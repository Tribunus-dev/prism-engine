//! PRISM-PRODUCTION-HETEROGENEOUS-EXECUTOR-0001 — Execution receipts with timing provenance.
//!
//! Every receipt captures wall-clock nanoseconds from [`std::time::SystemTime::now`]
//! converted to `duration_since(UNIX_EPOCH)`.  All fields come from real backend events
//! — no synthetic scheduler timestamps.
//!
//! The [`ReceiptCollector`] maintains an append-only log with a configurable maximum
//! capacity (oldest entries evicted).  Receipts can be exported as newline-delimited
//! JSON (JSONL) for offline analysis.

use serde::{Deserialize, Serialize};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::backend::placement::ExecutionLane;
use crate::compilation::activation_abi::{ActivationAbi, SlotLeaseId};
use crate::scheduling::completion_bridge::TimingQuality;
use crate::scheduling::lane_work::WorkId;
use crate::scheduling::work_registry::{WorkKey, WorkStatus};

// ---------------------------------------------------------------------------
// HeterogeneousExecutionReceipt
// ---------------------------------------------------------------------------

/// Complete execution receipt with full timing provenance.
///
/// Every field comes from real backend events — no synthetic scheduler
/// timestamps.  Wall-clock times are obtained from
/// [`SystemTime::now().duration_since(UNIX_EPOCH)`] at the relevant boundary
/// and stored as nanosecond-precision `u64` values.
///
/// Derived fields (`queue_wait_ns`, `execution_ns`, `overlap_ns`) are
/// populated at receipt construction time so the receipt is self-contained
/// and does not require further computation or external state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeterogeneousExecutionReceipt {
    /// Unique physical work identifier assigned at submission time.
    pub work_id: WorkId,

    /// Logical work identity across retries and fallback attempts.
    pub work_key: WorkKey,

    /// Execution lane that produced this receipt.
    pub lane: ExecutionLane,

    /// Attempt number (0 = original, 1+ = fallback retries).
    pub attempt: u32,

    /// Optional key referencing a backend-specific artifact (e.g. ANE
    /// compiled model handle, Metal pipeline state).
    pub artifact_key: Option<String>,

    /// Optional key referencing an ANE qualification record.
    pub qualification_key: Option<String>,

    /// Activation slot leases that were consumed as input.
    pub input_slots: Vec<SlotLeaseId>,

    /// Activation slot lease that received the output.
    pub output_slot: SlotLeaseId,

    /// ABI description of the input activation layout.
    pub input_abi: ActivationAbi,

    /// ABI description of the output activation layout.
    pub output_abi: ActivationAbi,

    // ── Wall-clock timestamps (ns since UNIX_EPOCH) ─────────────────────
    /// Wall-clock time when the work was submitted to the lane.
    pub submitted_at_ns: u64,

    /// Wall-clock time when the backend actually began execution.
    pub backend_started_at_ns: u64,

    /// Wall-clock time when the backend completed execution.
    pub backend_ended_at_ns: u64,

    /// Wall-clock time when the orchestrator received the completion.
    pub completion_received_at_ns: u64,

    // ── Derived timing fields ───────────────────────────────────────────
    /// Duration the item waited in the queue before backend execution began.
    /// Computed as `backend_started_at_ns - submitted_at_ns`.
    pub queue_wait_ns: u64,

    /// Duration of backend execution.
    /// Computed as `backend_ended_at_ns - backend_started_at_ns`.
    pub execution_ns: u64,

    /// Overlap with other work in the same epoch (ns).
    /// Set to 0 initially; updated externally by the epoch orchestrator
    /// when overlap tracking is available.
    pub overlap_ns: u64,

    // ── Fallback information ────────────────────────────────────────────
    /// Whether this execution used a fallback lane.
    pub fallback_used: bool,

    /// Reason the fallback was triggered, if applicable.
    pub fallback_reason: Option<String>,

    // ── Result status ───────────────────────────────────────────────────
    /// Terminal or intermediate status at completion time.
    pub status: WorkStatus,

    /// Quality of the timing data in this receipt.
    pub timing_quality: TimingQuality,
}

impl HeterogeneousExecutionReceipt {
    /// Queue wait duration in microseconds.
    pub fn queue_wait_us(&self) -> f64 {
        self.queue_wait_ns as f64 / 1_000.0
    }

    /// Execution duration in microseconds.
    pub fn execution_us(&self) -> f64 {
        self.execution_ns as f64 / 1_000.0
    }

    /// Total end-to-end latency in nanoseconds, from submission to
    /// completion receipt.
    pub fn total_latency_ns(&self) -> u64 {
        self.completion_received_at_ns
            .saturating_sub(self.submitted_at_ns)
    }
}

// ---------------------------------------------------------------------------
// Helpers for constructing receipts from event data
// ---------------------------------------------------------------------------

impl HeterogeneousExecutionReceipt {
    /// Record a wall-clock nanosecond timestamp since [`UNIX_EPOCH`].
    ///
    /// Returns `0` on platforms where `SystemTime::now()` is unreliable
    /// (should never happen on Darwin / Linux).
    pub fn now_ns() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_nanos() as u64
    }

    /// Build a receipt from the four wall-clock timestamps (submitted,
    /// backend started, backend ended, completion received) plus the
    /// semantic payload.
    ///
    /// Derived fields (`queue_wait_ns`, `execution_ns`) are computed
    /// automatically.  `overlap_ns` is set to 0 and can be updated
    /// externally.
    pub fn from_timestamps(
        work_id: WorkId,
        work_key: WorkKey,
        lane: ExecutionLane,
        attempt: u32,
        artifact_key: Option<String>,
        qualification_key: Option<String>,
        input_slots: Vec<SlotLeaseId>,
        output_slot: SlotLeaseId,
        input_abi: ActivationAbi,
        output_abi: ActivationAbi,
        submitted_at_ns: u64,
        backend_started_at_ns: u64,
        backend_ended_at_ns: u64,
        completion_received_at_ns: u64,
        fallback_used: bool,
        fallback_reason: Option<String>,
        status: WorkStatus,
        timing_quality: TimingQuality,
    ) -> Self {
        let queue_wait_ns = backend_started_at_ns.saturating_sub(submitted_at_ns);
        let execution_ns = backend_ended_at_ns.saturating_sub(backend_started_at_ns);

        Self {
            work_id,
            work_key,
            lane,
            attempt,
            artifact_key,
            qualification_key,
            input_slots,
            output_slot,
            input_abi,
            output_abi,
            submitted_at_ns,
            backend_started_at_ns,
            backend_ended_at_ns,
            completion_received_at_ns,
            queue_wait_ns,
            execution_ns,
            overlap_ns: 0,
            fallback_used,
            fallback_reason,
            status,
            timing_quality,
        }
    }
}

// ---------------------------------------------------------------------------
// ReceiptCollector — append-only, capped, JSONL-exportable
// ---------------------------------------------------------------------------

/// Append-only receipt collector with a configurable maximum capacity.
///
/// When the number of stored receipts reaches `max_receipts`, the oldest
/// entries are evicted to make room for new ones.  This prevents unbounded
/// memory growth in long-running production sessions.
///
/// The collector is designed for:
/// - On-line diagnostics (via [`snapshot`](ReceiptCollector::snapshot))
/// - Bulk export for offline analysis (via [`export_jsonl`](ReceiptCollector::export_jsonl))
/// - Periodic draining into a persistent store (via [`drain`](ReceiptCollector::drain))
pub struct ReceiptCollector {
    receipts: Vec<HeterogeneousExecutionReceipt>,
    max_receipts: usize,
}

impl ReceiptCollector {
    /// Create a new collector with the given capacity.
    ///
    /// `max_receipts` must be at least 1; values below 1 are clamped to 1.
    pub fn new(max_receipts: usize) -> Self {
        let max_receipts = max_receipts.max(1).min(1024 * 1024);
        Self {
            receipts: Vec::with_capacity(max_receipts),
            max_receipts,
        }
    }

    /// Record a receipt, evicting the oldest entry if at capacity.
    pub fn record(&mut self, receipt: HeterogeneousExecutionReceipt) {
        if self.receipts.len() >= self.max_receipts {
            // Evict the oldest receipt (O(1) removal from front via swap-remove
            // would reorder — use rotate_left for append-only ordering).
            // When at capacity and a new receipt arrives, drain the oldest.
            self.receipts.remove(0);
        }
        self.receipts.push(receipt);
    }

    /// Drain all stored receipts, leaving the collector empty.
    pub fn drain(&mut self) -> Vec<HeterogeneousExecutionReceipt> {
        std::mem::take(&mut self.receipts)
    }

    /// Returns an immutable snapshot of all stored receipts.
    pub fn snapshot(&self) -> &[HeterogeneousExecutionReceipt] {
        &self.receipts
    }

    /// Export all stored receipts as newline-delimited JSON (JSONL).
    ///
    /// Each line is a complete JSON object terminated by `\n`.
    /// Returns an empty string when no receipts are stored.
    pub fn export_jsonl(&self) -> String {
        let mut out = String::with_capacity(self.receipts.len() * 512);
        for receipt in &self.receipts {
            if let Ok(line) = serde_json::to_string(receipt) {
                out.push_str(&line);
                out.push('\n');
            }
        }
        out
    }

    /// Returns the number of stored receipts.
    pub fn len(&self) -> usize {
        self.receipts.len()
    }

    /// Returns `true` when no receipts are stored.
    pub fn is_empty(&self) -> bool {
        self.receipts.is_empty()
    }
}

// ---------------------------------------------------------------------------
// FallbackSummary
// ---------------------------------------------------------------------------

/// Summary of fallback activity for an epoch or request.
///
/// Produced as part of a higher-level execution result to provide a quick
/// overview of fallback-induced lane usage without requiring consumers to
/// scan the full receipt list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FallbackSummary {
    /// Total number of fallback activations across the epoch/request.
    pub total_fallbacks: u32,

    /// Ordered list of reasons for each fallback.
    pub reasons: Vec<String>,

    /// Lanes that were used as fallback targets.
    pub lanes_used: Vec<ExecutionLane>,
}

impl Default for FallbackSummary {
    fn default() -> Self {
        Self {
            total_fallbacks: 0,
            reasons: Vec::new(),
            lanes_used: Vec::new(),
        }
    }
}

impl FallbackSummary {
    /// Record a fallback occurrence.
    pub fn record_fallback(&mut self, reason: &str, lane: ExecutionLane) {
        self.total_fallbacks += 1;
        self.reasons.push(reason.to_string());
        if !self.lanes_used.contains(&lane) {
            self.lanes_used.push(lane);
        }
    }

    /// Merge another fallback summary into this one.
    pub fn merge(&mut self, other: &FallbackSummary) {
        self.total_fallbacks += other.total_fallbacks;
        self.reasons.extend(other.reasons.iter().cloned());
        for lane in &other.lanes_used {
            if !self.lanes_used.contains(lane) {
                self.lanes_used.push(*lane);
            }
        }
    }

    /// Returns `true` if no fallbacks occurred.
    pub fn is_clean(&self) -> bool {
        self.total_fallbacks == 0
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compilation::activation_abi::{DecodeActivationV1Params, PhysicalLayout};
    use crate::compilation::phase_ir::PhaseId;
    use crate::compilation::phase_ir::TensorDtype;

    fn test_abi() -> ActivationAbi {
        ActivationAbi::DecodeActivationV1(DecodeActivationV1Params {
            dtype: TensorDtype::Float16,
            seq_bucket: 0,
            hidden_dim: 4096,
            physical_layout: PhysicalLayout::ContiguousRowMajor,
            alignment: 64,
            stride_constraint: None,
        })
    }

    fn sample_receipt(work_id: WorkId, attempt: u32) -> HeterogeneousExecutionReceipt {
        let submitted = 1_000_000_000_000; // arbitrary base: 1000s since epoch
        let queue = 50_000; // 50 μs
        let exec = 200_000; // 200 μs
        let completion_delay = 5_000; // 5 μs

        HeterogeneousExecutionReceipt {
            work_id,
            work_key: WorkKey {
                session_id: "test-session".into(),
                request_id: "test-request".into(),
                sequence_id: 0,
                epoch_id: 0,
                phase_id: PhaseId(0),
                attempt,
            },
            lane: ExecutionLane::MlxGpu,
            attempt,
            artifact_key: None,
            qualification_key: None,
            input_slots: vec![SlotLeaseId(1), SlotLeaseId(2)],
            output_slot: SlotLeaseId(3),
            input_abi: test_abi(),
            output_abi: test_abi(),
            submitted_at_ns: submitted,
            backend_started_at_ns: submitted + queue,
            backend_ended_at_ns: submitted + queue + exec,
            completion_received_at_ns: submitted + queue + exec + completion_delay,
            queue_wait_ns: queue,
            execution_ns: exec,
            overlap_ns: 0,
            fallback_used: attempt > 0,
            fallback_reason: if attempt > 0 {
                Some("fallback for test".into())
            } else {
                None
            },
            status: WorkStatus::Released,
            timing_quality: TimingQuality::MetalCommandBufferCompletion,
        }
    }

    #[test]
    fn test_timing_derivation() {
        let r = sample_receipt(WorkId(1), 0);

        assert_eq!(r.queue_wait_ns, 50_000);
        assert_eq!(r.execution_ns, 200_000);
        assert_eq!(r.total_latency_ns(), 255_000);

        // microsecond helpers
        let queue_us = r.queue_wait_us();
        let exec_us = r.execution_us();
        assert!((queue_us - 50.0).abs() < 1e-9);
        assert!((exec_us - 200.0).abs() < 1e-9);
    }

    #[test]
    fn test_from_timestamps_builder() {
        let work_id = WorkId(42);
        let work_key = WorkKey {
            session_id: "sess".into(),
            request_id: "req".into(),
            sequence_id: 1,
            epoch_id: 2,
            phase_id: PhaseId(3),
            attempt: 0,
        };
        let input_slots = vec![SlotLeaseId(10)];
        let input_abi = test_abi();
        let output_abi = test_abi();

        let receipt = HeterogeneousExecutionReceipt::from_timestamps(
            work_id,
            work_key.clone(),
            ExecutionLane::CoreMlAne,
            0,
            Some("artifact-key".into()),
            Some("qual-key".into()),
            input_slots.clone(),
            SlotLeaseId(20),
            input_abi.clone(),
            output_abi.clone(),
            1_000_000,
            1_000_100, // 100 μs queue wait
            1_000_500, // 400 μs execution
            1_000_520, // 20 μs completion delay
            false,
            None,
            WorkStatus::Completed,
            TimingQuality::CoreMlWorkerBoundary,
        );

        assert_eq!(receipt.work_id, work_id);
        assert_eq!(receipt.work_key, work_key);
        assert_eq!(receipt.lane, ExecutionLane::CoreMlAne);
        assert_eq!(receipt.artifact_key.as_deref(), Some("artifact-key"));
        assert_eq!(receipt.input_slots, input_slots);
        assert_eq!(receipt.output_slot, SlotLeaseId(20));
        assert_eq!(receipt.queue_wait_ns, 100);
        assert_eq!(receipt.execution_ns, 400);
        assert_eq!(receipt.overlap_ns, 0);
        assert!(!receipt.fallback_used);
        assert_eq!(receipt.status, WorkStatus::Completed);
        assert_eq!(receipt.timing_quality, TimingQuality::CoreMlWorkerBoundary);
    }

    #[test]
    fn test_collector_record_and_snapshot() {
        let mut collector = ReceiptCollector::new(5);

        assert!(collector.is_empty());
        assert_eq!(collector.len(), 0);

        for i in 0..3 {
            collector.record(sample_receipt(WorkId(i as u64), 0));
        }

        assert!(!collector.is_empty());
        assert_eq!(collector.len(), 3);
        assert_eq!(collector.snapshot().len(), 3);
    }

    #[test]
    fn test_collector_eviction() {
        let mut collector = ReceiptCollector::new(3);

        for i in 0..5 {
            collector.record(sample_receipt(WorkId(i as u64), 0));
        }

        // Capacity is 3, oldest 2 should be evicted
        assert_eq!(collector.len(), 3);
        // Remaining: WorkId(2), WorkId(3), WorkId(4)
        assert_eq!(collector.snapshot()[0].work_id, WorkId(2));
        assert_eq!(collector.snapshot()[2].work_id, WorkId(4));
    }

    #[test]
    fn test_collector_drain() {
        let mut collector = ReceiptCollector::new(10);

        for i in 0..4 {
            collector.record(sample_receipt(WorkId(i as u64), 0));
        }

        let drained = collector.drain();
        assert_eq!(drained.len(), 4);
        assert!(collector.is_empty());
        assert_eq!(collector.len(), 0);
    }

    #[test]
    fn test_collector_export_jsonl() {
        let mut collector = ReceiptCollector::new(10);

        collector.record(sample_receipt(WorkId(1), 0));
        collector.record(sample_receipt(WorkId(2), 1));

        let jsonl = collector.export_jsonl();
        assert!(!jsonl.is_empty());

        let lines: Vec<&str> = jsonl.trim().lines().collect();
        assert_eq!(lines.len(), 2);

        // Each line must be valid JSON
        for line in &lines {
            let parsed: serde_json::Value =
                serde_json::from_str(line).expect("each JSONL line must be valid JSON");
            assert!(parsed.is_object());
            assert!(parsed.get("work_id").is_some());
            assert!(parsed.get("queue_wait_ns").is_some());
        }
    }

    #[test]
    fn test_collector_clamps_capacity() {
        let collector = ReceiptCollector::new(0);
        assert_eq!(collector.max_receipts, 1);

        let collector = ReceiptCollector::new(usize::MAX);
        assert_eq!(collector.max_receipts, 1024 * 1024);
    }

    #[test]
    fn test_fallback_summary_default() {
        let summary = FallbackSummary::default();
        assert_eq!(summary.total_fallbacks, 0);
        assert!(summary.reasons.is_empty());
        assert!(summary.lanes_used.is_empty());
        assert!(summary.is_clean());
    }

    #[test]
    fn test_fallback_summary_record() {
        let mut summary = FallbackSummary::default();
        summary.record_fallback("ANE unavailable", ExecutionLane::CandleCpu);
        summary.record_fallback("ANE timeout", ExecutionLane::CandleCpu);

        assert_eq!(summary.total_fallbacks, 2);
        assert_eq!(summary.reasons.len(), 2);
        assert_eq!(summary.lanes_used.len(), 1);
        assert_eq!(summary.lanes_used[0], ExecutionLane::CandleCpu);
        assert!(!summary.is_clean());
    }

    #[test]
    fn test_fallback_summary_merge() {
        let mut s1 = FallbackSummary::default();
        s1.record_fallback("ANE unavailable", ExecutionLane::CandleCpu);

        let mut s2 = FallbackSummary::default();
        s2.record_fallback("GPU OOM", ExecutionLane::AccelerateCpu);

        s1.merge(&s2);

        assert_eq!(s1.total_fallbacks, 2);
        assert_eq!(s1.reasons.len(), 2);
        // Both lanes should be present
        assert!(s1.lanes_used.contains(&ExecutionLane::CandleCpu));
        assert!(s1.lanes_used.contains(&ExecutionLane::AccelerateCpu));
    }

    #[test]
    fn test_fallback_summary_merge_deduplicates_lanes() {
        let mut s1 = FallbackSummary::default();
        s1.record_fallback("reason1", ExecutionLane::CandleCpu);
        s1.record_fallback("reason2", ExecutionLane::CandleCpu);

        let s2 = FallbackSummary::default(); // empty
        s1.merge(&s2);

        assert_eq!(s1.total_fallbacks, 2);
        assert_eq!(s1.lanes_used.len(), 1);
    }

    #[test]
    fn test_now_ns_returns_non_zero() {
        let ts = HeterogeneousExecutionReceipt::now_ns();
        assert!(
            ts > 0,
            "SystemTime::now() must return a post-epoch timestamp"
        );
    }

    #[test]
    fn test_receipt_serialize_roundtrip() {
        let r = sample_receipt(WorkId(7), 0);
        let json = serde_json::to_string(&r).expect("serialize receipt");
        let deserialized: HeterogeneousExecutionReceipt =
            serde_json::from_str(&json).expect("deserialize receipt");

        assert_eq!(deserialized.work_id, r.work_id);
        assert_eq!(deserialized.work_key, r.work_key);
        assert_eq!(deserialized.lane, r.lane);
        assert_eq!(deserialized.queue_wait_ns, r.queue_wait_ns);
        assert_eq!(deserialized.execution_ns, r.execution_ns);
        assert_eq!(deserialized.total_latency_ns(), r.total_latency_ns());
        assert_eq!(deserialized.timing_quality, r.timing_quality);
    }
}
