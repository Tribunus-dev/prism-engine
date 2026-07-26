//! Backpressure authority for the heterogeneous executor.
//!
//! Authority: this module owns the canonical admission-gate signal
//! for the heterogeneous executor — the typed backpressure events,
//! the severity level, the latency-window controller, and the
//! scheduling metrics that the orchestrator reads before admitting
//! new work and after dispatch. It does not own lane capacity
//! counters (those live in `lane_capacity`), the lane queue
//! (that lives in `lane_queue`), the orchestrator actor (that
//! lives in `heterogeneous_executor`), or the receipt collection
//! (that lives in `execution_receipts`).
//!
//! Constitutional notes:
//!
//! - The backpressure severity is [`BackpressureLevel`], a typed
//!   newtype that wraps a `u8` severity ordinal (0 = None, 1 =
//!   Mild, 2 = Moderate, 3 = Severe, 4 = Critical). The newtype
//!   prevents the integer-arithmetic mistakes that the engine's
//!   `enum` allows (e.g. accidentally using a level as a count).
//! - All canonical collections use `BTreeMap` per the
//!   "no HashMap/HashSet for canonical collections" rule.
//! - All fallible operations return [`Result<_, BackpressureError>`],
//!   a `thiserror`-derived enum.
//!
//! Two backpressure surfaces:
//!
//! 1. **Resource events** ([`BackpressureEventController`]) — the
//!    `AdmitSystem` reads this before admitting a new work item. It
//!    derives a [`BackpressureLevel`] from the most severe
//!    contributing reason category (resource > capacity > pool >
//!    transient).
//!
//! 2. **Latency window** ([`BackpressureController`]) — the
//!    `DispatchSystem` records a [`BatchCompletionRecord`] after
//!    each batch completes. When recent latencies exceed the
//!    configured window, the `AdmitSystem` throttles new
//!    admissions. The `SchedulingMetrics` struct couples the
//!    latency controller to a dynamic `max_num_scheduled_tokens`
//!    budget that the admission gate consumes.

use std::collections::{BTreeMap, VecDeque};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ── Backpressure reason ───────────────────────────────────────────────────

/// Specific reason for a backpressure signal. The
/// [`BackpressureEventController`] aggregates events by reason; the
/// [`BackpressureLevel`] is derived from the most severe reason
/// category present.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum BackpressureReason {
    /// Metal GPU command buffer / compute pipeline at capacity.
    MetalCapacity,
    /// Apple Neural Engine at capacity.
    AneCapacity,
    /// CPU compute lanes saturated.
    CpuCapacity,
    /// Activation arena slot exhaustion.
    ActivationSlots,
    /// IOSurface pool depleted or fragmented.
    IOSurfacePool,
    /// Per-session resource quota exceeded.
    SessionQuota,
    /// Global dispatch queue depth limit reached.
    GlobalQueue,
    /// ANE artifact-cache miss causing cold-load delay.
    ArtifactCold,
}

impl BackpressureReason {
    /// The severity category this reason contributes to the
    /// aggregate level. The categories are, from least to most
    /// severe: `Transient` < `Pool` < `Capacity` < `Quota`.
    pub const fn severity(self) -> BackpressureCategory {
        match self {
            BackpressureReason::ArtifactCold => BackpressureCategory::Transient,
            BackpressureReason::ActivationSlots | BackpressureReason::IOSurfacePool => {
                BackpressureCategory::Pool
            }
            BackpressureReason::MetalCapacity
            | BackpressureReason::AneCapacity
            | BackpressureReason::CpuCapacity => BackpressureCategory::Capacity,
            BackpressureReason::SessionQuota | BackpressureReason::GlobalQueue => {
                BackpressureCategory::Quota
            }
        }
    }

    /// Map a reason to its category ordinal for the
    /// [`BackpressureLevel`] derivation. The ordinals are not part
    /// of the public API; the level is.
    const fn severity_ordinal(self) -> u8 {
        match self.severity() {
            BackpressureCategory::Transient => 1,
            BackpressureCategory::Pool => 2,
            BackpressureCategory::Capacity => 3,
            BackpressureCategory::Quota => 4,
        }
    }

    /// Whether this reason is a transient (recoverable) condition.
    /// All built-in reasons are transient; the orchestrator can
    /// retry.
    pub const fn is_transient(self) -> bool {
        true
    }
}

/// Backpressure severity category. Used to derive the
/// [`BackpressureLevel`] from a set of [`BackpressureReason`]s.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum BackpressureCategory {
    /// Transient delay (cold cache, etc.) — Mild.
    Transient,
    /// Shared pool / slot exhaustion (IOSurface, activation arena) —
    /// Moderate.
    Pool,
    /// Lane capacity (Metal, ANE, CPU) — Severe.
    Capacity,
    /// Session or global quota — Critical.
    Quota,
}

// ── Backpressure level ────────────────────────────────────────────────────

/// Severity ordinal for the request scheduler's admission gate.
///
/// The level is derived from the most severe contributing reason
/// category. The orchestrator uses the level to throttle or refuse
/// new admissions; the level is also published as a metric for the
/// EXO autoscaler.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct BackpressureLevel(pub u8);

impl BackpressureLevel {
    /// Normal operation — no backpressure.
    pub const NONE: Self = Self(0);
    /// Mild — new work should be delayed briefly.
    pub const MILD: Self = Self(1);
    /// Moderate — only high-priority work should be admitted.
    pub const MODERATE: Self = Self(2);
    /// Severe — no new work should be admitted.
    pub const SEVERE: Self = Self(3);
    /// Critical — in-flight work may need to be cancelled.
    pub const CRITICAL: Self = Self(4);

    /// Construct a level from a raw `u8`. Clamps to the
    /// `[NONE, CRITICAL]` range; values above `CRITICAL` are
    /// pinned to `CRITICAL`.
    pub const fn from_raw(raw: u8) -> Self {
        if raw >= Self::CRITICAL.0 {
            Self::CRITICAL
        } else {
            Self(raw)
        }
    }

    /// Borrow the raw severity ordinal.
    pub const fn raw(self) -> u8 {
        self.0
    }

    /// True if the level is at or above `MODERATE` (admission
    /// throttling is in effect).
    pub const fn is_admission_throttling(self) -> bool {
        self.0 >= Self::MODERATE.0
    }

    /// True if the level is at `SEVERE` or `CRITICAL` (new
    /// admissions are refused).
    pub const fn is_admission_refused(self) -> bool {
        self.0 >= Self::SEVERE.0
    }

    /// True if the level is at `CRITICAL` (in-flight work may need
    /// to be cancelled).
    pub const fn is_cancellation_required(self) -> bool {
        self.0 >= Self::CRITICAL.0
    }
}

impl Default for BackpressureLevel {
    fn default() -> Self {
        Self::NONE
    }
}

// ── Backpressure event ────────────────────────────────────────────────────

/// A backpressure event with reason, severity, and affected lane.
///
/// Note: the `timestamp` field is `#[serde(skip)]` because
/// `std::time::Instant` does not implement `Serialize`/
/// `Deserialize` (the wall-clock representation is process-local).
/// The struct therefore derives `Serialize` only — it is
/// runtime-only state, and the durable record is the event store
/// entry. The orchestrator reconstructs the event from the event
/// history on replay.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BackpressureEvent {
    /// The specific resource or capacity reason.
    pub reason: BackpressureReason,
    /// Optional identifier for the lane experiencing backpressure.
    /// Constitutional side stores it as a typed `u32` (the orchestrator
    /// passes the `LaneId` raw ordinal).
    pub lane_ordinal: Option<u32>,
    /// Optional session identifier (typed as a `String`).
    pub affected_session: Option<String>,
    /// Severity ordinal (a [`BackpressureLevel`]).
    pub level: BackpressureLevel,
    /// Monotonic timestamp of when the event was observed.
    /// `#[serde(skip)]` — `Instant` is not serializable; the
    /// durable record is the event store entry.
    #[serde(skip)]
    pub timestamp: Instant,
    /// Free-form details for diagnostics or observability.
    pub details: String,
}

impl BackpressureEvent {
    /// Construct a new event with the given reason and level. The
    /// timestamp is set to the current instant.
    pub fn new(reason: BackpressureReason, level: BackpressureLevel) -> Self {
        Self {
            reason,
            lane_ordinal: None,
            affected_session: None,
            level,
            timestamp: Instant::now(),
            details: String::new(),
        }
    }

    /// Set the lane ordinal.
    pub fn with_lane(mut self, lane_ordinal: u32) -> Self {
        self.lane_ordinal = Some(lane_ordinal);
        self
    }

    /// Set the affected session.
    pub fn with_session(mut self, session: impl Into<String>) -> Self {
        self.affected_session = Some(session.into());
        self
    }

    /// Set the details string.
    pub fn with_details(mut self, details: impl Into<String>) -> Self {
        self.details = details.into();
        self
    }
}

// ── Backpressure event controller ─────────────────────────────────────────

/// Tracks resource-level backpressure events across all resources.
///
/// Maintains an ordered map of active [`BackpressureEvent`]s and
/// derives a current [`BackpressureLevel`] based on the most severe
/// category present. The map is keyed by event reason + lane so
/// distinct events on different lanes do not collapse into one.
#[derive(Debug)]
pub struct BackpressureEventController {
    /// Active events keyed by `(reason, lane_ordinal)`. `BTreeMap`
    /// so the iteration order is deterministic for the summary.
    active_events: BTreeMap<(BackpressureReason, Option<u32>), BackpressureEvent>,
    /// Current derived level.
    level: BackpressureLevel,
    /// Maximum number of distinct event entries to retain.
    max_events: usize,
}

impl Default for BackpressureEventController {
    fn default() -> Self {
        Self::new()
    }
}

impl BackpressureEventController {
    /// Create a new controller with the default capacity (256 events).
    pub fn new() -> Self {
        Self::with_max_events(256)
    }

    /// Create a new controller with a custom event capacity.
    pub fn with_max_events(max: usize) -> Self {
        let max = max.min(4096);
        Self {
            active_events: BTreeMap::new(),
            level: BackpressureLevel::NONE,
            max_events: max,
        }
    }

    /// Record a backpressure event. If the event buffer is at
    /// capacity, the oldest event (by key ordering) is evicted.
    pub fn report(&mut self, event: BackpressureEvent) {
        let key = (event.reason, event.lane_ordinal);
        if self.active_events.len() >= self.max_events
            && !self.active_events.contains_key(&key)
        {
            // Evict the oldest event (smallest key in BTreeMap order).
            if let Some(oldest_key) = self.active_events.keys().next().cloned() {
                self.active_events.remove(&oldest_key);
            }
        }
        self.active_events.insert(key, event);
        self.recalculate_level();
    }

    /// Clear expired/resolved events older than the given cutoff.
    /// After removal the controller recalculates the aggregate
    /// level.
    pub fn clear_before(&mut self, cutoff: Instant) {
        self.active_events
            .retain(|_, e| e.timestamp >= cutoff);
        self.recalculate_level();
    }

    /// Current backpressure level.
    pub fn level(&self) -> BackpressureLevel {
        self.level
    }

    /// Manually set backpressure level (e.g., after clearing
    /// events externally).
    pub fn set_level(&mut self, level: BackpressureLevel) {
        self.level = level;
    }

    /// All active events, ordered by `(reason, lane_ordinal)`.
    pub fn events(&self) -> Vec<&BackpressureEvent> {
        self.active_events.values().collect()
    }

    /// Number of distinct active event entries.
    pub fn event_count(&self) -> usize {
        self.active_events.len()
    }

    /// Clear all events and reset to `NONE`.
    pub fn clear(&mut self) {
        self.active_events.clear();
        self.level = BackpressureLevel::NONE;
    }

    /// Recalculate the aggregate level from active events.
    fn recalculate_level(&mut self) {
        self.level = derive_level(&self.active_events);
    }

    /// Produce a serializable summary of the current backpressure
    /// state.
    pub fn summary(&self) -> BackpressureSummary {
        let mut reasons: Vec<BackpressureReason> = Vec::new();
        let mut affected_lanes: Vec<u32> = Vec::new();

        for event in self.active_events.values() {
            if !reasons.contains(&event.reason) {
                reasons.push(event.reason);
            }
            if let Some(lane) = event.lane_ordinal {
                if !affected_lanes.contains(&lane) {
                    affected_lanes.push(lane);
                }
            }
        }

        BackpressureSummary {
            level: self.level,
            active_event_count: self.active_events.len(),
            reasons,
            affected_lanes,
        }
    }
}

/// Derive the aggregate [`BackpressureLevel`] from a map of active
/// events. Uses the highest-severity category present.
fn derive_level(
    events: &BTreeMap<(BackpressureReason, Option<u32>), BackpressureEvent>,
) -> BackpressureLevel {
    let mut max_ordinal: u8 = BackpressureLevel::NONE.0;
    for event in events.values() {
        let ordinal = event.reason.severity_ordinal();
        if ordinal > max_ordinal {
            max_ordinal = ordinal;
        }
    }
    BackpressureLevel::from_raw(max_ordinal)
}

/// Serializable summary of backpressure state for observability.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackpressureSummary {
    /// Aggregate severity level.
    pub level: BackpressureLevel,
    /// Number of active events contributing to the level.
    pub active_event_count: usize,
    /// Unique list of backpressure reasons present among active
    /// events.
    pub reasons: Vec<BackpressureReason>,
    /// Unique set of lane ordinals that have active backpressure
    /// events.
    pub affected_lanes: Vec<u32>,
}

// ── Latency-window backpressure ───────────────────────────────────────────

/// A record of a completed batch for latency-based backpressure
/// tracking.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchCompletionRecord {
    /// The session / request identifier.
    pub request_id: String,
    /// Number of tokens completed in this batch.
    pub tokens_completed: usize,
    /// Backend execution latency (ns).
    pub latency_ns: u64,
    /// Wall-clock time of completion (ns since some epoch; the
    /// latency controller treats this as an opaque u64).
    pub completed_at_ns: u64,
}

impl BatchCompletionRecord {
    /// Construct a record with the given fields.
    pub fn new(
        request_id: impl Into<String>,
        tokens_completed: usize,
        latency_ns: u64,
        completed_at_ns: u64,
    ) -> Self {
        Self {
            request_id: request_id.into(),
            tokens_completed,
            latency_ns,
            completed_at_ns,
        }
    }
}

/// Latency-based backpressure controller that tracks batch
/// completion records and determines whether to throttle new
/// admissions based on recent completion latency vs a target
/// window.
///
/// Distinct from [`BackpressureEventController`]: the event
/// controller tracks resource-level events (Metal capacity, ANE,
/// etc.), while this controller tracks actual batch completion
/// latencies to detect when the system is taking too long per
/// batch.
#[derive(Debug, Clone)]
pub struct BackpressureController {
    max_pending_microseconds: u64,
    pending_batches: VecDeque<BatchCompletionRecord>,
    target_occupancy: f32,
    /// Maximum number of records to retain.
    max_records: usize,
}

impl BackpressureController {
    /// Construct a new controller. `max_pending_us` is the
    /// latency window (in microseconds) within which a recent
    /// completion triggers backpressure. `target_occupancy` is the
    /// 0.0..=1.0 occupancy target (reserved for future use).
    pub fn new(max_pending_us: u64, target_occupancy: f32) -> Self {
        Self {
            max_pending_microseconds: max_pending_us,
            pending_batches: VecDeque::with_capacity(1000),
            target_occupancy,
            max_records: 1000,
        }
    }

    /// Construct a new controller with a custom record capacity.
    pub fn with_capacity(max_pending_us: u64, target_occupancy: f32, max_records: usize) -> Self {
        Self {
            max_pending_microseconds: max_pending_us,
            pending_batches: VecDeque::with_capacity(max_records),
            target_occupancy,
            max_records,
        }
    }

    /// Record a batch completion.
    pub fn record_completion(&mut self, record: BatchCompletionRecord) {
        self.pending_batches.push_back(record);
        while self.pending_batches.len() > self.max_records {
            self.pending_batches.pop_front();
        }
    }

    /// Whether the scheduler should pause new work. True if the
    /// oldest pending batch was completed within the configured
    /// latency window.
    pub fn is_backpressure(&self, current_time_ns: u64) -> bool {
        if let Some(oldest) = self.pending_batches.front() {
            return (current_time_ns.saturating_sub(oldest.completed_at_ns))
                < self.max_pending_microseconds.saturating_mul(1000);
        }
        false
    }

    /// Current average latency in ns.
    pub fn avg_latency_ns(&self) -> f64 {
        if self.pending_batches.is_empty() {
            return 0.0;
        }
        let total: u64 = self.pending_batches.iter().map(|r| r.latency_ns).sum();
        total as f64 / self.pending_batches.len() as f64
    }

    /// Number of pending batch records.
    pub fn pending_count(&self) -> usize {
        self.pending_batches.len()
    }

    /// The configured maximum pending microseconds.
    pub fn max_pending_microseconds(&self) -> u64 {
        self.max_pending_microseconds
    }

    /// The configured target occupancy.
    pub fn target_occupancy(&self) -> f32 {
        self.target_occupancy
    }
}

// ── Scheduling metrics ────────────────────────────────────────────────────

/// High-level scheduling metrics that feed admission decisions.
///
/// Combines the latency-window [`BackpressureController`] and a
/// dynamic `max_num_scheduled_tokens` budget that the admission
/// gate consumes. The orchestrator's `AdmitSystem` reads
/// [`SchedulingMetrics::current_admission_level`] and
/// [`SchedulingMetrics::max_num_scheduled_tokens`] before admitting
/// each work item.
#[derive(Debug, Clone)]
pub struct SchedulingMetrics {
    max_num_scheduled_tokens: usize,
    num_running_requests: usize,
    avg_prefill_latency_ns: f64,
    avg_decode_latency_ns: f64,
    kv_cache_usage_pct: f64,
    backpressure: BackpressureController,
}

impl SchedulingMetrics {
    /// Default initial token budget.
    pub const DEFAULT_TOKEN_BUDGET: usize = 4096;
    /// Minimum token budget under backpressure.
    pub const MIN_TOKEN_BUDGET: usize = 64;
    /// Step by which the budget is restored when latency is
    /// healthy.
    pub const TOKEN_BUDGET_RESTORE_STEP: usize = 256;

    /// Create new scheduling metrics with the given backpressure
    /// config.
    pub fn new(max_pending_us: u64, target_occupancy: f32) -> Self {
        Self {
            max_num_scheduled_tokens: Self::DEFAULT_TOKEN_BUDGET,
            num_running_requests: 0,
            avg_prefill_latency_ns: 0.0,
            avg_decode_latency_ns: 0.0,
            kv_cache_usage_pct: 0.0,
            backpressure: BackpressureController::new(max_pending_us, target_occupancy),
        }
    }

    /// Current `max_num_scheduled_tokens` budget.
    pub fn max_num_scheduled_tokens(&self) -> usize {
        self.max_num_scheduled_tokens
    }

    /// Number of running requests.
    pub fn num_running_requests(&self) -> usize {
        self.num_running_requests
    }

    /// Set the number of running requests.
    pub fn set_num_running_requests(&mut self, n: usize) {
        self.num_running_requests = n;
    }

    /// Average prefill latency in ns.
    pub fn avg_prefill_latency_ns(&self) -> f64 {
        self.avg_prefill_latency_ns
    }

    /// Set the average prefill latency in ns.
    pub fn set_avg_prefill_latency_ns(&mut self, ns: f64) {
        self.avg_prefill_latency_ns = ns;
    }

    /// Average decode latency in ns.
    pub fn avg_decode_latency_ns(&self) -> f64 {
        self.avg_decode_latency_ns
    }

    /// Set the average decode latency in ns.
    pub fn set_avg_decode_latency_ns(&mut self, ns: f64) {
        self.avg_decode_latency_ns = ns;
    }

    /// KV cache usage as a percentage (0.0..=100.0).
    pub fn kv_cache_usage_pct(&self) -> f64 {
        self.kv_cache_usage_pct
    }

    /// Set the KV cache usage percentage.
    pub fn set_kv_cache_usage_pct(&mut self, pct: f64) {
        self.kv_cache_usage_pct = pct;
    }

    /// Borrow the latency-window backpressure controller.
    pub fn backpressure(&self) -> &BackpressureController {
        &self.backpressure
    }

    /// Mutably borrow the latency-window backpressure controller.
    pub fn backpressure_mut(&mut self) -> &mut BackpressureController {
        &mut self.backpressure
    }

    /// Update the scheduled token budget based on the current
    /// backpressure state.
    ///
    /// When recent batches exceed the latency window
    /// (`is_backpressure == true`), gradually reduce
    /// `max_num_scheduled_tokens` (by 25%, minimum
    /// [`Self::MIN_TOKEN_BUDGET`]). When latency is healthy,
    /// gradually restore toward
    /// [`Self::DEFAULT_TOKEN_BUDGET`] (by
    /// [`Self::TOKEN_BUDGET_RESTORE_STEP`]).
    pub fn update_token_budget(&mut self, current_time_ns: u64) {
        if self.backpressure.is_backpressure(current_time_ns) {
            self.max_num_scheduled_tokens = (self.max_num_scheduled_tokens / 4 * 3)
                .max(Self::MIN_TOKEN_BUDGET);
        } else {
            self.max_num_scheduled_tokens = (self.max_num_scheduled_tokens
                + Self::TOKEN_BUDGET_RESTORE_STEP)
                .min(Self::DEFAULT_TOKEN_BUDGET);
        }
    }

    /// The current admission level derived from the latency-window
    /// controller. The `AdmitSystem` uses this to throttle or refuse
    /// new admissions.
    pub fn current_admission_level(&self, current_time_ns: u64) -> BackpressureLevel {
        if self.backpressure.is_backpressure(current_time_ns) {
            BackpressureLevel::MILD
        } else {
            BackpressureLevel::NONE
        }
    }
}

// ── Errors ────────────────────────────────────────────────────────────────

/// Errors emitted by the backpressure subsystem. The error type is
/// reserved for future fallible APIs (today the backpressure
/// subsystem is read-only from the orchestrator's perspective).
#[derive(Debug, Error, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackpressureError {
    /// An invalid parameter was provided to a backpressure
    /// constructor.
    #[error("invalid backpressure parameter: {0}")]
    InvalidParameter(String),
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── BackpressureReason tests ────────────────────────────────────

    #[test]
    fn reason_severity_derivation() {
        assert_eq!(
            BackpressureReason::ArtifactCold.severity(),
            BackpressureCategory::Transient
        );
        assert_eq!(
            BackpressureReason::ActivationSlots.severity(),
            BackpressureCategory::Pool
        );
        assert_eq!(
            BackpressureReason::IOSurfacePool.severity(),
            BackpressureCategory::Pool
        );
        assert_eq!(
            BackpressureReason::MetalCapacity.severity(),
            BackpressureCategory::Capacity
        );
        assert_eq!(
            BackpressureReason::AneCapacity.severity(),
            BackpressureCategory::Capacity
        );
        assert_eq!(
            BackpressureReason::CpuCapacity.severity(),
            BackpressureCategory::Capacity
        );
        assert_eq!(
            BackpressureReason::SessionQuota.severity(),
            BackpressureCategory::Quota
        );
        assert_eq!(
            BackpressureReason::GlobalQueue.severity(),
            BackpressureCategory::Quota
        );
    }

    #[test]
    fn all_reasons_are_transient() {
        let reasons = [
            BackpressureReason::MetalCapacity,
            BackpressureReason::AneCapacity,
            BackpressureReason::CpuCapacity,
            BackpressureReason::ActivationSlots,
            BackpressureReason::IOSurfacePool,
            BackpressureReason::SessionQuota,
            BackpressureReason::GlobalQueue,
            BackpressureReason::ArtifactCold,
        ];
        for r in &reasons {
            assert!(r.is_transient());
        }
    }

    // ── BackpressureLevel tests ─────────────────────────────────────

    #[test]
    fn level_constants_have_expected_ordering() {
        assert!(BackpressureLevel::NONE < BackpressureLevel::MILD);
        assert!(BackpressureLevel::MILD < BackpressureLevel::MODERATE);
        assert!(BackpressureLevel::MODERATE < BackpressureLevel::SEVERE);
        assert!(BackpressureLevel::SEVERE < BackpressureLevel::CRITICAL);
    }

    #[test]
    fn level_from_raw_clamps() {
        assert_eq!(BackpressureLevel::from_raw(0), BackpressureLevel::NONE);
        assert_eq!(BackpressureLevel::from_raw(1), BackpressureLevel::MILD);
        assert_eq!(BackpressureLevel::from_raw(2), BackpressureLevel::MODERATE);
        assert_eq!(BackpressureLevel::from_raw(3), BackpressureLevel::SEVERE);
        assert_eq!(BackpressureLevel::from_raw(4), BackpressureLevel::CRITICAL);
        // Above CRITICAL clamps to CRITICAL.
        assert_eq!(BackpressureLevel::from_raw(99), BackpressureLevel::CRITICAL);
    }

    #[test]
    fn level_throttling_predicates() {
        assert!(!BackpressureLevel::NONE.is_admission_throttling());
        assert!(!BackpressureLevel::MILD.is_admission_throttling());
        assert!(BackpressureLevel::MODERATE.is_admission_throttling());
        assert!(BackpressureLevel::SEVERE.is_admission_throttling());
        assert!(BackpressureLevel::CRITICAL.is_admission_throttling());

        assert!(!BackpressureLevel::NONE.is_admission_refused());
        assert!(!BackpressureLevel::MODERATE.is_admission_refused());
        assert!(BackpressureLevel::SEVERE.is_admission_refused());
        assert!(BackpressureLevel::CRITICAL.is_admission_refused());

        assert!(!BackpressureLevel::SEVERE.is_cancellation_required());
        assert!(BackpressureLevel::CRITICAL.is_cancellation_required());
    }

    #[test]
    fn level_serde_transparent() {
        let level = BackpressureLevel::MODERATE;
        let json = serde_json::to_string(&level).unwrap();
        assert_eq!(json, "2");
        let back: BackpressureLevel = serde_json::from_str(&json).unwrap();
        assert_eq!(back, BackpressureLevel::MODERATE);
    }

    // ── BackpressureEvent tests ─────────────────────────────────────

    #[test]
    fn event_builder_pattern() {
        let event = BackpressureEvent::new(BackpressureReason::MetalCapacity, BackpressureLevel::SEVERE)
            .with_lane(0)
            .with_session("session-1")
            .with_details("metal queue full");
        assert_eq!(event.reason, BackpressureReason::MetalCapacity);
        assert_eq!(event.lane_ordinal, Some(0));
        assert_eq!(event.affected_session.as_deref(), Some("session-1"));
        assert_eq!(event.details, "metal queue full");
    }

    // ── BackpressureEventController tests ───────────────────────────

    #[test]
    fn new_controller_is_at_none() {
        let ctrl = BackpressureEventController::new();
        assert_eq!(ctrl.level(), BackpressureLevel::NONE);
        assert_eq!(ctrl.event_count(), 0);
        assert!(ctrl.events().is_empty());
    }

    #[test]
    fn report_event_recalculates_level() {
        let mut ctrl = BackpressureEventController::new();
        ctrl.report(
            BackpressureEvent::new(BackpressureReason::ArtifactCold, BackpressureLevel::MILD),
        );
        assert_eq!(ctrl.level(), BackpressureLevel::MILD);
        assert_eq!(ctrl.event_count(), 1);

        ctrl.report(
            BackpressureEvent::new(BackpressureReason::MetalCapacity, BackpressureLevel::SEVERE)
                .with_lane(0),
        );
        // MetalCapacity is Capacity severity → SEVERE.
        assert_eq!(ctrl.level(), BackpressureLevel::SEVERE);
        assert_eq!(ctrl.event_count(), 2);
    }

    #[test]
    fn report_severity_uses_max_category_not_max_event() {
        let mut ctrl = BackpressureEventController::new();
        // A "MILD" event with capacity reason should still be SEVERE.
        ctrl.report(
            BackpressureEvent::new(BackpressureReason::MetalCapacity, BackpressureLevel::MILD),
        );
        assert_eq!(ctrl.level(), BackpressureLevel::SEVERE);
    }

    #[test]
    fn report_quota_overrides_capacity() {
        let mut ctrl = BackpressureEventController::new();
        ctrl.report(
            BackpressureEvent::new(BackpressureReason::MetalCapacity, BackpressureLevel::SEVERE)
                .with_lane(0),
        );
        assert_eq!(ctrl.level(), BackpressureLevel::SEVERE);
        ctrl.report(
            BackpressureEvent::new(BackpressureReason::SessionQuota, BackpressureLevel::CRITICAL),
        );
        assert_eq!(ctrl.level(), BackpressureLevel::CRITICAL);
    }

    #[test]
    fn same_reason_different_lane_keeps_both() {
        let mut ctrl = BackpressureEventController::new();
        ctrl.report(
            BackpressureEvent::new(BackpressureReason::MetalCapacity, BackpressureLevel::SEVERE)
                .with_lane(0),
        );
        ctrl.report(
            BackpressureEvent::new(BackpressureReason::MetalCapacity, BackpressureLevel::SEVERE)
                .with_lane(1),
        );
        assert_eq!(ctrl.event_count(), 2);
    }

    #[test]
    fn same_reason_same_lane_overwrites() {
        let mut ctrl = BackpressureEventController::new();
        ctrl.report(
            BackpressureEvent::new(BackpressureReason::MetalCapacity, BackpressureLevel::SEVERE)
                .with_lane(0),
        );
        ctrl.report(
            BackpressureEvent::new(BackpressureReason::MetalCapacity, BackpressureLevel::CRITICAL)
                .with_lane(0)
                .with_details("escalated"),
        );
        assert_eq!(ctrl.event_count(), 1);
        let ev = ctrl.events()[0];
        assert_eq!(ev.level, BackpressureLevel::CRITICAL);
        assert_eq!(ev.details, "escalated");
    }

    #[test]
    fn clear_resets_to_none() {
        let mut ctrl = BackpressureEventController::new();
        ctrl.report(
            BackpressureEvent::new(BackpressureReason::MetalCapacity, BackpressureLevel::SEVERE)
                .with_lane(0),
        );
        assert_eq!(ctrl.level(), BackpressureLevel::SEVERE);
        ctrl.clear();
        assert_eq!(ctrl.level(), BackpressureLevel::NONE);
        assert_eq!(ctrl.event_count(), 0);
    }

    #[test]
    fn clear_before_removes_old_events() {
        let mut ctrl = BackpressureEventController::new();
        let old = BackpressureEvent::new(BackpressureReason::MetalCapacity, BackpressureLevel::SEVERE)
            .with_lane(0);
        let cutoff = old.timestamp;
        ctrl.report(old);

        // The cutoff is the old event's timestamp; clear_before
        // should remove events with timestamp < cutoff, so the
        // old event itself stays.
        ctrl.clear_before(cutoff);
        assert_eq!(ctrl.event_count(), 1);

        ctrl.clear_before(cutoff + std::time::Duration::from_nanos(1));
        assert_eq!(ctrl.event_count(), 0);
        assert_eq!(ctrl.level(), BackpressureLevel::NONE);
    }

    #[test]
    fn summary_lists_unique_reasons_and_lanes() {
        let mut ctrl = BackpressureEventController::new();
        ctrl.report(
            BackpressureEvent::new(BackpressureReason::MetalCapacity, BackpressureLevel::SEVERE)
                .with_lane(0),
        );
        ctrl.report(
            BackpressureEvent::new(BackpressureReason::MetalCapacity, BackpressureLevel::SEVERE)
                .with_lane(1),
        );
        ctrl.report(
            BackpressureEvent::new(BackpressureReason::AneCapacity, BackpressureLevel::SEVERE)
                .with_lane(1),
        );
        let summary = ctrl.summary();
        assert_eq!(summary.level, BackpressureLevel::SEVERE);
        assert_eq!(summary.active_event_count, 3);
        assert_eq!(summary.reasons.len(), 2);
        assert!(summary.reasons.contains(&BackpressureReason::MetalCapacity));
        assert!(summary.reasons.contains(&BackpressureReason::AneCapacity));
        let mut lanes = summary.affected_lanes.clone();
        lanes.sort();
        assert_eq!(lanes, vec![0, 1]);
    }

    #[test]
    fn capacity_eviction_keeps_newest() {
        let mut ctrl = BackpressureEventController::with_max_events(2);
        ctrl.report(
            BackpressureEvent::new(BackpressureReason::MetalCapacity, BackpressureLevel::SEVERE)
                .with_lane(0),
        );
        ctrl.report(
            BackpressureEvent::new(BackpressureReason::AneCapacity, BackpressureLevel::SEVERE)
                .with_lane(1),
        );
        // Adding a third event evicts the first.
        ctrl.report(
            BackpressureEvent::new(BackpressureReason::CpuCapacity, BackpressureLevel::SEVERE)
                .with_lane(2),
        );
        assert_eq!(ctrl.event_count(), 2);
        // Metal capacity (lane 0) was evicted; ane and cpu remain.
        let mut lanes: Vec<u32> = ctrl.events().iter().filter_map(|e| e.lane_ordinal).collect();
        lanes.sort();
        assert_eq!(lanes, vec![1, 2]);
    }

    // ── BackpressureController (latency) tests ──────────────────────

    #[test]
    fn latency_controller_starts_empty() {
        let ctrl = BackpressureController::new(50_000, 0.9);
        assert_eq!(ctrl.pending_count(), 0);
        assert_eq!(ctrl.avg_latency_ns(), 0.0);
        assert!(!ctrl.is_backpressure(0));
    }

    #[test]
    fn latency_controller_triggers_within_window() {
        let mut ctrl = BackpressureController::new(50, 0.9); // 50 us window
        ctrl.record_completion(BatchCompletionRecord::new("req-1", 1, 1000, 1_000_000_000));
        // current_time_ns - 1_000_000_000 = 0, which is < 50 * 1000 = 50_000
        assert!(ctrl.is_backpressure(1_000_000_000));
    }

    #[test]
    fn latency_controller_clears_after_window() {
        let mut ctrl = BackpressureController::new(50, 0.9);
        ctrl.record_completion(BatchCompletionRecord::new("req-1", 1, 1000, 1_000_000_000));
        // current_time_ns - 1_000_000_000 = 60_000_000 > 50_000, so no backpressure.
        assert!(!ctrl.is_backpressure(1_060_000_000));
    }

    #[test]
    fn latency_controller_average() {
        let mut ctrl = BackpressureController::new(50, 0.9);
        ctrl.record_completion(BatchCompletionRecord::new("req-1", 1, 100, 0));
        ctrl.record_completion(BatchCompletionRecord::new("req-2", 1, 300, 0));
        ctrl.record_completion(BatchCompletionRecord::new("req-3", 1, 500, 0));
        // Average = (100 + 300 + 500) / 3 = 300.
        assert_eq!(ctrl.avg_latency_ns(), 300.0);
    }

    #[test]
    fn latency_controller_caps_record_count() {
        let mut ctrl = BackpressureController::with_capacity(50, 0.9, 3);
        for i in 0..5 {
            ctrl.record_completion(BatchCompletionRecord::new(
                format!("req-{i}"),
                1,
                100,
                i as u64,
            ));
        }
        assert_eq!(ctrl.pending_count(), 3);
    }

    // ── SchedulingMetrics tests ─────────────────────────────────────

    #[test]
    fn scheduling_metrics_default_budget() {
        let m = SchedulingMetrics::new(50, 0.9);
        assert_eq!(m.max_num_scheduled_tokens(), SchedulingMetrics::DEFAULT_TOKEN_BUDGET);
        assert_eq!(m.num_running_requests(), 0);
        assert_eq!(m.current_admission_level(0), BackpressureLevel::NONE);
    }

    #[test]
    fn token_budget_reduces_under_backpressure() {
        let mut m = SchedulingMetrics::new(50, 0.9);
        m.backpressure_mut()
            .record_completion(BatchCompletionRecord::new("req-1", 1, 1000, 0));
        // With a fresh completion in the window, is_backpressure
        // returns true and the budget should be reduced.
        m.update_token_budget(0);
        // Initial 4096 → 4096 * 3/4 = 3072.
        assert_eq!(m.max_num_scheduled_tokens(), 3072);

        // Step again — still in backpressure window.
        m.update_token_budget(0);
        // 3072 * 3/4 = 2304.
        assert_eq!(m.max_num_scheduled_tokens(), 2304);
    }

    #[test]
    fn token_budget_does_not_drop_below_minimum() {
        let mut m = SchedulingMetrics::new(50, 0.9);
        m.backpressure_mut()
            .record_completion(BatchCompletionRecord::new("req-1", 1, 1000, 0));
        // Step many times to drain the budget.
        for _ in 0..30 {
            m.update_token_budget(0);
        }
        assert_eq!(m.max_num_scheduled_tokens(), SchedulingMetrics::MIN_TOKEN_BUDGET);
    }

    #[test]
    fn token_budget_restores_when_healthy() {
        let mut m = SchedulingMetrics::new(50, 0.9);
        m.backpressure_mut()
            .record_completion(BatchCompletionRecord::new("req-1", 1, 1000, 0));
        m.update_token_budget(0);
        let reduced = m.max_num_scheduled_tokens();
        assert!(reduced < SchedulingMetrics::DEFAULT_TOKEN_BUDGET);

        // Move far enough into the future that the latency window
        // no longer triggers.
        m.update_token_budget(1_000_000_000_000);
        let restored = m.max_num_scheduled_tokens();
        assert!(restored > reduced);
    }

    #[test]
    fn current_admission_level_reflects_backpressure() {
        let mut m = SchedulingMetrics::new(50, 0.9);
        assert_eq!(m.current_admission_level(0), BackpressureLevel::NONE);
        m.backpressure_mut()
            .record_completion(BatchCompletionRecord::new("req-1", 1, 1000, 0));
        assert_eq!(m.current_admission_level(0), BackpressureLevel::MILD);
    }
}
