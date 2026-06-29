//! PRISM-ANE-TRI-LANE-PRODUCTION-0001 WS4 — borrow-safe tri-lane orchestrator.
//!
//! The [`TriLaneOrchestrator`] manages execution of phase variants across
//! three lanes (Metal/GPU, ANE/Core ML, Accelerate/CPU).  The critical
//! borrow-safety rule: [`select_best_idx`](TriLaneOrchestrator::select_best_idx)
//! returns an index (`usize`), never a reference into `self`, so
//! [`submit`](TriLaneOrchestrator::submit) can clone the variant before
//! mutating `self.lane_queues`.

use std::collections::HashMap;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::backend::placement::ExecutionLane;
use crate::compilation::activation_abi::{ActivationAbi, SlotLeaseId};
use crate::compilation::ane_admission_gate::{LaneAdmissionGate, RiskPolicy};
use crate::compilation::phase_ir::PhaseId;
use crate::compilation::tri_lane::{EpochRouteOrigin, NumericalStatus};
use crate::compute_image::compile::portfolio::CoreMlArtifactKey;
use crate::scheduling::ane_artifact_cache::{AneArtifactCache, ArtifactKey, ArtifactResidencyState};
use crate::scheduling::memory_pool::MemoryPoolAllocator;

// ── Type aliases ────────────────────────────────────────────────────────────

/// Epoch identifier within a session.
pub type EpochId = u64;

/// Variant identifier within a phase set.
pub type VariantId = u64;

// ── Admission status ────────────────────────────────────────────────────────

/// Whether a phase variant has been admitted for execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdmissionStatus {
    /// Fully admitted and ready for dispatch.
    Admitted,
    /// Rejected with a reason string.
    Denied(String),
    /// Admission gate has not yet evaluated this variant.
    NotAttempted,
}

// ── Cost estimate ───────────────────────────────────────────────────────────

/// Predicted cost of executing a phase variant on a specific lane.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseCostEstimate {
    /// Estimated queue delay before execution begins.
    pub queue_delay_ns: u64,
    /// Estimated pure execution time.
    pub execution_ns: u64,
    /// Estimated layout conversion overhead.
    pub layout_conversion_ns: u64,
    /// Estimated residency (weight-loading) cost.
    pub residency_cost_ns: u64,
    /// Estimated synchronisation overhead.
    pub sync_cost_ns: u64,
    /// How much this variant is on the critical path (0.0 = off, 1.0 = full).
    pub critical_path_effect: f32,
    /// Overlap gain if executed concurrently with another lane (ns saved).
    pub overlap_gain_ns: u64,
    /// Risk that the qualification will fail at runtime (0.0 = none, 1.0 = certain).
    pub qualification_risk: f32,
}

// ── Phase variant ───────────────────────────────────────────────────────────

/// A single execution variant for a phase — a specific lane + artifact binding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseVariant {
    /// Which execution lane this variant targets.
    pub lane: ExecutionLane,
    /// Core ML artifact key for ANE variants.
    pub artifact_key: Option<CoreMlArtifactKey>,
    /// Metal pipeline function name for GPU variants.
    pub metal_pipeline: Option<String>,
    /// Accelerate kernel name for CPU variants.
    pub accelerate_kernel: Option<String>,
    /// Input activation ABI contract.
    pub input_abi: ActivationAbi,
    /// Output activation ABI contract.
    pub output_abi: ActivationAbi,
    /// Estimated cost on this lane.
    pub cost_estimate: PhaseCostEstimate,
    /// Admission status from the gate.
    pub admission: AdmissionStatus,
}

// ── Phase variant set ───────────────────────────────────────────────────────

/// A collection of variants for one phase — the scheduler picks the best.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseVariantSet {
    /// Unique phase identifier.
    pub phase_id: PhaseId,
    /// Candidate execution variants (one per lane or lane-configuration).
    pub variants: Vec<PhaseVariant>,
}

// ── Lane queues ─────────────────────────────────────────────────────────────

/// Per-lane queues of phase identifiers pending execution.
#[derive(Debug)]
pub struct LaneQueues {
    /// Queue of phase ids for the Metal/GPU lane.
    pub metal_queue: Vec<PhaseId>,
    /// Queue of phase ids for the ANE/Core ML lane.
    pub ane_queue: Vec<PhaseId>,
    /// Queue of phase ids for the Accelerate/CPU lane.
    pub accelerate_queue: Vec<PhaseId>,
}

// ── Readiness tracking ──────────────────────────────────────────────────────

/// Tracks how many dependencies each phase has satisfied.
#[derive(Debug)]
pub struct ReadinessState {
    /// Number of ready dependencies per phase.
    pub ready_counters: HashMap<PhaseId, u32>,
    /// Phases whose completion satisfies the given phase's dependency.
    pub completed_dependencies: HashMap<PhaseId, Vec<PhaseId>>,
}

// ── Slot lease manager ──────────────────────────────────────────────────────

/// Manages leases on IOSurface / arena slots held by in-flight phases.
#[derive(Debug)]
pub struct SlotLeaseManager {
    /// Map from lease id to the phase that holds it.
    pub leased_slots: HashMap<SlotLeaseId, PhaseId>,
}

// ── Receipt collector ───────────────────────────────────────────────────────

/// Accumulates execution receipts for observability and admission feedback.
#[derive(Debug)]
pub struct ReceiptCollector {
    /// Collected execution receipts.
    pub receipts: Vec<TriLaneExecutionReceipt>,
}

// ── Dispatch policy ─────────────────────────────────────────────────────────

/// Dispatch policy controlling variant selection behaviour.
#[derive(Debug)]
pub struct DispatchPolicy {
    /// Risk tolerance for ANE admission.
    pub risk_policy: RiskPolicy,
}

// ── Lane state (per-lane dispatch-time view) ────────────────────────────────

/// Snapshot of a lane's state at dispatch time.
#[derive(Debug, Default)]
pub struct LaneState {
    /// Number of phases currently queued for this lane.
    pub queue_depth: u32,
}

// ── Work completion ─────────────────────────────────────────────────────────

/// Record produced when a phase variant finishes execution.
#[derive(Debug)]
pub struct WorkCompletion {
    /// Phase that completed.
    pub phase_id: PhaseId,
    /// Which variant was executed.
    pub variant_id: VariantId,
    /// Lane it ran on.
    pub lane: ExecutionLane,
    /// Wall-clock start time.
    pub start_time: Instant,
    /// Wall-clock completion time.
    pub completion_time: Instant,
    /// Whether execution succeeded.
    pub success: bool,
    /// Output slot lease id (if any).
    pub output_slot: SlotLeaseId,
}

// ── Execution receipt ───────────────────────────────────────────────────────

/// Persistent receipt for a tri-lane execution event, used for observability
/// and downstream admission qualification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriLaneExecutionReceipt {
    /// Session that issued the execution.
    pub session_id: String,
    /// Epoch within the session.
    pub epoch_id: u64,
    /// Phase that was executed.
    pub phase_id: PhaseId,
    /// Which variant was chosen.
    pub variant_id: VariantId,
    /// Lane that executed.
    pub lane: ExecutionLane,
    /// Core ML artifact key (ANE lane only).
    pub artifact_key: Option<CoreMlArtifactKey>,
    /// Input slot lease ids consumed by this execution.
    pub input_slots: Vec<SlotLeaseId>,
    /// Output slot lease id produced by this execution.
    pub output_slot: SlotLeaseId,
    /// Input activation ABI contract.
    pub input_abi: ActivationAbi,
    /// Output activation ABI contract.
    pub output_abi: ActivationAbi,
    /// Whether a fallback was used.
    pub fallback_used: bool,
    /// Origin of the execution route (scheduling decision provenance).
    pub route_origin: EpochRouteOrigin,
    /// Numerical outcome of the execution.
    pub numerical_status: NumericalStatus,
}

// ── TriLaneExecutionPlan ────────────────────────────────────────────────────

/// Compiled execution plan for an active model runtime.
/// Constructed once at runtime init, consumed at every epoch.
pub struct TriLaneExecutionPlan {
    pub model_identity: String,
    pub phase_templates: Vec<PhaseVariantSet>,
    pub fallback_policy: crate::compilation::tri_lane::AppleFallbackPlan,
}

// ── TriLaneOrchestrator ─────────────────────────────────────────────────────

/// Three-lane orchestrator that selects the best variant for each phase,
/// dispatches it to the appropriate lane queue, and tracks completions.
///
/// # Borrow safety
///
/// [`submit`](TriLaneOrchestrator::submit) delegates selection to
/// [`select_best_idx`](TriLaneOrchestrator::select_best_idx), which returns
/// an index (`usize`) instead of a reference.  This avoids a borrow conflict
/// because the variant is cloned from `phase_set` (accessed via shared
/// reference) before the method mutates `self.lane_queues`.
pub struct TriLaneOrchestrator {
    /// Placeholder executor for the Metal/GPU lane.
    pub metal_executor: (),
    /// Placeholder executor for the ANE/Core ML lane.
    pub ane_executor: (),
    /// Placeholder executor for the Accelerate/CPU lane.
    pub accelerate_executor: (),
    /// The full phase DAG — ordered list of variant sets.
    pub phase_dag: Vec<PhaseVariantSet>,
    /// Per-lane queues of pending phase ids.
    pub lane_queues: LaneQueues,
    /// Dependency readiness tracking.
    pub readiness: ReadinessState,
    /// Slot lease manager.
    pub slot_leases: SlotLeaseManager,
    /// Dispatch policy and risk tolerance.
    pub dispatch_policy: DispatchPolicy,
    /// Accumulated execution receipts.
    pub receipts: ReceiptCollector,
    /// Channel receiver for work completions from lane callbacks.
    pub completion_rx: Option<mpsc::UnboundedReceiver<WorkCompletion>>,
    /// Admission gate for ANE qualification checks.
    pub admission_gate: LaneAdmissionGate,
    /// ANE artifact cache for residency state tracking.
    pub ane_cache: AneArtifactCache,
    /// KV cache memory pool with token-stealing allocation.
    pub memory_pool: MemoryPoolAllocator,
}

impl TriLaneOrchestrator {
    /// Create a new orchestrator with the given phase DAG, dispatch policy,
    /// admission gate, and ANE artifact cache.
    pub fn new(
        variant_sets: Vec<PhaseVariantSet>,
        policy: DispatchPolicy,
        admission_gate: LaneAdmissionGate,
        ane_cache: AneArtifactCache,
        max_vram_bytes: u64,
        fixed_overhead_bytes: u64,
    ) -> Self {
        Self {
            metal_executor: (),
            ane_executor: (),
            accelerate_executor: (),
            phase_dag: variant_sets,
            lane_queues: LaneQueues {
                metal_queue: Vec::new(),
                ane_queue: Vec::new(),
                accelerate_queue: Vec::new(),
            },
            readiness: ReadinessState {
                ready_counters: HashMap::new(),
                completed_dependencies: HashMap::new(),
            },
            slot_leases: SlotLeaseManager {
                leased_slots: HashMap::new(),
            },
            dispatch_policy: policy,
            receipts: ReceiptCollector {
                receipts: Vec::new(),
            },
            admission_gate,
            completion_rx: None,
            ane_cache,
            memory_pool: MemoryPoolAllocator::new(max_vram_bytes, fixed_overhead_bytes),
        }
    }

    /// Submit a phase set for execution: select the best variant index,
    /// clone the variant, and enqueue it on the appropriate lane.
    ///
    /// Returns a [`WorkCompletion`] describing the enqueued work.
    ///
    /// # Borrow safety
    ///
    /// `select_best_idx` returns `Option<usize>` (not a reference into self),
    /// so we clone before the mutable borrow on `self.lane_queues`.
    pub fn submit(&mut self, phase_set: &PhaseVariantSet) -> Result<WorkCompletion, String> {
        let best_idx = self
            .select_best_idx(phase_set)
            .ok_or_else(|| "no admissible variant".to_string())?;

        let variant = phase_set.variants[best_idx].clone();

        // Preflight memory budget check.
        if let Err(oom) = self.check_memory_budget() {
            eprintln!("[tri-lane] REJECTED: {}", oom);
            return Err(oom);
        }

        // Push the phase id onto the appropriate lane queue.
        match variant.lane {
            ExecutionLane::MlxGpu => {
                self.lane_queues.metal_queue.push(phase_set.phase_id);
            }
            ExecutionLane::CoreMlAne => {
                self.lane_queues.ane_queue.push(phase_set.phase_id);
            }
            ExecutionLane::AccelerateCpu => {
                self.lane_queues.accelerate_queue.push(phase_set.phase_id);
            }
            _ => {
                // Fall back to accelerate queue for unknown/untargeted lanes.
                self.lane_queues.accelerate_queue.push(phase_set.phase_id);
            }
        }

        // Reserve page allocation for the activated slot.
        if let Err(e) = self.memory_pool.resolve_pool_allocation(phase_set.phase_id.0 as usize, 4096) {
            eprintln!("[tri-lane] page allocation failed: {}", e);
        }

        Ok(WorkCompletion {
            phase_id: phase_set.phase_id,
            variant_id: best_idx as VariantId,
            lane: variant.lane,
            start_time: Instant::now(),
            completion_time: Instant::now(),
            success: true,
            output_slot: SlotLeaseId(0),
        })
    }

    /// Select the index of the best variant from `phase_set`.
    ///
    /// Filters out:
    /// - Non-`Admitted` variants (denied / not-attempted).
    /// - ANE variants whose artifact is not warmed in the cache.
    /// Then scores the remaining variants and returns the index with the
    /// highest score.
    ///
    /// Returns `None` when no variant passes the filters.
    pub fn select_best_idx(&self, phase_set: &PhaseVariantSet) -> Option<usize> {
        let mut best_idx: Option<usize> = None;
        let mut best_score: f64 = f64::NEG_INFINITY;

        for (i, variant) in phase_set.variants.iter().enumerate() {
            // Must be fully admitted.
            if !matches!(variant.admission, AdmissionStatus::Admitted) {
                continue;
            }

            // ANE variants must have a warmed artifact in the cache.
            if variant.lane == ExecutionLane::CoreMlAne {
                if !self.is_ane_artifact_warmed(&variant) {
                    continue;
                }
            }

            let lane_state = self.lane_state_for(variant.lane);
            let score = self.score(variant, &lane_state);

            if score > best_score {
                best_score = score;
                best_idx = Some(i);
            }
        }

        best_idx
    }

    /// Score a variant for dispatch priority.  Higher values are better.
    ///
    /// The formula penalises execution cost, layout conversion, residency
    /// overhead, sync cost, queue depth, and qualification risk, while
    /// rewarding overlap gain.
    pub fn score(&self, variant: &PhaseVariant, lane_state: &LaneState) -> f64 {
        let est = &variant.cost_estimate;

        // Base cost (lower is better → negative contribution to score).
        let base_cost = est.execution_ns as f64
            + est.layout_conversion_ns as f64
            + est.residency_cost_ns as f64;

        // Sync and queue delay penalties.
        let sync_penalty = est.sync_cost_ns as f64 * 1.5;
        let delay_penalty = est.queue_delay_ns as f64;
        let queue_penalty = lane_state.queue_depth as f64 * 50.0;

        // Overlap gain (positive contribution).
        let overlap_bonus = est.overlap_gain_ns as f64 * 0.5;

        // Qualification risk penalty.
        let risk_penalty = est.qualification_risk as f64 * 100.0;

        -(base_cost + sync_penalty + delay_penalty + queue_penalty + risk_penalty - overlap_bonus)
    }

    /// Poll for completed work.  Currently a stub returning an empty vec.
    pub fn poll_completions(&mut self) -> Vec<WorkCompletion> {
        Vec::new()
    }

    /// Preflight memory budget check before accepting new work.
    /// Returns Ok(total_bytes) if within budget, Err(OOM description) otherwise.
    pub fn check_memory_budget(&self) -> Result<u64, String> {
        let static_base: u64 = 6_500_000_000; // ~6.5 GB base model + runtime
        self.memory_pool.verify_memory_budget().map(|used| used + static_base)
    }

    // ── Private helpers ────────────────────────────────────────────────────

    /// Check whether an ANE variant has a warmed artifact in the cache.
    fn is_ane_artifact_warmed(&self, variant: &PhaseVariant) -> bool {
        let Some(core_key) = &variant.artifact_key else {
            // No artifact key → cannot be warmed.
            return false;
        };

        // Convert the CoreMlArtifactKey to the cache's ArtifactKey format.
        let cache_key = ArtifactKey {
            model_family: core_key.model_identity.clone(),
            packet_kind: format!("{:?}", core_key.packet_kind),
            layer_start: core_key.layer_start,
            layer_end: core_key.layer_end,
            shape_bucket: core_key.shape_bucket.batch,
            precision: String::from("fp16"),
        };

        matches!(
            self.ane_cache.get_state(&cache_key),
            Some(ArtifactResidencyState::Warmed)
        )
    }

    /// Build a [`LaneState`] snapshot for the given lane.
    fn lane_state_for(&self, lane: ExecutionLane) -> LaneState {
        let queue_depth = match lane {
            ExecutionLane::MlxGpu => self.lane_queues.metal_queue.len() as u32,
            ExecutionLane::CoreMlAne => self.lane_queues.ane_queue.len() as u32,
            ExecutionLane::AccelerateCpu => self.lane_queues.accelerate_queue.len() as u32,
            _ => self.lane_queues.accelerate_queue.len() as u32,
        };
        LaneState { queue_depth }
    }

    /// Apply a work completion: update readiness, release leases, record receipt.
    pub fn apply_completion(&mut self, completion: WorkCompletion) -> Result<(), String> {
        // 1. Update readiness state
        // 2. Release output lease
        // 3. Record receipt
        // Stub: just record the receipt
        let receipt = TriLaneExecutionReceipt {
            session_id: String::new(),
            epoch_id: 0,
            phase_id: completion.phase_id,
            variant_id: completion.variant_id,
            lane: completion.lane,
            artifact_key: None,
            input_slots: vec![],
            output_slot: completion.output_slot,
            input_abi: crate::compilation::activation_abi::ActivationAbi::MetalOnly(
                crate::compilation::activation_abi::MetalOnlyParams {
                    name: String::new(),
                    dtype: crate::compilation::phase_ir::TensorDtype::Float16,
                    byte_count: 0,
                },
            ),
            output_abi: crate::compilation::activation_abi::ActivationAbi::MetalOnly(
                crate::compilation::activation_abi::MetalOnlyParams {
                    name: String::new(),
                    dtype: crate::compilation::phase_ir::TensorDtype::Float16,
                    byte_count: 0,
                },
            ),
            fallback_used: false,
            route_origin: crate::compilation::tri_lane::EpochRouteOrigin::CoreMlAne,
            numerical_status: crate::compilation::tri_lane::NumericalStatus::Pass,
        };
        self.receipts.receipts.push(receipt);
        Ok(())
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compilation::ane_eligibility::{ShapeBucket, ShapeBucketFamily};
    use crate::compute_image::compile::portfolio::{PacketKind, WeightEncoding};
    use crate::scheduling::ane_artifact_cache::AneEvictionPolicy;

    // ── Helpers ──────────────────────────────────────────────────────────

    fn sample_abi() -> ActivationAbi {
        // Use the simplest variant available.
        ActivationAbi::DecodeActivationV1(
            crate::compilation::activation_abi::DecodeActivationV1Params {
                dtype: crate::compilation::phase_ir::TensorDtype::Float16,
                seq_bucket: 128,
                hidden_dim: 512,
                physical_layout: crate::compilation::activation_abi::PhysicalLayout::ContiguousRowMajor,
                alignment: 64,
                stride_constraint: None,
            },
        )
    }

    fn sample_artifact_key() -> CoreMlArtifactKey {
        CoreMlArtifactKey {
            model_identity: "test-model".into(),
            packet_kind: PacketKind::MlpGateUp,
            layer_start: 0,
            layer_end: 1,
            function_name: "test_fn".into(),
            shape_bucket: ShapeBucket {
                batch: 1,
                sequence: 128,
                hidden: 4096,
                rank: 1,
                family: ShapeBucketFamily::Decode,
            },
            input_abi: sample_abi(),
            output_abi: sample_abi(),
            weight_encoding: WeightEncoding::Float16,
            source_package_digest: "abc123".into(),
        }
    }

    fn sample_phase_variant(lane: ExecutionLane, exec_ns: u64, admission: AdmissionStatus) -> PhaseVariant {
        PhaseVariant {
            lane,
            artifact_key: None,
            metal_pipeline: None,
            accelerate_kernel: None,
            input_abi: sample_abi(),
            output_abi: sample_abi(),
            cost_estimate: PhaseCostEstimate {
                queue_delay_ns: 0,
                execution_ns: exec_ns,
                layout_conversion_ns: 1_000,
                residency_cost_ns: 500,
                sync_cost_ns: 200,
                critical_path_effect: 0.5,
                overlap_gain_ns: 0,
                qualification_risk: 0.0,
            },
            admission,
        }
    }

    fn make_orchestrator(phase_sets: Vec<PhaseVariantSet>) -> TriLaneOrchestrator {
        TriLaneOrchestrator::new(
            phase_sets,
            DispatchPolicy {
                risk_policy: RiskPolicy::ProductionOnly,
            },
        LaneAdmissionGate::new(RiskPolicy::ProductionOnly),
            AneArtifactCache::new(32, 1_000_000_000, AneEvictionPolicy::Lru),
            16_000_000_000,  // max_vram_bytes: 16 GB
            1_000_000_000,   // fixed_overhead_bytes: 1 GB
        )
    }

    // ── Tests ─────────────────────────────────────────────────────────────

    #[test]
    fn test_new_orchestrator_empty() {
        let orch = make_orchestrator(Vec::new());
        assert!(orch.lane_queues.metal_queue.is_empty());
        assert!(orch.lane_queues.ane_queue.is_empty());
        assert!(orch.lane_queues.accelerate_queue.is_empty());
        assert!(orch.phase_dag.is_empty());
    }

    #[test]
    fn test_score_prefers_lower_cost() {
        let orch = make_orchestrator(Vec::new());

        let cheap = sample_phase_variant(ExecutionLane::MlxGpu, 100, AdmissionStatus::Admitted);
        let costly = sample_phase_variant(ExecutionLane::MlxGpu, 500, AdmissionStatus::Admitted);
        let lane_state = LaneState::default();

        let cheap_score = orch.score(&cheap, &lane_state);
        let costly_score = orch.score(&costly, &lane_state);

        // Lower execution cost → higher (less negative) score.
        assert!(
            cheap_score > costly_score,
            "cheap variant ({}) should score higher than costly variant ({})",
            cheap_score,
            costly_score
        );
    }

    #[test]
    fn test_select_filters_denied() {
        let phase_set = PhaseVariantSet {
            phase_id: PhaseId(1),
            variants: vec![
                sample_phase_variant(ExecutionLane::MlxGpu, 100, AdmissionStatus::Denied("no".into())),
                sample_phase_variant(ExecutionLane::MlxGpu, 200, AdmissionStatus::Admitted),
            ],
        };
        let orch = make_orchestrator(vec![phase_set]);

        let idx = orch.select_best_idx(&orch.phase_dag[0]);
        assert_eq!(idx, Some(1), "should select the admitted variant (index 1), not the denied one");
    }

    #[test]
    fn test_select_filters_cold_ane() {
        let key = sample_artifact_key();
        let phase_set = PhaseVariantSet {
            phase_id: PhaseId(2),
            variants: vec![
                sample_phase_variant(ExecutionLane::CoreMlAne, 50, AdmissionStatus::Admitted),
                sample_phase_variant(ExecutionLane::MlxGpu, 150, AdmissionStatus::Admitted),
            ],
        };
        // Make the ANE variant carry an artifact key.
        let phase_set = PhaseVariantSet {
            variants: vec![
                PhaseVariant {
                    artifact_key: Some(key),
                    ..sample_phase_variant(ExecutionLane::CoreMlAne, 50, AdmissionStatus::Admitted)
                },
                sample_phase_variant(ExecutionLane::MlxGpu, 150, AdmissionStatus::Admitted),
            ],
            ..phase_set
        };
        let orch = make_orchestrator(vec![phase_set]);

        // Cache is empty → ANE variant is cold → filtered → selects metal.
        let idx = orch.select_best_idx(&orch.phase_dag[0]);
        assert_eq!(idx, Some(1), "should select the metal variant (index 1) because ANE is cold");
    }

    #[test]
    fn test_submit_dispatches_to_selected_lane() {
        let phase_set = PhaseVariantSet {
            phase_id: PhaseId(10),
            variants: vec![
                sample_phase_variant(ExecutionLane::MlxGpu, 100, AdmissionStatus::Admitted),
                sample_phase_variant(ExecutionLane::AccelerateCpu, 300, AdmissionStatus::Admitted),
            ],
        };
        let mut orch = make_orchestrator(vec![phase_set]);

        let result = orch.submit(&orch.phase_dag[0].clone());
        assert!(result.is_ok(), "submit should succeed: {:?}", result);

        // The best variant is MlxGpu (cost 100), so it goes to metal_queue.
        assert_eq!(orch.lane_queues.metal_queue.len(), 1);
        assert_eq!(orch.lane_queues.metal_queue[0], PhaseId(10));
        assert!(orch.lane_queues.ane_queue.is_empty());
    }

    #[test]
    fn test_poll_completions_returns_empty() {
        let mut orch = make_orchestrator(Vec::new());
        let completions = orch.poll_completions();
        assert!(completions.is_empty());
    }

    #[test]
    fn test_tri_lane_receipt_serde_roundtrip() {
        let receipt = TriLaneExecutionReceipt {
            session_id: "sess-001".into(),
            epoch_id: 42,
            phase_id: PhaseId(7),
            variant_id: 0,
            lane: ExecutionLane::CoreMlAne,
            artifact_key: Some(sample_artifact_key()),
            input_slots: vec![SlotLeaseId(1), SlotLeaseId(2)],
            output_slot: SlotLeaseId(3),
            input_abi: sample_abi(),
            output_abi: sample_abi(),
            fallback_used: false,
            route_origin: EpochRouteOrigin::CoreMlAne,
            numerical_status: NumericalStatus::Pass,
        };

        let json = serde_json::to_string(&receipt).expect("serialize receipt");
        let deserialized: TriLaneExecutionReceipt =
            serde_json::from_str(&json).expect("deserialize receipt");

        assert_eq!(deserialized.session_id, "sess-001");
        assert_eq!(deserialized.epoch_id, 42);
        assert_eq!(deserialized.phase_id, PhaseId(7));
        assert_eq!(deserialized.variant_id, 0);
        assert_eq!(deserialized.lane, ExecutionLane::CoreMlAne);
        assert_eq!(deserialized.artifact_key, Some(sample_artifact_key()));
        assert_eq!(deserialized.input_slots, vec![SlotLeaseId(1), SlotLeaseId(2)]);
        assert_eq!(deserialized.output_slot, SlotLeaseId(3));
        assert_eq!(deserialized.fallback_used, false);
        assert_eq!(deserialized.route_origin, EpochRouteOrigin::CoreMlAne);
        assert_eq!(deserialized.numerical_status, NumericalStatus::Pass);
    }

    #[test]
    fn test_select_best_idx_returns_none_when_none_admitted() {
        let phase_set = PhaseVariantSet {
            phase_id: PhaseId(99),
            variants: vec![
                sample_phase_variant(ExecutionLane::MlxGpu, 100, AdmissionStatus::Denied("bad".into())),
                sample_phase_variant(ExecutionLane::AccelerateCpu, 200, AdmissionStatus::NotAttempted),
            ],
        };
        let orch = make_orchestrator(vec![phase_set]);

        let idx = orch.select_best_idx(&orch.phase_dag[0]);
        assert!(idx.is_none(), "no variant admitted → should return None");
    }
}
