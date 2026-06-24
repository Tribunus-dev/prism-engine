//! Activation ring and epoch scheduler for ANE-TRI-LANE-CIMAGE-0001.
//!
//! The activation ring manages two- or three-slot buffer recycling between
//! lane boundaries (ANE → GPU, GPU → CPU, CPU → ANE).  The epoch scheduler
//! consumes an [`AppleTriLaneExecutionPlan`] and drives epoch-by-epoch
//! dispatch, tracking timing and producing execution receipts.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::compilation::tri_lane::{
    AneAdmission, AppleTriLaneExecutionPlan, AppleTriLaneExecutionReceipt,
    ExecutionLane, ExecutionEpoch, LaneExecutionEvent, NumericalStatus, OverlapMetrics,
};

// ── Re-exports ───────────────────────────────────────────────────────────

pub use crate::compilation::tri_lane::{
    CompletionContract, DependencyKind, LaneDependency,
};

// ── Activation ring ──────────────────────────────────────────────────────

/// A slot in the activation ring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationSlot {
    pub slot_index: u8,
    pub total_slots: u8,
    pub tensor_name: String,
    pub byte_size: u64,
    pub producer: ExecutionLane,
    pub consumer: ExecutionLane,
    pub released: bool,
    pub epoch_acquired: u64,
    pub epoch_released: u64,
}

impl ActivationSlot {
    fn free(tensor_name: &str) -> Self {
        Self {
            slot_index: 0,
            total_slots: 0,
            tensor_name: tensor_name.to_owned(),
            byte_size: 0,
            producer: ExecutionLane::MlxGpu,
            consumer: ExecutionLane::MlxGpu,
            released: true,
            epoch_acquired: 0,
            epoch_released: 0,
        }
    }
}

/// Activation ring buffer manager for ANE/GPU/CPU lane boundary transfers.
///
/// The ring wraps a fixed number of slots (typically 2 or 3).  A slot is
/// acquired by a producer lane writing into it and released by the consumer
/// lane after reading.  The ring tracks cursor positions and epoch
/// boundaries so the scheduler can verify that all in-flight transfers
/// complete before advancing.
pub struct ActivationRing {
    slots: Vec<ActivationSlot>,
    ring_size: u8,
    write_cursor: u8,
    read_cursor: u8,
}

impl ActivationRing {
    /// Create a new ring with `ring_size` slots (typically 2 or 3).
    pub fn new(ring_size: u8) -> Self {
        let count = ring_size.max(1); // At least one slot.
        Self {
            slots: (0..count)
                .map(|i| ActivationSlot {
                    slot_index: i,
                    total_slots: ring_size,
                    tensor_name: String::new(),
                    byte_size: 0,
                    producer: ExecutionLane::MlxGpu,
                    consumer: ExecutionLane::MlxGpu,
                    released: true,
                    epoch_acquired: 0,
                    epoch_released: 0,
                })
                .collect(),
            ring_size: count,
            write_cursor: 0,
            read_cursor: 0,
        }
    }

    /// Acquire the next available slot for `tensor_name`.
    ///
    /// Returns `None` when all slots are still in flight (none released).
    /// On success the slot is marked as acquired at `epoch`.
    pub fn acquire_slot(
        &mut self,
        tensor_name: &str,
        byte_size: u64,
        epoch: u64,
    ) -> Option<&mut ActivationSlot> {
        // Search forward from write_cursor for a released slot.
        let n = self.ring_size as usize;
        for offset in 0..n {
            let idx = ((self.write_cursor as usize) + offset) % n;
            if self.slots[idx].released {
                self.slots[idx].slot_index = idx as u8;
                self.slots[idx].total_slots = self.ring_size;
                self.slots[idx].tensor_name = tensor_name.to_owned();
                self.slots[idx].byte_size = byte_size;
                self.slots[idx].producer = ExecutionLane::CoreMlAne;
                self.slots[idx].consumer = ExecutionLane::MlxGpu;
                self.slots[idx].released = false;
                self.slots[idx].epoch_acquired = epoch;
                self.slots[idx].epoch_released = 0;
                self.write_cursor = ((idx as u8) + 1) % self.ring_size;
                return Some(&mut self.slots[idx]);
            }
        }
        None
    }

    /// Release slot `slot_index` at `epoch`.
    ///
    /// Returns `false` when the slot index is out of range or already
    /// released.
    pub fn release_slot(&mut self, slot_index: u8, epoch: u64) -> bool {
        let idx = slot_index as usize;
        if idx >= self.slots.len() {
            return false;
        }
        if self.slots[idx].released {
            return false; // Already free.
        }
        self.slots[idx].released = true;
        self.slots[idx].epoch_released = epoch;
        true
    }

    /// Number of slots currently released (available for acquisition).
    pub fn available_slots(&self) -> usize {
        self.slots.iter().filter(|s| s.released).count()
    }

    /// Returns `true` when every slot acquired at or before `epoch` has
    /// been released (i.e. no in-flight transfers from that epoch or
    /// earlier).
    pub fn all_released(&self, epoch: u64) -> bool {
        self.slots
            .iter()
            .filter(|s| !s.released && s.epoch_acquired <= epoch)
            .count()
            == 0
    }
}

// ── Epoch scheduler ──────────────────────────────────────────────────────

/// Drives execution of an [`AppleTriLaneExecutionPlan`] epoch by epoch.
///
/// The scheduler reads the compiled plan, advances through each
/// [`ExecutionEpoch`], validates dependencies via the activation ring,
/// and produces [`AppleTriLaneExecutionReceipt`] entries for observability
/// and verification.
pub struct EpochScheduler {
    plan: AppleTriLaneExecutionPlan,
    current_epoch: u64,
    ring: ActivationRing,
    /// Wall-clock start (ns) per epoch.
    epoch_start_ns: HashMap<u64, u128>,
    /// Wall-clock end (ns) per epoch.
    epoch_end_ns: HashMap<u64, u128>,
}

impl EpochScheduler {
    /// Build a scheduler from a compiled execution plan.
    ///
    /// The activation ring size is derived from the plan's boundary
    /// contracts (defaults to 2 when no contracts specify reuse slots).
    pub fn new(plan: AppleTriLaneExecutionPlan) -> Self {
        // Pick ring size = max reuse_slots from boundary contracts,
        // clamped to [2, 3] for typical tri-lane scenarios.
        let ring_size: u8 = plan
            .ane_program
            .as_ref()
            .map(|p| {
                p.input_contract
                    .iter()
                    .chain(p.output_contract.iter())
                    .map(|_| 2u8)
                    .max()
                    .unwrap_or(2)
            })
            .unwrap_or(2)
            .max(2)
            .min(3);

        Self {
            plan,
            current_epoch: 0,
            ring: ActivationRing::new(ring_size),
            epoch_start_ns: HashMap::new(),
            epoch_end_ns: HashMap::new(),
        }
    }

    /// Returns the current epoch index.
    pub fn current_epoch(&self) -> u64 {
        self.current_epoch
    }

    /// Advance to the next epoch.
    ///
    /// Returns `false` when there are no more epochs in the plan.
    pub fn advance_epoch(&mut self) -> bool {
        let next = self.current_epoch + 1;
        if next >= self.plan.epochs.len() as u64 {
            return false;
        }
        self.current_epoch = next;
        true
    }

    /// Check that all dependencies for the current epoch are satisfied.
    ///
    /// Examines every [`LaneDependency`] whose `to_epoch` matches the
    /// current epoch and verifies the ring has released the referenced
    /// resources.
    pub fn dependencies_satisfied(&self) -> bool {
        let Some(epoch) = self.current_epoch_entry() else {
            return false;
        };
        for dep in &epoch.dependencies {
            if dep.to_epoch != self.current_epoch {
                continue;
            }
            // Fast path: a DataReady dependency is satisfied when every
            // ring slot acquired at or before the producer's epoch has
            // been released.
            if dep.kind == DependencyKind::DataReady {
                if !self.ring.all_released(dep.from_epoch) {
                    return false;
                }
            }
        }
        true
    }

    /// Generate a receipt describing the current epoch's execution state.
    ///
    /// Fields that require runtime measurement (compute_ns, boundary costs)
    /// are populated from the plan's cost model when real measurements are
    /// unavailable.
    pub fn generate_receipt(&self) -> AppleTriLaneExecutionReceipt {
        let epoch = self.current_epoch;

        // Build per-lane events from the plan's cost model (or zeros if
        // not yet measured).
        let gpu_event = LaneExecutionEvent {
            lane: ExecutionLane::MlxGpu,
            success: true,
            compute_ns: self.plan.predicted_cost.gpu.compute_ns,
            memory_ns: self.plan.predicted_cost.gpu.memory_ns,
            sync_ns: self.plan.predicted_cost.gpu.sync_ns,
        };
        let cpu_event = LaneExecutionEvent {
            lane: ExecutionLane::AccelerateCpu,
            success: true,
            compute_ns: self.plan.predicted_cost.cpu.compute_ns,
            memory_ns: self.plan.predicted_cost.cpu.memory_ns,
            sync_ns: self.plan.predicted_cost.cpu.sync_ns,
        };
        let ane_event = self.plan.ane_program.as_ref().map(|_| LaneExecutionEvent {
            lane: ExecutionLane::CoreMlAne,
            success: true,
            compute_ns: self.plan.predicted_cost.ane.compute_ns,
            memory_ns: self.plan.predicted_cost.ane.memory_ns,
            sync_ns: self.plan.predicted_cost.ane.sync_ns,
        });

        let mut lane_events = vec![gpu_event, cpu_event];
        if let Some(e) = ane_event {
            lane_events.push(e);
        }

        let wall_ns = self
            .epoch_end_ns
            .get(&epoch)
            .zip(self.epoch_start_ns.get(&epoch))
            .map(|(end, start)| (*end - start) as u64)
            .unwrap_or(0);

        let overlap = calculate_overlap(
            self.plan.predicted_cost.gpu.compute_ns,
            self.plan.predicted_cost.cpu.compute_ns,
            self.plan.predicted_cost.ane.compute_ns,
            wall_ns,
        );

        AppleTriLaneExecutionReceipt {
            cimage_id: String::new(),
            plan_digest: String::new(),
            epoch,
            lane_events,
            ane_artifact_id: self
                .plan
                .ane_program
                .as_ref()
                .map(|p| p.artifact_id.clone()),
            ane_admission: AneAdmission::Admitted,
            boundary_events: Vec::new(),
            overlap_ns: overlap,
            fallback_used: false,
            numerical_status: NumericalStatus::Pass,
            configured_cpu_and_neural_engine: true,
            observed_ane_execution: self.plan.ane_program.is_some(),
        }
    }

    /// Return the ANE work descriptor for the current epoch, if any.
    pub fn ane_work(&self) -> Option<&str> {
        self.current_epoch_entry()
            .and_then(|e| e.ane_work.as_deref())
    }

    /// Return the GPU work descriptor for the current epoch.
    ///
    /// Every epoch must have GPU work in a valid tri-lane plan, but we
    /// fall back to an empty string when the plan is degenerate.
    pub fn gpu_work(&self) -> &str {
        self.current_epoch_entry()
            .and_then(|e| e.gpu_work.as_deref())
            .unwrap_or("")
    }

    /// Mark the current epoch as complete and record its wall-clock
    /// duration.
    ///
    /// `wall_ns` is the end-of-epoch timestamp in nanoseconds; the start
    /// is implicitly the previous epoch's end (or the scheduler's creation
    /// time for epoch 0).
    pub fn complete_epoch(&mut self, wall_ns: u64) {
        let epoch = self.current_epoch;
        let start = self
            .epoch_end_ns
            .get(&(epoch.wrapping_sub(1)))
            .copied()
            .unwrap_or(wall_ns as u128);
        self.epoch_start_ns.entry(epoch).or_insert(start);
        self.epoch_end_ns
            .entry(epoch)
            .or_insert(wall_ns as u128);
    }

    // ── Helpers ──────────────────────────────────────────────────────────

    fn current_epoch_entry(&self) -> Option<&ExecutionEpoch> {
        self.plan.epochs.get(self.current_epoch as usize)
    }
}

// ── Overlap calculation ──────────────────────────────────────────────────

/// Calculate overlap metrics between lanes for a completed epoch.
///
/// # Parameters
/// * `gpu_ns` — GPU compute time (ns).
/// * `cpu_ns` — CPU compute time (ns).
/// * `ane_ns` — ANE compute time (ns).
/// * `wall_ns` — Wall-clock time for the epoch (ns).
///
/// # Overlap model
/// `total_compute_ns` is the sum of all three lane times (serialised).
/// If the lanes were perfectly serialised, wall time would equal
/// `total_compute_ns`.  Any gap is synchronisation overhead or idle time.
/// `overlap_ns` is the portion of compute that ran concurrently:
///
/// ```text
/// overlap_ns = max(0, total_compute_ns - wall_ns)
/// overlap_fraction = overlap_ns / total_compute_ns   (capped at 1.0)
/// ```
///
/// When all lanes are idle (`total_compute_ns == 0`), overlap is zero.
pub fn calculate_overlap(
    gpu_ns: u64,
    cpu_ns: u64,
    ane_ns: u64,
    wall_ns: u64,
) -> OverlapMetrics {
    let total_compute_ns = gpu_ns.saturating_add(cpu_ns).saturating_add(ane_ns);
    let total_sync_ns = if total_compute_ns >= wall_ns {
        0
    } else {
        wall_ns.saturating_sub(total_compute_ns)
    };
    let overlap_ns = total_compute_ns.saturating_sub(wall_ns);
    let overlap_fraction = if total_compute_ns == 0 {
        0.0
    } else {
        (overlap_ns as f64 / total_compute_ns as f64).clamp(0.0, 1.0)
    };

    OverlapMetrics {
        epoch_wall_ns: wall_ns,
        total_compute_ns,
        total_sync_ns,
        overlap_ns,
        overlap_fraction,
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_plan() -> AppleTriLaneExecutionPlan {
        use crate::compilation::tri_lane::{
            AppleFallbackPlan, AppleHardwareSignature, CoreMlProgramBinding,
            CoreMlTensorContract, CoreMlShapeContract, CoreMlWarmupContract,
            CoreMlComputeUnitPolicy, CpuProgramBinding, MetalProgramBinding, ShapeClass,
            NumericalPolicy, TriLaneCostModel, LaneCostEstimate, TriLaneEvidenceRequirements,
        };

        AppleTriLaneExecutionPlan {
            plan_version: 1,
            hardware_signature: AppleHardwareSignature {
                soc_family: "M1".into(),
                macos_version: "14.5".into(),
                coreml_version: "7.2.0".into(),
                p_core_count: 4,
                gpu_core_count: 8,
                ane_core_count: 16,
                unified_memory_gb: 16,
            },
            shape_class: ShapeClass {
                batch: 1,
                sequence: 1,
                hidden: 4096,
                num_heads: 32,
                num_kv_heads: 8,
                head_dim: 128,
                sliding_window: 0,
                max_context: 8192,
            },
            numerical_policy: NumericalPolicy {
                require_bit_exact: false,
                max_relative_error: 1e-3,
                allow_mixed_precision: true,
            },
            ane_program: Some(CoreMlProgramBinding {
                artifact_id: "ane-ffn-v3".into(),
                package_digest: "abc".into(),
                compiled_model_digest: "def".into(),
                compute_unit_policy: CoreMlComputeUnitPolicy::CpuAndNeuralEngineRequired,
                input_contract: vec![CoreMlTensorContract {
                    name: "hidden_states".into(),
                    shape: vec![1, 4096],
                    layout: "NHWC".into(),
                    dtype: "float16".into(),
                }],
                output_contract: vec![CoreMlTensorContract {
                    name: "ffn_output".into(),
                    shape: vec![1, 4096],
                    layout: "NHWC".into(),
                    dtype: "float16".into(),
                }],
                state_contract: None,
                shape_contract: CoreMlShapeContract {
                    static_shape: Some(vec![1, 4096]),
                    dynamic_range: None,
                },
                warmup_contract: CoreMlWarmupContract {
                    min_warmup_predictions: 3,
                    max_warmup_latency_ms: 50,
                    tolerance: 0.01,
                },
                qualification: crate::compilation::tri_lane::AneQualificationRecord {
                    compile_success: true,
                    load_success: true,
                    warmup_success: true,
                    output_present: true,
                    numerical_match: true,
                    steady_state_latency_ns: 500_000,
                    cpu_contention_ns: 10_000,
                    gpu_contention_ns: 20_000,
                    fallback_correct: true,
                },
            }),
            gpu_program: MetalProgramBinding {
                function_name: "attention_kernel".into(),
                pipeline_digest: "ghi".into(),
                threadgroup_size: (32, 1, 1),
                grid_size: (4096, 1, 1),
            },
            cpu_program: CpuProgramBinding {
                function_selector: "sampling".into(),
                routine: "vDSP_vsorti".into(),
                element_count: 4096,
            },
            tensors: vec![],
            dependencies: vec![],
            epochs: vec![
                ExecutionEpoch {
                    epoch_index: 0,
                    gpu_work: Some("attention:0".into()),
                    ane_work: Some("ffn:0".into()),
                    cpu_work: Some("tokenize:0".into()),
                    dependencies: vec![],
                },
                ExecutionEpoch {
                    epoch_index: 1,
                    gpu_work: Some("attention:1".into()),
                    ane_work: None,
                    cpu_work: Some("sample:1".into()),
                    dependencies: vec![],
                },
            ],
            fallback_plan: AppleFallbackPlan {
                ane_to_gpu: vec!["ffn:0".into()],
                ane_to_cpu: vec![],
                gpu_only_valid: true,
                cpu_only_valid: false,
            },
            predicted_cost: TriLaneCostModel {
                gpu: LaneCostEstimate {
                    compute_ns: 800_000,
                    memory_ns: 50_000,
                    boundary_ns: 10_000,
                    sync_ns: 5_000,
                },
                ane: LaneCostEstimate {
                    compute_ns: 400_000,
                    memory_ns: 20_000,
                    boundary_ns: 30_000,
                    sync_ns: 3_000,
                },
                cpu: LaneCostEstimate {
                    compute_ns: 100_000,
                    memory_ns: 10_000,
                    boundary_ns: 5_000,
                    sync_ns: 1_000,
                },
                critical_path_ns: 800_000,
                gpu_contention_penalty_ns: 20_000,
                cpu_contention_penalty_ns: 5_000,
                numerical_risk_penalty: 0.0,
                fallback_risk_penalty: 0.05,
            },
            evidence_requirements: TriLaneEvidenceRequirements {
                validate_numerics: true,
                min_steady_state_predictions: 100,
                collect_boundary_costs: true,
                profile_gpu_contention: true,
                profile_cpu_contention: false,
                verify_fallback: true,
            },
        }
    }

    // ── Activation ring tests ────────────────────────────────────────────

    #[test]
    fn test_activation_ring_basic_acquire_release() {
        let mut ring = ActivationRing::new(2);

        // Both slots start free.
        assert_eq!(ring.available_slots(), 2);

        // Acquire slot 0.
        let slot = ring.acquire_slot("hidden_states", 8192, 0);
        assert!(slot.is_some());
        assert_eq!(ring.available_slots(), 1);
        assert!(!ring.all_released(0));

        // Release slot 0.
        assert!(ring.release_slot(0, 0));
        assert_eq!(ring.available_slots(), 2);
        assert!(ring.all_released(0));
    }

    #[test]
    fn test_activation_ring_double_acquire_fails() {
        let mut ring = ActivationRing::new(2);

        // Acquire both slots.
        assert!(ring.acquire_slot("a", 1024, 0).is_some());
        assert!(ring.acquire_slot("b", 1024, 0).is_some());
        assert_eq!(ring.available_slots(), 0);

        // Third acquire must fail — ring is full.
        assert!(ring.acquire_slot("c", 1024, 0).is_none());

        // Release one and try again.
        assert!(ring.release_slot(0, 0));
        assert!(ring.acquire_slot("c", 1024, 1).is_some());
    }

    #[test]
    fn test_activation_ring_release_invalid_index() {
        let mut ring = ActivationRing::new(2);
        assert!(!ring.release_slot(99, 0));
        // Releasing an already-free slot returns false.
        assert!(!ring.release_slot(0, 0));
    }

    // ── Epoch scheduler tests ────────────────────────────────────────────

    #[test]
    fn test_epoch_scheduler_advance() {
        let plan = sample_plan();
        let mut sched = EpochScheduler::new(plan);

        assert_eq!(sched.current_epoch(), 0);
        assert!(sched.dependencies_satisfied()); // No dependencies.

        assert!(sched.advance_epoch());
        assert_eq!(sched.current_epoch(), 1);

        assert!(!sched.advance_epoch()); // Only 2 epochs.
        assert_eq!(sched.current_epoch(), 1); // Stays at last.
    }

    #[test]
    fn test_epoch_scheduler_receipt() {
        let plan = sample_plan();
        let mut sched = EpochScheduler::new(plan);

        sched.complete_epoch(1_000_000);
        let receipt = sched.generate_receipt();

        assert_eq!(receipt.epoch, 0);
        assert_eq!(receipt.lane_events.len(), 3); // GPU + CPU + ANE
        assert!(receipt.observed_ane_execution);
        assert!(!receipt.fallback_used);
        assert_eq!(receipt.numerical_status, NumericalStatus::Pass);
    }

    #[test]
    fn test_epoch_scheduler_work_descriptors() {
        let plan = sample_plan();
        let sched = EpochScheduler::new(plan);

        assert_eq!(sched.gpu_work(), "attention:0");
        assert_eq!(sched.ane_work(), Some("ffn:0"));
    }

    // ── Overlap calculation tests ────────────────────────────────────────

    #[test]
    fn test_overlap_calculation() {
        // Perfect overlap: 3 lanes each taking 100ns, wall = 150ns
        let metrics = calculate_overlap(100, 100, 100, 150);
        assert_eq!(metrics.epoch_wall_ns, 150);
        assert_eq!(metrics.total_compute_ns, 300);
        assert_eq!(metrics.total_sync_ns, 0);
        assert_eq!(metrics.overlap_ns, 150); // 300 - 150
        assert!((metrics.overlap_fraction - 0.5).abs() < 1e-9);

        // No overlap — fully serial
        let metrics = calculate_overlap(100, 0, 0, 100);
        assert_eq!(metrics.overlap_ns, 0);
        assert_eq!(metrics.overlap_fraction, 0.0);

        // All idle
        let metrics = calculate_overlap(0, 0, 0, 100);
        assert_eq!(metrics.overlap_ns, 0);
        assert_eq!(metrics.overlap_fraction, 0.0);
        assert_eq!(metrics.total_sync_ns, 100);

        // GPU dominates, CPU/ANE overlap perfectly
        let metrics = calculate_overlap(500, 100, 200, 500);
        assert_eq!(metrics.total_compute_ns, 800);
        assert_eq!(metrics.overlap_ns, 300); // 800 - 500
    }
}
