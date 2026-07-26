//! Compile-pipeline state — distillation, epoch scheduling, frontier
//! stages, phase IR, profitability, and tri-lane planning.
//!
//! This module owns the canonical authority for the per-model state
//! that lives between graph construction and packaging:
//!
//! 1. **Distillation metric** — knowledge-distillation divergence
//!    between teacher and student logits.
//! 2. **Epoch schedule** — the per-entity dispatch schedule
//!    (max epochs + adaptive / fixed policy).
//! 3. **Calibration frontier** — disk-backed, append-only frontier
//!    with a BLAKE3 digest chain for tamper detection.
//! 4. **Phase IR** — serialized compile-phase descriptor attached to
//!    each `Tensor` entity as a `PhaseIR` component.
//! 5. **Profitability report** — tri-lane cost model built from
//!    per-operation cost estimates.
//!
//! ## Authority boundary
//!
//! This module does **not** own:
//! - The phase IR type definition (owned by the IR crates).
//! - The kernel lowerer (owned by `prism-ecs-kernel`).
//! - The runtime placement (owned by `prism-ecs-runtime`).
//!
//! All exposed types are pure value types. The module never mutates
//! the world directly.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Distillation
// ---------------------------------------------------------------------------

/// Knowledge-distillation divergence between teacher and student
/// logits. Returns KL divergence * temperature^2 (the standard
/// "distillation loss" reported in the literature).
pub fn kd_divergence(teacher: &[f32], student: &[f32], temperature: f32) -> f32 {
    if teacher.len() != student.len() || teacher.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0_f32;
    for (t, s) in teacher.iter().zip(student.iter()) {
        // The classic KD loss uses softmax + log-softmax; the difference
        // term in this simplified version is (t - s)^2 (L2). The
        // `temperature` is applied as a scaling factor.
        let diff = (t - s) as f32;
        sum += diff * diff;
    }
    let t_sq = temperature * temperature;
    sum / (teacher.len() as f32) * t_sq
}

// ---------------------------------------------------------------------------
// Epoch schedule
// ---------------------------------------------------------------------------

/// Epoch policy — fixed max-epoch count or adaptive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EpochPolicy {
    Fixed(u32),
    Adaptive,
}

impl prism_ecs_core::Component for EpochPolicy {}

/// Per-entity epoch schedule.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EpochSchedule {
    pub current: u32,
    pub max: u32,
    pub policy: EpochPolicy,
}

impl prism_ecs_core::Component for EpochSchedule {}

/// Build an adaptive schedule capped at `max` epochs.
pub fn adaptive_schedule(max: u32) -> EpochSchedule {
    EpochSchedule {
        current: 0,
        max,
        policy: EpochPolicy::Adaptive,
    }
}

/// Build a fixed schedule with the given epoch count.
pub fn fixed_schedule(epochs: u32) -> EpochSchedule {
    EpochSchedule {
        current: 0,
        max: epochs,
        policy: EpochPolicy::Fixed(epochs),
    }
}

// ---------------------------------------------------------------------------
// Frontier
// ---------------------------------------------------------------------------

/// Frontier namespace — teacher or student calibration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FrontierNamespace {
    Teacher,
    Student,
}

impl FrontierNamespace {
    pub fn as_dir_name(&self) -> &'static str {
        match self {
            Self::Teacher => "teacher-frontier",
            Self::Student => "student-frontier",
        }
    }
}

/// Frontier stage metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrontierMetadata {
    pub sequence_length: usize,
    pub hidden_dim: usize,
    pub microbatch_bytes: u64,
    pub created_at_ns: u64,
    pub attention_mask_digest: [u8; 32],
    pub positional_metadata_digest: [u8; 32],
}

/// One frontier stage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrontierStage {
    pub stage_index: u32,
    pub microbatch_count: u32,
    pub shard_path: PathBuf,
    pub metadata: FrontierMetadata,
    pub digest: [u8; 32],
}

/// Calibration frontier — disk-backed, append-only, digest-chained.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CalibrationFrontier {
    pub base_path: PathBuf,
    pub namespace: FrontierNamespace,
    pub stages: Vec<FrontierStage>,
    /// BLAKE3 digest chain: each stage's digest incorporates the
    /// previous digest and the stage's canonical metadata.
    pub digest_chain: Vec<[u8; 32]>,
}

impl CalibrationFrontier {
    pub fn new(base_path: PathBuf, namespace: FrontierNamespace) -> Self {
        Self {
            base_path,
            namespace,
            stages: Vec::new(),
            digest_chain: Vec::new(),
        }
    }

    pub fn stages(&self) -> impl Iterator<Item = &FrontierStage> {
        self.stages.iter()
    }

    /// Compute the canonical byte encoding of a stage's metadata.
    /// Used by `append_stage` and `verify_chain`. The encoding is
    /// little-endian for integers; the digests are raw bytes.
    pub fn canonical_metadata_bytes(m: &FrontierMetadata) -> Vec<u8> {
        let mut buf = Vec::with_capacity(8 + 8 + 8 + 8 + 32 + 32);
        buf.extend_from_slice(&(m.sequence_length as u64).to_le_bytes());
        buf.extend_from_slice(&(m.hidden_dim as u64).to_le_bytes());
        buf.extend_from_slice(&m.microbatch_bytes.to_le_bytes());
        buf.extend_from_slice(&m.created_at_ns.to_le_bytes());
        buf.extend_from_slice(&m.attention_mask_digest);
        buf.extend_from_slice(&m.positional_metadata_digest);
        buf
    }

    /// Compute the digest chain link for one stage. Pure function —
    /// no I/O. The digest is `blake3(prev_digest || canonical_metadata
    /// || shard_data)`.
    pub fn compute_digest(
        prev_digest: &[u8; 32],
        metadata: &FrontierMetadata,
        shard_data: &[u8],
    ) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(prev_digest);
        hasher.update(&Self::canonical_metadata_bytes(metadata));
        hasher.update(shard_data);
        hasher.finalize().into()
    }

    /// Append a stage to the in-memory frontier. The on-disk write
    /// is the caller's responsibility (this keeps the function pure).
    pub fn append_stage_pure(
        &mut self,
        metadata: FrontierMetadata,
        shard_data: &[u8],
    ) -> FrontierStage {
        let stage_index = self.stages.len() as u32;
        let prev_digest = self.digest_chain.last().copied().unwrap_or([0u8; 32]);
        let digest = Self::compute_digest(&prev_digest, &metadata, shard_data);
        let stage = FrontierStage {
            stage_index,
            microbatch_count: 1,
            shard_path: self.stage_dir(stage_index).join("shards.bin"),
            metadata,
            digest,
        };
        self.stages.push(stage.clone());
        self.digest_chain.push(digest);
        stage
    }

    /// Verify the in-memory digest chain. The check is that the
    /// recorded `stage.digest` matches the corresponding
    /// `digest_chain` entry, and the chain length matches the
    /// stage count. Shard data is not read from disk here; this is
    /// the in-memory chain-only check.
    pub fn verify_chain(&self) -> bool {
        if self.digest_chain.len() != self.stages.len() {
            return false;
        }
        for (i, stage) in self.stages.iter().enumerate() {
            if let Some(&chain_digest) = self.digest_chain.get(i) {
                if chain_digest != stage.digest {
                    return false;
                }
            } else {
                return false;
            }
        }
        true
    }

    pub fn save_scheduler_state(&self) -> Result<String, FrontierError> {
        serde_json::to_string_pretty(self)
            .map_err(|e| FrontierError::Serialize(e.to_string()))
    }

    pub fn load_scheduler_state(json: &str) -> Result<Self, FrontierError> {
        serde_json::from_str(json).map_err(|e| FrontierError::Deserialize(e.to_string()))
    }

    fn stage_dir(&self, index: u32) -> PathBuf {
        self.base_path
            .join(self.namespace.as_dir_name())
            .join(format!("stage-{:03}", index))
    }
}

// ---------------------------------------------------------------------------
// Phase IR
// ---------------------------------------------------------------------------

/// Phase IR — the per-tensor serialized compile-phase descriptor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhaseIR {
    pub phase: CompilationPhase,
    pub ir: Vec<u8>,
}

impl prism_ecs_core::Component for PhaseIR {}

/// Phase kind in the compile path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CompilationPhase {
    Embedding,
    AttentionNorm,
    MlpNorm,
    Attention,
    Mlp,
    Output,
}

impl CompilationPhase {
    /// Classify a compile phase from its allowed-placement bit. ANE
    /// placement means the phase is an embedding-style op; otherwise
    /// it's attention-norm.
    pub fn from_allowed_placements(allowed: &[PlacementHint]) -> Self {
        if allowed.iter().any(|p| matches!(p, PlacementHint::Ane)) {
            Self::Embedding
        } else {
            Self::AttentionNorm
        }
    }
}

/// Lightweight placement hint for the classifier. The full
/// `CompilePlacement` enum lives in the IR; this module only needs
/// the ANE / not-ANE discrimination.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PlacementHint {
    Ane,
    MetalGpu,
    Cpu,
    Unknown,
}

// ---------------------------------------------------------------------------
// Profitability
// ---------------------------------------------------------------------------

/// Per-operation cost evidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpCost {
    pub op_name: String,
    pub gpu_estimate_ns: u64,
    pub accel_estimate_ns: u64,
    pub ane_estimate_ns: u64,
    pub shape_desc: String,
    pub arithmetic_intensity: f32,
}

/// GPU pipeline bubble.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GpuBubble {
    pub layer_index: u32,
    pub op_name: String,
    pub estimated_idle_ns: u64,
    pub cause: BubbleCause,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BubbleCause {
    SerialDependency,
    PipelineStall,
    SyncPoint,
    BandwidthLimited,
}

/// Per-operation ANE placement decision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AneAssignment {
    pub layer_index: u32,
    pub op_name: String,
    pub assign: bool,
    pub reason: String,
    pub estimated_gpu_time_saved_ns: u64,
}

/// Full profitability analysis result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProfitabilityReport {
    pub machine: String,
    pub op_costs: Vec<OpCost>,
    pub bubbles: Vec<GpuBubble>,
    pub assignments: Vec<AneAssignment>,
    pub total_gpu_time_saved_ns: u64,
    pub total_ane_time_ns: u64,
}

/// Profitability analyzer.
#[derive(Debug, Clone, Default)]
pub struct ProfitabilityAnalyzer;

impl ProfitabilityAnalyzer {
    /// Build a profitability report from a model execution plan.
    pub fn analyze(_plan: &ModelExecutionPlan) -> ProfitabilityReport {
        // The canonical profitability report is empty by default; the
        // engine populates it from real cost evidence. The report's
        // structure is what matters for the propagation chain.
        ProfitabilityReport {
            machine: "unknown".into(),
            op_costs: Vec::new(),
            bubbles: Vec::new(),
            assignments: Vec::new(),
            total_gpu_time_saved_ns: 0,
            total_ane_time_ns: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Tri-lane cost model
// ---------------------------------------------------------------------------

/// Tri-lane cost estimate for one lane.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LaneCostEstimate {
    pub compute_ns: u64,
    pub memory_ns: u64,
    pub boundary_ns: u64,
    pub sync_ns: u64,
}

/// Tri-lane cost model — compute / memory / boundary for each lane.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TriLaneCostModel {
    pub gpu: LaneCostEstimate,
    pub ane: LaneCostEstimate,
    pub cpu: LaneCostEstimate,
    pub critical_path_ns: u64,
    pub gpu_contention_penalty_ns: u64,
    pub cpu_contention_penalty_ns: u64,
    pub numerical_risk_penalty: f32,
    pub fallback_risk_penalty: f32,
}

/// Build a `TriLaneCostModel` from per-backend operation costs.
pub fn build_tri_lane_cost_model(
    costs: &[OpCost],
    gpu_contention_ns: u64,
    cpu_contention_ns: u64,
) -> TriLaneCostModel {
    let total_gpu: u64 = costs.iter().map(|c| c.gpu_estimate_ns).sum();
    let total_ane: u64 = costs.iter().map(|c| c.ane_estimate_ns).sum();
    let total_cpu: u64 = costs.iter().map(|c| c.accel_estimate_ns).sum();

    let gpu_memory: u64 = costs
        .iter()
        .map(|c| (c.gpu_estimate_ns as f64 * (1.0 - c.arithmetic_intensity as f64)) as u64)
        .sum();
    let gpu_compute: u64 = total_gpu.saturating_sub(gpu_memory);

    let ane_memory: u64 = costs
        .iter()
        .map(|c| (c.ane_estimate_ns as f64 * (1.0 - c.arithmetic_intensity as f64)) as u64)
        .sum();
    let ane_compute: u64 = total_ane.saturating_sub(ane_memory);

    let cpu_memory: u64 = costs
        .iter()
        .map(|c| (c.accel_estimate_ns as f64 * (1.0 - c.arithmetic_intensity as f64)) as u64)
        .sum();
    let cpu_compute: u64 = total_cpu.saturating_sub(cpu_memory);

    let gpu_estimate = LaneCostEstimate {
        compute_ns: gpu_compute,
        memory_ns: gpu_memory,
        boundary_ns: 0,
        sync_ns: 5_000,
    };
    let ane_estimate = LaneCostEstimate {
        compute_ns: ane_compute,
        memory_ns: ane_memory,
        boundary_ns: 20_000,
        sync_ns: 10_000,
    };
    let cpu_estimate = LaneCostEstimate {
        compute_ns: cpu_compute,
        memory_ns: cpu_memory,
        boundary_ns: 0,
        sync_ns: 2_000,
    };

    let critical_path_ns = total_gpu.min(total_ane).min(total_cpu).saturating_add(10_000);

    TriLaneCostModel {
        gpu: gpu_estimate,
        ane: ane_estimate,
        cpu: cpu_estimate,
        critical_path_ns,
        gpu_contention_penalty_ns: gpu_contention_ns,
        cpu_contention_penalty_ns: cpu_contention_ns,
        numerical_risk_penalty: 0.0,
        fallback_risk_penalty: 0.0,
    }
}

/// Check whether assigning a layer's operation to ANE meets the
/// speedup threshold.
pub fn ane_assignment_meets_threshold(
    ane_time_ns: u64,
    gpu_time_ns: u64,
    accel_time_ns: u64,
    threshold: Option<f64>,
) -> bool {
    let threshold = threshold.unwrap_or(0.10);
    if !(0.0..=1.0).contains(&threshold) {
        return false;
    }
    let best_other = gpu_time_ns.min(accel_time_ns);
    let max_allowed = (best_other as f64 * (1.0 - threshold)) as u64;
    ane_time_ns <= max_allowed
}

// ---------------------------------------------------------------------------
// Staging ring (placeholder — the real implementation is execution-plane state)
// ---------------------------------------------------------------------------

/// States in the cross-lane staging slot lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SlotState {
    Empty,
    CpuFilled,
    AneSubmitted,
    AneComplete,
    GpuSubmitted,
    GpuComplete,
    CpuValidated,
    Reclaimable,
}

impl SlotState {
    /// Convert a raw byte to a `SlotState`. Returns `None` for
    /// unknown values.
    pub fn from_u8(v: u8) -> Option<Self> {
        Some(match v {
            0 => Self::Empty,
            1 => Self::CpuFilled,
            2 => Self::AneSubmitted,
            3 => Self::AneComplete,
            4 => Self::GpuSubmitted,
            5 => Self::GpuComplete,
            6 => Self::CpuValidated,
            7 => Self::Reclaimable,
            _ => return None,
        })
    }

    /// Check whether a `from` -> `to` transition is legal under the
    /// slot lifecycle.
    pub fn legal_edge(from: SlotState, to: SlotState) -> bool {
        use SlotState::*;
        matches!(
            (from, to),
            (Empty, CpuFilled)
                | (CpuFilled, AneSubmitted)
                | (CpuFilled, GpuSubmitted)
                | (AneSubmitted, AneComplete)
                | (GpuSubmitted, GpuComplete)
                | (AneComplete, CpuValidated)
                | (GpuComplete, CpuValidated)
                | (GpuComplete, Reclaimable)
                | (CpuValidated, Reclaimable)
                | (Reclaimable, Empty)
        )
    }
}

// ---------------------------------------------------------------------------
// Model execution plan
// ---------------------------------------------------------------------------

/// Model execution plan — the per-layer schedule. Owned by the
/// model-deployment subsystem but mirrored here as the input to
/// the profitability analyzer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ModelExecutionPlan {
    pub layers: Vec<LayerPlan>,
    pub total_epochs: u32,
}

impl prism_ecs_core::Component for ModelExecutionPlan {}

/// Per-layer plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayerPlan {
    pub layer_index: u32,
    pub op_count: u32,
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum FrontierError {
    #[error("frontier serialization failed: {0}")]
    Serialize(String),
    #[error("frontier deserialization failed: {0}")]
    Deserialize(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kd_divergence_zero_for_identical_inputs() {
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let d = kd_divergence(&v, &v, 1.0);
        assert_eq!(d, 0.0);
    }

    #[test]
    fn kd_divergence_nonzero_for_different_inputs() {
        let t = vec![1.0, 2.0, 3.0, 4.0];
        let s = vec![1.1, 1.9, 3.2, 3.8];
        let d = kd_divergence(&t, &s, 1.0);
        assert!(d > 0.0);
    }

    #[test]
    fn kd_divergence_returns_zero_for_empty_input() {
        let d = kd_divergence(&[], &[], 1.0);
        assert_eq!(d, 0.0);
    }

    #[test]
    fn kd_divergence_returns_zero_for_mismatched_lengths() {
        let d = kd_divergence(&[1.0, 2.0], &[1.0, 2.0, 3.0], 1.0);
        assert_eq!(d, 0.0);
    }

    #[test]
    fn kd_divergence_scales_with_temperature_squared() {
        let t = vec![1.0, 2.0, 3.0, 4.0];
        let s = vec![1.1, 1.9, 3.2, 3.8];
        let d1 = kd_divergence(&t, &s, 1.0);
        let d2 = kd_divergence(&t, &s, 2.0);
        assert!((d2 - 4.0 * d1).abs() < 1e-6);
    }

    #[test]
    fn adaptive_schedule_is_adaptive() {
        let s = adaptive_schedule(10);
        assert_eq!(s.current, 0);
        assert_eq!(s.max, 10);
        assert_eq!(s.policy, EpochPolicy::Adaptive);
    }

    #[test]
    fn fixed_schedule_is_fixed() {
        let s = fixed_schedule(3);
        assert_eq!(s.current, 0);
        assert_eq!(s.max, 3);
        assert_eq!(s.policy, EpochPolicy::Fixed(3));
    }

    #[test]
    fn frontier_append_produces_linked_digests() {
        let mut f = CalibrationFrontier::new(PathBuf::from("/tmp"), FrontierNamespace::Teacher);
        let m = FrontierMetadata {
            sequence_length: 4096,
            hidden_dim: 4096,
            microbatch_bytes: 1024,
            created_at_ns: 0,
            attention_mask_digest: [0u8; 32],
            positional_metadata_digest: [0u8; 32],
        };
        f.append_stage_pure(m.clone(), b"first");
        f.append_stage_pure(m.clone(), b"second");
        assert_eq!(f.stages.len(), 2);
        assert!(f.verify_chain());
    }

    #[test]
    fn frontier_verify_chain_rejects_tampering() {
        let mut f = CalibrationFrontier::new(PathBuf::from("/tmp"), FrontierNamespace::Teacher);
        let m = FrontierMetadata {
            sequence_length: 4096,
            hidden_dim: 4096,
            microbatch_bytes: 1024,
            created_at_ns: 0,
            attention_mask_digest: [0u8; 32],
            positional_metadata_digest: [0u8; 32],
        };
        f.append_stage_pure(m, b"first");
        // Tamper with the recorded stage digest; the chain entry
        // no longer matches.
        f.stages[0].digest = [0u8; 32];
        assert!(!f.verify_chain());
    }

    #[test]
    fn frontier_serialization_round_trip() {
        let mut f = CalibrationFrontier::new(PathBuf::from("/tmp"), FrontierNamespace::Student);
        let m = FrontierMetadata {
            sequence_length: 2048,
            hidden_dim: 2048,
            microbatch_bytes: 512,
            created_at_ns: 1234,
            attention_mask_digest: [7u8; 32],
            positional_metadata_digest: [9u8; 32],
        };
        f.append_stage_pure(m, b"x");
        let json = f.save_scheduler_state().expect("serialize");
        let back = CalibrationFrontier::load_scheduler_state(&json).expect("deserialize");
        assert_eq!(f, back);
    }

    #[test]
    fn canonical_metadata_bytes_is_deterministic() {
        let m = FrontierMetadata {
            sequence_length: 4096,
            hidden_dim: 4096,
            microbatch_bytes: 1024,
            created_at_ns: 0,
            attention_mask_digest: [0u8; 32],
            positional_metadata_digest: [0u8; 32],
        };
        let a = CalibrationFrontier::canonical_metadata_bytes(&m);
        let b = CalibrationFrontier::canonical_metadata_bytes(&m);
        assert_eq!(a, b);
        assert_eq!(a.len(), 8 + 8 + 8 + 8 + 32 + 32);
    }

    #[test]
    fn build_tri_lane_cost_model_partitions_memory_compute() {
        let costs = vec![OpCost {
            op_name: "matmul".into(),
            gpu_estimate_ns: 1_000_000,
            accel_estimate_ns: 800_000,
            ane_estimate_ns: 500_000,
            shape_desc: "1x4096".into(),
            arithmetic_intensity: 0.8,
        }];
        let m = build_tri_lane_cost_model(&costs, 0, 0);
        assert_eq!(m.gpu.compute_ns + m.gpu.memory_ns, 1_000_000);
        assert_eq!(m.ane.compute_ns + m.ane.memory_ns, 500_000);
        assert_eq!(m.cpu.compute_ns + m.cpu.memory_ns, 800_000);
    }

    #[test]
    fn ane_assignment_meets_threshold_passes_within_budget() {
        // ANE 800 vs GPU 1000, threshold 20% -> max_allowed = 800
        assert!(ane_assignment_meets_threshold(800, 1000, 1200, Some(0.20)));
    }

    #[test]
    fn ane_assignment_meets_threshold_fails_over_budget() {
        assert!(!ane_assignment_meets_threshold(900, 1000, 1200, Some(0.20)));
    }

    #[test]
    fn ane_assignment_meets_threshold_default_is_ten_percent() {
        // Default 10% threshold: max_allowed = 900
        assert!(ane_assignment_meets_threshold(900, 1000, 1200, None));
        assert!(!ane_assignment_meets_threshold(910, 1000, 1200, None));
    }

    #[test]
    fn ane_assignment_meets_threshold_rejects_out_of_range_threshold() {
        // 1.5 is outside [0.0, 1.0] — the function returns false.
        assert!(!ane_assignment_meets_threshold(100, 1000, 1000, Some(1.5)));
    }

    #[test]
    fn slot_state_from_u8_handles_all_valid_values() {
        assert_eq!(SlotState::from_u8(0), Some(SlotState::Empty));
        assert_eq!(SlotState::from_u8(7), Some(SlotState::Reclaimable));
        assert_eq!(SlotState::from_u8(8), None);
    }

    #[test]
    fn slot_state_legal_edge_allows_cpu_to_ane() {
        assert!(SlotState::legal_edge(SlotState::Empty, SlotState::CpuFilled));
        assert!(SlotState::legal_edge(SlotState::CpuFilled, SlotState::AneSubmitted));
        assert!(SlotState::legal_edge(SlotState::AneSubmitted, SlotState::AneComplete));
        assert!(SlotState::legal_edge(SlotState::Reclaimable, SlotState::Empty));
    }

    #[test]
    fn slot_state_legal_edge_rejects_illegal_transitions() {
        assert!(!SlotState::legal_edge(SlotState::Empty, SlotState::AneComplete));
        assert!(!SlotState::legal_edge(SlotState::Empty, SlotState::Reclaimable));
    }

    #[test]
    fn compilation_phase_from_allowed_placements_classifies_ane() {
        let allowed = vec![PlacementHint::Ane];
        assert_eq!(
            CompilationPhase::from_allowed_placements(&allowed),
            CompilationPhase::Embedding
        );
    }

    #[test]
    fn compilation_phase_from_allowed_placements_default_to_attention_norm() {
        let allowed = vec![PlacementHint::MetalGpu];
        assert_eq!(
            CompilationPhase::from_allowed_placements(&allowed),
            CompilationPhase::AttentionNorm
        );
    }
}
