//! Receipt manifest — stable across all execution levels (1–3).
//!
//! The top-level manifest contains 11 sections: source, target, calibration,
//! policy, memory, execution, numerical_evidence, runtime_evidence,
//! bridge_evidence, artifact, and certification.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::phase_ir::CompilationId;
use super::phase_types::PhaseId;

// ── Bridge receipt (shared with bridge_provider) ────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeReceipt {
    pub source_slot: u64,
    pub destination_slot: u64,
    pub requested_route: String,
    pub actual_route: String,
    pub materialized_bytes: u64,
    pub cpu_copy_bytes: u64,
    pub gpu_copy_bytes: u64,
    pub bridge_latency_ns: u64,
    pub zero_copy_verified: bool,
    pub verification_method: String,
    pub failure_reason: Option<String>,
}

// ── Sub-structures ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageGeometry {
    pub lane: usize,          // 20
    pub page: usize,          // 640
    pub trits_per_word: usize, // 20
}

impl Default for PageGeometry {
    fn default() -> Self {
        PageGeometry { lane: 20, page: 640, trits_per_word: 20 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SidecarLimits {
    pub per_region_bytes: u64,
    pub total_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateSearchLimits {
    pub max_candidates_per_page: usize,
    pub beam_width: usize,
    pub sensitivity_rank_cutoff: usize,
}

impl Default for CandidateSearchLimits {
    fn default() -> Self {
        CandidateSearchLimits {
            max_candidates_per_page: 3,
            beam_width: 1,
            sensitivity_rank_cutoff: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpillEvent {
    pub region: String,
    pub bytes_spilled: u64,
    pub spilled_at_ns: u64,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockMetric {
    pub block_index: usize,
    pub output_mse: f64,
    pub cosine_similarity: f64,
    pub residual_relative_error: f64,
    pub rmsnorm_stat_drift: f64,
    pub attention_score_divergence: f64,
    pub topk_logit_overlap: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollingMetric {
    pub window: String,
    pub mse: f64,
    pub count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseExecutionRecord {
    pub phase_id: PhaseId,
    pub phase_type: String,
    pub provider: String,
    pub started_at_ns: u64,
    pub completed_at_ns: u64,
    pub input_slots: Vec<u64>,
    pub output_slots: Vec<u64>,
    pub peak_bytes: u64,
    pub transition_count: u64,
}

// ── Manifest sections ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceSection {
    pub dense_checkpoint_digest: [u8; 32],
    pub qat_checkpoint: bool,
    pub qat_checkpoint_digest: [u8; 32],
    pub tokenizer_digest: [u8; 32],
    pub model_architecture_digest: [u8; 32],
    pub coreai_teacher_digests: Vec<(String, [u8; 32])>,
    pub compiler_binary_digest: [u8; 32],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetSection {
    pub device_family: String,
    pub os_build: String,
    pub metal_feature_set: String,
    pub coreai_configuration: String,
    pub page_geometry: PageGeometry,
    pub kernel_abi: u32,
    pub allowed_backends: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationSection {
    pub dataset_provenance: String,
    pub sampling_seed: u64,
    pub sequence_lengths: Vec<usize>,
    pub attention_masks_digest: [u8; 32],
    pub teacher_frontier_digest: [u8; 32],
    pub student_frontier_digest: [u8; 32],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicySection {
    pub objective_weights: ObjectiveWeights,
    pub sidecar_limits: SidecarLimits,
    pub scale_policy: String,
    pub deadzone_policy: String,
    pub rounding_algorithm: String,
    pub deterministic_seed: u64,
    pub candidate_search_limits: CandidateSearchLimits,
    pub fallback_decisions: Vec<String>,
}

/// The 8-term loss function from spec §9:
/// L = λ_out·L_block_output + λ_res·L_residual_stream + λ_attn·L_attention
///     + λ_norm·L_normalization_stats + λ_logit·L_KL + λ_rollout·L_generation
///     + λ_cost·C_runtime + λ_bytes·C_representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectiveWeights {
    pub lambda_output: f64,
    pub lambda_residual: f64,
    pub lambda_attention: f64,
    pub lambda_norm: f64,
    pub lambda_logit: f64,
    pub lambda_rollout: f64,
    pub lambda_cost: f64,
    pub lambda_bytes: f64,
    /// Per-modality weight overrides. Keys: "text", "image", "audio", "video", "embedding".
    /// Absent keys fall through to the top-level lambda values.
    pub per_modality: HashMap<String, ObjectiveWeights>,
}

impl ObjectiveWeights {
    /// Resolve the effective weights for a given modality.
    ///
    /// Returns the per-modality weights if present, otherwise the top-level defaults.
    pub fn resolve(&self, modality: &str) -> &ObjectiveWeights {
        self.per_modality.get(modality).unwrap_or(self)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySection {
    pub budget_bytes: u64,
    pub emergency_ceiling_bytes: u64,
    pub peak_resident_bytes: u64,
    pub peak_provider_bytes: HashMap<String, u64>,
    pub spills: Vec<SpillEvent>,
    pub microbatch_changes: Vec<String>,
    pub emergency_serialization_events: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionSection {
    pub phases: Vec<PhaseExecutionRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericalEvidenceSection {
    pub local_metrics: Vec<BlockMetric>,
    pub block_metrics: Vec<BlockMetric>,
    pub rolling_window_metrics: Vec<RollingMetric>,
    pub rollout_metrics: Vec<RollingMetric>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeEvidenceSection {
    pub kernel_timing_ns: HashMap<String, Vec<u64>>,
    pub dispatch_counts: HashMap<String, u64>,
    pub allocation_bytes: HashMap<String, u64>,
    pub bytes_modeled: u64,
    pub bytes_measured: u64,
    pub cost_model_residuals: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeEvidenceSection {
    pub receipts: Vec<BridgeReceipt>,
    pub bridge_proof_status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactSection {
    pub cimage_digest: [u8; 32],
    pub region_digests: Vec<(String, [u8; 32])>,
    pub page_counts: HashMap<String, usize>,
    pub sidecar_bytes: u64,
    pub scale_bytes: u64,
    pub total_effective_bits_per_param: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificationSection {
    pub level1_pass: bool,
    pub level2_pass: bool,
    pub level3_pass: bool,
    pub test_corpus_digest: [u8; 32],
}

// ── Master manifest ─────────────────────────────────────────────────────────

/// Top-level receipt manifest.
///
/// Stable across all levels. The certification section records which level
/// gates were passed and links to the exact test corpus digests.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MasterManifest {
    pub manifest_version: u32,
    pub compilation_id: CompilationId,
    pub created_at_ns: u64,
    pub source: SourceSection,
    pub target: TargetSection,
    pub calibration: CalibrationSection,
    pub policy: PolicySection,
    pub memory: MemorySection,
    pub execution: ExecutionSection,
    pub numerical_evidence: NumericalEvidenceSection,
    pub runtime_evidence: RuntimeEvidenceSection,
    pub bridge_evidence: BridgeEvidenceSection,
    pub artifact: ArtifactSection,
    pub certification: CertificationSection,
}

impl MasterManifest {
    /// Create a new manifest with a fresh compilation id and default geometry.
    pub fn new(
        source: SourceSection,
        target: TargetSection,
        calibration: CalibrationSection,
        policy: PolicySection,
    ) -> Self {
        MasterManifest {
            manifest_version: 1,
            compilation_id: CompilationId::next(),
            created_at_ns: 0,
            source,
            target,
            calibration,
            policy,
            memory: MemorySection {
                budget_bytes: 10_000_000_000,
                emergency_ceiling_bytes: 10_750_000_000,
                peak_resident_bytes: 0,
                peak_provider_bytes: HashMap::new(),
                spills: Vec::new(),
                microbatch_changes: Vec::new(),
                emergency_serialization_events: Vec::new(),
            },
            execution: ExecutionSection { phases: Vec::new() },
            numerical_evidence: NumericalEvidenceSection {
                local_metrics: Vec::new(),
                block_metrics: Vec::new(),
                rolling_window_metrics: Vec::new(),
                rollout_metrics: Vec::new(),
            },
            runtime_evidence: RuntimeEvidenceSection {
                kernel_timing_ns: HashMap::new(),
                dispatch_counts: HashMap::new(),
                allocation_bytes: HashMap::new(),
                bytes_modeled: 0,
                bytes_measured: 0,
                cost_model_residuals: Vec::new(),
            },
            bridge_evidence: BridgeEvidenceSection {
                receipts: Vec::new(),
                bridge_proof_status: "none".into(),
            },
            artifact: ArtifactSection {
                cimage_digest: [0u8; 32],
                region_digests: Vec::new(),
                page_counts: HashMap::new(),
                sidecar_bytes: 0,
                scale_bytes: 0,
                total_effective_bits_per_param: 0.0,
            },
            certification: CertificationSection {
                level1_pass: false,
                level2_pass: false,
                level3_pass: false,
                test_corpus_digest: [0u8; 32],
            },
        }
    }
}

// ── Distillation processing receipt ────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingReceipt {
    pub manifest_version: String,
    pub artifact_digest: String,
    pub source_checkpoint_digest: String,
    pub calibration_corpus_digest: String,
    pub deterministic_seed: u64,
    pub block_receipts: Vec<BlockReceipt>,
    pub global_metrics: GlobalMetrics,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockReceipt {
    pub block_index: usize,
    pub modality_tag: String,
    pub input_frontier_hash: String,
    pub output_frontier_hash: String,
    pub sidecar_fraction: f32,
    pub optimal_scale_dtype: String,
    pub numerical_drift: HashMap<String, f32>,
    pub execution_provenance: EngineExecutionLog,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineExecutionLog {
    pub backend_requested: String,
    pub backend_observed: String,
    pub zero_copy_verified: bool,
    pub wall_time_ms: f64,
    pub peak_arena_bytes: u64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalMetrics {
    pub final_perplexity: f32,
    pub mtp_baseline_acceptance_rate: f32,
    pub student_mtp_acceptance_rate: f32,
    pub tokens_per_joule_estimate: f32,
    pub bits_per_weight_effective: f32,
}
