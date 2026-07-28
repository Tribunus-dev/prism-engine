//! Provenance, receipt, and lifecycle contract types.
//!
//! Every compilation, evaluation, and promotion event produces evidence
//! captured in these types. No evidence is reconstructed post-hoc — the
//! artifacts reference each other by digest, forming an auditable chain.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use prism_ecs_constitutional::canonical::generation::CimageGeneration;
use prism_ecs_constitutional::canonical::identity::{
    CandidateId, GenerationId, LogicalTensorId, PhysicalSegmentId, ReceiptId,
};
use prism_ecs_constitutional::canonical::kernel_abi::{ArtifactProvenance, KernelAbi, KernelSemanticId};
use prism_ecs_ir::evolution::receipts::{NumericalReceipt, PerformanceReceipt};

/// Aggregate of all evidence produced during one compilation lifecycle.
///
/// A gate accepts or rejects based on the complete bundle — no gate
/// makes a decision without seeing compiler, numerical, quality,
/// performance, policy, and promotion evidence together.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecycleReceiptBundle {
    /// Receipt ids for each stage.
    pub compiler_receipt: ReceiptId,
    pub numerical_receipt: ReceiptId,
    pub quality_receipt: ReceiptId,
    pub performance_receipt: ReceiptId,
    pub policy_receipt: ReceiptId,
    pub promotion_receipt: ReceiptId,
    /// The generation this bundle applies to.
    pub generation_id: GenerationId,
    /// When the bundle was sealed.
    pub sealed_at: String,
}

impl LifecycleReceiptBundle {
    /// Verify that every receipt field is non-empty.
    ///
    /// A bundle with any empty/missing receipt is rejected as incomplete.
    pub fn verify_complete(&self) -> Result<(), String> {
        if self.compiler_receipt.0.is_empty() {
            return Err("compiler_receipt is empty".into());
        }
        if self.numerical_receipt.0.is_empty() {
            return Err("numerical_receipt is empty".into());
        }
        if self.quality_receipt.0.is_empty() {
            return Err("quality_receipt is empty".into());
        }
        if self.performance_receipt.0.is_empty() {
            return Err("performance_receipt is empty".into());
        }
        if self.policy_receipt.0.is_empty() {
            return Err("policy_receipt is empty".into());
        }
        if self.promotion_receipt.0.is_empty() {
            return Err("promotion_receipt is empty".into());
        }
        if self.generation_id.0.is_empty() {
            return Err("generation_id is empty".into());
        }
        if self.sealed_at.is_empty() {
            return Err("sealed_at is empty".into());
        }
        Ok(())
    }
}

/// Persisted record of one measured candidate in an evolutionary search.
///
/// Survives serialization so the search, frontier, replay, and promotion
/// paths all reference the same evidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasuredCandidateRecord {
    /// Candidate identity (hash of the genome).
    pub candidate_id: CandidateId,
    /// Provenance chain for every compiled artifact in this candidate.
    pub provenance: Vec<ArtifactProvenance>,
    /// Numeric receipt from the evaluator.
    pub numerical_receipt_id: ReceiptId,
    /// Performance receipt from the evaluator.
    pub performance_receipt_id: ReceiptId,
    /// Quality receipt from the evaluator.
    pub quality_receipt_id: ReceiptId,
    /// Optional rejection reason if the candidate was rejected.
    pub rejection_reason: Option<String>,
    /// Pareto rank if the candidate was part of a multi-objective frontier.
    pub pareto_rank: Option<usize>,
    /// When this record was created.
    pub created_at: String,
}

/// A request to promote a new generation.
///
/// Carries everything required for the promotion gate to verify
/// identity, evidence, and policy compliance before committing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromotionRequest {
    /// Parent generation identity.
    pub parent_generation: GenerationId,
    /// Content digests of all payload segments in the new generation.
    pub payload_digests: Vec<(PhysicalSegmentId, String)>,
    /// Artifact digests for the compiled kernel artifacts.
    pub artifact_digests: Vec<(KernelSemanticId, String)>,
    /// Identity of the promotion policy that must be satisfied.
    pub policy_id: String,
    /// Identity of the receipt bundle containing all evidence.
    pub receipt_bundle_id: ReceiptId,
    /// The new generation to promote.
    pub generation: CimageGeneration,
}

/// Manifest that resolves everything required to reproduce a promoted
/// generation without ambient state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayManifest {
    /// The generation to replay.
    pub generation: CimageGeneration,
    /// Source payloads keyed by segment id.
    pub payloads: BTreeMap<PhysicalSegmentId, Vec<u8>>,
    /// Compiled kernel artifacts keyed by semantic id.
    pub artifacts: BTreeMap<KernelSemanticId, ArtifactProvenance>,
    /// Compiled kernel artifact bytes keyed by semantic ID.
    /// Present when the replay has access to the original compiled artifacts.
    /// When absent, the replay must recompile from the catalogue source.
    pub compiled_artifacts: BTreeMap<KernelSemanticId, Vec<u8>>,
    /// ABI contracts for replay dispatch.
    pub abi: KernelAbi,
    /// The receipt bundle that was accepted at promotion time.
    pub receipt_bundle: LifecycleReceiptBundle,
    /// The numerical receipt accepted at promotion time.
    pub numerical_receipt: NumericalReceipt,
    /// The performance receipt accepted at promotion time.
    pub performance_receipt: PerformanceReceipt,
    /// Whether this replay is expected to produce identical numerical output.
    pub expects_numerical_parity: bool,
}

/// Typed cimage payload bindings for phase execution.
///
/// Every weight, scale, activation, KV block, and compiled kernel
/// reference is resolved to a specific offset, shape, and dtype
/// from the loaded cimage. No zero/symbolic tensors in production.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionBindings {
    /// Generation identity these bindings belong to.
    pub generation_id: GenerationId,
    /// Weight offsets: logical tensor → (segment_id, byte_offset, byte_length)
    pub weight_offsets: BTreeMap<LogicalTensorId, (PhysicalSegmentId, u64, u64)>,
    /// Scale offsets: logical tensor → (segment_id, byte_offset, byte_length)
    pub scale_offsets: BTreeMap<LogicalTensorId, (PhysicalSegmentId, u64, u64)>,
    /// Activation binding metadata.
    pub activations: ExecutionActivationSet,
    /// KV block binding metadata.
    pub kv_state: ExecutionKvSet,
    /// Compiled kernel semantic ids and their provenance.
    pub kernels: BTreeMap<KernelSemanticId, ArtifactProvenance>,
    /// Maximum layer count for dynamic KV allocation.
    pub max_layers: usize,
}

/// Activation binding set for a single phase execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionActivationSet {
    /// Number of activation buffers needed.
    pub num_buffers: usize,
    /// Total activation byte size across all buffers.
    pub total_bytes: u64,
    /// Per-buffer metadata.
    pub buffers: Vec<ActivationBufferMeta>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationBufferMeta {
    pub logical_id: LogicalTensorId,
    pub byte_offset: u64,
    pub byte_size: u64,
    pub dtype: String,
}

/// KV block binding metadata for a phase execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionKvSet {
    /// Total KV cache bytes.
    pub total_bytes: u64,
    /// Number of KV layers.
    pub num_layers: usize,
    /// Per-layer block count and byte size.
    pub layers: Vec<KvLayerMeta>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvLayerMeta {
    pub layer_idx: usize,
    pub num_blocks: usize,
    pub bytes_per_block: u64,
    pub dtype: String,
}
