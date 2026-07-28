//! Phase program IR — the canonical launch IR for a compiled
//! executable.

use serde::{Deserialize, Serialize};

use crate::compute_image_runtime::ContentHash;
use crate::compute_image_runtime::ExecutionShapeClass;

/// Opaque program identifier.
pub type ProgramId = String;

/// Opaque arena plan identifier.
pub type ArenaPlanId = String;

/// Opaque residency plan identifier.
pub type ResidencyPlanId = String;

/// Opaque receipt identifier.
pub type ReceiptId = String;

/// Opaque state domain identifier.
pub type StateDomainId = String;

/// Phase program — the canonical launch IR.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseProgram {
    /// Program identifier.
    pub program_id: ProgramId,
    /// Content hash of the program.
    pub program_hash: ContentHash,
    /// Execution shape class this program targets.
    pub shape_class: ExecutionShapeClass,
    /// Ordered list of phase operations.
    pub phases: Vec<PhaseOperation>,
    /// Arena plan identifier.
    pub arena_plan_id: ArenaPlanId,
    /// Residency plan identifier.
    pub residency_plan_id: ResidencyPlanId,
}

/// A single phase operation in a phase program.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseOperation {
    /// Operation identifier.
    pub operation_id: String,
    /// Semantic operation kind.
    pub semantic: SemanticOperation,
    /// Execution lane this operation runs on.
    pub lane: ExecutionLane,
    /// Input tensor identifiers.
    pub inputs: Vec<String>,
    /// Output tensor identifiers.
    pub outputs: Vec<String>,
}

/// Semantic operation kinds — the language of phase programs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SemanticOperation {
    /// RMS normalization.
    RmsNorm,
    /// Rotary position embedding.
    Rope,
    /// Attention.
    Attention,
    /// Matrix multiplication.
    MatMul,
    /// Element-wise SiLU activation.
    Silu,
    /// Element-wise GeLU activation.
    Gelu,
    /// Softmax.
    Softmax,
    /// Embedding lookup.
    Embedding,
    /// Sampling.
    Sample,
    /// Custom user-defined operation.
    Custom,
}

/// Execution lane identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExecutionLane {
    /// Apple Neural Engine via Core ML.
    CoreAi,
    /// Apple Metal GPU.
    Metal,
    /// CPU via Accelerate framework.
    Cpu,
    /// Generic GPU.
    Gpu,
}

/// Serialized phase program — the on-disk form.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedPhaseProgram {
    /// Program identifier.
    pub program_id: ProgramId,
    /// Serialized program bytes.
    pub bytes: Vec<u8>,
    /// Content hash of the serialized program.
    pub program_hash: ContentHash,
    /// Format version of the serialization.
    pub format_version: u32,
    /// State domain identifier.
    pub state_domain_id: StateDomainId,
    /// Receipt identifier.
    pub receipt_id: ReceiptId,
}
