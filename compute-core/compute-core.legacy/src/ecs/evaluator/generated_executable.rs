//! Backend-neutral identity of generated code, ABI, bindings, geometry, codec,
//! and semantic operation.

use serde::{Deserialize, Serialize};

use crate::ecs::canonical::kernel_abi::KernelAbi;
use super::binding_plan::BindingPlan;

/// Backend-neutral identity of one generated executable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedExecutable {
    /// Digest of the source genome/program used to generate this executable.
    pub source_digest: [u8; 32],
    /// Semantic operation this executable performs.
    pub operation_id: String,
    /// Codec family (NF4, ternary, INT8, FP16).
    pub codec_id: String,
    /// Packed-layout identity.
    pub layout_id: String,
    /// Entry point name.
    pub entry_point: String,
    /// ABI contract — buffer bindings, constants, threadgroup geometry.
    pub abi: KernelAbi,
    /// Binding plan — maps buffer/constant slots to fixture fields.
    pub binding_plan: BindingPlan,
    /// Target backend for this executable.
    pub backend_target: String,
    /// Machine capability requirements.
    pub machine_requirements: Vec<String>,
    /// Compiler identity that produced this executable.
    pub compiler_identity: String,
    /// Digest of the compiled artifact bytes.
    pub artifact_digest: [u8; 32],
}
