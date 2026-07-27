//! GeneratedExecutable — the backend-neutral identity of a generated
//! executable.
//!
//! This module owns the canonical authority for the immutable
//! identity of one generated executable: source digest, semantic
//! operation, codec family, packed-layout identity, entry point,
//! ABI, binding plan, target backend, machine requirements,
//! compiler identity, and compiled artifact digest. Every field is
//! part of the executable's content address.

use serde::{Deserialize, Serialize};

use super::binding_plan::BindingPlan;
use super::kernel_abi::KernelAbi;

/// Backend-neutral identity of one generated executable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluator::binding_plan::{BindingPlan, BindingSlot, ConstantSlot};
    use crate::evaluator::kernel_abi::{BufferBinding, DispatchGeometryPolicy, KernelAbi};

    fn sample_abi() -> KernelAbi {
        KernelAbi {
            version: 1,
            buffers: vec![BufferBinding {
                slot: 0,
                name: "input".to_string(),
                byte_size: 1024,
                optional: false,
            }],
            constants: vec![],
            threadgroup_memory: vec![],
            dispatch_geometry: DispatchGeometryPolicy::Fixed(1, 1, 1),
            threads_per_threadgroup: (32, 1, 1),
        }
    }

    fn sample_binding_plan() -> BindingPlan {
        BindingPlan {
            buffers: vec![BindingSlot {
                name: "input".to_string(),
                slot: 0,
                byte_size: 1024,
                alignment: 16,
            }],
            constants: vec![ConstantSlot {
                name: "tile_m".to_string(),
                index: 0,
                value: 64,
            }],
            output_buffer: "output".to_string(),
            output_size: 1024,
        }
    }

    #[test]
    fn generated_executable_is_constructible_and_serializable() {
        let exe = GeneratedExecutable {
            source_digest: [0u8; 32],
            operation_id: "prism.linear.nf4.v1".to_string(),
            codec_id: "nf4".to_string(),
            layout_id: "tile640".to_string(),
            entry_point: "linear_nf4".to_string(),
            abi: sample_abi(),
            binding_plan: sample_binding_plan(),
            backend_target: "metal".to_string(),
            machine_requirements: vec!["metal3".to_string()],
            compiler_identity: "prism-ane-builder-1.0".to_string(),
            artifact_digest: [1u8; 32],
        };

        let json = serde_json::to_string(&exe).expect("serialize");
        let restored: GeneratedExecutable = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored, exe);
    }

    #[test]
    fn generated_executable_distinguishes_two_executables() {
        let mut exe_a = GeneratedExecutable {
            source_digest: [0u8; 32],
            operation_id: "op".to_string(),
            codec_id: "nf4".to_string(),
            layout_id: "tile640".to_string(),
            entry_point: "ep".to_string(),
            abi: sample_abi(),
            binding_plan: sample_binding_plan(),
            backend_target: "metal".to_string(),
            machine_requirements: vec![],
            compiler_identity: "ci".to_string(),
            artifact_digest: [1u8; 32],
        };
        let exe_b = GeneratedExecutable {
            artifact_digest: [2u8; 32],
            ..exe_a.clone()
        };
        exe_a.artifact_digest = [1u8; 32];
        assert_ne!(exe_a, exe_b, "different artifact digests → distinct identities");
    }
}
