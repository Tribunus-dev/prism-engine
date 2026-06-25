//! PRISM-ACTIVATION-ABI-0001: Lane-specific ABI contracts.
//!
//! Defines the ABI contract for activation tensors crossing lane boundaries
//! between ANE, GPU, and CPU execution lanes on Apple Silicon. Each variant
//! captures the physical layout and alignment constraints for a specific
//! activation class (decode, attention, vision, or metal-only).
//!
//! The [`ActivationContract`] bundles a full ABI description with computed
//! element/byte counts, shape, and stride — and provides a
//! [`validate_contract`](ActivationContract::validate_contract) method to
//! verify that a candidate contract matches exactly.

use serde::{Deserialize, Serialize};

use crate::backend::placement::ExecutionLane;
use crate::compilation::phase_ir::TensorDtype;
use crate::compilation::tri_lane::MaterializationMode;
use crate::compute_image::apple_shared_arena::SlotState;

// ── Activation ABI variants ──────────────────────────────────────────────

/// Per-variant ABI for an activation tensor crossing a lane boundary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ActivationAbi {
    /// Decode-step activation (KV-cache projections, MLP intermediates).
    DecodeActivationV1(DecodeActivationV1Params),
    /// MHA / GQA attention heads.
    AttentionHeads(AttentionHeadsParams),
    /// Vision encoder/decoder image tensors.
    VisionImage(VisionImageParams),
    /// Opaque metal-only buffer (no tensor semantics).
    MetalOnly(MetalOnlyParams),
}

/// Parameters for a decode-step activation V1.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecodeActivationV1Params {
    pub dtype: TensorDtype,
    pub seq_bucket: u32,
    pub hidden_dim: u32,
    pub physical_layout: PhysicalLayout,
    pub alignment: u32,
    pub stride_constraint: Option<Vec<u64>>,
}

/// Parameters for attention head projections.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AttentionHeadsParams {
    pub dtype: TensorDtype,
    pub num_heads: u32,
    pub seq_bucket: u32,
    pub head_dim: u32,
    pub physical_layout: PhysicalLayout,
    pub alignment: u32,
}

/// Parameters for vision image tensors.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VisionImageParams {
    pub dtype: TensorDtype,
    pub channel_count: u32,
    pub height: u32,
    pub width: u32,
    pub physical_layout: PhysicalLayout,
    pub alignment: u32,
}

/// Parameters for an opaque metal-only buffer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetalOnlyParams {
    pub name: String,
    pub dtype: TensorDtype,
    pub byte_count: u64,
}

// ── Physical layout ──────────────────────────────────────────────────────

/// Describes how a tensor's logical dimensions map to physical memory.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PhysicalLayout {
    /// Row-major contiguous (C-order).
    ContiguousRowMajor,
    /// NCHW channel-first.
    NCHW,
    /// NHWC channel-last.
    NHWC,
    /// Custom stride-defined layout.
    Custom(Vec<u64>),
}

// ── Slot lease / tensor identity ─────────────────────────────────────────

/// Opaque lease identifier for a slot reservation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SlotLeaseId(pub u64);

/// Logical tensor identifier (model-scoped name).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LogicalTensorId(pub String);

// ── Slot backing ─────────────────────────────────────────────────────────

/// How a slot's backing memory is provisioned.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SlotBacking {
    /// IOSurface shared memory (ANE ↔ GPU).
    IOSurface,
    /// Metal buffer (GPU-private).
    MetalBuffer,
    /// POSIX shared memory (CPU ↔ GPU / CPU ↔ ANE).
    SharedMemory,
}

// ── Slot descriptor ──────────────────────────────────────────────────────

/// Full descriptor for a single activation slot tracked in the shared arena.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActivationSlotDescriptor {
    pub slot_index: u32,
    pub logical_tensor_id: LogicalTensorId,
    pub producer_lane: ExecutionLane,
    pub consumer_candidates: Vec<ExecutionLane>,
    pub abi: ActivationAbi,
    pub backing: SlotBacking,
    pub state: SlotState,
    pub lease_id: SlotLeaseId,
    pub materialization_mode: MaterializationMode,
}

// ── Activation contract ──────────────────────────────────────────────────

/// Concrete ABI contract for one activation tensor — the validated
/// description used at dispatch time to match producer and consumer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActivationContract {
    pub abi: ActivationAbi,
    pub element_count: u64,
    pub byte_count: u64,
    pub shape: Vec<u64>,
    pub stride: Vec<u64>,
    pub physical_layout: PhysicalLayout,
    pub alignment: u32,
}

impl ActivationContract {
    /// Validate that `candidate` matches this contract exactly on all fields.
    ///
    /// Returns `Ok(())` when every field matches, or `Err(reason)` on the
    /// first mismatch.
    pub fn validate_contract(&self, candidate: &ActivationContract) -> Result<(), String> {
        if self.abi != candidate.abi {
            return Err(format!(
                "ABI mismatch: {:?} vs {:?}",
                self.abi, candidate.abi
            ));
        }
        if self.element_count != candidate.element_count {
            return Err(format!(
                "element_count mismatch: {} vs {}",
                self.element_count, candidate.element_count
            ));
        }
        if self.byte_count != candidate.byte_count {
            return Err(format!(
                "byte_count mismatch: {} vs {}",
                self.byte_count, candidate.byte_count
            ));
        }
        if self.shape != candidate.shape {
            return Err(format!(
                "shape mismatch: {:?} vs {:?}",
                self.shape, candidate.shape
            ));
        }
        if self.stride != candidate.stride {
            return Err(format!(
                "stride mismatch: {:?} vs {:?}",
                self.stride, candidate.stride
            ));
        }
        if self.physical_layout != candidate.physical_layout {
            return Err(format!(
                "physical_layout mismatch: {:?} vs {:?}",
                self.physical_layout, candidate.physical_layout
            ));
        }
        if self.alignment != candidate.alignment {
            return Err(format!(
                "alignment mismatch: {} vs {}",
                self.alignment, candidate.alignment
            ));
        }

        Ok(())
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────

    fn sample_decode_v1_abi() -> ActivationAbi {
        ActivationAbi::DecodeActivationV1(DecodeActivationV1Params {
            dtype: TensorDtype::Float16,
            seq_bucket: 64,
            hidden_dim: 4096,
            physical_layout: PhysicalLayout::ContiguousRowMajor,
            alignment: 64,
            stride_constraint: None,
        })
    }

    fn sample_decode_v1_contract() -> ActivationContract {
        ActivationContract {
            abi: sample_decode_v1_abi(),
            element_count: 262144,       // 64 * 4096
            byte_count: 524288,          // 262144 * 2 (FP16)
            shape: vec![64, 4096],
            stride: vec![4096, 1],
            physical_layout: PhysicalLayout::ContiguousRowMajor,
            alignment: 64,
        }
    }

    fn sample_attention_abi() -> ActivationAbi {
        ActivationAbi::AttentionHeads(AttentionHeadsParams {
            dtype: TensorDtype::Float16,
            num_heads: 32,
            seq_bucket: 64,
            head_dim: 128,
            physical_layout: PhysicalLayout::ContiguousRowMajor,
            alignment: 64,
        })
    }

    fn sample_vision_abi() -> ActivationAbi {
        ActivationAbi::VisionImage(VisionImageParams {
            dtype: TensorDtype::Float32,
            channel_count: 3,
            height: 224,
            width: 224,
            physical_layout: PhysicalLayout::NHWC,
            alignment: 64,
        })
    }

    fn sample_metal_only_abi() -> ActivationAbi {
        ActivationAbi::MetalOnly(MetalOnlyParams {
            name: "scratch_buffer".into(),
            dtype: TensorDtype::Float16,
            byte_count: 1_048_576,
        })
    }

    // ── Tests ────────────────────────────────────────────────────────────

    #[test]
    fn test_decode_activation_v1_constructs() {
        let params = DecodeActivationV1Params {
            dtype: TensorDtype::Float16,
            seq_bucket: 128,
            hidden_dim: 8192,
            physical_layout: PhysicalLayout::ContiguousRowMajor,
            alignment: 256,
            stride_constraint: Some(vec![8192, 1]),
        };
        let abi = ActivationAbi::DecodeActivationV1(params);
        let contract = ActivationContract {
            abi,
            element_count: 1_048_576,       // 128 * 8192
            byte_count: 2_097_152,           // 1_048_576 * 2
            shape: vec![128, 8192],
            stride: vec![8192, 1],
            physical_layout: PhysicalLayout::ContiguousRowMajor,
            alignment: 256,
        };

        // Verify contract round-trips
        assert_eq!(
            contract.abi,
            ActivationAbi::DecodeActivationV1(DecodeActivationV1Params {
                dtype: TensorDtype::Float16,
                seq_bucket: 128,
                hidden_dim: 8192,
                physical_layout: PhysicalLayout::ContiguousRowMajor,
                alignment: 256,
                stride_constraint: Some(vec![8192, 1]),
            })
        );
        assert_eq!(contract.element_count, 1_048_576);
        assert_eq!(contract.byte_count, 2_097_152);
    }

    #[test]
    fn test_contract_rejects_shape_mismatch() {
        let expected = sample_decode_v1_contract();
        let mut candidate = expected.clone();
        candidate.shape = vec![128, 4096];

        let result = expected.validate_contract(&candidate);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("shape"));
    }

    #[test]
    fn test_contract_rejects_dtype_mismatch() {
        let expected = sample_decode_v1_contract();
        // Same ABI variant but different dtype inside params
        let candidate = ActivationContract {
            abi: ActivationAbi::DecodeActivationV1(DecodeActivationV1Params {
                dtype: TensorDtype::Float32,
                seq_bucket: 64,
                hidden_dim: 4096,
                physical_layout: PhysicalLayout::ContiguousRowMajor,
                alignment: 64,
                stride_constraint: None,
            }),
            ..expected.clone()
        };

        let result = expected.validate_contract(&candidate);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("ABI"));
    }

    #[test]
    fn test_contract_accepts_matching() {
        let expected = sample_decode_v1_contract();
        let candidate = expected.clone();
        assert!(expected.validate_contract(&candidate).is_ok());
    }

    #[test]
    fn test_serde_roundtrip_all_variants() {
        let variants: Vec<ActivationAbi> = vec![
            sample_decode_v1_abi(),
            sample_attention_abi(),
            sample_vision_abi(),
            sample_metal_only_abi(),
        ];

        for abi in &variants {
            let json = serde_json::to_string(abi).unwrap();
            let deserialized: ActivationAbi = serde_json::from_str(&json).unwrap();
            assert_eq!(*abi, deserialized, "serde roundtrip failed for {:?}", abi);
        }
    }

    #[test]
    fn test_slot_descriptor_constructs() {
        let descriptor = ActivationSlotDescriptor {
            slot_index: 3,
            logical_tensor_id: LogicalTensorId("proj_k.weight".into()),
            producer_lane: ExecutionLane::MlxGpu,
            consumer_candidates: vec![ExecutionLane::CoreMlAne, ExecutionLane::MlxGpu],
            abi: sample_decode_v1_abi(),
            backing: SlotBacking::IOSurface,
            state: SlotState::Free,
            lease_id: SlotLeaseId(42),
            materialization_mode: MaterializationMode::ReusedProviderBuffer,
        };

        assert_eq!(descriptor.slot_index, 3);
        assert_eq!(descriptor.logical_tensor_id.0, "proj_k.weight");
        assert_eq!(descriptor.producer_lane, ExecutionLane::MlxGpu);
        assert_eq!(descriptor.consumer_candidates.len(), 2);
        assert!(matches!(descriptor.backing, SlotBacking::IOSurface));
        assert!(matches!(descriptor.state, SlotState::Free));
        assert_eq!(descriptor.lease_id.0, 42);
        assert!(matches!(
            descriptor.materialization_mode,
            MaterializationMode::ReusedProviderBuffer
        ));
    }
}
