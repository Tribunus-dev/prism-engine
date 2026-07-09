//! Receipt types for cimage runtime execution.

use serde::{Deserialize, Serialize};

use crate::cimage::ReceiptEvidenceKind;
use crate::execution_plan::backend_capability::BackendLoweringTarget;
use crate::execution_plan::HardwareProfileId;

use super::tensor_store::MlpRegionExecutionMode;

/// Receipt emitted after running an MLP shard through Metal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CImageRegionExecutionReceipt {
    pub receipt_version: u32,
    pub cimage_digest: String,
    pub region_id: String,
    pub backend: BackendLoweringTarget,
    pub hardware_profile: HardwareProfileId,
    pub execution_mode: MlpRegionExecutionMode,
    pub evidence_kind: ReceiptEvidenceKind,
    pub tensor_count: usize,
    pub kernel_count: usize,
    pub buffer_count: usize,
    pub total_bound_bytes: u64,
    pub scratch_bytes: u64,
    pub cpu_reconstructed_output_digest: String,
    pub metal_output_digest: String,
    pub metal_vs_cpu_nrmse: f64,
    pub metal_vs_cpu_cosine: f64,
    pub metal_vs_cpu_max_abs_error: f64,
    pub rawf32_vs_cpu_reconstructed_nrmse: f64,
    pub rawf32_vs_metal_nrmse: f64,
    pub command_buffer_ms: f64,
    pub encode_ms: f64,
    pub readback_ms: f64,
    pub hazard_safe: bool,
    pub validation_passed: bool,
    pub warnings: Vec<String>,
}

/// Per-op binding receipt for debugging Metal execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CImageBindingReceipt {
    pub region_id: String,
    pub op_id: String,
    pub kernel_name: String,
    pub bindings: Vec<CImageKernelBindingInfo>,
    pub all_bindings_resolved: bool,
}

/// One binding entry in a binding receipt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CImageKernelBindingInfo {
    pub slot: u32,
    pub buffer_id: String,
    pub role: String,
    pub byte_offset: u64,
    pub byte_len: u64,
    pub resolved: bool,
}

/// Per-layer validation comparison between Metal and CPU reference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CImageLayerValidationReceipt {
    pub layer: usize,
    pub hidden_nrmse: f64,
    pub hidden_cosine: f64,
    pub max_abs_error: f64,
    pub passed: bool,
}

/// Per-layer timing statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CImageLayerTiming {
    pub layer: usize,
    pub weight_upload_ms: f64,
    pub command_buffer_ms: f64,
}

/// Full model execution receipt — aggregate of all layer runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CImageModelExecutionReceipt {
    pub cimage_digest: String,
    pub num_layers: usize,
    pub hidden_dim: usize,
    pub seq_len: usize,
    pub layer_validations: Vec<CImageLayerValidationReceipt>,
    pub layer_timings: Vec<CImageLayerTiming>,
    pub total_command_buffer_ms: f64,
    pub validation_passed: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_region_execution_receipt_serde() {
        let r = CImageRegionExecutionReceipt {
            receipt_version: 0,
            cimage_digest: "abc".into(),
            region_id: "mlp_shard_region".into(),
            backend: BackendLoweringTarget::MetalFusedGpu,
            hardware_profile: HardwareProfileId::AppleMProBalanced,
            execution_mode: MlpRegionExecutionMode::StagedKernels,
            evidence_kind: ReceiptEvidenceKind::SyntheticNumericalProof,
            tensor_count: 4,
            kernel_count: 7,
            buffer_count: 12,
            total_bound_bytes: 65536,
            scratch_bytes: 16384,
            cpu_reconstructed_output_digest: "def".into(),
            metal_output_digest: "ghi".into(),
            metal_vs_cpu_nrmse: 0.0003,
            metal_vs_cpu_cosine: 0.99999,
            metal_vs_cpu_max_abs_error: 0.001,
            rawf32_vs_cpu_reconstructed_nrmse: 0.05,
            rawf32_vs_metal_nrmse: 0.05,
            command_buffer_ms: 0.18,
            encode_ms: 0.05,
            readback_ms: 0.01,
            hazard_safe: true,
            validation_passed: true,
            warnings: vec![],
        };
        let json = serde_json::to_string_pretty(&r).unwrap();
        let back: CImageRegionExecutionReceipt = serde_json::from_str(&json).unwrap();
        assert_eq!(back.kernel_count, 7);
        assert!(back.validation_passed);
    }

    #[test]
    fn test_binding_receipt_serde() {
        let r = CImageBindingReceipt {
            region_id: "r0".into(),
            op_id: "op_0_rmsnorm".into(),
            kernel_name: "cimage_rmsnorm_f32".into(),
            bindings: vec![
                CImageKernelBindingInfo {
                    slot: 0,
                    buffer_id: "hidden_in".into(),
                    role: "InputActivation".into(),
                    byte_offset: 0,
                    byte_len: 256,
                    resolved: true,
                },
                CImageKernelBindingInfo {
                    slot: 1,
                    buffer_id: "rmsnorm_weight".into(),
                    role: "WeightCodes".into(),
                    byte_offset: 0,
                    byte_len: 256,
                    resolved: true,
                },
            ],
            all_bindings_resolved: true,
        };
        let json = serde_json::to_string_pretty(&r).unwrap();
        let back: CImageBindingReceipt = serde_json::from_str(&json).unwrap();
        assert_eq!(back.bindings.len(), 2);
        assert!(back.all_bindings_resolved);
    }

    #[test]
    fn test_layer_validation_receipt_serde() {
        let r = CImageLayerValidationReceipt {
            layer: 0,
            hidden_nrmse: 0.0002,
            hidden_cosine: 0.99998,
            max_abs_error: 0.001,
            passed: true,
        };
        let json = serde_json::to_string_pretty(&r).unwrap();
        let back: CImageLayerValidationReceipt = serde_json::from_str(&json).unwrap();
        assert_eq!(back.layer, 0);
        assert!(back.passed);
    }

    #[test]
    fn test_layer_timing_serde() {
        let r = CImageLayerTiming {
            layer: 5,
            weight_upload_ms: 1.2,
            command_buffer_ms: 3.4,
        };
        let json = serde_json::to_string_pretty(&r).unwrap();
        let back: CImageLayerTiming = serde_json::from_str(&json).unwrap();
        assert_eq!(back.layer, 5);
        assert!((back.command_buffer_ms - 3.4).abs() < 1e-10);
    }

    #[test]
    fn test_model_execution_receipt_serde() {
        let r = CImageModelExecutionReceipt {
            cimage_digest: "abc123".into(),
            num_layers: 30,
            hidden_dim: 2560,
            seq_len: 128,
            layer_validations: vec![
                CImageLayerValidationReceipt {
                    layer: 0,
                    hidden_nrmse: 0.0002,
                    hidden_cosine: 0.99998,
                    max_abs_error: 0.001,
                    passed: true,
                },
                CImageLayerValidationReceipt {
                    layer: 29,
                    hidden_nrmse: 0.005,
                    hidden_cosine: 0.9995,
                    max_abs_error: 0.01,
                    passed: true,
                },
            ],
            layer_timings: vec![
                CImageLayerTiming {
                    layer: 0,
                    weight_upload_ms: 1.2,
                    command_buffer_ms: 3.4,
                },
                CImageLayerTiming {
                    layer: 29,
                    weight_upload_ms: 1.1,
                    command_buffer_ms: 3.2,
                },
            ],
            total_command_buffer_ms: 99.0,
            validation_passed: true,
        };
        let json = serde_json::to_string_pretty(&r).unwrap();
        let back: CImageModelExecutionReceipt = serde_json::from_str(&json).unwrap();
        assert_eq!(back.num_layers, 30);
        assert_eq!(back.layer_validations.len(), 2);
        assert!(back.validation_passed);
    }
}
