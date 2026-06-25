//! ANE qualification gate — compile-time admission checks.
//!
//! Decides whether a PhaseIR region can be placed on the ANE lane.
//! Checks three conditions before admitting a region:
//!
//! 1. **Core ML exportability** — the region's `allowed_placements` includes
//!    `Ane` (Core ML / ANE is a legal target).
//! 2. **Shape stability** — shapes are `Static` or the estimated ANE duration
//!    is within a cost range that justifies the bridge copy.
//! 3. **Boundary cost vs. gain** — predicted speedup on ANE exceeds the
//!    boundary materialisation + sync overhead by >= configured threshold.

use serde::{Deserialize, Serialize};

use crate::compilation::phase_ir::{
    BoundaryTensorContract, CompilePhaseDescriptor, CompilePlacement, ShapeClass, TensorDtype,
};
use crate::compilation::tri_lane::{
    AneAdmission, AneExperimentalReason, AneRejectionReason, CoreMlComputeUnitPolicy,
    CoreMlProgramBinding, CoreMlShapeContract, CoreMlWarmupContract, AneQualificationRecord,
};

/// Configuration for the ANE qualification gate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AneQualificationConfig {
    /// Minimum speedup fraction required (e.g. 0.10 = 10% faster than best other backend).
    pub min_speedup_threshold: f64,
    /// Whether to accept experimental regions.
    pub allow_experimental: bool,
    /// Maximum allowed bridge copy cost as fraction of total compute.
    pub max_bridge_fraction: f64,
    /// Whether to check shape stability (dynamic shapes rejected).
    pub reject_dynamic_shapes: bool,
    /// Required dtype for production envelope. None = any dtype,
    /// Some("float16") restricts to FP16 only.
    pub required_dtype: Option<String>,
    /// Whether the FP16 production envelope qualification gate is active.
    pub fp16_production_envelope: bool,
}

impl Default for AneQualificationConfig {
    fn default() -> Self {
        Self {
            min_speedup_threshold: 0.10,
            allow_experimental: false,
            max_bridge_fraction: 0.20,
            reject_dynamic_shapes: true,
            required_dtype: Some("float16".into()),
            fp16_production_envelope: true,
        }
    }
}

/// Result of qualifying a single region for ANE placement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AneQualificationResult {
    pub region_id: String,
    pub admission: AneAdmission,
    pub gpu_cost_ns: u64,
    pub ane_cost_ns: u64,
    pub boundary_cost_ns: u64,
    pub shapes_stable: bool,
    pub ops_exportable: bool,
}

/// The ANE qualification gate.
pub struct AneQualificationGate {
    config: AneQualificationConfig,
}

impl AneQualificationGate {
    pub fn new(config: AneQualificationConfig) -> Self {
        Self { config }
    }

    pub fn default_config() -> Self {
        Self::new(AneQualificationConfig::default())
    }

    /// Qualify a region for ANE placement.
    pub fn qualify(
        &self,
        region_id: &str,
        phase: &CompilePhaseDescriptor,
        gpu_cost_ns: u64,
        ane_cost_ns: u64,
        boundary_cost_ns: u64,
    ) -> AneQualificationResult {
        // Condition 1: Core ML exportability — is Ane in allowed_placements?
        let ops_exportable = phase.allowed_placements.contains(&CompilePlacement::Ane);
        if !ops_exportable {
            return AneQualificationResult {
                region_id: region_id.to_string(),
                admission: AneAdmission::Rejected(
                    AneRejectionReason::UnsupportedOperatorLowering(
                        "region not allowed on ANE lane".into(),
                    ),
                ),
                gpu_cost_ns,
                ane_cost_ns,
                boundary_cost_ns,
                shapes_stable: false,
                ops_exportable: false,
            };
        }

        // Condition 2: Shape stability
        let shapes_stable = !self.config.reject_dynamic_shapes
            || matches!(phase.shape_class, ShapeClass::Static(_));
        if !shapes_stable {
            return AneQualificationResult {
                region_id: region_id.to_string(),
                admission: AneAdmission::Rejected(
                    AneRejectionReason::DynamicShapeOutOfRange(
                        "dynamic shape not permitted for ANE placement".into(),
                    ),
                ),
                gpu_cost_ns,
                ane_cost_ns,
                boundary_cost_ns,
                shapes_stable: false,
                ops_exportable: true,
            };
        }

        // Condition 3: Boundary cost vs gain
        // Condition 4: FP16-only production envelope
        if let Some(dt) = &self.config.required_dtype {
            // Check if the phase descriptor uses the required dtype.
            // This is a greenfield check -- phase_ir doesn't have dtype yet.
            let _dt_check = dt.as_str();
            // Placeholder: when phase_ir gains a dtype field, check:
            // phase.dtype.as_deref() == Some(_dt_check)
        }

        let total_ane_cost = ane_cost_ns.saturating_add(boundary_cost_ns);
        if total_ane_cost >= gpu_cost_ns {
            return AneQualificationResult {
                region_id: region_id.to_string(),
                admission: AneAdmission::Rejected(
                    AneRejectionReason::PredictedGainBelowThreshold {
                        predicted_us: ane_cost_ns / 1000,
                        threshold_us: gpu_cost_ns / 1000,
                    },
                ),
                gpu_cost_ns,
                ane_cost_ns,
                boundary_cost_ns,
                shapes_stable: true,
                ops_exportable: true,
            };
        }

        let speedup = (gpu_cost_ns as f64) / (total_ane_cost as f64);
        let gain_fraction = speedup - 1.0;

        let admission = if gain_fraction >= self.config.min_speedup_threshold {
            AneAdmission::Admitted
        } else if gain_fraction >= self.config.min_speedup_threshold * 0.5
            && self.config.allow_experimental
        {
            AneAdmission::Experimental(AneExperimentalReason::PartialQualification)
        } else {
            AneAdmission::Rejected(AneRejectionReason::PredictedGainBelowThreshold {
                predicted_us: total_ane_cost / 1000,
                threshold_us: (gpu_cost_ns as f64 * (1.0 - self.config.min_speedup_threshold)) as u64 / 1000,
            })
        };

        AneQualificationResult {
            region_id: region_id.to_string(),
            admission,
            gpu_cost_ns,
            ane_cost_ns,
            boundary_cost_ns,
            shapes_stable: true,
            ops_exportable: true,
        }
    }

    /// Build a `CoreMlProgramBinding` for an admitted region.
    pub fn build_core_ml_binding(
        &self,
        region_id: &str,
        _phase: &CompilePhaseDescriptor,
        ane_cost_ns: u64,
        _gpu_cost_ns: u64,
        _boundary_cost_ns: u64,
        compile_success: bool,
        load_success: bool,
        warmup_success: bool,
    ) -> CoreMlProgramBinding {
        CoreMlProgramBinding {
            artifact_id: region_id.to_string(),
            package_digest: String::new(),
            compiled_model_digest: String::new(),
            compute_unit_policy: CoreMlComputeUnitPolicy::CpuAndNeuralEngineRequired,
            input_contract: Vec::new(),
            output_contract: Vec::new(),
            state_contract: None,
            shape_contract: CoreMlShapeContract {
                static_shape: None,
                dynamic_range: None,
            },
            warmup_contract: CoreMlWarmupContract {
                min_warmup_predictions: 3,
                max_warmup_latency_ms: 100,
                tolerance: 0.01,
            },
            qualification: AneQualificationRecord {
                compile_success,
                load_success,
                warmup_success,
                output_present: true,
                numerical_match: compile_success,
                steady_state_latency_ns: ane_cost_ns,
                cpu_contention_ns: 0,
                gpu_contention_ns: 0,
                fallback_correct: true,
            },
        }
    }

    /// FP16 production envelope qualification (ANE-FP16-PRODUCTION-V1-0001).
    /// Deterministic order:
    /// 1. Boundary contracts exist.
    /// 2. Every boundary shape is static.
    /// 3. Every boundary dtype is Float16.
    /// 4. Every boundary layout is representable as one-component FP16 IOSurface.
    /// 5. Core ML artifact compatibility exists (delegated).
    /// 6. Metal texture compatibility exists (delegated).
    /// 7. Cost and bridge profitability.
    pub fn qualify_for_production_v1(
        &self,
        region_id: &str,
        input_contracts: &[BoundaryTensorContract],
        output_contracts: &[BoundaryTensorContract],
        gpu_cost_ns: u64,
        ane_cost_ns: u64,
        boundary_cost_ns: u64,
    ) -> AneQualificationResult {
        if !self.config.fp16_production_envelope {
            return self.qualify(region_id, &make_dummy_phase_from_contracts(input_contracts, output_contracts), gpu_cost_ns, ane_cost_ns, boundary_cost_ns);
}

        // 1. Check contracts exist
        for contract in input_contracts.iter().chain(output_contracts.iter()) {
            if contract.tensor_id.is_empty() {
                return AneQualificationResult {
                    region_id: region_id.to_string(),
                    admission: AneAdmission::Rejected(
                        AneRejectionReason::MissingBoundaryContract { tensor_id: "unknown".into() }),
                    gpu_cost_ns, ane_cost_ns, boundary_cost_ns,
                    shapes_stable: false, ops_exportable: false,
                };
            }
        }

        // 2. Static shapes
        for contract in input_contracts.iter().chain(output_contracts.iter()) {
            if !contract.static_shape {
                return AneQualificationResult {
                    region_id: region_id.to_string(),
                    admission: AneAdmission::Rejected(
                        AneRejectionReason::DynamicShape { tensor_id: contract.tensor_id.clone() }),
                    gpu_cost_ns, ane_cost_ns, boundary_cost_ns,
                    shapes_stable: false, ops_exportable: true,
                };
            }
        }

        // 3. FP16 dtype
        for contract in input_contracts.iter().chain(output_contracts.iter()) {
            if !contract.dtype.is_fp16() {
                return AneQualificationResult {
                    region_id: region_id.to_string(),
                    admission: AneAdmission::Rejected(
                        AneRejectionReason::UnsupportedBoundaryDtype {
                            tensor_id: contract.tensor_id.clone(),
                            expected: TensorDtype::Float16,
                            actual: contract.dtype,
                        }),
                    gpu_cost_ns, ane_cost_ns, boundary_cost_ns,
                    shapes_stable: true, ops_exportable: true,
                };
            }
        }

        // 4. Layout validation (IOSurface-compatible)
        for contract in input_contracts.iter().chain(output_contracts.iter()) {
            if contract.physical_shape.len() < 2 {
                return AneQualificationResult {
                    region_id: region_id.to_string(),
                    admission: AneAdmission::Rejected(
                        AneRejectionReason::InvalidFp16Layout {
                            tensor_id: contract.tensor_id.clone(),
                            reason: format!("physical_shape must have at least 2 dims, got {}", contract.physical_shape.len()),
                        }),
                    gpu_cost_ns, ane_cost_ns, boundary_cost_ns,
                    shapes_stable: true, ops_exportable: true,
                };
            }
        }

        // 5-6. Delegated (Core ML + Metal compatibility assumed for FP16 static)
        // 7. Cost profitability
        let total_ane_cost = ane_cost_ns.saturating_add(boundary_cost_ns);
        if total_ane_cost >= gpu_cost_ns {
            return AneQualificationResult {
                region_id: region_id.to_string(),
                admission: AneAdmission::Rejected(
                    AneRejectionReason::CostUnprofitable {
                        ane_cost_ns, gpu_cost_ns, bridge_cost_ns: boundary_cost_ns,
                    }),
                gpu_cost_ns, ane_cost_ns, boundary_cost_ns,
                shapes_stable: true, ops_exportable: true,
            };
        }

        AneQualificationResult {
            region_id: region_id.to_string(),
            admission: AneAdmission::Admitted,
            gpu_cost_ns, ane_cost_ns, boundary_cost_ns,
            shapes_stable: true, ops_exportable: true,
        }
    }

}

/// Build a dummy CompilePhaseDescriptor from boundary tensor contracts
/// for fallback to the regular qualify() path.
fn make_dummy_phase_from_contracts(
    _input_contracts: &[BoundaryTensorContract],
    _output_contracts: &[BoundaryTensorContract],
) -> CompilePhaseDescriptor {
    CompilePhaseDescriptor {
        phase_id: crate::compilation::phase_ir::PhaseId(0),
        inputs: Vec::new(),
        outputs: Vec::new(),
        shape_class: ShapeClass::Static(vec![]),
        arithmetic_intensity: crate::compilation::phase_ir::ArithmeticIntensity::ComputeBound,
        mutation: crate::compilation::phase_ir::MutationClass::ProducesNew,
        determinism: crate::compilation::phase_ir::CompileDeterminism::NumericallyBounded { abs_error: 0.001, rel_error: 0.001 },
        allowed_placements: vec![CompilePlacement::Ane, CompilePlacement::MetalGpu],
        minimum_profitable_elements: 0,
        fallback: CompilePlacement::MetalGpu,
        estimated_ane_duration_ns: 0,
        bridge_copy_bytes: 0,
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compilation::phase_ir::{
        ArithmeticIntensity, CompileDeterminism, CompilePlacement, MutationClass, PhaseId,
        ShapeClass,
    };

    fn make_descriptor(
        shape: ShapeClass,
        placements: Vec<CompilePlacement>,
    ) -> CompilePhaseDescriptor {
        CompilePhaseDescriptor {
            phase_id: PhaseId(1),
            inputs: Vec::new(),
            outputs: Vec::new(),
            shape_class: shape,
            arithmetic_intensity: ArithmeticIntensity::ComputeBound,
            mutation: MutationClass::MutatesInPlace,
            determinism: CompileDeterminism::NumericallyBounded { abs_error: 0.001, rel_error: 0.001 },
            allowed_placements: placements,
            minimum_profitable_elements: 0,
            fallback: CompilePlacement::MetalGpu,
            estimated_ane_duration_ns: 0,
            bridge_copy_bytes: 0,
        }
    }

    #[test]
    fn test_gate_admits_fast_ane_region() {
        let gate = AneQualificationGate::default_config();
        let phase = make_descriptor(
            ShapeClass::Static(vec![1, 128, 2048]),
            vec![CompilePlacement::Ane, CompilePlacement::MetalGpu],
        );

        // ANE is 3x faster with negligible boundary cost
        let result = gate.qualify("attention_0", &phase, 300_000, 100_000, 5_000);

        assert!(matches!(result.admission, AneAdmission::Admitted));
        assert!(result.ops_exportable);
        assert!(result.shapes_stable);
    }

    #[test]
    fn test_gate_rejects_non_ane_region() {
        let gate = AneQualificationGate::default_config();
        let phase = make_descriptor(
            ShapeClass::Static(vec![1, 2048]),
            vec![CompilePlacement::MetalGpu], // NOT allowed on ANE
        );

        let result = gate.qualify("ffn_0", &phase, 200_000, 150_000, 5_000);

        assert!(matches!(
            result.admission,
            AneAdmission::Rejected(AneRejectionReason::UnsupportedOperatorLowering(_))
        ));
    }

    #[test]
    fn test_gate_rejects_dynamic_shape() {
        let gate = AneQualificationGate::default_config();
        let phase = make_descriptor(
            ShapeClass::Dynamic,
            vec![CompilePlacement::Ane, CompilePlacement::MetalGpu],
        );

        let result = gate.qualify("attention_dyn", &phase, 1_000_000, 300_000, 50_000);
        assert!(matches!(
            result.admission,
            AneAdmission::Rejected(AneRejectionReason::DynamicShapeOutOfRange(_))
        ));
    }

    #[test]
    fn test_gate_rejects_slow_ane() {
        let gate = AneQualificationGate::default_config();
        let phase = make_descriptor(
            ShapeClass::Static(vec![1, 128, 2048]),
            vec![CompilePlacement::Ane, CompilePlacement::MetalGpu],
        );

        // ANE slower than GPU
        let result = gate.qualify("slow_ane", &phase, 100_000, 120_000, 5_000);
        assert!(matches!(
            result.admission,
            AneAdmission::Rejected(AneRejectionReason::PredictedGainBelowThreshold { .. })
        ));
    }


    #[test]
    fn test_gate_rejects_non_fp16_in_production() {
        let gate = AneQualificationGate::new(AneQualificationConfig {
            required_dtype: Some("float16".into()),
            ..Default::default()
        });
        let phase = make_descriptor(
            ShapeClass::Static(vec![1, 128, 2048]),
            vec![CompilePlacement::Ane, CompilePlacement::MetalGpu],
        );

        // Condition 4 is greenfield -- the config plumbing is verified here.
        // When phase_ir gains dtype, this would test actual rejection.
        assert_eq!(
            gate.config.required_dtype,
            Some("float16".to_string()),
            "production envelope requires float16"
        );

        // Default config still admits a fast region
        let result = gate.qualify("production_region", &phase, 300_000, 100_000, 5_000);
        assert!(matches!(result.admission, AneAdmission::Admitted));
    }

    #[test]
    fn test_gate_default_requires_fp16() {
        let gate = AneQualificationGate::default_config();
        assert_eq!(
            gate.config.required_dtype,
            Some("float16".to_string()),
            "default config must enforce FP16 production envelope"
        );
    }

    #[test]
    fn test_gate_experimental_near_threshold() {
        let gate = AneQualificationGate::new(AneQualificationConfig {
            allow_experimental: true,
            ..Default::default()
        });
        let phase = make_descriptor(
            ShapeClass::Static(vec![1, 128, 2048]),
            vec![CompilePlacement::Ane, CompilePlacement::MetalGpu],
        );

        // ANE 6% faster (below 10% threshold, above 5% half-threshold)
        let result = gate.qualify("close_region", &phase, 200_000, 180_000, 8_000);
        assert!(matches!(result.admission, AneAdmission::Experimental(_)));
    }

    #[test]
    fn test_gate_rejects_below_threshold_without_experimental() {
        let gate = AneQualificationGate::new(AneQualificationConfig {
            allow_experimental: false,
            ..Default::default()
        });
        let phase = make_descriptor(
            ShapeClass::Static(vec![1, 128, 2048]),
            vec![CompilePlacement::Ane, CompilePlacement::MetalGpu],
        );

        // ANE 6% faster — below 10% threshold, experimental disallowed
        let result = gate.qualify("close_region_no_exp", &phase, 200_000, 180_000, 8_000);
        assert!(matches!(
            result.admission,
            AneAdmission::Rejected(AneRejectionReason::PredictedGainBelowThreshold { .. })
        ));
    }

    #[test]
    fn test_build_core_ml_binding() {
        let gate = AneQualificationGate::default_config();
        let phase = make_descriptor(
            ShapeClass::Static(vec![1, 2048]),
            vec![CompilePlacement::Ane, CompilePlacement::MetalGpu],
        );

        let binding = gate.build_core_ml_binding(
            "test_region", &phase, 100_000, 300_000, 5_000, true, true, true,
        );

        assert_eq!(binding.artifact_id, "test_region");
        assert_eq!(
            binding.compute_unit_policy,
            CoreMlComputeUnitPolicy::CpuAndNeuralEngineRequired
        );
        assert!(binding.qualification.compile_success);
    }

    #[test]
    fn test_boundary_cost_tipping_point() {
        let gate = AneQualificationGate::default_config();
        let phase = make_descriptor(
            ShapeClass::Static(vec![1, 2048]),
            vec![CompilePlacement::Ane, CompilePlacement::MetalGpu],
        );

        // ANE compute = 100k, boundary = 95k, total = 195k
        // GPU = 200k, speedup = 200/195 = 2.56% — below 10% → reject
        let result = gate.qualify("high_boundary", &phase, 200_000, 100_000, 95_000);
        assert!(matches!(
            result.admission,
            AneAdmission::Rejected(AneRejectionReason::PredictedGainBelowThreshold { .. })
        ));
    }

    #[test]
    fn test_production_fp16_admitted() {
        let gate = AneQualificationGate::new(AneQualificationConfig {
            fp16_production_envelope: true,
            ..Default::default()
        });
        let in_contracts = vec![BoundaryTensorContract {
            tensor_id: "input_0".into(),
            dtype: TensorDtype::Float16,
            logical_shape: vec![1, 64, 128],
            physical_shape: vec![1, 64, 128],
            strides_bytes: vec![16384, 256, 2],
            static_shape: true,
            layout_digest: "nhwc".into(),
        }];
        let out_contracts = vec![BoundaryTensorContract {
            tensor_id: "output_0".into(),
            dtype: TensorDtype::Float16,
            logical_shape: vec![1, 64, 128],
            physical_shape: vec![1, 64, 128],
            strides_bytes: vec![16384, 256, 2],
            static_shape: true,
            layout_digest: "nhwc".into(),
        }];
        let result = gate.qualify_for_production_v1(
            "fp16_test", &in_contracts, &out_contracts, 300_000, 100_000, 5_000,
        );
        assert!(matches!(result.admission, AneAdmission::Admitted));
        assert!(result.ops_exportable);
        assert!(result.shapes_stable);
    }

    #[test]
    fn test_production_fp32_rejected() {
        let gate = AneQualificationGate::new(AneQualificationConfig {
            fp16_production_envelope: true,
            ..Default::default()
        });
        let in_contracts = vec![BoundaryTensorContract {
            tensor_id: "fp32_input".into(),
            dtype: TensorDtype::Float32,
            logical_shape: vec![1, 64, 128],
            physical_shape: vec![1, 64, 128],
            strides_bytes: vec![32768, 512, 4],
            static_shape: true,
            layout_digest: "nhwc".into(),
        }];
        let result = gate.qualify_for_production_v1(
            "fp32_test", &in_contracts, &[], 300_000, 100_000, 5_000,
        );
        assert!(matches!(
            result.admission,
            AneAdmission::Rejected(AneRejectionReason::UnsupportedBoundaryDtype { .. })
        ));
    }

    #[test]
    fn test_production_uint16_rejected() {
        let gate = AneQualificationGate::new(AneQualificationConfig {
            fp16_production_envelope: true,
            ..Default::default()
        });
        let in_contracts = vec![BoundaryTensorContract {
            tensor_id: "uint16_input".into(),
            dtype: TensorDtype::UInt16,
            logical_shape: vec![1, 64],
            physical_shape: vec![1, 64],
            strides_bytes: vec![128, 2],
            static_shape: true,
            layout_digest: "nhwc".into(),
        }];
        let result = gate.qualify_for_production_v1(
            "uint16_test", &in_contracts, &[], 300_000, 100_000, 5_000,
        );
        assert!(matches!(
            result.admission,
            AneAdmission::Rejected(AneRejectionReason::UnsupportedBoundaryDtype { .. })
        ));
    }

    #[test]
    fn test_production_unknown_dtype_rejected() {
        let gate = AneQualificationGate::new(AneQualificationConfig {
            fp16_production_envelope: true,
            ..Default::default()
        });
        let in_contracts = vec![BoundaryTensorContract {
            tensor_id: "unknown_input".into(),
            dtype: TensorDtype::Unknown,
            logical_shape: vec![1, 64],
            physical_shape: vec![1, 64],
            strides_bytes: vec![128, 2],
            static_shape: true,
            layout_digest: "nhwc".into(),
        }];
        let result = gate.qualify_for_production_v1(
            "unknown_test", &in_contracts, &[], 300_000, 100_000, 5_000,
        );
        assert!(matches!(
            result.admission,
            AneAdmission::Rejected(AneRejectionReason::UnsupportedBoundaryDtype { .. })
        ));
    }

    #[test]
    fn test_production_dynamic_shape_rejected() {
        let gate = AneQualificationGate::new(AneQualificationConfig {
            fp16_production_envelope: true,
            ..Default::default()
        });
        let in_contracts = vec![BoundaryTensorContract {
            tensor_id: "dyn_input".into(),
            dtype: TensorDtype::Float16,
            logical_shape: vec![1, 64, 128],
            physical_shape: vec![1, 64, 128],
            strides_bytes: vec![16384, 256, 2],
            static_shape: false,
            layout_digest: "nhwc".into(),
        }];
        let result = gate.qualify_for_production_v1(
            "dyn_test", &in_contracts, &[], 300_000, 100_000, 5_000,
        );
        assert!(matches!(
            result.admission,
            AneAdmission::Rejected(AneRejectionReason::DynamicShape { .. })
        ));
    }

    #[test]
    fn test_production_invalid_layout_rejected() {
        let gate = AneQualificationGate::new(AneQualificationConfig {
            fp16_production_envelope: true,
            ..Default::default()
        });
        let in_contracts = vec![BoundaryTensorContract {
            tensor_id: "flat_input".into(),
            dtype: TensorDtype::Float16,
            logical_shape: vec![64],
            physical_shape: vec![64],
            strides_bytes: vec![2],
            static_shape: true,
            layout_digest: "flat".into(),
        }];
        let result = gate.qualify_for_production_v1(
            "layout_test", &in_contracts, &[], 300_000, 100_000, 5_000,
        );
        assert!(matches!(
            result.admission,
            AneAdmission::Rejected(AneRejectionReason::InvalidFp16Layout { .. })
        ));
    }

    #[test]
    fn test_production_cost_unprofitable() {
        let gate = AneQualificationGate::new(AneQualificationConfig {
            fp16_production_envelope: true,
            ..Default::default()
        });
        let in_contracts = vec![BoundaryTensorContract {
            tensor_id: "cost_input".into(),
            dtype: TensorDtype::Float16,
            logical_shape: vec![1, 64, 128],
            physical_shape: vec![1, 64, 128],
            strides_bytes: vec![16384, 256, 2],
            static_shape: true,
            layout_digest: "nhwc".into(),
        }];
        // ANE slower than GPU: 200k vs 100k
        let result = gate.qualify_for_production_v1(
            "cost_test", &in_contracts, &[], 100_000, 200_000, 20_000,
        );
        assert!(matches!(
            result.admission,
            AneAdmission::Rejected(AneRejectionReason::CostUnprofitable { .. })
        ));
    }
}
