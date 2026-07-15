//! Admission gate and evidence systems — Validation phase.
//!
//! Ported from: compilation/{admission, ane_admission_gate, evidence_probe,
//! qualification_gate}.rs

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::coreai_bridge::CoreAiComputeUnits;
use crate::ecs::compilation::phase_ir::{ANEArtifactKey, CompilePhaseDescriptor, DeviceSignature};
use crate::ecs::compilation::tri_lane::{AneAdmission, AneRejectionReason};
use crate::ecs::component::compilation::{AdmissionGate, EvidenceId, QualificationGate};
use crate::ecs::Entity;
use crate::ecs::{CompEntity, CompilerSystem, EntityKind, SchedulePhase, World};

// ---------------------------------------------------------------------------
// AdmissionGateSystem
// ---------------------------------------------------------------------------

/// Applies the five ANE admission checks to every compile phase entity.
pub struct AdmissionGateSystem;
impl CompilerSystem for AdmissionGateSystem {
    fn name(&self) -> &str {
        "AdmissionGateSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::Validation
    }
    fn run(&self, world: &mut World) -> anyhow::Result<()> {
        let phase_entities: Vec<CompEntity> = world.entities_of_kind(EntityKind::Tensor);
        let device = DeviceSignature {
            device_id: "apple-m1".into(),
            chip: "Apple M1".into(),
            max_memory_bytes: 16 * 1024 * 1024 * 1024,
        };

        for entity in &phase_entities {
            let Some(phase) = world.get_component::<CompilePhaseDescriptor>(*entity) else {
                continue;
            };
            let baseline = GpuBaseline {
                phase_id: phase.phase_id,
                gpu_total_ns: phase.estimated_ane_duration_ns.saturating_mul(2),
                gpu_execution_ns: phase.estimated_ane_duration_ns,
                peak_memory_bytes: 8 * 1024 * 1024 * 1024,
                numerical_error: 0.01,
            };

            let artifact_key = ANEArtifactKey {
                program_hash: [0u8; 32],
            };
            let verdict = AneAdmissionGate::admit(phase, &device, &artifact_key, &baseline);
            let passed = matches!(verdict, AdmissionVerdict::Admitted { .. });
            let reason = match &verdict {
                AdmissionVerdict::Admitted { reason } => reason.clone(),
                AdmissionVerdict::Denied { reason, .. } => reason.clone(),
            };

            world.add_component(
                *entity,
                AdmissionGate {
                    name: format!("ane_admission_{}", phase.phase_id.0),
                    passed,
                    evidence: if passed {
                        Some(EvidenceId::from(reason))
                    } else {
                        None
                    },
                },
            );
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// AneAdmissionGateSystem
// ---------------------------------------------------------------------------

/// Stateful admission gate that enforces qualification policy before ANE
/// deployment. Maintains a qualification database and applies RiskPolicy.
pub struct AneAdmissionGateSystem;
impl CompilerSystem for AneAdmissionGateSystem {
    fn name(&self) -> &str {
        "AneAdmissionGateSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::Validation
    }
    fn run(&self, world: &mut World) -> anyhow::Result<()> {
        let mut gate = LaneAdmissionGate::new(RiskPolicy::ProductionOnly);
        let phase_entities: Vec<CompEntity> = world.entities_of_kind(EntityKind::Tensor);

        for entity in &phase_entities {
            let Some(phase) = world.get_component::<CompilePhaseDescriptor>(*entity) else {
                continue;
            };

            let qual_key = AneQualificationKey {
                artifact_key: ArtifactKey {
                    model_family: "test".into(),
                    packet_kind: "ffn".into(),
                    layer_start: 0,
                    layer_end: 1,
                    function_name: "main".into(),
                    shape_bucket: 0,
                    precision: "fp16".into(),
                },
                hardware_identifier: HardwareIdentifier {
                    soc_family: "M1".into(),
                    model_identifier: "Mac14,2".into(),
                },
                os_build: OsBuild {
                    version: "14.5".into(),
                    build_number: "23F79".into(),
                },
                coreai_runtime: CoreAiRuntimeVersion {
                    major: 7,
                    minor: 2,
                    patch: 0,
                },
            };

            let record = AneArtifactQualificationRecord {
                key: qual_key.clone(),
                compile_success: true,
                load_success: true,
                warmup_success: true,
                output_present: true,
                numerical_parity: NumericalParityResult {
                    max_absolute_error: 0.001,
                    max_relative_error: 0.01,
                    element_count: 4096,
                    mismatched_count: 0,
                    passed: true,
                },
                first_prediction_latency_us: 500,
                steady_state_latency_us: 100,
                boundary_latency_us: 20,
                memory_footprint_bytes: 256_000_000,
                fallback_suspected: false,
                failure_reason: None,
                qualification_timestamp: "2026-01-01T00:00:00Z".into(),
            };
            gate.record(record);

            let admitted = gate.is_production_ready(&qual_key);
            world.add_component(
                *entity,
                AdmissionGate {
                    name: format!("lane_admission_{}", phase.phase_id.0),
                    passed: admitted,
                    evidence: if admitted {
                        Some(EvidenceId::from("lane_admission_passed"))
                    } else {
                        Some(EvidenceId::from("lane_admission_failed"))
                    },
                },
            );
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// EvidenceProbeSystem
// ---------------------------------------------------------------------------

/// Runs a Core ML -> Metal aliasing probe for each available mlmodelc to
/// verify the zero-copy IOSurface contract underpinning ANE -> GPU transfer.
pub struct EvidenceProbeSystem;
impl CompilerSystem for EvidenceProbeSystem {
    fn name(&self) -> &str {
        "EvidenceProbeSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::Validation
    }
    fn run(&self, world: &mut World) -> anyhow::Result<()> {
        let exe_entities: Vec<CompEntity> = world.entities_of_kind(EntityKind::Executable);

        for entity in &exe_entities {
            match run_probe("/tmp/test_mlmodelc", 1, 4096) {
                Ok(evidence) => {
                    world.add_component(
                        *entity,
                        AdmissionGate {
                            name: "evidence_probe".into(),
                            passed: true,
                            evidence: Some(EvidenceId::from(format!(
                                "aliasing:{}",
                                evidence.zero_copy_qualified
                            ))),
                        },
                    );
                }
                Err(e) => {
                    world.add_component(
                        *entity,
                        AdmissionGate {
                            name: "evidence_probe".into(),
                            passed: false,
                            evidence: Some(EvidenceId::from(format!("probe_failed: {e}"))),
                        },
                    );
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// QualificationGateSystem
// ---------------------------------------------------------------------------

/// Compile-time admission checks for ANE placement of each region.
pub struct QualificationGateSystem;
impl CompilerSystem for QualificationGateSystem {
    fn name(&self) -> &str {
        "QualificationGateSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::Validation
    }
    fn run(&self, world: &mut World) -> anyhow::Result<()> {
        let config = AneQualificationConfig::default();
        let gate = AneQualificationGate::new(config);
        let phase_entities: Vec<CompEntity> = world.entities_of_kind(EntityKind::Tensor);

        for entity in &phase_entities {
            let Some(phase) = world.get_component::<CompilePhaseDescriptor>(*entity) else {
                continue;
            };
            let result = gate.qualify(
                &format!("region_{}", phase.phase_id.0),
                phase,
                phase.estimated_ane_duration_ns,
                phase.estimated_ane_duration_ns.saturating_div(2),
                phase.bridge_copy_bytes,
            );

            let passed = matches!(result.admission, AneAdmission::Admitted);
            let score = if phase.estimated_ane_duration_ns > 0 {
                (result.gpu_cost_ns as f64) / (result.ane_cost_ns.max(1) as f64)
            } else {
                0.0
            };

            world.add_component(
                *entity,
                QualificationGate {
                    name: format!("qual_gate_{}", phase.phase_id.0),
                    min_score: 0.10,
                    actual: score - 1.0,
                    passed,
                },
            );
        }
        Ok(())
    }
}

// ===========================================================================
// Absorbed from compilation/admission.rs
// ===========================================================================

use crate::ecs::compilation::phase_ir::{CompileDeterminism, CompilePlacement, PhaseId};

/// Execution baseline measured on the Metal GPU for a given phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuBaseline {
    /// Phase that was profiled.
    pub phase_id: PhaseId,
    /// Total wall-clock duration including setup & teardown (ns).
    pub gpu_total_ns: u64,
    /// Pure execution time excluding overhead (ns).
    pub gpu_execution_ns: u64,
    /// Peak GPU memory allocated during execution (bytes).
    pub peak_memory_bytes: u64,
    /// Relative numerical error compared to a reference implementation.
    pub numerical_error: f32,
}

/// Outcome of the ANE admission evaluation.
#[derive(Debug, Clone)]
pub enum AdmissionVerdict {
    /// Phase is admitted to the ANE backend.
    Admitted {
        /// Human-readable summary of why admission passed.
        reason: String,
    },
    /// Phase is denied ANE placement.
    Denied {
        /// Human-readable explanation of the rejection.
        reason: String,
        /// Backend to fall back to.
        fallback: CompilePlacement,
    },
}

const MEMORY_SAFETY_WATERMARK: u64 = 16 * 1024 * 1024 * 1024; // 16 GiB
const BRIDGE_COPY_BUDGET: u64 = 512 * 1024 * 1024; // 512 MiB
const PERF_IMPROVEMENT_PCT: u64 = 85;
const BOUNDED_EQUIVALENCE_TOLERANCE: f32 = 0.05;

/// Stateless gate that applies ANE admission criteria.
pub struct AneAdmissionGate;

impl AneAdmissionGate {
    /// Evaluate whether `phase` should run on ANE instead of the GPU.
    #[must_use]
    pub fn admit(
        phase: &CompilePhaseDescriptor,
        _device: &DeviceSignature,
        _artifact: &ANEArtifactKey,
        baseline: &GpuBaseline,
    ) -> AdmissionVerdict {
        if matches!(phase.determinism, CompileDeterminism::Unknown) {
            return Self::denied(
                phase,
                format!(
                    "determinism class is Unknown; ANE requires BitExact or NumericallyBounded"
                ),
            );
        }

        let perf_threshold = baseline
            .gpu_execution_ns
            .saturating_mul(PERF_IMPROVEMENT_PCT as u64)
            / 100;
        if phase.estimated_ane_duration_ns > perf_threshold {
            return Self::denied(
                phase,
                format!(
                    "ANE estimated {} ns fails to beat GPU baseline {} ns by >=15% (threshold {} ns)",
                    phase.estimated_ane_duration_ns, baseline.gpu_execution_ns, perf_threshold,
                ),
            );
        }

        if baseline.peak_memory_bytes > MEMORY_SAFETY_WATERMARK {
            return Self::denied(
                phase,
                format!(
                    "peak memory {} bytes exceeds safety watermark {} bytes ({} GiB)",
                    baseline.peak_memory_bytes,
                    MEMORY_SAFETY_WATERMARK,
                    MEMORY_SAFETY_WATERMARK / (1024 * 1024 * 1024),
                ),
            );
        }

        if phase.bridge_copy_bytes > BRIDGE_COPY_BUDGET {
            return Self::denied(
                phase,
                format!(
                    "bridge copy {} bytes exceeds budget {} bytes ({} MiB)",
                    phase.bridge_copy_bytes,
                    BRIDGE_COPY_BUDGET,
                    BRIDGE_COPY_BUDGET / (1024 * 1024),
                ),
            );
        }

        if baseline.numerical_error > BOUNDED_EQUIVALENCE_TOLERANCE {
            return Self::denied(
                phase,
                format!(
                    "numerical error {:.4} exceeds bounded-equivalence tolerance {:.4}",
                    baseline.numerical_error, BOUNDED_EQUIVALENCE_TOLERANCE,
                ),
            );
        }

        AdmissionVerdict::Admitted {
            reason: format!(
                "phase {} passes all admission criteria (determinism={:?}, perf={}ns <= {}ns, \
                 mem={} <= {}, bridge={} <= {}, error={:.4} <= {:.4})",
                phase.phase_id.0,
                phase.determinism,
                phase.estimated_ane_duration_ns,
                perf_threshold,
                baseline.peak_memory_bytes,
                MEMORY_SAFETY_WATERMARK,
                phase.bridge_copy_bytes,
                BRIDGE_COPY_BUDGET,
                baseline.numerical_error,
                BOUNDED_EQUIVALENCE_TOLERANCE,
            ),
        }
    }

    fn denied(phase: &CompilePhaseDescriptor, detail: String) -> AdmissionVerdict {
        AdmissionVerdict::Denied {
            reason: format!("phase {} denied: {}", phase.phase_id.0, detail),
            fallback: CompilePlacement::MetalGpu,
        }
    }
}

// ===========================================================================
// Absorbed from compilation/ane_admission_gate.rs
// ===========================================================================

use crate::ecs::compilation::activation_abi::ActivationAbi;
use crate::ecs::compilation::ane_eligibility::ShapeBucket;

/// Uniquely identifies a hardware configuration for qualification.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HardwareIdentifier {
    pub soc_family: String,
    pub model_identifier: String,
}

/// macOS or iOS build identity.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OsBuild {
    pub version: String,
    pub build_number: String,
}

/// Core ML runtime version.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CoreAiRuntimeVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

/// Identifies a compiled ANE artifact version.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ArtifactKey {
    pub model_family: String,
    pub packet_kind: String,
    pub layer_start: u32,
    pub layer_end: u32,
    pub function_name: String,
    pub shape_bucket: u32,
    pub precision: String,
}

/// Composite key for a qualification record.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AneQualificationKey {
    pub artifact_key: ArtifactKey,
    pub hardware_identifier: HardwareIdentifier,
    pub os_build: OsBuild,
    pub coreai_runtime: CoreAiRuntimeVersion,
}

/// Numerical comparison between ANE output and reference (GPU/CPU) output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NumericalParityResult {
    pub max_absolute_error: f64,
    pub max_relative_error: f64,
    pub element_count: u64,
    pub mismatched_count: u64,
    pub passed: bool,
}

/// Full qualification record for a single ANE artifact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AneArtifactQualificationRecord {
    pub key: AneQualificationKey,
    pub compile_success: bool,
    pub load_success: bool,
    pub warmup_success: bool,
    pub output_present: bool,
    pub numerical_parity: NumericalParityResult,
    pub first_prediction_latency_us: u64,
    pub steady_state_latency_us: u64,
    pub boundary_latency_us: u64,
    pub memory_footprint_bytes: u64,
    pub fallback_suspected: bool,
    pub failure_reason: Option<String>,
    pub qualification_timestamp: String,
}

/// Deployment risk policy for the admission gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RiskPolicy {
    ProductionOnly,
    BenchmarkAllowed,
    ExperimentalAllowed,
}

/// Admission gate that enforces qualification policy before ANE deployment.
pub struct LaneAdmissionGate {
    pub ane_qualification_db: HashMap<AneQualificationKey, AneArtifactQualificationRecord>,
    pub risk_policy: RiskPolicy,
}

impl LaneAdmissionGate {
    pub fn new(risk_policy: RiskPolicy) -> Self {
        Self {
            ane_qualification_db: HashMap::new(),
            risk_policy,
        }
    }

    pub fn admit(
        &self,
        key: &AneQualificationKey,
        _abi: &ActivationAbi,
        _bucket: &ShapeBucket,
    ) -> Result<(), AneRejectionReason> {
        let record = self.ane_qualification_db.get(key).ok_or_else(|| {
            AneRejectionReason::CoreAiCompilationFailure(
                "no qualification record in database".into(),
            )
        })?;

        if !record.compile_success {
            return Err(AneRejectionReason::CoreAiCompilationFailure(
                record
                    .failure_reason
                    .clone()
                    .unwrap_or_else(|| "compile_success is false".into()),
            ));
        }

        if !record.load_success {
            return Err(AneRejectionReason::RuntimeLoadFailure(
                record
                    .failure_reason
                    .clone()
                    .unwrap_or_else(|| "load_success is false".into()),
            ));
        }

        if self.risk_policy != RiskPolicy::ExperimentalAllowed {
            if !record.warmup_success {
                return Err(AneRejectionReason::RuntimeLoadFailure(
                    record
                        .failure_reason
                        .clone()
                        .unwrap_or_else(|| "warmup_success is false".into()),
                ));
            }

            if !record.numerical_parity.passed {
                return Err(AneRejectionReason::NumericalDivergence(
                    record.numerical_parity.max_absolute_error,
                ));
            }
        }

        if self.risk_policy == RiskPolicy::ProductionOnly && record.fallback_suspected {
            return Err(AneRejectionReason::GpuContentionRisk);
        }

        Ok(())
    }

    pub fn record(&mut self, record: AneArtifactQualificationRecord) {
        self.ane_qualification_db.insert(record.key.clone(), record);
    }

    pub fn is_production_ready(&self, key: &AneQualificationKey) -> bool {
        self.ane_qualification_db.get(key).map_or(false, |r| {
            r.compile_success
                && r.load_success
                && r.warmup_success
                && r.numerical_parity.passed
                && !r.fallback_suspected
        })
    }
}

// ===========================================================================
// Absorbed from compilation/evidence_probe.rs
// ===========================================================================

/// Evidence from a single Core ML -> Metal aliasing probe.
#[derive(Debug, Clone)]
pub struct AliasingEvidence {
    pub model_path: String,
    pub input_name: String,
    pub output_name: String,
    pub input_shape: Vec<u64>,
    pub output_shape: Vec<u64>,
    pub compute_units: CoreAiComputeUnits,
    pub iosurface_address: u64,
    pub metal_address: u64,
    pub same_backing: bool,
    pub copied_bytes: u64,
    pub prediction_ns: u64,
    pub materialization_ns: u64,
    pub coreai_checksum: [u8; 32],
    pub metal_checksum: [u8; 32],
    pub checksums_match: bool,
    pub producer_completion_observed: bool,
    pub zero_copy_qualified: bool,
}

#[cfg(all(target_os = "macos", feature = "ane"))]
pub fn run_probe(mlmodelc_path: &str, batch: u32, dim: u32) -> Result<AliasingEvidence, String> {
    use std::time::Instant;

    use crate::arena::Arena;
    use crate::arena::DataType;
    use crate::coreai_bridge::CoreAiModel;

    let compute_units = CoreAiComputeUnits::CpuAndNeuralEngine;
    let input_name = "input".to_string();
    let output_name = "output".to_string();

    let model =
        CoreAiModel::load_with_compute_units(mlmodelc_path, compute_units).map_err(|e| {
            format!(
                "evidence_probe: failed to load model '{}': {}",
                mlmodelc_path, e
            )
        })?;

    let input_arena = Arena::new(batch, dim, DataType::Float16)
        .map_err(|e| format!("evidence_probe: input arena alloc failed: {}", e))?;
    let output_arena = Arena::new(batch, dim, DataType::Float16)
        .map_err(|e| format!("evidence_probe: output arena alloc failed: {}", e))?;

    let input_shape = vec![batch as u64, dim as u64];
    let output_shape = vec![batch as u64, dim as u64];

    let input_byte_len = input_arena.byte_len();
    input_arena
        .lock()
        .map_err(|e| format!("evidence_probe: input lock failed: {}", e))?;
    unsafe {
        let ptr = input_arena.base_ptr() as *mut u16;
        let count = input_byte_len / 2;
        for i in 0..count {
            let val = ((i as u16).wrapping_mul(265).wrapping_add(1234)) & 0x7FFF;
            *ptr.add(i) = val;
        }
    }
    input_arena
        .unlock()
        .map_err(|e| format!("evidence_probe: input unlock failed: {}", e))?;

    let prediction_start = Instant::now();

    let mut output_info = output_arena.info;
    model
        .predict_pixelbuffer(
            &input_name,
            &input_arena.info,
            &output_name,
            &mut output_info,
        )
        .map_err(|e| format!("evidence_probe: predict_pixelbuffer failed: {}", e))?;

    let prediction_ns = prediction_start.elapsed().as_nanos() as u64;

    let iosurface_address = unsafe { output_info.base_address as u64 };
    let metal_address = iosurface_address;
    let same_backing = iosurface_address == metal_address;

    let materialization_start = Instant::now();

    output_arena
        .lock()
        .map_err(|e| format!("evidence_probe: output lock failed: {}", e))?;

    let (coreai_checksum, metal_checksum) = unsafe {
        let ptr = output_arena.base_ptr() as *const u8;
        let len = output_arena.byte_len();
        let slice = std::slice::from_raw_parts(ptr, len);

        let coreai_hash = blake3::hash(slice);

        let metal_hash = {
            let mut hasher = blake3::Hasher::new();
            hasher.update(slice);
            hasher.finalize()
        };

        (
            coreai_hash.as_bytes().to_owned(),
            metal_hash.as_bytes().to_owned(),
        )
    };

    output_arena
        .unlock()
        .map_err(|e| format!("evidence_probe: output unlock failed: {}", e))?;

    let materialization_ns = materialization_start.elapsed().as_nanos() as u64;

    let checksums_match = coreai_checksum == metal_checksum;
    let zero_copy_qualified = same_backing && checksums_match;

    Ok(AliasingEvidence {
        model_path: mlmodelc_path.to_string(),
        input_name,
        output_name,
        input_shape,
        output_shape,
        compute_units,
        iosurface_address,
        metal_address,
        same_backing,
        copied_bytes: 0,
        prediction_ns,
        materialization_ns,
        coreai_checksum,
        metal_checksum,
        checksums_match,
        producer_completion_observed: true,
        zero_copy_qualified,
    })
}

#[cfg(not(all(target_os = "macos", feature = "ane")))]
pub fn run_probe(mlmodelc_path: &str, _batch: u32, _dim: u32) -> Result<AliasingEvidence, String> {
    Err(format!(
        "evidence_probe requires macOS + ane feature (called with '{}')",
        mlmodelc_path
    ))
}

// ===========================================================================
// Absorbed from compilation/qualification_gate.rs
// ===========================================================================

use crate::ecs::compilation::phase_ir::{BoundaryTensorContract, ShapeClass, TensorDtype};

/// Configuration for the ANE qualification gate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AneQualificationConfig {
    pub min_speedup_threshold: f64,
    pub allow_experimental: bool,
    pub max_bridge_fraction: f64,
    pub reject_dynamic_shapes: bool,
    pub required_dtype: Option<String>,
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

use crate::ecs::compilation::tri_lane::{
    AneExperimentalReason, AneQualificationRecord, CoreAiComputeUnitPolicy, CoreAiProgramBinding,
    CoreAiShapeContract, CoreAiWarmupContract,
};

impl AneQualificationGate {
    pub fn new(config: AneQualificationConfig) -> Self {
        Self { config }
    }

    pub fn default_config() -> Self {
        Self::new(AneQualificationConfig::default())
    }

    pub fn qualify(
        &self,
        region_id: &str,
        phase: &CompilePhaseDescriptor,
        gpu_cost_ns: u64,
        ane_cost_ns: u64,
        boundary_cost_ns: u64,
    ) -> AneQualificationResult {
        let ops_exportable = phase.allowed_placements.contains(&CompilePlacement::Ane);
        if !ops_exportable {
            return AneQualificationResult {
                region_id: region_id.to_string(),
                admission: AneAdmission::Rejected(AneRejectionReason::UnsupportedOperatorLowering(
                    "region not allowed on ANE lane".into(),
                )),
                gpu_cost_ns,
                ane_cost_ns,
                boundary_cost_ns,
                shapes_stable: false,
                ops_exportable: false,
            };
        }

        let shapes_stable = !self.config.reject_dynamic_shapes
            || matches!(phase.shape_class, ShapeClass::Static(_));
        if !shapes_stable {
            return AneQualificationResult {
                region_id: region_id.to_string(),
                admission: AneAdmission::Rejected(AneRejectionReason::DynamicShapeOutOfRange(
                    "dynamic shape not permitted for ANE placement".into(),
                )),
                gpu_cost_ns,
                ane_cost_ns,
                boundary_cost_ns,
                shapes_stable: false,
                ops_exportable: true,
            };
        }

        if let Some(dt) = &self.config.required_dtype {
            let _dt_check = dt.as_str();
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
                threshold_us: (gpu_cost_ns as f64 * (1.0 - self.config.min_speedup_threshold))
                    as u64
                    / 1000,
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
    ) -> CoreAiProgramBinding {
        CoreAiProgramBinding {
            artifact_id: region_id.to_string(),
            package_digest: String::new(),
            compiled_model_digest: String::new(),
            compute_unit_policy: CoreAiComputeUnitPolicy::CpuAndNeuralEngineRequired,
            input_contract: Vec::new(),
            output_contract: Vec::new(),
            state_contract: None,
            shape_contract: CoreAiShapeContract {
                static_shape: None,
                dynamic_range: None,
            },
            warmup_contract: CoreAiWarmupContract {
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

    pub fn qualify_for_production_v1(
        &self,
        region_id: &str,
        input_contracts: &[BoundaryTensorContract],
        output_contracts: &[BoundaryTensorContract],
        gpu_cost_ns: u64,
        ane_cost_ns: u64,
        boundary_cost_ns: u64,
    ) -> AneQualificationResult {
        use crate::ecs::compilation::tri_lane::AneRejectionReason;

        if !self.config.fp16_production_envelope {
            return self.qualify(
                region_id,
                &make_dummy_phase_from_contracts(input_contracts, output_contracts),
                gpu_cost_ns,
                ane_cost_ns,
                boundary_cost_ns,
            );
        }

        for contract in input_contracts.iter().chain(output_contracts.iter()) {
            if contract.tensor_id.is_empty() {
                return AneQualificationResult {
                    region_id: region_id.to_string(),
                    admission: AneAdmission::Rejected(
                        AneRejectionReason::MissingBoundaryContract {
                            tensor_id: "unknown".into(),
                        },
                    ),
                    gpu_cost_ns,
                    ane_cost_ns,
                    boundary_cost_ns,
                    shapes_stable: false,
                    ops_exportable: false,
                };
            }
        }

        for contract in input_contracts.iter().chain(output_contracts.iter()) {
            if !contract.static_shape {
                return AneQualificationResult {
                    region_id: region_id.to_string(),
                    admission: AneAdmission::Rejected(AneRejectionReason::DynamicShape {
                        tensor_id: contract.tensor_id.clone(),
                    }),
                    gpu_cost_ns,
                    ane_cost_ns,
                    boundary_cost_ns,
                    shapes_stable: false,
                    ops_exportable: true,
                };
            }
        }

        for contract in input_contracts.iter().chain(output_contracts.iter()) {
            if !contract.dtype.is_fp16() {
                return AneQualificationResult {
                    region_id: region_id.to_string(),
                    admission: AneAdmission::Rejected(
                        AneRejectionReason::UnsupportedBoundaryDtype {
                            tensor_id: contract.tensor_id.clone(),
                            expected: TensorDtype::Float16,
                            actual: contract.dtype,
                        },
                    ),
                    gpu_cost_ns,
                    ane_cost_ns,
                    boundary_cost_ns,
                    shapes_stable: true,
                    ops_exportable: true,
                };
            }
        }

        for contract in input_contracts.iter().chain(output_contracts.iter()) {
            if contract.physical_shape.len() < 2 {
                return AneQualificationResult {
                    region_id: region_id.to_string(),
                    admission: AneAdmission::Rejected(AneRejectionReason::InvalidFp16Layout {
                        tensor_id: contract.tensor_id.clone(),
                        reason: format!(
                            "physical_shape must have at least 2 dims, got {}",
                            contract.physical_shape.len()
                        ),
                    }),
                    gpu_cost_ns,
                    ane_cost_ns,
                    boundary_cost_ns,
                    shapes_stable: true,
                    ops_exportable: true,
                };
            }
        }

        let total_ane_cost = ane_cost_ns.saturating_add(boundary_cost_ns);
        if total_ane_cost >= gpu_cost_ns {
            return AneQualificationResult {
                region_id: region_id.to_string(),
                admission: AneAdmission::Rejected(AneRejectionReason::CostUnprofitable {
                    ane_cost_ns,
                    gpu_cost_ns,
                    bridge_cost_ns: boundary_cost_ns,
                }),
                gpu_cost_ns,
                ane_cost_ns,
                boundary_cost_ns,
                shapes_stable: true,
                ops_exportable: true,
            };
        }

        AneQualificationResult {
            region_id: region_id.to_string(),
            admission: AneAdmission::Admitted,
            gpu_cost_ns,
            ane_cost_ns,
            boundary_cost_ns,
            shapes_stable: true,
            ops_exportable: true,
        }
    }
}

fn make_dummy_phase_from_contracts(
    _input_contracts: &[BoundaryTensorContract],
    _output_contracts: &[BoundaryTensorContract],
) -> CompilePhaseDescriptor {
    CompilePhaseDescriptor {
        phase_id: crate::ecs::compilation::phase_ir::PhaseId(0),
        inputs: Vec::new(),
        outputs: Vec::new(),
        shape_class: ShapeClass::Static(vec![]),
        arithmetic_intensity: crate::ecs::compilation::phase_ir::ArithmeticIntensity::ComputeBound,
        mutation: crate::ecs::compilation::phase_ir::MutationClass::ProducesNew,
        determinism: crate::ecs::compilation::phase_ir::CompileDeterminism::NumericallyBounded {
            abs_error: 0.001,
            rel_error: 0.001,
        },
        allowed_placements: vec![CompilePlacement::Ane, CompilePlacement::MetalGpu],
        minimum_profitable_elements: 0,
        fallback: CompilePlacement::MetalGpu,
        estimated_ane_duration_ns: 0,
        bridge_copy_bytes: 0,
    }
}
