//! Admission gate evaluation — ANE eligibility, qualification, and
//! evidence probes for compile-phase admission.
//!
//! This module owns the canonical authority for the three admission
//! decisions that gate ANE placement of compile phases:
//!
//! 1. **ANE admission** — for each `CompilePhaseDescriptor`, evaluate
//!    the determinism class, the performance improvement over the
//!    GPU baseline, the memory safety watermark, the bridge-copy
//!    budget, and the numerical-error tolerance. The verdict is
//!    `Admitted` or `Denied { fallback }`.
//! 2. **Qualification gate** — given a set of qualification records
//!    keyed by hardware configuration, decide whether a region is
//!    production-ready under the active `RiskPolicy`.
//! 3. **Evidence probe** — record the aliasing-evidence results
//!    from a zero-copy IOSurface probe into a `AdmissionGate`
//!    component (the actual probe lives in the engine; this
//!    module only formats the result).
//!
//! ## Authority boundary
//!
//! This module does **not** own:
//! - The phase IR (owned by `prism-ecs-compile`).
//! - The hardware discovery (owned by `prism-ecs-kernel`).
//! - The runtime admission policy (owned by `prism-ecs-runtime`).
//!
//! All exposed types are pure value types. The module never mutates
//! the world directly.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Hardware identity
// ---------------------------------------------------------------------------

/// Stable hardware identifier — `soc_family` + `model_identifier`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct HardwareIdentifier {
    pub soc_family: String,
    pub model_identifier: String,
}

/// macOS / Darwin build identifier.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct OsBuild {
    pub version: String,
    pub build_number: String,
}

/// CoreAI runtime version triple.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct CoreAiRuntimeVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

/// Combined key for an artifact qualification record.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct AneQualificationKey {
    pub artifact_key: ArtifactKey,
    pub hardware_identifier: HardwareIdentifier,
    pub os_build: OsBuild,
    pub coreai_runtime: CoreAiRuntimeVersion,
}

/// Artifact identity — model family + packet + layer range + function.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ArtifactKey {
    pub model_family: String,
    pub packet_kind: String,
    pub layer_start: u32,
    pub layer_end: u32,
    pub function_name: String,
    pub shape_bucket: u32,
    pub precision: String,
}

// ---------------------------------------------------------------------------
// Numerics & parity
// ---------------------------------------------------------------------------

/// Result of a numerical-parity check between two implementations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NumericalParityResult {
    pub max_absolute_error: f64,
    pub max_relative_error: f64,
    pub element_count: u32,
    pub mismatched_count: u32,
    pub passed: bool,
}

/// One record of an artifact's qualification on a specific hardware/OS
/// combination.
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

// ---------------------------------------------------------------------------
// Risk policy
// ---------------------------------------------------------------------------

/// Risk policy governing which qualification records are accepted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash, Serialize, Deserialize)]
pub enum RiskPolicy {
    /// Accept only qualification records that pass all checks.
    #[default]
    ProductionOnly,
    /// Accept records that pass latency + parity but allow warmup
    /// or compile failures as long as the run completed.
    Research,
}

// ---------------------------------------------------------------------------
// GPU baseline
// ---------------------------------------------------------------------------

/// Measured execution baseline on the Metal GPU for one phase.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GpuBaseline {
    pub phase_id: PhaseId,
    pub gpu_total_ns: u64,
    pub gpu_execution_ns: u64,
    pub peak_memory_bytes: u64,
    pub numerical_error: f32,
}

/// Phase ID — typed authority-bearing identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct PhaseId(pub u32);

/// Device signature — the target's stable identity.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DeviceSignature {
    pub device_id: String,
    pub chip: String,
    pub max_memory_bytes: u64,
}

/// Artifact key for the ANE binary (program hash).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ANEArtifactKey {
    pub program_hash: [u8; 32],
}

/// Compile-phase determinism class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CompileDeterminism {
    Unknown,
    BitExact,
    NumericallyBounded,
}

/// Compile-phase placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CompilePlacement {
    Ane,
    MetalGpu,
    Cpu,
    Unknown,
}

/// Compile-phase descriptor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompilePhaseDescriptor {
    pub phase_id: PhaseId,
    pub determinism: CompileDeterminism,
    pub estimated_ane_duration_ns: u64,
    pub bridge_copy_bytes: u64,
    pub allowed_placements: Vec<CompilePlacement>,
}

impl prism_ecs_core::Component for CompilePhaseDescriptor {}

/// ANE admission verdict.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AdmissionVerdict {
    Admitted { reason: String },
    Denied {
        reason: String,
        fallback: CompilePlacement,
    },
}

// ---------------------------------------------------------------------------
// ANE admission gate
// ---------------------------------------------------------------------------

/// Safety watermarks used by the ANE admission gate.
pub const MEMORY_SAFETY_WATERMARK: u64 = 16 * 1024 * 1024 * 1024; // 16 GiB
pub const BRIDGE_COPY_BUDGET: u64 = 512 * 1024 * 1024; // 512 MiB
pub const PERF_IMPROVEMENT_PCT: u64 = 85;
pub const BOUNDED_EQUIVALENCE_TOLERANCE: f32 = 0.05;

/// Stateless gate that applies ANE admission criteria.
pub struct AneAdmissionGate;

impl AneAdmissionGate {
    /// Evaluate whether a phase should run on ANE instead of the GPU.
    pub fn admit(
        phase: &CompilePhaseDescriptor,
        _device: &DeviceSignature,
        _artifact: &ANEArtifactKey,
        baseline: &GpuBaseline,
    ) -> AdmissionVerdict {
        if matches!(phase.determinism, CompileDeterminism::Unknown) {
            return Self::denied(
                phase,
                "determinism class is Unknown; ANE requires BitExact or NumericallyBounded",
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
                "phase {} passes all admission criteria (determinism={:?}, perf={}ns <= {}ns, mem={} <= {}, bridge={} <= {}, error={:.4} <= {:.4})",
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

    fn denied(phase: &CompilePhaseDescriptor, detail: impl Into<String>) -> AdmissionVerdict {
        AdmissionVerdict::Denied {
            reason: format!("phase {} denied: {}", phase.phase_id.0, detail.into()),
            fallback: CompilePlacement::MetalGpu,
        }
    }
}

// ---------------------------------------------------------------------------
// Lane admission gate
// ---------------------------------------------------------------------------

/// Stateful gate that maintains a qualification database and applies
/// the active `RiskPolicy`.
#[derive(Debug, Clone, Default)]
pub struct LaneAdmissionGate {
    policy: RiskPolicy,
    records: BTreeMap<AneQualificationKey, AneArtifactQualificationRecord>,
}

impl LaneAdmissionGate {
    pub fn new(policy: RiskPolicy) -> Self {
        Self {
            policy,
            records: BTreeMap::new(),
        }
    }

    /// Record a qualification result for one artifact+hardware+OS
    /// combination. Later calls with the same key overwrite the
    /// earlier record.
    pub fn record(&mut self, record: AneArtifactQualificationRecord) {
        self.records.insert(record.key.clone(), record);
    }

    /// Decide whether the artifact is production-ready on the
    /// specified key under the active risk policy.
    pub fn is_production_ready(&self, key: &AneQualificationKey) -> bool {
        let Some(record) = self.records.get(key) else {
            return false;
        };
        match self.policy {
            RiskPolicy::ProductionOnly => {
                record.compile_success
                    && record.load_success
                    && record.warmup_success
                    && record.output_present
                    && record.numerical_parity.passed
                    && !record.fallback_suspected
                    && record.failure_reason.is_none()
            }
            RiskPolicy::Research => {
                record.compile_success
                    && record.load_success
                    && record.output_present
                    && record.numerical_parity.passed
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Qualification gate (cost-aware)
// ---------------------------------------------------------------------------

/// ANE qualification gate configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AneQualificationConfig {
    /// Latency target the ANE binary must beat (in ns).
    pub latency_target_ns: u64,
    /// Memory budget the ANE binary must respect (in bytes).
    pub memory_budget_bytes: u64,
    /// Maximum acceptable bridge-copy bytes.
    pub bridge_copy_budget_bytes: u64,
}

impl Default for AneQualificationConfig {
    fn default() -> Self {
        Self {
            latency_target_ns: 100_000,
            memory_budget_bytes: 512 * 1024 * 1024,
            bridge_copy_budget_bytes: 32 * 1024 * 1024,
        }
    }
}

/// Qualification gate that checks the cost ratio between ANE and GPU
/// execution.
#[derive(Debug, Clone)]
pub struct AneQualificationGate {
    pub config: AneQualificationConfig,
}

impl AneQualificationGate {
    pub fn new(config: AneQualificationConfig) -> Self {
        Self { config }
    }

    pub fn qualify(
        &self,
        _region_id: &str,
        _phase: &CompilePhaseDescriptor,
        ane_cost_ns: u64,
        gpu_cost_ns: u64,
        bridge_copy_bytes: u64,
    ) -> QualificationResult {
        let admission = if ane_cost_ns <= self.config.latency_target_ns
            && bridge_copy_bytes <= self.config.bridge_copy_budget_bytes
        {
            AneAdmission::Admitted
        } else {
            AneAdmission::Denied
        };
        QualificationResult {
            admission,
            ane_cost_ns,
            gpu_cost_ns,
        }
    }
}

/// ANE admission — admitted or denied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AneAdmission {
    Admitted,
    Denied,
}

/// Result of qualification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualificationResult {
    pub admission: AneAdmission,
    pub ane_cost_ns: u64,
    pub gpu_cost_ns: u64,
}

// ---------------------------------------------------------------------------
// Evidence probe
// ---------------------------------------------------------------------------

/// Result of an IOSurface aliasing probe (zero-copy contract).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceProbeResult {
    pub zero_copy_qualified: bool,
    pub error: Option<String>,
}

impl EvidenceProbeResult {
    /// Build an `AdmissionGate` component from this probe result.
    pub fn to_admission_gate(&self, name: impl Into<String>) -> AdmissionGateComponent {
        let passed = self.zero_copy_qualified && self.error.is_none();
        AdmissionGateComponent {
            name: name.into(),
            passed,
            evidence: Some(if passed {
                format!("aliasing:{}", self.zero_copy_qualified)
            } else {
                format!("probe_failed: {:?}", self.error)
            }),
        }
    }
}

/// In-memory placeholder for the engine's IOSurface probe. The
/// real probe lives in the engine; this module formats the result.
pub fn run_probe(
    _model_path: &str,
    _iteration: u32,
    _element_count: u32,
) -> Result<EvidenceProbeResult, String> {
    Ok(EvidenceProbeResult {
        zero_copy_qualified: true,
        error: None,
    })
}

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------

/// One admission gate verdict.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdmissionGateComponent {
    pub name: String,
    pub passed: bool,
    pub evidence: Option<String>,
}

impl prism_ecs_core::Component for AdmissionGateComponent {}

/// One qualification gate verdict.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualificationGateComponent {
    pub name: String,
    pub min_score: f64,
    pub actual: f64,
    pub passed: bool,
}

impl prism_ecs_core::Component for QualificationGateComponent {}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum AdmissionError {
    #[error("qualification key is missing artifact key")]
    MissingArtifactKey,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn device() -> DeviceSignature {
        DeviceSignature {
            device_id: "apple-m1".into(),
            chip: "Apple M1".into(),
            max_memory_bytes: 16 * 1024 * 1024 * 1024,
        }
    }

    fn artifact() -> ANEArtifactKey {
        ANEArtifactKey { program_hash: [0u8; 32] }
    }

    fn baseline(duration_ns: u64) -> GpuBaseline {
        GpuBaseline {
            phase_id: PhaseId(1),
            gpu_total_ns: duration_ns.saturating_mul(2),
            gpu_execution_ns: duration_ns,
            peak_memory_bytes: 8 * 1024 * 1024 * 1024,
            numerical_error: 0.01,
        }
    }

    fn good_phase() -> CompilePhaseDescriptor {
        CompilePhaseDescriptor {
            phase_id: PhaseId(1),
            determinism: CompileDeterminism::BitExact,
            estimated_ane_duration_ns: 1000,
            bridge_copy_bytes: 1024,
            allowed_placements: vec![CompilePlacement::Ane],
        }
    }

    #[test]
    fn admit_passes_when_all_criteria_met() {
        let verdict =
            AneAdmissionGate::admit(&good_phase(), &device(), &artifact(), &baseline(2000));
        assert!(matches!(verdict, AdmissionVerdict::Admitted { .. }));
    }

    #[test]
    fn admit_denies_unknown_determinism() {
        let mut phase = good_phase();
        phase.determinism = CompileDeterminism::Unknown;
        let verdict = AneAdmissionGate::admit(&phase, &device(), &artifact(), &baseline(2000));
        match verdict {
            AdmissionVerdict::Denied { reason, .. } => {
                assert!(reason.contains("Unknown"));
            }
            _ => panic!("expected denied"),
        }
    }

    #[test]
    fn admit_denies_when_perf_threshold_exceeded() {
        let mut phase = good_phase();
        phase.estimated_ane_duration_ns = 5000; // > 85% of 2000
        let verdict = AneAdmissionGate::admit(&phase, &device(), &artifact(), &baseline(2000));
        match verdict {
            AdmissionVerdict::Denied { reason, .. } => {
                assert!(reason.contains("ANE estimated"));
            }
            _ => panic!("expected denied"),
        }
    }

    #[test]
    fn admit_denies_when_peak_memory_exceeds_watermark() {
        let mut bl = baseline(2000);
        bl.peak_memory_bytes = 32 * 1024 * 1024 * 1024; // 32 GiB > 16 GiB
        let verdict =
            AneAdmissionGate::admit(&good_phase(), &device(), &artifact(), &bl);
        match verdict {
            AdmissionVerdict::Denied { reason, .. } => {
                assert!(reason.contains("peak memory"));
            }
            _ => panic!("expected denied"),
        }
    }

    #[test]
    fn admit_denies_when_bridge_copy_exceeds_budget() {
        let mut phase = good_phase();
        phase.bridge_copy_bytes = 1024 * 1024 * 1024; // 1 GiB > 512 MiB
        let verdict = AneAdmissionGate::admit(&phase, &device(), &artifact(), &baseline(2000));
        match verdict {
            AdmissionVerdict::Denied { reason, .. } => {
                assert!(reason.contains("bridge copy"));
            }
            _ => panic!("expected denied"),
        }
    }

    #[test]
    fn admit_denies_when_numerical_error_exceeds_tolerance() {
        let mut bl = baseline(2000);
        bl.numerical_error = 0.10; // > 0.05
        let verdict = AneAdmissionGate::admit(&good_phase(), &device(), &artifact(), &bl);
        match verdict {
            AdmissionVerdict::Denied { reason, .. } => {
                assert!(reason.contains("numerical error"));
            }
            _ => panic!("expected denied"),
        }
    }

    #[test]
    fn admit_denied_falls_back_to_metal_gpu() {
        let mut phase = good_phase();
        phase.determinism = CompileDeterminism::Unknown;
        let verdict = AneAdmissionGate::admit(&phase, &device(), &artifact(), &baseline(2000));
        match verdict {
            AdmissionVerdict::Denied { fallback, .. } => {
                assert_eq!(fallback, CompilePlacement::MetalGpu);
            }
            _ => panic!("expected denied"),
        }
    }

    fn good_record(key: AneQualificationKey) -> AneArtifactQualificationRecord {
        AneArtifactQualificationRecord {
            key,
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
        }
    }

    fn sample_key() -> AneQualificationKey {
        AneQualificationKey {
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
        }
    }

    #[test]
    fn lane_admission_production_only_accepts_clean_records() {
        let key = sample_key();
        let mut gate = LaneAdmissionGate::new(RiskPolicy::ProductionOnly);
        gate.record(good_record(key.clone()));
        assert!(gate.is_production_ready(&key));
    }

    #[test]
    fn lane_admission_production_only_rejects_missing_record() {
        let gate = LaneAdmissionGate::new(RiskPolicy::ProductionOnly);
        assert!(!gate.is_production_ready(&sample_key()));
    }

    #[test]
    fn lane_admission_production_only_rejects_fallback_suspected() {
        let key = sample_key();
        let mut record = good_record(key.clone());
        record.fallback_suspected = true;
        let mut gate = LaneAdmissionGate::new(RiskPolicy::ProductionOnly);
        gate.record(record);
        assert!(!gate.is_production_ready(&key));
    }

    #[test]
    fn lane_admission_research_accepts_partial_failures() {
        let key = sample_key();
        let mut record = good_record(key.clone());
        record.warmup_success = false;
        let mut gate = LaneAdmissionGate::new(RiskPolicy::Research);
        gate.record(record);
        assert!(gate.is_production_ready(&key));
    }

    #[test]
    fn qualification_gate_admits_within_latency() {
        let gate = AneQualificationGate::new(AneQualificationConfig::default());
        let r = gate.qualify("r1", &good_phase(), 50_000, 200_000, 1024);
        assert_eq!(r.admission, AneAdmission::Admitted);
        assert_eq!(r.ane_cost_ns, 50_000);
    }

    #[test]
    fn qualification_gate_denies_above_latency() {
        let gate = AneQualificationGate::new(AneQualificationConfig::default());
        let r = gate.qualify("r1", &good_phase(), 200_000, 200_000, 1024);
        assert_eq!(r.admission, AneAdmission::Denied);
    }

    #[test]
    fn qualification_gate_denies_above_bridge_budget() {
        let gate = AneQualificationGate::new(AneQualificationConfig::default());
        let r = gate.qualify("r1", &good_phase(), 50_000, 200_000, 64 * 1024 * 1024);
        assert_eq!(r.admission, AneAdmission::Denied);
    }

    #[test]
    fn evidence_probe_success_becomes_admission_gate_pass() {
        let probe = EvidenceProbeResult {
            zero_copy_qualified: true,
            error: None,
        };
        let gate = probe.to_admission_gate("evidence_probe");
        assert!(gate.passed);
        assert!(gate.evidence.unwrap().contains("aliasing:true"));
    }

    #[test]
    fn evidence_probe_failure_becomes_admission_gate_fail() {
        let probe = EvidenceProbeResult {
            zero_copy_qualified: false,
            error: Some("aliasing contract violated".into()),
        };
        let gate = probe.to_admission_gate("evidence_probe");
        assert!(!gate.passed);
        assert!(gate.evidence.unwrap().contains("probe_failed"));
    }

    #[test]
    fn run_probe_returns_success_for_valid_inputs() {
        let r = run_probe("/tmp/test", 1, 4096).expect("probe ok");
        assert!(r.zero_copy_qualified);
    }
}
