//! PRISM-ANE-EVIDENCE-AND-ADMISSION-0001: ANE admission gate.
//!
//! Qualification gate and admission policy for ANE artifact deployment.
//! The gate maintains a database of qualification records keyed by
//! [`AneQualificationKey`] and enforces a configurable [`RiskPolicy`]
//! before allowing an artifact onto the ANE lane.
//!
//! Admission rules (applied in order):
//!
//! 1. **Catalogue presence** — no qualification record for the key →
//!    [`AneRejectionReason::CoreMlCompilationFailure`].
//! 2. **Compilation success** — `compile_success` must be `true`.
//! 3. **Warmup success** — `warmup_success` must be `true` (bypassed
//!    by [`RiskPolicy::ExperimentalAllowed`]).
//! 4. **Numerical parity** — `numerical_parity.passed` must be `true`.
//! 5. **Fallback suspicion** — `fallback_suspected = true` AND
//!    [`RiskPolicy::ProductionOnly`] → rejected.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use crate::compilation::activation_abi::ActivationAbi;
use crate::compilation::ane_eligibility::ShapeBucket;
use crate::compilation::ane_eligibility::ShapeBucketFamily;
use crate::compilation::tri_lane::AneRejectionReason;

// ── Identity types ────────────────────────────────────────────────────────

/// Uniquely identifies a hardware configuration for qualification.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HardwareIdentifier {
    /// Apple SoC family (e.g. `"M1"`, `"M2"`, `"M3"`).
    pub soc_family: String,
    /// Model identifier (e.g. `"Mac14,2"`, `"Mac15,3"`).
    pub model_identifier: String,
}

/// macOS or iOS build identity.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OsBuild {
    /// Semantic version (e.g. `"14.5"`).
    pub version: String,
    /// Build number (e.g. `"23F79"`).
    pub build_number: String,
}

/// Core ML runtime version.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CoreMlRuntimeVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

/// Identifies a compiled ANE artifact version.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ArtifactKey {
    /// Model family name (e.g. `"qwen3-tts"`, `"flux-klein"`).
    pub model_family: String,
    /// Packet kind (e.g. `"cross_attention"`, `"ffn"`).
    pub packet_kind: String,
    /// First layer index in this artifact.
    pub layer_start: u32,
    /// Last layer index in this artifact (inclusive).
    pub layer_end: u32,
    /// Function name within the Core ML model.
    pub function_name: String,
    /// Index into the shape bucket table.
    pub shape_bucket: u32,
    /// Numerical precision (e.g. `"fp16"`).
    pub precision: String,
}

// ── Qualification key ────────────────────────────────────────────────────

/// Composite key for a qualification record.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AneQualificationKey {
    /// Compile-time artifact descriptor.
    pub artifact_key: ArtifactKey,
    /// Hardware configuration this key targets.
    pub hardware_identifier: HardwareIdentifier,
    /// OS build this key was qualified on.
    pub os_build: OsBuild,
    /// Core ML runtime version at qualification time.
    pub coreml_runtime: CoreMlRuntimeVersion,
}

// ── Numerical parity ────────────────────────────────────────────────────

/// Numerical comparison between ANE output and reference (GPU/CPU) output.
///
/// `passed` is `true` only when both `max_absolute_error` and
/// `max_relative_error` are within the tolerance thresholds for the
/// target precision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NumericalParityResult {
    /// Maximum absolute error across all elements.
    pub max_absolute_error: f64,
    /// Maximum relative error across all elements.
    pub max_relative_error: f64,
    /// Total number of elements compared.
    pub element_count: u64,
    /// Number of elements exceeding the error threshold.
    pub mismatched_count: u64,
    /// Whether the numerical parity check passed.
    pub passed: bool,
}

// ── Qualification record ─────────────────────────────────────────────────

/// Full qualification record for a single ANE artifact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AneArtifactQualificationRecord {
    /// Composite key identifying the artifact and environment.
    pub key: AneQualificationKey,
    /// Core ML model compilation succeeded.
    pub compile_success: bool,
    /// Core ML model loaded and bound successfully.
    pub load_success: bool,
    /// Warmup predictions completed without error.
    pub warmup_success: bool,
    /// Output tensors were present after prediction.
    pub output_present: bool,
    /// Numerical comparison against reference output.
    pub numerical_parity: NumericalParityResult,
    /// Time-to-first-prediction latency in microseconds.
    pub first_prediction_latency_us: u64,
    /// Steady-state per-prediction latency in microseconds.
    pub steady_state_latency_us: u64,
    /// Boundary materialisation latency in microseconds.
    pub boundary_latency_us: u64,
    /// Peak memory footprint in bytes.
    pub memory_footprint_bytes: u64,
    /// Whether the runtime suspects ANE fell back to CPU/GPU.
    pub fallback_suspected: bool,
    /// Human-readable reason if any step failed.
    pub failure_reason: Option<String>,
    /// ISO 8601 timestamp of qualification.
    pub qualification_timestamp: String,
}

// ── Risk policy ──────────────────────────────────────────────────────────

/// Deployment risk policy for the admission gate.
///
/// Each level relaxes or enforces specific checks to allow staged
/// rollout from experimental through benchmark to production.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RiskPolicy {
    /// Full qualification required — no fallback suspicion tolerated.
    ProductionOnly,
    /// Fallback suspicion is tolerated.
    BenchmarkAllowed,
    /// Warmup, parity, and fallback checks bypassed.
    ExperimentalAllowed,
}

// ── Admission gate ───────────────────────────────────────────────────────

/// Admission gate that enforces qualification policy before ANE deployment.
///
/// Maintains an in-memory database of qualification records and applies
/// the configured [`RiskPolicy`] to each admission request.
pub struct LaneAdmissionGate {
    /// Qualification database keyed by artifact + environment.
    pub ane_qualification_db: HashMap<AneQualificationKey, AneArtifactQualificationRecord>,
    /// Active risk policy for this gate instance.
    pub risk_policy: RiskPolicy,
}

impl LaneAdmissionGate {
    /// Create a new admission gate with the given risk policy.
    pub fn new(risk_policy: RiskPolicy) -> Self {
        Self {
            ane_qualification_db: HashMap::new(),
            risk_policy,
        }
    }

    /// Evaluate whether the artifact identified by `key` is admitted to
    /// the ANE lane.
    ///
    /// Admission applies the configured [`RiskPolicy`] to the matching
    /// qualification record:
    ///
    /// | Check | ProductionOnly | BenchmarkAllowed | ExperimentalAllowed |
    /// |---|---|---|---|
    /// | Record exists | yes | yes | yes |
    /// | Compile succeeded | yes | yes | yes |
    /// | Load succeeded | yes | yes | yes |
    /// | Warmup succeeded | yes | yes | **bypassed** |
    /// | Numerical parity | yes | yes | **bypassed** |
    /// | No fallback suspicion | yes | **bypassed** | **bypassed** |
    pub fn admit(
        &self,
        key: &AneQualificationKey,
        _abi: &ActivationAbi,
        _bucket: &ShapeBucket,
    ) -> Result<(), AneRejectionReason> {
        let record = self
            .ane_qualification_db
            .get(key)
            .ok_or_else(|| {
                AneRejectionReason::CoreMlCompilationFailure(
                    "no qualification record in database".into(),
                )
            })?;

        // Compilation success — enforced at every policy level.
        if !record.compile_success {
            return Err(AneRejectionReason::CoreMlCompilationFailure(
                record
                    .failure_reason
                    .clone()
                    .unwrap_or_else(|| "compile_success is false".into()),
            ));
        }

        // Load success — enforced at every policy level.
        if !record.load_success {
            return Err(AneRejectionReason::RuntimeLoadFailure(
                record
                    .failure_reason
                    .clone()
                    .unwrap_or_else(|| "load_success is false".into()),
            ));
        }

        // ExperimentalAllowed bypasses warmup, parity, and fallback checks.
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

        // ProductionOnly also rejects suspected fallback.
        if self.risk_policy == RiskPolicy::ProductionOnly && record.fallback_suspected {
            return Err(AneRejectionReason::GpuContentionRisk);
        }

        Ok(())
    }

    /// Insert or update a qualification record in the database.
    ///
    /// The record is stored keyed by its own `key` field, replacing any
    /// prior record for the same key.
    pub fn record(&mut self, record: AneArtifactQualificationRecord) {
        self.ane_qualification_db.insert(record.key.clone(), record);
    }

    /// Check whether a fully production-qualified record exists for `key`.
    ///
    /// A record is production-ready when all of:
    /// - Record exists in the database.
    /// - `compile_success`, `load_success`, and `warmup_success` are all
    ///   `true`.
    /// - `numerical_parity.passed` is `true`.
    /// - `fallback_suspected` is `false`.
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

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compilation::activation_abi::DecodeActivationV1Params;
    use crate::compilation::activation_abi::PhysicalLayout;
    use crate::compilation::phase_ir::TensorDtype;
    use serde_json;

    fn sample_key() -> AneQualificationKey {
        AneQualificationKey {
            artifact_key: ArtifactKey {
                model_family: "qwen3-tts".into(),
                packet_kind: "cross_attention".into(),
                layer_start: 0,
                layer_end: 5,
                function_name: "ane_cross_attn".into(),
                shape_bucket: 2,
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
            coreml_runtime: CoreMlRuntimeVersion {
                major: 8,
                minor: 0,
                patch: 0,
            },
        }
    }

    fn qualified_record(key: AneQualificationKey) -> AneArtifactQualificationRecord {
        AneArtifactQualificationRecord {
            key,
            compile_success: true,
            load_success: true,
            warmup_success: true,
            output_present: true,
            numerical_parity: NumericalParityResult {
                max_absolute_error: 1e-4,
                max_relative_error: 1e-3,
                element_count: 65536,
                mismatched_count: 0,
                passed: true,
            },
            first_prediction_latency_us: 1500,
            steady_state_latency_us: 320,
            boundary_latency_us: 85,
            memory_footprint_bytes: 16_384_000,
            fallback_suspected: false,
            failure_reason: None,
            qualification_timestamp: "2026-06-25T12:00:00Z".into(),
        }
    }

    fn sample_abi() -> ActivationAbi {
        ActivationAbi::DecodeActivationV1(DecodeActivationV1Params {
            dtype: TensorDtype::Float16,
            seq_bucket: 2,
            hidden_dim: 4096,
            physical_layout: PhysicalLayout::ContiguousRowMajor,
            alignment: 64,
            stride_constraint: None,
        })
    }

    // ── Admission tests ──────────────────────────────────────────────────

    #[test]
    fn test_admit_qualified_production_record() {
        let key = sample_key();
        let record = qualified_record(key.clone());
        let mut gate = LaneAdmissionGate::new(RiskPolicy::ProductionOnly);
        gate.record(record);

        let abi = sample_abi();
        // ShapeBucket is a forward reference to ane_eligibility module
        let bucket = ShapeBucket { batch: 1, sequence: 128, hidden: 4096, rank: 1, family: ShapeBucketFamily::Decode };

        assert!(gate.admit(&key, &abi, &bucket).is_ok());
    }

    #[test]
    fn test_reject_missing_record() {
        let gate = LaneAdmissionGate::new(RiskPolicy::ProductionOnly);
        let key = sample_key();
        let abi = sample_abi();
        let bucket = ShapeBucket { batch: 1, sequence: 128, hidden: 4096, rank: 1, family: ShapeBucketFamily::Decode };

        let result = gate.admit(&key, &abi, &bucket);
        assert!(result.is_err());
        match result.unwrap_err() {
            AneRejectionReason::CoreMlCompilationFailure(msg) => {
                assert!(msg.contains("no qualification record"));
            }
            other => panic!("expected CoreMlCompilationFailure, got {other:?}"),
        }
    }

    #[test]
    fn test_reject_failed_compile() {
        let key = sample_key();
        let mut record = qualified_record(key.clone());
        record.compile_success = false;
        record.failure_reason = Some("MIL operator not supported".into());
        let mut gate = LaneAdmissionGate::new(RiskPolicy::ProductionOnly);
        gate.record(record);
        let abi = sample_abi();
        let bucket = ShapeBucket { batch: 1, sequence: 128, hidden: 4096, rank: 1, family: ShapeBucketFamily::Decode };

        let result = gate.admit(&key, &abi, &bucket);
        assert!(result.is_err());
        match result.unwrap_err() {
            AneRejectionReason::CoreMlCompilationFailure(msg) => {
                assert!(msg.contains("MIL operator"), "{msg}");
            }
            other => panic!("expected CoreMlCompilationFailure, got {other:?}"),
        }
    }

    #[test]
    fn test_reject_failed_parity() {
        let key = sample_key();
        let mut record = qualified_record(key.clone());
        record.numerical_parity.passed = false;
        record.numerical_parity.max_absolute_error = 0.25;
        let mut gate = LaneAdmissionGate::new(RiskPolicy::ProductionOnly);
        gate.record(record);
        let abi = sample_abi();
        let bucket = ShapeBucket { batch: 1, sequence: 128, hidden: 4096, rank: 1, family: ShapeBucketFamily::Decode };

        let result = gate.admit(&key, &abi, &bucket);
        assert!(result.is_err());
        match result.unwrap_err() {
            AneRejectionReason::NumericalDivergence(err) => {
                assert!((err - 0.25).abs() < 1e-12);
            }
            other => panic!("expected NumericalDivergence, got {other:?}"),
        }
    }

    #[test]
    fn test_benchmark_allows_suspected_fallback() {
        let key = sample_key();
        let mut record = qualified_record(key.clone());
        record.fallback_suspected = true;
        let mut gate = LaneAdmissionGate::new(RiskPolicy::BenchmarkAllowed);
        gate.record(record);
        let abi = sample_abi();
        let bucket = ShapeBucket { batch: 1, sequence: 128, hidden: 4096, rank: 1, family: ShapeBucketFamily::Decode };

        // BenchmarkAllowed tolerates fallback suspicion.
        assert!(gate.admit(&key, &abi, &bucket).is_ok());
    }

    #[test]
    fn test_experimental_bypasses_warmup() {
        let key = sample_key();
        let mut record = qualified_record(key.clone());
        record.warmup_success = false;   // would fail ProductionOnly
        record.numerical_parity.passed = false; // would also fail
        record.fallback_suspected = true; // would fail ProductionOnly
        let mut gate = LaneAdmissionGate::new(RiskPolicy::ExperimentalAllowed);
        gate.record(record);
        let abi = sample_abi();
        let bucket = ShapeBucket { batch: 1, sequence: 128, hidden: 4096, rank: 1, family: ShapeBucketFamily::Decode };

        // ExperimentalAllowed bypasses warmup, parity, and fallback checks.
        assert!(gate.admit(&key, &abi, &bucket).is_ok());
    }

    // ── Serde round-trip ─────────────────────────────────────────────────

    #[test]
    fn test_serde_roundtrip_all_types() {
        let key = sample_key();
        let record = qualified_record(key);

        // AneQualificationKey round-trip
        let key_json = serde_json::to_string(&record.key).expect("serialize key");
        let key_back: AneQualificationKey =
            serde_json::from_str(&key_json).expect("deserialize key");
        assert_eq!(record.key, key_back);

        // NumericalParityResult round-trip
        let parity_json =
            serde_json::to_string(&record.numerical_parity).expect("serialize parity");
        let parity_back: NumericalParityResult =
            serde_json::from_str(&parity_json).expect("deserialize parity");
        assert_eq!(record.numerical_parity, parity_back);

        // AneArtifactQualificationRecord round-trip
        let rec_json =
            serde_json::to_string(&record).expect("serialize record");
        let rec_back: AneArtifactQualificationRecord =
            serde_json::from_str(&rec_json).expect("deserialize record");
        assert_eq!(record, rec_back);

        // RiskPolicy round-trip
        for policy in &[
            RiskPolicy::ProductionOnly,
            RiskPolicy::BenchmarkAllowed,
            RiskPolicy::ExperimentalAllowed,
        ] {
            let pol_json = serde_json::to_string(policy).expect("serialize policy");
            let pol_back: RiskPolicy =
                serde_json::from_str(&pol_json).expect("deserialize policy");
            assert_eq!(*policy, pol_back);
        }

        // HardwareIdentifier round-trip
        let hw_json =
            serde_json::to_string(&record.key.hardware_identifier).expect("serialize hw");
        let hw_back: HardwareIdentifier =
            serde_json::from_str(&hw_json).expect("deserialize hw");
        assert_eq!(record.key.hardware_identifier, hw_back);

        // OsBuild round-trip
        let os_json = serde_json::to_string(&record.key.os_build).expect("serialize os");
        let os_back: OsBuild = serde_json::from_str(&os_json).expect("deserialize os");
        assert_eq!(record.key.os_build, os_back);

        // CoreMlRuntimeVersion round-trip
        let rt_json =
            serde_json::to_string(&record.key.coreml_runtime).expect("serialize runtime");
        let rt_back: CoreMlRuntimeVersion =
            serde_json::from_str(&rt_json).expect("deserialize runtime");
        assert_eq!(record.key.coreml_runtime, rt_back);

        // ArtifactKey round-trip
        let ak_json =
            serde_json::to_string(&record.key.artifact_key).expect("serialize artifact key");
        let ak_back: ArtifactKey =
            serde_json::from_str(&ak_json).expect("deserialize artifact key");
        assert_eq!(record.key.artifact_key, ak_back);
    }

    // ── Production readiness ─────────────────────────────────────────────

    #[test]
    fn test_is_production_ready() {
        let key = sample_key();
        let mut gate = LaneAdmissionGate::new(RiskPolicy::ProductionOnly);

        // No record yet.
        assert!(!gate.is_production_ready(&key));

        // Fully qualified.
        let record = qualified_record(key.clone());
        gate.record(record);
        assert!(gate.is_production_ready(&key));

        // Corrupt each required flag.
        let checks: Vec<(&str, Box<dyn FnMut(&mut AneArtifactQualificationRecord)>)> = vec![
            ("compile_success", Box::new(|r| r.compile_success = false)),
            ("load_success", Box::new(|r| r.load_success = false)),
            ("warmup_success", Box::new(|r| r.warmup_success = false)),
            ("parity_passed", Box::new(|r| r.numerical_parity.passed = false)),
            ("fallback_suspected", Box::new(|r| r.fallback_suspected = true)),
        ];
        for (field, mut check) in checks {
            let mut bad = qualified_record(key.clone());
            check(&mut bad);
            let mut gate = LaneAdmissionGate::new(RiskPolicy::ProductionOnly);
            gate.record(bad);
            assert!(!gate.is_production_ready(&key), "{field} should make record not production-ready");
        }
    }
}
