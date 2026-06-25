//! ANE admission gate — evaluates whether a compile phase is suitable for
//! the Apple Neural Engine.
//!
//! The gate applies five sequential checks:
//!
//! 1. **Numerical contract** — the phase's determinism must be
//!    [`CompileDeterminism::BitExact`] or
//!    [`CompileDeterminism::NumericallyBounded`] (never `Unknown`).
//! 2. **Performance** — ANE estimated runtime must beat the GPU baseline by
//!    at least 15 %.
//! 3. **Memory** — peak memory must stay below a configurable safety watermark.
//! 4. **Bridge** — bytes copied over the ANE bridge must fit within budget.
//! 5. **Fallback bound** — the GPU numerical error must be within bounded-
//!    equivalence tolerance.
//!
//! If every check passes the phase is `Admitted`; otherwise it is `Denied`
//! with an explicit fallback to [`CompilePlacement::MetalGpu`].

use super::phase_ir::{
    CompilePhaseDescriptor, CompilePlacement, CompileDeterminism,
    DeviceSignature, PhaseId,
};
use super::phase_ir::ANEArtifactKey;
use serde::{Deserialize, Serialize};

// ── Performance baseline ──────────────────────────────────────────────────

/// Execution baseline measured on the Metal GPU for a given phase.
///
/// Used by the admission gate to decide whether ANE is worth the switch.
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

// ── Verdict ───────────────────────────────────────────────────────────────

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

// ── Constants ─────────────────────────────────────────────────────────────

/// Peak memory watermark: phases exceeding this many bytes are denied.
const MEMORY_SAFETY_WATERMARK: u64 = 16 * 1024 * 1024 * 1024; // 16 GiB

/// Maximum bridge-copy budget per phase.
const BRIDGE_COPY_BUDGET: u64 = 512 * 1024 * 1024; // 512 MiB

/// ANE must be at most this fraction of GPU execution time to be admitted.
/// Equivalent to "beats GPU by >= 15 %".
const PERF_IMPROVEMENT_PCT: u64 = 85;

/// Maximum relative numerical error tolerated for bounded-equivalent output.
const BOUNDED_EQUIVALENCE_TOLERANCE: f32 = 0.05;

// ── Admission gate ────────────────────────────────────────────────────────

/// Stateless gate that applies ANE admission criteria.
pub struct AneAdmissionGate;

impl AneAdmissionGate {
    /// Evaluate whether `phase` should run on ANE instead of the GPU.
    ///
    /// Returns `Admitted` only when **every** check passes.  On denial the
    /// returned `CompilePlacement` is always `MetalGpu`.
    #[must_use]
    pub fn admit(
        phase: &CompilePhaseDescriptor,
        device: &DeviceSignature,
        artifact: &ANEArtifactKey,
        baseline: &GpuBaseline,
    ) -> AdmissionVerdict {
        // ── 1. Numerical contract ──────────────────────────────────────
        if matches!(phase.determinism, CompileDeterminism::Unknown) {
            return Self::denied(
                phase,
                format!(
                    "determinism class is Unknown; ANE requires BitExact or NumericallyBounded"
                ),
            );
        }

        // ── 2. Performance: ANE must beat GPU by >= 15 % ──────────────
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

        // ── 3. Memory: peak below safety watermark ────────────────────
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

        // ── 4. Bridge budget ──────────────────────────────────────────
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

        // ── 5. Bounded-equivalent output ──────────────────────────────
        if baseline.numerical_error > BOUNDED_EQUIVALENCE_TOLERANCE {
            return Self::denied(
                phase,
                format!(
                    "numerical error {:.4} exceeds bounded-equivalence tolerance {:.4}",
                    baseline.numerical_error, BOUNDED_EQUIVALENCE_TOLERANCE,
                ),
            );
        }

        // All checks passed.
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

    // ── helpers ────────────────────────────────────────────────────────

    fn denied(phase: &CompilePhaseDescriptor, detail: String) -> AdmissionVerdict {
        AdmissionVerdict::Denied {
            reason: format!("phase {} denied: {}", phase.phase_id.0, detail),
            fallback: CompilePlacement::MetalGpu,
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compilation::phase_ir::{
        ANEArtifactKey, CompileDeterminism, CompilePhaseDescriptor, PhaseId,
    };

    fn sample_phase(determinism: CompileDeterminism) -> CompilePhaseDescriptor {
        CompilePhaseDescriptor {
            phase_id: PhaseId(42),
            inputs: vec![],
            outputs: vec![],
            shape_class: crate::compilation::phase_ir::ShapeClass::Static(vec![1, 768]),
            arithmetic_intensity: crate::compilation::phase_ir::ArithmeticIntensity::ComputeBound,
            mutation: crate::compilation::phase_ir::MutationClass::ReadOnly,
            determinism,
            allowed_placements: vec![
                CompilePlacement::MetalGpu,
                CompilePlacement::Ane,
            ],
            minimum_profitable_elements: 0,
            fallback: CompilePlacement::MetalGpu,
            estimated_ane_duration_ns: 1_000_000, // 1 ms
            bridge_copy_bytes: 64 * 1024 * 1024,  // 64 MiB
        }
    }

    fn sample_device() -> DeviceSignature {
        DeviceSignature {
            device_id: "M1-MBP-2021".into(),
            chip: "Apple M1".into(),
            max_memory_bytes: 16 * 1024 * 1024 * 1024,
        }
    }

    fn sample_artifact() -> ANEArtifactKey {
        ANEArtifactKey {
            program_hash: [0xab; 32],
        }
    }

    fn sample_baseline(peak_mem: u64, exec_ns: u64, err: f32) -> GpuBaseline {
        GpuBaseline {
            phase_id: PhaseId(42),
            gpu_total_ns: exec_ns + 50_000,
            gpu_execution_ns: exec_ns,
            peak_memory_bytes: peak_mem,
            numerical_error: err,
        }
    }

    // ── 1. Admitted when every criterion is satisfied ──────────────────

    #[test]
    fn admitted_when_all_criteria_met() {
        let phase = sample_phase(CompileDeterminism::NumericallyBounded {
            abs_error: 0.001,
            rel_error: 0.01,
        });
        let device = sample_device();
        let artifact = sample_artifact();
        // GPU: 2 ms exec -> ANE must beat 1.7 ms; ANE is 1 ms -> passes.
        // Memory: 4 GiB < 16 GiB.
        let baseline = sample_baseline(4 * 1024 * 1024 * 1024, 2_000_000, 0.01);

        let verdict = AneAdmissionGate::admit(&phase, &device, &artifact, &baseline);

        match verdict {
            AdmissionVerdict::Admitted { reason } => {
                assert!(
                    reason.contains("passes all"),
                    "unexpected admitted reason: {reason}"
                );
            }
            AdmissionVerdict::Denied { reason, fallback } => {
                panic!("expected Admitted, got Denied({reason}, {fallback:?})");
            }
        }
    }

    // ── 2. Denied when determinism is Unknown ──────────────────────────

    #[test]
    fn denied_when_numerical_mismatch() {
        let phase = sample_phase(CompileDeterminism::Unknown);
        let baseline = sample_baseline(4 * 1024 * 1024 * 1024, 2_000_000, 0.01);

        let verdict =
            AneAdmissionGate::admit(&phase, &sample_device(), &sample_artifact(), &baseline);

        match verdict {
            AdmissionVerdict::Denied { reason, fallback } => {
                assert!(reason.contains("determinism"), "reason: {reason}");
                assert_eq!(fallback, CompilePlacement::MetalGpu);
            }
            _ => panic!("expected Denied"),
        }
    }

    // ── 3. Denied when ANE slower than GPU threshold ───────────────────

    #[test]
    fn denied_when_performance_regression() {
        let phase = sample_phase(CompileDeterminism::BitExact);
        // GPU exec = 1 ms, ANE est = 1 ms (same -- not >=15% faster).
        // Threshold = 1_000_000 * 85 / 100 = 850_000 ns -- ANE 1_000_000 > 850_000 -> deny.
        let baseline = sample_baseline(4 * 1024 * 1024 * 1024, 1_000_000, 0.01);

        let verdict =
            AneAdmissionGate::admit(&phase, &sample_device(), &sample_artifact(), &baseline);

        match verdict {
            AdmissionVerdict::Denied { reason, fallback } => {
                assert!(reason.contains("fails to beat"), "reason: {reason}");
                assert_eq!(fallback, CompilePlacement::MetalGpu);
            }
            _ => panic!("expected Denied"),
        }
    }

    // ── 4. Denied when peak memory exceeds watermark ───────────────────

    #[test]
    fn denied_when_memory_exceeds_watermark() {
        let phase = sample_phase(CompileDeterminism::BitExact);
        // Peak mem = 20 GiB > 16 GiB watermark.
        let baseline = sample_baseline(20 * 1024 * 1024 * 1024, 2_000_000, 0.01);

        let verdict =
            AneAdmissionGate::admit(&phase, &sample_device(), &sample_artifact(), &baseline);

        match verdict {
            AdmissionVerdict::Denied { reason, fallback } => {
                assert!(reason.contains("watermark"), "reason: {reason}");
                assert_eq!(fallback, CompilePlacement::MetalGpu);
            }
            _ => panic!("expected Denied"),
        }
    }

    // ── 5. Denied when bridge copy exceeds budget ──────────────────────

    #[test]
    fn denied_when_bridge_budget_exceeded() {
        let mut phase = sample_phase(CompileDeterminism::BitExact);
        phase.bridge_copy_bytes = 1024 * 1024 * 1024; // 1 GiB > 512 MiB
        let baseline = sample_baseline(4 * 1024 * 1024 * 1024, 2_000_000, 0.01);

        let verdict =
            AneAdmissionGate::admit(&phase, &sample_device(), &sample_artifact(), &baseline);

        match verdict {
            AdmissionVerdict::Denied { reason, fallback } => {
                assert!(reason.contains("bridge copy"), "reason: {reason}");
                assert_eq!(fallback, CompilePlacement::MetalGpu);
            }
            _ => panic!("expected Denied"),
        }
    }

    // ── 6. Denied when numerical error exceeds tolerance ───────────────

    #[test]
    fn denied_when_numerical_error_exceeds_tolerance() {
        let phase = sample_phase(CompileDeterminism::BitExact);
        let baseline = sample_baseline(4 * 1024 * 1024 * 1024, 2_000_000, 0.10); // 10% > 5%

        let verdict =
            AneAdmissionGate::admit(&phase, &sample_device(), &sample_artifact(), &baseline);

        match verdict {
            AdmissionVerdict::Denied { reason, fallback } => {
                assert!(reason.contains("numerical error"), "reason: {reason}");
                assert_eq!(fallback, CompilePlacement::MetalGpu);
            }
            _ => panic!("expected Denied"),
        }
    }

    // ── 7. Every deny verdict returns MetalGpu fallback ────────────────

    #[test]
    fn fallback_returns_metal_gpu() {
        let deny_cases: Vec<(&str, CompilePhaseDescriptor, GpuBaseline)> = vec![
            // Unknown determinism
            (
                "unknown determinism",
                sample_phase(CompileDeterminism::Unknown),
                sample_baseline(4 * 1024 * 1024 * 1024, 2_000_000, 0.01),
            ),
            // Performance regression
            (
                "perf regression",
                sample_phase(CompileDeterminism::BitExact),
                sample_baseline(4 * 1024 * 1024 * 1024, 500_000, 0.01),
            ),
            // Memory over watermark
            (
                "memory overload",
                sample_phase(CompileDeterminism::BitExact),
                sample_baseline(20 * 1024 * 1024 * 1024, 2_000_000, 0.01),
            ),
            // Bridge budget
            (
                "bridge overload",
                {
                    let mut p = sample_phase(CompileDeterminism::BitExact);
                    p.bridge_copy_bytes = 1024 * 1024 * 1024;
                    p
                },
                sample_baseline(4 * 1024 * 1024 * 1024, 2_000_000, 0.01),
            ),
            // Numerical error
            (
                "numerical drift",
                sample_phase(CompileDeterminism::BitExact),
                sample_baseline(4 * 1024 * 1024 * 1024, 2_000_000, 0.10),
            ),
        ];

        for (label, phase, baseline) in deny_cases {
            let verdict =
                AneAdmissionGate::admit(&phase, &sample_device(), &sample_artifact(), &baseline);
            match &verdict {
                AdmissionVerdict::Denied { fallback, reason } => {
                    assert_eq!(
                        *fallback,
                        CompilePlacement::MetalGpu,
                        "case '{label}': expected MetalGpu fallback, got {fallback:?}; reason: {reason}"
                    );
                }
                _ => panic!("case '{label}': expected Denied, got Admitted"),
            }
        }
    }
}
