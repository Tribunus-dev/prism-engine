//! CPU backend capability registration — Accelerate + Rayon as a fusion candidate.
//!
//! This module defines the CPU-specific capability types and the registration
//! function that advertises CPU capabilities to the compiler pipeline.

use serde::{Deserialize, Serialize};

// ── CpuProgramOp ────────────────────────────────────────────────────────────

/// Operations the CPU (Accelerate + Rayon) fusion backend can execute natively.
///
/// These map to the five supported roles for which the backend has hand-tuned
/// or library-accelerated implementations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CpuProgramOp {
    /// Matrix multiplication (via Accelerate BLAS `cblas_sgemm` or BNNS).
    Matmul,
    /// Root-mean-squared layer normalization.
    RmsNorm,
    /// GELU activation.
    Gelu,
    /// Residual add with optional scaling.
    AddResidual,
    /// Attention score (Q·K^T) computation.
    AttentionScore,
}

// ── CpuBackendCapability ────────────────────────────────────────────────────

/// Self-contained capability descriptor for the CPU (Accelerate + Rayon) backend.
///
/// Mirrors the fields that the fusion scheduler and lowering pipeline need
/// without depending on the still-evolving `execution_plan::backend_capability`
/// types. A bridge function converts this to the registry entry once the
/// dependency types stabilize.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuBackendCapability {
    /// Codec families the CPU backend supports natively.
    pub supported_codecs: Vec<String>,
    /// Supported op names (matches `CpuProgramOp` names).
    pub supported_ops: Vec<String>,
    /// Maximum number of ops that can be fused into one program.
    pub max_ops_per_group: usize,
    /// Maximum dense materialization in bytes (0 = unlimited).
    pub max_dense_bytes: u64,
    /// Codec families that are expressly rejected (require custom kernel).
    pub rejected_codecs: Vec<String>,
}

impl Default for CpuBackendCapability {
    fn default() -> Self {
        Self {
            supported_codecs: vec!["RawF32".into(), "Fp16".into(), "Int8".into()],
            supported_ops: vec![
                "Matmul".into(),
                "RmsNorm".into(),
                "Gelu".into(),
                "AddResidual".into(),
                "AttentionScore".into(),
            ],
            max_ops_per_group: 5,
            max_dense_bytes: 512 * 1024 * 1024, // 512 MB
            rejected_codecs: vec!["Nf4".into(), "Ternary".into()],
        }
    }
}

impl CpuBackendCapability {
    /// Returns `true` iff the named codec is supported (and not rejected).
    pub fn supports_codec(&self, codec: &str) -> bool {
        self.supported_codecs.iter().any(|c| c == codec) && !self.is_rejected(codec)
    }

    /// Returns `true` iff the named op is supported.
    pub fn supports_op(&self, op: &CpuProgramOp) -> bool {
        self.supported_ops.iter().any(|o| o == op.as_str())
    }

    /// Returns `true` iff this codec is in the reject list.
    pub fn is_rejected(&self, codec: &str) -> bool {
        self.rejected_codecs.iter().any(|c| c == codec)
    }

    /// Returns `Ok` if a materialization of `byte_size` is within budget.
    pub fn check_materialization_budget(&self, byte_size: u64) -> Result<(), String> {
        if self.max_dense_bytes > 0 && byte_size > self.max_dense_bytes {
            return Err(format!(
                "materialization of {byte_size} bytes exceeds CPU limit of {}",
                self.max_dense_bytes
            ));
        }
        Ok(())
    }
}

impl CpuProgramOp {
    /// Return the string name of this op variant.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Matmul => "Matmul",
            Self::RmsNorm => "RmsNorm",
            Self::Gelu => "Gelu",
            Self::AddResidual => "AddResidual",
            Self::AttentionScore => "AttentionScore",
        }
    }
}

// ── Registration stub ───────────────────────────────────────────────────────

/// Return the default stable capability descriptor for the Accelerate + Rayon
/// CPU backend. External callers use this to register the CPU backend into
/// whatever capability system is in use.
///
/// Once `execution_plan::backend_capability::BackendCapabilityRegistry`
/// stabilises, this will feed its `register()` method directly.
pub fn accelerate_rayon_capability() -> CpuBackendCapability {
    CpuBackendCapability::default()
}

/// Register the CPU backend capabilities by populating the given capability
/// descriptor. This is the canonical entry point — idempotent, safe to call
/// multiple times.
pub fn register_accelerate_rayon_capabilities(cap: &mut CpuBackendCapability) {
    cap.supported_codecs = vec!["RawF32".into(), "Fp16".into(), "Int8".into()];
    cap.supported_ops = vec![
        "Matmul".into(),
        "RmsNorm".into(),
        "Gelu".into(),
        "AddResidual".into(),
        "AttentionScore".into(),
    ];
    cap.max_ops_per_group = 5;
    cap.max_dense_bytes = 512 * 1024 * 1024;
    cap.rejected_codecs = vec!["Nf4".into(), "Ternary".into()];
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_capability_registered() {
        let mut cap = CpuBackendCapability::default();
        register_accelerate_rayon_capabilities(&mut cap);

        // Codec support
        assert!(cap.supports_codec("RawF32"));
        assert!(cap.supports_codec("Fp16"));
        assert!(cap.supports_codec("Int8"));

        // Rejected codecs
        assert!(!cap.supports_codec("Nf4"));
        assert!(!cap.supports_codec("Ternary"));

        // Op support
        assert!(cap.supports_op(&CpuProgramOp::Matmul));
        assert!(cap.supports_op(&CpuProgramOp::RmsNorm));
        assert!(cap.supports_op(&CpuProgramOp::Gelu));
        assert!(cap.supports_op(&CpuProgramOp::AddResidual));
        assert!(cap.supports_op(&CpuProgramOp::AttentionScore));

        assert_eq!(cap.max_ops_per_group, 5);
        assert_eq!(cap.max_dense_bytes, 512 * 1024 * 1024);
    }

    #[test]
    fn cpu_accepts_rawf32_rmsnorm() {
        let cap = accelerate_rayon_capability();
        assert!(cap.supports_codec("RawF32"), "CPU must accept RawF32");
        assert!(
            cap.supports_op(&CpuProgramOp::RmsNorm),
            "CPU must accept RmsNorm"
        );
        assert!(
            cap.check_materialization_budget(64 * 1024 * 1024).is_ok(),
            "64 MB must be within budget"
        );
    }

    #[test]
    fn cpu_rejects_nf4() {
        let cap = accelerate_rayon_capability();
        assert!(
            !cap.supports_codec("Nf4"),
            "CPU must reject NF4 without custom kernel"
        );
        assert!(
            !cap.supports_codec("Ternary"),
            "CPU must reject Ternary without custom kernel"
        );
    }

    #[test]
    fn cpu_candidate_competes_in_fusion_evaluation() {
        let cap = accelerate_rayon_capability();
        assert_eq!(cap.max_ops_per_group, 5, "CPU supports up to 5 fused ops");
        assert!(cap.supports_op(&CpuProgramOp::Matmul));
        assert!(cap.supports_op(&CpuProgramOp::Gelu));
        assert!(cap.supports_op(&CpuProgramOp::AddResidual));
    }

    #[test]
    fn materialization_limit_enforced() {
        let cap = accelerate_rayon_capability();
        assert!(cap.check_materialization_budget(128 * 1024 * 1024).is_ok());
        assert!(cap
            .check_materialization_budget(1024 * 1024 * 1024)
            .is_err());
    }
}
