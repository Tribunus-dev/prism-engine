//! Evaluation receipts — the immutable evidence emitted by one
//! evaluation run.
//!
//! This module owns the canonical authority for the structured
//! evidence a backend evaluation emits. Receipts are content-bearing
//! and survive across processes; an admission gate consumes the
//! bundle to decide whether to admit a candidate. Each receipt type
//! is independently optional — a backend may skip stages (e.g. an
//! oracle role does not need a performance receipt) — but the
//! bundle as a whole is the durable evidence.

use serde::{Deserialize, Serialize};

/// Complete evidence bundle from one evaluation run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct EvaluationReceiptBundle {
    /// Static validation receipt.
    pub static_validation: Option<StaticValidationReceipt>,
    /// Compilation receipt.
    pub compilation: Option<CompilationReceipt>,
    /// Codec correctness receipt.
    pub codec: Option<CodecReceipt>,
    /// Numerical comparison receipt.
    pub numerical: Option<NumericalReceipt>,
    /// Performance measurement receipt.
    pub performance: Option<PerformanceReceipt>,
    /// Repeatability receipt.
    pub repeatability: Option<RepeatabilityReceipt>,
    /// Provenance chain.
    pub provenance: Option<ProvenanceReceipt>,
    /// Rejection evidence (if evaluation failed).
    pub rejection: Option<RejectionReceipt>,
}

/// Static validation receipt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StaticValidationReceipt {
    pub passed: bool,
    pub messages: Vec<String>,
}

/// Compilation receipt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompilationReceipt {
    pub compile_duration_ms: u64,
    pub artifact_digest: [u8; 32],
    pub success: bool,
}

/// Codec correctness receipt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CodecReceipt {
    pub packed_digest: [u8; 32],
    pub decode_correct: bool,
    pub dimensions_match: bool,
}

/// Numerical comparison receipt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NumericalReceipt {
    pub max_abs_error: f64,
    pub mean_abs_error: f64,
    pub passed: bool,
}

/// Performance measurement receipt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerformanceReceipt {
    pub wall_time_ns: u64,
    pub sample_count: usize,
    pub median_ns: u64,
}

/// Repeatability receipt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RepeatabilityReceipt {
    pub std_dev_pct: f64,
    pub passed: bool,
}

/// Provenance receipt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProvenanceReceipt {
    pub candidate_id: String,
    pub machine_id: String,
    pub timestamp: String,
}

/// Rejection receipt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RejectionReceipt {
    pub stage: String,
    pub reason: String,
    pub detail: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_bundle() -> EvaluationReceiptBundle {
        EvaluationReceiptBundle {
            static_validation: Some(StaticValidationReceipt {
                passed: true,
                messages: vec!["ok".to_string()],
            }),
            compilation: Some(CompilationReceipt {
                compile_duration_ms: 100,
                artifact_digest: [1u8; 32],
                success: true,
            }),
            codec: Some(CodecReceipt {
                packed_digest: [2u8; 32],
                decode_correct: true,
                dimensions_match: true,
            }),
            numerical: Some(NumericalReceipt {
                max_abs_error: 0.001,
                mean_abs_error: 0.0001,
                passed: true,
            }),
            performance: Some(PerformanceReceipt {
                wall_time_ns: 1_000_000,
                sample_count: 10,
                median_ns: 100_000,
            }),
            repeatability: Some(RepeatabilityReceipt {
                std_dev_pct: 0.5,
                passed: true,
            }),
            provenance: Some(ProvenanceReceipt {
                candidate_id: "cand-1".to_string(),
                machine_id: "machine-a".to_string(),
                timestamp: "2026-07-27T00:00:00Z".to_string(),
            }),
            rejection: None,
        }
    }

    #[test]
    fn all_receipt_types_serialize_round_trip() {
        let bundle = sample_bundle();
        let json = serde_json::to_string(&bundle).expect("serialize");
        let restored: EvaluationReceiptBundle = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored, bundle);
    }

    #[test]
    fn default_bundle_is_empty() {
        let bundle = EvaluationReceiptBundle::default();
        assert!(bundle.static_validation.is_none());
        assert!(bundle.compilation.is_none());
        assert!(bundle.codec.is_none());
        assert!(bundle.numerical.is_none());
        assert!(bundle.performance.is_none());
        assert!(bundle.repeatability.is_none());
        assert!(bundle.provenance.is_none());
        assert!(bundle.rejection.is_none());
    }

    #[test]
    fn rejection_receipt_carries_stage_reason_detail() {
        let r = RejectionReceipt {
            stage: "numerical".to_string(),
            reason: "max error exceeded".to_string(),
            detail: "0.05 > 0.02".to_string(),
        };
        assert_eq!(r.stage, "numerical");
        assert_eq!(r.reason, "max error exceeded");
        assert_eq!(r.detail, "0.05 > 0.02");
    }
}
