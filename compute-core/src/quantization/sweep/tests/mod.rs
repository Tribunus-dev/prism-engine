use std::collections::HashMap;

use crate::quantization::contract::{SourceMatrixLayout, WeightValidationReport};
use crate::quantization::sweep::runner::default_validation_config;
use crate::quantization::sweep::spec::SweepScoringConfig;
use crate::quantization::sweep::{
    ByteAccounting, FamilyPolicyEntry, MatrixShape, PackedTileLayout, PerClassPolicy,
    QuantFamilyId, QuantSweepReceipt, SweepCandidateStatus, score_receipt,
};
use crate::quantization::TensorClass;

// ── Test 1: score_receipt higher for lower NRMSE ──────────────────────────

/// Lower weight-space NRMSE must produce a higher score when all other
/// factors (bytes, source shape, family) are identical.
#[test]
fn score_ordering_lower_nrmse_higher_score() {
    let mut max_weight_nrmse_by_family = HashMap::new();
    max_weight_nrmse_by_family.insert("Nf4".to_string(), 0.15);
    let config = SweepScoringConfig {
        max_weight_nrmse_by_family,
        max_zero_collapse: 0.01,
        byte_weight: 0.3,
    };

    let bytes = ByteAccounting {
        code_bytes: 320,
        metadata_bytes: 8,
        residual_bytes: 0,
        routing_bytes: 0,
        total_bytes: 328,
        // 640 × 640 = 409,600 elements → 1,638,400 f32 bytes
        f32_baseline_bytes: 1_638_400,
        compression_ratio_vs_f32: 1_638_400.0 / 328.0,
    };
    let source_shape = vec![640, 640];

    let make_receipt = |nrmse: f64| QuantSweepReceipt {
        receipt_version: 1,
        run_id: "test-run".into(),
        tensor_key: "model.layers.0.self_attn.q_proj.weight".into(),
        tensor_class: TensorClass::DecoderAttentionProjection,
        source_shape: source_shape.clone(),
        family: QuantFamilyId::Nf4,
        parameters: serde_json::json!({}),
        bytes,
        source_layout: SourceMatrixLayout::PrismInByOut,
        logical_shape: MatrixShape {
            in_features: 640,
            out_features: 640,
        },
        packed_layout: PackedTileLayout::OutputChannelContiguousReductionTiles,
        weight: WeightValidationReport {
            nrmse,
            ..Default::default()
        },
        status: SweepCandidateStatus::Passed,
        score: 0.0,
        wall_ms: 100,
    };

    let good = make_receipt(0.05);
    let bad = make_receipt(0.12);

    let score_good = score_receipt(&good, &config);
    let score_bad = score_receipt(&bad, &config);

    assert!(
        score_good > score_bad,
        "lower NRMSE ({}) should score higher than higher NRMSE ({}): got {:.6} vs {:.6}",
        0.05, 0.12, score_good, score_bad
    );
}

// ── Test 2: ByteAccounting compression ratio ──────────────────────────────

/// `compression_ratio_vs_f32` must equal `f32_baseline_bytes / total_bytes`.
#[test]
fn byte_accounting_compression_ratio() {
    let accounting = ByteAccounting::from_payloads(
        &[0u8; 320], // code
        &[0u8; 8],   // metadata
        &[0u8; 0],   // residual
        &[0u8; 0],   // routing
        640,          // elem_count
    );

    let expected_ratio = accounting.f32_baseline_bytes as f64 / accounting.total_bytes as f64;
    assert!(
        (accounting.compression_ratio_vs_f32 - expected_ratio).abs() < f64::EPSILON,
        "compression_ratio_vs_f32 expected {:.6}, got {:.6}",
        expected_ratio,
        accounting.compression_ratio_vs_f32
    );
}

/// Division by zero path: empty payloads return ratio 1.0.
#[test]
fn byte_accounting_zero_total_bytes() {
    let accounting = ByteAccounting::from_payloads(&[], &[], &[], &[], 0);
    assert!(
        (accounting.compression_ratio_vs_f32 - 1.0).abs() < f64::EPSILON,
        "zero-size payloads should yield ratio 1.0, got {}",
        accounting.compression_ratio_vs_f32
    );
}

// ── Test 3: Policy types have expected fields (compile-time layout) ───────

/// `PerClassPolicy` fields must be accessible at the structural level
/// expected by serialization and downstream consumers.
#[test]
fn per_class_policy_fields() {
    let entry = FamilyPolicyEntry {
        family: "Nf4".into(),
        parameters: serde_json::json!({"group_size": 32}),
        weight_nrmse: 0.05,
        score: 0.78,
        total_bytes: 328,
    };

    let policy = PerClassPolicy {
        tensor_class: TensorClass::DecoderAttentionProjection,
        preferred: vec![entry],
        fallback: "Int8".into(),
    };

    // Field existence assertions (compile-time structural check).
    assert_eq!(policy.preferred.len(), 1);
    assert_eq!(&policy.fallback, "Int8");
    assert_eq!(
        policy.tensor_class,
        TensorClass::DecoderAttentionProjection,
        "PerClassPolicy should carry tensor_class"
    );
    assert_eq!(
        policy.preferred[0].weight_nrmse, 0.05,
        "FamilyPolicyEntry should carry weight_nrmse"
    );
    assert_eq!(
        policy.preferred[0].total_bytes, 328,
        "FamilyPolicyEntry should carry total_bytes"
    );
    assert_eq!(
        policy.preferred[0].score, 0.78,
        "FamilyPolicyEntry should carry score"
    );
    assert_eq!(
        &policy.preferred[0].family, "Nf4",
        "FamilyPolicyEntry should carry family"
    );
}

// ── Test 4: SweepValidationConfig default ────────────────────────────────

/// `default_validation_config()` must set `max_candidates_per_tensor` to 200.
#[test]
fn default_validation_config_max_candidates() {
    let config = default_validation_config();
    assert_eq!(
        config.max_candidates_per_tensor, 200,
        "default maximum candidates per tensor should be 200"
    );
}
