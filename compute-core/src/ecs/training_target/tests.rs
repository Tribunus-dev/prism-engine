//! Integration tests for the training_target module.
//!
//! # Test plan
//!
//! 1. `test_spec_serde_roundtrip` — create, serialize, deserialize, verify fields
//! 2. `test_resolver_generates_ternary_target` — policy with ternary produces target
//! 3. `test_resolver_skips_rawf32` — RawF32 required produces no target
//! 4. `test_feedback_zero_collapse` — evidence with high zero-collapse feedback item
//! 5. `test_feedback_satisfied_target` — evidence meeting all gates → Satisfied
//! 6. `test_training_target_digest_changes_when_gate_changes` — digest == f(gates)
//! 7. `test_export_deterministic` — same spec → same output bytes
//! 8. `test_check_validates_consistency` — invalid spec raises error

use std::collections::HashMap;

use crate::execution_plan::CodecFamily;
use crate::ecs::execution_profile::{
    GroupAxis, MetadataLayout, PhysicalTileLayout, StorageOrder, TileFamily, TileShape,
};
use crate::ecs::training_target::export::{export_spec, spec_digest_from_bytes};
use crate::ecs::training_target::feedback::{
    EvidenceEntry, GateThresholds, TargetWithGates, TrainingFeedbackBuilder,
};
use crate::ecs::training_target::gates::{
    QuantTrainingMethod, RequiredEvidenceLevel, TrainingTargetStatus, WeightTrainingGates,
};
use crate::ecs::training_target::resolve::TrainingTargetResolveOptions;
use crate::ecs::training_target::resolve::TrainingTargetResolver;
use crate::ecs::training_target::spec::{
    TrainingEvidenceGate, TrainingTargetPriority, TrainingTargetSpec, WeightTrainingTarget,
};

/// Helper: construct a minimal PhysicalTileLayout for test use.
fn test_tile_layout() -> PhysicalTileLayout {
    PhysicalTileLayout {
        format: "NF4".into(),
        tile_family: TileFamily::tile640(),
        logical_shape: [4096, 4096],
        storage_order: StorageOrder::RowMajor,
        tile_shape: TileShape::tile640(),
        group_size: 32,
        group_axis: GroupAxis::PackedContiguous,
        metadata_layout: MetadataLayout::AdjacentTile,
        padding_policy: "ZeroPadToTile".into(),
        alignment_bytes: 256,
        interleave: "None".into(),
    }
}

/// Helper: make a simple weight target for tests.
fn make_weight_target(
    target_id: &str,
    codec: CodecFamily,
    priority: TrainingTargetPriority,
) -> WeightTrainingTarget {
    WeightTrainingTarget {
        target_id: target_id.into(),
        tensor_class: "attention.q_proj".into(),
        tensor_key_match: vec!["*".to_string()],
        target_codec: codec,
        physical_layout: test_tile_layout(),
        training_method: QuantTrainingMethod::ShadowWeightsSte,
        gates: WeightTrainingGates {
            max_weight_nrmse: Some(0.05),
            max_zero_collapse_ratio: Some(0.01),
            max_operator_nrmse: Some(0.10),
            min_operator_cosine: Some(0.95),
            max_operator_abs_error: Some(2.0),
            min_byte_savings_ratio: Some(0.5),
            required_evidence_level: RequiredEvidenceLevel::WeightSpace,
        },
        priority,
    }
}

// ── test 1: spec serde roundtrip ───────────────────────────────────────────

#[test]
fn test_spec_serde_roundtrip() {
    let spec = TrainingTargetSpec {
        spec_version: 1,
        model_family: "gemma-2-9b".into(),
        model_digest: Some("abcdef0123456789".into()),
        source_policy_digest: "policy_digest_123".into(),
        target_cimage_profile: "production".into(),
        weight_targets: vec![make_weight_target(
            "wt_gemm_proj",
            CodecFamily::Ternary,
            TrainingTargetPriority::Required,
        )],
        kv_cache_target: None,
        speculative_targets: vec![],
        engram_targets: vec![],
        attention_shape_targets: vec![],
        evidence_gates: vec![TrainingEvidenceGate {
            gate_id: "operator_cosine".into(),
            gate_type: "cosine_similarity".into(),
            threshold: 0.95,
            weight: 1.0,
            required: true,
        }],
    };

    // Serialize to JSON string.
    let json = serde_json::to_string_pretty(&spec).expect("serialize spec");
    assert!(!json.is_empty(), "JSON output must not be empty");
    assert!(json.contains("spec_version"), "JSON should contain spec_version");
    assert!(json.contains("gemma-2-9b"), "JSON should contain model_family");
    assert!(json.contains("Ternary"), "JSON should contain target codec");

    // Deserialize back.
    let decoded: TrainingTargetSpec =
        serde_json::from_str(&json).expect("deserialize spec");

    // Verify top-level fields.
    assert_eq!(decoded.spec_version, 1);
    assert_eq!(decoded.model_family, "gemma-2-9b");
    assert_eq!(decoded.model_digest.as_deref(), Some("abcdef0123456789"));
    assert_eq!(decoded.source_policy_digest, "policy_digest_123");
    assert_eq!(decoded.target_cimage_profile, "production");

    // Verify weight target.
    assert_eq!(decoded.weight_targets.len(), 1);
    let wt = &decoded.weight_targets[0];
    assert_eq!(wt.target_id, "wt_gemm_proj");
    assert_eq!(wt.target_codec, CodecFamily::Ternary);
    assert!(matches!(wt.training_method, QuantTrainingMethod::ShadowWeightsSte));
    assert_eq!(wt.priority, TrainingTargetPriority::Required);

    // Verify evidence gate.
    assert_eq!(decoded.evidence_gates.len(), 1);
    let eg = &decoded.evidence_gates[0];
    assert_eq!(eg.gate_id, "operator_cosine");
    assert_eq!(eg.gate_type, "cosine_similarity");
    assert!((eg.threshold - 0.95).abs() < 1e-9);
    assert!(eg.required);

    // Empty optional fields.
    assert!(decoded.kv_cache_target.is_none());
    assert!(decoded.speculative_targets.is_empty());
    assert!(decoded.engram_targets.is_empty());
    assert!(decoded.attention_shape_targets.is_empty());
}

// ── test 2: resolver generates ternary target ──────────────────────────────

#[test]
fn test_resolver_generates_ternary_target() {
    let policy: serde_json::Value = serde_json::json!({
        "model_family": "gemma-2-9b",
        "entries": [
            {
                "tensor_class": "attention.q_proj",
                "codec": "Ternary",
                "priority": "required",
                "gates": {
                    "max_weight_nrmse": 0.05
                }
            }
        ]
    });

    let options = TrainingTargetResolveOptions::default();
    let resolver = TrainingTargetResolver;
    let specs = resolver
        .resolve(&policy, &options)
        .expect("resolve should succeed");
    let spec = &specs[0];

    assert_eq!(spec.weight_targets.len(), 1, "expected 1 ternary target");
    let wt = &spec.weight_targets[0];
    assert_eq!(wt.target_codec, CodecFamily::Ternary);
    assert_eq!(wt.priority, TrainingTargetPriority::Required);
    assert_eq!(wt.tensor_class, "attention.q_proj");
}

// ── test 3: resolver skips RawF32 ──────────────────────────────────────────

#[test]
fn test_resolver_skips_rawf32() {
    let policy: serde_json::Value = serde_json::json!({
        "model_family": "gemma-2-9b",
        "entries": [
            {
                "tensor_class": "attention.out_proj",
                "codec": "RawF32",
                "priority": "required"
            }
        ]
    });

    let options = TrainingTargetResolveOptions::default();
    let resolver = TrainingTargetResolver;
    let specs = resolver
        .resolve(&policy, &options)
        .expect("resolve should succeed");
    let spec = &specs[0];

    assert!(
        spec.weight_targets.is_empty(),
        "RawF32 target should be skipped, got {}",
        spec.weight_targets.len()
    );
}

// ── test 4: feedback zero collapse ─────────────────────────────────────────

#[test]
fn test_feedback_zero_collapse() {
    // Build a TargetWithGates that has a strict zero-collapse threshold.
    let target = TargetWithGates {
        target_id: "wt_zc".into(),
        tensor_key_match: vec!["model.layers.0.attention.q_proj.weight".into()],
        tensor_class: "attention.q_proj".into(),
        gates: GateThresholds {
            max_zero_collapse_ratio: Some(0.01), // 1% max
            ..GateThresholds::default()
        },
    };

    // Evidence with high zero-collapse (12% > 1% threshold).
    let mut evidence: HashMap<String, Vec<EvidenceEntry>> = HashMap::new();
    evidence.insert(
        "model.layers.0.attention.q_proj.weight".into(),
        vec![EvidenceEntry {
            tensor_key: "model.layers.0.attention.q_proj.weight".into(),
            tensor_class: "attention.q_proj".into(),
            operator: "matmul".into(),
            observed_nrmse: None,
            observed_zero_collapse: Some(0.12),
            observed_cosine: None,
            observed_max_abs: None,
            byte_savings_ratio: None,
        }],
    );

    let report = TrainingFeedbackBuilder::build(
        &[target],
        &evidence,
        "spec_digest_zc",
        "ckpt_abc",
        "ledger_xyz",
    );

    // Should have at least one item with ReduceZeroCollapse.
    let has_zc_item = report.items.iter().any(|item| {
        item.failure_mode == crate::training_target::gates::TrainingFailureMode::ZeroCollapseTooHigh
    });
    assert!(
        has_zc_item,
        "expected ZeroCollapseTooHigh item for high zero-collapse"
    );

    // Status should not be Satisfied.
    assert_ne!(
        report.status,
        TrainingTargetStatus::Satisfied,
        "high zero-collapse should not satisfy target"
    );
}

// ── test 5: feedback satisfied target ───────────────────────────────────────

#[test]
fn test_feedback_satisfied_target() {
    // Target with all gates set.
    let target = TargetWithGates {
        target_id: "wt_sat".into(),
        tensor_key_match: vec!["model.layers.0.attention.q_proj.weight".into()],
        tensor_class: "attention.q_proj".into(),
        gates: GateThresholds {
            max_weight_nrmse: Some(0.05),
            max_zero_collapse_ratio: Some(0.01),
            max_operator_nrmse: Some(0.10),
            min_operator_cosine: Some(0.95),
            max_operator_abs_error: Some(2.0),
            min_byte_savings_ratio: Some(0.5),
        },
    };

    // Evidence well within all thresholds.
    let mut evidence: HashMap<String, Vec<EvidenceEntry>> = HashMap::new();
    evidence.insert(
        "model.layers.0.attention.q_proj.weight".into(),
        vec![EvidenceEntry {
            tensor_key: "model.layers.0.attention.q_proj.weight".into(),
            tensor_class: "attention.q_proj".into(),
            operator: "matmul".into(),
            observed_nrmse: Some(0.01),          // < 0.05 ✓
            observed_zero_collapse: Some(0.001), // < 0.01 ✓
            observed_cosine: Some(0.98),         // > 0.95 ✓
            observed_max_abs: Some(0.5),         // < 2.0 ✓
            byte_savings_ratio: Some(0.6),       // > 0.5 ✓
        }],
    );

    let report = TrainingFeedbackBuilder::build(
        &[target],
        &evidence,
        "spec_digest_sat",
        "ckpt_abc",
        "ledger_xyz",
    );

    // All gates satisfied — no items, status Satisfied.
    assert!(
        report.items.is_empty(),
        "expected no failed items for satisfied target, got {}",
        report.items.len()
    );
    assert_eq!(
        report.status,
        TrainingTargetStatus::Satisfied,
        "status should be Satisfied when all gates pass"
    );
}

// ── test 6: digest changes when gate changes ────────────────────────────────

#[test]
fn test_training_target_digest_changes_when_gate_changes() {
    let make_spec = |threshold: f64| -> TrainingTargetSpec {
        let gates = WeightTrainingGates {
            max_weight_nrmse: Some(threshold),
            max_zero_collapse_ratio: None,
            max_operator_nrmse: None,
            min_operator_cosine: None,
            max_operator_abs_error: None,
            min_byte_savings_ratio: None,
            required_evidence_level: RequiredEvidenceLevel::WeightSpace,
        };
        TrainingTargetSpec {
            spec_version: 1,
            model_family: "test".into(),
            model_digest: None,
            source_policy_digest: "p1".into(),
            target_cimage_profile: "test".into(),
            weight_targets: vec![WeightTrainingTarget {
                target_id: "wt_digest".into(),
                tensor_class: "attention.q_proj".into(),
                tensor_key_match: vec!["*".into()],
                target_codec: CodecFamily::Ternary,
                physical_layout: test_tile_layout(),
                training_method: QuantTrainingMethod::ShadowWeightsSte,
                gates,
                priority: TrainingTargetPriority::Required,
            }],
            kv_cache_target: None,
            speculative_targets: vec![],
            engram_targets: vec![],
            attention_shape_targets: vec![],
            evidence_gates: vec![],
        }
    };

    let spec_a = make_spec(0.05);
    let spec_b = make_spec(0.10);

    let digest_a = spec_a.digest();
    let digest_b = spec_b.digest();

    assert_ne!(
        digest_a, digest_b,
        "digests must differ when gate thresholds differ"
    );
    assert!(!digest_a.is_empty(), "digest must not be empty");
    assert_eq!(digest_a.len(), 64, "BLAKE3 hex digest should be 64 chars");
}

// ── test 7: export deterministic ───────────────────────────────────────────

#[test]
fn test_export_deterministic() {
    let spec = TrainingTargetSpec {
        spec_version: 1,
        model_family: "test".into(),
        model_digest: None,
        source_policy_digest: "p1".into(),
        target_cimage_profile: "test".into(),
        weight_targets: vec![make_weight_target(
            "wt_det",
            CodecFamily::Ternary,
            TrainingTargetPriority::Required,
        )],
        kv_cache_target: None,
        speculative_targets: vec![],
        engram_targets: vec![],
        attention_shape_targets: vec![],
        evidence_gates: vec![],
    };

    let dir = tempfile::tempdir().expect("create temp dir");
    let path_a = dir.path().join("spec_a.json");
    let path_b = dir.path().join("spec_b.json");

    export_spec(&spec, &path_a).expect("first export");
    export_spec(&spec, &path_b).expect("second export");

    let bytes_a = std::fs::read(&path_a).expect("read first export");
    let bytes_b = std::fs::read(&path_b).expect("read second export");

    assert_eq!(bytes_a, bytes_b, "same spec must produce identical bytes");

    let digest_a = spec_digest_from_bytes(&bytes_a);
    let digest_b = spec_digest_from_bytes(&bytes_b);
    assert_eq!(digest_a, digest_b, "same bytes must produce same digest");
}

// ── test 8: check validates consistency ─────────────────────────────────────

#[test]
fn test_check_validates_consistency() {
    // An invalid spec: weight target with negative max_weight_nrmse.
    let invalid_target = WeightTrainingTarget {
        target_id: "wt_bad".into(),
        tensor_class: "attention.q_proj".into(),
        tensor_key_match: vec!["*".into()],
        target_codec: CodecFamily::Ternary,
        physical_layout: test_tile_layout(),
        training_method: QuantTrainingMethod::ShadowWeightsSte,
        gates: WeightTrainingGates {
            max_weight_nrmse: Some(-0.01), // invalid: negative
            max_zero_collapse_ratio: None,
            max_operator_nrmse: None,
            min_operator_cosine: None,
            max_operator_abs_error: None,
            min_byte_savings_ratio: None,
            required_evidence_level: RequiredEvidenceLevel::WeightSpace,
        },
        priority: TrainingTargetPriority::Required,
    };

    let spec = TrainingTargetSpec {
        spec_version: 1,
        model_family: "test".into(),
        model_digest: None,
        source_policy_digest: "p1".into(),
        target_cimage_profile: "test".into(),
        weight_targets: vec![invalid_target],
        kv_cache_target: None,
        speculative_targets: vec![],
        engram_targets: vec![],
        attention_shape_targets: vec![],
        evidence_gates: vec![],
    };

    let result = spec.check_consistency();
    assert!(
        result.is_err(),
        "check_consistency should reject negative NRMSE threshold"
    );

    // A valid spec should pass.
    let valid_spec = TrainingTargetSpec {
        spec_version: 1,
        model_family: "test".into(),
        model_digest: None,
        source_policy_digest: "p1".into(),
        target_cimage_profile: "test".into(),
        weight_targets: vec![WeightTrainingTarget {
            target_id: "wt_good".into(),
            tensor_class: "attention.q_proj".into(),
            tensor_key_match: vec!["*".into()],
            target_codec: CodecFamily::Ternary,
            physical_layout: test_tile_layout(),
            training_method: QuantTrainingMethod::ShadowWeightsSte,
            gates: WeightTrainingGates {
                max_weight_nrmse: Some(0.05),
                max_zero_collapse_ratio: None,
                max_operator_nrmse: None,
                min_operator_cosine: None,
                max_operator_abs_error: None,
                min_byte_savings_ratio: None,
                required_evidence_level: RequiredEvidenceLevel::WeightSpace,
            },
            priority: TrainingTargetPriority::Required,
        }],
        kv_cache_target: None,
        speculative_targets: vec![],
        engram_targets: vec![],
        attention_shape_targets: vec![],
        evidence_gates: vec![],
    };

    assert!(
        valid_spec.check_consistency().is_ok(),
        "valid spec should pass check_consistency"
    );
}
