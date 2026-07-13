//! Tests for reasoning evidence types, guardrails, and promotion checks.

use crate::ecs::reasoning_evidence::distillation_guard::{
    DistillationGuardrail, DistillationSignalDecompositionReceipt, EpistemicThresholds,
    PromotionCheck,
};
use crate::ecs::reasoning_evidence::epistemic::{
    EpistemicBehaviorReceipt, EpistemicDegradationFlag, EpistemicMarkerSet,
};
use crate::ecs::reasoning_evidence::receipts::EvidenceReceiptHeader;

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn clean_epistemic_receipt() -> EpistemicBehaviorReceipt {
    EpistemicBehaviorReceipt {
        receipt_id: "r1".into(),
        model_or_partition_id: "model-a".into(),
        evaluation_set_id: "eval-1".into(),
        trace_count: 500,
        uncertainty_marker_rate: 0.15,
        self_correction_marker_rate: 0.08,
        average_reasoning_length: 42.0,
        ood_accuracy: Some(0.92),
        in_domain_accuracy: Some(0.95),
        degradation_flags: vec![],
        evidence_kind: "Measured".into(),
        promotion_eligible: false,
    }
}

fn default_guardrail() -> DistillationGuardrail {
    DistillationGuardrail::default()
}

// ---------------------------------------------------------------------------
// epistemic_marker_receipt_serializes
// ---------------------------------------------------------------------------

#[test]
fn epistemic_marker_receipt_serializes() {
    let receipt = clean_epistemic_receipt();
    let json = serde_json::to_string(&receipt).expect("serialize");
    let deserialized: EpistemicBehaviorReceipt = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(deserialized.receipt_id, "r1");
    assert_eq!(deserialized.model_or_partition_id, "model-a");
    assert_eq!(deserialized.trace_count, 500);
    assert!((deserialized.uncertainty_marker_rate - 0.15).abs() < 1e-9);
    assert_eq!(deserialized.evidence_kind, "Measured");
    assert!(!deserialized.promotion_eligible);
}

// ---------------------------------------------------------------------------
// distillation_guard_rejects_trace_collapse_with_ood_drop
// ---------------------------------------------------------------------------

#[test]
fn distillation_guard_rejects_trace_collapse_with_ood_drop() {
    let guard = default_guardrail();
    let epistemic = EpistemicBehaviorReceipt {
        average_reasoning_length: 2.0, // collapsed
        ood_accuracy: Some(0.60),      // well below 0.85
        uncertainty_marker_rate: 0.15,
        self_correction_marker_rate: 0.08,
        ..clean_epistemic_receipt()
    };

    let result = guard.check_promotion(&epistemic, &None);
    assert!(!result.promotion_eligible);
    assert!(result
        .reasons
        .iter()
        .any(|r| r.contains("Reasoning trace length")));
    assert!(result.reasons.iter().any(|r| r.contains("OOD accuracy")));
}

// ---------------------------------------------------------------------------
// purified_opsd_metadata_marks_residual_pmi_training_as_promotion_eligible
// ---------------------------------------------------------------------------

#[test]
fn purified_opsd_metadata_marks_residual_pmi_training_as_promotion_eligible() {
    let guard = default_guardrail();
    let epistemic = clean_epistemic_receipt(); // all thresholds pass

    let distillation = DistillationSignalDecompositionReceipt {
        teacher_id: "teacher-v3".into(),
        student_id: "student-x".into(),
        dataset_id: "ds-purified".into(),
        has_reference_conditioned_teacher: true,
        has_reference_only_teacher: false,
        uses_inference_transferable_residual: true,
        uses_pmi_target_distribution: true,
        promotion_eligible: false,
    };

    let result = guard.check_promotion(&epistemic, &Some(distillation));
    assert!(result.promotion_eligible);
    assert!(result.reasons.iter().any(|r| r.contains("Purified OPSD")));
}

// ---------------------------------------------------------------------------
// epistemic_marker_set_default_contains_markers
// ---------------------------------------------------------------------------

#[test]
fn epistemic_marker_set_default_contains_markers() {
    let markers = EpistemicMarkerSet::default();
    assert_eq!(markers.marker_set_id, "english_default_v1");
    assert!(!markers.uncertainty_markers.is_empty());
    assert!(!markers.self_correction_markers.is_empty());
    assert!(!markers.reformulation_markers.is_empty());
    assert!(markers.uncertainty_markers.contains(&"maybe".into()));
    assert!(markers.self_correction_markers.contains(&"actually".into()));
    assert!(markers
        .reformulation_markers
        .contains(&"in other words".into()));
}

// ---------------------------------------------------------------------------
// serde roundtrips for all types
// ---------------------------------------------------------------------------

#[test]
fn serde_roundtrip_epistemic_behavior_receipt() {
    let receipt = clean_epistemic_receipt();
    let json = serde_json::to_string(&receipt).unwrap();
    let recovered: EpistemicBehaviorReceipt = serde_json::from_str(&json).unwrap();
    assert_eq!(recovered.receipt_id, receipt.receipt_id);
    assert_eq!(recovered.trace_count, receipt.trace_count);
    assert_eq!(
        recovered.degradation_flags.len(),
        receipt.degradation_flags.len()
    );
}

#[test]
fn serde_roundtrip_epistemic_marker_set() {
    let markers = EpistemicMarkerSet::default();
    let json = serde_json::to_string(&markers).unwrap();
    let recovered: EpistemicMarkerSet = serde_json::from_str(&json).unwrap();
    assert_eq!(recovered.marker_set_id, markers.marker_set_id);
    assert_eq!(recovered.uncertainty_markers.len(), 9);
    assert_eq!(recovered.self_correction_markers.len(), 7);
    assert_eq!(recovered.reformulation_markers.len(), 4);
}

#[test]
fn serde_roundtrip_distillation_signal_decomposition_receipt() {
    let receipt = DistillationSignalDecompositionReceipt {
        teacher_id: "t1".into(),
        student_id: "s1".into(),
        dataset_id: "d1".into(),
        has_reference_conditioned_teacher: true,
        has_reference_only_teacher: false,
        uses_inference_transferable_residual: true,
        uses_pmi_target_distribution: false,
        promotion_eligible: false,
    };
    let json = serde_json::to_string(&receipt).unwrap();
    let recovered: DistillationSignalDecompositionReceipt = serde_json::from_str(&json).unwrap();
    assert_eq!(recovered.teacher_id, "t1");
    assert!(recovered.has_reference_conditioned_teacher);
    assert!(!recovered.uses_pmi_target_distribution);
}

#[test]
fn serde_roundtrip_distillation_guardrail() {
    let guard = default_guardrail();
    let json = serde_json::to_string(&guard).unwrap();
    let recovered: DistillationGuardrail = serde_json::from_str(&json).unwrap();
    assert_eq!(recovered.guard_id, "default_v1");
    assert!((recovered.ood_threshold - 0.85).abs() < 1e-9);
}

#[test]
fn serde_roundtrip_promotion_check() {
    let check = PromotionCheck {
        promotion_eligible: false,
        reasons: vec!["OOD too low".into(), "markers collapsed".into()],
    };
    let json = serde_json::to_string(&check).unwrap();
    let recovered: PromotionCheck = serde_json::from_str(&json).unwrap();
    assert!(!recovered.promotion_eligible);
    assert_eq!(recovered.reasons.len(), 2);
}

#[test]
fn serde_roundtrip_epistemic_thresholds() {
    let thresholds = EpistemicThresholds::default();
    let json = serde_json::to_string(&thresholds).unwrap();
    let recovered: EpistemicThresholds = serde_json::from_str(&json).unwrap();
    assert!((recovered.min_uncertainty_marker_rate - 0.05).abs() < 1e-9);
    assert!((recovered.max_in_domain_gain_for_ood_loss - 0.03).abs() < 1e-9);
}

#[test]
fn serde_roundtrip_evidence_receipt_header() {
    let header = EvidenceReceiptHeader {
        receipt_id: "hdr-001".into(),
        version: 1,
        created_at_unix_ms: 1_700_000_000_000,
    };
    let json = serde_json::to_string(&header).unwrap();
    let recovered: EvidenceReceiptHeader = serde_json::from_str(&json).unwrap();
    assert_eq!(recovered.receipt_id, "hdr-001");
    assert_eq!(recovered.version, 1);
}

#[test]
fn serde_roundtrip_epistemic_degradation_flag() {
    let flags = vec![
        EpistemicDegradationFlag::SuppressedUncertaintyMarkers,
        EpistemicDegradationFlag::ReasoningTraceCollapse,
        EpistemicDegradationFlag::InDomainGainOodLoss,
        EpistemicDegradationFlag::ShortcutStyleIncrease,
        EpistemicDegradationFlag::SelfCorrectionDrop,
    ];
    let json = serde_json::to_string(&flags).unwrap();
    let recovered: Vec<EpistemicDegradationFlag> = serde_json::from_str(&json).unwrap();
    assert_eq!(recovered.len(), 5);
    assert_eq!(
        recovered[0],
        EpistemicDegradationFlag::SuppressedUncertaintyMarkers
    );
    assert_eq!(recovered[4], EpistemicDegradationFlag::SelfCorrectionDrop);
}

// ---------------------------------------------------------------------------
// promotion_check passes for clean profile
// ---------------------------------------------------------------------------

#[test]
fn promotion_check_passes_for_clean_profile() {
    let guard = default_guardrail();
    let epistemic = clean_epistemic_receipt();

    let result = guard.check_promotion(&epistemic, &None);
    assert!(
        result.promotion_eligible,
        "clean profile should pass: {:?}",
        result.reasons
    );
}

#[test]
fn promotion_check_fails_on_low_ood() {
    let guard = default_guardrail();
    let epistemic = EpistemicBehaviorReceipt {
        ood_accuracy: Some(0.70),
        ..clean_epistemic_receipt()
    };

    let result = guard.check_promotion(&epistemic, &None);
    assert!(!result.promotion_eligible);
    assert!(result.reasons.iter().any(|r| r.contains("OOD accuracy")));
}

#[test]
fn promotion_check_fails_on_collapsed_markers() {
    let guard = default_guardrail();
    let epistemic = EpistemicBehaviorReceipt {
        uncertainty_marker_rate: 0.01,
        self_correction_marker_rate: 0.005,
        ..clean_epistemic_receipt()
    };

    let result = guard.check_promotion(&epistemic, &None);
    assert!(!result.promotion_eligible);
    assert!(result
        .reasons
        .iter()
        .any(|r| r.contains("Uncertainty marker rate")));
    assert!(result
        .reasons
        .iter()
        .any(|r| r.contains("Self-correction marker rate")));
}

#[test]
fn promotion_check_fails_on_indomain_gain_with_ood_loss() {
    let guard = default_guardrail();
    let epistemic = EpistemicBehaviorReceipt {
        ood_accuracy: Some(0.75),
        in_domain_accuracy: Some(0.95),
        ..clean_epistemic_receipt()
    };

    let result = guard.check_promotion(&epistemic, &None);
    assert!(!result.promotion_eligible);
    assert!(result.reasons.iter().any(|r| r.contains("In-domain gain")));
}

#[test]
fn purified_opsd_still_rejected_when_ood_below_threshold() {
    let guard = default_guardrail();
    let epistemic = EpistemicBehaviorReceipt {
        ood_accuracy: Some(0.60),
        ..clean_epistemic_receipt()
    };

    let distillation = DistillationSignalDecompositionReceipt {
        teacher_id: "t".into(),
        student_id: "s".into(),
        dataset_id: "d".into(),
        has_reference_conditioned_teacher: true,
        has_reference_only_teacher: false,
        uses_inference_transferable_residual: true,
        uses_pmi_target_distribution: true,
        promotion_eligible: false,
    };

    let result = guard.check_promotion(&epistemic, &Some(distillation));
    assert!(!result.promotion_eligible, "OOD drop overrides OPSD flag");
}

#[test]
fn epistemic_behavior_receipt_constructor() {
    let header = EvidenceReceiptHeader {
        receipt_id: "hdr-abc".into(),
        version: 1,
        created_at_unix_ms: 42,
    };
    let receipt = EpistemicBehaviorReceipt::new(
        &header,
        "model-1",
        "eval-42",
        100,
        0.12,
        0.06,
        30.0,
        "Synthetic",
    );
    assert_eq!(receipt.receipt_id, "hdr-abc");
    assert_eq!(receipt.model_or_partition_id, "model-1");
    assert_eq!(receipt.trace_count, 100);
    assert!(receipt.ood_accuracy.is_none());
    assert!(receipt.degradation_flags.is_empty());
    assert!(!receipt.promotion_eligible);
}

#[test]
fn distillation_signal_decomposition_is_purified_opsd() {
    let r1 = DistillationSignalDecompositionReceipt {
        teacher_id: "t".into(),
        student_id: "s".into(),
        dataset_id: "d".into(),
        has_reference_conditioned_teacher: true,
        has_reference_only_teacher: false,
        uses_inference_transferable_residual: true,
        uses_pmi_target_distribution: true,
        promotion_eligible: false,
    };
    assert!(r1.is_purified_opsd());

    let r2 = DistillationSignalDecompositionReceipt {
        has_reference_conditioned_teacher: false,
        ..r1.clone()
    };
    assert!(!r2.is_purified_opsd());

    let r3 = DistillationSignalDecompositionReceipt {
        uses_pmi_target_distribution: false,
        ..r1.clone()
    };
    assert!(!r3.is_purified_opsd());
}

#[test]
fn degradation_flags_recorded_in_promotion_check() {
    let guard = default_guardrail();
    let epistemic = EpistemicBehaviorReceipt {
        degradation_flags: vec![
            EpistemicDegradationFlag::SuppressedUncertaintyMarkers,
            EpistemicDegradationFlag::SelfCorrectionDrop,
        ],
        // Still pass thresholds
        uncertainty_marker_rate: 0.15,
        self_correction_marker_rate: 0.08,
        ood_accuracy: Some(0.92),
        ..clean_epistemic_receipt()
    };

    let result = guard.check_promotion(&epistemic, &None);
    // Degradation flags are recorded but don't alone cause rejection
    // (the promotion rules check the actual rates/values, not just flags)
    assert!(result
        .reasons
        .iter()
        .any(|r| r.contains("SuppressedUncertaintyMarkers")));
    assert!(result
        .reasons
        .iter()
        .any(|r| r.contains("SelfCorrectionDrop")));
}
