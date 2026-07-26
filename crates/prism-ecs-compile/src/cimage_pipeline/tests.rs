//! Test module for the CImage pipeline.
//!
//! The tests exercise the constitutional surface of the pipeline:
//! admission gating, fixture ceiling, receipt emission, and
//! post-emission diagnostics. The tests are ported from the engine
//! `compile/pipeline.rs` test module and follow the same
//! invariant-named convention.

use super::admission::{
    verify_fixture_ceiling, FixtureCeilingError, FixtureCeilingPolicy, FIXTURE_MAX_LAYERS,
    FIXTURE_MAX_SOURCE_BYTES, FIXTURE_MAX_VOCAB,
};
use super::authority::CompilationAuthority;
use super::diagnostics::{run_diagnostics, DiagnosticIssue, DiagnosticSeverity};
use super::receipts::{build_compile_receipt, CompileReceipt, StageProfile, StageTimings};
use super::CImagePipelineError;
use std::collections::BTreeMap;

#[test]
fn compilation_authority_default_is_sealed() {
    assert_eq!(
        CompilationAuthority::default(),
        CompilationAuthority::SealedComputeImage
    );
}

#[test]
fn compilation_authority_str_roundtrip() {
    assert_eq!(CompilationAuthority::TestFixture.as_str(), "TestFixture");
    assert_eq!(
        CompilationAuthority::SealedComputeImage.as_str(),
        "SealedComputeImage"
    );
}

#[test]
fn fixture_ceiling_policy_defaults_match_constants() {
    let policy = FixtureCeilingPolicy::default();
    assert_eq!(policy.max_layers, FIXTURE_MAX_LAYERS);
    assert_eq!(policy.max_vocab, FIXTURE_MAX_VOCAB);
    assert_eq!(policy.max_source_bytes, FIXTURE_MAX_SOURCE_BYTES);
}

#[test]
fn fixture_ceiling_missing_directory_passes() {
    let result = verify_fixture_ceiling("/nonexistent/path/that/should/not/exist");
    assert!(result.is_ok());
}

#[test]
fn stage_profile_total_is_sum_of_stages() {
    let timings = StageTimings {
        source_discovery_ms: 1,
        header_parsing_ms: 2,
        architecture_normalization_ms: 3,
        binding_validation_ms: 4,
        source_hashing_ms: 5,
        layout_planning_ms: 6,
        payload_emission_ms: 7,
        segment_hashing_ms: 8,
        manifest_generation_ms: 9,
        verification_ms: 10,
    };
    assert_eq!(timings.total_ms(), 55);
    let profile = StageProfile::from_timings(&timings);
    assert_eq!(profile.total_ms(), 55);
}

#[test]
fn compile_receipt_construction_collects_hashes() {
    struct TwoShards;
    impl super::receipts::BuildReceiptSource for TwoShards {
        fn shard_hashes(&self) -> Vec<String> {
            vec!["aaaa".to_string(), "bbbb".to_string()]
        }
        fn tokenizer_hashes(&self) -> Vec<String> {
            vec!["tttt".to_string()]
        }
        fn auxiliary_hashes(&self) -> Vec<String> {
            vec![]
        }
        fn namespace(&self) -> String {
            "model".to_string()
        }
    }

    let receipt: CompileReceipt = build_compile_receipt(
        &TwoShards,
        &serde_json::Value::Null,
        123,
        StageProfile::default(),
        BTreeMap::new(),
        Some(4096),
    );
    assert_eq!(receipt.shard_hashes.get("shard_0000").map(String::as_str), Some("aaaa"));
    assert_eq!(receipt.shard_hashes.get("shard_0001").map(String::as_str), Some("bbbb"));
    assert_eq!(receipt.tokenizer_hashes.get("tokenizer_0000").map(String::as_str), Some("tttt"));
    assert_eq!(receipt.namespace, "model");
    assert_eq!(receipt.elapsed_ms, 123);
    assert_eq!(receipt.total_source_bytes, Some(4096));
}

#[test]
fn compile_receipt_uses_btreemap_for_canonical_iteration() {
    struct ThreeShards;
    impl super::receipts::BuildReceiptSource for ThreeShards {
        fn shard_hashes(&self) -> Vec<String> {
            vec!["z".to_string(), "a".to_string(), "m".to_string()]
        }
        fn tokenizer_hashes(&self) -> Vec<String> {
            Vec::new()
        }
        fn auxiliary_hashes(&self) -> Vec<String> {
            Vec::new()
        }
        fn namespace(&self) -> String {
            String::new()
        }
    }

    let receipt = build_compile_receipt(
        &ThreeShards,
        &serde_json::Value::Null,
        0,
        StageProfile::default(),
        BTreeMap::new(),
        None,
    );

    let keys: Vec<&String> = receipt.shard_hashes.keys().collect();
    // BTreeMap iterates in key-sorted order regardless of insertion order.
    assert_eq!(keys, vec!["shard_0000", "shard_0001", "shard_0002"]);
}

#[test]
fn diagnostic_issue_error_severity() {
    let issue = DiagnosticIssue::error("MANIFEST_MISSING", "missing");
    assert_eq!(issue.severity, DiagnosticSeverity::Error);
    assert_eq!(issue.code, "MANIFEST_MISSING");
}

#[test]
fn diagnostic_issue_warning_severity() {
    let issue = DiagnosticIssue::warning("RECEIPT_MISSING", "missing");
    assert_eq!(issue.severity, DiagnosticSeverity::Warning);
}

#[test]
fn run_diagnostics_on_empty_directory_reports_missing_manifest() {
    let tmp = std::env::temp_dir().join("prism-cimage-pipeline-test-empty");
    let _ = std::fs::create_dir_all(&tmp);
    let report = run_diagnostics(&tmp).unwrap();
    assert!(!report.overall_pass);
    assert_eq!(report.issues.len(), 1);
    assert_eq!(report.issues[0].code, "MANIFEST_MISSING");
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn cimage_pipeline_error_categories() {
    let r = CImagePipelineError::rejected("rejected");
    let f = CImagePipelineError::failed("failed");
    let s = CImagePipelineError::stale("stale");
    assert!(format!("{r}").contains("rejected"));
    assert!(format!("{f}").contains("failed"));
    assert!(format!("{s}").contains("stale"));
}

#[test]
fn image_build_attestation_serializes_to_canonical_json() {
    let json = super::authority::image_build_attestation_json();
    assert_eq!(json["event"], "compiler_profile");
    assert!(json.get("profile").is_some());
    assert!(json.get("authorized").is_some());
}

#[test]
fn verify_fixture_ceiling_rejects_oversized_vocab() {
    // Construct a temporary directory with a config.json whose vocab_size
    // exceeds the ceiling. The test only requires the JSON parser to
    // surface the value; the fixture ceiling should reject it.
    let tmp = std::env::temp_dir().join("prism-cimage-pipeline-test-vocab");
    let _ = std::fs::create_dir_all(&tmp);
    let config = serde_json::json!({
        "num_hidden_layers": 2,
        "vocab_size": FIXTURE_MAX_VOCAB + 1,
    });
    std::fs::write(
        tmp.join("config.json"),
        serde_json::to_string_pretty(&config).unwrap(),
    )
    .unwrap();
    let result = verify_fixture_ceiling(tmp.to_str().unwrap());
    assert!(matches!(
        result,
        Err(FixtureCeilingError::VocabTooLarge { .. })
    ));
    let _ = std::fs::remove_dir_all(&tmp);
}
