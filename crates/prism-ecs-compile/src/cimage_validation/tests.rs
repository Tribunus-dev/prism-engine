//! Test module for the CImage validation matrix.
//!
//! The tests exercise the constitutional surface of the validation
//! matrix: result construction, fail / record_error semantics,
//! per-kernel validators, and the flatten entry point.

use super::result::{TestName, ValidationMatrix, ValidationResult};
use super::run::{run_validation_matrix, run_validation_results, DeviceCapability, ValidationDevice};
use super::validators::{
    validate_attention_probe, validate_candidate_score, validate_dense_projection,
    validate_error_partial, validate_mlp_activation_probe, validate_rmsnorm_residual_probe,
    validate_sidecar_apply_verify, validate_ternary_projection, validate_unpack_verify,
};
use super::CImageValidationError;
use super::KernelName;

struct MockDevice;

impl ValidationDevice for MockDevice {
    fn supports(&self, _capability: DeviceCapability) -> bool {
        true
    }
}

#[test]
fn validation_result_construction_starts_passing() {
    let r = ValidationResult::new("kernel", "test");
    assert_eq!(r.kernel_name, KernelName::new("kernel"));
    assert_eq!(r.test_name, TestName::new("test"));
    assert!(r.passed);
    assert_eq!(r.max_abs_error, 0.0);
    assert!(r.details.is_empty());
}

#[test]
fn validation_result_fail_records_error_and_details() {
    let mut r = ValidationResult::new("kernel", "test");
    r.fail(0.1, "first");
    assert!(!r.passed);
    assert_eq!(r.max_abs_error, 0.1);
    assert_eq!(r.details, "first");
    r.fail(0.05, "second");
    // Smaller errors must not lower the recorded max.
    assert_eq!(r.max_abs_error, 0.1);
    assert!(r.details.contains("first"));
    assert!(r.details.contains("second"));
}

#[test]
fn validation_result_record_error_does_not_fail() {
    let mut r = ValidationResult::new("kernel", "test");
    r.record_error(0.01, "max_abs");
    assert!(r.passed);
    assert_eq!(r.max_abs_error, 0.01);
    assert!(r.details.contains("max_abs=1.00e-2"));
}

#[test]
fn validation_matrix_starts_passing_and_empty() {
    let m = ValidationMatrix::new("kernel");
    assert_eq!(m.kernel_name, KernelName::new("kernel"));
    assert!(m.overall_pass);
    assert!(m.is_empty());
    assert_eq!(m.len(), 0);
}

#[test]
fn validation_matrix_push_propagates_fail_verdict() {
    let mut m = ValidationMatrix::new("kernel");
    m.push(ValidationResult::new("kernel", "t1"));
    assert!(m.overall_pass);
    let mut r = ValidationResult::new("kernel", "t2");
    r.fail(0.1, "boom");
    m.push(r);
    assert!(!m.overall_pass);
    assert_eq!(m.len(), 2);
}

#[test]
fn validate_ternary_projection_returns_six_tests() {
    let m = validate_ternary_projection();
    assert_eq!(m.kernel_name, KernelName::new("ternary_projection"));
    assert!(m.overall_pass);
    assert_eq!(m.len(), 6);
}

#[test]
fn validate_dense_projection_returns_five_tests() {
    let m = validate_dense_projection();
    assert_eq!(m.kernel_name, KernelName::new("dense_projection_f16"));
    assert!(m.overall_pass);
    assert_eq!(m.len(), 5);
}

#[test]
fn validate_error_partial_returns_four_tests() {
    let m = validate_error_partial();
    assert_eq!(m.kernel_name, KernelName::new("error_partial"));
    assert_eq!(m.len(), 4);
}

#[test]
fn validate_attention_probe_returns_four_tests() {
    let m = validate_attention_probe();
    assert_eq!(m.kernel_name, KernelName::new("attention_score_probe"));
    assert_eq!(m.len(), 4);
}

#[test]
fn validate_candidate_score_returns_four_tests() {
    let m = validate_candidate_score();
    assert_eq!(m.kernel_name, KernelName::new("page_candidate_score"));
    assert_eq!(m.len(), 4);
}

#[test]
fn validate_unpack_verify_returns_four_tests() {
    let m = validate_unpack_verify();
    assert_eq!(m.kernel_name, KernelName::new("page_unpack_verify"));
    assert_eq!(m.len(), 4);
}

#[test]
fn validate_sidecar_apply_verify_returns_four_tests() {
    let m = validate_sidecar_apply_verify();
    assert_eq!(m.kernel_name, KernelName::new("sidecar_apply_verify"));
    assert_eq!(m.len(), 4);
}

#[test]
fn validate_rmsnorm_residual_probe_returns_four_tests() {
    let m = validate_rmsnorm_residual_probe();
    assert_eq!(m.kernel_name, KernelName::new("rmsnorm_residual_probe"));
    assert_eq!(m.len(), 4);
}

#[test]
fn validate_mlp_activation_probe_returns_four_tests() {
    let m = validate_mlp_activation_probe();
    assert_eq!(m.kernel_name, KernelName::new("mlp_activation_probe"));
    assert_eq!(m.len(), 4);
}

#[test]
fn run_validation_matrix_returns_one_matrix_per_kernel() {
    let matrices = run_validation_matrix(&MockDevice);
    assert_eq!(matrices.len(), 9);
    let names: Vec<String> = matrices.iter().map(|m| m.kernel_name.0.clone()).collect();
    assert!(names.contains(&"ternary_projection".to_string()));
    assert!(names.contains(&"dense_projection_f16".to_string()));
    assert!(names.contains(&"error_partial".to_string()));
    assert!(names.contains(&"attention_score_probe".to_string()));
    assert!(names.contains(&"page_candidate_score".to_string()));
    assert!(names.contains(&"page_unpack_verify".to_string()));
    assert!(names.contains(&"sidecar_apply_verify".to_string()));
    assert!(names.contains(&"rmsnorm_residual_probe".to_string()));
    assert!(names.contains(&"mlp_activation_probe".to_string()));
}

#[test]
fn run_validation_results_flattens_to_per_test() {
    let results = run_validation_results(&MockDevice);
    // 6 + 5 + 4 + 4 + 4 + 4 + 4 + 4 + 4 = 39 tests
    assert_eq!(results.len(), 39);
}

#[test]
fn cimage_validation_error_categories() {
    let r = CImageValidationError::rejected("rejected");
    let f = CImageValidationError::failed("failed");
    assert!(format!("{r}").contains("rejected"));
    assert!(format!("{f}").contains("failed"));
}
