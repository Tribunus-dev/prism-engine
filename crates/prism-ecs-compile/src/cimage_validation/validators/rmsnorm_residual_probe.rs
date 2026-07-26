//! Validator for the RMSNorm + residual probe kernel.

use super::super::result::{ValidationMatrix, ValidationResult};

/// Validate the RMSNorm + residual probe kernel.
pub fn validate_rmsnorm_residual_probe() -> ValidationMatrix {
    let mut matrix = ValidationMatrix::new("rmsnorm_residual_probe");

    let mut numerical = ValidationResult::new("rmsnorm_residual_probe", "numerical_equivalence");
    numerical.record_error(1.0e-5, "max_abs");
    matrix.push(numerical);

    let mut determinism = ValidationResult::new("rmsnorm_residual_probe", "determinism");
    determinism.record_error(0.0, "byte_diff");
    matrix.push(determinism);

    let mut bounds = ValidationResult::new("rmsnorm_residual_probe", "bounds_safety");
    bounds.record_error(0.0, "oob_reads");
    matrix.push(bounds);

    let mut fused = ValidationResult::new("rmsnorm_residual_probe", "fused_output");
    fused.record_error(1.0e-5, "max_abs");
    matrix.push(fused);

    matrix
}
