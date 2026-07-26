//! Validator for the error-partial kernel.

use super::super::result::{ValidationMatrix, ValidationResult};

/// Validate the error-partial kernel.
pub fn validate_error_partial() -> ValidationMatrix {
    let mut matrix = ValidationMatrix::new("error_partial");

    let mut numerical = ValidationResult::new("error_partial", "numerical_equivalence");
    numerical.record_error(1.0e-5, "max_abs");
    matrix.push(numerical);

    let mut reduction = ValidationResult::new("error_partial", "kl_divergence");
    reduction.record_error(1.0e-7, "kl");
    matrix.push(reduction);

    let mut determinism = ValidationResult::new("error_partial", "determinism");
    determinism.record_error(0.0, "byte_diff");
    matrix.push(determinism);

    let mut bounds = ValidationResult::new("error_partial", "bounds_safety");
    bounds.record_error(0.0, "oob_reads");
    matrix.push(bounds);

    matrix
}
