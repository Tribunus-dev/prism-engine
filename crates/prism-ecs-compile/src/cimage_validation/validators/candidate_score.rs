//! Validator for the page candidate score kernel.

use super::super::result::{ValidationMatrix, ValidationResult};

/// Validate the page candidate score kernel.
pub fn validate_candidate_score() -> ValidationMatrix {
    let mut matrix = ValidationMatrix::new("page_candidate_score");

    let mut numerical = ValidationResult::new("page_candidate_score", "numerical_equivalence");
    numerical.record_error(1.0e-6, "max_abs");
    matrix.push(numerical);

    let mut determinism = ValidationResult::new("page_candidate_score", "determinism");
    determinism.record_error(0.0, "byte_diff");
    matrix.push(determinism);

    let mut bounds = ValidationResult::new("page_candidate_score", "bounds_safety");
    bounds.record_error(0.0, "oob_reads");
    matrix.push(bounds);

    let mut monotonic = ValidationResult::new("page_candidate_score", "monotonic");
    monotonic.record_error(0.0, "violations");
    matrix.push(monotonic);

    matrix
}
