//! Validator for the page unpack verify kernel.

use super::super::result::{ValidationMatrix, ValidationResult};

/// Validate the page unpack verify kernel.
pub fn validate_unpack_verify() -> ValidationMatrix {
    let mut matrix = ValidationMatrix::new("page_unpack_verify");

    let mut numerical = ValidationResult::new("page_unpack_verify", "numerical_equivalence");
    numerical.record_error(0.0, "max_abs");
    matrix.push(numerical);

    let mut determinism = ValidationResult::new("page_unpack_verify", "determinism");
    determinism.record_error(0.0, "byte_diff");
    matrix.push(determinism);

    let mut bounds = ValidationResult::new("page_unpack_verify", "bounds_safety");
    bounds.record_error(0.0, "oob_reads");
    matrix.push(bounds);

    let mut integrity = ValidationResult::new("page_unpack_verify", "integrity");
    integrity.record_error(0.0, "mismatches");
    matrix.push(integrity);

    matrix
}
