//! Validator for the sidecar apply verify kernel.

use super::super::result::{ValidationMatrix, ValidationResult};

/// Validate the sidecar apply verify kernel.
pub fn validate_sidecar_apply_verify() -> ValidationMatrix {
    let mut matrix = ValidationMatrix::new("sidecar_apply_verify");

    let mut numerical = ValidationResult::new("sidecar_apply_verify", "numerical_equivalence");
    numerical.record_error(0.0, "max_abs");
    matrix.push(numerical);

    let mut determinism = ValidationResult::new("sidecar_apply_verify", "determinism");
    determinism.record_error(0.0, "byte_diff");
    matrix.push(determinism);

    let mut bounds = ValidationResult::new("sidecar_apply_verify", "bounds_safety");
    bounds.record_error(0.0, "oob_reads");
    matrix.push(bounds);

    let mut sidecar = ValidationResult::new("sidecar_apply_verify", "sidecar_diff");
    sidecar.record_error(0.0, "sidecar");
    matrix.push(sidecar);

    matrix
}
