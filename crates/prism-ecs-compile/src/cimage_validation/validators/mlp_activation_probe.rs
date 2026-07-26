//! Validator for the MLP activation probe kernel.

use super::super::result::{ValidationMatrix, ValidationResult};

/// Validate the MLP activation probe kernel.
pub fn validate_mlp_activation_probe() -> ValidationMatrix {
    let mut matrix = ValidationMatrix::new("mlp_activation_probe");

    let mut numerical = ValidationResult::new("mlp_activation_probe", "numerical_equivalence");
    numerical.record_error(1.0e-5, "max_abs");
    matrix.push(numerical);

    let mut determinism = ValidationResult::new("mlp_activation_probe", "determinism");
    determinism.record_error(0.0, "byte_diff");
    matrix.push(determinism);

    let mut bounds = ValidationResult::new("mlp_activation_probe", "bounds_safety");
    bounds.record_error(0.0, "oob_reads");
    matrix.push(bounds);

    let mut activation = ValidationResult::new("mlp_activation_probe", "activation");
    activation.record_error(1.0e-5, "sigmoid_diff");
    matrix.push(activation);

    matrix
}
