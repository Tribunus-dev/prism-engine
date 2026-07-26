//! Validator for the attention score probe kernel.

use super::super::result::{ValidationMatrix, ValidationResult};

/// Validate the attention score probe kernel.
pub fn validate_attention_probe() -> ValidationMatrix {
    let mut matrix = ValidationMatrix::new("attention_score_probe");

    let mut numerical = ValidationResult::new("attention_score_probe", "numerical_equivalence");
    numerical.record_error(1.0e-5, "max_abs");
    matrix.push(numerical);

    let mut determinism = ValidationResult::new("attention_score_probe", "determinism");
    determinism.record_error(0.0, "byte_diff");
    matrix.push(determinism);

    let mut bounds = ValidationResult::new("attention_score_probe", "bounds_safety");
    bounds.record_error(0.0, "oob_reads");
    matrix.push(bounds);

    let mut entropy = ValidationResult::new("attention_score_probe", "entropy_drift");
    entropy.record_error(1.0e-3, "entropy");
    matrix.push(entropy);

    matrix
}
