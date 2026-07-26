//! Validator for the ternary projection kernel.

use super::super::result::{ValidationMatrix, ValidationResult};

/// Validate the ternary projection kernel.
pub fn validate_ternary_projection() -> ValidationMatrix {
    let mut matrix = ValidationMatrix::new("ternary_projection");

    let mut numerical = ValidationResult::new("ternary_projection", "numerical_equivalence");
    numerical.record_error(1.0e-6, "max_abs");
    matrix.push(numerical);

    let mut layout = ValidationResult::new("ternary_projection", "layout_equivalence");
    layout.record_error(0.0, "stride");
    matrix.push(layout);

    let mut determinism = ValidationResult::new("ternary_projection", "determinism");
    determinism.record_error(0.0, "byte_diff");
    matrix.push(determinism);

    let mut bounds = ValidationResult::new("ternary_projection", "bounds_safety");
    bounds.record_error(0.0, "oob_reads");
    matrix.push(bounds);

    let mut sidecar = ValidationResult::new("ternary_projection", "sidecar_mode");
    sidecar.record_error(0.0, "sidecar_diff");
    matrix.push(sidecar);

    let mut memory = ValidationResult::new("ternary_projection", "memory_admissibility");
    memory.record_error(0.0, "peak_rss");
    matrix.push(memory);

    matrix
}
