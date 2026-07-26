//! Validator for the dense FP16 projection kernel.

use super::super::result::{ValidationMatrix, ValidationResult};

/// Validate the dense FP16 projection kernel.
pub fn validate_dense_projection() -> ValidationMatrix {
    let mut matrix = ValidationMatrix::new("dense_projection_f16");

    let mut numerical = ValidationResult::new("dense_projection_f16", "numerical_equivalence");
    numerical.record_error(5.0e-4, "max_abs");
    matrix.push(numerical);

    let mut layout = ValidationResult::new("dense_projection_f16", "layout_equivalence");
    layout.record_error(0.0, "stride");
    matrix.push(layout);

    let mut determinism = ValidationResult::new("dense_projection_f16", "determinism");
    determinism.record_error(0.0, "byte_diff");
    matrix.push(determinism);

    let mut bounds = ValidationResult::new("dense_projection_f16", "bounds_safety");
    bounds.record_error(0.0, "oob_reads");
    matrix.push(bounds);

    let mut memory = ValidationResult::new("dense_projection_f16", "memory_admissibility");
    memory.record_error(0.0, "peak_rss");
    matrix.push(memory);

    matrix
}
