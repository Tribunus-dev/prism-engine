//! Program validation — pure data types for reporting validation errors.

use serde::{Deserialize, Serialize};

use super::phase_program::PhaseProgram;

/// Reason a program validation failed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProgramValidationError {
    /// A referenced input tensor is not produced by any prior phase.
    MissingInput {
        /// Operation identifier.
        operation_id: String,
        /// Tensor identifier.
        tensor_id: String,
    },
    /// A tensor is produced by multiple phases.
    DuplicateProduction {
        /// Tensor identifier.
        tensor_id: String,
    },
    /// The program is empty.
    EmptyProgram,
    /// A phase dependency forms a cycle.
    CyclicDependency,
    /// The shape class is invalid for the program.
    InvalidShapeClass,
}

/// Validation report for a program.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgramValidationReport {
    /// Whether the program passed validation.
    pub valid: bool,
    /// Errors found during validation.
    pub errors: Vec<ProgramValidationError>,
}

impl ProgramValidationReport {
    /// Validate a phase program and return the report.
    pub fn from_program(program: &PhaseProgram) -> Self {
        let mut errors = Vec::new();
        if program.phases.is_empty() {
            errors.push(ProgramValidationError::EmptyProgram);
        }
        Self {
            valid: errors.is_empty(),
            errors,
        }
    }
}
