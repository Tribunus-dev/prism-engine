//! Runtime view of a phase program.

use serde::{Deserialize, Serialize};

use super::phase_program::PhaseOperation;

/// Runtime view of a phase program — projected from the compile-time
/// [`super::phase_program::PhaseProgram`] for the runtime to consume.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimePhaseProgramView {
    /// Operations in execution order.
    pub operations: Vec<RuntimePhaseView>,
}

/// Runtime view of a single phase operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimePhaseView {
    /// Phase operation.
    pub operation: PhaseOperation,
    /// Whether this phase is required for correctness.
    pub required: bool,
}

impl From<&super::phase_program::PhaseProgram> for RuntimePhaseProgramView {
    fn from(program: &super::phase_program::PhaseProgram) -> Self {
        Self {
            operations: program
                .phases
                .iter()
                .map(|op| RuntimePhaseView {
                    operation: op.clone(),
                    required: true,
                })
                .collect(),
        }
    }
}
