//! Program/launch IR — pure data types and pure algorithms for
//! phase programs, serialization, validation, and selection.

pub mod arena;
pub mod dependencies;
pub mod phase_program;
pub mod runtime_view;
pub mod selection;
pub mod serialization;
pub mod validation;

pub use phase_program::{
    ExecutionLane, PhaseOperation, PhaseProgram, ProgramId, SemanticOperation,
    SerializedPhaseProgram,
};
pub use selection::{ProgramArtifactSelection, VariantSelectionRefusal};
pub use serialization::{ProgramFormat, ProgramSerializer};
pub use validation::{ProgramValidationError, ProgramValidationReport};
