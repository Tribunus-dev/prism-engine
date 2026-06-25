//! Phase program schema — compiled, shape-specialized execution programs
//! for the SealedComputeImageExecutable.
//!
//! A phase program ([`SerializedPhaseProgram`]) is the serializable output
//! of the compiler's phase-graph lowering pass, specialized for one
//! workload shape class ([`ExecutionShapeClass`]).  It contains all phases
//! (units of schedulable work), dependency edges, arena and residency plan
//! references, artifact selection decisions, and fallback chains for
//! compatible variant switches.
//!
//! Every type in this module derives [`Serialize`] and [`Deserialize`] via
//! serde, making the phase program persistable within the
//! SealedComputeImageExecutable payload.

pub mod arena;
pub mod dependencies;
pub mod phase_program;
pub mod runtime_view;
pub mod selection;
pub mod serialization;
pub mod validation;

#[allow(ambiguous_glob_reexports)]
pub use arena::*;
#[allow(ambiguous_glob_reexports)]
pub use dependencies::*;
#[allow(ambiguous_glob_reexports)]
pub use phase_program::*;
pub use runtime_view::*;
pub use serialization::*;
pub use validation::*;
