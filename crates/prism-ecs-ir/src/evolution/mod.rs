//! Evolution search pipeline for the Prism ECS compiler.
//!
//! This module absorbs the compute-core evolutionary search pipeline,
//! providing ECS-native components and systems for format/operation/layout
//! co-evolution. The module is structured as follows:
//!
//! - `mutation_table`: per-tensor format + operation mutation operators
//! - `foundation`: base identity types (CandidateId, LogicalTensorId),
//!   scoring (FitnessScore), and the 8-dimensional CandidateGenome
//! - `budget`: search budget constraints (wall time, memory, energy)
//! - `sensitivity`: per-tensor sensitivity analysis + budget classification
//! - `frontier`: Pareto frontier maintenance and exploration system
//! - `decompose`: genome decomposition into independently evolvable sub-problems
//! - `joint`: joint evolution orchestrator (crossover, mutation, generation loop)
//! - `evaluate`: pluggable evaluation system trait + synthetic evaluator
//! - `compile_plan`: ECS components for compilation plan + format/tile assignment

pub mod assembly;
pub mod budget;
pub mod compile_plan;
pub mod decompose;
pub mod evaluate;
pub mod foundation;
pub mod frontier;
pub mod joint;
pub mod mutation_table;
pub mod sensitivity;

// Re-export everything from the sub-modules for backward compatibility.
pub use budget::*;
pub use compile_plan::*;
pub use decompose::*;
pub use evaluate::*;
pub use foundation::*;
pub use frontier::*;
pub use joint::*;
pub use mutation_table::*;
pub use sensitivity::*;
