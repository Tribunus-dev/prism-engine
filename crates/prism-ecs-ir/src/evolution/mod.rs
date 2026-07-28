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
pub mod chromosome;
pub mod compile_plan;
pub mod decompose;
pub mod emitters;
pub mod evaluate;
pub mod foundation;
pub mod frontier;
pub mod hierarchical;
pub mod joint;
pub mod memory;
pub mod mutation_table;
pub mod objectives;
pub mod pareto;
pub mod progressive;
pub mod receipts;
pub mod runtime;
pub mod sensitivity;
pub mod variation;

// Re-export everything from the sub-modules for backward compatibility.
pub use budget::*;
pub use compile_plan::*;
pub use decompose::*;
pub use emitters::*;
pub use evaluate::*;
pub use foundation::*;
pub use frontier::*;
pub use hierarchical::*;
pub use joint::*;
pub use memory::*;
pub use mutation_table::*;
pub use objectives::*;
pub use pareto::*;
pub use progressive::*;
pub use receipts::*;
pub use runtime::*;
pub use sensitivity::*;
pub use variation::*;
