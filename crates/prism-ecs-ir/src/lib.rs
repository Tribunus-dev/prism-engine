//! Prism ECS-native IR kernel.
//!
//! ECS-native intermediate representation: Ops, Regions, Blocks, Values,
//! Types, Attributes, serialization, rewriting, and type inference.
//! Every operation is an Entity in the ECS World.

pub mod arith;
pub mod block;
pub mod bonsai;
pub mod builder;
pub mod codegen_metal;
pub mod dominance;
pub mod evolution;
pub mod func;
pub mod ir_attrs;
pub mod ir_types;
pub mod linalg;
pub mod lowering;
pub mod op;
pub mod region;
pub mod rewrite_driver;
pub mod scf;
pub mod serde;
pub mod symbol_table;
pub mod type_inference;
pub mod value;

pub use symbol_table::{SymbolConflict, SymbolTable};
