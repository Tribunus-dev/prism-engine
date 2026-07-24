//! Prism ECS-native IR kernel.
//!
//! ECS-native intermediate representation: Ops, Regions, Blocks, Values,
//! Types, Attributes, serialization, rewriting, and type inference.
//! Every operation is an Entity in the ECS World.

pub mod arith;
pub mod backend_apple_gpu;
pub mod backend_cpu;
pub mod backend_dispatch;
pub mod backend_intel_gpu;
pub mod block;
pub mod block_analysis;
pub mod block_program;
pub mod bonsai;
pub mod builder;
pub mod cimage_types;
pub mod codegen_metal;
pub mod dialect_registry;
pub mod dominance;
pub mod evolution;
pub mod func;
pub mod hash_dispatch;
pub mod interfaces;
pub mod ir_attrs;
pub mod ir_types;
pub mod linalg;
pub mod lowering;
pub mod manifest;
pub mod model_graph;
pub mod op;
pub mod pass_manager;
pub mod pass_pipeline;
pub mod pattern_rewriter;
pub mod region;
pub mod rewrite_driver;
pub mod scf;
pub mod semantic_region;
pub mod serde;
pub mod symbol_table;
pub mod traits;
pub mod type_inference;
pub mod value;

pub use block_analysis::{AnalysisResult, BlockAnalyzer, FusionSuggestion, PatternKind};
pub use hash_dispatch::{DispatchHash, HashDispatchSystem, HashDispatchTable};
pub use model_graph::{
    generate_plan, ActivationFunction, ArchitectureFamily, AttentionBackend, BackendAssignment,
    ComputeNode, ExecutionPlan, KVCacheConfig, ModelGraph, TensorBlueprint, TensorRole,
    UnifiedConfig,
};
pub use symbol_table::{SymbolConflict, SymbolTable};
