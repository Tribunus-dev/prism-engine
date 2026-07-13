//! Canonical compiler types — unified IR, plans, and receipts.
//!
//! These types define the single ownership chain for the compiler:
//!   Source → ModelIr → RepresentationPlan → ExecutionGraph → KernelPlan → CompiledKernelArtifact → CimageBuildInput
//!
//! Every backend, fusion strategy, and quantization format is expressed
//! through these types. No path-specific state escapes into the compiler
//! pipeline without being captured here.
//!
//! PR B — Canonical compiler types. Introduced without changing existing
//! behavior. The existing procedural pipeline is adapted to emit these
//! types alongside its current internal representations.

pub mod compile_plan;
pub mod execution_graph;
pub mod generation;
pub mod identity;
pub mod kernel_abi;
pub mod model_ir;
pub mod representation;

pub use compile_plan::*;
pub use execution_graph::*;
pub use generation::*;
pub use identity::{
    CandidateId, CompilerIdentity, CorpusId, EngramArtifactId, EngramId, GenerationId,
    HardwareProfileId, LogicalTensorId, ModelSourceId, PhysicalSegmentId, ReceiptId,
    RepresentationId, TensorShape, Timestamp,
};
pub use kernel_abi::*;
pub use model_ir::*;
pub use representation::*;
