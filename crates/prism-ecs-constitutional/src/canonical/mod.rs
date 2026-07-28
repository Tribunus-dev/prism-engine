//! `prism_ecs_constitutional::canonical` — canonical compiler types.
//!
//! This is the constitutional home for the engine-coupled
//! canonical/ surface: identity, generation, kernel ABI, execution
//! graph, provenance, representation, compile plan, model IR, and
//! receipt store. The surface is the canonical authority for the
//! `Source → ModelIr → RepresentationPlan → ExecutionGraph →
//!  KernelPlan → CompiledKernelArtifact → CimageBuildInput`
//! ownership chain. Every backend, fusion strategy, and
//! quantization format is expressed through these types; no
//! path-specific state escapes into the compiler pipeline without
//! being captured here.
//!
//! Higher-leverage, engine-coupled adapter code (the engine's
//! Metal compiler, the prism-backend Metal/Ane dispatch bridges,
//! and the engine binaries) remains engine-internal at
//! `compute-core/src/ecs/legacy_canonical/` because it depends on
//! engine FFI bridges and the per-backend executor stack. This
//! surface is the cross-platform, constitutional home for the data
//! types.
//!
//! Submodules (one authority per file):
//! - [`identity`] — canonical identity primitives: generation,
//!   candidate, engram, hardware, model, compiler. Authority: the
//!   type system.
//! - [`generation`] — `CimageGeneration`, the resolved output of
//!   one compiler invocation, with binding metadata. Authority:
//!   one-generation-per-compilation.
//! - [`kernel_abi`] — `KernelAbi`, `KernelPlan`, dispatch geometry
//!   policy, and compiled-kernel artifact digest helpers.
//!   Authority: the kernel interface contract.
//! - [`execution_graph`] — execution-oriented graph produced from
//!   `ModelIr` + `RepresentationPlan`. Authority: the
//!   execution-oriented view.
//! - [`provenance`] — receipt bundles, measured-candidate records,
//!   promotion requests, replay manifests, and execution bindings.
//!   Authority: the evidence chain.
//! - [`representation`] — per-tensor representation decision
//!   (codec, scale structure, residual plan). Authority: the
//!   per-tensor representation decision.
//! - [`compile_plan`] — `CompileRequest` / `CompilePlan` /
//!   `CompileEventStream`, the public compilation API. Authority:
//!   the compiler pipeline contract.
//! - [`model_ir`] — platform-independent model representation.
//!   Authority: the model semantics.
//! - [`receipt_store`] — content-addressed receipt persistence.
//!   Authority: the receipt store.
//!
//! # Migration status
//!
//! This surface was added when the canonical/ engine-deletion
//! migration was being completed (2026-07-28). The engine's
//! `compute-core/src/ecs/canonical/` directory was renamed to
//! `compute-core/src/ecs/legacy_canonical/`; the directory is the
//! engine-internal migration inventory and is the engine-internal
//! home for the engine-coupled adapter code (Metal compiler,
//! prism-metal-runtime bridges, engine binaries). This surface
//! is the cross-platform, constitutional home for the data
//! types and re-exports the IR primitives from `prism_ecs_ir`.

pub mod compile_plan;
pub mod execution_graph;
pub mod generation;
pub mod identity;
pub mod kernel_abi;
pub mod model_ir;
pub mod provenance;
pub mod receipt_store;
pub mod representation;

// Re-export every public symbol at the canonical module root for
// `prism_ecs_constitutional::canonical::*` ergonomics. The
// `mod` statements above remain authoritative for documentation
// purposes; the `pub use` block is the public surface.
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
pub use provenance::*;
pub use receipt_store::*;
pub use representation::*;
