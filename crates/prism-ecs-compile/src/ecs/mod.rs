//! ECS compilation components and pipeline orchestration.
//!
//! Defines ECS [`Component`] types that represent compilation pipeline state
//! attached to a session [`Entity`], plus system functions that operate on
//! a [`World`] to drive each pipeline stage. The [`CompilationOrchestrator`]
//! owns a world + session entity and runs the full pipeline.
//!
//! # Architecture
//!
//! Each stage attaches its output as a component on the session entity.
//! Shared read-only data lives as [`World`] resources where the type is
//! `Sync`, or as [`World`] extensions (via `set_extension`) when the type
//! is `Send` but not `Sync` (e.g. [`CanonicalSource`]).
//!
//! ```text
//! World resources/extensions:            Session entity components:
//! ┌──────────────────┐                  ┌──────────────────────┐
//! │ SessionHandle (R)│──► session Entity│ CompilationSession   │
//! │ CurrentSource (E)│                  │ SourceModel          │
//! │ VecEventSink (R) │                  │ TensorCollection     │
//! │ TargetCaps (R)   │                  │ SpatialGraphComponent│
//! │ ExecutionMode (R)│                  │ SearchStateComponent │
//! └──────────────────┘                  │ LegalizedPlan        │
//!                                       │ KernelCollection     │
//!                                       │ CImageArtifact       │
//!                                       └──────────────────────┘
//! ```
//!
//! # Module layout
//!
//! - [`components`] — session-entity components (one struct per authority).
//! - [`resources`] — world resources and extensions.
//! - [`orchestrator`] — pipeline driver that owns the world and the session
//!   entity and dispatches stages in order.
//! - [`stages`] — per-stage system functions, split by pipeline phase.

pub mod components;
pub mod orchestrator;
pub mod resources;
pub mod stages;

// Re-exports for backward compatibility with callers that previously
// imported from `prism_ecs_compile::ecs::*` directly.
pub use components::{
    CImageArtifact, CompilationReceipt, CompilationSession, KernelCollection, LegalizedPlan,
    SearchStateComponent, SessionStatus, SourceModel, SpatialGraphComponent, TensorCollection,
};
pub use orchestrator::CompilationOrchestrator;
pub use resources::{
    CImagePlanDigest, CurrentSource, EvaluatorOption, ModelManifestResource, SessionHandle,
    SourceAdapterList, VecEventSink,
};
pub use stages::{
    system_build_graph, system_build_receipt, system_certify, system_detect_source,
    system_emit_cimage, system_generate_kernels, system_legalize, system_run_search,
};
