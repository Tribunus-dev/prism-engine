//! Engine-internal shim for the constitutional `config` surface.
//!
//! The canonical authority for the product-shape configuration types
//! (text/vision/audio architectures, layer plans, hardware target,
//! operation route, server config, planning limits, parser) lives in
//! `prism_ecs_constitutional::config`. This module re-exports every
//! public type and function from the constitutional surface so engine
//! code can keep reading the types via the engine-internal
//! `crate::ecs::legacy_config::*` path. New engine code should prefer
//! the constitutional `prism_ecs_constitutional::config::*` import
//! path; the re-exports here are the migration bridge.
//!
//! # History
//!
//! This module was the engine's `compute-core/src/ecs/config/` directory
//! (6 files, 2,583 LOC). The constitutional engine absorption
//! (`changelogs/2026-07-28-phase-a-config-to-constitutional.md`)
//! moved the canonical types to
//! `prism_ecs_constitutional::config::*` and renamed the engine's
//! directory to `legacy_config/`. The architecture safety net
//! `workspace_legacy_config_imports` enforces that no NEW engine
//! code imports the legacy path; the re-exports below are the
//! migration bridge for the few remaining engine-coupled callers.

// Architecture (text/vision/audio) types.
pub use prism_ecs_constitutional::config::architecture::{
    AttentionKind, AudioArchitecture, CommitPolicy, ConfidenceType, DiffusionAttentionKind,
    DiffusionConfig, DiffusionExecutionPlan, DiffusionForwardRoute, DiffusionStage,
    GenerationRegime, KvCacheMode, MaskSelection, MoEConfig, NoiseScheduleType,
    QuantizationMeta, QuantizationMode, RopeSpec, SamplerPolicy, StopCondition,
    TextArchitecture, VisionArchitecture,
};

// Compile-time quantization mode.
pub use prism_ecs_constitutional::config::compile_quant_mode::CompileQuantMode;

// Hardware target.
pub use prism_ecs_constitutional::config::hardware_target::HardwareTarget;

// Per-layer compile plan.
pub use prism_ecs_constitutional::config::layer_plan::{
    ExecutionSpec, LayerSpec, PackedLinearShapes, TensorBinding, TensorRole,
};

// Planning limits.
pub use prism_ecs_constitutional::config::limits::{
    CompilationPlan, PlannedSegment, PlannedTensor, TensorDisposition,
};

// Model execution plan.
pub use prism_ecs_constitutional::config::model_execution_plan::{
    AneFusedIsland, EpiloguePlan, FusedOperation, LayerPlan, ModelExecutionPlan, ProloguePlan,
    SpeculativeModelConfig,
};

// Namespace binding + resolver.
pub use prism_ecs_constitutional::config::namespace_binding::{
    NamespaceBinding, resolve_namespace,
};

// Server config + sections.
pub use prism_ecs_constitutional::config::network::{
    generate_backend_plans, CacheConfigSection, ClusterConfigSection, ModelConfigSection,
    ServerConfig, ServerConfigSection, SpecConfigSection,
};

// Per-operation backend route.
pub use prism_ecs_constitutional::config::operation_route::OperationRoute;

// Manifest + parser.
pub use prism_ecs_constitutional::config::parser::{
    parse_config, ArchitectureConfig, CimageManifest, ManifestModality, ModelManifest,
    ShardManifest,
};

// Layer-plan assembly helpers.
pub use prism_ecs_constitutional::config::layer_plan::{
    build_execution_plan, compile, filter_spec_to_existing,
};

// Compatibility alias — the engine's
// `config::hardware::VisionArchitecture` import style resolves to the
// canonical `architecture::VisionArchitecture`. The constitutional
// surface exposes `config::hardware` as a re-export alias of
// `config::architecture`; engine code that uses
// `legacy_config::hardware::VisionArchitecture` continues to work.
pub mod hardware {
    pub use prism_ecs_constitutional::config::architecture::*;
}

// Compatibility alias for `config::parser::*` callers.
pub mod parser {
    pub use prism_ecs_constitutional::config::parser::*;
}

// Compatibility alias for `config::operation_route::*` callers.
pub mod operation_route {
    pub use prism_ecs_constitutional::config::operation_route::*;
}

// Compatibility alias for `config::network::*` callers (the original
// engine had `crate::ecs::config::network::ServerConfig`).
pub mod network {
    pub use prism_ecs_constitutional::config::network::*;
}

// Compatibility alias for `config::limits::*` callers.
pub mod limits {
    pub use prism_ecs_constitutional::config::limits::*;
}
