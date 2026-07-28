//! Engine-internal adapter structs that bridge the constitutional
//! `prism_ecs_runtime::systems::*` data types onto the engine's
//! `CompilerSystem` trait.
//!
//! # Authority
//!
//! This module is the **only** engine-internal authority for
//! `CompilerSystem` impls that wrap the constitutional system
//! data types. Every system the engine's `compile_session.rs`
//! registers has its adapter defined here; the body of each
//! adapter's `run` is a no-op stub (the real system work is
//! still wired through the engine's `runtime/compilation_systems.rs`
//! and `runtime/ecs_components.rs`, which this module complements
//! but does not replace).
//!
//! # Module convention
//!
//! Each sub-module is named after the constitutional surface it
//! wraps and re-exports the adapter struct under the original
//! system name. This lets the engine's `compile_session.rs`
//! keep its `use crate::ecs::system_adapters::*;` shape and
//! its `Box::new(metal_init::MetalInitSystem)` call sites
//! unchanged — only the import path changes.
//!
//! # Migration status
//!
//! Added as part of the engine-absorption recipe (E-2). After the
//! engine's `compute-core/src/ecs/system/` directory is deleted
//! in E-{N+1}, this module is the sole remaining place where
//! the engine's `CompilerSystem` trait is implemented for
//! the system data types.

use crate::ecs::{CompilerSystem, SchedulePhase, World};

// ── Phase A: ModelLoading ────────────────────────────────────────

pub mod metal_init {
    use super::*;
    pub struct MetalInitSystem(pub prism_ecs_runtime::systems::metal_init::MetalInitSystem);
    impl CompilerSystem for MetalInitSystem {
        fn name(&self) -> &str { "MetalInitSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::ModelLoading }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
}
pub use metal_init::MetalInitSystem;

pub mod phase_engine_init {
    use super::*;
    pub struct PhaseEngineInitSystem(pub prism_ecs_runtime::systems::phase_engine_init::PhaseEngineInitSystem);
    impl CompilerSystem for PhaseEngineInitSystem {
        fn name(&self) -> &str { "PhaseEngineInitSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::ModelLoading }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
}
pub use phase_engine_init::PhaseEngineInitSystem;

pub mod session_init {
    use super::*;
    pub struct SessionInitSystem(pub prism_ecs_runtime::systems::session_init::SessionInitSystem);
    impl CompilerSystem for SessionInitSystem {
        fn name(&self) -> &str { "SessionInitSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::ModelLoading }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
}
pub use session_init::SessionInitSystem;

pub mod source_load {
    use super::*;
    pub struct SourceLoadingSystem(pub prism_ecs_runtime::systems::source_load::SourceLoadingSystem);
    impl CompilerSystem for SourceLoadingSystem {
        fn name(&self) -> &str { "SourceLoadingSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::ModelLoading }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
    pub struct TensorTableLoadingSystem(pub prism_ecs_runtime::systems::source_load::TensorTableLoadingSystem);
    impl CompilerSystem for TensorTableLoadingSystem {
        fn name(&self) -> &str { "TensorTableLoadingSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::ModelLoading }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
}
pub use source_load::{SourceLoadingSystem, TensorTableLoadingSystem};

pub mod tts {
    use super::*;
    pub struct TTSSystem(pub prism_ecs_runtime::systems::tts::TTSSystem);
    impl CompilerSystem for TTSSystem {
        fn name(&self) -> &str { "TTSSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::ModelLoading }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
}
pub use tts::TTSSystem;

pub mod download {
    use super::*;
    pub struct DownloadSystem(pub prism_ecs_runtime::systems::download::DownloadSystem);
    impl CompilerSystem for DownloadSystem {
        fn name(&self) -> &str { "DownloadSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::ModelLoading }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
    pub struct HfSourceParsingSystem(pub prism_ecs_runtime::systems::download::HfSourceParsingSystem);
    impl CompilerSystem for HfSourceParsingSystem {
        fn name(&self) -> &str { "HfSourceParsingSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::ModelLoading }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
}
pub use download::{DownloadSystem, HfSourceParsingSystem};

pub mod archive {
    use super::*;
    pub struct ArchiveSystem(pub prism_ecs_runtime::systems::archive::ArchiveSystem);
    impl CompilerSystem for ArchiveSystem {
        fn name(&self) -> &str { "ArchiveSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::ModelLoading }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
    pub struct PrecompiledAneSystem(pub prism_ecs_runtime::systems::archive::PrecompiledAneSystem);
    impl CompilerSystem for PrecompiledAneSystem {
        fn name(&self) -> &str { "PrecompiledAneSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::ModelLoading }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
}
pub use archive::{ArchiveSystem, PrecompiledAneSystem};

pub mod draft_model {
    use super::*;
    pub struct DraftModelSystem(pub prism_ecs_runtime::systems::draft_model::DraftModelSystem);
    impl CompilerSystem for DraftModelSystem {
        fn name(&self) -> &str { "DraftModelSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::ModelLoading }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
}
pub use draft_model::DraftModelSystem;

pub mod engine_systems {
    use super::*;
    pub struct EngineInitSystem(pub prism_ecs_runtime::systems::engine_systems::EngineInitSystem);
    impl CompilerSystem for EngineInitSystem {
        fn name(&self) -> &str { "EngineInitSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::ModelLoading }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
    pub struct ModelInstallSystem(pub prism_ecs_runtime::systems::engine_systems::ModelInstallSystem);
    impl CompilerSystem for ModelInstallSystem {
        fn name(&self) -> &str { "ModelInstallSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::ModelLoading }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
    pub struct ModelLoadSystem(pub prism_ecs_runtime::systems::engine_systems::ModelLoadSystem);
    impl CompilerSystem for ModelLoadSystem {
        fn name(&self) -> &str { "ModelLoadSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::ModelLoading }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
    pub struct CimageLoadSystem(pub prism_ecs_runtime::systems::engine_systems::CimageLoadSystem);
    impl CompilerSystem for CimageLoadSystem {
        fn name(&self) -> &str { "CimageLoadSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::ModelLoading }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
    pub struct HostInferenceInitSystem(pub prism_ecs_runtime::systems::engine_systems::HostInferenceInitSystem);
    impl CompilerSystem for HostInferenceInitSystem {
        fn name(&self) -> &str { "HostInferenceInitSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::ModelLoading }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
    pub struct ModelUnloadSystem(pub prism_ecs_runtime::systems::engine_systems::ModelUnloadSystem);
    impl CompilerSystem for ModelUnloadSystem {
        fn name(&self) -> &str { "ModelUnloadSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::Packaging }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
    pub struct EngineMetricsSystem(pub prism_ecs_runtime::systems::engine_systems::EngineMetricsSystem);
    impl CompilerSystem for EngineMetricsSystem {
        fn name(&self) -> &str { "EngineMetricsSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::Packaging }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
    pub struct EngineShutdownSystem(pub prism_ecs_runtime::systems::engine_systems::EngineShutdownSystem);
    impl CompilerSystem for EngineShutdownSystem {
        fn name(&self) -> &str { "EngineShutdownSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::Packaging }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
    pub struct GenerationRequestSystem(pub prism_ecs_runtime::systems::engine_systems::GenerationRequestSystem);
    impl CompilerSystem for GenerationRequestSystem {
        fn name(&self) -> &str { "GenerationRequestSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::Execution }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
    pub struct CimageGenerateSystem(pub prism_ecs_runtime::systems::engine_systems::CimageGenerateSystem);
    impl CompilerSystem for CimageGenerateSystem {
        fn name(&self) -> &str { "CimageGenerateSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::Execution }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
    pub struct InferenceCycleSystem(pub prism_ecs_runtime::systems::engine_systems::InferenceCycleSystem);
    impl CompilerSystem for InferenceCycleSystem {
        fn name(&self) -> &str { "InferenceCycleSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::Execution }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
    pub struct TokenBudgetInferenceSystem(pub prism_ecs_runtime::systems::engine_systems::TokenBudgetInferenceSystem);
    impl CompilerSystem for TokenBudgetInferenceSystem {
        fn name(&self) -> &str { "TokenBudgetInferenceSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::Execution }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
    pub struct CancelSystem(pub prism_ecs_runtime::systems::engine_systems::CancelSystem);
    impl CompilerSystem for CancelSystem {
        fn name(&self) -> &str { "CancelSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::Execution }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
    pub struct MemoryPressureSystem(pub prism_ecs_runtime::systems::engine_systems::MemoryPressureSystem);
    impl CompilerSystem for MemoryPressureSystem {
        fn name(&self) -> &str { "MemoryPressureSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::Execution }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
}
pub use engine_systems::{
    CancelSystem, CimageGenerateSystem, CimageLoadSystem, EngineInitSystem,
    EngineMetricsSystem, EngineShutdownSystem, GenerationRequestSystem,
    HostInferenceInitSystem, InferenceCycleSystem, MemoryPressureSystem,
    ModelInstallSystem, ModelLoadSystem, ModelUnloadSystem,
    TokenBudgetInferenceSystem,
};

pub mod model_load {
    use super::*;
    pub struct ModelAdapterSystem(pub prism_ecs_runtime::systems::model_load::ModelAdapterSystem);
    impl CompilerSystem for ModelAdapterSystem {
        fn name(&self) -> &str { "ModelAdapterSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::ModelLoading }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
}
pub use model_load::ModelAdapterSystem;

// ── Phase B: Quantization ───────────────────────────────────────

pub mod int4_pack {
    use super::*;
    pub struct Int4PackSystem(pub prism_ecs_runtime::systems::int4_pack::Int4PackSystem);
    impl CompilerSystem for Int4PackSystem {
        fn name(&self) -> &str { "Int4PackSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::Quantization }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
}
pub use int4_pack::Int4PackSystem;

pub mod quant_plan {
    use super::*;
    pub struct CodecSelectionSystem(pub prism_ecs_runtime::systems::quant_plan::CodecSelectionSystem);
    impl CompilerSystem for CodecSelectionSystem {
        fn name(&self) -> &str { "CodecSelectionSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::Quantization }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
    pub struct PrecisionPlanSystem(pub prism_ecs_runtime::systems::quant_plan::PrecisionPlanSystem);
    impl CompilerSystem for PrecisionPlanSystem {
        fn name(&self) -> &str { "PrecisionPlanSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::Quantization }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
}
pub use quant_plan::{CodecSelectionSystem, PrecisionPlanSystem};

pub mod moe_budget {
    use super::*;
    pub struct MoERoutingSystem(pub prism_ecs_runtime::systems::moe_budget::MoERoutingSystem);
    impl CompilerSystem for MoERoutingSystem {
        fn name(&self) -> &str { "MoERoutingSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::Quantization }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
    pub struct MemoryBudgetSystem(pub prism_ecs_runtime::systems::moe_budget::MemoryBudgetSystem);
    impl CompilerSystem for MemoryBudgetSystem {
        fn name(&self) -> &str { "MemoryBudgetSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::Quantization }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
}
pub use moe_budget::{MemoryBudgetSystem, MoERoutingSystem};

pub mod planning_core {
    use super::*;
    pub struct AneEligibilitySystem(pub prism_ecs_runtime::systems::planning_core::AneEligibilitySystem);
    impl CompilerSystem for AneEligibilitySystem {
        fn name(&self) -> &str { "AneEligibilitySystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::QuantizationPlanning }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
    pub struct MemoryBudgetSystemV2(pub prism_ecs_runtime::systems::planning_core::MemoryBudgetSystemV2);
    impl CompilerSystem for MemoryBudgetSystemV2 {
        fn name(&self) -> &str { "MemoryBudgetSystemV2" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::MemoryPlanning }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
    pub struct RegionPlannerSystem(pub prism_ecs_runtime::systems::planning_core::RegionPlannerSystem);
    impl CompilerSystem for RegionPlannerSystem {
        fn name(&self) -> &str { "RegionPlannerSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::MemoryPlanning }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
    pub struct RegionCatalogueSystem(pub prism_ecs_runtime::systems::planning_core::RegionCatalogueSystem);
    impl CompilerSystem for RegionCatalogueSystem {
        fn name(&self) -> &str { "RegionCatalogueSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::FusionDispatch }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
    pub struct ReceiptSystem(pub prism_ecs_runtime::systems::planning_core::ReceiptSystem);
    impl CompilerSystem for ReceiptSystem {
        fn name(&self) -> &str { "ReceiptSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::Packaging }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
}
pub use planning_core::{
    AneEligibilitySystem, MemoryBudgetSystemV2, ReceiptSystem,
    RegionCatalogueSystem, RegionPlannerSystem,
};

// ── Phase C: MemoryPlanning ─────────────────────────────────────

pub mod memory_plan {
    use super::*;
    pub struct MemoryDomainAssignmentSystem(pub prism_ecs_runtime::systems::memory_plan::MemoryDomainAssignmentSystem);
    impl CompilerSystem for MemoryDomainAssignmentSystem {
        fn name(&self) -> &str { "MemoryDomainAssignmentSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::MemoryPlanning }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
    pub struct BufferAllocationSystem(pub prism_ecs_runtime::systems::memory_plan::BufferAllocationSystem);
    impl CompilerSystem for BufferAllocationSystem {
        fn name(&self) -> &str { "BufferAllocationSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::MemoryPlanning }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
}
pub use memory_plan::{BufferAllocationSystem, MemoryDomainAssignmentSystem};

pub mod buffer_lifetime {
    use super::*;
    pub struct LifetimeAnalysisSystem(pub prism_ecs_runtime::systems::buffer_lifetime::LifetimeAnalysisSystem);
    impl CompilerSystem for LifetimeAnalysisSystem {
        fn name(&self) -> &str { "LifetimeAnalysisSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::MemoryPlanning }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
    pub struct ScratchPlanningSystem(pub prism_ecs_runtime::systems::buffer_lifetime::ScratchPlanningSystem);
    impl CompilerSystem for ScratchPlanningSystem {
        fn name(&self) -> &str { "ScratchPlanningSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::MemoryPlanning }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
}
pub use buffer_lifetime::{LifetimeAnalysisSystem, ScratchPlanningSystem};

pub mod backend_residency {
    use super::*;
    pub struct BackendResidencySystem(pub prism_ecs_runtime::systems::backend_residency::BackendResidencySystem);
    impl CompilerSystem for BackendResidencySystem {
        fn name(&self) -> &str { "BackendResidencySystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::MemoryPlanning }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
}
pub use backend_residency::BackendResidencySystem;

// ── Phase D: FusionDispatch ─────────────────────────────────────

pub mod fusion {
    use super::*;
    pub mod analysis {
        use super::*;
        pub struct FusionAnalysisSystem(pub prism_ecs_runtime::systems::fusion::analysis::FusionAnalysisSystem);
        impl CompilerSystem for FusionAnalysisSystem {
            fn name(&self) -> &str { "FusionAnalysisSystem" }
            fn phase(&self) -> SchedulePhase { SchedulePhase::FusionDispatch }
            fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
        }
    }
    pub mod heuristic {
        use super::*;
        pub struct FusionHeuristicSystem(pub prism_ecs_runtime::systems::fusion::heuristic::FusionHeuristicSystem);
        impl CompilerSystem for FusionHeuristicSystem {
            fn name(&self) -> &str { "FusionHeuristicSystem" }
            fn phase(&self) -> SchedulePhase { SchedulePhase::FusionDispatch }
            fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
        }
    }
    pub mod dispatch {
        use super::*;
        pub struct DispatchFormationSystem(pub prism_ecs_runtime::systems::fusion::dispatch::DispatchFormationSystem);
        impl CompilerSystem for DispatchFormationSystem {
            fn name(&self) -> &str { "DispatchFormationSystem" }
            fn phase(&self) -> SchedulePhase { SchedulePhase::FusionDispatch }
            fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
        }
    }
    pub mod scalar {
        use super::*;
        pub struct ScalarDispatchSystem(pub prism_ecs_runtime::systems::fusion::scalar::ScalarDispatchSystem);
        impl CompilerSystem for ScalarDispatchSystem {
            fn name(&self) -> &str { "ScalarDispatchSystem" }
            fn phase(&self) -> SchedulePhase { SchedulePhase::FusionDispatch }
            fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
        }
    }
}
pub use fusion::{
    analysis::FusionAnalysisSystem, dispatch::DispatchFormationSystem,
    heuristic::FusionHeuristicSystem, scalar::ScalarDispatchSystem,
};

pub mod compiler_systems {
    use super::*;
    pub struct GraphEqualizationSystem(pub prism_ecs_runtime::systems::compiler_systems::GraphEqualizationSystem);
    impl CompilerSystem for GraphEqualizationSystem {
        fn name(&self) -> &str { "GraphEqualizationSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::FusionDispatch }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
    pub struct GraphOptimizerSystem(pub prism_ecs_runtime::systems::compiler_systems::GraphOptimizerSystem);
    impl CompilerSystem for GraphOptimizerSystem {
        fn name(&self) -> &str { "GraphOptimizerSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::FusionDispatch }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
    pub struct CompileScheduleSystem(pub prism_ecs_runtime::systems::compiler_systems::CompileScheduleSystem);
    impl CompilerSystem for CompileScheduleSystem {
        fn name(&self) -> &str { "CompileScheduleSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::Compilation }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
    pub struct BackendAssessmentSystem(pub prism_ecs_runtime::systems::compiler_systems::BackendAssessmentSystem);
    impl CompilerSystem for BackendAssessmentSystem {
        fn name(&self) -> &str { "BackendAssessmentSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::Compilation }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
}
pub use compiler_systems::{
    BackendAssessmentSystem, CompileScheduleSystem, GraphEqualizationSystem,
    GraphOptimizerSystem,
};

pub mod execution_graph {
    use super::*;
    pub struct ExecutionGraphSystem(pub prism_ecs_runtime::systems::execution_graph::ExecutionGraphSystem);
    impl CompilerSystem for ExecutionGraphSystem {
        fn name(&self) -> &str { "ExecutionGraphSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::FusionDispatch }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
}
pub use execution_graph::ExecutionGraphSystem;

pub mod capability_registry_sys {
    use super::*;
    pub struct CapabilityRegistrySystem(pub prism_ecs_runtime::systems::capability_registry_sys::CapabilityRegistrySystem);
    impl CompilerSystem for CapabilityRegistrySystem {
        fn name(&self) -> &str { "CapabilityRegistrySystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::FusionDispatch }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
}
pub use capability_registry_sys::CapabilityRegistrySystem;

// ── Phase E: KernelGeneration ───────────────────────────────────

pub mod kernel_gen {
    use super::*;
    pub struct TemplateSelectionSystem(pub prism_ecs_runtime::systems::kernel_gen::TemplateSelectionSystem);
    impl CompilerSystem for TemplateSelectionSystem {
        fn name(&self) -> &str { "TemplateSelectionSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::KernelGeneration }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
    pub struct ParameterResolutionSystem(pub prism_ecs_runtime::systems::kernel_gen::ParameterResolutionSystem);
    impl CompilerSystem for ParameterResolutionSystem {
        fn name(&self) -> &str { "ParameterResolutionSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::KernelGeneration }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
    pub struct TemplateExpansionSystem(pub prism_ecs_runtime::systems::kernel_gen::TemplateExpansionSystem);
    impl CompilerSystem for TemplateExpansionSystem {
        fn name(&self) -> &str { "TemplateExpansionSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::KernelGeneration }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
}
pub use kernel_gen::{
    ParameterResolutionSystem, TemplateExpansionSystem, TemplateSelectionSystem,
};

pub mod tuning {
    use super::*;
    pub struct AutoTuningSystem(pub prism_ecs_runtime::systems::tuning::AutoTuningSystem);
    impl CompilerSystem for AutoTuningSystem {
        fn name(&self) -> &str { "AutoTuningSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::KernelGeneration }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
    pub struct AOTProfileMatchSystem(pub prism_ecs_runtime::systems::tuning::AOTProfileMatchSystem);
    impl CompilerSystem for AOTProfileMatchSystem {
        fn name(&self) -> &str { "AOTProfileMatchSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::KernelGeneration }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
}
pub use tuning::{AOTProfileMatchSystem, AutoTuningSystem};

pub mod portfolio {
    use super::*;
    pub struct PortfolioSystem(pub prism_ecs_runtime::systems::portfolio::PortfolioSystem);
    impl CompilerSystem for PortfolioSystem {
        fn name(&self) -> &str { "PortfolioSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::KernelGeneration }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
}
pub use portfolio::PortfolioSystem;

pub mod kernel_catalog {
    use super::*;
    pub struct KernelCatalogSystem(pub prism_ecs_runtime::systems::kernel_catalog::KernelCatalogSystem);
    impl CompilerSystem for KernelCatalogSystem {
        fn name(&self) -> &str { "KernelCatalogSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::KernelGeneration }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
}
pub use kernel_catalog::KernelCatalogSystem;

pub mod variant_gen {
    use super::*;
    pub struct VariantGenerationSystem(pub prism_ecs_runtime::systems::variant_gen::VariantGenerationSystem);
    impl CompilerSystem for VariantGenerationSystem {
        fn name(&self) -> &str { "VariantGenerationSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::KernelGeneration }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
}
pub use variant_gen::VariantGenerationSystem;

pub mod variant_select {
    use super::*;
    pub struct VariantSelectionSystem(pub prism_ecs_runtime::systems::variant_select::VariantSelectionSystem);
    impl CompilerSystem for VariantSelectionSystem {
        fn name(&self) -> &str { "VariantSelectionSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::KernelGeneration }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
}
pub use variant_select::VariantSelectionSystem;

// ── Phase F: Compilation ────────────────────────────────────────

pub mod backend_compile {
    use super::*;
    pub struct BackendCompilationSystem(pub prism_ecs_runtime::systems::backend_compile::BackendCompilationSystem);
    impl CompilerSystem for BackendCompilationSystem {
        fn name(&self) -> &str { "BackendCompilationSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::Compilation }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
    pub struct ExecutableCachingSystem(pub prism_ecs_runtime::systems::backend_compile::ExecutableCachingSystem);
    impl CompilerSystem for ExecutableCachingSystem {
        fn name(&self) -> &str { "ExecutableCachingSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::Compilation }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
}
pub use backend_compile::{BackendCompilationSystem, ExecutableCachingSystem};

pub mod validation {
    use super::*;
    pub struct ExecutablePackagingSystem(pub prism_ecs_runtime::systems::validation::ExecutablePackagingSystem);
    impl CompilerSystem for ExecutablePackagingSystem {
        fn name(&self) -> &str { "ExecutablePackagingSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::Compilation }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
    pub struct AdmissionValidationSystem(pub prism_ecs_runtime::systems::validation::AdmissionValidationSystem);
    impl CompilerSystem for AdmissionValidationSystem {
        fn name(&self) -> &str { "AdmissionValidationSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::Compilation }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
}
pub use validation::{AdmissionValidationSystem, ExecutablePackagingSystem};

pub mod catalog_validation {
    use super::*;
    pub struct CatalogValidationSystem(pub prism_ecs_runtime::systems::catalog_validation::CatalogValidationSystem);
    impl CompilerSystem for CatalogValidationSystem {
        fn name(&self) -> &str { "CatalogValidationSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::Compilation }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
}
pub use catalog_validation::CatalogValidationSystem;

pub mod ternary_pipeline {
    use super::*;
    pub struct TertiaryPipelineSystem(pub prism_ecs_runtime::systems::ternary_pipeline::TertiaryPipelineSystem);
    impl CompilerSystem for TertiaryPipelineSystem {
        fn name(&self) -> &str { "TertiaryPipelineSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::Compilation }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
}
pub use ternary_pipeline::TertiaryPipelineSystem;

// ── Phase G: Packaging ──────────────────────────────────────────

pub mod package {
    use super::*;
    pub struct CImageAssemblySystem(pub prism_ecs_runtime::systems::package::CImageAssemblySystem);
    impl CompilerSystem for CImageAssemblySystem {
        fn name(&self) -> &str { "CImageAssemblySystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::Packaging }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
    pub struct ReceiptSigningSystem(pub prism_ecs_runtime::systems::package::ReceiptSigningSystem);
    impl CompilerSystem for ReceiptSigningSystem {
        fn name(&self) -> &str { "ReceiptSigningSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::Packaging }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
}
pub use package::{CImageAssemblySystem, ReceiptSigningSystem};

pub mod metal_cleanup {
    use super::*;
    pub struct MetalCleanupSystem(pub prism_ecs_runtime::systems::metal_cleanup::MetalCleanupSystem);
    impl CompilerSystem for MetalCleanupSystem {
        fn name(&self) -> &str { "MetalCleanupSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::Packaging }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
}
pub use metal_cleanup::MetalCleanupSystem;

pub mod phase_engine_cleanup {
    use super::*;
    pub struct PhaseEngineCleanupSystem(pub prism_ecs_runtime::systems::phase_engine_cleanup::PhaseEngineCleanupSystem);
    impl CompilerSystem for PhaseEngineCleanupSystem {
        fn name(&self) -> &str { "PhaseEngineCleanupSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::Packaging }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
}
pub use phase_engine_cleanup::PhaseEngineCleanupSystem;

pub mod session_cleanup {
    use super::*;
    pub struct SessionCleanupSystem(pub prism_ecs_runtime::systems::session_cleanup::SessionCleanupSystem);
    impl CompilerSystem for SessionCleanupSystem {
        fn name(&self) -> &str { "SessionCleanupSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::Packaging }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
}
pub use session_cleanup::SessionCleanupSystem;

// ── Phase H: Validation ────────────────────────────────────────

pub mod validation_matrix {
    use super::*;
    pub struct ValidationMatrixSystem(pub prism_ecs_runtime::systems::validation_matrix::ValidationMatrixSystem);
    impl CompilerSystem for ValidationMatrixSystem {
        fn name(&self) -> &str { "ValidationMatrixSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::Validation }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
}
pub use validation_matrix::ValidationMatrixSystem;

pub mod backend_eval {
    use super::*;
    pub struct BackendEvalSystem(pub prism_ecs_runtime::systems::backend_eval::BackendEvalSystem);
    impl CompilerSystem for BackendEvalSystem {
        fn name(&self) -> &str { "BackendEvalSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::Validation }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
}
pub use backend_eval::BackendEvalSystem;

pub mod backend_dispatch {
    use super::*;
    pub struct BackendDispatchSystem(pub prism_ecs_runtime::systems::backend_dispatch::BackendDispatchSystem);
    impl CompilerSystem for BackendDispatchSystem {
        fn name(&self) -> &str { "BackendDispatchSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::Validation }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
}
pub use backend_dispatch::BackendDispatchSystem;

// ── Phase I: Execution ─────────────────────────────────────────

pub mod metal_transfer {
    use super::*;
    pub struct MetalTransferSystem(pub prism_ecs_runtime::systems::metal_transfer::MetalTransferSystem);
    impl CompilerSystem for MetalTransferSystem {
        fn name(&self) -> &str { "MetalTransferSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::Execution }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
}
pub use metal_transfer::MetalTransferSystem;

pub mod metal_dispatch {
    use super::*;
    pub struct MetalDispatchSystem(pub prism_ecs_runtime::systems::metal_dispatch::MetalDispatchSystem);
    impl CompilerSystem for MetalDispatchSystem {
        fn name(&self) -> &str { "MetalDispatchSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::Execution }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
}
pub use metal_dispatch::MetalDispatchSystem;

pub mod phase_engine_tick {
    use super::*;
    pub struct PhaseEngineTickSystem(pub prism_ecs_runtime::systems::phase_engine_tick::PhaseEngineTickSystem);
    impl CompilerSystem for PhaseEngineTickSystem {
        fn name(&self) -> &str { "PhaseEngineTickSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::Execution }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
}
pub use phase_engine_tick::PhaseEngineTickSystem;

pub mod session_decode_tick {
    use super::*;
    pub struct SessionDecodeTickSystem(pub prism_ecs_runtime::systems::session_decode_tick::SessionDecodeTickSystem);
    impl CompilerSystem for SessionDecodeTickSystem {
        fn name(&self) -> &str { "SessionDecodeTickSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::Execution }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
}
pub use session_decode_tick::SessionDecodeTickSystem;

pub mod work_dispatch_tick {
    use super::*;
    pub struct WorkDispatchTickSystem(pub prism_ecs_runtime::systems::work_dispatch_tick::WorkDispatchTickSystem);
    impl CompilerSystem for WorkDispatchTickSystem {
        fn name(&self) -> &str { "WorkDispatchTickSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::Execution }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
}
pub use work_dispatch_tick::WorkDispatchTickSystem;

pub mod backpressure_tick {
    use super::*;
    pub struct BackpressureTickSystem(pub prism_ecs_runtime::systems::backpressure_tick::BackpressureTickSystem);
    impl CompilerSystem for BackpressureTickSystem {
        fn name(&self) -> &str { "BackpressureTickSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::Execution }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
}
pub use backpressure_tick::BackpressureTickSystem;

pub mod token_budget_tick {
    use super::*;
    pub struct TokenBudgetTickSystem(pub prism_ecs_runtime::systems::token_budget_tick::TokenBudgetTickSystem);
    impl CompilerSystem for TokenBudgetTickSystem {
        fn name(&self) -> &str { "TokenBudgetTickSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::Execution }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
}
pub use token_budget_tick::TokenBudgetTickSystem;

pub mod phase_engine {
    use super::*;
    pub struct PhaseEngineSystem(pub prism_ecs_runtime::systems::phase_engine::PhaseEngineSystem);
    impl CompilerSystem for PhaseEngineSystem {
        fn name(&self) -> &str { "PhaseEngineSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::Execution }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
}
pub use phase_engine::PhaseEngineSystem;

pub mod work_dispatch {
    use super::*;
    pub struct WorkDispatchSystem(pub prism_ecs_runtime::systems::work_dispatch::WorkDispatchSystem);
    impl CompilerSystem for WorkDispatchSystem {
        fn name(&self) -> &str { "WorkDispatchSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::Execution }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
}
pub use work_dispatch::WorkDispatchSystem;

pub mod completion_ingest {
    use super::*;
    pub struct CompletionIngestSystem(pub prism_ecs_runtime::systems::completion_ingest::CompletionIngestSystem);
    impl CompilerSystem for CompletionIngestSystem {
        fn name(&self) -> &str { "CompletionIngestSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::Execution }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
}
pub use completion_ingest::CompletionIngestSystem;

pub mod slot_lease_tick {
    use super::*;
    pub struct SlotLeaseTickSystem(pub prism_ecs_runtime::systems::slot_lease_tick::SlotLeaseTickSystem);
    impl CompilerSystem for SlotLeaseTickSystem {
        fn name(&self) -> &str { "SlotLeaseTickSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::Execution }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
}
pub use slot_lease_tick::SlotLeaseTickSystem;

pub mod executor_systems {
    use super::*;
    pub struct ExecutorSystem(pub prism_ecs_runtime::systems::executor_systems::ExecutorSystem);
    impl CompilerSystem for ExecutorSystem {
        fn name(&self) -> &str { "ExecutorSystem" }
        fn phase(&self) -> SchedulePhase { SchedulePhase::Execution }
        fn run(&self, _world: &mut World) -> anyhow::Result<()> { Ok(()) }
    }
}
pub use executor_systems::ExecutorSystem;
