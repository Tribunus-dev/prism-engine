//! Constitutional surface tests for the `systems` module.
//!
//! These tests verify that every sub-module compiles, the
//! `prism_ecs_runtime::systems::*` surface re-exports the data types
//! the engine's `compile_session.rs` and other engine callers
//! reference, and the data types are constructible where the engine
//! callers use `::new()` or `::default()`.
//!
//! # Migration status
//!
//! These tests were added when the engine's `system/` subsystem was
//! being absorbed into the constitutional surface. After the engine's
//! `compute-core/src/ecs/system/` directory is deleted, these tests
//! are the only remaining surface test for the data types.

#![allow(clippy::default_constructed_unit_structs)]

use crate::systems::*;

#[test]
fn unit_systems_construct() {
    // Unit structs the engine's compile_session.rs uses directly.
    let _ = metal_init::MetalInitSystem;
    let _ = phase_engine_init::PhaseEngineInitSystem;
    let _ = session_init::SessionInitSystem;
    let _ = download::DownloadSystem;
    let _ = download::HfSourceParsingSystem;
    let _ = archive::ArchiveSystem;
    let _ = int4_pack::Int4PackSystem;
    let _ = quant_plan::CodecSelectionSystem;
    let _ = quant_plan::PrecisionPlanSystem;
    let _ = moe_budget::MoERoutingSystem;
    let _ = moe_budget::MemoryBudgetSystem;
    let _ = memory_plan::MemoryDomainAssignmentSystem;
    let _ = memory_plan::BufferAllocationSystem;
    let _ = backend_residency::BackendResidencySystem;
    let _ = compiler_systems::GraphEqualizationSystem;
    let _ = compiler_systems::GraphOptimizerSystem;
    let _ = compiler_systems::CompileScheduleSystem;
    let _ = compiler_systems::BackendAssessmentSystem;
    let _ = catalog_validation::CatalogValidationSystem;
    let _ = execution_graph::ExecutionGraphSystem;
    let _ = capability_registry_sys::CapabilityRegistrySystem;
    let _ = kernel_catalog::KernelCatalogSystem;
    let _ = variant_gen::VariantGenerationSystem;
    let _ = variant_select::VariantSelectionSystem;
    let _ = backend_compile::ExecutableCachingSystem;
    let _ = validation::ExecutablePackagingSystem;
    let _ = validation::AdmissionValidationSystem;
    let _ = validation_matrix::ValidationMatrixSystem;
    let _ = backend_eval::BackendEvalSystem;
    let _ = backend_dispatch::BackendDispatchSystem;
    let _ = metal_transfer::MetalTransferSystem;
    let _ = metal_dispatch::MetalDispatchSystem;
    let _ = phase_engine_tick::PhaseEngineTickSystem;
    let _ = session_decode_tick::SessionDecodeTickSystem;
    let _ = work_dispatch_tick::WorkDispatchTickSystem;
    let _ = backpressure_tick::BackpressureTickSystem;
    let _ = token_budget_tick::TokenBudgetTickSystem;
    let _ = phase_engine::PhaseEngineSystem;
    let _ = work_dispatch::WorkDispatchSystem;
    let _ = completion_ingest::CompletionIngestSystem;
    let _ = slot_lease_tick::SlotLeaseTickSystem;
    let _ = executor_systems::ExecutorSystem;
    let _ = metal_cleanup::MetalCleanupSystem;
    let _ = phase_engine_cleanup::PhaseEngineCleanupSystem;
    let _ = session_cleanup::SessionCleanupSystem;
}

#[test]
fn struct_literal_systems_construct() {
    // Structs the engine's compile_session.rs uses with struct-literal syntax.
    let _ = source_load::SourceLoadingSystem;
    let _ = source_load::TensorTableLoadingSystem;
    let _ = source_load::SourceTensorMeta;
    let _ = source_load::DiffSystem;
    let _ = source_load::TensorTableComp;
    let _ = archive::PrecompiledAneSystem;
    let _ = draft_model::DraftModelSystem;
    let _ = tts::TTSSystem;
    let _ = portfolio::PortfolioSystem;
    let _ = package::CImageAssemblySystem;
    let _ = package::ReceiptSigningSystem;
}

#[test]
fn engine_systems_construct() {
    let _ = engine_systems::EngineInitSystem;
    let _ = engine_systems::ModelInstallSystem;
    let _ = engine_systems::ModelLoadSystem;
    let _ = engine_systems::CimageLoadSystem;
    let _ = engine_systems::HostInferenceInitSystem;
    let _ = engine_systems::ModelUnloadSystem;
    let _ = engine_systems::EngineMetricsSystem;
    let _ = engine_systems::EngineShutdownSystem;
    let _ = engine_systems::GenerationRequestSystem;
    let _ = engine_systems::CimageGenerateSystem;
    let _ = engine_systems::InferenceCycleSystem;
    let _ = engine_systems::TokenBudgetInferenceSystem;
    let _ = engine_systems::CancelSystem;
    let _ = engine_systems::MemoryPressureSystem;
    let _ = engine_systems::CimageLoadRequest;
}

#[test]
fn kernel_gen_construct() {
    let _ = kernel_gen::TemplateSelectionSystem;
    let _ = kernel_gen::ParameterResolutionSystem;
    let _ = kernel_gen::TemplateExpansionSystem::new();
    let _ = kernel_gen::TemplateExpansionSystem::default();
    let _ = kernel_gen::TemplateExpander::new();
}

#[test]
fn planning_core_construct() {
    let _ = planning_core::AneEligibilitySystem;
    let _ = planning_core::MemoryBudgetSystemV2;
    let _ = planning_core::RegionPlannerSystem;
    let _ = planning_core::RegionCatalogueSystem;
    let _ = planning_core::ReceiptSystem;
    let _ = planning_core::MemoryBudget;
    let _ = planning_core::MemoryPlan;
    let _ = planning_core::RegionKind;
    let _ = planning_core::SpillPolicy;
}

#[test]
fn buffer_lifetime_construct() {
    let _ = buffer_lifetime::LifetimeAnalysisSystem::new();
    let _ = buffer_lifetime::ScratchPlanningSystem::default();
}

#[test]
fn model_load_construct() {
    let _ = model_load::ModelAdapterSystem;
}

#[test]
fn tuning_construct() {
    let _ = tuning::AutoTuningSystem;
    let _ = tuning::AOTProfileMatchSystem;
}

#[test]
fn fusion_analysis_construct() {
    let _ = fusion::analysis::FusionAnalysisSystem;
    let _ = fusion::heuristic::FusionHeuristicSystem::default();
    let _ = fusion::dispatch::DispatchFormationSystem;
    let _ = fusion::scalar::ScalarDispatchSystem;
}

#[test]
fn ternary_pipeline_construct() {
    let _ = ternary_pipeline::TertiaryPipelineSystem::new();
}

#[test]
fn backend_compile_default_construct() {
    let _ = backend_compile::BackendCompilationSystem::default();
}
