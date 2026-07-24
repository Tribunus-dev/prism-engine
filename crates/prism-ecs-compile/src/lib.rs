//! Canonical compiler orchestration — format-independent compilation pipeline.
//!
//! This crate defines the contract types for the prism-engine compilation
//! pipeline. Every format-specific compiler (GGUF, Safetensors, ONNX, etc.)
//! implements its compilation path against these interfaces.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

pub use prism_ecs_core::identity::CompilerIdentity;
pub use prism_ecs_kernel::BackendKind;
pub use prism_ecs_source::SourceIdentity;
pub use prism_ecs_source::{CanonicalSource, SourceCapabilities};

pub mod graph;
pub use graph::{CanonicalGraphBuilder, GraphBuildError, GraphBuildResult};

pub mod model;
pub mod qwen3_6_moe;
pub use model::{
    adapter_for_model_dir, classify_tensor, DenseTransformerAdapter, ModelAdapter,
    TensorDescriptor, TensorRole,
};
pub use qwen3_6_moe::{
    classify_qwen36_tensor, MappedLayerStream, Qwen36Config, Qwen36MappedLayerStream,
    Qwen36TensorDescriptor, Qwen36TensorRole,
};

pub mod active_window;
pub mod adapter_training;
pub mod agentic_workload;
pub mod engram_learning;
pub mod evaluator;
pub mod execution_graph_evolution;
pub mod knowledge_correction;
pub mod kv_cache_compaction;
pub mod kv_cache_evolution;
pub mod living_cimage;
pub mod living_ecs;
pub mod living_promotion;
pub mod model_export;
pub mod progressive_ternary;
pub mod representation_cache;
pub mod search;
pub mod semantic_region_discovery;
pub mod semantic_region_evaluation;
pub mod semantic_region_manifest;
pub mod semantic_region_probe;
pub mod semantic_region_search;
pub mod semantic_region_spec;
pub mod shadow_calibration;
pub mod speculative_inference;
pub mod workload_search;

pub use adapter_training::{
    admit_adapter, AdapterAdmissionPolicy, AdapterArtifact, AdapterKind, AdapterTrainingError,
    AdapterTrainingReceipt, AdapterTrainingRequest,
};
pub use agentic_workload::{
    AgenticCalibrationCorpus, AgenticOutcome, AgenticWorkloadClass, AgenticWorkloadEpisode,
    AgenticWorkloadError, EngramAccessEvent, PrivacyClass, RetentionPolicy,
};
pub use engram_learning::{
    admit_engram_generation, EngramAdmissionPolicy, EngramAdmissionReceipt, EngramEntry,
    EngramEvaluation, EngramGeneration, EngramLearningError, EngramRepresentation,
    EngramRouterPolicy,
};
pub use execution_graph_evolution::{
    admit_execution_graph, mutate_execution_graph, ExecutableUnitCandidate, ExecutableUnitKind,
    ExecutionGraphAdmissionPolicy, ExecutionGraphAdmissionReceipt, ExecutionGraphEdge,
    ExecutionGraphError, ExecutionGraphMeasurement, ExecutionGraphMutation, TargetExecutionGraph,
};
pub use knowledge_correction::{
    admit_knowledge_correction, CorrectionMechanism, KnowledgeCorrectionError,
    KnowledgeCorrectionEvaluation, KnowledgeCorrectionPolicy, KnowledgeCorrectionProposal,
    KnowledgeCorrectionReceipt, ModelErrorClass, ObservedModelError, RefreshPolicy,
    TemporalKnowledgeContract,
};
pub use kv_cache_compaction::{
    evaluate_kv_compaction, propose_compaction_candidates, KvCompactionAdmissionPolicy,
    KvCompactionAlgorithm, KvCompactionCandidate, KvCompactionError, KvCompactionMeasurement,
    KvCompactionReceipt,
};
pub use kv_cache_evolution::*;
pub use living_cimage::{
    AdaptationKind, AdaptationLifecycle, CImageGeneration, LivingCImage, LivingCImageError,
    LivingCImageGeneration, LivingCImageId,
};
pub use living_ecs::{
    event_from_command, project_living_cimage, validate_command_against_authority,
    LivingAdaptationIndex, LivingCalibrationAuthority, LivingCImageAuthority, LivingCommand,
    LivingCommandKind, LivingEcsError, LivingEntityKind, LivingEvent, LivingGenerationAuthority,
    LivingLifecycle, LivingLifecycleComponent, LivingProjection, LivingReplayRegistry,
};
pub use living_promotion::{
    build_rollback_receipt, evaluate_promotion, DomainAdmissionEvidence,
    LivingGenerationPromotionPolicy, LivingGenerationPromotionReceipt,
    LivingGenerationPromotionRequest, LivingGenerationRollbackReceipt, LivingPromotionError,
    RefinementDomain,
};
pub use model_export::{
    validate_export, CanonicalModelFormat, EngramExportPolicy, ExportEquivalenceClass,
    ExportPrecisionPolicy, ExportedComponent, ModelExportError, ModelExportPolicy,
    ModelExportReceipt, ModelExportRequest, ModelExportValidation,
};
pub use progressive_ternary::*;
pub use search::{
    EvaluationStrategy, SearchCoordinator, SearchError, SearchResult, SearchSelectionReceipt,
};
pub use semantic_region_discovery::{
    discover_semantic_partition, ArchitectureDiscoverer, GraphExplicitDiscoverer, GraphRegionHint,
    LogicalTensorDescriptor, SemanticDiscoveryError, SemanticModelConfig,
    SemanticRegionDiscoverer,
};
pub use semantic_region_evaluation::{
    SemanticRegionAblation, SemanticRegionBaseline, SemanticRegionEvaluationMetrics,
    SemanticRegionEvaluationRecord, SemanticRegionEvaluationStudy,
};
pub use semantic_region_manifest::{
    SemanticRegionManifest, SemanticRegionManifestError, SEMANTIC_REGION_MANIFEST_V1,
};
pub use semantic_region_probe::{
    selector_digest, MappedTensorRegionProbeContext, RegionProbeError, RegionSensitivityReceipt,
    RegionView,
};
pub use semantic_region_search::{
    build_palettes, enforce_regularization, objective_score, select_bounded_plan,
    RegionCandidatePalette, RegionRegularizationPolicy, RegionTemplateId, RegionalCandidate,
    RegionalSearchError, RegionalSearchObjectives,
};
pub use semantic_region_spec::{
    SemanticRegionDiscoveryReceipt, SemanticRegionSpec, SemanticRegionSpecEntry,
    SemanticRegionSpecError, SEMANTIC_REGION_SPEC_V1,
};
pub use shadow_calibration::{
    evaluate_shadow_candidate, ShadowCalibrationError, ShadowCalibrationPolicy,
    ShadowCalibrationReceipt, ShadowEpisodeComparison,
};
pub use speculative_inference::*;

pub mod legalize;
pub use legalize::{
    apply_legalization, validate_fusion_legality, validate_kernel_bindings,
    validate_memory_constraints, validate_plan, validate_precision_compatibility,
    validate_tensor_layouts, validate_tile_geometry, BindingCheck, CompilerLegalizer, FusionCheck,
    LayoutCheck, LegalizationError, LegalizationReport, MemoryCheck, PlanCheck, PrecisionCheck,
    TileCheck,
};

pub mod assembly;
pub mod cimage;
pub mod uop;
pub use assembly::{assemble, AssemblyModelSource, AssemblyReceipt, AssemblyRequest};
pub mod model_manifest;
pub use model_manifest::{
    HardwareCapabilities, ModelIoBinding, ModelIoKind, ModelManifest, ModelModality,
    ModelProjectorBinding, ModelRequirements, MultiModelManifest,
};
pub mod compiler;
pub use cimage::{emit_int8_ane_program, CImageError, TensorPayloadEntry, UniversalCImageWriter};
pub use compiler::{
    compile_ecs_op_to_xdna_cimage, compile_gguf_compat, compile_int8_ane_tile_to_cimage,
    compile_path, compile_path_with_backend, compile_source, compile_source_ecs,
    compile_to_cimage_compat, compile_with_autodetect, detect_source,
};
pub use uop::{
    benchmark_uop_graph_strategies, benchmark_uop_graph_strategies_with_runner,
    benchmark_uop_graph_workloads, benchmark_uop_graph_workloads_with_runner,
    benchmark_uop_strategy_candidates, classify_custom_operation, compile_and_validate_uop_capture,
    compile_and_validate_uop_graph_strategies, compile_spatial_graph,
    compile_spatial_graph_strategies, compile_spatial_matmul, compile_spatial_node,
    compile_spatial_node_with_metadata, compile_uop_capture, compile_uop_graph_strategies,
    compile_uop_graph_with_strategy, execute_uop_reference, select_measured_uop_strategy,
    select_measured_uop_workloads, validate_and_classify_custom_operation, CustomOperationClass,
    UOpCompileCache, UOpCompiledProgram, UOpDispatchResult, UOpMeasurementSource,
    UOpTuningCandidate, UOpTuningReceipt, UOpTuningScenario, UOpWorkloadMeasurement,
    UOpWorkloadSelection, CUSTOM_OPERATION_CANDIDATES, VALIDATED_CUSTOM_OPERATIONS,
};

pub mod forensic;
pub use forensic::{build_forensic_receipt, create_event, load_events_from_file, FileEventSink};

pub mod ecs;
pub use ecs::{
    CImageArtifact, CompilationOrchestrator, CompilationReceipt, CompilationSession,
    KernelCollection, LegalizedPlan, SearchStateComponent, SessionStatus, SourceModel,
    SpatialGraphComponent, TensorCollection,
};

pub mod compilation_entity;
pub use compilation_entity::{CompilationEntity, CompilationStatus};

pub mod compilation_systems;
pub use compilation_systems::*;

pub mod runtime;
pub use runtime::{CImageXdnaRouteDispatcher, ExecutionMode, RuntimeError, RuntimeModel};
pub mod observability;
pub use observability::{EcsCorrelation, EcsStateEvent, EcsStateSnapshot, EcsStateStream};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CompilationStage {
    SourceDetection,
    SourceIngestion,
    GraphConstruction,
    EvolutionarySearch,
    CandidateMeasurement,
    Legalization,
    TargetLowering,
    KernelGeneration,
    CImageEmission,
    ReceiptBuild,
    Certification,
    Certify,
}

impl std::fmt::Display for CompilationStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SourceDetection => write!(f, "source_detection"),
            Self::SourceIngestion => write!(f, "source_ingestion"),
            Self::GraphConstruction => write!(f, "graph_construction"),
            Self::EvolutionarySearch => write!(f, "evolutionary_search"),
            Self::CandidateMeasurement => write!(f, "candidate_measurement"),
            Self::Legalization => write!(f, "legalization"),
            Self::TargetLowering => write!(f, "target_lowering"),
            Self::KernelGeneration => write!(f, "kernel_generation"),
            Self::CImageEmission => write!(f, "cimage_emission"),
            Self::ReceiptBuild => write!(f, "receipt_build"),
            Self::Certification | Self::Certify => write!(f, "certification"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageResult {
    pub stage: CompilationStage,
    pub success: bool,
    pub duration_ms: u64,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EventKind {
    StageStarted,
    StageCompleted,
    Warning,
    Error,
    Info,
}

impl std::fmt::Display for EventKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StageStarted => write!(f, "stage_started"),
            Self::StageCompleted => write!(f, "stage_completed"),
            Self::Warning => write!(f, "warning"),
            Self::Error => write!(f, "error"),
            Self::Info => write!(f, "info"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilationEvent {
    pub sequence: u64,
    pub timestamp: DateTime<Utc>,
    pub phase: CompilationStage,
    pub event_type: EventKind,
    pub entity_id: Option<String>,
    pub duration_ms: u64,
    pub detail: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub digests: Vec<String>,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompileConfig {
    pub production_mode: bool,
    pub max_candidates: u32,
    pub max_generations: u32,
    pub max_search_time_ms: u64,
    pub target_backends: Vec<BackendKind>,
    pub calibration_policy: CalibrationPolicy,
    pub validation_policy: ValidationPolicy,
    pub enable_search: bool,
    pub enable_legalization: bool,
    pub enable_kernel_gen: bool,
}

impl Default for CompileConfig {
    fn default() -> Self {
        Self {
            production_mode: false,
            max_candidates: 20,
            max_generations: 1,
            max_search_time_ms: 300_000,
            target_backends: vec![BackendKind::CPU],
            calibration_policy: CalibrationPolicy::None,
            validation_policy: ValidationPolicy::Structural,
            enable_search: true,
            enable_legalization: true,
            enable_kernel_gen: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CalibrationPolicy {
    None,
    FromFile(String),
    Auto,
}
impl Default for CalibrationPolicy {
    fn default() -> Self {
        Self::Auto
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ValidationPolicy {
    Structural,
    Production,
}
impl Default for ValidationPolicy {
    fn default() -> Self {
        Self::Structural
    }
}

#[derive(Debug, Clone)]
pub struct CompilationPolicy {
    stages: Vec<CompilationStage>,
}

impl Default for CompilationPolicy {
    fn default() -> Self {
        Self {
            stages: vec![
                CompilationStage::SourceDetection,
                CompilationStage::GraphConstruction,
                CompilationStage::EvolutionarySearch,
                CompilationStage::Legalization,
                CompilationStage::KernelGeneration,
                CompilationStage::CImageEmission,
                CompilationStage::Certification,
                CompilationStage::ReceiptBuild,
            ],
        }
    }
}

impl CompilationPolicy {
    pub fn enabled_stages(&self) -> &[CompilationStage] {
        &self.stages
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchConfig {
    pub max_generations: u32,
    pub population_size: u32,
    pub mutation_rate: f64,
    pub crossover_rate: f64,
    pub tournament_size: u32,
    pub elite_count: u32,
    pub early_stop_generations: u32,
    pub production_mode: bool,
    #[serde(default = "default_surrogate_measurement_fraction")]
    pub surrogate_measurement_fraction: f64,
    #[serde(default)]
    pub min_quality: Option<f64>,
    #[serde(default)]
    pub max_p99_latency_ms: Option<f64>,
    #[serde(default)]
    pub max_peak_memory_bytes: Option<u64>,
}

fn default_surrogate_measurement_fraction() -> f64 {
    0.2
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            max_generations: 1,
            population_size: 20,
            mutation_rate: 0.1,
            crossover_rate: 0.7,
            tournament_size: 3,
            elite_count: 2,
            early_stop_generations: 10,
            production_mode: false,
            surrogate_measurement_fraction: default_surrogate_measurement_fraction(),
            min_quality: None,
            max_p99_latency_ms: None,
            max_peak_memory_bytes: None,
        }
    }
}

impl SearchConfig {
    pub fn effective_surrogate_measurement_fraction(&self) -> f64 {
        if self.surrogate_measurement_fraction.is_finite() {
            self.surrogate_measurement_fraction.clamp(0.01, 1.0)
        } else {
            default_surrogate_measurement_fraction()
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchTrace {
    pub search_id: String,
    pub config: SearchConfig,
    #[serde(default)]
    pub generations: Vec<GenerationRecord>,
    #[serde(default)]
    pub pareto_frontier: Vec<CandidateRecord>,
    #[serde(default)]
    pub quality_diversity_archive: Vec<prism_ecs_ir::evolution::ArchiveEntry>,
    pub best_genome: Option<String>,
    #[serde(default)]
    pub trace_digest: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CandidateStatus {
    Proposed,
    Evaluated,
    Rejected,
    Promoted,
}

impl Default for CandidateStatus {
    fn default() -> Self {
        Self::Proposed
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateRecord {
    pub candidate_digest: String,
    #[serde(default)]
    pub parent_digests: Vec<String>,
    pub genome: String,
    #[serde(default)]
    pub tensor_scope: Vec<String>,
    #[serde(default)]
    pub score_vector: Vec<f64>,
    pub measurements: Option<serde_json::Value>,
    pub status: CandidateStatus,
    pub rejection_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationRecord {
    pub generation: u32,
    #[serde(default)]
    pub candidates: Vec<CandidateRecord>,
    pub best_score: f64,
    pub diversity: f64,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Error)]
pub enum CompileError {
    #[error("source detection failed: {0}")]
    SourceDetectionFailed(String),
    #[error("source ingestion failed: {0}")]
    SourceIngestionFailed(String),
    #[error("graph build failed: {0}")]
    GraphBuildFailed(String),
    #[error("search failed: {0}")]
    SearchFailed(String),
    #[error("legalization failed: {0}")]
    LegalizationFailed(String),
    #[error("kernel generation failed: {0}")]
    KernelGenerationFailed(String),
    #[error("CImage emission failed: {0}")]
    CImageEmissionFailed(String),
    #[error("CImage emission failed: {0}")]
    CImageEmitFailed(String),
    #[error("receipt construction failed: {0}")]
    ReceiptBuildFailed(String),
    #[error("certification failed: {0}")]
    CertificationFailed(String),
    #[error("invalid configuration: {0}")]
    InvalidConfiguration(String),
    #[error("compilation failed: {0}")]
    CompilationFailed(String),
    #[error("policy violation: {0}")]
    PolicyViolation(String),
}

pub trait CompilationEventSink: Send + Sync {
    fn emit(&self, event: CompilationEvent);
}

#[derive(Debug, Default)]
pub struct VecEventSink {
    events: std::sync::Mutex<Vec<CompilationEvent>>,
}

impl VecEventSink {
    pub fn events(&self) -> Vec<CompilationEvent> {
        self.events.lock().expect("event sink lock").clone()
    }
}

impl CompilationEventSink for VecEventSink {
    fn emit(&self, event: CompilationEvent) {
        self.events.lock().expect("event sink lock").push(event);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompileReceipt {
    pub compilation_id: Uuid,
    pub compiler: CompilerIdentity,
    pub source: SourceIdentity,
    pub config: CompileConfig,
    pub stages: Vec<StageResult>,
    pub output_path: Option<String>,
    pub output_digest: Option<String>,
    pub status: CompileStatus,
    #[serde(default)]
    pub metadata: std::collections::BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompileStatus {
    Success,
    Failed,
    Cancelled,
}

#[derive(Debug)]
pub struct CompileResult {
    pub receipt: CompileReceipt,
    pub output_path: Option<std::path::PathBuf>,
}
