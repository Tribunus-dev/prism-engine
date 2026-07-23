//! Canonical compiler orchestration — format-independent compilation pipeline.
//!
//! This crate defines the contract types for the prism-engine compilation
//! pipeline. Every format-specific compiler (GGUF, Safetensors, ONNX, etc.)
//! implements its compilation path against these interfaces.
//!
//! # Phase 1 — Contract Types
//!
//! All types are [`Serialize`] + [`Deserialize`] where they cross crate
//! boundaries. Error types use [`thiserror`]. Timestamps use [`chrono`].
//! Identifiers use [`uuid`] for ephemeral request IDs and SHA-256 digests
//! for durable artifact identities.
//!
//! # Phase 4 — Orchestration
//!
//! The [`CanonicalCompiler::compile`] method delegates to the full stage
//! runner that wires source detection,
//! tensor ingestion, graph construction, evolutionary search, measurement,
//! legalization, kernel generation, CImage emission, certification, and
//! receipt building.
//!
//! ```text
//! Source
//!   │ detect_source
//!   ▼
//! CanonicalSource
//!   │ ingest_tensors
//!   ▼
//! TensorProvider stream
//!   │ build_graph
//!   ▼
//! SpatialGraph + TensorCodec bindings
//!   │ run_search           ─── SearchTrace
//!   ▼
//! ScoredCandidate[]
//!   │ measure_candidates   ─── CandidateMeasurements
//!   ▼
//! Selected candidate
//!   │ legalize             ─── LegalizedGraph
//!   ▼
//! Legalized graph
//!   │ generate_kernels     ─── KernelArtifact
//!   ▼
//! Compiled kernels
//!   │ emit_cimage          ─── CImage
//!   ▼
//! CImage artifact
//!   │ certify              ─── Certificate
//!   ▼
//! Certified artifact
//!   │ build_receipt        ─── CompileReceipt
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

// Re-exports for downstream convenience.
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

pub mod evaluator;
pub mod search;
pub use search::{EvaluationStrategy, SearchCoordinator, SearchError, SearchResult};

pub mod legalize;
pub use legalize::{
    apply_legalization, validate_fusion_legality, validate_kernel_bindings,
    validate_memory_constraints, validate_plan, validate_precision_compatibility,
    validate_tensor_layouts, validate_tile_geometry, BindingCheck, CompilerLegalizer, FusionCheck,
    LayoutCheck, LegalizationError, LegalizationReport, MemoryCheck, PlanCheck, PrecisionCheck,
    TileCheck,
};

// ---------------------------------------------------------------------------
// CImage emission
// ---------------------------------------------------------------------------

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
    compile_ecs_op_to_xdna_cimage, compile_int8_ane_tile_to_cimage, compile_path,
    compile_path_with_backend, compile_source, compile_source_ecs, detect_source,
};
pub use uop::{
    benchmark_uop_graph_strategies, benchmark_uop_graph_strategies_with_runner,
    benchmark_uop_graph_workloads, benchmark_uop_graph_workloads_with_runner,
    classify_custom_operation, compile_and_validate_uop_capture,
    compile_and_validate_uop_graph_strategies, compile_spatial_graph,
    compile_spatial_graph_strategies, compile_spatial_matmul, compile_spatial_node,
    compile_spatial_node_with_metadata, compile_uop_capture, compile_uop_graph_strategies,
    compile_uop_graph_with_strategy, execute_uop_reference, select_measured_uop_strategy,
    select_measured_uop_workloads, validate_and_classify_custom_operation, CustomOperationClass,
    UOpCompileCache, UOpCompiledProgram, UOpDispatchResult, UOpWorkloadMeasurement,
    UOpWorkloadSelection, CUSTOM_OPERATION_CANDIDATES, VALIDATED_CUSTOM_OPERATIONS,
};

// ---------------------------------------------------------------------------
// Forensic observability module
// ---------------------------------------------------------------------------

pub mod forensic;
pub use forensic::{build_forensic_receipt, create_event, load_events_from_file, FileEventSink};

// ---------------------------------------------------------------------------
// ECS compilation components and pipeline orchestration
// ---------------------------------------------------------------------------

pub mod ecs;
pub use ecs::{
    CImageArtifact, CompilationOrchestrator, CompilationSession, KernelCollection, LegalizedPlan,
    SearchStateComponent, SessionStatus, SourceModel, SpatialGraphComponent, TensorCollection,
};

pub mod compilation_entity;
pub use compilation_entity::{CompilationEntity, CompilationStatus};

pub mod compilation_systems;
pub use compilation_systems::*;

// ---------------------------------------------------------------------------
// Unified runtime
// ---------------------------------------------------------------------------

pub mod runtime;
pub use runtime::{CImageXdnaRouteDispatcher, ExecutionMode, RuntimeError, RuntimeModel};

// ---------------------------------------------------------------------------
// Legacy compatibility layer
// ---------------------------------------------------------------------------

pub mod legacy;
pub use legacy::{compile_gguf_compat, compile_to_cimage_compat, compile_with_autodetect};

// ---------------------------------------------------------------------------
// Top-level types
// ---------------------------------------------------------------------------

/// Compilation stage identifier.
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
            Self::Certification => write!(f, "certification"),
        }
    }
}

/// Result of a single compilation stage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageResult {
    pub stage: CompilationStage,
    pub success: bool,
    pub duration_ms: u64,
    pub error: Option<String>,
}

/// Event kind for compilation observability.
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

/// Compilation event for forensic tracing.
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

/// Compilation configuration.
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

/// Calibration policy for quantization.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CalibrationPolicy {
    None,
    FromFile(String),
    Auto,
}

/// Validation policy for compilation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ValidationPolicy {
    Structural,
    Production,
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
    /// Fraction of warm-start offspring sent to hardware measurement after
    /// the contextual surrogate has enough evidence.
    #[serde(default = "default_surrogate_measurement_fraction")]
    pub surrogate_measurement_fraction: f64,
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
pub enum CandidateStatus {
    Evaluated,
    Rejected,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateMeasurements {
    pub wall_time_ms: f64,
    pub gpu_time_ms: f64,
    pub bandwidth_gbps: f64,
    pub peak_memory_mb: f64,
    pub reconstruction_error: f64,
    pub accuracy_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateRecord {
    pub candidate_digest: String,
    pub parent_digests: Vec<String>,
    pub genome: String,
    pub tensor_scope: Vec<String>,
    pub score_vector: Vec<f64>,
    pub measurements: Option<serde_json::Value>,
    pub status: CandidateStatus,
    pub rejection_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationRecord {
    pub generation: u32,
    pub candidates: Vec<CandidateRecord>,
    pub best_score: f64,
    pub diversity: f64,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchTrace {
    pub search_id: String,
    pub config: SearchConfig,
    pub generations: Vec<GenerationRecord>,
    pub pareto_frontier: Vec<CandidateRecord>,
    /// Measured quality-diversity elites retained across compiler sessions.
    /// The legacy scalar frontier remains for compatibility, while this map
    /// is the authoritative multi-objective search artifact.
    #[serde(default)]
    pub quality_diversity_archive: Vec<prism_ecs_ir::evolution::objectives::ArchiveEntry>,
    pub best_genome: Option<String>,
    pub trace_digest: String,
}

/// Compilation status.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompileStatus {
    Pending,
    InProgress,
    Completed,
    Failed(String),
    Partial(Vec<StageResult>),
}

/// Compilation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompileResult {
    pub receipt: CompileReceipt,
    pub status: CompileStatus,
    pub request_id: Uuid,
    pub events: Vec<CompilationEvent>,
    pub output_digest: String,
    pub output_path: std::path::PathBuf,
}

/// Compilation receipt (forensic artifact).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompileReceipt {
    pub receipt_id: String,
    pub request_id: Uuid,
    pub compiler_identity: CompilerIdentity,
    pub source_identity: SourceIdentity,
    pub started_at: DateTime<Utc>,
    pub completed_at: DateTime<Utc>,
    pub finished_at: DateTime<Utc>,
    pub duration_ms: u64,
    pub stages: Vec<StageResult>,
    pub candidate_count: u32,
    pub generations: u32,
    pub output_digest: String,
    pub output_path: std::path::PathBuf,
    pub schema_version: String,
    pub status: CompileStatus,
    pub error: Option<String>,
    pub source_digest: Option<String>,
    pub graph_digest: Option<String>,
    pub search_trace_digest: Option<String>,
    pub kernel_manifest_digest: Option<String>,
    pub events_digest: Option<String>,
    pub legalization_mode: Option<String>,
}

impl Default for CompileReceipt {
    fn default() -> Self {
        Self {
            receipt_id: String::new(),
            request_id: Uuid::nil(),
            compiler_identity: CompilerIdentity {
                name: String::new(),
                version: String::new(),
                build_hash: None,
                build_timestamp: None,
            },
            source_identity: SourceIdentity {
                format: prism_ecs_core::identity::SourceFormat::Raw,
                source_digest: String::new(),
                architecture: String::new(),
                model_family: String::new(),
            },
            started_at: Utc::now(),
            completed_at: Utc::now(),
            finished_at: Utc::now(),
            duration_ms: 0,
            stages: Vec::new(),
            candidate_count: 0,
            generations: 0,
            output_digest: String::new(),
            output_path: std::path::PathBuf::new(),
            schema_version: String::new(),
            status: CompileStatus::Pending,
            error: None,
            source_digest: None,
            graph_digest: None,
            search_trace_digest: None,
            kernel_manifest_digest: None,
            events_digest: None,
            legalization_mode: None,
        }
    }
}

/// Compilation error.
#[derive(Debug, Error)]
pub enum CompileError {
    #[error("Policy violation: {0}")]
    PolicyViolation(String),
    #[error("Source detection failed: {0}")]
    SourceDetectionFailed(String),

    #[error("Source ingestion failed: {0}")]
    SourceIngestionFailed(String),

    #[error("Graph build failed: {0}")]
    GraphBuildFailed(String),

    #[error("Search failed: {0}")]
    SearchFailed(String),

    #[error("Legalization failed: {0}")]
    LegalizationFailed(String),

    #[error("Kernel generation failed: {0}")]
    KernelGenFailed(String),

    #[error("CImage emission failed: {0}")]
    CImageEmitFailed(String),

    #[error("Compilation failed: {0}")]
    CompilationFailed(String),

    #[error("Session not found")]
    SessionNotFound,

    #[error("Invalid state transition: {0}")]
    InvalidStateTransition(String),
}

impl From<prism_ecs_core::WorldError> for CompileError {
    fn from(error: prism_ecs_core::WorldError) -> Self {
        Self::CompilationFailed(error.to_string())
    }
}

/// Event sink trait for compilation observability.
pub trait CompilationEventSink: Send + Sync {
    fn emit(&mut self, event: CompilationEvent) -> Result<(), String>;
    fn events(&self) -> Vec<CompilationEvent>;
}

/// Vector-based event sink for testing.
#[derive(Debug, Default)]
pub struct VecEventSink {
    events: std::sync::Arc<std::sync::Mutex<Vec<CompilationEvent>>>,
}

impl VecEventSink {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn stage_completed(
        &mut self,
        _stage: CompilationStage,
        _duration_ms: u64,
        _detail: &str,
    ) -> Result<(), String> {
        Ok(())
    }
    pub fn stage_failed(
        &mut self,
        _stage: CompilationStage,
        _duration_ms: u64,
        _detail: &str,
    ) -> Result<(), String> {
        Ok(())
    }
}

impl CompilationEventSink for VecEventSink {
    fn emit(&mut self, event: CompilationEvent) -> Result<(), String> {
        self.events.lock().unwrap().push(event);
        Ok(())
    }

    fn events(&self) -> Vec<CompilationEvent> {
        self.events.lock().unwrap().clone()
    }
}

/// Canonical compiler orchestrator.
pub struct CanonicalCompiler {
    pub config: CompileConfig,
    pub output_path: Option<std::path::PathBuf>,
    /// Original model directory, retained so emission can consume resumable
    /// native tensor cache records rather than silently falling back to BF16.
    pub source_path: Option<std::path::PathBuf>,
    /// Model-level configuration carried from a model directory into CImage
    /// emission when the source is Qwen3.6-MoE.
    pub qwen36_config: Option<qwen3_6_moe::Qwen36Config>,
    /// Model-neutral family adapter used by graph, search, and promotion.
    pub model_adapter: Option<std::sync::Arc<dyn ModelAdapter>>,
    pub event_sink: Option<Box<dyn CompilationEventSink>>,
    #[cfg(feature = "phase4_evaluation")]
    pub evaluator: Option<Box<dyn EvaluationStrategy>>,
}

impl CanonicalCompiler {
    pub fn new(config: CompileConfig) -> Self {
        Self {
            config,
            output_path: None,
            source_path: None,
            qwen36_config: None,
            model_adapter: None,
            event_sink: None,
            #[cfg(feature = "phase4_evaluation")]
            evaluator: None,
        }
    }

    /// Compile an already-ingested canonical source through the unified
    /// compiler pipeline. Keeping this entry point on the compiler object
    /// makes configuration, output selection, and event sinks travel through
    /// one API instead of requiring callers to know the free function.
    pub fn compile(&mut self, source: CanonicalSource) -> Result<CompileResult, CompileError> {
        compiler::compile_source(self, source)
    }
}

impl Default for CanonicalCompiler {
    fn default() -> Self {
        Self::new(CompileConfig {
            production_mode: false,
            max_candidates: 100,
            max_generations: 5,
            max_search_time_ms: 300_000,
            target_backends: vec![BackendKind::Metal],
            calibration_policy: CalibrationPolicy::None,
            validation_policy: ValidationPolicy::Structural,
            enable_search: true,
            enable_legalization: true,
            enable_kernel_gen: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calibration_policy_serialization() {
        let policy = CalibrationPolicy::None;
        let serialized = serde_json::to_string(&policy).unwrap();
        let deserialized: CalibrationPolicy = serde_json::from_str(&serialized).unwrap();
        match deserialized {
            CalibrationPolicy::None => {}
            _ => panic!("Expected CalibrationPolicy::None"),
        }
    }

    #[test]
    fn test_validation_policy_serialization() {
        let policy = ValidationPolicy::Structural;
        let serialized = serde_json::to_string(&policy).unwrap();
        let deserialized: ValidationPolicy = serde_json::from_str(&serialized).unwrap();
        match deserialized {
            ValidationPolicy::Structural => {}
            _ => panic!("Expected ValidationPolicy::Structural"),
        }
    }

    #[test]
    fn test_compile_status_serialization() {
        let status = CompileStatus::Completed;
        let serialized = serde_json::to_string(&status).unwrap();
        let deserialized: CompileStatus = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized, CompileStatus::Completed);
    }
}
