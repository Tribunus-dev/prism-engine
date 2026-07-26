//! CompilePlan, CompilerReceipt — the public compilation API types.
//!
//! These types define the request/response contract for the PrismCompiler.
//! All binary entry points, server endpoints, tests, and constitutional
//! commands call through this API.

use super::execution_graph::ExecutionGraph;
use super::kernel_abi::{CompiledKernelArtifact, KernelPlan};
use super::model_ir::ModelIr;
use super::representation::RepresentationPlan;
use serde::{Deserialize, Serialize};

/// A request to inspect (not compile) a model source.
#[derive(Debug, Clone)]
pub struct InspectRequest {
    pub source_path: String,
    pub source_type: Option<String>,
}

/// The result of inspecting a model source.
#[derive(Debug, Clone)]
pub struct ModelInspection {
    pub identity: super::model_ir::ModelIdentity,
    pub architecture: super::model_ir::ArchitectureId,
    pub configuration: super::model_ir::ModelConfiguration,
    pub tensor_count: usize,
    pub total_weight_bytes: u64,
}

/// A complete compilation request.
#[derive(Debug, Clone)]
pub struct CompileRequest {
    pub source_path: String,
    pub source_type: Option<String>,
    pub output_path: Option<String>,
    pub target_lanes: Vec<super::execution_graph::ExecutionLane>,
    pub policy_path: Option<String>,
    pub quant_mode: Option<String>,
    /// Authority mode: "sealed" | "test_fixture" | None (unchecked)
    pub authority: Option<String>,
    /// Path to a draft GGUF model for speculative decoding
    pub draft_path: Option<String>,
    /// Directory containing pre-compiled ANE .mlmodelc bundles.
    /// When set, the compiler skips MIL generation and uses these directly.
    pub ane_models_dir: Option<String>,
    /// Path to a pre-compiled .metallib file (Metal inference kernels).
    /// When set, the compiler embeds this library directly instead of
    /// compiling shader templates.
    pub metallib_path: Option<String>,
    /// Directory containing MLX JIT-captured Metal source (generated.metal)
    /// for AOT compilation. When set and generated.metal exists, the
    /// compiler uses it instead of template kernels.
    pub mlx_capture_dir: Option<String>,
    /// Target hardware identifier (e.g. "m1", "m1pro", "m2", "m2ultra").
    /// Auto-detected if None.
    pub target_hardware: Option<String>,
}

impl Default for CompileRequest {
    fn default() -> Self {
        Self {
            source_path: String::new(),
            source_type: None,
            output_path: None,
            target_lanes: Vec::new(),
            policy_path: None,
            quant_mode: None,
            authority: None,
            draft_path: None,
            ane_models_dir: None,
            metallib_path: None,
            mlx_capture_dir: None,
            target_hardware: None,
        }
    }
}

/// The full plan for a compilation, produced by `PrismCompiler::plan()`.
#[derive(Debug, Clone)]
pub struct CompilePlan {
    pub model_ir: ModelIr,
    pub representation_plan: RepresentationPlan,
    pub execution_graph: ExecutionGraph,
    pub kernel_plan: KernelPlan,
    pub estimated_output_size: u64,
}

/// A compiled kernel artifact with its metadata.
#[derive(Debug, Clone)]
pub struct CompiledKernelEntry {
    pub artifact: CompiledKernelArtifact,
    pub compile_duration_ms: f64,
    pub cache_hit: bool,
}

/// A tensor payload ready for packaging.
#[derive(Debug, Clone)]
pub struct TensorPayload {
    pub name: String,
    pub data: Vec<u8>,
    pub byte_size: u64,
}

/// Input to the cimage packer — fully compiled, ready for packaging only.
#[derive(Debug, Clone)]
pub struct CimageBuildInput {
    pub model_ir_digest: [u8; 32],
    pub representation_plan: RepresentationPlan,
    pub execution_graph: ExecutionGraph,
    pub compiled_kernels: Vec<CompiledKernelArtifact>,
    pub tensor_payloads: Vec<TensorPayload>,
    pub receipts: CompilerReceiptSet,
}

/// A compiler receipt captures the outcome of one compilation stage.
#[derive(Debug, Clone)]
pub struct CompilerReceipt {
    pub stage: CompilerStage,
    pub success: bool,
    pub duration_ms: f64,
    pub message: Option<String>,
}

/// Named stages in the compiler pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompilerStage {
    /// Source located and parsed into a handle.
    SourceResolution,
    /// Model imported through the frontend into canonical ModelIr.
    FrontendImport,
    /// Representation plan produced (quantization strategy, codec selection).
    RepresentationPlanning,
    /// Execution plan produced (regions, lanes, memory plan).
    ExecutionPlanning,
    /// Kernels selected and mapped to backends.
    KernelSelection,
    /// Backend lowers kernels to concrete artifacts (Metal, MLX, ANE).
    BackendLowering,
    /// Payloads packed into segments with alignment and metadata.
    PayloadPacking,
    /// Cimage assembled from segments, kernels, and manifest.
    CimageAssembly,
    /// Compiled image verified for integrity and structural correctness.
    Verification,
    /// Image sealed with policy attestation.
    Sealing,
    /// Image registered in the local catalog.
    Registration,
}

/// Set of receipts covering an entire compilation.
#[derive(Debug, Clone)]
pub struct CompilerReceiptSet {
    pub receipts: Vec<CompilerReceipt>,
}

impl CompilerReceiptSet {
    pub fn new() -> Self {
        Self {
            receipts: Vec::new(),
        }
    }

    pub fn push(&mut self, receipt: CompilerReceipt) {
        self.receipts.push(receipt);
    }

    pub fn all_success(&self) -> bool {
        self.receipts.iter().all(|r| r.success)
    }
}

impl Default for CompilerReceiptSet {
    fn default() -> Self {
        Self::new()
    }
}

/// Full event from one compilation stage with provenance data.
/// Richer than CompilerReceipt — includes digests and toolchain identity
/// for evidence chaining between compilation and execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompileEvent {
    pub stage: CompilerStage,
    pub success: bool,
    /// ISO 8601 timestamp when this event was created.
    pub timestamp: String,
    pub duration_ms: f64,
    pub message: Option<String>,
    /// SHA-256 of the source model file (e.g., GGUF).
    pub source_digest: Option<String>,
    /// SHA-256 of the compilation policy / authority manifest.
    pub policy_digest: Option<String>,
    /// SHA-256 of any intermediate artifact produced at this stage.
    pub artifact_digest: Option<String>,
    /// Compiler/toolchain version string (e.g., metal --version).
    pub toolchain_version: Option<String>,
    /// Optional failure detail — populated on CompilationFailed.
    pub failure_detail: Option<String>,
}

/// Ordered event stream from one compilation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompileEventStream {
    pub events: Vec<CompileEvent>,
    /// Unique compilation run identifier (UUID v4 or equivalent).
    pub compilation_id: String,
    pub source_path: String,
    pub source_digest: Option<String>,
    /// ISO 8601 timestamp when compilation started.
    pub started_at: String,
    /// ISO 8601 timestamp when compilation completed (or failed).
    pub completed_at: Option<String>,
}

impl CompileEventStream {
    pub fn new(source_path: &str) -> Self {
        Self {
            events: Vec::new(),
            compilation_id: compile_id(),
            source_path: source_path.to_string(),
            source_digest: None,
            started_at: compile_timestamp(),
            completed_at: None,
        }
    }

    pub fn push(&mut self, event: CompileEvent) {
        self.events.push(event);
    }

    pub fn all_success(&self) -> bool {
        self.events.iter().all(|e| e.success)
    }

    /// Find the first failure event, if any.
    pub fn first_failure(&self) -> Option<&CompileEvent> {
        self.events.iter().find(|e| !e.success)
    }

    /// The last event is the terminal one (CimageSealed or CompilationFailed).
    pub fn terminal_event(&self) -> Option<&CompileEvent> {
        self.events.last()
    }

    /// Produce a summary CompilerReceiptSet from the event stream.
    pub fn to_receipt_set(&self) -> CompilerReceiptSet {
        let mut set = CompilerReceiptSet::new();
        for event in &self.events {
            set.push(CompilerReceipt {
                stage: event.stage,
                success: event.success,
                duration_ms: event.duration_ms,
                message: event.message.clone(),
            });
        }
        set
    }
}

/// Outcome of a full compilation.
#[derive(Debug, Clone)]
pub struct CompileOutcome {
    pub plan: CompilePlan,
    pub compiled_kernels: Vec<CompiledKernelEntry>,
    pub build_input: CimageBuildInput,
    pub receipts: CompilerReceiptSet,
    pub output_path: Option<String>,
    /// Event stream with full provenance from this compilation run.
    pub event_stream: CompileEventStream,
}

/// Generate a unique-ish compilation ID from a monotonic timestamp.
pub fn compile_id() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("compile-{:016x}", nanos)
}

/// Monotonic timestamp string (epoch nanoseconds, zero-padded).
/// Sortable and unique per-call within a process.
pub fn compile_timestamp() -> String {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| format!("{:020}", d.as_nanos()))
        .unwrap_or_else(|_| "0".to_string())
}
