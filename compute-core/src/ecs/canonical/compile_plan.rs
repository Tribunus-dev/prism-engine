//! CompilePlan, CompilerReceipt — the public compilation API types.
//!
//! These types define the request/response contract for the PrismCompiler.
//! All binary entry points, server endpoints, tests, and constitutional
//! commands call through this API.

use super::execution_graph::ExecutionGraph;
use super::kernel_abi::{CompiledKernelArtifact, KernelPlan};
use super::model_ir::ModelIr;
use super::representation::RepresentationPlan;

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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompilerStage {
    SourceResolved,
    SourceFileVerified,
    ModelInspected,
    ModelNormalized,
    RepresentationPlanned,
    ExecutionGraphBuilt,
    KernelLoweringStarted,
    KernelCompiled,
    KernelVerified,
    PayloadPacked,
    CimageWritten,
    CimageVerified,
    CimageSealed,
    CompilationFailed,
}

/// Set of receipts covering an entire compilation.
#[derive(Debug, Clone)]
pub struct CompilerReceiptSet {
    pub receipts: Vec<CompilerReceipt>,
}

impl CompilerReceiptSet {
    pub fn new() -> Self {
        Self { receipts: Vec::new() }
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

/// Outcome of a full compilation.
#[derive(Debug, Clone)]
pub struct CompileOutcome {
    pub plan: CompilePlan,
    pub compiled_kernels: Vec<CompiledKernelEntry>,
    pub build_input: CimageBuildInput,
    pub receipts: CompilerReceiptSet,
    pub output_path: Option<String>,
}
