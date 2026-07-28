//! Compilation-related ECS component types for admission, scheduling,
//! profitability, and region planning.

use crate::ecs::Component;

// ── Identity type aliases ─────────────────────────────────────────────────

pub type EvidenceId = String;
pub type FrontierNodeId = String;
pub type PathId = String;
pub type NodeId = String;
pub type BackendTarget = String;
pub type OpId = String;
use prism_ecs_compile::compilation::phase_ir::CompilePhaseDescriptor;
use crate::ecs::config::ModelExecutionPlan;

impl Component for CompilePhaseDescriptor {}
impl Component for ModelExecutionPlan {}

// ── Epoch policy ───────────────────────────────────────────────────────────

/// Policy for epoch scheduling — either a fixed number of epochs or adaptive
/// (auto-determined by the scheduler).
#[derive(Debug, Clone)]
pub enum EpochPolicy {
    Fixed(u64),
    Adaptive,
}

// ── Graph node kind (mirrors graph_optimizer::OpKind) ─────────────────────

/// Kind of a graph node in the compilation graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphNodeKind {
    EmbeddingLookup,
    RmsNorm,
    QProj,
    KProj,
    VProj,
    QNorm,
    KNorm,
    RoPE,
    Attention,
    OProj,
    GateProj,
    UpProj,
    SiLU,
    GateTimesUp,
    DownProj,
    ResidualAdd,
    FinalNorm,
    OutputProjection,
    Softcap,
    Argmax,
    Matmul,
    QuantizedMatmul,
    Softmax,
    Add,
    Multiply,
    Transpose,
    Reshape,
    Reduction,
    DecoderLayer,
    MlpBlock,
    AttentionBlock,
    Unknown(String),
}

// ── Compilation phase ─────────────────────────────────────────────────────

/// Identifies the type of compilation phase for a pipeline stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompilationPhase {
    Embedding,
    AttentionNorm,
    QProjection,
    KProjection,
    VProjection,
    QNorm,
    KNorm,
    RoPE,
    AttentionScore,
    SoftmaxPhase,
    AttentionValueAggregation,
    OProjection,
    ResidualAdd,
    MlpNorm,
    GateProjection,
    SiLUActivation,
    UpProjection,
    DownProjection,
    FinalNorm,
    LogitsProjection,
}

// ═══════════════════════════════════════════════════════════════════════════
// Component types
// ═══════════════════════════════════════════════════════════════════════════

/// Result of an admission gate check.
#[derive(Debug, Clone)]
pub struct AdmissionGate {
    pub name: String,
    pub passed: bool,
    pub evidence: Option<EvidenceId>,
}
impl Component for AdmissionGate {}

/// Current and maximum epoch schedule managed by the epoch scheduler.
#[derive(Debug, Clone)]
pub struct EpochSchedule {
    pub current: u64,
    pub max: u64,
    pub policy: EpochPolicy,
}
impl Component for EpochSchedule {}

/// State of the calibration frontier — active frontier nodes and the
/// currently-active evaluation path.
#[derive(Debug, Clone)]
pub struct FrontierState {
    pub nodes: Vec<FrontierNodeId>,
    pub active_path: PathId,
}
impl Component for FrontierState {}

/// A node in the compilation graph with dependencies and kind.
#[derive(Debug, Clone)]
pub struct GraphNode {
    pub id: NodeId,
    pub deps: Vec<NodeId>,
    pub kind: GraphNodeKind,
}
impl Component for GraphNode {}

/// Intermediate representation for a compilation phase.
#[derive(Debug, Clone)]
pub struct PhaseIR {
    pub phase: CompilationPhase,
    pub ir: Vec<u8>,
}
impl Component for PhaseIR {}

/// Profitability score for a compilation unit.
#[derive(Debug, Clone)]
pub struct ProfitabilityScore {
    pub score: f64,
    pub confidence: f64,
    pub reason: String,
}
impl Component for ProfitabilityScore {}

/// Result of a qualification gate check.
#[derive(Debug, Clone)]
pub struct QualificationGate {
    pub name: String,
    pub min_score: f64,
    pub actual: f64,
    pub passed: bool,
}
impl Component for QualificationGate {}

/// Planned region for execution on a specific backend.
#[derive(Debug, Clone)]
pub struct RegionPlan {
    pub region_id: String,
    pub backend: BackendTarget,
    pub schedule: Vec<OpId>,
}
impl Component for RegionPlan {}
