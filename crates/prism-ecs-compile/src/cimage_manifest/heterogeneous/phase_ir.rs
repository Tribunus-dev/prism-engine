//! Canonical PhaseIR — the compiler's intermediate graph before
//! backend decisions.
//!
//! This module owns the **semantic graph**: [`PhaseNode`],
//! [`PhaseEdge`], [`PhaseValue`], and the [`PhaseKind`] /
//! [`PhaseEdgeKind`] / [`DependencyClass`] enums. The compiler lowers
//! all frontends into this canonical graph before any backend
//! placement, resource planning, or concurrency analysis runs.
//!
//! A [`PhaseNode`] is an executable semantic region — not
//! necessarily a single operator. Phase boundaries are drawn where
//! the compiler has decided to insert materialization, change lanes,
//! or split for concurrency. The graph is guaranteed acyclic at
//! emission time.

use serde::{Deserialize, Serialize};

/// Identifies a phase within the compilation session.
pub type PhaseId = u64;

/// Identifies a value in the phase graph.
pub type ValueId = u64;

/// Identifies an operator within a phase.
pub type OperatorId = String;

/// The compiler lowers all frontends into one canonical graph before
/// backend decisions. A [`PhaseNode`] represents an executable
/// semantic region, not necessarily one operator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseGraph {
    pub phases: Vec<PhaseNode>,
    pub edges: Vec<PhaseEdge>,
    pub values: Vec<PhaseValue>,
}

/// A single executable semantic region in the canonical PhaseIR.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseNode {
    pub phase_id: PhaseId,
    pub kind: PhaseKind,
    pub operators: Vec<OperatorId>,
    pub inputs: Vec<ValueId>,
    pub outputs: Vec<ValueId>,
    pub shape_contract: ShapeContract,
    pub numerical_contract: NumericalContract,
    pub dependency_class: DependencyClass,
}

/// A dependency edge between two PhaseIR nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseEdge {
    pub from: PhaseId,
    pub to: PhaseId,
    pub value: Option<ValueId>,
    pub kind: PhaseEdgeKind,
}

/// A value (tensor / activation) flowing between phases.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseValue {
    pub value_id: ValueId,
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: String,
    pub producer: Option<PhaseId>,
    pub consumers: Vec<PhaseId>,
}

/// Kind of executable semantic region.
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum PhaseKind {
    Attention,
    MlpGate,
    MlpUp,
    MlpDown,
    MlpActivation,
    RmsNorm,
    RoPE,
    ResidualAdd,
    LogitsProjection,
    Sampling,
    Softmax,
    KvUpdate,
    KvCacheLookup,
    Prologue,
    Epilogue,
    Fusion,
    DataTransfer,
}

/// How a dependency class affects concurrency.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum DependencyClass {
    /// Strict token-autoregressive dependency — must serialize.
    StrictTokenDependency,
    /// Intra-layer dependency (e.g., attention → MLP within one layer).
    IntraLayerDependency,
    /// Cross-sequence independent — phases from different sequences
    /// can overlap.
    CrossSequenceIndependent,
    /// Prefill batch independent — phases within a prefill batch are
    /// independent.
    PrefillBatchIndependent,
    /// Background or speculative work — can overlap with decode.
    BackgroundSpeculativeIndependent,
    /// Host-only dependency (e.g., tokenization, metadata).
    HostOnlyDependency,
}

/// Shape contract for a phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeContract {
    pub batch_dim: Option<u64>,
    pub seq_len: Option<u64>,
    pub hidden_dim: u64,
    pub num_heads: u64,
    pub head_dim: u64,
}

/// Numerical contract for a phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericalContract {
    pub accumulation_dtype: String,
    pub activation_dtype: String,
    pub requires_determinism: bool,
}

/// Kind of edge between PhaseIR nodes.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum PhaseEdgeKind {
    Data,
    Control,
    State,
}
