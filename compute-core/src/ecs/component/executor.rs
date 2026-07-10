//! Runtime executor components — decode loop state machine, step results,
//! and next-token input. Attached to Session entities during Phase I execution.
//!
//! See also: ecs/system/executor_systems.rs for the system that drives these.

use crate::ecs::Component;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// ExecutorState — the sequential decode loop state machine
// ---------------------------------------------------------------------------

/// Stage within the decode loop state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutorStage {
    /// Initial state — waiting to begin.
    Idle,
    /// Loading weights into memory / binding resources.
    Loading,
    /// Running the prefill (prompt ingestion) pass.
    Prefill,
    /// Running auto-regressive decode (one token at a time).
    Decode,
    /// Draining — finishing up, releasing resources.
    Draining,
}

/// Tracks the runtime state of one active inference execution session.
///
/// One `ExecutorState` component per Session entity. The state machine
/// advances: Idle → Loading → Prefill → Decode → Draining.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutorState {
    /// Current stage in the decode loop.
    pub stage: ExecutorStage,
    /// How many decode steps have been produced (total tokens emitted).
    pub step_counter: u64,
    /// Count of errors encountered during this session.
    pub error_count: u64,
    /// Maximum allowed decode steps before forcing a drain.
    pub max_steps: u64,
}

impl Component for ExecutorState {}

// ---------------------------------------------------------------------------
// ExecutorStep — one decode step's result
// ---------------------------------------------------------------------------

/// One decode step result: the generated token, its logits (if available),
/// and the KV block indices written during this step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutorStep {
    /// The generated token id.
    pub token_id: u32,
    /// Logits vector from the LM head (optional, for scoring / rejection).
    pub logits: Option<Vec<f32>>,
    /// Indices of the KV cache blocks used / written in this step.
    pub kv_block_indices: Vec<u32>,
}

impl Component for ExecutorStep {}

// ---------------------------------------------------------------------------
// DecodeInput — next token to feed into the decoder
// ---------------------------------------------------------------------------

/// Input for the next decode step — the token id that will be embedded
/// and passed through the decoder layers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecodeInput {
    /// Token id to embed and decode.
    pub token_id: u32,
}

impl Component for DecodeInput {}

// ---------------------------------------------------------------------------
// Resource wrappers (stored in CompWorld resources, not per-entity)
// ---------------------------------------------------------------------------

/// Global weight store — wraps the resolved MLX Array references for all
/// decoder layers. In the initial ECS scaffolding phase this is a placeholder;
/// Phase 2+ will populate it from the existing `LoadedProfiledModel`.
#[derive(Debug, Clone, Default)]
pub struct WeightStore {
    /// Number of layers whose weights are stored.
    pub num_layers: usize,
    /// Placeholder — Phase 2 will hold a Vec<LayerWeights> or similar.
    pub loaded: bool,
}

/// Global routing store — holds the per-layer backend routing decisions
/// (which backend handles which operation for each layer).
#[derive(Debug, Clone, Default)]
pub struct RouteStore {
    /// Number of layers with routing decisions.
    pub num_layers: usize,
    /// Placeholder — Phase 2+ will hold OperationRoute per layer.
    pub resolved: bool,
}

/// Global ANE (Apple Neural Engine) handle store — holds CoreML model
/// handles and the MoE scheduler.
#[derive(Debug, Clone, Default)]
pub struct AneStore {
    /// Placeholder — Phase 2+ holds CoreAiModel handles and MoE scheduler.
    pub initialized: bool,
}
