//! Protocol payloads — the 16 type-specific JSON schemas that ride
//! inside a [`Frame::payload`](crate::worker_protocol::Frame::payload)
//! field.
//!
//! Each payload struct corresponds to one [`HostCommand`] or
//! [`WorkerEvent`] variant. The protocol is intentionally
//! self-describing: the [`MessageKind`] discriminator tells the
//! receiver which payload schema to expect, and the payload itself is
//! a plain JSON object with the fields below.
//!
//! # Tensor payloads do not cross the wire
//!
//! All tensor data stays in the worker's address space; only metadata,
//! token IDs, and control messages flow over this channel. The protocol
//! is intentionally small.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};

use super::types::GenerationRegime;

// ── Payload Schemas ────────────────────────────────────────────────────────

/// Payload for [`HostCommand::StartGeneration`](super::HostCommand::StartGeneration).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartGenerationPayload {
    /// Token IDs of the prompt.
    pub prompt_token_ids: Vec<u32>,
    /// Maximum number of output tokens to generate.
    pub max_output_tokens: u32,
    /// Absolute deadline in milliseconds (epoch-relative or monotonic,
    /// depending on the worker's time base).
    pub deadline_ms: u64,
    /// Opaque request identifier echoed in all response events.
    pub request_id: String,
    /// Temperature for sampling (None = use model default).
    pub temperature: Option<f32>,
    /// Top-K for sampling (None = no limit).
    pub top_k: Option<u32>,
    /// Top-P (nucleus) for sampling (None = no filtering).
    pub top_p: Option<f32>,
    /// Random seed for reproducible generation.
    pub seed: Option<u64>,
    /// Token IDs that stop generation when sampled.
    pub stop_token_ids: Vec<u32>,
    /// Generation regime (diffusion vs. autoregressive).
    #[serde(default)]
    pub generation_regime: GenerationRegime,
    /// Number of denoising steps for diffusion generation.
    #[serde(default)]
    pub denoising_steps: Option<u32>,
    /// Confidence threshold for committing positions.
    #[serde(default)]
    pub confidence_threshold: Option<f32>,
    /// Number of canvas tokens for diffusion generation.
    #[serde(default)]
    pub canvas_tokens: Option<u32>,
}

/// Payload for [`WorkerEvent::Token`](super::WorkerEvent::Token).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenPayload {
    /// The generation request this token belongs to.
    pub request_id: String,
    /// Sampled token ID from the model vocabulary.
    pub token_id: u32,
    /// Position (index) of this token in the output sequence.
    pub position: u32,
    /// Log-probability of the token, if available.
    pub logprob: Option<f32>,
}

/// Payload for [`WorkerEvent::DiffusionStepStarted`](super::WorkerEvent::DiffusionStepStarted).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffusionStepStartedPayload {
    pub request_id: String,
    pub step: u32,
    pub total_steps: u32,
    pub unresolved_positions: u32,
    pub committed_positions: u32,
}

/// Payload for [`WorkerEvent::DiffusionStepCompleted`](super::WorkerEvent::DiffusionStepCompleted).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffusionStepCompletedPayload {
    pub request_id: String,
    pub step: u32,
    pub total_steps: u32,
    pub committed_this_step: u32,
    pub newly_committed_tokens: Vec<u32>,
    pub newly_committed_positions: Vec<u32>,
    pub avg_confidence: f32,
}

/// Payload for [`WorkerEvent::CanvasUpdated`](super::WorkerEvent::CanvasUpdated).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanvasUpdatedPayload {
    pub request_id: String,
    pub step: u32,
    pub canvas_width: u32,
    pub resolved_count: u32,
    pub unresolved_count: u32,
}

/// Payload for [`WorkerEvent::PositionsCommitted`](super::WorkerEvent::PositionsCommitted).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionsCommittedPayload {
    pub request_id: String,
    pub positions: Vec<u32>,
    pub token_ids: Vec<u32>,
    pub confidence_scores: Vec<f32>,
    pub text_preview: String,
}

/// Payload for [`WorkerEvent::Converged`](super::WorkerEvent::Converged).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergedPayload {
    pub request_id: String,
    pub total_steps: u32,
    pub final_committed_count: u32,
    pub final_unresolved_count: u32,
    pub reason: String,
}

/// Payload for [`WorkerEvent::DiffusionGenerationCompleted`](super::WorkerEvent::DiffusionGenerationCompleted).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffusionGenerationCompletedPayload {
    pub request_id: String,
    pub token_count: u32,
    pub ttft_ms: u64,
    pub total_ms: u64,
    pub total_diffusion_steps: u32,
    pub final_committed_count: u32,
    pub final_unresolved_count: u32,
    pub convergence_reason: String,
}

/// Payload for [`WorkerEvent::Heartbeat`](super::WorkerEvent::Heartbeat).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeartbeatPayload {
    /// Current generation phase, if any (e.g. `"prefill"`, `"decode"`).
    pub request_phase: Option<String>,
    /// Transformer layer currently being processed, if applicable.
    pub current_layer: Option<u32>,
    /// Resident set size of the worker process in bytes.
    pub process_rss_bytes: u64,
    /// Elapsed time since the worker started, in milliseconds.
    pub elapsed_ms: u64,
    /// Index of the most recently completed decode step, if any.
    pub last_completed_step: Option<u32>,
    /// Request ID of the currently active generation, if any.
    pub active_request_id: Option<String>,
    /// MLX Metal active memory in bytes.
    pub mlx_active_memory: u64,
    /// MLX Metal cache memory in bytes.
    pub mlx_cache_memory: u64,
    /// MLX Metal peak memory in bytes.
    pub mlx_peak_memory: u64,
}

/// Payload for [`WorkerEvent::GenerationCompleted`](super::WorkerEvent::GenerationCompleted).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationCompletedPayload {
    /// The generation request that completed.
    pub request_id: String,
    /// Total number of output tokens produced.
    pub token_count: u32,
    /// Time-to-first-token in milliseconds.
    pub ttft_ms: u64,
    /// Total generation time in milliseconds.
    pub total_ms: u64,
}

/// Payload for [`WorkerEvent::GenerationFailed`](super::WorkerEvent::GenerationFailed).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationFailedPayload {
    /// The generation request that failed.
    pub request_id: String,
    /// Machine-readable error code.
    pub error_code: String,
    /// Human-readable error description.
    pub message: String,
    /// Phase during which the failure occurred.
    pub phase: String,
    /// Optional diagnostic hints (e.g. stack snippets, log excerpts).
    pub diagnostics: Option<Vec<String>>,
}

/// Payload for [`WorkerEvent::WorkerFatal`](super::WorkerEvent::WorkerFatal).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerFatalPayload {
    /// Machine-readable error code.
    pub error_code: String,
    /// Human-readable error description.
    pub message: String,
    /// Phase during which the fatal error occurred.
    pub phase: String,
    /// Optional diagnostic hints (e.g. stack snippets, log excerpts).
    pub diagnostics: Option<Vec<String>>,
}

/// Payload for [`WorkerEvent::HelloAck`](super::WorkerEvent::HelloAck) /
/// [`HostCommand::LoadModel`](super::HostCommand::LoadModel) carrying
/// policy limits from the host to the worker.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PolicySnapshotPayload {
    /// Maximum active memory for MLX operations, in bytes.
    pub mlx_active_memory_limit_bytes: u64,
    /// Maximum MLX cache size, in bytes.
    pub mlx_cache_limit_bytes: u64,
    /// Maximum number of prompt tokens the worker should accept.
    pub prompt_token_ceiling: usize,
    /// Maximum number of output tokens per generation.
    pub output_token_ceiling: u32,
}

/// Payload for [`WorkerEvent::ResearchTraceBatch`](super::WorkerEvent::ResearchTraceBatch).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchTraceBatchPayload {
    pub request_id: String,
    pub batch_index: u64,
    pub events: Vec<ResearchTraceEventJson>,
    pub buffer_drops: u64,
    pub buffer_overflowed: bool,
}

/// JSON-serializable version of TraceEvent for wire transport.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchTraceEventJson {
    pub monotonic_ns: u64,
    pub stage_id: u16,
    pub substrate_id: u8,
    pub clock_domain: u8,
    pub layer_index: u8,
    pub attention_kind: u8,
    pub status: u8,
    pub graph_build_ns: u32,
    pub eval_ns: u32,
    pub sync_ns: u32,
    pub mlx_active_delta: i32,
    pub mlx_cache_delta: i32,
    pub rss_delta: i32,
    pub materialized_bytes: u32,
    pub file_read_bytes: u32,
    pub kv_delta: i32,
}
