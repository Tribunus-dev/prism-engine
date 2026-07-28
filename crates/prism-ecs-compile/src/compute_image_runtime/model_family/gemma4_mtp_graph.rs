//! Gemma 4 + MTP execution graph — pure data types and pure
//! algorithms.

use serde::{Deserialize, Serialize};

/// Phase types in the Gemma 4 + MTP execution pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MTPPhase {
    /// Embedding lookup + RoPE precomputation.
    InputEmbedding,
    /// Target model prefill over prompt tokens.
    TargetPrefill,
    /// MTP drafter predicts K future tokens in parallel.
    MtpDraft,
    /// Target model verifies draft tokens.
    TargetVerify,
    /// Accept matching prefix of draft tokens.
    Acceptance,
    /// KV commit for accepted tokens.
    KvCommit,
    /// Sample fallback token at first rejection.
    SampleFallback,
    /// Single-token decode for verified position.
    TargetDecode,
    /// Project to vocabulary and sample.
    OutputProjection,
}

/// Edge type between execution graph nodes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MTPEdge {
    /// Source phase.
    pub from: MTPPhase,
    /// Target phase.
    pub to: MTPPhase,
    /// How many tokens flow along this edge.
    pub token_count: u32,
    /// Whether this edge is taken conditionally.
    pub conditional: bool,
}

/// Canonical execution graph for Gemma 4 with optional MTP
/// speculative decoding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MTPExecutionGraph {
    /// All phases in the graph.
    pub phases: Vec<MTPPhase>,
    /// Directed edges between phases.
    pub edges: Vec<MTPEdge>,
}

/// MTP share contract — what shared state is allowed between
/// concurrent MTP drafts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MTPShareContract {
    /// Whether KV cache is shared between drafts.
    pub share_kv_cache: bool,
    /// Whether position embeddings are shared.
    pub share_position_embeddings: bool,
    /// Maximum number of concurrent drafts.
    pub max_concurrent_drafts: u32,
}
