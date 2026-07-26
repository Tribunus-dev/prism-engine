//! Phase contract catalog — the canonical list of
//! [`PhaseContract`]s, one per [`PipelinePhase`](super::phase::PipelinePhase).
//!
//! Shape patterns use symbolic dimensions (e.g. `{hidden_dim}`,
//! `{vocab_size}`) to express which dimensions are shared across
//! phases. Multi-input phases (AttentionScores, MaskApply, etc.)
//! declare all inputs.

#![forbid(unsafe_code)]

use super::dim::{Dim, TensorContract, TensorRole};
use super::phase::PipelinePhase;

use Dim::{Known, Symbol};

/// Full contract for a single inference pipeline phase.
#[derive(Debug, Clone, Copy)]
pub struct PhaseContract {
    /// Which phase this contract describes.
    pub phase: PipelinePhase,
    /// Input tensor contracts (primary input first, then secondary inputs).
    pub inputs: &'static [TensorContract],
    /// Output tensor contracts.
    pub outputs: &'static [TensorContract],
    /// Default reference tolerance for numerical conformance.
    pub tolerance: f64,
    /// Human-readable description of what this phase does.
    pub description: &'static str,
}

/// Canonical contracts for all 21 inference pipeline phases.
///
/// Shape patterns use symbolic dimensions (e.g. `{hidden_dim}`,
/// `{vocab_size}`) to express which dimensions are shared across
/// phases. Multi-input phases (AttentionScores, MaskApply, etc.)
/// declare all inputs.
pub const PHASE_CONTRACTS: &[PhaseContract] = &[
    PhaseContract {
        phase: PipelinePhase::TokenEmbedding,
        inputs: &[
            TensorContract { name: "token_ids", role: TensorRole::PrimaryInput, shape_pattern: &[Known(1), Symbol("seq_len")], dtype: "int32" },
        ],
        outputs: &[
            TensorContract { name: "hidden", role: TensorRole::Output, shape_pattern: &[Known(1), Symbol("seq_len"), Symbol("hidden_dim")], dtype: "float32" },
        ],
        tolerance: 1e-4,
        description: "Embed token ID sequences into dense hidden-state vectors.",
    },
    PhaseContract {
        phase: PipelinePhase::PositionEncodingOrRope,
        inputs: &[
            TensorContract { name: "hidden", role: TensorRole::PrimaryInput, shape_pattern: &[Known(1), Symbol("seq_len"), Symbol("hidden_dim")], dtype: "float32" },
        ],
        outputs: &[
            TensorContract { name: "hidden", role: TensorRole::Output, shape_pattern: &[Known(1), Symbol("seq_len"), Symbol("hidden_dim")], dtype: "float32" },
        ],
        tolerance: 1e-4,
        description: "Apply Rotary Position Embedding (RoPE) or learned positional encoding.",
    },
    PhaseContract {
        phase: PipelinePhase::QkvProjection,
        inputs: &[
            TensorContract { name: "hidden", role: TensorRole::PrimaryInput, shape_pattern: &[Known(1), Symbol("hidden_dim")], dtype: "float32" },
        ],
        outputs: &[
            TensorContract { name: "q", role: TensorRole::Output, shape_pattern: &[Known(1), Symbol("hidden_dim")], dtype: "float32" },
            TensorContract { name: "k", role: TensorRole::Output, shape_pattern: &[Known(1), Symbol("hidden_dim")], dtype: "float32" },
            TensorContract { name: "v", role: TensorRole::Output, shape_pattern: &[Known(1), Symbol("hidden_dim")], dtype: "float32" },
        ],
        tolerance: 1e-3,
        description: "Project hidden state to Q, K, V subspaces. Weight matrices: Wq, Wk, Wv.",
    },
    PhaseContract {
        phase: PipelinePhase::KvRead,
        inputs: &[
            TensorContract { name: "cache_k", role: TensorRole::PrimaryInput, shape_pattern: &[Known(1), Symbol("cache_len"), Symbol("head_dim")], dtype: "float32" },
            TensorContract { name: "cache_v", role: TensorRole::PrimaryInput, shape_pattern: &[Known(1), Symbol("cache_len"), Symbol("head_dim")], dtype: "float32" },
        ],
        outputs: &[
            TensorContract { name: "k", role: TensorRole::Output, shape_pattern: &[Known(1), Known(1), Symbol("head_dim")], dtype: "float32" },
            TensorContract { name: "v", role: TensorRole::Output, shape_pattern: &[Known(1), Known(1), Symbol("head_dim")], dtype: "float32" },
        ],
        tolerance: 1e-4,
        description: "Read current-position K, V entries from KV cache (pre-filled or cached).",
    },
    PhaseContract {
        phase: PipelinePhase::KvWrite,
        inputs: &[
            TensorContract { name: "k", role: TensorRole::PrimaryInput, shape_pattern: &[Known(1), Symbol("num_kv_heads"), Known(1), Symbol("head_dim")], dtype: "float32" },
            TensorContract { name: "v", role: TensorRole::SecondaryInput, shape_pattern: &[Known(1), Symbol("num_kv_heads"), Known(1), Symbol("head_dim")], dtype: "float32" },
            TensorContract { name: "position", role: TensorRole::SecondaryInput, shape_pattern: &[Known(1)], dtype: "int32" },
        ],
        outputs: &[
            TensorContract { name: "cache_k", role: TensorRole::Output, shape_pattern: &[Known(1), Symbol("num_kv_heads"), Symbol("cache_len"), Symbol("head_dim")], dtype: "float32" },
            TensorContract { name: "cache_v", role: TensorRole::Output, shape_pattern: &[Known(1), Symbol("num_kv_heads"), Symbol("cache_len"), Symbol("head_dim")], dtype: "float32" },
        ],
        tolerance: 1e-4,
        description: "Write K, V at position into KV cache (single-position write).",
    },
    PhaseContract {
        phase: PipelinePhase::KvAppend,
        inputs: &[
            TensorContract { name: "k", role: TensorRole::PrimaryInput, shape_pattern: &[Known(1), Symbol("num_kv_heads"), Known(1), Symbol("head_dim")], dtype: "float32" },
            TensorContract { name: "v", role: TensorRole::SecondaryInput, shape_pattern: &[Known(1), Symbol("num_kv_heads"), Known(1), Symbol("head_dim")], dtype: "float32" },
            TensorContract { name: "cache_k", role: TensorRole::SecondaryInput, shape_pattern: &[Known(1), Symbol("num_kv_heads"), Symbol("cache_len"), Symbol("head_dim")], dtype: "float32" },
            TensorContract { name: "cache_v", role: TensorRole::SecondaryInput, shape_pattern: &[Known(1), Symbol("num_kv_heads"), Symbol("cache_len"), Symbol("head_dim")], dtype: "float32" },
        ],
        outputs: &[
            TensorContract { name: "cache_k", role: TensorRole::Output, shape_pattern: &[Known(1), Symbol("num_kv_heads"), Symbol("extended_len"), Symbol("head_dim")], dtype: "float32" },
            TensorContract { name: "cache_v", role: TensorRole::Output, shape_pattern: &[Known(1), Symbol("num_kv_heads"), Symbol("extended_len"), Symbol("head_dim")], dtype: "float32" },
        ],
        tolerance: 1e-4,
        description: "Append K, V to the end of KV cache (extend sequence dimension).",
    },
    PhaseContract {
        phase: PipelinePhase::KvView,
        inputs: &[
            TensorContract { name: "cache_k", role: TensorRole::PrimaryInput, shape_pattern: &[Known(1), Symbol("num_kv_heads"), Symbol("cache_len"), Symbol("head_dim")], dtype: "float32" },
            TensorContract { name: "cache_v", role: TensorRole::SecondaryInput, shape_pattern: &[Known(1), Symbol("num_kv_heads"), Symbol("cache_len"), Symbol("head_dim")], dtype: "float32" },
        ],
        outputs: &[
            TensorContract { name: "k_view", role: TensorRole::Output, shape_pattern: &[Known(1), Symbol("num_kv_heads"), Symbol("visible_len"), Symbol("head_dim")], dtype: "float32" },
            TensorContract { name: "v_view", role: TensorRole::Output, shape_pattern: &[Known(1), Symbol("num_kv_heads"), Symbol("visible_len"), Symbol("head_dim")], dtype: "float32" },
        ],
        tolerance: 1e-4,
        description: "Extract a view/slice of KV cache without mutation.",
    },
    PhaseContract {
        phase: PipelinePhase::AttentionScores,
        inputs: &[
            TensorContract { name: "q", role: TensorRole::PrimaryInput, shape_pattern: &[Known(1), Symbol("num_heads"), Symbol("seq_len"), Symbol("head_dim")], dtype: "float32" },
            TensorContract { name: "k", role: TensorRole::SecondaryInput, shape_pattern: &[Known(1), Symbol("num_heads"), Symbol("kv_len"), Symbol("head_dim")], dtype: "float32" },
        ],
        outputs: &[
            TensorContract { name: "scores", role: TensorRole::Output, shape_pattern: &[Known(1), Symbol("num_heads"), Symbol("seq_len"), Symbol("kv_len")], dtype: "float32" },
        ],
        tolerance: 1e-3,
        description: "Compute attention scores: Q @ K^T over head dimensions.",
    },
    PhaseContract {
        phase: PipelinePhase::MaskApply,
        inputs: &[
            TensorContract { name: "scores", role: TensorRole::PrimaryInput, shape_pattern: &[Known(1), Symbol("num_heads"), Symbol("seq_len"), Symbol("kv_len")], dtype: "float32" },
            TensorContract { name: "mask", role: TensorRole::SecondaryInput, shape_pattern: &[Known(1), Known(1), Symbol("seq_len"), Symbol("kv_len")], dtype: "float32" },
        ],
        outputs: &[
            TensorContract { name: "scores", role: TensorRole::Output, shape_pattern: &[Known(1), Symbol("num_heads"), Symbol("seq_len"), Symbol("kv_len")], dtype: "float32" },
        ],
        tolerance: 1e-4,
        description: "Apply causal/attention mask to scores (addition of -inf or large negative).",
    },
    PhaseContract {
        phase: PipelinePhase::Softmax,
        inputs: &[
            TensorContract { name: "scores", role: TensorRole::PrimaryInput, shape_pattern: &[Known(1), Symbol("num_heads"), Symbol("seq_len"), Symbol("kv_len")], dtype: "float32" },
        ],
        outputs: &[
            TensorContract { name: "probs", role: TensorRole::Output, shape_pattern: &[Known(1), Symbol("num_heads"), Symbol("seq_len"), Symbol("kv_len")], dtype: "float32" },
        ],
        tolerance: 1e-4,
        description: "Softmax over attention scores (last dimension = kv_len).",
    },
    PhaseContract {
        phase: PipelinePhase::AttentionWeightedSum,
        inputs: &[
            TensorContract { name: "probs", role: TensorRole::PrimaryInput, shape_pattern: &[Known(1), Symbol("num_heads"), Symbol("seq_len"), Symbol("kv_len")], dtype: "float32" },
            TensorContract { name: "v", role: TensorRole::SecondaryInput, shape_pattern: &[Known(1), Symbol("num_heads"), Symbol("kv_len"), Symbol("head_dim")], dtype: "float32" },
        ],
        outputs: &[
            TensorContract { name: "context", role: TensorRole::Output, shape_pattern: &[Known(1), Symbol("num_heads"), Symbol("seq_len"), Symbol("head_dim")], dtype: "float32" },
        ],
        tolerance: 1e-3,
        description: "Weighted sum of V: softmax(QK^T) @ V.",
    },
    PhaseContract {
        phase: PipelinePhase::AttentionOutputProjection,
        inputs: &[
            TensorContract { name: "context", role: TensorRole::PrimaryInput, shape_pattern: &[Known(1), Symbol("hidden_dim")], dtype: "float32" },
        ],
        outputs: &[
            TensorContract { name: "attention_output", role: TensorRole::Output, shape_pattern: &[Known(1), Symbol("hidden_dim")], dtype: "float32" },
        ],
        tolerance: 1e-3,
        description: "Project concatenated attention head outputs back to model dimension via Wo.",
    },
    PhaseContract {
        phase: PipelinePhase::ResidualAdd1,
        inputs: &[
            TensorContract { name: "attention_output", role: TensorRole::PrimaryInput, shape_pattern: &[Known(1), Symbol("hidden_dim")], dtype: "float32" },
            TensorContract { name: "residual", role: TensorRole::SecondaryInput, shape_pattern: &[Known(1), Symbol("hidden_dim")], dtype: "float32" },
        ],
        outputs: &[
            TensorContract { name: "hidden", role: TensorRole::Output, shape_pattern: &[Known(1), Symbol("hidden_dim")], dtype: "float32" },
        ],
        tolerance: 1e-4,
        description: "First residual add: attention_output + residual (input before attention sublayer).",
    },
    PhaseContract {
        phase: PipelinePhase::Norm1,
        inputs: &[
            TensorContract { name: "hidden", role: TensorRole::PrimaryInput, shape_pattern: &[Known(1), Symbol("hidden_dim")], dtype: "float32" },
        ],
        outputs: &[
            TensorContract { name: "hidden", role: TensorRole::Output, shape_pattern: &[Known(1), Symbol("hidden_dim")], dtype: "float32" },
        ],
        tolerance: 1e-4,
        description: "First layer normalization (RMS norm or LayerNorm) at pre-MLP boundary.",
    },
    PhaseContract {
        phase: PipelinePhase::MlpGateUp,
        inputs: &[
            TensorContract { name: "hidden", role: TensorRole::PrimaryInput, shape_pattern: &[Known(1), Symbol("hidden_dim")], dtype: "float32" },
        ],
        outputs: &[
            TensorContract { name: "gate", role: TensorRole::Output, shape_pattern: &[Known(1), Symbol("ffw_dim")], dtype: "float32" },
            TensorContract { name: "up", role: TensorRole::Output, shape_pattern: &[Known(1), Symbol("ffw_dim")], dtype: "float32" },
        ],
        tolerance: 1e-3,
        description: "MLP gate and up projection (gate_proj, up_proj for SwiGLU).",
    },
    PhaseContract {
        phase: PipelinePhase::Activation,
        inputs: &[
            TensorContract { name: "gate", role: TensorRole::PrimaryInput, shape_pattern: &[Known(1), Symbol("ffw_dim")], dtype: "float32" },
        ],
        outputs: &[
            TensorContract { name: "activated", role: TensorRole::Output, shape_pattern: &[Known(1), Symbol("ffw_dim")], dtype: "float32" },
        ],
        tolerance: 1e-4,
        description: "Activation function (SiLU, ReLU, GELU) applied to MLP gate output.",
    },
    PhaseContract {
        phase: PipelinePhase::MlpDown,
        inputs: &[
            TensorContract { name: "up_gated", role: TensorRole::PrimaryInput, shape_pattern: &[Known(1), Symbol("ffw_dim")], dtype: "float32" },
        ],
        outputs: &[
            TensorContract { name: "mlp_output", role: TensorRole::Output, shape_pattern: &[Known(1), Symbol("hidden_dim")], dtype: "float32" },
        ],
        tolerance: 1e-3,
        description: "MLP down projection (down_proj) after activation.",
    },
    PhaseContract {
        phase: PipelinePhase::ResidualAdd2,
        inputs: &[
            TensorContract { name: "mlp_output", role: TensorRole::PrimaryInput, shape_pattern: &[Known(1), Symbol("hidden_dim")], dtype: "float32" },
            TensorContract { name: "residual", role: TensorRole::SecondaryInput, shape_pattern: &[Known(1), Symbol("hidden_dim")], dtype: "float32" },
        ],
        outputs: &[
            TensorContract { name: "hidden", role: TensorRole::Output, shape_pattern: &[Known(1), Symbol("hidden_dim")], dtype: "float32" },
        ],
        tolerance: 1e-4,
        description: "Second residual add: mlp_output + residual (pre-MLP input after Norm1).",
    },
    PhaseContract {
        phase: PipelinePhase::Norm2,
        inputs: &[
            TensorContract { name: "hidden", role: TensorRole::PrimaryInput, shape_pattern: &[Known(1), Symbol("hidden_dim")], dtype: "float32" },
        ],
        outputs: &[
            TensorContract { name: "hidden", role: TensorRole::Output, shape_pattern: &[Known(1), Symbol("hidden_dim")], dtype: "float32" },
        ],
        tolerance: 1e-4,
        description: "Second layer normalization (RMS norm or LayerNorm) at pre-LM Head boundary.",
    },
    PhaseContract {
        phase: PipelinePhase::LmHead,
        inputs: &[
            TensorContract { name: "hidden", role: TensorRole::PrimaryInput, shape_pattern: &[Known(1), Symbol("hidden_dim")], dtype: "float32" },
        ],
        outputs: &[
            TensorContract { name: "logits", role: TensorRole::Output, shape_pattern: &[Known(1), Symbol("vocab_size")], dtype: "float32" },
        ],
        tolerance: 1e-3,
        description: "Language model head: project hidden states to logits over vocabulary.",
    },
    PhaseContract {
        phase: PipelinePhase::SamplingOrLogitsPostprocess,
        inputs: &[
            TensorContract { name: "logits", role: TensorRole::PrimaryInput, shape_pattern: &[Known(1), Symbol("vocab_size")], dtype: "float32" },
        ],
        outputs: &[
            TensorContract { name: "token_id", role: TensorRole::Output, shape_pattern: &[Known(1)], dtype: "int32" },
        ],
        tolerance: 0.0,
        description: "Sample from logits or apply post-processing (temperature, top-k, top-p). Non-differentiable.",
    },
];
