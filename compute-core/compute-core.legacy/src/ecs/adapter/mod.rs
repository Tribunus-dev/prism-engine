//! Model-family adapter layer: normalises diverse model sources (safetensors +
//! HuggingFace config.json) into a canonical representation that the ComputeImage
//! compiler can lower to backend-specific artifacts.
//!
//! # Architecture
//!
//! ```text
//! SourceModel (raw config.json + safetensor shards)
//!   → AdapterRegistry::select(config, tensor_names) → ModelFamilyAdapter
//!   → adapter.normalize(source) → CanonicalModel
//!   → compile.rs bridge → LoadedSource
//! ```
//!
//! # Adding a new family
//!
//! 1. Write a struct implementing `ModelFamilyAdapter`
//! 2. Register it in `AdapterRegistry::new()`
//! 3. Add a synthetic fixture in `fixtures.rs`
//! 4. Add a conformance case in `conformance.rs`

use crate::ecs::config::TextArchitecture;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::path::PathBuf;

// ═══════════════════════════════════════════════════════════════════════════
// Canonical roles
// ═══════════════════════════════════════════════════════════════════════════

/// Canonical tensor roles that every adapter maps source names to.
///
/// Layers are indexed from 0. Every layer index that the architecture declares
/// MUST have a complete set of required roles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CanonicalRole {
    Embedding,
    FinalNorm,
    LmHead,
    AttnNorm(u32),
    Q(u32),
    K(u32),
    V(u32),
    O(u32),
    MlpNorm(u32),
    Gate(u32),
    Up(u32),
    Down(u32),
    QNorm(u32),
    KNorm(u32),
    // ── DeepSeek V4 MoE ─────────────────────────────────────────────────
    /// Routed expert gate projection.
    GateEx(u32, u32),
    /// Routed expert up projection.
    UpEx(u32, u32),
    /// Routed expert down projection.
    DownEx(u32, u32),
    /// Expert router weight (per-layer).
    RouterWeight(u32),
    // ── Shared expert ───────────────────────────────────────────────────
    SharedGate,
    SharedUp,
    SharedDown,
    /// Per-layer shared expert gate projection.
    SharedGateL(u32),
    /// Per-layer shared expert up projection.
    SharedUpL(u32),
    /// Per-layer shared expert down projection.
    SharedDownL(u32),
    // ── Compressed sparse attention ─────────────────────────────────────
    /// KV compressor weight.
    CompressWeight(u32),
    /// Selective attention indexer weight.
    IndexerWeight(u32),
    /// Raw window KV cache projection.
    WindowK(u32),
    WindowV(u32),
    // ── Hyper-connection ────────────────────────────────────────────────
    /// Hyper-connection residual merge weight.
    HCWeight(u32),
}

impl std::fmt::Display for CanonicalRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CanonicalRole::Embedding => write!(f, "Embedding"),
            CanonicalRole::FinalNorm => write!(f, "FinalNorm"),
            CanonicalRole::LmHead => write!(f, "LmHead"),
            CanonicalRole::AttnNorm(i) => write!(f, "AttnNorm({})", i),
            CanonicalRole::Q(i) => write!(f, "Q({})", i),
            CanonicalRole::K(i) => write!(f, "K({})", i),
            CanonicalRole::V(i) => write!(f, "V({})", i),
            CanonicalRole::O(i) => write!(f, "O({})", i),
            CanonicalRole::MlpNorm(i) => write!(f, "MlpNorm({})", i),
            CanonicalRole::Gate(i) => write!(f, "Gate({})", i),
            CanonicalRole::Up(i) => write!(f, "Up({})", i),
            CanonicalRole::Down(i) => write!(f, "Down({})", i),
            CanonicalRole::QNorm(i) => write!(f, "QNorm({})", i),
            CanonicalRole::KNorm(i) => write!(f, "KNorm({})", i),
            CanonicalRole::GateEx(l, e) => write!(f, "GateEx({},{})", l, e),
            CanonicalRole::UpEx(l, e) => write!(f, "UpEx({},{})", l, e),
            CanonicalRole::DownEx(l, e) => write!(f, "DownEx({},{})", l, e),
            CanonicalRole::RouterWeight(l) => write!(f, "RouterWeight({})", l),
            CanonicalRole::SharedGate => write!(f, "SharedGate"),
            CanonicalRole::SharedUp => write!(f, "SharedUp"),
            CanonicalRole::SharedDown => write!(f, "SharedDown"),
            CanonicalRole::SharedGateL(l) => write!(f, "SharedGateL({})", l),
            CanonicalRole::SharedUpL(l) => write!(f, "SharedUpL({})", l),
            CanonicalRole::SharedDownL(l) => write!(f, "SharedDownL({})", l),
            CanonicalRole::CompressWeight(l) => write!(f, "CompressWeight({})", l),
            CanonicalRole::IndexerWeight(l) => write!(f, "IndexerWeight({})", l),
            CanonicalRole::WindowK(l) => write!(f, "WindowK({})", l),
            CanonicalRole::WindowV(l) => write!(f, "WindowV({})", l),
            CanonicalRole::HCWeight(l) => write!(f, "HCWeight({})", l),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Data types
// ═══════════════════════════════════════════════════════════════════════════

/// A single canonical tensor with raw bytes and metadata.
#[derive(Clone, Debug)]
pub struct TensorData {
    pub dtype: String,
    pub shape: Vec<u32>,
    pub data: Vec<u8>,
}

/// Raw source model before normalisation.
#[derive(Clone, Debug)]
pub struct SourceModel {
    pub config: Value,
    pub config_path: PathBuf,
    pub model_type: String,
    pub tensor_names: Vec<String>,
    /// Raw tensor data keyed by source tensor name.
    /// Each entry: (dtype, shape, raw_bytes).
    pub tensors: HashMap<String, (String, Vec<u32>, Vec<u8>)>,
}

/// Fully normalised model consumed by the compiler pipeline.
#[derive(Clone, Debug)]
pub struct CanonicalModel {
    /// Architecture parameters extracted and validated from config.
    pub architecture: TextArchitecture,
    /// Canonical role → actual tensor data.
    pub tensors: HashMap<CanonicalRole, TensorData>,
}

/// Human-readable normalisation failure.
#[derive(Clone, Debug)]
pub struct NormalizationReport {
    pub family: String,
    pub errors: Vec<String>,
    pub missing_roles: Vec<CanonicalRole>,
    pub shape_mismatches: Vec<String>,
}

impl std::fmt::Display for NormalizationReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "normalization failed for family '{}':", self.family)?;
        for e in &self.errors {
            writeln!(f, "  {}", e)?;
        }
        if !self.missing_roles.is_empty() {
            let roles: Vec<String> = self.missing_roles.iter().map(|r| r.to_string()).collect();
            writeln!(f, "  missing roles: {}", roles.join(", "))?;
        }
        if !self.shape_mismatches.is_empty() {
            for m in &self.shape_mismatches {
                writeln!(f, "  shape mismatch: {}", m)?;
            }
        }
        Ok(())
    }
}

/// A single tensor name pattern that maps source names to canonical roles.
#[derive(Clone)]
pub struct TensorPattern {
    /// Tensor name pattern with {layer} and {expert} placeholders.
    /// Global patterns have no placeholders.
    pub pattern: &'static str,
    /// Role constructor given layer and expert indices.
    /// For global roles, both are 0.
    pub role: fn(u32, u32) -> CanonicalRole,
    /// True if this is a global (non-layer, non-expert) pattern.
    pub is_global: bool,
    /// True if this pattern includes an expert index.
    pub is_expert: bool,
}

// ────────────────────────────────────────────────────────────────────────
// Pattern-based tensor name matching
// ────────────────────────────────────────────────────────────────────────
//
// All of the following was moved from ecs::system::model_load to consolidate
// the pattern engine alongside TensorPattern. The old copies in model_load.rs
// will be removed in a follow-up.

// ── Role constructor helpers (function pointers usable in statics) ─────

pub fn r_emb(_l: u32, _e: u32) -> CanonicalRole {
    CanonicalRole::Embedding
}
pub fn r_fnorm(_l: u32, _e: u32) -> CanonicalRole {
    CanonicalRole::FinalNorm
}
pub fn r_lmhead(_l: u32, _e: u32) -> CanonicalRole {
    CanonicalRole::LmHead
}
pub fn r_attn_norm(l: u32, _e: u32) -> CanonicalRole {
    CanonicalRole::AttnNorm(l)
}
pub fn r_q(l: u32, _e: u32) -> CanonicalRole {
    CanonicalRole::Q(l)
}
pub fn r_k(l: u32, _e: u32) -> CanonicalRole {
    CanonicalRole::K(l)
}
pub fn r_v(l: u32, _e: u32) -> CanonicalRole {
    CanonicalRole::V(l)
}
pub fn r_o(l: u32, _e: u32) -> CanonicalRole {
    CanonicalRole::O(l)
}
pub fn r_mlp_norm(l: u32, _e: u32) -> CanonicalRole {
    CanonicalRole::MlpNorm(l)
}
pub fn r_gate(l: u32, _e: u32) -> CanonicalRole {
    CanonicalRole::Gate(l)
}
pub fn r_up(l: u32, _e: u32) -> CanonicalRole {
    CanonicalRole::Up(l)
}
pub fn r_down(l: u32, _e: u32) -> CanonicalRole {
    CanonicalRole::Down(l)
}
pub fn r_qnorm(l: u32, _e: u32) -> CanonicalRole {
    CanonicalRole::QNorm(l)
}
pub fn r_knorm(l: u32, _e: u32) -> CanonicalRole {
    CanonicalRole::KNorm(l)
}
pub fn r_router(l: u32, _e: u32) -> CanonicalRole {
    CanonicalRole::RouterWeight(l)
}
pub fn r_gate_ex(l: u32, e: u32) -> CanonicalRole {
    CanonicalRole::GateEx(l, e)
}
pub fn r_up_ex(l: u32, e: u32) -> CanonicalRole {
    CanonicalRole::UpEx(l, e)
}
pub fn r_down_ex(l: u32, e: u32) -> CanonicalRole {
    CanonicalRole::DownEx(l, e)
}
pub fn r_shared_gate(_l: u32, _e: u32) -> CanonicalRole {
    CanonicalRole::SharedGate
}
pub fn r_shared_up(_l: u32, _e: u32) -> CanonicalRole {
    CanonicalRole::SharedUp
}
pub fn r_shared_down(_l: u32, _e: u32) -> CanonicalRole {
    CanonicalRole::SharedDown
}
pub fn r_shared_gate_l(l: u32, _e: u32) -> CanonicalRole {
    CanonicalRole::SharedGateL(l)
}
pub fn r_shared_up_l(l: u32, _e: u32) -> CanonicalRole {
    CanonicalRole::SharedUpL(l)
}
pub fn r_shared_down_l(l: u32, _e: u32) -> CanonicalRole {
    CanonicalRole::SharedDownL(l)
}
pub fn r_compress(l: u32, _e: u32) -> CanonicalRole {
    CanonicalRole::CompressWeight(l)
}
pub fn r_indexer(l: u32, _e: u32) -> CanonicalRole {
    CanonicalRole::IndexerWeight(l)
}
pub fn r_window_k(l: u32, _e: u32) -> CanonicalRole {
    CanonicalRole::WindowK(l)
}
pub fn r_window_v(l: u32, _e: u32) -> CanonicalRole {
    CanonicalRole::WindowV(l)
}
pub fn r_hc(l: u32, _e: u32) -> CanonicalRole {
    CanonicalRole::HCWeight(l)
}

// ── Pattern matching engine ────────────────────────────────────────────

/// Match a tensor name against a pattern, extracting layer and expert indices.
pub fn match_tensor_pattern(name: &str, tp: &TensorPattern) -> Option<(u32, u32)> {
    if tp.is_global {
        return if name == tp.pattern {
            Some((0, 0))
        } else {
            None
        };
    }

    let (prefix, suffix) = tp.pattern.split_once("{layer}")?;

    if !name.starts_with(prefix) {
        return None;
    }
    let after_prefix = &name[prefix.len()..];

    if tp.is_expert {
        // pattern = prefix + {layer} + infix + {expert} + suffix
        let (infix, suffix) = suffix.split_once("{expert}")?;
        let layer_end = after_prefix.find(|c: char| !c.is_ascii_digit())?;
        let layer: u32 = after_prefix[..layer_end].parse().ok()?;
        let after_layer_dot = after_prefix[layer_end..].strip_prefix(infix)?;
        let expert_end = after_layer_dot.find(|c: char| !c.is_ascii_digit())?;
        let expert: u32 = after_layer_dot[..expert_end].parse().ok()?;
        let rest = &after_layer_dot[expert_end..];
        if rest != suffix {
            return None;
        }
        Some((layer, expert))
    } else {
        // pattern = prefix + {layer} + suffix
        let layer_end = after_prefix.find(|c: char| !c.is_ascii_digit())?;
        let layer: u32 = after_prefix[..layer_end].parse().ok()?;
        let rest = &after_prefix[layer_end..];
        if rest != suffix {
            return None;
        }
        Some((layer, 0))
    }
}

// ────────────────────────────────────────────────────────────────────────
// Pattern arrays — one combined static per architecture family
// ────────────────────────────────────────────────────────────────────────

// ── Standard HuggingFace (qwen2, qwen3, kimi, moonshot) ────────────────

static STANDARD_HF_PATTERNS: &[TensorPattern] = &[
    // Global
    TensorPattern {
        pattern: "model.embed_tokens.weight",
        role: r_emb,
        is_global: true,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.norm.weight",
        role: r_fnorm,
        is_global: true,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.lm_head.weight",
        role: r_lmhead,
        is_global: true,
        is_expert: false,
    },
    // Per-layer attention
    TensorPattern {
        pattern: "model.layers.{layer}.input_layernorm.weight",
        role: r_attn_norm,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.self_attn.q_proj.weight",
        role: r_q,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.self_attn.k_proj.weight",
        role: r_k,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.self_attn.v_proj.weight",
        role: r_v,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.self_attn.o_proj.weight",
        role: r_o,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.post_attention_layernorm.weight",
        role: r_mlp_norm,
        is_global: false,
        is_expert: false,
    },
    // Per-layer MLP (dense)
    TensorPattern {
        pattern: "model.layers.{layer}.mlp.gate_proj.weight",
        role: r_gate,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.mlp.up_proj.weight",
        role: r_up,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.mlp.down_proj.weight",
        role: r_down,
        is_global: false,
        is_expert: false,
    },
    // MoE (qwen3_moe, etc.)
    TensorPattern {
        pattern: "model.layers.{layer}.mlp.router.weight",
        role: r_router,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight",
        role: r_gate_ex,
        is_global: false,
        is_expert: true,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.mlp.experts.{expert}.up_proj.weight",
        role: r_up_ex,
        is_global: false,
        is_expert: true,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.mlp.experts.{expert}.down_proj.weight",
        role: r_down_ex,
        is_global: false,
        is_expert: true,
    },
];

// ── Gemma / Ornith (model or model.language_model prefix, Q/K norms, MoE) ─

static GEMMA_PATTERNS: &[TensorPattern] = &[
    // Global — model prefix
    TensorPattern {
        pattern: "model.embed_tokens.weight",
        role: r_emb,
        is_global: true,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.norm.weight",
        role: r_fnorm,
        is_global: true,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.lm_head.weight",
        role: r_lmhead,
        is_global: true,
        is_expert: false,
    },
    // Global — model.language_model prefix
    TensorPattern {
        pattern: "model.language_model.embed_tokens.weight",
        role: r_emb,
        is_global: true,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.language_model.norm.weight",
        role: r_fnorm,
        is_global: true,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.language_model.lm_head.weight",
        role: r_lmhead,
        is_global: true,
        is_expert: false,
    },
    // Per-layer attention — model prefix
    TensorPattern {
        pattern: "model.layers.{layer}.input_layernorm.weight",
        role: r_attn_norm,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.self_attn.q_proj.weight",
        role: r_q,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.self_attn.k_proj.weight",
        role: r_k,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.self_attn.v_proj.weight",
        role: r_v,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.self_attn.o_proj.weight",
        role: r_o,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.post_attention_layernorm.weight",
        role: r_mlp_norm,
        is_global: false,
        is_expert: false,
    },
    // Per-layer attention — model.language_model prefix
    TensorPattern {
        pattern: "model.language_model.layers.{layer}.input_layernorm.weight",
        role: r_attn_norm,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.language_model.layers.{layer}.self_attn.q_proj.weight",
        role: r_q,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.language_model.layers.{layer}.self_attn.k_proj.weight",
        role: r_k,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.language_model.layers.{layer}.self_attn.v_proj.weight",
        role: r_v,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.language_model.layers.{layer}.self_attn.o_proj.weight",
        role: r_o,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.language_model.layers.{layer}.post_attention_layernorm.weight",
        role: r_mlp_norm,
        is_global: false,
        is_expert: false,
    },
    // Q/K norms (Gemma2+) — both prefixes
    TensorPattern {
        pattern: "model.layers.{layer}.self_attn.q_norm.weight",
        role: r_qnorm,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.self_attn.k_norm.weight",
        role: r_knorm,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.language_model.layers.{layer}.self_attn.q_norm.weight",
        role: r_qnorm,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.language_model.layers.{layer}.self_attn.k_norm.weight",
        role: r_knorm,
        is_global: false,
        is_expert: false,
    },
    // Dense MLP — both prefixes
    TensorPattern {
        pattern: "model.layers.{layer}.mlp.gate_proj.weight",
        role: r_gate,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.mlp.up_proj.weight",
        role: r_up,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.mlp.down_proj.weight",
        role: r_down,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.language_model.layers.{layer}.mlp.gate_proj.weight",
        role: r_gate,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.language_model.layers.{layer}.mlp.up_proj.weight",
        role: r_up,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.language_model.layers.{layer}.mlp.down_proj.weight",
        role: r_down,
        is_global: false,
        is_expert: false,
    },
    // MoE router — both prefixes
    TensorPattern {
        pattern: "model.layers.{layer}.mlp.router.weight",
        role: r_router,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.language_model.layers.{layer}.mlp.router.weight",
        role: r_router,
        is_global: false,
        is_expert: false,
    },
    // MoE experts (standard naming) — both prefixes
    TensorPattern {
        pattern: "model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight",
        role: r_gate_ex,
        is_global: false,
        is_expert: true,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.mlp.experts.{expert}.up_proj.weight",
        role: r_up_ex,
        is_global: false,
        is_expert: true,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.mlp.experts.{expert}.down_proj.weight",
        role: r_down_ex,
        is_global: false,
        is_expert: true,
    },
    TensorPattern {
        pattern: "model.language_model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight",
        role: r_gate_ex,
        is_global: false,
        is_expert: true,
    },
    TensorPattern {
        pattern: "model.language_model.layers.{layer}.mlp.experts.{expert}.up_proj.weight",
        role: r_up_ex,
        is_global: false,
        is_expert: true,
    },
    TensorPattern {
        pattern: "model.language_model.layers.{layer}.mlp.experts.{expert}.down_proj.weight",
        role: r_down_ex,
        is_global: false,
        is_expert: true,
    },
    // Gemma MoE alternative naming: mlp.{proj}_proj.{ex}.weight
    TensorPattern {
        pattern: "model.layers.{layer}.mlp.gate_proj.{expert}.weight",
        role: r_gate_ex,
        is_global: false,
        is_expert: true,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.mlp.up_proj.{expert}.weight",
        role: r_up_ex,
        is_global: false,
        is_expert: true,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.mlp.down_proj.{expert}.weight",
        role: r_down_ex,
        is_global: false,
        is_expert: true,
    },
    TensorPattern {
        pattern: "model.language_model.layers.{layer}.mlp.gate_proj.{expert}.weight",
        role: r_gate_ex,
        is_global: false,
        is_expert: true,
    },
    TensorPattern {
        pattern: "model.language_model.layers.{layer}.mlp.up_proj.{expert}.weight",
        role: r_up_ex,
        is_global: false,
        is_expert: true,
    },
    TensorPattern {
        pattern: "model.language_model.layers.{layer}.mlp.down_proj.{expert}.weight",
        role: r_down_ex,
        is_global: false,
        is_expert: true,
    },
];

// ── DiffusionGemma (same as Gemma but model prefix only) ───────────────

static DIFFUSION_GEMMA_PATTERNS: &[TensorPattern] = &[
    TensorPattern {
        pattern: "model.embed_tokens.weight",
        role: r_emb,
        is_global: true,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.norm.weight",
        role: r_fnorm,
        is_global: true,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.lm_head.weight",
        role: r_lmhead,
        is_global: true,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.input_layernorm.weight",
        role: r_attn_norm,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.self_attn.q_proj.weight",
        role: r_q,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.self_attn.k_proj.weight",
        role: r_k,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.self_attn.v_proj.weight",
        role: r_v,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.self_attn.o_proj.weight",
        role: r_o,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.post_attention_layernorm.weight",
        role: r_mlp_norm,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.self_attn.q_norm.weight",
        role: r_qnorm,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.self_attn.k_norm.weight",
        role: r_knorm,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.mlp.gate_proj.weight",
        role: r_gate,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.mlp.up_proj.weight",
        role: r_up,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.mlp.down_proj.weight",
        role: r_down,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.mlp.router.weight",
        role: r_router,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight",
        role: r_gate_ex,
        is_global: false,
        is_expert: true,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.mlp.experts.{expert}.up_proj.weight",
        role: r_up_ex,
        is_global: false,
        is_expert: true,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.mlp.experts.{expert}.down_proj.weight",
        role: r_down_ex,
        is_global: false,
        is_expert: true,
    },
];

// ── Phi (model or transformer prefix, wte / ln_f legacy names) ─────

static PHI_PATTERNS: &[TensorPattern] = &[
    // model prefix globals
    TensorPattern {
        pattern: "model.embed_tokens.weight",
        role: r_emb,
        is_global: true,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.wte.weight",
        role: r_emb,
        is_global: true,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.norm.weight",
        role: r_fnorm,
        is_global: true,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.ln_f.weight",
        role: r_fnorm,
        is_global: true,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.lm_head.weight",
        role: r_lmhead,
        is_global: true,
        is_expert: false,
    },
    // transformer prefix globals
    TensorPattern {
        pattern: "transformer.embed_tokens.weight",
        role: r_emb,
        is_global: true,
        is_expert: false,
    },
    TensorPattern {
        pattern: "transformer.wte.weight",
        role: r_emb,
        is_global: true,
        is_expert: false,
    },
    TensorPattern {
        pattern: "transformer.norm.weight",
        role: r_fnorm,
        is_global: true,
        is_expert: false,
    },
    TensorPattern {
        pattern: "transformer.ln_f.weight",
        role: r_fnorm,
        is_global: true,
        is_expert: false,
    },
    TensorPattern {
        pattern: "transformer.lm_head.weight",
        role: r_lmhead,
        is_global: true,
        is_expert: false,
    },
    // model.layers.{l}.*
    TensorPattern {
        pattern: "model.layers.{layer}.input_layernorm.weight",
        role: r_attn_norm,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.self_attn.q_proj.weight",
        role: r_q,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.self_attn.k_proj.weight",
        role: r_k,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.self_attn.v_proj.weight",
        role: r_v,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.self_attn.o_proj.weight",
        role: r_o,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.post_attention_layernorm.weight",
        role: r_mlp_norm,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.mlp.gate_proj.weight",
        role: r_gate,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.mlp.up_proj.weight",
        role: r_up,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.mlp.down_proj.weight",
        role: r_down,
        is_global: false,
        is_expert: false,
    },
    // transformer.h.{l}.*
    TensorPattern {
        pattern: "transformer.h.{layer}.input_layernorm.weight",
        role: r_attn_norm,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "transformer.h.{layer}.self_attn.q_proj.weight",
        role: r_q,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "transformer.h.{layer}.self_attn.k_proj.weight",
        role: r_k,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "transformer.h.{layer}.self_attn.v_proj.weight",
        role: r_v,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "transformer.h.{layer}.self_attn.o_proj.weight",
        role: r_o,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "transformer.h.{layer}.post_attention_layernorm.weight",
        role: r_mlp_norm,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "transformer.h.{layer}.mlp.gate_proj.weight",
        role: r_gate,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "transformer.h.{layer}.mlp.up_proj.weight",
        role: r_up,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "transformer.h.{layer}.mlp.down_proj.weight",
        role: r_down,
        is_global: false,
        is_expert: false,
    },
];

// ── GLM (transformer.encoder prefix, self_attention naming) ────────

static GLM_PATTERNS: &[TensorPattern] = &[
    // Standard HF globals
    TensorPattern {
        pattern: "model.embed_tokens.weight",
        role: r_emb,
        is_global: true,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.norm.weight",
        role: r_fnorm,
        is_global: true,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.lm_head.weight",
        role: r_lmhead,
        is_global: true,
        is_expert: false,
    },
    // GLM legacy globals
    TensorPattern {
        pattern: "transformer.embedding.word_embeddings.weight",
        role: r_emb,
        is_global: true,
        is_expert: false,
    },
    TensorPattern {
        pattern: "transformer.encoder.final_layernorm.weight",
        role: r_fnorm,
        is_global: true,
        is_expert: false,
    },
    TensorPattern {
        pattern: "transformer.output_layer.weight",
        role: r_lmhead,
        is_global: true,
        is_expert: false,
    },
    // Standard HF per-layer
    TensorPattern {
        pattern: "model.layers.{layer}.input_layernorm.weight",
        role: r_attn_norm,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.self_attn.q_proj.weight",
        role: r_q,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.self_attn.k_proj.weight",
        role: r_k,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.self_attn.v_proj.weight",
        role: r_v,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.self_attn.o_proj.weight",
        role: r_o,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.post_attention_layernorm.weight",
        role: r_mlp_norm,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.mlp.gate_proj.weight",
        role: r_gate,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.mlp.up_proj.weight",
        role: r_up,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.mlp.down_proj.weight",
        role: r_down,
        is_global: false,
        is_expert: false,
    },
    // GLM legacy per-layer (transformer.encoder.layers.{l}.*)
    TensorPattern {
        pattern: "transformer.encoder.layers.{layer}.input_layernorm.weight",
        role: r_attn_norm,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "transformer.encoder.layers.{layer}.self_attention.q_proj.weight",
        role: r_q,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "transformer.encoder.layers.{layer}.self_attention.k_proj.weight",
        role: r_k,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "transformer.encoder.layers.{layer}.self_attention.v_proj.weight",
        role: r_v,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "transformer.encoder.layers.{layer}.self_attention.o_proj.weight",
        role: r_o,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "transformer.encoder.layers.{layer}.post_attention_layernorm.weight",
        role: r_mlp_norm,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "transformer.encoder.layers.{layer}.mlp.gate_proj.weight",
        role: r_gate,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "transformer.encoder.layers.{layer}.mlp.up_proj.weight",
        role: r_up,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "transformer.encoder.layers.{layer}.mlp.down_proj.weight",
        role: r_down,
        is_global: false,
        is_expert: false,
    },
    // model.layers.{l}.self_attention fallback
    TensorPattern {
        pattern: "model.layers.{layer}.self_attention.q_proj.weight",
        role: r_q,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.self_attention.k_proj.weight",
        role: r_k,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.self_attention.v_proj.weight",
        role: r_v,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.self_attention.o_proj.weight",
        role: r_o,
        is_global: false,
        is_expert: false,
    },
];

// ── Llama (standard HF + ffn_gate/ex ffn_up/ex ffn_down/ex MoE) ────

static LLAMA_PATTERNS: &[TensorPattern] = &[
    TensorPattern {
        pattern: "model.embed_tokens.weight",
        role: r_emb,
        is_global: true,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.norm.weight",
        role: r_fnorm,
        is_global: true,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.lm_head.weight",
        role: r_lmhead,
        is_global: true,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.input_layernorm.weight",
        role: r_attn_norm,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.self_attn.q_proj.weight",
        role: r_q,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.self_attn.k_proj.weight",
        role: r_k,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.self_attn.v_proj.weight",
        role: r_v,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.self_attn.o_proj.weight",
        role: r_o,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.post_attention_layernorm.weight",
        role: r_mlp_norm,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.mlp.gate_proj.weight",
        role: r_gate,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.mlp.up_proj.weight",
        role: r_up,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.mlp.down_proj.weight",
        role: r_down,
        is_global: false,
        is_expert: false,
    },
    // Llama 4 MoE
    TensorPattern {
        pattern: "model.layers.{layer}.ffn_gate.{expert}.weight",
        role: r_gate_ex,
        is_global: false,
        is_expert: true,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.ffn_up.{expert}.weight",
        role: r_up_ex,
        is_global: false,
        is_expert: true,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.ffn_down.{expert}.weight",
        role: r_down_ex,
        is_global: false,
        is_expert: true,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.router.weight",
        role: r_router,
        is_global: false,
        is_expert: false,
    },
];

// ── Mistral (standard HF + BlockSparseMoe + MlpExperts) ─────────────

static MISTRAL_PATTERNS: &[TensorPattern] = &[
    TensorPattern {
        pattern: "model.embed_tokens.weight",
        role: r_emb,
        is_global: true,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.norm.weight",
        role: r_fnorm,
        is_global: true,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.lm_head.weight",
        role: r_lmhead,
        is_global: true,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.input_layernorm.weight",
        role: r_attn_norm,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.self_attn.q_proj.weight",
        role: r_q,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.self_attn.k_proj.weight",
        role: r_k,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.self_attn.v_proj.weight",
        role: r_v,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.self_attn.o_proj.weight",
        role: r_o,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.post_attention_layernorm.weight",
        role: r_mlp_norm,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.mlp.gate_proj.weight",
        role: r_gate,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.mlp.up_proj.weight",
        role: r_up,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.mlp.down_proj.weight",
        role: r_down,
        is_global: false,
        is_expert: false,
    },
    // BlockSparseMoe
    TensorPattern {
        pattern: "model.layers.{layer}.block_sparse_moe.gate.weight",
        role: r_router,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.block_sparse_moe.experts.{expert}.w1.weight",
        role: r_gate_ex,
        is_global: false,
        is_expert: true,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.block_sparse_moe.experts.{expert}.w2.weight",
        role: r_down_ex,
        is_global: false,
        is_expert: true,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.block_sparse_moe.experts.{expert}.w3.weight",
        role: r_up_ex,
        is_global: false,
        is_expert: true,
    },
    // MlpExperts
    TensorPattern {
        pattern: "model.layers.{layer}.mlp.router.weight",
        role: r_router,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight",
        role: r_gate_ex,
        is_global: false,
        is_expert: true,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.mlp.experts.{expert}.up_proj.weight",
        role: r_up_ex,
        is_global: false,
        is_expert: true,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.mlp.experts.{expert}.down_proj.weight",
        role: r_down_ex,
        is_global: false,
        is_expert: true,
    },
];

// ── DeepSeek V4 (blk.{l}.* prefix, MLA, shared expert, compressor) ──

static DS4_PATTERNS: &[TensorPattern] = &[
    // Global
    TensorPattern {
        pattern: "embed.word.weight",
        role: r_emb,
        is_global: true,
        is_expert: false,
    },
    TensorPattern {
        pattern: "embed_tokens.weight",
        role: r_emb,
        is_global: true,
        is_expert: false,
    },
    TensorPattern {
        pattern: "output_norm.weight",
        role: r_fnorm,
        is_global: true,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.norm.weight",
        role: r_fnorm,
        is_global: true,
        is_expert: false,
    },
    TensorPattern {
        pattern: "head.weight",
        role: r_lmhead,
        is_global: true,
        is_expert: false,
    },
    TensorPattern {
        pattern: "lm_head.weight",
        role: r_lmhead,
        is_global: true,
        is_expert: false,
    },
    // Shared expert
    TensorPattern {
        pattern: "shared_ffn.gate.weight",
        role: r_shared_gate,
        is_global: true,
        is_expert: false,
    },
    TensorPattern {
        pattern: "shared_ffn.up.weight",
        role: r_shared_up,
        is_global: true,
        is_expert: false,
    },
    TensorPattern {
        pattern: "shared_ffn.down.weight",
        role: r_shared_down,
        is_global: true,
        is_expert: false,
    },
    // Per-layer
    TensorPattern {
        pattern: "blk.{layer}.attn_norm.weight",
        role: r_attn_norm,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "blk.{layer}.attn_q_a.weight",
        role: r_q,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "blk.{layer}.attn_q_b.weight",
        role: r_qnorm,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "blk.{layer}.attn_kv_a.weight",
        role: r_k,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "blk.{layer}.attn_kv_b.weight",
        role: r_v,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "blk.{layer}.attn_o.weight",
        role: r_o,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "blk.{layer}.ffn_norm.weight",
        role: r_mlp_norm,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "blk.{layer}.ffn_gate.{expert}.weight",
        role: r_gate_ex,
        is_global: false,
        is_expert: true,
    },
    TensorPattern {
        pattern: "blk.{layer}.ffn_up.{expert}.weight",
        role: r_up_ex,
        is_global: false,
        is_expert: true,
    },
    TensorPattern {
        pattern: "blk.{layer}.ffn_down.{expert}.weight",
        role: r_down_ex,
        is_global: false,
        is_expert: true,
    },
    TensorPattern {
        pattern: "blk.{layer}.ffn_router.weight",
        role: r_router,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "blk.{layer}.compress.weight",
        role: r_compress,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "blk.{layer}.indexer.weight",
        role: r_indexer,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "blk.{layer}.hc.weight",
        role: r_hc,
        is_global: false,
        is_expert: false,
    },
];

// ── MiniMax (language_model.model prefix, shared expert per-layer) ───

static MINIMAX_PATTERNS: &[TensorPattern] = &[
    TensorPattern {
        pattern: "language_model.model.embed_tokens.weight",
        role: r_emb,
        is_global: true,
        is_expert: false,
    },
    TensorPattern {
        pattern: "language_model.model.norm.weight",
        role: r_fnorm,
        is_global: true,
        is_expert: false,
    },
    TensorPattern {
        pattern: "language_model.lm_head.weight",
        role: r_lmhead,
        is_global: true,
        is_expert: false,
    },
    TensorPattern {
        pattern: "language_model.model.lm_head.weight",
        role: r_lmhead,
        is_global: true,
        is_expert: false,
    },
    TensorPattern {
        pattern: "language_model.model.layers.{layer}.input_layernorm.weight",
        role: r_attn_norm,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "language_model.model.layers.{layer}.self_attn.q_proj.weight",
        role: r_q,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "language_model.model.layers.{layer}.self_attn.k_proj.weight",
        role: r_k,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "language_model.model.layers.{layer}.self_attn.v_proj.weight",
        role: r_v,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "language_model.model.layers.{layer}.self_attn.o_proj.weight",
        role: r_o,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "language_model.model.layers.{layer}.self_attn.q_norm.weight",
        role: r_qnorm,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "language_model.model.layers.{layer}.self_attn.k_norm.weight",
        role: r_knorm,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "language_model.model.layers.{layer}.post_attention_layernorm.weight",
        role: r_mlp_norm,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "language_model.model.layers.{layer}.mlp.router.weight",
        role: r_router,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "language_model.model.layers.{layer}.block_sparse_moe.gate.weight",
        role: r_router,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "language_model.model.layers.{layer}.mlp.shared_expert.gate_proj.weight",
        role: r_shared_gate_l,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "language_model.model.layers.{layer}.mlp.shared_expert.up_proj.weight",
        role: r_shared_up_l,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "language_model.model.layers.{layer}.mlp.shared_expert.down_proj.weight",
        role: r_shared_down_l,
        is_global: false,
        is_expert: false,
    },
    // MiniMax block_sparse_moe naming (w1=gate, w2=down, w3=up)
    TensorPattern {
        pattern: "language_model.model.layers.{layer}.block_sparse_moe.experts.{expert}.w1.weight",
        role: r_gate_ex,
        is_global: false,
        is_expert: true,
    },
    TensorPattern {
        pattern: "language_model.model.layers.{layer}.block_sparse_moe.experts.{expert}.w2.weight",
        role: r_down_ex,
        is_global: false,
        is_expert: true,
    },
    TensorPattern {
        pattern: "language_model.model.layers.{layer}.block_sparse_moe.experts.{expert}.w3.weight",
        role: r_up_ex,
        is_global: false,
        is_expert: true,
    },
    // MiniMax also uses mlp.experts naming
    TensorPattern {
        pattern: "language_model.model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight",
        role: r_gate_ex,
        is_global: false,
        is_expert: true,
    },
    TensorPattern {
        pattern: "language_model.model.layers.{layer}.mlp.experts.{expert}.up_proj.weight",
        role: r_up_ex,
        is_global: false,
        is_expert: true,
    },
    TensorPattern {
        pattern: "language_model.model.layers.{layer}.mlp.experts.{expert}.down_proj.weight",
        role: r_down_ex,
        is_global: false,
        is_expert: true,
    },
    // Dense MLP for early layers
    TensorPattern {
        pattern: "language_model.model.layers.{layer}.mlp.gate_proj.weight",
        role: r_gate,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "language_model.model.layers.{layer}.mlp.up_proj.weight",
        role: r_up,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "language_model.model.layers.{layer}.mlp.down_proj.weight",
        role: r_down,
        is_global: false,
        is_expert: false,
    },
];

// ── GPT-OSS (standard HF + MoE) ─────────────────────────────────────

static GPT_OSS_PATTERNS: &[TensorPattern] = &[
    TensorPattern {
        pattern: "model.embed_tokens.weight",
        role: r_emb,
        is_global: true,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.norm.weight",
        role: r_fnorm,
        is_global: true,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.lm_head.weight",
        role: r_lmhead,
        is_global: true,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.input_layernorm.weight",
        role: r_attn_norm,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.self_attn.q_proj.weight",
        role: r_q,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.self_attn.k_proj.weight",
        role: r_k,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.self_attn.v_proj.weight",
        role: r_v,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.self_attn.o_proj.weight",
        role: r_o,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.post_attention_layernorm.weight",
        role: r_mlp_norm,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.mlp.gate_proj.weight",
        role: r_gate,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.mlp.up_proj.weight",
        role: r_up,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.mlp.down_proj.weight",
        role: r_down,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.mlp.router.weight",
        role: r_router,
        is_global: false,
        is_expert: false,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight",
        role: r_gate_ex,
        is_global: false,
        is_expert: true,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.mlp.experts.{expert}.up_proj.weight",
        role: r_up_ex,
        is_global: false,
        is_expert: true,
    },
    TensorPattern {
        pattern: "model.layers.{layer}.mlp.experts.{expert}.down_proj.weight",
        role: r_down_ex,
        is_global: false,
        is_expert: true,
    },
];

// ────────────────────────────────────────────────────────────────────────
// Architecture → pattern dispatch
// ────────────────────────────────────────────────────────────────────────

pub fn architecture_patterns(model_type: &str) -> &'static [TensorPattern] {
    match model_type {
        "qwen2" | "qwen3" | "qwen3_moe" | "kimi" | "kimi_k2_7" | "moonshot" => STANDARD_HF_PATTERNS,
        "gemma"
        | "gemma2"
        | "gemma4"
        | "gemma4_unified"
        | "gemma4_unified_text"
        | "gemma4_unified_assistant"
        | "gemma4_moe_26b"
        | "gemma4_dense_31b"
        | "gemma4_e2b" => GEMMA_PATTERNS,
        "diffusion_gemma" => DIFFUSION_GEMMA_PATTERNS,
        "phi" | "phi3" | "phimoe" => PHI_PATTERNS,
        "glm" | "glm_5" | "glm_5_2" => GLM_PATTERNS,
        "llama" | "llama4" | "llama4_scout" | "llama4_maverick" | "llama4_moe" => LLAMA_PATTERNS,
        "mistral" | "mistral_large_3" | "mistral_small_4" | "leanstral" | "leanstral_1_5" => {
            MISTRAL_PATTERNS
        }
        "deepseek4_flash" | "deepseek4" => DS4_PATTERNS,
        "minimax_m3_vl" => MINIMAX_PATTERNS,
        "gpt_oss" | "gpt_oss_moe" => GPT_OSS_PATTERNS,
        "ornith" | "ornith_moe" => GEMMA_PATTERNS,
        _ => &[],
    }
}

// ────────────────────────────────────────────────────────────────────────
// Adapter registry — backward-compat shim
// ────────────────────────────────────────────────────────────────────────

/// Trait for model-family adapters that normalise raw HuggingFace model
/// sources into canonical form.
pub trait ModelFamilyAdapter: Send + Sync {
    fn family_name(&self) -> &'static str;
    fn claimed_config_types(&self) -> &'static [&'static str];
    fn detect(&self, config: &Value, tensor_names: &[String]) -> bool;
    fn normalize(&self, source: &SourceModel) -> Result<CanonicalModel, NormalizationReport>;
}

/// A default adapter implementation that uses the pattern engine to map
/// tensor names to canonical roles.
pub struct PatternAdapter {
    pub family: &'static str,
    pub model_types: &'static [&'static str],
}

impl ModelFamilyAdapter for PatternAdapter {
    fn family_name(&self) -> &'static str {
        self.family
    }

    fn claimed_config_types(&self) -> &'static [&'static str] {
        self.model_types
    }

    fn detect(&self, config: &Value, tensor_names: &[String]) -> bool {
        let mt = config
            .get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let matches_type = self
            .model_types
            .iter()
            .any(|t| *t == mt || (mt.starts_with(t) && t.len() >= 2));
        matches_type && tensor_names.iter().any(|n| n.contains(".weight"))
    }

    fn normalize(&self, source: &SourceModel) -> Result<CanonicalModel, NormalizationReport> {
        let patterns = architecture_patterns(&source.model_type);
        if patterns.is_empty() {
            return Err(NormalizationReport {
                family: self.family.into(),
                errors: vec![format!(
                    "no patterns registered for model_type '{}'",
                    source.model_type
                )],
                missing_roles: vec![],
                shape_mismatches: vec![],
            });
        }
        let mut tensors = HashMap::new();
        for name in &source.tensor_names {
            for tp in patterns.iter() {
                if let Some((layer, expert)) = match_tensor_pattern(name, tp) {
                    let role = (tp.role)(layer, expert);
                    if let Some(t) = source.tensors.get(name) {
                        tensors.insert(
                            role,
                            TensorData {
                                dtype: t.0.clone(),
                                shape: t.1.clone(),
                                data: t.2.clone(),
                            },
                        );
                    }
                }
            }
        }
        // Construct TextArchitecture manually — it does not derive Default.
        // We extract model_type so the dispatch works; other fields are left
        // at zero/default values for this backward-compat shim. Production
        // callers should use the ECS ModelAdapterSystem instead.
        let architecture = TextArchitecture {
            hidden_size: 0,
            intermediate_size: 0,
            num_attention_heads: 0,
            num_key_value_heads: 0,
            head_dim: 0,
            global_head_dim: None,
            num_global_key_value_heads: None,
            num_hidden_layers: 0,
            vocab_size: 0,
            sliding_window: 0,
            max_position_embeddings: 0,
            rms_norm_eps: 0.0,
            tie_word_embeddings: false,
            attention_k_eq_v: false,
            final_logit_softcapping: None,
            hidden_size_per_layer_input: 0,
            layer_types: vec![],
            rope_local: crate::ecs::config::hardware::RopeSpec {
                theta: 10000.0,
                rope_type: "default".to_string(),
                partial_rotary_factor: None,
            },
            rope_global: None,
            model_type: source.model_type.clone(),
            moe_config: None,
            diffusion_config: None,
            thinking_mode: false,
        };
        Ok(CanonicalModel {
            architecture,
            tensors,
        })
    }
}

/// A registry of model-family adapters.
///
/// This is a backward-compatible shim; new code should use the ECS
/// `ModelAdapterSystem` instead.
pub struct AdapterRegistry {
    adapters: Vec<PatternAdapter>,
}

impl AdapterRegistry {
    pub fn new() -> Self {
        Self {
            adapters: BUILTIN_ADAPTERS
                .iter()
                .map(|a| PatternAdapter {
                    family: a.family,
                    model_types: a.model_types,
                })
                .collect(),
        }
    }

    pub fn select_by_config_type(&self, config: &Value) -> Result<&dyn ModelFamilyAdapter, String> {
        let model_type = config
            .get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        for adapter in &self.adapters {
            if adapter.model_types.iter().any(|t| *t == model_type) {
                return Ok(adapter as &dyn ModelFamilyAdapter);
            }
        }
        Err(format!("unsupported model_type '{}'", model_type))
    }

    pub fn select(
        &self,
        config: &Value,
        tensor_names: &[String],
    ) -> Result<&dyn ModelFamilyAdapter, String> {
        let model_type = config
            .get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        // First pass: try exact model_type match with tensor name evidence
        for adapter in &self.adapters {
            if adapter
                .claimed_config_types()
                .iter()
                .any(|t| *t == model_type)
                && adapter.detect(config, tensor_names)
            {
                return Ok(adapter as &dyn ModelFamilyAdapter);
            }
        }
        // Second pass: fall back to any adapter that claims this type
        for adapter in &self.adapters {
            if adapter
                .claimed_config_types()
                .iter()
                .any(|t| *t == model_type)
            {
                return Ok(adapter as &dyn ModelFamilyAdapter);
            }
        }
        Err(format!(
            "unsupported model_type '{}': no adapter matched",
            model_type
        ))
    }

    pub fn register(&mut self, adapter: PatternAdapter) {
        self.adapters.push(adapter);
    }
}

impl Default for AdapterRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Static adapter instances for the pattern-based system.
static STANDARD_ADAPTER: PatternAdapter = PatternAdapter {
    family: "qwen2",
    model_types: &[
        "qwen2",
        "qwen3",
        "qwen3_moe",
        "kimi",
        "kimi_k2_7",
        "moonshot",
    ],
};
static LLAMA_ADAPTER: PatternAdapter = PatternAdapter {
    family: "llama",
    model_types: &[
        "llama",
        "llama4",
        "llama4_moe",
        "llama_scout",
        "llama_guard",
    ],
};
static GEMMA_ADAPTER: PatternAdapter = PatternAdapter {
    family: "gemma",
    model_types: &[
        "gemma",
        "gemma2",
        "gemma4",
        "gemma4_unified",
        "gemma4_unified_text",
        "gemma4_unified_assistant",
        "gemma4_moe_26b",
        "gemma4_dense_31b",
        "gemma4_e2b",
        "gemma4_e4b",
        "ornith",
        "ornith_moe",
    ],
};
static MISTRAL_ADAPTER: PatternAdapter = PatternAdapter {
    family: "mistral",
    model_types: &[
        "mistral",
        "mistral_large_3",
        "mistral_small_4",
        "leanstral",
        "leanstral_1_5",
    ],
};
static PHI_ADAPTER: PatternAdapter = PatternAdapter {
    family: "phi",
    model_types: &["phi", "phi3", "phimoe"],
};
static GLM_ADAPTER: PatternAdapter = PatternAdapter {
    family: "glm",
    model_types: &["glm", "glm_5", "glm_5_2"],
};
static DS4_ADAPTER: PatternAdapter = PatternAdapter {
    family: "deepseek4_flash",
    model_types: &["deepseek4_flash", "deepseek4"],
};
static MINIMAX_ADAPTER: PatternAdapter = PatternAdapter {
    family: "minimax",
    model_types: &["minimax_m3_vl"],
};
static GPT_OSS_ADAPTER: PatternAdapter = PatternAdapter {
    family: "gpt_oss",
    model_types: &["gpt_oss", "gpt_oss_moe"],
};
static DIFFUSION_GEMMA_ADAPTER: PatternAdapter = PatternAdapter {
    family: "diffusion_gemma",
    model_types: &["diffusion_gemma"],
};

/// All built-in adapters.
const BUILTIN_ADAPTERS: &[&PatternAdapter] = &[
    &STANDARD_ADAPTER,
    &LLAMA_ADAPTER,
    &GEMMA_ADAPTER,
    &MISTRAL_ADAPTER,
    &PHI_ADAPTER,
    &GLM_ADAPTER,
    &DS4_ADAPTER,
    &MINIMAX_ADAPTER,
    &GPT_OSS_ADAPTER,
    &DIFFUSION_GEMMA_ADAPTER,
];
