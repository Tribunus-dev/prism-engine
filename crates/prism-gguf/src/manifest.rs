//! Manifest extraction — read GGUF metadata into a typed model architecture.
//!
//! This file owns the canonical authority for converting GGUF metadata
//! (the `Vec<(String, String)>` key/value pairs returned by the format
//! parser in [`crate::lib`]) into a typed [`TextArchitecture`]. The
//! extraction logic — architecture-prefixed key resolution with fallback
//! to the generic `llama.*` namespace — is the unique design contribution
//! that used to live in the engine's `compute-core/src/ecs/core/gguf.rs`.
//!
//! Per the project-absorption rules, the format adapter (`prism-gguf`)
//! is the right home for this authority: the GGUF key namespace is the
//! public contract the file serves, and the adapter's job is to turn
//! that namespace into Prism's domain types. Downstream code consumes
//! the resulting [`TextArchitecture`] as input; it never re-parses the
//! raw metadata.
//!
//! # Key resolution
//!
//! GGUF keys are architecture-prefixed: a Llama model stores
//! `llama.embedding_length`, while a Gemma model stores
//! `gemma4.embedding_length`. The extraction logic looks up the
//! architecture-prefixed key first (using the value of
//! `general.architecture`) and falls back to the generic `llama.*`
//! namespace when no prefix-specific key is present. This is the
//! canonical pattern in llama.cpp's GGUF reader and is preserved here.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::GgufImportResult;

// ── GGUF metadata key namespace ─────────────────────────────────────────────

/// Canonical GGUF metadata keys used by the manifest extractor. These are
/// the public keys as defined by the GGUF specification and by
/// llama.cpp's reference implementation. Keys with the `llama.` prefix
/// are the generic namespace; architecture-specific keys (e.g.
/// `gemma4.embedding_length`) take precedence when present.
pub mod keys {
    pub const VOCAB_SIZE: &str = "llama.vocab_size";
    pub const HIDDEN_SIZE: &str = "llama.embedding_length";
    pub const INTERMEDIATE_SIZE: &str = "llama.feed_forward_length";
    pub const NUM_HIDDEN_LAYERS: &str = "llama.block_count";
    pub const NUM_ATTENTION_HEADS: &str = "llama.attention.head_count";
    pub const NUM_KV_HEADS: &str = "llama.attention.head_count_kv";
    pub const HEAD_DIM: &str = "llama.attention.head_dim";
    pub const GLOBAL_HEAD_DIM: &str = "llama.attention.key_length";
    pub const MAX_SEQ_LEN: &str = "llama.context_length";
    pub const ROPE_THETA: &str = "llama.rope.freq_base";
    pub const NORM_EPS: &str = "llama.attention.layer_norm_rms_epsilon";
    pub const SLIDING_WINDOW: &str = "llama.attention.sliding_window";
    pub const TIE_WORD_EMBEDDINGS: &str = "llama.tie_embeddings";
    pub const MODEL_TYPE: &str = "general.architecture";
    pub const QUANTIZATION_VERSION: &str = "general.quantization_version";
    pub const FILE_TYPE: &str = "general.file_type";
    pub const EXPERT_COUNT: &str = "llama.expert_count";
    pub const EXPERT_USED_COUNT: &str = "llama.expert_used_count";
    pub const MOE_FEED_FORWARD_LENGTH: &str = "llama.expert_feed_forward_length";
}

// ── Error type ─────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum ManifestError {
    /// A required metadata key was missing entirely from the GGUF file.
    /// The file is malformed or comes from a model family we do not yet
    /// support.
    #[error("missing required GGUF key: {0}")]
    MissingKey(&'static str),

    /// A metadata key was present but its value could not be parsed into
    /// the expected type. We distinguish this from `MissingKey` because
    /// the file *has* the key — it just has a malformed value.
    #[error("invalid value for GGUF key {key}: {reason}")]
    InvalidValue {
        key: &'static str,
        reason: String,
    },
}

// ── Domain types ───────────────────────────────────────────────────────────

/// Attention kind for a single transformer layer. Maps to the
/// `llama.attention.layer_types` array when present; defaults to
/// [`AttentionKind::Sliding`] for models that do not declare per-layer
/// kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionKind {
    /// Sliding-window attention (used in Gemma 3, Mistral).
    #[default]
    Sliding,
    /// Full causal attention (used in most dense transformers).
    Full,
}

impl AttentionKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Sliding => "sliding_attention",
            Self::Full => "full_attention",
        }
    }
}

/// Rotary Position Embedding specification. The RoPE theta controls the
/// base of the frequency spectrum; some models (Gemma 3, Qwen) use
/// distinct RoPE configs for local and global attention.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RopeSpec {
    /// Frequency base (default 10000.0).
    pub theta: f64,
    /// Optional partial-rotary factor (Qwen2/3 use 0.5).
    pub partial_rotary_factor: Option<f64>,
}

impl Default for RopeSpec {
    fn default() -> Self {
        Self {
            theta: 10_000.0,
            partial_rotary_factor: None,
        }
    }
}

/// Mixture-of-Experts configuration, if the model uses MoE.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MoeConfig {
    /// Total number of experts per layer.
    pub num_experts: u32,
    /// Number of experts activated per token.
    pub num_experts_used: u32,
}

/// Text-architecture manifest extracted from a GGUF file.
///
/// This is the typed output of manifest extraction. The struct is
/// serde-serialisable so downstream code (the compile path, the engine)
/// can persist or transmit it without re-parsing the raw GGUF metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextArchitecture {
    /// Hidden dimension of the transformer.
    pub hidden_size: u32,
    /// Feed-forward intermediate dimension.
    pub intermediate_size: u32,
    /// Number of attention heads.
    pub num_attention_heads: u32,
    /// Number of key/value heads (GQA: usually less than
    /// `num_attention_heads`).
    pub num_key_value_heads: u32,
    /// Per-head dimension.
    pub head_dim: u32,
    /// Per-head dimension for global attention (Gemma 3 global layers).
    /// `None` when the model has a single head dim.
    pub global_head_dim: Option<u32>,
    /// Number of transformer layers.
    pub num_hidden_layers: u32,
    /// Vocabulary size.
    pub vocab_size: u32,
    /// Sliding-window size for layers that use it. `0` when the model
    /// has no sliding window.
    pub sliding_window: u32,
    /// Maximum sequence length the model supports.
    pub max_position_embeddings: u32,
    /// RMS normalization epsilon.
    pub rms_norm_eps: f64,
    /// Whether input and output embeddings are tied.
    pub tie_word_embeddings: bool,
    /// Attention kind per layer. Defaults to all-sliding when the GGUF
    /// does not declare per-layer kinds.
    pub layer_types: Vec<AttentionKind>,
    /// RoPE specification for local (sliding) attention.
    pub rope_local: RopeSpec,
    /// RoPE specification for global attention, when distinct.
    pub rope_global: Option<RopeSpec>,
    /// Model type (e.g. "gemma4", "llama", "qwen2"). From
    /// `general.architecture`.
    pub model_type: String,
    /// MoE configuration, when applicable.
    pub moe_config: Option<MoeConfig>,
}

impl TextArchitecture {
    /// Total parameter count (rough, embedding + per-layer only).
    /// This is the canonical "weight count" used by the engine's
    /// admission estimator and by the quantisation sweep.
    pub fn approx_weight_count(&self) -> u64 {
        let embed = (self.vocab_size as u64).saturating_mul(self.hidden_size as u64);
        let per_layer = self.per_layer_weight_count();
        embed
            .saturating_mul(2) // embedding + lm head (untied) or 1× if tied
            .saturating_add(
                per_layer.saturating_mul(self.num_hidden_layers as u64),
            )
    }

    fn per_layer_weight_count(&self) -> u64 {
        // Q, K, V, O projections: 4 × hidden × (head_dim × heads).
        let qkv = (self.hidden_size as u64)
            .saturating_mul((self.num_attention_heads as u64).saturating_mul(self.head_dim as u64))
            .saturating_mul(4);
        // MLP gate/up/down: 3 × hidden × intermediate.
        let mlp = (self.hidden_size as u64)
            .saturating_mul(self.intermediate_size as u64)
            .saturating_mul(3);
        qkv.saturating_add(mlp)
    }
}

// ── Public API ─────────────────────────────────────────────────────────────

/// Extract a [`TextArchitecture`] from a parsed GGUF result.
///
/// The metadata is taken from the [`GgufImportResult`] returned by the
/// format parser. The extraction is the canonical Prism-domain way to
/// turn raw GGUF metadata into a typed manifest — downstream code should
/// never re-implement the arch-prefixed key resolution.
///
/// # Errors
///
/// Returns [`ManifestError::MissingKey`] if a field required by the
/// architecture is absent from the metadata, and
/// [`ManifestError::InvalidValue`] if a field is present but cannot be
/// parsed. The default extraction is conservative: missing optional
/// fields (e.g. `global_head_dim`, `moe_config`) are set to `None`, not
/// treated as errors.
pub fn extract_architecture(import: &GgufImportResult) -> Result<TextArchitecture, ManifestError> {
    extract_architecture_from_metadata(&import.metadata)
}

/// Lower-level entry point that operates directly on the metadata slice.
/// Useful when the caller has already split out the metadata and does
/// not need the rest of the [`GgufImportResult`].
pub fn extract_architecture_from_metadata(
    metadata: &[(String, String)],
) -> Result<TextArchitecture, ManifestError> {
    let model_type = meta_str(metadata, keys::MODEL_TYPE).unwrap_or("unknown").to_string();
    let arch = model_type.as_str();

    let hidden_size = meta_u64(metadata, arch, keys::HIDDEN_SIZE, "hidden_size")?;
    let intermediate_size = meta_u64(metadata, arch, keys::INTERMEDIATE_SIZE, "intermediate_size")?;
    let num_attention_heads = meta_u64(metadata, arch, keys::NUM_ATTENTION_HEADS, "num_attention_heads")?;
    let num_key_value_heads = meta_u64(metadata, arch, keys::NUM_KV_HEADS, "num_key_value_heads")?;
    let head_dim = meta_u64(metadata, arch, keys::HEAD_DIM, "head_dim")?;
    let global_head_dim = meta_u64_flex(metadata, arch, keys::GLOBAL_HEAD_DIM);
    let num_hidden_layers = meta_u64(metadata, arch, keys::NUM_HIDDEN_LAYERS, "num_hidden_layers")?;
    let vocab_size = meta_u64(metadata, arch, keys::VOCAB_SIZE, "vocab_size")?;
    let max_position_embeddings = meta_u64(metadata, arch, keys::MAX_SEQ_LEN, "max_position_embeddings")?;
    let rms_norm_eps = meta_f64_flex(metadata, arch, keys::NORM_EPS).unwrap_or(1e-6);
    let sliding_window = meta_u64_flex(metadata, arch, keys::SLIDING_WINDOW);
    let rope_theta = meta_f64_flex(metadata, arch, keys::ROPE_THETA).unwrap_or(10_000.0);
    let tie_word_embeddings = meta_bool_flex(metadata, arch, keys::TIE_WORD_EMBEDDINGS);

    let moe_config = if let (Some(expert_count), Some(expert_used_count)) = (
        meta_u64_flex(metadata, arch, keys::EXPERT_COUNT),
        meta_u64_flex(metadata, arch, keys::EXPERT_USED_COUNT),
    ) {
        Some(MoeConfig {
            num_experts: expert_count as u32,
            num_experts_used: expert_used_count as u32,
        })
    } else {
        None
    };

    // Per-layer attention kinds. The GGUF spec does not standardise a
    // single key for this; the engine reads `llama.attention.layer_types`
    // as a comma-separated string. When absent, every layer defaults to
    // sliding attention (Gemma 3 convention) — downstream code overrides
    // this when it knows the model family.
    let layer_types = read_layer_types(metadata, arch, num_hidden_layers as usize);

    let rope_local = RopeSpec {
        theta: rope_theta,
        partial_rotary_factor: None,
    };
    let rope_global = if global_head_dim.is_some() {
        Some(rope_local)
    } else {
        None
    };

    Ok(TextArchitecture {
        hidden_size: hidden_size as u32,
        intermediate_size: intermediate_size as u32,
        num_attention_heads: num_attention_heads as u32,
        num_key_value_heads: num_key_value_heads as u32,
        head_dim: head_dim as u32,
        global_head_dim: global_head_dim.map(|v| v as u32),
        num_hidden_layers: num_hidden_layers as u32,
        vocab_size: vocab_size as u32,
        sliding_window: sliding_window.unwrap_or(0) as u32,
        max_position_embeddings: max_position_embeddings as u32,
        rms_norm_eps,
        tie_word_embeddings,
        layer_types,
        rope_local,
        rope_global,
        model_type,
        moe_config,
    })
}

/// Read the per-layer attention kinds. The GGUF format does not define
/// a single canonical key for this; the engine stored the kinds in
/// `llama.attention.layer_types` as a comma-separated string of
/// `"sliding"` or `"full"`. Returns a vector of `num_layers` entries
/// defaulting to [`AttentionKind::Sliding`] when the key is absent.
fn read_layer_types(
    metadata: &[(String, String)],
    arch: &str,
    num_layers: usize,
) -> Vec<AttentionKind> {
    let key = format!("{arch}.attention.layer_types");
    let raw = metadata
        .iter()
        .find(|(k, _)| *k == key || *k == "llama.attention.layer_types")
        .map(|(_, v)| v.as_str());
    let parsed: Option<Vec<AttentionKind>> = raw.and_then(|s| {
        s.split(',')
            .map(|tok| match tok.trim() {
                "sliding" => Some(AttentionKind::Sliding),
                "full" => Some(AttentionKind::Full),
                _ => None,
            })
            .collect()
    });
    parsed.unwrap_or_else(|| vec![AttentionKind::Sliding; num_layers])
}

// ── Key-resolution helpers ─────────────────────────────────────────────────

fn meta_str<'a>(metadata: &'a [(String, String)], key: &str) -> Option<&'a str> {
    metadata
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.as_str())
}

/// Look up a metadata value, trying the architecture-prefixed key first
/// and falling back to the generic key.
///
/// The generic key is the full `llama.X` key (e.g.
/// `"llama.embedding_length"`). The architecture-prefixed key replaces
/// the `llama.` prefix with `<arch>.` — so for arch `gemma4` and
/// generic key `llama.embedding_length`, the prefixed key is
/// `gemma4.embedding_length` (matching the llama.cpp convention). This
/// is the correct semantics: the engine's original helper formed
/// `gemma4.llama.embedding_length`, which never matched.
fn meta_val<'a>(
    metadata: &'a [(String, String)],
    arch: &str,
    generic_key: &str,
) -> Option<(String, &'a str)> {
    let without_llama = generic_key.strip_prefix("llama.").unwrap_or(generic_key);
    let with_prefix = format!("{arch}.{without_llama}");
    if let Some((k, v)) = metadata.iter().find(|(k, _)| k == &with_prefix) {
        return Some((k.clone(), v.as_str()));
    }
    metadata
        .iter()
        .find(|(k, _)| k == generic_key)
        .map(|(k, v)| (k.clone(), v.as_str()))
}

fn meta_u64_flex(metadata: &[(String, String)], arch: &str, generic_key: &str) -> Option<u64> {
    meta_val(metadata, arch, generic_key)
        .and_then(|(_, v)| v.parse::<u64>().ok())
}

fn meta_f64_flex(metadata: &[(String, String)], arch: &str, generic_key: &str) -> Option<f64> {
    meta_val(metadata, arch, generic_key)
        .and_then(|(_, v)| v.parse::<f64>().ok())
}

fn meta_bool_flex(metadata: &[(String, String)], arch: &str, generic_key: &str) -> bool {
    meta_val(metadata, arch, generic_key)
        .and_then(|(_, v)| v.parse::<bool>().ok())
        .unwrap_or(false)
}

/// Strict u64 lookup that returns [`ManifestError`] on miss or parse
/// failure. The `name` parameter is used in error messages and must
/// match the struct field name; it does not need to be unique.
fn meta_u64(
    metadata: &[(String, String)],
    arch: &str,
    generic_key: &str,
    name: &str,
) -> Result<u64, ManifestError> {
    match meta_val(metadata, arch, generic_key) {
        None => Err(ManifestError::MissingKey(generic_key_for(name))),
        Some((key, value)) => value
            .parse::<u64>()
            .map_err(|e| ManifestError::InvalidValue {
                key: leaked_static(&key),
                reason: format!("expected u64, got {:?}: {}", value, e),
            }),
    }
}

/// Map a Rust field name to a static GGUF key. For now this is a 1:1
/// table; the mapping is here so that a future rename of a field
/// automatically updates the error message.
fn generic_key_for(field: &str) -> &'static str {
    match field {
        "hidden_size" => keys::HIDDEN_SIZE,
        "intermediate_size" => keys::INTERMEDIATE_SIZE,
        "num_attention_heads" => keys::NUM_ATTENTION_HEADS,
        "num_key_value_heads" => keys::NUM_KV_HEADS,
        "head_dim" => keys::HEAD_DIM,
        "num_hidden_layers" => keys::NUM_HIDDEN_LAYERS,
        "vocab_size" => keys::VOCAB_SIZE,
        "max_position_embeddings" => keys::MAX_SEQ_LEN,
        _ => "unknown",
    }
}

/// `&String` → `&'static str` for error messages. The leak is bounded —
/// error formatting happens at most once per error, and the leaked
/// strings are never read in any performance-sensitive path. The keys
/// are short (< 80 bytes each) so the leak is small in practice.
fn leaked_static(s: &str) -> &'static str {
    Box::leak(s.to_string().into_boxed_str())
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    fn full_meta() -> Vec<(String, String)> {
        meta(&[
            ("general.architecture", "gemma4"),
            ("llama.vocab_size", "256128"),
            ("llama.embedding_length", "3072"),
            ("llama.feed_forward_length", "8192"),
            ("llama.block_count", "32"),
            ("llama.attention.head_count", "8"),
            ("llama.attention.head_count_kv", "4"),
            ("llama.attention.head_dim", "256"),
            ("llama.context_length", "8192"),
            ("llama.rope.freq_base", "10000.0"),
            ("llama.attention.layer_norm_rms_epsilon", "1e-6"),
            ("llama.attention.sliding_window", "4096"),
            ("gemma4.attention.layer_types", "sliding,sliding,sliding,sliding,full,full,full,full,sliding,sliding,sliding,sliding,sliding,sliding,sliding,sliding,sliding,sliding,sliding,sliding,sliding,sliding,sliding,sliding,sliding,sliding,sliding,sliding,sliding,sliding,sliding,full"),
        ])
    }

    #[test]
    fn extract_full_architecture() {
        let arch = extract_architecture_from_metadata(&full_meta()).expect("extract");
        assert_eq!(arch.model_type, "gemma4");
        assert_eq!(arch.hidden_size, 3072);
        assert_eq!(arch.intermediate_size, 8192);
        assert_eq!(arch.num_attention_heads, 8);
        assert_eq!(arch.num_key_value_heads, 4);
        assert_eq!(arch.head_dim, 256);
        assert_eq!(arch.num_hidden_layers, 32);
        assert_eq!(arch.vocab_size, 256128);
        assert_eq!(arch.max_position_embeddings, 8192);
        assert_eq!(arch.sliding_window, 4096);
        assert!((arch.rms_norm_eps - 1e-6).abs() < 1e-12);
        assert_eq!(arch.rope_local.theta, 10_000.0);

        // The test data has 5 "full" tokens (positions 4, 5, 6, 7, 31)
        // and 27 "sliding" tokens.
        let full_count = arch
            .layer_types
            .iter()
            .filter(|k| **k == AttentionKind::Full)
            .count();
        let sliding_count = arch
            .layer_types
            .iter()
            .filter(|k| **k == AttentionKind::Sliding)
            .count();
        assert_eq!(sliding_count + full_count, 32);
        assert_eq!(full_count, 5);
        assert_eq!(sliding_count, 27);
    }

    #[test]
    fn extract_minimal_architecture() {
        // No per-layer kinds declared → all default to sliding.
        let m = meta(&[
            ("general.architecture", "llama"),
            ("llama.vocab_size", "32000"),
            ("llama.embedding_length", "4096"),
            ("llama.feed_forward_length", "11008"),
            ("llama.block_count", "32"),
            ("llama.attention.head_count", "32"),
            ("llama.attention.head_count_kv", "32"),
            ("llama.attention.head_dim", "128"),
            ("llama.context_length", "4096"),
        ]);
        let arch = extract_architecture_from_metadata(&m).expect("extract");
        assert_eq!(arch.model_type, "llama");
        assert_eq!(arch.hidden_size, 4096);
        assert_eq!(arch.num_key_value_heads, 32);
        assert_eq!(arch.sliding_window, 0);
        assert!(arch.moe_config.is_none());
        assert_eq!(arch.layer_types.len(), 32);
        assert!(arch
            .layer_types
            .iter()
            .all(|k| *k == AttentionKind::Sliding));
    }

    #[test]
    fn arch_prefixed_key_takes_precedence() {
        // `gemma4.embedding_length` overrides `llama.embedding_length`.
        let m = meta(&[
            ("general.architecture", "gemma4"),
            ("gemma4.embedding_length", "1024"),
            ("llama.embedding_length", "9999"),
            ("llama.vocab_size", "100"),
            ("llama.feed_forward_length", "100"),
            ("llama.block_count", "1"),
            ("llama.attention.head_count", "1"),
            ("llama.attention.head_count_kv", "1"),
            ("llama.attention.head_dim", "32"),
            ("llama.context_length", "512"),
        ]);
        let arch = extract_architecture_from_metadata(&m).expect("extract");
        // The arch-prefixed value wins.
        assert_eq!(arch.hidden_size, 1024);
    }

    #[test]
    fn missing_required_key_returns_error() {
        let m = meta(&[("general.architecture", "llama")]);
        let err = extract_architecture_from_metadata(&m).unwrap_err();
        assert!(matches!(err, ManifestError::MissingKey(_)));
    }

    #[test]
    fn invalid_value_returns_error() {
        let m = meta(&[
            ("general.architecture", "llama"),
            ("llama.embedding_length", "not-a-number"),
            ("llama.vocab_size", "100"),
            ("llama.feed_forward_length", "100"),
            ("llama.block_count", "1"),
            ("llama.attention.head_count", "1"),
            ("llama.attention.head_count_kv", "1"),
            ("llama.attention.head_dim", "32"),
            ("llama.context_length", "512"),
        ]);
        let err = extract_architecture_from_metadata(&m).unwrap_err();
        match err {
            ManifestError::InvalidValue { key, .. } => {
                assert_eq!(key, keys::HIDDEN_SIZE);
            }
            other => panic!("expected InvalidValue, got {other:?}"),
        }
    }

    #[test]
    fn moe_config_extracted_when_present() {
        let m = meta(&[
            ("general.architecture", "mixtral"),
            ("llama.vocab_size", "32000"),
            ("llama.embedding_length", "4096"),
            ("llama.feed_forward_length", "14336"),
            ("llama.block_count", "32"),
            ("llama.attention.head_count", "32"),
            ("llama.attention.head_count_kv", "8"),
            ("llama.attention.head_dim", "128"),
            ("llama.context_length", "32768"),
            ("llama.expert_count", "8"),
            ("llama.expert_used_count", "2"),
        ]);
        let arch = extract_architecture_from_metadata(&m).expect("extract");
        let moe = arch.moe_config.expect("moe present");
        assert_eq!(moe.num_experts, 8);
        assert_eq!(moe.num_experts_used, 2);
    }

    #[test]
    fn attention_kind_default_is_sliding() {
        assert_eq!(AttentionKind::default(), AttentionKind::Sliding);
        assert_eq!(AttentionKind::Sliding.as_str(), "sliding_attention");
        assert_eq!(AttentionKind::Full.as_str(), "full_attention");
    }

    #[test]
    fn text_architecture_serializes_round_trip() {
        let arch = extract_architecture_from_metadata(&full_meta()).expect("extract");
        let json = serde_json::to_string(&arch).expect("serialize");
        let back: TextArchitecture = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(arch, back);
    }

    #[test]
    fn approx_weight_count_is_positive() {
        let arch = extract_architecture_from_metadata(&full_meta()).expect("extract");
        let wc = arch.approx_weight_count();
        // 32 layers × ~70M per layer + 1.5B embedding ≈ 4B weights.
        assert!(wc > 1_000_000_000);
        assert!(wc < 100_000_000_000);
    }
}
