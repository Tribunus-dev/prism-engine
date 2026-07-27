//! Pure measurement calculations for compiled-model profiling.
//!
//! Single authority: the constitutional peak-bytes and RoPE-table budget
//! calculations that the engine previously implemented inline against a
//! `CompiledImageReader`. The constitutional version operates on the
//! canonical [`ArchitectureMeta`] and a slice of byte sizes so it has no
//! dependency on the engine, no FFI, no process-local state, and no
//! hardware observation. The engine's `estimate_profiled_peak_bytes` is
//! a thin adapter that constructs an [`ArchitectureMeta`] and a sorted
//! byte-size vector and delegates to [`peak_bytes_for_manifest`].

use serde::{Deserialize, Serialize};

use super::data::ByteCount;

// ---------------------------------------------------------------------------
// Architecture metadata (canonical, engine-free)
// ---------------------------------------------------------------------------

/// Canonical, engine-free architecture metadata for peak-bytes and
/// rope-table measurement.
///
/// Captures the architectural parameters that the engine's
/// `estimate_profiled_peak_bytes` and `build_rope_tables` read from the
/// compiled-image manifest, lifted into a constitutional type so callers do
/// not have to depend on the engine's `TextArchitecture`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArchitectureMeta {
    /// Hidden size of the transformer (`hidden_size`).
    pub hidden_size: u32,
    /// Vocabulary size (`vocab_size`).
    pub vocab_size: u32,
    /// Max position embeddings (`max_position_embeddings`).
    pub max_position_embeddings: u32,
    /// Number of attention heads (`num_attention_heads`).
    pub num_attention_heads: u32,
    /// Number of key-value heads (`num_key_value_heads`); GQA models have
    /// `num_kv_heads < num_attention_heads`.
    pub num_key_value_heads: u32,
    /// Per-head dimension (`head_dim`).
    pub head_dim: u32,
    /// Global head dimension (`global_head_dim`) when the model uses
    /// distinct RoPE tables for the global attention. `None` reuses
    /// [`Self::head_dim`].
    pub global_head_dim: Option<u32>,
}

impl ArchitectureMeta {
    /// Effective global head dim, defaulting to [`Self::head_dim`].
    #[must_use]
    pub fn effective_global_head_dim(&self) -> u32 {
        self.global_head_dim.unwrap_or(self.head_dim)
    }
}

// ---------------------------------------------------------------------------
// Peak bytes newtype
// ---------------------------------------------------------------------------

/// Peak-byte estimate for a profiled model load.
///
/// Wraps a `u64` so callers cannot accidentally use a peak estimate as a
/// runtime byte count or vice versa.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct PeakBytesEstimate(pub u64);

impl PeakBytesEstimate {
    /// The fixed 2 GiB extra budget the engine adds for the runtime
    /// (Accelerate, scratch, KV cache headroom beyond the model estimate).
    /// Exposed as a constant so constitutional callers can apply the same
    /// rule without duplicating the magic number.
    pub const RUNTIME_BUDGET_BYTES: u64 = 2 * 1024 * 1024 * 1024;

    /// The fixed 25% margin the engine adds to the IOSurface pool.
    pub const IOSURFACE_POOL_MARGIN_NUMERATOR: u64 = 125;
    pub const IOSURFACE_POOL_MARGIN_DENOMINATOR: u64 = 100;

    /// The minimum 16 MiB IOSurface pool the engine guarantees.
    pub const MIN_IOSURFACE_POOL_BYTES: u64 = 16 * 1024 * 1024;

    /// Inner raw value.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl Default for PeakBytesEstimate {
    fn default() -> Self {
        Self(0)
    }
}

// ---------------------------------------------------------------------------
// Peak bytes for a manifest
// ---------------------------------------------------------------------------

/// Compute the peak byte estimate for a model load, given a sorted list of
/// per-tensor byte sizes, the largest single tensor size, and the largest
/// segment size.
///
/// `tensor_total_bytes` is the sum of all tensor byte lengths in the
/// compiled image (mapped, copied, or otherwise). The engine's original
/// `estimate_profiled_peak_bytes` adds a fixed `+2 GiB` runtime budget on
/// top of the architectural budget; we expose that constant on
/// [`PeakBytesEstimate::RUNTIME_BUDGET_BYTES`].
#[must_use]
pub fn peak_bytes_for_manifest(
    arch: &ArchitectureMeta,
    tensor_total_bytes: u64,
    max_tensor_bytes: u64,
    max_segment_bytes: u64,
) -> PeakBytesEstimate {
    let rope = rope_table_bytes(arch);
    let embed = embedding_dequant_bytes(arch);
    let peak = tensor_total_bytes
        .saturating_add(max_tensor_bytes)
        .saturating_add(max_segment_bytes)
        .saturating_add(rope)
        .saturating_add(embed)
        .saturating_add(PeakBytesEstimate::RUNTIME_BUDGET_BYTES);
    PeakBytesEstimate(peak)
}

// ---------------------------------------------------------------------------
// RoPE table byte budget
// ---------------------------------------------------------------------------

/// Compute the byte budget for the four RoPE tables the engine materializes
/// for a profiled load (local cos/sin + global cos/sin). Each entry is a
/// `f32` tensor of shape `[max_position_embeddings, head_dim]` (or
/// `[max_position_embeddings, global_head_dim]` for the global tables).
#[must_use]
pub fn rope_table_bytes(arch: &ArchitectureMeta) -> u64 {
    let local = u64::from(arch.max_position_embeddings)
        .saturating_mul(u64::from(arch.head_dim))
        .saturating_mul(4);
    let global = u64::from(arch.max_position_embeddings)
        .saturating_mul(u64::from(arch.effective_global_head_dim()))
        .saturating_mul(4);
    local.saturating_add(global)
}

// ---------------------------------------------------------------------------
// Embedding dequantization byte budget
// ---------------------------------------------------------------------------

/// Compute the byte budget for dequantizing the embedding table to
/// `f32` at load time (`vocab_size * hidden_size * 4`).
#[must_use]
pub fn embedding_dequant_bytes(arch: &ArchitectureMeta) -> u64 {
    u64::from(arch.vocab_size)
        .saturating_mul(u64::from(arch.hidden_size))
        .saturating_mul(4)
}

// ---------------------------------------------------------------------------
// IOSurface pool sizing
// ---------------------------------------------------------------------------

/// Compute the IOSurface pool size for the engine's shared memory island,
/// given a `total_memory` value the engine observed via sysctl (or
/// otherwise).
///
/// Constitutional callers pass in a `total_memory` value they trust
/// (typically the engine's `system_memory_bytes` result); this function
/// applies the engine's rule: `min(computed_pool, total_memory/4)` floored
/// at [`PeakBytesEstimate::MIN_IOSURFACE_POOL_BYTES`], with a
/// `+25%` margin applied to the computed pool before clamping.
#[must_use]
pub fn iosurface_pool_bytes(total_memory: u64, computed_pool: u64) -> u64 {
    let with_margin = computed_pool
        .saturating_mul(PeakBytesEstimate::IOSURFACE_POOL_MARGIN_NUMERATOR)
        / PeakBytesEstimate::IOSURFACE_POOL_MARGIN_DENOMINATOR;
    if total_memory > 0 {
        with_margin
            .min(total_memory / 4)
            .max(PeakBytesEstimate::MIN_IOSURFACE_POOL_BYTES)
    } else {
        with_margin.max(PeakBytesEstimate::MIN_IOSURFACE_POOL_BYTES)
    }
}

// ---------------------------------------------------------------------------
// Admission safety budget
// ---------------------------------------------------------------------------

/// Compute the safe memory budget the engine compares a peak estimate
/// against. The engine refuses to load if the estimate exceeds
/// `total_memory - safety_margin`. We use the engine's `2 GiB` margin
/// (see [`PeakBytesEstimate::RUNTIME_BUDGET_BYTES`]).
#[must_use]
pub fn admission_safe_budget(total_memory: u64) -> ByteCount {
    ByteCount(total_memory.saturating_sub(PeakBytesEstimate::RUNTIME_BUDGET_BYTES))
}

/// Returns `true` when `peak` is within the safe budget for `total_memory`.
#[must_use]
pub fn peak_within_budget(peak: PeakBytesEstimate, total_memory: u64) -> bool {
    total_memory == 0 || peak.0 <= admission_safe_budget(total_memory).0
}

// ---------------------------------------------------------------------------
// Attention workspace byte budget
// ---------------------------------------------------------------------------

/// Compute the attention-score workspace budget the engine uses when
/// sizing the IOSurface pool. The engine caps the effective sequence at
/// 4096 and multiplies by `num_heads * head_dim * 4`.
#[must_use]
pub fn attention_scores_bytes(arch: &ArchitectureMeta) -> u64 {
    let cap = u64::from(arch.max_position_embeddings).min(4096);
    cap.saturating_mul(u64::from(arch.num_attention_heads))
        .saturating_mul(u64::from(arch.head_dim))
        .saturating_mul(4)
}

/// Compute the per-token KV-cache byte cost for the engine's IOSurface
/// pool sizing. The engine uses FP16 (`*2`).
#[must_use]
pub fn kv_per_token_bytes(arch: &ArchitectureMeta) -> u64 {
    2u64
        .saturating_mul(u64::from(arch.num_key_value_heads))
        .saturating_mul(u64::from(arch.head_dim))
        .saturating_mul(2)
}

/// The KV cache headroom for 4096 tokens at the current architecture.
#[must_use]
pub fn kv_headroom_bytes(arch: &ArchitectureMeta) -> u64 {
    kv_per_token_bytes(arch).saturating_mul(4096)
}

/// The scratch-byte budget for 10 f32 hidden-size scratch tensors.
#[must_use]
pub fn scratch_bytes(arch: &ArchitectureMeta) -> u64 {
    u64::from(arch.hidden_size).saturating_mul(10).saturating_mul(4)
}

/// The full IOSurface pool computation, given the per-stage components
/// the engine sums. This is the constitutional counterpart of the engine's
/// `computed_pool` calculation that adds `scratch + attn + kv_headroom`
/// before the 25% margin is applied.
#[must_use]
pub fn computed_iosurface_pool(arch: &ArchitectureMeta) -> u64 {
    let scratch = scratch_bytes(arch);
    let attn = attention_scores_bytes(arch);
    let kv = kv_headroom_bytes(arch);
    scratch.saturating_add(attn).saturating_add(kv)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn qwen_arch() -> ArchitectureMeta {
        ArchitectureMeta {
            hidden_size: 4096,
            vocab_size: 152_064,
            max_position_embeddings: 32_768,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            global_head_dim: Some(128),
        }
    }

    #[test]
    fn rope_table_bytes_is_local_plus_global() {
        let arch = qwen_arch();
        let expected = 2 * u64::from(arch.max_position_embeddings)
            * u64::from(arch.head_dim)
            * 4;
        assert_eq!(rope_table_bytes(&arch), expected);
    }

    #[test]
    fn rope_table_bytes_handles_distinct_global_head_dim() {
        let arch = ArchitectureMeta {
            head_dim: 128,
            global_head_dim: Some(256),
            ..qwen_arch()
        };
        let local = u64::from(arch.max_position_embeddings) * 128 * 4;
        let global = u64::from(arch.max_position_embeddings) * 256 * 4;
        assert_eq!(rope_table_bytes(&arch), local + global);
    }

    #[test]
    fn rope_table_bytes_falls_back_when_global_unset() {
        let arch = ArchitectureMeta {
            global_head_dim: None,
            ..qwen_arch()
        };
        let expected = 2 * u64::from(arch.max_position_embeddings)
            * u64::from(arch.head_dim)
            * 4;
        assert_eq!(rope_table_bytes(&arch), expected);
    }

    #[test]
    fn embedding_dequant_bytes_is_vocab_times_hidden_times_4() {
        let arch = qwen_arch();
        assert_eq!(
            embedding_dequant_bytes(&arch),
            u64::from(arch.vocab_size) * u64::from(arch.hidden_size) * 4
        );
    }

    #[test]
    fn peak_bytes_for_manifest_sums_three_components_plus_rope_plus_embed() {
        let arch = qwen_arch();
        let peak = peak_bytes_for_manifest(&arch, 100, 50, 25);
        let expected = 100u64
            .saturating_add(50)
            .saturating_add(25)
            .saturating_add(rope_table_bytes(&arch))
            .saturating_add(embedding_dequant_bytes(&arch))
            .saturating_add(PeakBytesEstimate::RUNTIME_BUDGET_BYTES);
        assert_eq!(peak, PeakBytesEstimate(expected));
    }

    #[test]
    fn admission_safe_budget_subtracts_runtime_margin() {
        let total = 16 * 1024 * 1024 * 1024_u64;
        let expected = total - PeakBytesEstimate::RUNTIME_BUDGET_BYTES;
        assert_eq!(admission_safe_budget(total), ByteCount(expected));
    }

    #[test]
    fn peak_within_budget_when_total_is_zero() {
        // 0 total_memory means the engine skipped the admission check.
        assert!(peak_within_budget(PeakBytesEstimate(u64::MAX), 0));
    }

    #[test]
    fn peak_within_budget_rejects_oversize() {
        let total = 16 * 1024 * 1024 * 1024_u64;
        let safe = admission_safe_budget(total).0;
        assert!(peak_within_budget(PeakBytesEstimate(safe), total));
        assert!(!peak_within_budget(
            PeakBytesEstimate(safe.saturating_add(1)),
            total
        ));
    }

    #[test]
    fn iosurface_pool_bytes_applies_margin_then_clamps() {
        let arch = qwen_arch();
        let computed = computed_iosurface_pool(&arch);
        let total = 16 * 1024 * 1024 * 1024_u64;
        let with_margin =
            computed * PeakBytesEstimate::IOSURFACE_POOL_MARGIN_NUMERATOR
                / PeakBytesEstimate::IOSURFACE_POOL_MARGIN_DENOMINATOR;
        // total/4 = 4 GiB; with_margin for a realistic Qwen is much smaller.
        let expected = with_margin.min(total / 4).max(PeakBytesEstimate::MIN_IOSURFACE_POOL_BYTES);
        assert_eq!(iosurface_pool_bytes(total, computed), expected);
    }

    #[test]
    fn iosurface_pool_bytes_zero_total_uses_minimum() {
        let arch = qwen_arch();
        let computed = computed_iosurface_pool(&arch);
        let with_margin =
            computed * PeakBytesEstimate::IOSURFACE_POOL_MARGIN_NUMERATOR
                / PeakBytesEstimate::IOSURFACE_POOL_MARGIN_DENOMINATOR;
        // total_memory == 0 means "unknown"; pool is the min of with_margin and 16 MiB.
        let expected = with_margin.max(PeakBytesEstimate::MIN_IOSURFACE_POOL_BYTES);
        assert_eq!(iosurface_pool_bytes(0, computed), expected);
    }

    #[test]
    fn attention_scores_bytes_caps_at_4096() {
        let arch = qwen_arch();
        // max_position_embeddings=32768 is well above the 4096 cap.
        let expected = 4096 * u64::from(arch.num_attention_heads)
            * u64::from(arch.head_dim)
            * 4;
        assert_eq!(attention_scores_bytes(&arch), expected);
    }

    #[test]
    fn kv_per_token_bytes_uses_fp16() {
        let arch = qwen_arch();
        // 2 (K+V) * num_kv_heads * head_dim * 2 (FP16 bytes)
        let expected = 2 * u64::from(arch.num_key_value_heads)
            * u64::from(arch.head_dim)
            * 2;
        assert_eq!(kv_per_token_bytes(&arch), expected);
    }

    #[test]
    fn computed_iosurface_pool_is_scratch_plus_attn_plus_kv_headroom() {
        let arch = qwen_arch();
        let expected = scratch_bytes(&arch)
            .saturating_add(attention_scores_bytes(&arch))
            .saturating_add(kv_headroom_bytes(&arch));
        assert_eq!(computed_iosurface_pool(&arch), expected);
    }

    #[test]
    fn saturating_arithmetic_does_not_panic() {
        let arch = ArchitectureMeta {
            hidden_size: u32::MAX,
            vocab_size: u32::MAX,
            max_position_embeddings: u32::MAX,
            num_attention_heads: u32::MAX,
            num_key_value_heads: u32::MAX,
            head_dim: u32::MAX,
            global_head_dim: Some(u32::MAX),
        };
        // Should not panic.
        let _ = rope_table_bytes(&arch);
        let _ = embedding_dequant_bytes(&arch);
        let _ = attention_scores_bytes(&arch);
        let _ = kv_per_token_bytes(&arch);
        let _ = computed_iosurface_pool(&arch);
        let _ = peak_bytes_for_manifest(&arch, u64::MAX, u64::MAX, u64::MAX);
        let _ = iosurface_pool_bytes(u64::MAX, u64::MAX);
    }
}
