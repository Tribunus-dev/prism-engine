//! Top-level inference orchestrator.
//!
//! Ties together `.cimage` loading, the full-transformer GPU megakernel
//! (RMSNorm → GQA attention → RoPE → SwiGLU MLP), and the tree
//! speculative decoding verification kernel.
//!
//! Each call to [`decode_token`](Orchestrator::decode_token) dispatches a
//! full 48-layer transformer pass on GPU, reads back FP16 logits, and
//! performs greedy argmax sampling.
//!
//! # ANE Prefill ↔ GPU Decode Handoff
//!
//! The orchestrator supports a split pipeline where the ANE (Apple Neural
//! Engine) runs the prefill (prompt processing) while the GPU runs
//! autoregressive decode. This avoids saturating the shared SLC with both
//! ANE and GPU working sets simultaneously.
//!
//! ## KV Cache Layout (Metal side)
//!
//! The KV cache is stored as ternary-packed nibbles (kv_k_nibbles/kv_v_nibbles)
//! with block scales (kv_k_scales/kv_v_scales). The scratch buffers
//! (kv_scratch_k/kv_scratch_v) hold FP16 for one decompressed layer during decode.
//! organised as `[layer][position][head][dim]`.

mod runner;
mod compilation;
mod loading;

pub use runner::Orchestrator;

// ── Architecture constants (shared with megakernel) ───────────────

/// Number of KV heads (GQA). Must match megakernel::NUM_KV_HEADS.
pub(crate) const NUM_KV_HEADS: u32 = 8;
/// KV head dimension (global, after RoPE). Must match megakernel::GLOBAL_HEAD_DIM.
pub(crate) const GLOBAL_HEAD_DIM: u32 = 512;
/// Maximum context length (KV cache slots). Must match megakernel::MAX_CONTEXT.
pub(crate) const MAX_CONTEXT: u32 = 2048;
/// Number of transformer layers.
pub(crate) const LAYERS: u32 = 48;
/// Number of concurrent work queue slots.
#[allow(dead_code)]
pub(crate) const NUM_SLOTS: u32 = 32;
/// Maximum survivor count per slot (20480 = ~1M context at 50:1 compaction).
pub const MAX_SURVIVORS: u32 = 20480;

// ── Half-precision conversion helpers ──────────────────────────────

/// Convert a single-precision float to IEEE 754 FP16 bit pattern.
pub fn half_from_f32(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = (bits >> 23) & 0xFF;
    let mant = bits & 0x7FFFFF;
    if exp == 0 {
        return sign;
    }
    if exp == 0xFF {
        return if mant == 0 {
            if (bits >> 31) != 0 {
                0xFC00
            } else {
                0x7C00
            }
        } else {
            0x7E00
        };
    }
    let exp_f16: i32 = exp as i32 - 127 + 15;
    if exp_f16 >= 0x1F {
        return if (bits >> 31) != 0 { 0xFC00 } else { 0x7C00 };
    }
    if exp_f16 <= 0 {
        return sign;
    }
    sign | ((exp_f16 as u16) << 10) | ((mant >> 13) as u16)
}

/// Convert an IEEE 754 FP16 bit pattern to f32.
pub fn f32_from_half(x: u16) -> f32 {
    let bits = x as u32;
    let sign = bits & 0x8000;
    let exp = (bits >> 10) & 0x1F;
    let mant = bits & 0x3FF;
    if exp == 0 {
        if mant == 0 {
            return 0.0;
        }
        let norm_exp: i32 = -14;
        let norm_mant = mant;
        let fp32_bits = sign << 16 | ((norm_exp + 127) as u32) << 23 | norm_mant << 13;
        return f32::from_bits(fp32_bits);
    }
    if exp == 0x1F {
        let fp32_bits = sign << 16 | 0x7F800000u32 | mant << 13;
        return f32::from_bits(fp32_bits);
    }
    let fp32_exp = exp.wrapping_add(127 - 15);
    let fp32_bits = sign << 16 | fp32_exp << 23 | mant << 13;
    f32::from_bits(fp32_bits)
}

/// Greedy argmax over FP16 logits.
pub fn sample_argmax(logits: &[u16]) -> u32 {
    let mut best = 0u32;
    let mut best_v = f32_from_half(logits[0]);
    for (i, &l) in logits.iter().enumerate().skip(1) {
        let v = f32_from_half(l);
        if v > best_v {
            best_v = v;
            best = i as u32;
        }
    }
    best
}

/// Extract the top-K token IDs from FP16 logits by finding the
/// highest-valued positions. Used by MTP speculative decode.
pub fn generate_speculative_candidates(logits: &[u16], count: usize) -> Vec<u32> {
    if logits.is_empty() || count == 0 {
        return Vec::new();
    }
    let mut indices: Vec<u32> = (0..logits.len() as u32).collect();
    // Partial sort: find top `count` by FP16 value
    indices.sort_by(|&a, &b| {
        let va = f32_from_half(logits[a as usize]);
        let vb = f32_from_half(logits[b as usize]);
        vb.partial_cmp(&va).unwrap_or(std::cmp::Ordering::Equal)
    });
    indices.truncate(count);
    indices
}

// ── Scheduling phase ───────────────────────────────────────────────

/// Scheduling phase — controls SLC occupancy on M1.
///
/// The M1's 8 MB System Level Cache is shared across CPU, GPU, and ANE.
/// When the GPU streams ~2.7 GB of Base-3 ternary weights for a single
/// decode pass it evicts the ANE working set (~2 MB+) from the SLC.
/// These phases are informational/diagnostic — the ANE and GPU run
/// concurrently without blocking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SLCPhase {
    /// ANE owns the SLC (prefill). GPU dispatches are forbidden.
    ANEPrefill,
    /// GPU owns the SLC (decode). ANE dispatches are forbidden.
    GPUDecode,
    /// No active compute — idle.
    Idle,
}
