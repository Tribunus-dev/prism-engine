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

// ── Module declarations ──────────────────────────────────────────────

#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
mod compilation;
pub mod kernel_fusion;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
mod loading;
mod multimodal_assembly;
pub mod multimodal_receipt;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
mod runner;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub mod vision_projection;

#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub use runner::Orchestrator;

pub use crate::ecs::compute_image::legacy_compute_image_runtime::multimodal::{
    InputModality, ModalityError, MultimodalArtifactSummary, MultimodalCapabilities,
    ProjectionBackend, ProjectionPrecision,
};
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
use multimodal_assembly::PromptPart;

// ── Architecture constants (shared with megakernel) ───────────────

/// Number of KV heads (GQA). Must match megakernel::NUM_KV_HEADS.
#[allow(dead_code)]
pub(crate) const NUM_KV_HEADS: u32 = 8;
/// KV head dimension (global, after RoPE). Must match megakernel::GLOBAL_HEAD_DIM.
#[allow(dead_code)]
pub(crate) const GLOBAL_HEAD_DIM: u32 = 512;
/// Maximum context length (KV cache slots). Must match megakernel::MAX_CONTEXT.
#[allow(dead_code)]
pub(crate) const MAX_CONTEXT: u32 = 2048;
/// Number of transformer layers.
#[allow(dead_code)]
pub(crate) const LAYERS: u32 = 48;
/// Number of concurrent work queue slots.
#[allow(dead_code)]
pub(crate) const NUM_SLOTS: u32 = 32;
/// Maximum survivor count per slot (20480 = ~1M context at 50:1 compaction).
pub const MAX_SURVIVORS: u32 = 20480;

// ── Execution phase kinds ─────────────────────────────────────────

/// Execution phases for the compute orchestrator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputePhaseKind {
    /// Multimodal preparation — resize, patchify, normalize.
    MultimodalPrepare,
    /// Multimodal projection — patch embedding → decoder-width activations.
    MultimodalProject,
    /// Decoder prefill over assembled embeddings.
    DecoderPrefill,
    /// Autoregressive single-token decode.
    DecoderDecode,
}

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

/// Greedy argmax over f32 logits (the scoring-API side of the boundary —
/// `decode_slot_logits` converts the megakernel's FP16 bits to f32 once, and
/// everything downstream of it works in f32).
pub fn sample_argmax_f32(logits: &[f32]) -> u32 {
    let mut best = 0u32;
    let mut best_v = f32::MIN;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i as u32;
        }
    }
    best
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

// ── Concurrency policy ───────────────────────────────────────────

/// Encodes chip-family bandwidth and cache class.
/// Drives default scheduling policy rather than per-model guesswork.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppleMemoryPressureClass {
    /// Base M1: ~68 GB/s, 8–16 MB SLC — tight bandwidth, high contention risk.
    BaseM1Constrained,
    /// M1 Pro: ~200 GB/s, 24 MB SLC — moderate headroom.
    ProClassModerate,
    /// M1 Max: ~400 GB/s, 48 MB SLC — wide fabric.
    MaxClassWide,
    /// M1 Ultra: ~800 GB/s, 96 MB SLC — very wide, dual-die.
    UltraClassVeryWide,
    /// M2/M3/M4/M5 or unidentifiable device.
    Unknown,
}

/// GPU weight-streaming cache mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuWeightCacheMode {
    /// Streaming / cache-hostile: weights have no temporal reuse;
    /// prevent them from polluting the SLC.
    Streaming,
    /// Metal-default cache policy.
    Default,
    /// Aggressive cache retention for small-model or fine-tuned paths.
    AggressiveReuse,
}

/// Per-device decode policy — replaces binary SLCPhase gating.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AppleDecodePolicy {
    pub gpu_weight_cache_mode: GpuWeightCacheMode,
    pub allow_ane_gpu_overlap: bool,
    pub require_overlap_benchmark: bool,
    pub max_ane_inflight: u8,
}

impl AppleDecodePolicy {
    pub const fn for_pressure_class(pc: AppleMemoryPressureClass) -> Self {
        match pc {
            AppleMemoryPressureClass::BaseM1Constrained => Self {
                gpu_weight_cache_mode: GpuWeightCacheMode::Streaming,
                allow_ane_gpu_overlap: false,
                require_overlap_benchmark: true,
                max_ane_inflight: 1,
            },
            AppleMemoryPressureClass::ProClassModerate => Self {
                gpu_weight_cache_mode: GpuWeightCacheMode::Default,
                allow_ane_gpu_overlap: true,
                require_overlap_benchmark: true,
                max_ane_inflight: 1,
            },
            AppleMemoryPressureClass::MaxClassWide
            | AppleMemoryPressureClass::UltraClassVeryWide => Self {
                gpu_weight_cache_mode: GpuWeightCacheMode::Default,
                allow_ane_gpu_overlap: true,
                require_overlap_benchmark: false,
                max_ane_inflight: 1,
            },
            AppleMemoryPressureClass::Unknown => Self {
                gpu_weight_cache_mode: GpuWeightCacheMode::Default,
                allow_ane_gpu_overlap: false,
                require_overlap_benchmark: true,
                max_ane_inflight: 1,
            },
        }
    }
}

/// Legacy diagnostic phase — kept for backward compat in existing callers.
/// Write-only; dispatch is gated by AppleDecodePolicy, not this enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[deprecated(since = "0.3.0", note = "use AppleDecodePolicy instead")]
pub enum SLCPhase {
    ANEPrefill,
    GPUDecode,
    Idle,
}

// ── ANE Worker types ──────────────────────────────────────────────

/// Request sent to the dedicated ANE MTP decode worker.
#[derive(Debug, Clone)]
pub struct MtpDecodeRequest {
    pub session_id: u64,
    pub token_id: u32,
    pub position: u32,
    pub kv_generation: u64,
}

/// Result from one ANE MTP decode step.
#[derive(Debug, Clone)]
pub struct MtpDecodeResult {
    pub session_id: u64,
    /// The KV generation this result corresponds to.
    pub kv_generation: u64,
    /// Draft token IDs produced by the MTP model.
    pub draft_tokens: Vec<u32>,
    /// Wall-clock ANE execution time in nanoseconds.
    pub elapsed_ns: u64,
}

/// A structured multi-token candidate proposal produced by the ANE MTP model.
///
/// The MTP model predicts a bounded continuation of up to 3 future tokens
/// per invocation. Verification by the main 12B model treats this as a
/// candidate chain: positions are accepted in prefix order, stopping at the
/// first rejection.
#[derive(Debug, Clone)]
pub struct MtpProposal {
    /// Absolute sequence position of the first proposed token (t+1).
    pub base_position: u32,
    /// KV generation this proposal was built from. Must match the main model's
    /// generation for verification to proceed.
    pub kv_generation: u64,
    /// Proposed token IDs for positions t+1, t+2, t+3.
    pub tokens: [u32; 3],
    /// Number of proposed tokens (1-3).
    pub token_count: u8,
    /// Per-token confidence (FP16, higher = more certain).
    pub confidence: [u16; 3],
    /// SHA-256 digest of the raw logits buffer for diagnosability.
    pub logits_digest: [u8; 32],
}

/// IOSurface-backed MTP proposal transport -- written by ANE, read by Metal.
///
/// Allocated as a shared IOSurface, bound to both Core ML (as output) and
/// Metal (as a MTLBuffer or MTLTexture). The E-core only writes `epoch`
/// and signals `MTLSharedEvent` after Core ML completion; the payload
/// itself is populated directly by the ANE.
///
/// Fixed size to avoid dynamic allocation in the hot path.
#[repr(C, align(64))]
pub struct MtpTreeSurface {
    /// Monotonically increasing proposal epoch. Metal waits on this value
    /// via MTLSharedEvent before consuming the tree.
    pub epoch: u64,
    /// Session identifier -- must match the active decode session.
    pub session_id: u64,
    /// KV generation this tree was built from.
    pub kv_generation: u64,
    /// Number of active nodes in the tree (1-16).
    pub node_count: u16,
    /// Tree depth (1-3).
    pub tree_depth: u8,
    /// Flags: bit 0 = ready, bit 1 = stale, bits 2-7 reserved.
    pub flags: u8,
    /// Token IDs indexed by node (0 = root/input token).
    pub token_ids: [u32; 16],
    /// Parent node index for each node. Root has parent_id = 0xFF.
    pub parent_ids: [u8; 16],
    /// Rank within sibling group (0 = best).
    pub rank: [u8; 16],
    /// Confidence in Q2.15 fixed-point (higher = more certain).
    pub confidence_q15: [u16; 16],
    /// Set to 1 by the E-core after all fields including epoch are valid.
    pub ready: u32,
    /// Padding to reach 64-byte alignment boundary.
    pub _pad: [u8; 96],
}

// Compile-time size check -- IOSurface allocations need exact size.
const _: () = assert!(std::mem::size_of::<MtpTreeSurface>() == 256);

impl Default for MtpTreeSurface {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

// ── Orchestrator methods (multimodal validation) ───────────────────

#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
impl Orchestrator {
    /// Validate that this deployment can handle the given prompt parts.
    pub fn validate_multimodal_prompt(&self, parts: &[PromptPart]) -> Result<(), String> {
        let has_image = parts.iter().any(|p| matches!(p, PromptPart::Image(_)));
        let has_audio = parts.iter().any(|p| matches!(p, PromptPart::Audio(_)));

        if has_image && !self.deployment.multimodal_capabilities().image {
            return Err("image prompt requires multimodal capabilities".into());
        }
        if has_audio && !self.deployment.multimodal_capabilities().audio {
            return Err("audio modality is feature-gated".into());
        }
        Ok(())
    }
}
/// Per-session MTP KV state with speculative rollback.
///
/// The ANE worker MUST NOT advance its local KV state irreversibly
/// until the main-model verification commits. Rejection sampling can
/// roll back the speculative generation.
#[derive(Debug, Clone)]
pub struct MtpKvState {
    pub committed_generation: u64,
    pub speculative_generation: u64,
}

impl MtpKvState {
    pub fn new() -> Self {
        Self {
            committed_generation: 0,
            speculative_generation: 0,
        }
    }

    /// Record a new speculative step. Does not commit.
    pub fn speculate(&mut self, gen: u64) {
        self.speculative_generation = gen;
    }

    /// Commit the speculative generation as the new baseline.
    pub fn commit(&mut self) {
        self.committed_generation = self.speculative_generation;
    }

    /// Revert speculative state after rejection.
    pub fn rollback(&mut self) {
        self.speculative_generation = self.committed_generation;
    }
}

/// Dedicated ANE MTP decode worker -- owns the Core ML model permanently.
///
/// Designed to be pinned to an E-core. Uses cpuAndNeuralEngine mode to
/// exclude the GPU. Produces MtpProposal blocks via IOSurface-backed
/// transport: the ANE writes candidate tokens directly into a shared
/// MtpTreeSurface; the E-core advances the epoch and signals
/// MTLSharedEvent; Metal reads the surface at a legal verification
/// boundary without an intermediate CPU copy.
pub struct AneMtpWorker {
    pub model: crate::ecs::coreai_bridge::CoreAiModel,
    pub request_rx: std::sync::mpsc::Receiver<MtpDecodeRequest>,
    pub result_tx: std::sync::mpsc::Sender<MtpDecodeResult>,
    pub session_states: std::collections::HashMap<u64, MtpKvState>,
    pub handle: Option<std::thread::JoinHandle<()>>,
}
