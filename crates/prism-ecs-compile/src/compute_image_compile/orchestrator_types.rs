//! Orchestrator-level types and helpers — execution phases, decode
//! policy, MTP proposal/transport types, sample-argmax helpers.
//!
//! Authority: orchestrator data types that are decoupled from the heavy
//! MLX / Metal runtime. The actual `Orchestrator` struct (which holds
//! an MLX `Deployment` and a metal pipeline) lives in the engine's
//! `legacy_compute_image_compile::orchestrator::runner`.

use crate::compute_image_compile::fp16::f32_from_half;

/// Number of KV heads (GQA).
pub const NUM_KV_HEADS: u32 = 8;
/// KV head dimension (global, after RoPE).
pub const GLOBAL_HEAD_DIM: u32 = 512;
/// Maximum context length (KV cache slots).
pub const MAX_CONTEXT: u32 = 2048;
/// Number of transformer layers.
pub const LAYERS: u32 = 48;
/// Number of concurrent work queue slots.
pub const NUM_SLOTS: u32 = 32;
/// Maximum survivor count per slot.
pub const MAX_SURVIVORS: u32 = 20480;

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

/// Encodes chip-family bandwidth and cache class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppleMemoryPressureClass {
    /// Base M1: tight bandwidth, high contention risk.
    BaseM1Constrained,
    /// M1 Pro: moderate headroom.
    ProClassModerate,
    /// M1 Max: wide fabric.
    MaxClassWide,
    /// M1 Ultra: very wide, dual-die.
    UltraClassVeryWide,
    /// M2/M3/M4/M5 or unidentifiable device.
    Unknown,
}

/// GPU weight-streaming cache mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuWeightCacheMode {
    /// Streaming / cache-hostile: weights have no temporal reuse.
    Streaming,
    /// Metal-default cache policy.
    Default,
    /// Aggressive cache retention for small-model or fine-tuned paths.
    AggressiveReuse,
}

/// Per-device decode policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AppleDecodePolicy {
    /// GPU weight-streaming cache mode.
    pub gpu_weight_cache_mode: GpuWeightCacheMode,
    /// Whether ANE and GPU may overlap.
    pub allow_ane_gpu_overlap: bool,
    /// Whether the overlap must be benchmarked before being enabled.
    pub require_overlap_benchmark: bool,
    /// Maximum ANE inflight requests.
    pub max_ane_inflight: u8,
}

impl AppleDecodePolicy {
    /// Default decode policy for a given memory-pressure class.
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

/// Greedy argmax over f32 logits.
pub fn sample_argmax_f32(logits: &[f32]) -> u32 {
    if logits.is_empty() {
        return 0;
    }
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
    if logits.is_empty() {
        return 0;
    }
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

/// Request sent to the dedicated ANE MTP decode worker.
#[derive(Debug, Clone)]
pub struct MtpDecodeRequest {
    /// Session identifier.
    pub session_id: u64,
    /// Token ID to start decoding from.
    pub token_id: u32,
    /// Sequence position to decode at.
    pub position: u32,
    /// KV generation this request is based on.
    pub kv_generation: u64,
}

/// Result from one ANE MTP decode step.
#[derive(Debug, Clone)]
pub struct MtpDecodeResult {
    /// Session identifier.
    pub session_id: u64,
    /// The KV generation this result corresponds to.
    pub kv_generation: u64,
    /// Draft token IDs produced by the MTP model.
    pub draft_tokens: Vec<u32>,
    /// Wall-clock ANE execution time in nanoseconds.
    pub elapsed_ns: u64,
}

/// A structured multi-token candidate proposal produced by the ANE MTP model.
#[derive(Debug, Clone)]
pub struct MtpProposal {
    /// Absolute sequence position of the first proposed token (t+1).
    pub base_position: u32,
    /// KV generation this proposal was built from.
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

/// IOSurface-backed MTP proposal transport — written by ANE, read by Metal.
///
/// Allocated as a shared IOSurface, bound to both Core ML (as output) and
/// Metal (as a `MTLBuffer` or `MTLTexture`). The E-core only writes `epoch`
/// and signals `MTLSharedEvent` after Core ML completion; the payload itself
/// is populated directly by the ANE.
#[repr(C, align(64))]
pub struct MtpTreeSurface {
    /// Monotonically increasing proposal epoch.
    pub epoch: u64,
    /// Session identifier.
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
    /// Parent node index for each node. Root has `parent_id = 0xFF`.
    pub parent_ids: [u8; 16],
    /// Rank within sibling group (0 = best).
    pub rank: [u8; 16],
    /// Confidence in Q2.15 fixed-point (higher = more certain).
    pub confidence_q15: [u16; 16],
    /// Set to 1 by the E-core after all fields including `epoch` are valid.
    pub ready: u32,
    /// Padding to reach 64-byte alignment boundary.
    pub _pad: [u8; 96],
}

// Compile-time size check — IOSurface allocations need exact size.
const _: () = assert!(std::mem::size_of::<MtpTreeSurface>() == 256);

impl Default for MtpTreeSurface {
    fn default() -> Self {
        // SAFETY: `MtpTreeSurface` is `#[repr(C, align(64))]` and every field
        // is a primitive integer type or array of primitives. A zeroed
        // representation is a valid default.
        unsafe { std::mem::zeroed() }
    }
}

/// Per-session MTP KV state with speculative rollback.
#[derive(Debug, Clone)]
pub struct MtpKvState {
    /// Committed KV generation.
    pub committed_generation: u64,
    /// Speculative KV generation.
    pub speculative_generation: u64,
}

impl Default for MtpKvState {
    fn default() -> Self {
        Self::new()
    }
}

impl MtpKvState {
    /// Construct a fresh `MtpKvState` with both generations at 0.
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
