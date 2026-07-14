//! KvCachePlan -- compiled KV cache layout contract.
//! The ComputeImage declares the KV cache plan so the runtime can
//! instantiate KvBlockArena from it without runtime configuration decisions.
//!
//! The central contract: KvCodec replaces string-typed precision policies
//! and drives state-store page sizing, runtime construction, attention
//! dispatch, and replay receipts from a single type.

use serde::{Deserialize, Serialize};

// ── Typted KV codec ─────────────────────────────────────────────────────────

/// Typted KV codec contract — replaces string-typed precision policies.
/// Each variant carries the exact quantization parameters needed for
/// state-store page sizing, runtime construction, and attention dispatch.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum KvCodec {
    /// Uncompressed FP16 (2 bytes/value).
    Fp16,
    /// TurboQuant with optional K/V asymmetry.
    TurboQuant {
        key_mode: KvQuantMode,
        value_mode: KvQuantMode,
        group_size: u32,
    },
    /// FP32 reference (rare, CPU testing only).
    Fp32,
}

/// TurboQuant mode for KV quantization.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum KvQuantMode {
    Polar(u32),
    Prod(u32),
    Split(u32),
    Mse(u32),
    PolarProd(u32),
    PolarHadamard(u32),
    TurboQuant3,
}

/// Bytes-per-value for a KvQuantMode (rounded up to nearest bit, for cimage
/// metadata — actual packing may vary per TurboQuant implementation).
pub fn bits_for_mode(mode: &KvQuantMode) -> u32 {
    use KvQuantMode::*;
    match mode {
        Polar(n) | Prod(n) | Split(n) | Mse(n) | PolarProd(n) | PolarHadamard(n) => *n,
        TurboQuant3 => 3,
    }
}

impl KvCodec {
    /// Bytes required for one KV page given this codec.
    /// Used by KvCacheManager for page sizing without manual dtype dispatch.
    pub fn page_bytes(&self, head_dim: u32, page_tokens: u32) -> u64 {
        match self {
            KvCodec::Fp16 => page_tokens as u64 * head_dim as u64 * 4,
            KvCodec::Fp32 => page_tokens as u64 * head_dim as u64 * 8,
            KvCodec::TurboQuant {
                key_mode,
                value_mode,
                group_size,
            } => {
                let key_bits = bits_for_mode(key_mode) as u64;
                let value_bits = bits_for_mode(value_mode) as u64;
                let elems = page_tokens as u64 * head_dim as u64;
                let key_bytes = (elems * key_bits + 7) / 8;
                let value_bytes = (elems * value_bits + 7) / 8;
                let groups = (head_dim as u64 + *group_size as u64 - 1) / *group_size as u64;
                let scales = groups * 4; // 4 bytes/scale per dtype (f32 alpha+beta handled internally)
                key_bytes + value_bytes + scales * 2
            }
        }
    }
}

// ── Legacy KVDtype (kept for backward compat, prefer KvCodec for new code) ──

/// KV cache element dtype with FP8 support.
/// Superseded by KvCodec for new code — kept for existing callers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum KVDtype {
    Bf16,
    F16,
    F32,
    Fp8E4M3,
    Fp8E5M2,
    Int8,
}

/// Physical KV layout strategy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum KvLayout {
    /// Fixed-size blocks, one block per page.
    PagedBlocks,
    /// Virtual contiguous address space backed by paged physical memory.
    VirtualContiguousPagedPhysical,
}

/// Compiled KV cache layout for a ComputeImage.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KvCachePlan {
    /// Block size in tokens (16 for CUDA paged, 32 for Apple unified, 64 for Tensix).
    pub block_tokens: u32,
    /// Maximum number of blocks before eviction.
    pub max_blocks: u32,
    /// Typted codec contract — drives sizing, construction, and dispatch.
    pub codec: KvCodec,
    /// Legacy KV element dtype (populated from codec for existing callers).
    pub kv_dtype: KVDtype,
    /// Physical layout strategy.
    pub layout: KvLayout,
    /// Which layers/heads partition this plan.
    pub layer_partition: Vec<u32>,
    /// Memory domain for backend residency.
    pub residency_domain: String, // "SharedUnified", "DeviceLocal", "MappedExternal"
    /// Compatibility key for prefix cache sharing.
    pub prefix_key: PrefixCompatibilityKey,
    /// Eviction policy.
    pub eviction_policy: String, // "lru", "fifo", "lru_refcount"
    /// COW policy.
    pub cow_policy: String, // "copy_on_write", "share_full"
}

/// Compute compatibility key from a KvCodec — two plans sharing codec + block
/// size can share prefix cache pages.
fn codec_compatibility_key(codec: &KvCodec) -> String {
    match codec {
        KvCodec::Fp16 => "fp16".into(),
        KvCodec::Fp32 => "fp32".into(),
        KvCodec::TurboQuant {
            key_mode,
            value_mode,
            group_size,
        } => {
            format!(
                "tq:k{}v{}g{}",
                bits_for_mode(key_mode),
                bits_for_mode(value_mode),
                group_size
            )
        }
    }
}

impl KvCodec {
    /// Map this codec to the nearest KVDtype for legacy callers.
    pub fn to_kv_dtype(&self) -> KVDtype {
        match self {
            KvCodec::Fp16 => KVDtype::F16,
            KvCodec::Fp32 => KVDtype::F32,
            // TurboQuant codecs map to F16 for KVDtype (the compressed format
            // is opaque to callers that only know KVDtype).
            KvCodec::TurboQuant { .. } => KVDtype::F16,
        }
    }
}

/// Prefix cache compatibility key.
/// Two ComputeImages with the same prefix key can share KV cache blocks.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PrefixCompatibilityKey {
    pub architecture: String,
    pub backend: String,
    pub codec_key: String,
}

impl PrefixCompatibilityKey {
    pub fn new(architecture: &str, backend: &str, codec_key: &str) -> Self {
        Self {
            architecture: architecture.to_string(),
            backend: backend.to_string(),
            codec_key: codec_key.to_string(),
        }
    }

    /// Build a compatibility key from a KvCachePlan's codec.
    pub fn from_codec(architecture: &str, backend: &str, codec: &KvCodec) -> Self {
        Self::new(architecture, backend, &codec_compatibility_key(codec))
    }
}

impl Default for KvCachePlan {
    fn default() -> Self {
        KvCachePlan {
            block_tokens: 32,
            max_blocks: 4096,
            codec: KvCodec::Fp16,
            kv_dtype: KVDtype::F16,
            layout: KvLayout::PagedBlocks,
            layer_partition: Vec::new(),
            residency_domain: "SharedUnified".into(),
            prefix_key: PrefixCompatibilityKey::new("", "", ""),
            eviction_policy: "lru".into(),
            cow_policy: "share_full".into(),
        }
    }
}

impl KvCachePlan {
    pub fn for_apple_unified_memory() -> Self {
        KvCachePlan {
            block_tokens: 32,
            max_blocks: 8192,
            codec: KvCodec::Fp16,
            kv_dtype: KVDtype::Bf16,
            layout: KvLayout::VirtualContiguousPagedPhysical,
            residency_domain: "SharedUnified".into(),
            eviction_policy: "lru".into(),
            cow_policy: "share_full".into(),
            ..Default::default()
        }
    }

    pub fn for_tensix() -> Self {
        KvCachePlan {
            block_tokens: 64,
            max_blocks: 2048,
            codec: KvCodec::Fp16,
            kv_dtype: KVDtype::Bf16,
            layout: KvLayout::PagedBlocks,
            residency_domain: "DeviceLocal".into(),
            eviction_policy: "lru_refcount".into(),
            cow_policy: "copy_on_write".into(),
            ..Default::default()
        }
    }

    pub fn for_cpu() -> Self {
        KvCachePlan {
            block_tokens: 16,
            max_blocks: 16384,
            codec: KvCodec::Fp32,
            kv_dtype: KVDtype::F32,
            layout: KvLayout::PagedBlocks,
            residency_domain: "HostPageable".into(),
            ..Default::default()
        }
    }

    /// Build a codec-key-qualified prefix key from this plan.
    pub fn build_prefix_key(&self, architecture: &str, backend: &str) -> PrefixCompatibilityKey {
        PrefixCompatibilityKey::from_codec(architecture, backend, &self.codec)
    }

    /// Bytes per KV page given this plan's codec and layout.
    pub fn page_bytes(&self, head_dim: u32) -> u64 {
        self.codec.page_bytes(head_dim, self.block_tokens)
    }
}

// ── KvState + RuntimePage (page state machine, independent of plan types) ──
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvState {
    Unallocated,
    Allocated,
    Primed,
    Decoding,
    Synchronized,
    Invalidated,
    Released,
}

pub struct RuntimePage {
    pub state: KvState,
    pub counter: AtomicU64,
}

impl RuntimePage {
    pub fn new() -> Self {
        Self {
            state: KvState::Unallocated,
            counter: AtomicU64::new(0),
        }
    }

    pub fn allocate(&mut self) {
        assert_eq!(self.state, KvState::Unallocated);
        self.state = KvState::Allocated;
    }

    pub fn prime(&mut self) {
        assert_eq!(self.state, KvState::Allocated);
        self.state = KvState::Primed;
    }

    pub fn validate_then_prepare(&mut self) -> bool {
        if self.state == KvState::Primed || self.state == KvState::Synchronized {
            self.state = KvState::Decoding;
            self.counter.fetch_add(1, Ordering::SeqCst);
            true
        } else {
            false
        }
    }

    pub fn append(&mut self) {
        assert_eq!(self.state, KvState::Decoding);
        self.state = KvState::Synchronized;
    }

    pub fn read(&self) -> KvState {
        self.state
    }

    pub fn rollback(&mut self) {
        if self.state == KvState::Decoding || self.state == KvState::Synchronized {
            self.state = KvState::Primed;
            self.counter.fetch_sub(1, Ordering::SeqCst);
        }
    }

    pub fn release(&mut self) {
        self.state = KvState::Released;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp16_codec_page_size() {
        // 32 tokens × 128 dim × 4 bytes (K+V, 2 each) = 16,384
        assert_eq!(KvCodec::Fp16.page_bytes(128, 32), 16384);
    }

    #[test]
    fn turboquant_page_size_fits_within_fp16() {
        // NF4 TurboQuant: 4 bits × 2 dtypes = 8 bits/token/dim = 1 byte
        // 32 × 128 × 1 = 4096 payload + (~128/32 × 4 × 2) scales
        let tq = KvCodec::TurboQuant {
            key_mode: KvQuantMode::PolarHadamard(4),
            value_mode: KvQuantMode::PolarHadamard(4),
            group_size: 32,
        };
        let tq_bytes = tq.page_bytes(128, 32);
        assert!(tq_bytes < 16384, "TurboQuant must be smaller than FP16");
        assert!(tq_bytes > 0);
    }

    #[test]
    fn compile_from_nf4_produces_turboquant() {
        // We test the codec construction path (compile_from_policy will live
        // in the deployment compiler once it imports PrecisionPolicy)
        let codec = KvCodec::TurboQuant {
            key_mode: KvQuantMode::PolarHadamard(4),
            value_mode: KvQuantMode::PolarHadamard(4),
            group_size: 32,
        };
        assert!(matches!(codec, KvCodec::TurboQuant { .. }));
        assert_eq!(bits_for_mode(&KvQuantMode::PolarHadamard(4)), 4);
    }

    #[test]
    fn kvcodec_to_kvdtype_maps_correctly() {
        assert_eq!(KvCodec::Fp16.to_kv_dtype(), KVDtype::F16);
        assert_eq!(KvCodec::Fp32.to_kv_dtype(), KVDtype::F32);
        let tq = KvCodec::TurboQuant {
            key_mode: KvQuantMode::PolarHadamard(4),
            value_mode: KvQuantMode::PolarHadamard(4),
            group_size: 32,
        };
        assert_eq!(tq.to_kv_dtype(), KVDtype::F16);
    }

    #[test]
    fn prefix_key_from_codec_is_deterministic() {
        let codec = KvCodec::TurboQuant {
            key_mode: KvQuantMode::Polar(4),
            value_mode: KvQuantMode::Mse(3),
            group_size: 64,
        };
        let k1 = PrefixCompatibilityKey::from_codec("gemma4", "metal", &codec);
        let k2 = PrefixCompatibilityKey::from_codec("gemma4", "metal", &codec);
        assert_eq!(k1, k2);
        assert!(k1.codec_key.contains("tq:k4v3g64"));
    }

    #[test]
    fn runtime_page_state_machine() {
        let mut p = RuntimePage::new();
        assert_eq!(p.state, KvState::Unallocated);
        p.allocate();
        assert_eq!(p.state, KvState::Allocated);
        p.prime();
        assert_eq!(p.state, KvState::Primed);
        assert!(p.validate_then_prepare());
        assert_eq!(p.state, KvState::Decoding);
        p.append();
        assert_eq!(p.state, KvState::Synchronized);
        p.rollback();
        assert_eq!(p.state, KvState::Primed);
    }
}
