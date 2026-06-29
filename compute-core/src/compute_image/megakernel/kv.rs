//! KV cache ternary packing and decompression constants.
//!
//! The Gemma 4 GPU megakernel stores K/V cache entries as ternary-packed
//! nibbles (Base-3 digits) with FP16 block-scales.  Each block of 256
//! dimensions packs into 13 u32 words (20 ternary digits per u32) plus
//! one half-precision scale = 54 bytes.
//!
//! The actual pack/unpack logic runs inside the Metal shader
//! (`kernels::SHADER_SRC`).  Module-level constants here are shared
//! between the host-side buffer allocation (`pipeline::Megakernel::launch`)
//! and the GPU kernel.


// ── Ternary KV block constants ────────────────────────────────────
#[allow(dead_code)]
pub const KV_BLOCK: u32 = 256;
pub const KV_NIBBLES_U32: u32 = 13; // 256 values / 20 per u32 = 13 u32
pub const KV_BLOCK_BYTES: u64 = (KV_NIBBLES_U32 as u64) * 4 + 2; // 13*4 + 2 = 54 bytes per block
