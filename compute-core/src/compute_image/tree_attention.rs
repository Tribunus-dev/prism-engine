//! Tree speculative decoding verification kernel.
//!
//! Processes N candidate tokens in parallel using a boolean ancestor mask,
//! computing per-candidate output logits via a 640-weight Base-3 tile GEMV.
//! Each threadgroup handles one candidate; a 32-thread warp processes the
//! tile.  The ancestor mask ([N] u32 bitmask) controls which lanes contribute
//! before the warp reduction, implementing tree-structured attention:
//! candidate gid's logits combine embeddings from ancestor candidates only.
//!
//! # Buffer layout (Metal kernel)
//!
//! | Index | Content | Type |
//! |---|---|---|
//! | 0 | packed Base-3 weights (same layout as `CimageDeployment.weights_buffer`) | `[VOCAB × BLOCKS × 32] u32` |
//! | 1 | candidate token embeddings | `[N × HEAD_DIM] half` |
//! | 2 | ancestor mask | `[N] u32` (bit `j` = candidate j is ancestor of gid) |
//! | 3 | output logits | `[N × VOCAB] half` |
//! | 4 | num_candidates (N) | `uint` |
//! | 5 | head_dim | `uint` |
//! | 6 | vocab_size | `uint` |

use std::path::PathBuf;

// ── Constants ────────────────────────────────────────────────────────────
const LANES: u32 = 32;
const PER_LANE: u32 = 20;
const TILE: u32 = LANES * PER_LANE; // 640 weights per warp wave

// ── Metal Kernel Source ─────────────────────────────────────────────────

const KERNEL_SRC: &str = r##"#include <metal_stdlib>
using namespace metal;

constant uint LANES = 32;
constant uint PER_LANE = 20;
constant uint TILE = 640;
constant uint MAGIC_DIV3 = 2863311531u;

inline uint fast_div3(uint v) {
    return ((uint64_t)v * (uint64_t)MAGIC_DIV3) >> 33;
}

inline uint fast_mod3(uint v) {
    return v - fast_div3(v) * 3u;
}

/// Tree attention verification kernel.
///
/// Each threadgroup handles one candidate token (gid = candidate index).
/// Each warp lane loads one u32 of packed Base-3 weights and the
/// corresponding slice of candidate gid's ancestor embedding. Before the
/// warp reduction, the ancestor mask zeros contributions from non-ancestor
/// positions.
///
/// Dispatch: N threadgroups × 32 threads.
kernel void tree_attention_verify(
    device const uint*   packed_weights [[buffer(0)]],
    device const half*   candidates     [[buffer(1)]],
    device const uint*   ancestor_mask  [[buffer(2)]],
    device half*         logits_out     [[buffer(3)]],
    constant uint&       num_candidates [[buffer(4)]],
    constant uint&       head_dim       [[buffer(5)]],
    constant uint&       vocab_size     [[buffer(6)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lane_id [[thread_index_in_simdgroup]])
{
    if (gid >= num_candidates) return;

    uint blocks = (head_dim + TILE - 1) / TILE;
    uint mask = ancestor_mask[gid];

    for (uint row = 0; row < vocab_size; ++row) {
        uint row_base = row * blocks * LANES;
        float block_sum = 0.0;

        for (uint b = 0; b < blocks; ++b) {
            uint base = row_base + b * LANES;
            uint val = packed_weights[base + lane_id];

            float partial = 0.0;
            uint dim_base = b * TILE + lane_id * PER_LANE;
            uint v = val;

            for (uint i = 0; i < PER_LANE; ++i) {
                uint dim = dim_base + i;
                uint rem = fast_mod3(v);
                int w = (int)rem - 1;
                if (dim < head_dim && lane_id < num_candidates) {
                    half embed_val = candidates[lane_id * head_dim + dim];
                    partial += (float)w * (float)embed_val;
                }
                v = fast_div3(v);
            }
            block_sum += partial;
        }

        // Apply ancestor mask: zero out non-ancestor lane contributions
        if ((mask & (1u << lane_id)) == 0) {
            block_sum = 0.0;
        }

        // Warp reduction (all 32 lanes → lane 0)
        block_sum += simd_shuffle_xor(block_sum, 1);
        block_sum += simd_shuffle_xor(block_sum, 2);
        block_sum += simd_shuffle_xor(block_sum, 4);
        block_sum += simd_shuffle_xor(block_sum, 8);
        block_sum += simd_shuffle_xor(block_sum, 16);

        if (lane_id == 0) {
            logits_out[gid * vocab_size + row] = (half)block_sum;
        }
    }
}
"##;

// ── TreeAttention ────────────────────────────────────────────────────────

/// Tree speculative decoding verification engine.
///
/// Wraps a compiled Metal kernel that verifies N candidate tokens in
/// parallel using a boolean ancestor mask.  Each candidate's output logit
/// is the warp-reduced sum over ancestor-contributed lane partials.
pub struct TreeAttention {
    pso: metal::ComputePipelineState,
}

impl TreeAttention {
    /// Compile the tree attention Metal kernel.
    ///
    /// Uses `xcrun` to compile the Metal source and build a
    /// `ComputePipelineState`.  Calling this is O(1) per process — the
    /// same pipeline is reused for all invocation sizes.
    pub fn new(device: &metal::Device) -> Result<Self, String> {
        let tmp = std::env::temp_dir().join("tribunus-tree-attn");
        std::fs::create_dir_all(&tmp).map_err(|e| format!("temp dir: {e}"))?;

        let metal_path: PathBuf = tmp.join("tree_attention.metal");
        let air_path: PathBuf = tmp.join("tree_attention.air");
        let lib_path: PathBuf = tmp.join("tree_attention.metallib");

        std::fs::write(&metal_path, KERNEL_SRC).map_err(|e| format!("write .metal: {e}"))?;

        let status = std::process::Command::new("xcrun")
            .args(["-sdk", "macosx", "metal", "-std=metal3.2", "-O3", "-c"])
            .arg(metal_path.to_str().unwrap())
            .arg("-o")
            .arg(air_path.to_str().unwrap())
            .status()
            .map_err(|e| format!("xcrun metal: {e}"))?;
        if !status.success() {
            return Err("metal compilation failed".into());
        }

        let status = std::process::Command::new("xcrun")
            .args(["-sdk", "macosx", "metallib", "-o"])
            .arg(lib_path.to_str().unwrap())
            .arg(air_path.to_str().unwrap())
            .status()
            .map_err(|e| format!("xcrun metallib: {e}"))?;
        if !status.success() {
            return Err("metallib linking failed".into());
        }

        let bytes = std::fs::read(&lib_path).map_err(|e| format!("read metallib: {e}"))?;
        let library = device
            .new_library_with_data(&bytes)
            .map_err(|e| format!("new_library: {e}"))?;
        let function = library
            .get_function("tree_attention_verify", None)
            .map_err(|e| format!("get_function: {e}"))?;
        let pso = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| format!("new_pso: {e}"))?;

        Ok(Self { pso })
    }

    /// Verify N candidate tokens in parallel using a tree ancestor mask.
    ///
    /// # Parameters
    ///
    /// * `queue` — Metal command queue.
    /// * `weights` — Metal buffer containing packed Base-3 weights (same
    ///   layout as `CimageDeployment::weights_buffer`).
    /// * `candidates` — Metal buffer with `[N × HEAD_DIM]` FP16 candidate
    ///    token embeddings.
    /// * `mask` — Ancestor bitmask array: `mask[i]` bit `j` set means
    ///   candidate `j` is an ancestor of candidate `i`.  Must have exactly
    ///   `N` elements.  `N` must be ≤ 32.
    /// * `logits_out` — Metal buffer for the output `[N × VOCAB]` FP16
    ///   logits.  Must have capacity `N × VOCAB × sizeof(half)`.
    ///
    /// # Constraints
    ///
    /// * N (number of candidates) must be ≤ 32 (one warp per candidate).
    /// * The weights buffer, candidates buffer, and output buffer must
    ///   already exist and have sufficient capacity.
    /// * HEAD_DIM and VOCAB are inferred from the weights buffer geometry
    ///   (not passed explicitly — callers must ensure buffer sizes match).
    pub fn verify_candidates(
        &self,
        queue: &metal::CommandQueue,
        weights: &metal::Buffer,
        candidates: &metal::Buffer,
        mask: &[u32],
        logits_out: &metal::Buffer,
    ) -> Result<(), String> {
        let n = mask.len() as u32;
        if n > 32 {
            return Err(format!("N={} exceeds max 32 (one warp per candidate)", n));
        }
        if n == 0 {
            return Ok(());
        }

        // We need HEAD_DIM and VOCAB_SIZE to dispatch correctly.
        // The weights buffer layout is [VOCAB × blocks × 32] u32.
        // We don't know HEAD_DIM or VOCAB at compile time, so compute
        // the griddim from the output buffer size and candidate buffer.
        //
        // Candidates:   [N × HEAD_DIM] half   → HEAD_DIM = cand_len / N
        // Logits out:   [N × VOCAB] half      → VOCAB = logits_len / N
        // Weights:      [VOCAB × blocks × 32] u32 → verify consistency
        let cand_len = candidates.length() as usize / 2; // half = 2 bytes
        let logits_len = logits_out.length() as usize / 2;
        let weights_len = weights.length() as usize / 4; // u32 = 4 bytes

        let head_dim = cand_len / n as usize;
        let vocab_size = logits_len / n as usize;

        if head_dim == 0 || vocab_size == 0 {
            return Err("zero-dimension buffers".into());
        }
        if head_dim * n as usize != cand_len {
            return Err(format!(
                "candidates buffer inconsistent: {cand_len} halves, N={n}, HEAD_DIM={head_dim}"
            ));
        }
        if vocab_size * n as usize != logits_len {
            return Err(format!(
                "logits_out buffer inconsistent: {logits_len} halves, N={n}, VOCAB={vocab_size}"
            ));
        }

        // Validate weight buffer geometry
        let blocks = (head_dim + TILE as usize - 1) / TILE as usize;
        let expected_weights_len = vocab_size * blocks * LANES as usize;
        if weights_len < expected_weights_len {
            return Err(format!(
                "weights buffer too small: have {weights_len} u32, need {expected_weights_len} for VOCAB={vocab_size}, blocks={blocks}"
            ));
        }

        // Create a Metal buffer for the mask
        let mask_buf = queue.device().new_buffer_with_data(
            mask.as_ptr() as *const std::ffi::c_void,
            (n as usize * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();

        enc.set_compute_pipeline_state(&self.pso);
        enc.set_buffer(0, Some(weights), 0);
        enc.set_buffer(1, Some(candidates), 0);
        enc.set_buffer(2, Some(&mask_buf), 0);
        enc.set_buffer(3, Some(logits_out), 0);
        enc.set_bytes(4, 4, &n as *const u32 as *const std::ffi::c_void);
        let hd = head_dim as u32;
        enc.set_bytes(5, 4, &hd as *const u32 as *const std::ffi::c_void);
        let vs = vocab_size as u32;
        enc.set_bytes(6, 4, &vs as *const u32 as *const std::ffi::c_void);

        enc.dispatch_thread_groups(
            metal::MTLSize {
                width: n as u64,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );

        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        Ok(())
    }
}
