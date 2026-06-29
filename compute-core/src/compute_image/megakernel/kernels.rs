//! Metal kernel source, architecture constants, and on-the-fly compilation.
//!
//! Provides the full Gemma 4 48-layer transformer Metal shader source string
//! alongside Rust-side compilation helpers and shared layout constants.

use metal::*;

// ── Architecture constants (Gemma 4 12B Unified) ───────────────────
#[allow(dead_code)]
pub const HIDDEN_DIM: u32 = 3840;
pub const LAYERS: u32 = 48;
#[allow(dead_code)]
pub const NUM_Q_HEADS: u32 = 16;
pub const NUM_KV_HEADS: u32 = 8;
#[allow(dead_code)]
pub const HEAD_DIM: u32 = 256;
pub const GLOBAL_HEAD_DIM: u32 = 512;
#[allow(dead_code)]
pub const FFN_INTERMEDIATE: u32 = 15360;
pub const VOCAB_SIZE: u32 = 262144;
pub const MAX_CONTEXT: u32 = 2048; // KV cache slots (limited by 16 GB SRAM + device mem)
pub const NUM_CENTROIDS: u32 = 256;
pub const NUM_MTP_HEADS: u32 = 4;
pub const MTP_HIDDEN: u32 = 2048;
pub const MTP_FFN_INTER: u32 = 8192;
pub const MTP_TILES: u32 = (MTP_HIDDEN + 640) / 640; // 4
pub const MTP_TILES_FFN: u32 = (MTP_FFN_INTER + 640) / 640; // 13
pub const MAX_DRAFT_CANDIDATES: u32 = 5;
pub const DRAFT_HIDDEN: u32 = 768;
pub const TILE: u32 = 640;
#[allow(dead_code)]
const MAGIC_DIV3: u32 = 2863311531;

// ── Work queue constants ───────────────────────────────────────────
pub const NUM_SLOTS: u32 = 256;
pub const SLOT_U32_COUNT: u32 = 4 + VOCAB_SIZE; // 262148
pub const SLOT_BYTE_COUNT: u64 = SLOT_U32_COUNT as u64 * 4; // 1,048,592

// ── Ternary KV block constants ────────────────────────────────────

// ── Weight-offset constants for the per-layer matrix layout ────────
// Each matrix's flat element count BEFORE Base-3 nibble packing.
// For tile-GEMV indexing we compute nibble offsets at runtime.
#[allow(dead_code)]
const Q_COLS: u32 = NUM_Q_HEADS * HEAD_DIM; // 4096
#[allow(dead_code)]
const KV_COLS: u32 = NUM_KV_HEADS * HEAD_DIM; // 2048
#[allow(dead_code)]
const O_ROWS: u32 = Q_COLS; // 4096
#[allow(dead_code)]
const DOWN_ROWS: u32 = FFN_INTERMEDIATE; // 15360

// ====================================================================
//  Metal Shader Source
// ====================================================================

pub const SHADER_SRC: &str = r##"#include <metal_stdlib>
using namespace metal;

constant uint HIDDEN_DIM      = 3840;
constant uint LAYERS          = 48;
constant uint NUM_Q_HEADS     = 16;
constant uint NUM_KV_HEADS    = 8;
constant uint HEAD_DIM        = 256;
constant uint GLOBAL_HEAD_DIM = 512;
constant uint FFN_INTER      = 15360;
constant uint VOCAB_SIZE      = 262144;
constant uint MAX_CTX         = 2048;
constant uint MAGIC_DIV3      = 2863311531u;
constant uint O_ROWS          = 4096;
constant uint DOWN_ROWS       = 15360;
constant uint NUM_CENTROIDS   = 256;

constant uint NUM_SINKS = 4;     // first 4 positions are permanent attention sinks (StreamingLLM)
constant uint KV_BLOCK           = 256;
constant uint KV_NIBBLES_U32     = 13;


// -- Work queue constants -------------------------------------------------
constant uint SLOT_WORDS = 4 + VOCAB_SIZE; // 262148
constant uint NUM_SLOTS  = 256;             // concurrent decode slots
constant uint RING_SIZE = 512;

// -- Tile dimensions ------------------------------------------------
constant uint LANES    = 32u;
constant uint PER_LANE = 20u;
constant uint TILE     = 640u;     // 32 × 20 weights per warp-coalesced wave

// Tile count per matrix (ceil(dim / 640))
constant uint Q_TILES     = (NUM_Q_HEADS * HEAD_DIM + TILE - 1) / TILE;  // 7
constant uint KV_TILES    = (NUM_KV_HEADS * HEAD_DIM + TILE - 1) / TILE; // 4
constant uint HID_TILES   = (HIDDEN_DIM + TILE - 1) / TILE;              // 6
constant uint FFN_TILES   = (FFN_INTER + TILE - 1) / TILE;              // 24
constant uint DOWN_TILES  __attribute__((unused)) = (FFN_INTER + TILE - 1) / TILE;              // 24
constant uint VOCAB_TILES __attribute__((unused)) = (VOCAB_SIZE + TILE - 1) / TILE;             // 410
constant uint NUM_MTP_HEADS  = 4;  // number of future-token predictors
constant uint MTP_HIDDEN     = 2048;
constant uint MTP_FFN_INTER  = 8192;
constant uint MTP_TILES      = (MTP_HIDDEN + TILE - 1) / TILE;  // 4
constant uint MTP_TILES_FFN  = (MTP_FFN_INTER + TILE - 1) / TILE; // 13
// ── Draft model architecture (100M params, lightweight speculative drafter) ──
constant uint DRAFT_LAYERS       = 8u;
constant uint DRAFT_HIDDEN       = 768u;
constant uint DRAFT_NUM_HEADS    = 8u;
constant uint DRAFT_NUM_KV_HEADS = 4u;  // GQA ratio 2:1
constant uint DRAFT_HEAD_DIM     = 96u;  // 768 / 8
constant uint DRAFT_FFN_INTER    = 2048u;
constant uint DRAFT_TILES        = (DRAFT_HIDDEN + TILE - 1) / TILE;   // 2
constant uint DRAFT_FFN_TILES    = (DRAFT_FFN_INTER + TILE - 1) / TILE; // 4
constant uint DRAFT_Q_TILES      = (DRAFT_NUM_HEADS * DRAFT_HEAD_DIM + TILE - 1) / TILE;   // 2
constant uint DRAFT_KV_TILES     = (DRAFT_NUM_KV_HEADS * DRAFT_HEAD_DIM + TILE - 1) / TILE; // 1
constant uint DRAFT_HID_TILES    = (DRAFT_HIDDEN + TILE - 1) / TILE;  // 2
// Per-layer nibble offsets for draft model weight layout
constant uint DRAFT_Q_OFF    = 0u;
constant uint DRAFT_K_OFF    = DRAFT_Q_OFF + DRAFT_HIDDEN * DRAFT_Q_TILES * LANES;
constant uint DRAFT_V_OFF    = DRAFT_K_OFF + DRAFT_HIDDEN * DRAFT_KV_TILES * LANES;
constant uint DRAFT_O_OFF    = DRAFT_V_OFF + DRAFT_HIDDEN * DRAFT_KV_TILES * LANES;
constant uint DRAFT_GATE_OFF = DRAFT_O_OFF + DRAFT_HIDDEN * DRAFT_HID_TILES * LANES;
constant uint DRAFT_UP_OFF   = DRAFT_GATE_OFF + DRAFT_HIDDEN * DRAFT_FFN_TILES * LANES;
constant uint DRAFT_DOWN_OFF = DRAFT_UP_OFF + DRAFT_HIDDEN * DRAFT_FFN_TILES * LANES;
constant uint DRAFT_LAYER_STRIDE = DRAFT_DOWN_OFF + DRAFT_FFN_INTER * DRAFT_HID_TILES * LANES;

// Per-layer nibble offsets (in u32 units) for each matrix.
// Computed from row × tile_count × LANES.
constant uint Q_OFF    = 0u;
constant uint K_OFF    = Q_OFF    + HIDDEN_DIM * Q_TILES * LANES;   // 3840×7×32
constant uint V_OFF    = K_OFF    + HIDDEN_DIM * KV_TILES * LANES;  // 3840×4×32
constant uint O_OFF    = V_OFF    + HIDDEN_DIM * KV_TILES * LANES;  // 3840×4×32
constant uint GATE_OFF = O_OFF    + O_ROWS     * HID_TILES * LANES; // 4096×6×32
constant uint UP_OFF   = GATE_OFF + HIDDEN_DIM * FFN_TILES * LANES; // 3840×24×32
constant uint DOWN_OFF = UP_OFF   + HIDDEN_DIM * FFN_TILES * LANES; // 3840×24×32
constant uint LAYER_STRIDE = DOWN_OFF + DOWN_ROWS * HID_TILES * LANES; // 15360×6×32

// ---- Helpers -------------------------------------------------------

inline uint fast_div3(uint v) {
    return ((uint64_t)v * (uint64_t)MAGIC_DIV3) >> 33;
}
inline uint fast_mod3(uint v) {
    return v - fast_div3(v) * 3u;
}

/// Single-thread tile GEMV returning one dot product.
///   packed_weights[tile_base + b*LANES + lane_id]  =  u32 holding 20 Base-3 weights
///   in_vec[act_base + i]                            =  FP16 activation
float tile_gemv(device const uint* w, uint tile_base, uint ntiles, uint lane,
                threadgroup const half* in_vec) {
    float acc = 0.0;
    for (uint b = 0; b < ntiles; ++b) {
        uint val = w[tile_base + b * LANES + lane];
        uint act_base = b * TILE + lane * PER_LANE;
        for (uint i = 0; i < PER_LANE; ++i) {
            uint rem = fast_mod3(val);
            int wgt = (int)rem - 1;
            if (wgt != 0) {
                acc += (float)in_vec[act_base + i] * (float)wgt;
            }
            val = fast_div3(val);
        }
    }
    return acc;
}

/// Warp reduction tree (5 shuffle steps, result on lane 0).
inline float warp_sum(float val) {
    val += simd_shuffle_xor(val, 1);
    val += simd_shuffle_xor(val, 2);
    val += simd_shuffle_xor(val, 4);
    val += simd_shuffle_xor(val, 8);
    val += simd_shuffle_xor(val, 16);
    return val;
}

// ---- RMSNorm -------------------------------------------------------

/// In-place RMSNorm on a 3840-d vector using all tg_size threads.
inline void fast_rmsnorm(threadgroup half* vec,
                         device const half* weight,
                         uint tid, uint tg_size,
                         threadgroup float* sums) {
    sums[tid] = 0.0;
    for (uint i = tid; i < HIDDEN_DIM; i += tg_size) {
        float v = (float)vec[i];
        sums[tid] += v * v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sums[tid] += sums[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rcp = rsqrt(sums[0] / (float)HIDDEN_DIM + 1e-6);
    for (uint i = tid; i < HIDDEN_DIM; i += tg_size) {
        vec[i] = (half)((float)vec[i] * rcp * (float)weight[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// ---- p-RoPE --------------------------------------------------------

inline void apply_rope(threadgroup half* qk, uint num_heads, uint h_dim,
                       uint seq_pos, uint tid, uint tg_size) {
    uint rope_dim = 64u; // partial factor 0.25 of 256
    float theta = 1e6;
    for (uint h = 0; h < num_heads; ++h) {
        uint base = h * h_dim;
        for (uint i = tid; i < rope_dim / 2; i += tg_size) {
            uint c = base + 2 * i;
            float freq = 1.0 / pow(theta, (float)(2 * i) / (float)rope_dim);
            float ang = (float)seq_pos * freq;
            float c0 = cos(ang), s0 = sin(ang);
            float x0 = (float)qk[c], x1 = (float)qk[c + 1];
            qk[c]     = (half)(x0 * c0 - x1 * s0);
            qk[c + 1] = (half)(x0 * s0 + x1 * c0);
        }
    }
}

// ---- SwiGLU --------------------------------------------------------

inline float swiglu(float g, float u) {
    return (g / (1.0 + exp(-g))) * u;
}

// ---- GQA Attention (inner loop over KV heads) ----------------------

/// Process one KV head group (2 query heads, 1 KV head).
/// Reads/writes q_chunk[k][d] via lane-level indexing.
/// Loads K_cache/V_cache from device memory for all past positions.
/// num_cached = number of valid KV cache positions (may be capped at MAX_CTX with eviction).
void gqa_group(device const half* kv_k, device const half* kv_v,
               threadgroup const half* q_buf, uint kv_h,
               uint num_cached, uint active_h_dim,
               uint tid, uint tg_size,
               threadgroup float* scores,   // [2 × MAX_CTX] float scratch
               threadgroup half* out_accum) // [2 × active_h_dim] output
{

    uint N = NUM_KV_HEADS * active_h_dim; // per-position KV stride

    // First pass: compute scores and global max
    float global_max = -1e10;
    for (uint p = tid; p < num_cached; p += tg_size) {
        uint kv_base = p * N + kv_h * active_h_dim;
        for (uint qh = 0; qh < 2; ++qh) {
            float s = 0.0;
            uint q_base = qh * active_h_dim;
            for (uint d = 0; d < active_h_dim; ++d) {
                s += (float)q_buf[q_base + d] * (float)kv_k[kv_base + d];
            }
            scores[qh * MAX_CTX + p] = s;
            if (s > global_max) global_max = s;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Second pass: exp(score - max) → per-head sum
    float sum_exp_q0 = 0.0;
    float sum_exp_q1 = 0.0;
    for (uint p = tid; p < num_cached; p += tg_size) {
        float e0 = exp(scores[0 * MAX_CTX + p] - global_max);
        float e1 = exp(scores[1 * MAX_CTX + p] - global_max);
        scores[0 * MAX_CTX + p] = e0;
        scores[1 * MAX_CTX + p] = e1;
        sum_exp_q0 += e0;
        sum_exp_q1 += e1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_sum_q0 = 1.0 / sum_exp_q0;
    float inv_sum_q1 = 1.0 / sum_exp_q1;

    // Third pass: weighted sum of V (per-head softmax)
    for (uint d = tid; d < active_h_dim; d += tg_size) {
        float v0 = 0.0, v1 = 0.0;
        for (uint p = 0; p < num_cached; ++p) {
            float s0 = scores[0 * MAX_CTX + p] * inv_sum_q0;
            float s1 = scores[1 * MAX_CTX + p] * inv_sum_q1;
            uint kv_base = p * N + kv_h * active_h_dim;
            float vv = (float)kv_v[kv_base + d];
            v0 += s0 * vv;
            v1 += s1 * vv;
        }
        out_accum[0 * active_h_dim + d] = (half)v0;
        out_accum[1 * active_h_dim + d] = (half)v1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}


// ---- Main Kernel ---------------------------------------------------

kernel void gemma4_full_decode_persistent(
    device const uint*    ternary_w     [[buffer(0)]],
    device const half*    scales        [[buffer(1)]],
    device const half*    norms         [[buffer(2)]],  // aux: first part is norms
    device const uint*    embed_clust   [[buffer(3)]],  // ternary nibbles (reordered by cluster)
    device const uint*    centroids_ternary [[buffer(4)]],  // ternary-packed centroids (256 x 192 u32)
    device const uint*    cluster_map   [[buffer(5)]],  // VOCAB_SIZE x u32 cluster assignments
    device uint*          kv_k_nibbles  [[buffer(6)]],  // ternary-packed K nibbles
    device uint*          kv_v_nibbles  [[buffer(7)]],  // ternary-packed V nibbles
    device half*          kv_k_scales   [[buffer(8)]],  // FP16 block scales for K
    device half*          kv_v_scales   [[buffer(9)]],  // FP16 block scales for V
    device const half*    embed_scales  [[buffer(14)]],  // FP16 block scales for embed
    device const half*    centroid_scales   [[buffer(15)]], // FP16 block scales for centroids (256 x 15)
    device half*          centroid_scratch  [[buffer(16)]], // decompressed FP16 centroids (256 x HIDDEN_DIM)
    device atomic_uint*   centroid_decompress_progress [[buffer(17)]], // atomic progress counter
    device half*          kv_scratch_k  [[buffer(19)]], // decompressed K scratch (1 layer)
    device half*          kv_scratch_v  [[buffer(20)]], // decompressed V scratch (1 layer)
    device half*          entropy_map   [[buffer(21)]],
    device uint*          ring_entries  [[buffer(22)]],  // submission ring entries (WorkEntry[4 x RING_SIZE])
    device atomic_uint*   ring_tail     [[buffer(23)]],  // GPU-claimed tail offset
    device half*          slot_logits_base [[buffer(24)]], // per-slot logits (NUM_SLOTS x VOCAB_SIZE half)
    device atomic_uint*   completion_counter [[buffer(25)]], // incremented after COMPLETED
    device const uint*    mtp_ternary_w     [[buffer(26)]], // MTP head ternary weights
    device const uint*    draft_ternary_w   [[buffer(10)]],  // draft model ternary nibble weights
    device const half*    draft_scales      [[buffer(11)]],  // draft model block scales
    device const half*    draft_norms       [[buffer(12)]],  // draft model RMSNorm weights
    device uint*          draft_output      [[buffer(28)]],  // draft output: [N, tok_id0..4, logprob0..4]
    uint tid    [[thread_index_in_threadgroup]],
    uint tg_sz  [[threads_per_threadgroup]])
{
    // --- SRAM ----------------------------------------------------------
    // Budget: 7.5 + 7.5 + 2 + 1 + 1 + 0.008 = ~19 KB < 32 KB

    threadgroup half h_buf[HIDDEN_DIM];      // 7.5 KB --- residual stream
    threadgroup half n_buf[HIDDEN_DIM];      // 7.5 KB --- norm scratch
    threadgroup half q_chunk[1024];           // 2 KB  --- 2 Q-heads x 512 max
    threadgroup float shared_sums[256];       // 1 KB  --- tree reduction
    threadgroup float centroid_scores[256];   // 1 KB  --- centroid scout
    threadgroup uint cluster_bounds[2];       // 8 B   --- [cluster_start, cluster_end)
    threadgroup float entropy_acc[MAX_CTX];   // 8 KB  --- per-position entropy accumulator

    while (true) {
        // --- Idle work: centroid decompression --------------------------
        uint decomp_progress = atomic_load_explicit(
            centroid_decompress_progress, memory_order_relaxed);
        if (decomp_progress < NUM_CENTROIDS) {
            uint idx = NUM_CENTROIDS;
            if ((tid & 31) == 0) {
                idx = atomic_fetch_add_explicit(
                    centroid_decompress_progress, 1, memory_order_relaxed);
            }
            idx = simd_broadcast(idx, 0);
            if (idx < NUM_CENTROIDS) {
                device const uint* src = centroids_ternary + idx * HID_TILES * LANES;
                device half* dst = centroid_scratch + idx * HIDDEN_DIM;
                uint lane = tid & 31;
                for (uint b = 0; b < HID_TILES; ++b) {
                    uint val = src[b * LANES + lane];
                    uint act_base = b * TILE + lane * PER_LANE;
                    for (uint i = 0; i < PER_LANE; ++i) {
                        uint rem = fast_mod3(val);
                        int wgt = (int)rem - 1;
                        uint flat_idx = act_base + i;
                        uint block_idx = flat_idx / 256;
                        half s = centroid_scales[
                            idx * ((HIDDEN_DIM + 255) / 256) + block_idx];
                        dst[act_base + i] = (half)((float)wgt * (float)s);
                        val = fast_div3(val);
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_device);

        // --- Atomic ring dequeue from submission queue ------------------
        bool processed = false;
        uint my_tail = atomic_fetch_add_explicit(
            ring_tail, 1, memory_order_relaxed);
        uint idx = my_tail % RING_SIZE;
        device uint* entry = ring_entries + idx * 4;
        uint entry_state = atomic_load_explicit(
            (device atomic_uint*)entry, memory_order_relaxed);
        uint kind = entry_state >> 2;
        if ((entry_state & 3) == 1) {  // SUBMITTED (low 2 bits = state, upper = kind)
            uint expected = entry_state;
            if (atomic_compare_exchange_weak_explicit(
                (device atomic_uint*)entry, &expected, 2 | (kind << 2),  // CLAIMED
                memory_order_relaxed, memory_order_relaxed)) {
                uint current_token = entry[1];
                uint current_pos   = entry[2];
                uint kv_slot_id    = entry[3];

                // Number of valid KV cache positions
                uint num_cached = min(current_pos + 1, MAX_CTX);

                // KV cache offset for this partition
                uint slot_kv_offset = kv_slot_id * MAX_CTX * NUM_KV_HEADS * GLOBAL_HEAD_DIM * LAYERS;

                // Logits output goes to the slot's logits region
                device half* slot_logits = slot_logits_base + kv_slot_id * VOCAB_SIZE;

    // --- Stage 0: Embedding lookup from embed_clust via cluster_map ----------
    if (tid == 0) {
        uint c = cluster_map[current_token];
        uint cluster_start = 0;
        for (uint pos = 0; pos < VOCAB_SIZE; ++pos) {
            if (cluster_map[pos] < c) ++cluster_start;
        }
        uint rank = 0;
        for (uint pos = 0; pos < current_token; ++pos) {
            if (cluster_map[pos] == c) ++rank;
        }
        uint embed_row = cluster_start + rank;
        cluster_bounds[0] = embed_row;
    }

    // Tile-GEMV: decode ternary embed row into h_buf
    uint simd_lane = tid & 31;
    uint simd_id = tid / 32;
    uint sel_row = cluster_bounds[0];
    uint tile_base = sel_row * HID_TILES * LANES;
    for (uint b = simd_id; b < HID_TILES; b += tg_sz / 32) {
        uint val = embed_clust[tile_base + b * LANES + simd_lane];
        uint act_base = b * TILE + simd_lane * PER_LANE;
        for (uint i = 0; i < PER_LANE; ++i) {
            uint rem = fast_mod3(val);
            int wgt = (int)rem - 1;
            uint flat_idx = b * TILE + simd_lane * PER_LANE + i;
            uint block_idx = flat_idx / 256;
            float s = (float)embed_scales[block_idx];
            h_buf[act_base + i] = (half)((float)wgt * s);
            val = fast_div3(val);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (kind == 0) {
    // --- 48-layer loop -------------------------------------------------
    for (uint layer = 0; layer < LAYERS; ++layer) {
        bool shared = ((layer + 1) % 6 == 0);
        uint h_dim = shared ? GLOBAL_HEAD_DIM : HEAD_DIM;
        uint layer_base = layer * LAYER_STRIDE;

        // --- 1. Input RMSNorm ------------------------------------------
        device const half* in_norm_w = norms + layer * HIDDEN_DIM;
        for (uint i = tid; i < HIDDEN_DIM; i += tg_sz) n_buf[i] = h_buf[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        fast_rmsnorm(n_buf, in_norm_w, tid, tg_sz, shared_sums);

        // --- 2. Q projection -------------------------------------------
        uint qw_base = layer_base + Q_OFF;
        uint kw_base = layer_base + K_OFF;
        uint vw_base = layer_base + V_OFF;
        uint ow_base = layer_base + O_OFF;

        // Init attention-output accumulator in n_buf
        for (uint i = tid; i < HIDDEN_DIM; i += tg_sz) n_buf[i] = 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute cache position with StreamingLLM + cyclic FIFO eviction
        uint kv_cache_pos = current_pos;
        if (kv_cache_pos >= MAX_CTX) {
            kv_cache_pos = NUM_SINKS + (kv_cache_pos - NUM_SINKS) % (MAX_CTX - NUM_SINKS);
        }

        // ── Decompress K/V for this layer from ternary ──
        uint scratch_stride = NUM_KV_HEADS * GLOBAL_HEAD_DIM;
        for (uint i = tid; i < MAX_CTX * scratch_stride; i += tg_sz) {
            kv_scratch_k[i] = 0;
            kv_scratch_v[i] = 0;
        }
        threadgroup_barrier(mem_flags::mem_device);

        uint blocks_per_head = (h_dim + 255) / 256;
        uint bytes_per_kv_block = KV_NIBBLES_U32 * 4u;  // 52 bytes = 260 elements, 256 used
        for (uint p = 0; p < num_cached; ++p) {
            for (uint h = 0; h < NUM_KV_HEADS; ++h) {
                uint pos_head_vals = slot_kv_offset + layer * MAX_CTX * scratch_stride
                                   + p * scratch_stride + h * GLOBAL_HEAD_DIM;
                for (uint b = 0; b < blocks_per_head; ++b) {
                    uint val_offset = pos_head_vals + b * KV_BLOCK;
                    uint block_idx = val_offset / KV_BLOCK;
                    uint nibble_idx = block_idx * KV_NIBBLES_U32;

                    // Decompress K
                    half scale_k = kv_k_scales[block_idx];
                    for (uint t = (tid & 31u); t < KV_NIBBLES_U32; t += 32) {
                        uint val = kv_k_nibbles[nibble_idx + t];
                        uint dim_start = b * KV_BLOCK + t * PER_LANE;
                        for (uint i = 0; i < PER_LANE; ++i) {
                            uint rem = fast_mod3(val);
                            int wgt = (int)rem - 1;
                            uint dim = dim_start + i;
                            if (dim < h_dim) {
                                uint scratch_pos = p * scratch_stride + h * GLOBAL_HEAD_DIM + dim;
                                kv_scratch_k[scratch_pos] = (half)((float)wgt * (float)scale_k);
                            }
                            val = fast_div3(val);
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_device);

                    // Decompress V
                    half scale_v = kv_v_scales[block_idx];
                    for (uint t = (tid & 31u); t < KV_NIBBLES_U32; t += 32) {
                        uint val = kv_v_nibbles[nibble_idx + t];
                        uint dim_start = b * KV_BLOCK + t * PER_LANE;
                        for (uint i = 0; i < PER_LANE; ++i) {
                            uint rem = fast_mod3(val);
                            int wgt = (int)rem - 1;
                            uint dim = dim_start + i;
                            if (dim < h_dim) {
                                uint scratch_pos = p * scratch_stride + h * GLOBAL_HEAD_DIM + dim;
                                kv_scratch_v[scratch_pos] = (half)((float)wgt * (float)scale_v);
                            }
                            val = fast_div3(val);
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_device);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_device);

        // --- 2-6. KV-group loop: K/V scatter + Q GQA + accum attn ---
        for (uint kv_h = 0; kv_h < NUM_KV_HEADS; ++kv_h) {

            // --- K proj -> q_chunk[0..h_dim], scatter ---
            for (uint o = 0; o < h_dim; o += 32) {
                uint row = o + (tid & 31u);
                if (row < h_dim) {
                    uint flat_row = kv_h * h_dim + row;
                    float dk = tile_gemv(ternary_w, kw_base + flat_row * KV_TILES * LANES,
                                     KV_TILES, tid & 31u, n_buf);
                    dk = warp_sum(dk);
                    if ((tid & 31u) == 0) q_chunk[row] = (half)dk;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            // --- V proj -> q_chunk[h_dim..2*h_dim], scatter ---
            for (uint o = 0; o < h_dim; o += 32) {
                uint row = o + (tid & 31u);
                if (row < h_dim) {
                    float dv = tile_gemv(ternary_w, vw_base + (kv_h * h_dim + row) * KV_TILES * LANES,
                                     KV_TILES, tid & 31u, n_buf);
                    dv = warp_sum(dv);
                    if ((tid & 31u) == 0) q_chunk[h_dim + row] = (half)dv;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            // --- RoPE on K ---
            apply_rope(q_chunk, 1, h_dim, current_pos, tid, tg_sz);

            // --- Scatter K, V to cache ---
            for (uint i = tid; i < h_dim; i += tg_sz) {
                uint scratch_pos = kv_cache_pos * scratch_stride + kv_h * GLOBAL_HEAD_DIM + i;
                kv_scratch_k[scratch_pos] = q_chunk[i];
                kv_scratch_v[scratch_pos] = q_chunk[h_dim + i];
            }
            threadgroup_barrier(mem_flags::mem_device);

            // --- Process 2 Q heads in this KV group ---
            for (uint q_pair = 0; q_pair < 2; ++q_pair) {
                uint qh = 2 * kv_h + q_pair;

                // --- Q proj -> q_chunk[0..h_dim] ---
                for (uint o = 0; o < h_dim; o += 32) {
                    uint row = o + (tid & 31u);
                    if (row < h_dim) {
                        uint flat_row = qh * h_dim + row;
                        float dp = tile_gemv(ternary_w, qw_base + flat_row * Q_TILES * LANES,
                                         Q_TILES, tid & 31u, n_buf);
                        dp = warp_sum(dp);
                        if ((tid & 31u) == 0) q_chunk[row] = (half)dp;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }

                // --- RoPE on Q ---
                apply_rope(q_chunk, 1, h_dim, current_pos, tid, tg_sz);
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // --- GQA attention for this Q head ---
                // Pass 1: Q*K dot product for all cached positions
                float max_val = -1e10;
                for (uint p = tid; p < num_cached; p += tg_sz) {
                    float s = 0.0;
                    for (uint d = 0; d < h_dim; ++d)
                        s += (float)q_chunk[d] * (float)kv_scratch_k[p * scratch_stride + kv_h * GLOBAL_HEAD_DIM + d];
                    slot_logits[p] = (half)s;
                    if (s > max_val) max_val = s;
                }
                // Reduce max across threads
                shared_sums[tid] = max_val;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint stride = tg_sz / 2; stride > 0; stride >>= 1) {
                    if (tid < stride && shared_sums[tid + stride] > shared_sums[tid])
                        shared_sums[tid] = shared_sums[tid + stride];
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                float g_max = shared_sums[0];
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // Pass 2: exp(score - max), accumulate sum
                float sum_exp = 0.0;
                for (uint p = tid; p < num_cached; p += tg_sz) {
                    float e = exp((float)slot_logits[p] - g_max);
                    slot_logits[p] = (half)e;
                    sum_exp += e;
                }
                // Reduce sum_exp
                shared_sums[tid] = sum_exp;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint stride = tg_sz / 2; stride > 0; stride >>= 1) {
                    if (tid < stride) shared_sums[tid] += shared_sums[tid + stride];
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                threadgroup_barrier(mem_flags::mem_device);
                float inv_s = 1.0 / shared_sums[0];

                // Pass 3: weighted sum of V
                for (uint d = tid; d < h_dim; d += tg_sz) {
                    float acc = 0.0;
                    for (uint p = 0; p < num_cached; ++p) {
                        float s = (float)slot_logits[p] * inv_s;
                        acc += s * (float)kv_scratch_v[p * scratch_stride + kv_h * GLOBAL_HEAD_DIM + d];
                    }
                    uint write_pos = qh * h_dim + d;
                    if (write_pos < HIDDEN_DIM)
                        n_buf[write_pos] = (half)acc;
                }
                // Accumulate per-position entropy (-q * log2(q))
                for (uint p = tid; p < num_cached; p += tg_sz) {
                    float q = (float)slot_logits[p] * inv_s;
                    entropy_acc[p] += -q * log2(q + 1e-10);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Pack current position's K/V to ternary ──
        {
            uint blocks_per_head_pack = (h_dim + 255) / 256;
            for (uint h = 0; h < NUM_KV_HEADS; ++h) {
                uint pos_head_vals = slot_kv_offset + layer * MAX_CTX * scratch_stride
                                   + kv_cache_pos * scratch_stride + h * GLOBAL_HEAD_DIM;
                for (uint b = 0; b < blocks_per_head_pack; ++b) {
                    uint block_idx = (pos_head_vals + b * KV_BLOCK) / KV_BLOCK;
                    uint nibble_idx = block_idx * KV_NIBBLES_U32;

                    // ── Pack K block ──
                    float max_mag_k = 0.0;
                    for (uint i = tid; i < KV_BLOCK && (b * KV_BLOCK + i) < h_dim; i += tg_sz) {
                        uint dim = b * KV_BLOCK + i;
                        float v = (float)kv_scratch_k[
                            kv_cache_pos * scratch_stride + h * GLOBAL_HEAD_DIM + dim];
                        if (v < 0) v = -v;
                        if (v > max_mag_k) max_mag_k = v;
                    }
                    shared_sums[tid] = max_mag_k;
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    for (uint stride = tg_sz / 2; stride > 0; stride >>= 1) {
                        if (tid < stride && shared_sums[tid + stride] > shared_sums[tid])
                            shared_sums[tid] = shared_sums[tid + stride];
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                    }
                    float scale_k = shared_sums[0];
                    float rcp_scale_k = (scale_k > 1e-12) ? 1.0 / scale_k : 1.0;
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    if (tid == 0) {
                        kv_k_scales[block_idx] = (half)scale_k;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    for (uint t = (tid & 31u); t < KV_NIBBLES_U32; t += 32) {
                        uint val = 0;
                        uint dim_start = b * KV_BLOCK + t * PER_LANE;
                        for (uint i = 0; i < PER_LANE; ++i) {
                            uint dim = dim_start + i;
                            float f = (dim < h_dim)
                                ? (float)kv_scratch_k[kv_cache_pos * scratch_stride
                                    + h * GLOBAL_HEAD_DIM + dim] * rcp_scale_k
                                : 0.0f;
                            int q = (int)round(f);
                            if (q > 1) q = 1;
                            if (q < -1) q = -1;
                            uint digit = (q == -1) ? 2u : (uint)q;
                            val = val * 3 + digit;
                        }
                        kv_k_nibbles[nibble_idx + t] = val;
                    }
                    threadgroup_barrier(mem_flags::mem_device);

                    // ── Pack V block ──
                    float max_mag_v = 0.0;
                    for (uint i = tid; i < KV_BLOCK && (b * KV_BLOCK + i) < h_dim; i += tg_sz) {
                        uint dim = b * KV_BLOCK + i;
                        float v = (float)kv_scratch_v[
                            kv_cache_pos * scratch_stride + h * GLOBAL_HEAD_DIM + dim];
                        if (v < 0) v = -v;
                        if (v > max_mag_v) max_mag_v = v;
                    }
                    shared_sums[tid] = max_mag_v;
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    for (uint stride = tg_sz / 2; stride > 0; stride >>= 1) {
                        if (tid < stride && shared_sums[tid + stride] > shared_sums[tid])
                            shared_sums[tid] = shared_sums[tid + stride];
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                    }
                    float scale_v = shared_sums[0];
                    float rcp_scale_v = (scale_v > 1e-12) ? 1.0 / scale_v : 1.0;
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    if (tid == 0) {
                        kv_v_scales[block_idx] = (half)scale_v;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    for (uint t = (tid & 31u); t < KV_NIBBLES_U32; t += 32) {
                        uint val = 0;
                        uint dim_start = b * KV_BLOCK + t * PER_LANE;
                        for (uint i = 0; i < PER_LANE; ++i) {
                            uint dim = dim_start + i;
                            float f = (dim < h_dim)
                                ? (float)kv_scratch_v[kv_cache_pos * scratch_stride
                                    + h * GLOBAL_HEAD_DIM + dim] * rcp_scale_v
                                : 0.0f;
                            int q = (int)round(f);
                            if (q > 1) q = 1;
                            if (q < -1) q = -1;
                            uint digit = (q == -1) ? 2u : (uint)q;
                            val = val * 3 + digit;
                        }
                        kv_v_nibbles[nibble_idx + t] = val;
                    }
                    threadgroup_barrier(mem_flags::mem_device);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_device);

        // --- 7. Output projection GEMV + residual ---------------------
        uint ow_stride = HID_TILES * LANES;
        for (uint o = 0; o < HIDDEN_DIM; o += 32) {
            uint row = o + (tid & 31u);
            if (row < HIDDEN_DIM) {
                float dp = tile_gemv(ternary_w, ow_base + row * ow_stride,
                                     HID_TILES, tid & 31u, n_buf);
                dp = warp_sum(dp);
                if ((tid & 31u) == 0) h_buf[row] += (half)dp;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // --- 8. Post-Attention RMSNorm ------------------------------
        for (uint i = tid; i < HIDDEN_DIM; i += tg_sz) n_buf[i] = h_buf[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        fast_rmsnorm(n_buf, in_norm_w, tid, tg_sz, shared_sums);

        // --- 9. MLP Gate/Up projections -----------------------------
        uint gate_base = layer_base + GATE_OFF;
        uint up_base   = layer_base + UP_OFF;
        uint gate_stride = HID_TILES * LANES;
        uint ffn_batch = 32u;
        for (uint o = 0; o < FFN_INTER; o += ffn_batch) {
            for (uint gb = 0; gb < ffn_batch; gb += 32) {
                uint row = o + gb + (tid & 31u);
                if (row < FFN_INTER) {
                    float dp = tile_gemv(ternary_w, gate_base + row * gate_stride,
                                         HID_TILES, tid & 31u, n_buf);
                    dp = warp_sum(dp);
                    if ((tid & 31u) == 0)
                        slot_logits[row] = (half)dp; // gate value
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            for (uint ub = 0; ub < ffn_batch; ub += 32) {
                uint row = o + ub + (tid & 31u);
                if (row < FFN_INTER) {
                    float dp = tile_gemv(ternary_w, up_base + row * gate_stride,
                                         HID_TILES, tid & 31u, n_buf);
                    dp = warp_sum(dp);
                    if ((tid & 31u) == 0)
                        slot_logits[FFN_INTER + row] = (half)dp; // up value
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        // --- 10. SwiGLU activation + Down projection ----------------
        uint down_base = layer_base + DOWN_OFF;
        uint down_stride = FFN_TILES * LANES;
        for (uint o = 0; o < HIDDEN_DIM; o += 32) {
            uint row = o + (tid & 31u);
            float dp_total = 0.0;
            for (uint t = 0; t < FFN_TILES; ++t) {
                uint tile_offset = t * TILE;
                for (uint i = tid; i < TILE; i += tg_sz) {
                    float g = (float)slot_logits[tile_offset + i];
                    float u = (float)slot_logits[FFN_INTER + tile_offset + i];
                    float silu_g = g / (1.0 + exp(-g));
                    n_buf[i] = (half)(silu_g * u);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (row < HIDDEN_DIM) {
                    uint tile_base = down_base + row * down_stride + t * LANES;
                    float dp = tile_gemv(ternary_w, tile_base, 1, tid & 31u, n_buf);
                    dp_total += warp_sum(dp);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (row < HIDDEN_DIM && (tid & 31u) == 0)
                h_buf[row] += (half)dp_total;
        }
    }
    // --- Entropy: normalize by total query heads and write to device buffer ---
    {
        uint total_queries = LAYERS * NUM_Q_HEADS;
        for (uint p = tid; p < num_cached; p += tg_sz) {
            float h = entropy_acc[p] / (float)total_queries;
            entropy_map[p] = (half)h;
        }
        for (uint p = tid; p < MAX_CTX; p += tg_sz) {
            entropy_acc[p] = 0.0;
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Stage: Centroid Scout + Targeted Unpack ------------------------
    // Step A: Compute dot product of h_buf against all 256 centroids.
    for (uint c = tid; c < NUM_CENTROIDS; c += tg_sz) {
        float score = 0.0;
        for (uint d = 0; d < HIDDEN_DIM; ++d) {
            score += (float)h_buf[d] * (float)centroid_scratch[c * HIDDEN_DIM + d];
        }
        centroid_scores[c] = score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step B: Find best cluster --- simple sequential scan by thread 0.
    threadgroup uint best_cluster = 0;
    if (tid == 0) {
        float best_val = -1e10;
        for (uint i = 0; i < NUM_CENTROIDS; ++i) {
            if (centroid_scores[i] > best_val) {
                best_val = centroid_scores[i];
                best_cluster = i;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step C: Find cluster [start, end) positions in the vocabulary.
    if (tid == 0) {
        uint start = VOCAB_SIZE;
        uint end = 0;
        for (uint pos = 0; pos < VOCAB_SIZE; ++pos) {
            if (cluster_map[pos] == best_cluster) {
                if (pos < start) start = pos;
                end = pos + 1;
            }
        }
        cluster_bounds[0] = start;
        cluster_bounds[1] = end;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step D: Compute logits only for tokens in the winning cluster.
    uint cstart = cluster_bounds[0];
    uint cend = cluster_bounds[1];
    for (uint row = cstart; row < cend; ++row) {
        uint simd_lane = tid & 31;
        uint simd_id = tid / 32;
        if ((row - cstart) % (tg_sz / 32) == simd_id) {
            uint tile_base = row * HID_TILES * LANES;
            float acc = 0.0;
            for (uint b = 0; b < HID_TILES; ++b) {
                uint val = embed_clust[tile_base + b * LANES + simd_lane];
                uint act_base = b * TILE + simd_lane * PER_LANE;
                for (uint i = 0; i < PER_LANE; ++i) {
                    uint rem = fast_mod3(val);
                    int wgt = (int)rem - 1;
                    if (wgt != 0) {
                        acc += (float)h_buf[act_base + i] * (float)wgt;
                    }
                    val = fast_div3(val);
                }
            }
            acc = warp_sum(acc);
            if (simd_lane == 0) {
                uint block_idx = row / 256;
                float s = (float)embed_scales[block_idx];
                slot_logits[row] = (half)(acc * s);
            }
        }
    }
    // Fill non-cluster logits with -inf (never selected by argmax).
    for (uint row = tid; row < VOCAB_SIZE; row += tg_sz) {
        if (row < cstart || row >= cend) {
            slot_logits[row] = as_type<half>((unsigned short)0xFC00u);
        }
    }
    threadgroup_barrier(mem_flags::mem_device);

    // ── MTP: Multi-Token Prediction heads ─────────────────────
    // Compute NUM_MTP_HEADS additional logit vectors by running the MTP
    // 2-layer MLP (up_proj + down_proj with residual) then centroid scout
    // on h_buf for each head. Weights are in buffer(26).
    {
        uint mtp_w_base = 0;  // buffer(26) starts at MTP weights
        uint per_head = MTP_HIDDEN * HID_TILES * LANES + HIDDEN_DIM * MTP_TILES * LANES;

        for (uint mtp_head = 0; mtp_head < NUM_MTP_HEADS; ++mtp_head) {
            device half* mtp_scratch = slot_logits + (mtp_head + 1) * VOCAB_SIZE;
            uint head_w_off = mtp_w_base + mtp_head * per_head;

            // Up projection: h_buf -> mtp_hidden (MTP_HIDDEN rows, HID_TILES tiles each)
            // 0. Save h_buf to device scratch (for residual after down-projection)
            for (uint i = tid; i < HIDDEN_DIM; i += tg_sz) mtp_scratch[i] = h_buf[i];
            threadgroup_barrier(mem_flags::mem_device);

            // 1. Up projection: h_buf -> mtp_scratch[HIDDEN_DIM ... HIDDEN_DIM+MTP_HIDDEN]
            device half* up_out = mtp_scratch + HIDDEN_DIM;
            for (uint row = 0; row < MTP_HIDDEN; row += 32) {
                uint r = row + (tid & 31u);
                if (r < MTP_HIDDEN) {
                    float acc = tile_gemv(mtp_ternary_w, head_w_off + r * HID_TILES * LANES,
                                         HID_TILES, tid & 31u, h_buf);
                    acc = warp_sum(acc);
                    if ((tid & 31u) == 0) up_out[r] = (half)acc;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            // 2. Copy up-out to n_buf with zero-padding to tile boundary
            for (uint i = tid; i < MTP_TILES * TILE; i += tg_sz) {
                n_buf[i] = (i < MTP_HIDDEN) ? up_out[i] : (half)0;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // 3. Down projection: n_buf (read) -> h_buf (write, overwriting input)
            uint down_base = head_w_off + MTP_HIDDEN * HID_TILES * LANES;
            for (uint row = 0; row < HIDDEN_DIM; row += 32) {
                uint r = row + (tid & 31u);
                if (r < HIDDEN_DIM) {
                    float acc = tile_gemv(mtp_ternary_w, down_base + r * MTP_TILES * LANES,
                                         MTP_TILES, tid & 31u, n_buf);
                    acc = warp_sum(acc);
                    if ((tid & 31u) == 0) h_buf[r] = (half)acc;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            // 4. Residual: h_buf (MLP output) += original h_buf from device backup
            for (uint i = tid; i < HIDDEN_DIM; i += tg_sz) h_buf[i] += mtp_scratch[i];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // RMS norm on h_buf
            for (uint i = tid; i < HIDDEN_DIM; i += tg_sz) n_buf[i] = h_buf[i];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            fast_rmsnorm(n_buf, norms + 0, tid, tg_sz, shared_sums);
            // Copy normalized result back to h_buf for centroid scout
            for (uint i = tid; i < HIDDEN_DIM; i += tg_sz) h_buf[i] = n_buf[i];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Centroid scout for this MTP head's logits
            device half* head_logits = slot_logits + (mtp_head + 1) * VOCAB_SIZE;

            // Step A: dot products against all centroids
            for (uint c = tid; c < NUM_CENTROIDS; c += tg_sz) {
                float score = 0.0;
                for (uint d = 0; d < HIDDEN_DIM; ++d) {
                    score += (float)h_buf[d] * (float)centroid_scratch[c * HIDDEN_DIM + d];
                }
                centroid_scores[c] = score;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Step B: find best cluster
            threadgroup uint head_best_cluster = 0;
            if (tid == 0) {
                float best_val = -1e10;
                for (uint i = 0; i < NUM_CENTROIDS; ++i) {
                    if (centroid_scores[i] > best_val) {
                        best_val = centroid_scores[i];
                        head_best_cluster = i;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Step C: find cluster bounds
            if (tid == 0) {
                uint start = VOCAB_SIZE;
                uint end = 0;
                for (uint pos = 0; pos < VOCAB_SIZE; ++pos) {
                    if (cluster_map[pos] == head_best_cluster) {
                        if (pos < start) start = pos;
                        end = pos + 1;
                    }
                }
                cluster_bounds[0] = start;
                cluster_bounds[1] = end;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Step D: compute logits for the winning cluster
            uint hcstart = cluster_bounds[0];
            uint hcend = cluster_bounds[1];
            for (uint row = hcstart; row < hcend; ++row) {
                uint simd_lane = tid & 31;
                uint simd_id = tid / 32;
                if ((row - hcstart) % (tg_sz / 32) == simd_id) {
                    uint tile_base = row * HID_TILES * LANES;
                    float acc = 0.0;
                    for (uint b = 0; b < HID_TILES; ++b) {
                        uint val = embed_clust[tile_base + b * LANES + simd_lane];
                        uint act_base = b * TILE + simd_lane * PER_LANE;
                        for (uint i = 0; i < PER_LANE; ++i) {
                            uint rem = fast_mod3(val);
                            int wgt = (int)rem - 1;
                            if (wgt != 0) {
                                acc += (float)h_buf[act_base + i] * (float)wgt;
                            }
                            val = fast_div3(val);
                        }
                    }
                    acc = warp_sum(acc);
                    if (simd_lane == 0) {
                        uint block_idx = row / 256;
                        float s = (float)embed_scales[block_idx];
                        head_logits[row] = (half)(acc * s);
                    }
                }
            }
            // Fill non-cluster logits with -inf
            for (uint row = tid; row < VOCAB_SIZE; row += tg_sz) {
                if (row < hcstart || row >= hcend) {
                    head_logits[row] = as_type<half>((unsigned short)0xFC00u);
                }
            }
            threadgroup_barrier(mem_flags::mem_device);
        }
    }

    }  // end if (kind == 0)
    else if (kind == 3) {
        // ── Draft model forward pass (sub-1ms speculative drafter) ──
        // h_buf already contains the embedded input from Stage 0
        // Reads draft_ternary_w (same ternary nibble format as main model)
        // Processes through DRAFT_LAYERS layers
        // Uses kv_scratch_k/v as FP16 KV cache for draft (small window)
        // No MTP heads, no centroid scout, no entropy accumulation

        uint draft_kv_stride = DRAFT_NUM_KV_HEADS * DRAFT_HEAD_DIM;
        uint draft_cache_pos = 0u;

        for (uint layer = 0; layer < DRAFT_LAYERS; ++layer) {
            uint h_dim = DRAFT_HEAD_DIM;
            uint layer_base = layer * DRAFT_LAYER_STRIDE;  // uses draft nibble weight layout stride
            uint scratch_layer_base = layer * MAX_CTX * draft_kv_stride;

            // --- 1. Input RMSNorm (inlined with DRAFT_HIDDEN dimension) ---
            device const half* in_norm_w = draft_norms + layer * DRAFT_HIDDEN;
            shared_sums[tid] = 0.0;
            for (uint i = tid; i < DRAFT_HIDDEN; i += tg_sz) {
                float v = (float)h_buf[i];
                shared_sums[tid] += v * v;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = tg_sz / 2; stride > 0; stride >>= 1) {
                if (tid < stride) shared_sums[tid] += shared_sums[tid + stride];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float rcp = rsqrt(shared_sums[0] / (float)DRAFT_HIDDEN + 1e-6);
            for (uint i = tid; i < DRAFT_HIDDEN; i += tg_sz) {
                n_buf[i] = (half)((float)h_buf[i] * rcp * (float)in_norm_w[i]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // --- 2. K/V projections via tile_gemv (nibble-based ternary) ---
            uint kw_base = layer_base + DRAFT_K_OFF;
            uint vw_base = layer_base + DRAFT_V_OFF;
            uint qw_base = layer_base + DRAFT_Q_OFF;
            uint ow_base = layer_base + DRAFT_O_OFF;

            // K projection: DRAFT_HIDDEN -> h_dim (per KV head)
            for (uint kv_h = 0; kv_h < DRAFT_NUM_KV_HEADS; ++kv_h) {
                for (uint o = 0; o < h_dim; o += 32) {
                    uint r = o + (tid & 31u);
                    if (r < h_dim) {
                        uint flat_row = kv_h * h_dim + r;
                        float dp = tile_gemv(draft_ternary_w, kw_base + flat_row * DRAFT_KV_TILES * LANES,
                                         DRAFT_KV_TILES, tid & 31u, n_buf);
                        dp = warp_sum(dp);
                        if ((tid & 31u) == 0) n_buf[DRAFT_HIDDEN + r] = (half)dp;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }

            // V projection: DRAFT_HIDDEN -> h_dim (per KV head)
            for (uint kv_h = 0; kv_h < DRAFT_NUM_KV_HEADS; ++kv_h) {
                for (uint o = 0; o < h_dim; o += 32) {
                    uint r = o + (tid & 31u);
                    if (r < h_dim) {
                        uint flat_row = kv_h * h_dim + r;
                        float dp = tile_gemv(draft_ternary_w, vw_base + flat_row * DRAFT_KV_TILES * LANES,
                                         DRAFT_KV_TILES, tid & 31u, n_buf);
                        dp = warp_sum(dp);
                        if ((tid & 31u) == 0) n_buf[DRAFT_HIDDEN + DRAFT_KV_TILES * TILE + r] = (half)dp;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }

            // Store K,V to FP16 device cache (kv_scratch_k/v, one position per layer)
            for (uint kv_h = 0; kv_h < DRAFT_NUM_KV_HEADS; ++kv_h) {
                for (uint i = tid; i < h_dim; i += tg_sz) {
                    kv_scratch_k[scratch_layer_base + draft_cache_pos * draft_kv_stride + kv_h * h_dim + i] =
                        n_buf[DRAFT_HIDDEN + kv_h * h_dim + i];
                    kv_scratch_v[scratch_layer_base + draft_cache_pos * draft_kv_stride + kv_h * h_dim + i] =
                        n_buf[DRAFT_HIDDEN + DRAFT_KV_TILES * TILE + kv_h * h_dim + i];
                }
            }
            threadgroup_barrier(mem_flags::mem_device);

            // --- 3. Full Q projection: DRAFT_HIDDEN -> DRAFT_HIDDEN (all 8 heads) ---
            for (uint qh = 0; qh < DRAFT_NUM_HEADS; ++qh) {
                for (uint o = 0; o < h_dim; o += 32) {
                    uint r = o + (tid & 31u);
                    if (r < h_dim) {
                        uint flat_row = qh * h_dim + r;
                        float dp = tile_gemv(draft_ternary_w, qw_base + flat_row * DRAFT_Q_TILES * LANES,
                                         DRAFT_Q_TILES, tid & 31u, n_buf);
                        dp = warp_sum(dp);
                        if ((tid & 31u) == 0) q_chunk[qh * h_dim + r] = (half)dp;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }

            // --- 4. RoPE on all Q heads ---
            apply_rope(q_chunk, DRAFT_NUM_HEADS, h_dim, draft_cache_pos, tid, tg_sz);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // --- 5. Attention init ---
            for (uint i = tid; i < DRAFT_HIDDEN; i += tg_sz) {
                n_buf[DRAFT_HIDDEN + 2 * h_dim + i] = 0;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // GQA: 8 Q heads, 4 KV heads (2:1). For each KV head, process 2 Q heads.
            for (uint kv_h = 0; kv_h < DRAFT_NUM_KV_HEADS; ++kv_h) {
                for (uint q_pair = 0; q_pair < 2; ++q_pair) {
                    uint qh = 2 * kv_h + q_pair;
                    threadgroup half* q_head = q_chunk + qh * h_dim;

                    // Q*K dot products for all cached positions (0..draft_cache_pos)
                    float max_val = -1e10;
                    for (uint p = tid; p <= draft_cache_pos; p += tg_sz) {
                        float s = 0.0;
                        device half* kv_k_ptr = kv_scratch_k + scratch_layer_base + p * draft_kv_stride + kv_h * h_dim;
                        for (uint d = 0; d < h_dim; ++d)
                            s += (float)q_head[d] * (float)kv_k_ptr[d];
                        slot_logits[p] = (half)s;
                        if (s > max_val) max_val = s;
                    }
                    shared_sums[tid] = max_val;
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    for (uint stride = tg_sz / 2; stride > 0; stride >>= 1) {
                        if (tid < stride && shared_sums[tid + stride] > shared_sums[tid])
                            shared_sums[tid] = shared_sums[tid + stride];
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                    }
                    float g_max = shared_sums[0];
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    // softmax
                    float sum_exp = 0.0;
                    for (uint p = tid; p <= draft_cache_pos; p += tg_sz) {
                        float e = exp((float)slot_logits[p] - g_max);
                        slot_logits[p] = (half)e;
                        sum_exp += e;
                    }
                    shared_sums[tid] = sum_exp;
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    for (uint stride = tg_sz / 2; stride > 0; stride >>= 1) {
                        if (tid < stride) shared_sums[tid] += shared_sums[tid + stride];
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                    }
                    threadgroup_barrier(mem_flags::mem_device);
                    float inv_s = 1.0 / shared_sums[0];

                    // Weighted sum of V
                    for (uint d = tid; d < h_dim; d += tg_sz) {
                        float acc = 0.0;
                        for (uint p = 0; p <= draft_cache_pos; ++p) {
                            float s = (float)slot_logits[p] * inv_s;
                            device half* kv_v_ptr = kv_scratch_v + scratch_layer_base + p * draft_kv_stride + kv_h * h_dim;
                            acc += s * (float)kv_v_ptr[d];
                        }
                        n_buf[DRAFT_HIDDEN + 2 * h_dim + qh * h_dim + d] = (half)acc;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }

            // --- 6. O projection + residual ---
            for (uint i = tid; i < DRAFT_HIDDEN; i += tg_sz) {
                n_buf[i] = n_buf[DRAFT_HIDDEN + 2 * h_dim + i];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // O projection: DRAFT_HIDDEN -> DRAFT_HIDDEN
            for (uint row = 0; row < DRAFT_HIDDEN; row += 32) {
                uint r = row + (tid & 31u);
                if (r < DRAFT_HIDDEN) {
                    float dp = tile_gemv(draft_ternary_w, ow_base + row * DRAFT_HID_TILES * LANES,
                                     DRAFT_HID_TILES, tid & 31u, n_buf);
                    dp = warp_sum(dp);
                    if ((tid & 31u) == 0) h_buf[r] += (half)dp;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            // --- 7. Post-Attention RMSNorm (inlined) ---
            shared_sums[tid] = 0.0;
            for (uint i = tid; i < DRAFT_HIDDEN; i += tg_sz) {
                float v = (float)h_buf[i];
                shared_sums[tid] += v * v;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = tg_sz / 2; stride > 0; stride >>= 1) {
                if (tid < stride) shared_sums[tid] += shared_sums[tid + stride];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float rcp2 = rsqrt(shared_sums[0] / (float)DRAFT_HIDDEN + 1e-6);
            for (uint i = tid; i < DRAFT_HIDDEN; i += tg_sz) {
                n_buf[i] = (half)((float)h_buf[i] * rcp2 * (float)in_norm_w[i]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // --- 8. MLP: Gate projection ---
            uint gate_base = layer_base + DRAFT_GATE_OFF;
            uint up_base   = layer_base + DRAFT_UP_OFF;
            for (uint row = 0; row < DRAFT_FFN_INTER; row += 32) {
                uint r = row + (tid & 31u);
                if (r < DRAFT_FFN_INTER) {
                    float dp = tile_gemv(draft_ternary_w, gate_base + row * DRAFT_HID_TILES * LANES,
                                     DRAFT_HID_TILES, tid & 31u, n_buf);
                    dp = warp_sum(dp);
                    if ((tid & 31u) == 0) slot_logits[r] = (half)dp;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            // --- 9. MLP: Up projection ---
            for (uint row = 0; row < DRAFT_FFN_INTER; row += 32) {
                uint r = row + (tid & 31u);
                if (r < DRAFT_FFN_INTER) {
                    float dp = tile_gemv(draft_ternary_w, up_base + row * DRAFT_HID_TILES * LANES,
                                     DRAFT_HID_TILES, tid & 31u, n_buf);
                    dp = warp_sum(dp);
                    if ((tid & 31u) == 0) slot_logits[DRAFT_FFN_INTER + r] = (half)dp;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            // --- 10. SwiGLU + Down projection ---
            uint down_base = layer_base + DRAFT_DOWN_OFF;
            for (uint row = 0; row < DRAFT_HIDDEN; row += 32) {
                float dp_total = 0.0;
                for (uint t = 0; t < DRAFT_FFN_TILES; ++t) {
                    uint tile_off = t * TILE;
                    uint n_off = t * TILE;
                    for (uint i = tid; i < TILE; i += tg_sz) {
                        float g = (float)slot_logits[tile_off + i];
                        float u = (float)slot_logits[DRAFT_FFN_INTER + tile_off + i];
                        n_buf[n_off + i] = (half)((g / (1.0 + exp(-g))) * u);
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    uint r = row + (tid & 31u);
                    if (r < DRAFT_HIDDEN) {
                        uint tile_base = down_base + row * DRAFT_FFN_TILES * LANES + t * LANES;
                        float dp = tile_gemv(draft_ternary_w, tile_base, 1, tid & 31u, n_buf);
                        dp_total += warp_sum(dp);
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                float result = warp_sum(dp_total);
                if ((tid & 31u) == 0) h_buf[row] += (half)result;
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }  // end for (uint layer = 0; layer < DRAFT_LAYERS; ++layer)

        // ── After all layers: output projection to vocab via centroid scout ──
        // Step A: dot products against all centroids.
        for (uint c = tid; c < NUM_CENTROIDS; c += tg_sz) {
            float score = 0.0;
            for (uint d = 0; d < DRAFT_HIDDEN; ++d) {
                score += (float)h_buf[d] * (float)centroid_scratch[c * HIDDEN_DIM + d];
            }
            centroid_scores[c] = score;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step B: Find best cluster
        threadgroup uint best_cluster = 0;
        if (tid == 0) {
            float best_val = -1e10;
            for (uint i = 0; i < NUM_CENTROIDS; ++i) {
                if (centroid_scores[i] > best_val) {
                    best_val = centroid_scores[i];
                    best_cluster = i;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step C: Find cluster [start, end) positions in the vocabulary.
        if (tid == 0) {
            uint start = VOCAB_SIZE;
            uint end = 0;
            for (uint pos = 0; pos < VOCAB_SIZE; ++pos) {
                if (cluster_map[pos] == best_cluster) {
                    if (pos < start) start = pos;
                    end = pos + 1;
                }
            }
            cluster_bounds[0] = start;
            cluster_bounds[1] = end;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step D: Compute logits for the winning cluster
        uint cstart = cluster_bounds[0];
        uint cend = cluster_bounds[1];
        for (uint row = cstart; row < cend; ++row) {
            uint simd_lane = tid & 31;
            uint simd_id = tid / 32;
            if ((row - cstart) % (tg_sz / 32) == simd_id) {
                uint tile_base = row * HID_TILES * LANES;
                float acc = 0.0;
                for (uint b = 0; b < HID_TILES; ++b) {
                    uint val = embed_clust[tile_base + b * LANES + simd_lane];
                    uint act_base = b * TILE + simd_lane * PER_LANE;
                    for (uint i = 0; i < PER_LANE; ++i) {
                        uint rem = fast_mod3(val);
                        int wgt = (int)rem - 1;
                        if (wgt != 0) {
                            acc += (float)h_buf[act_base + i] * (float)wgt;
                        }
                        val = fast_div3(val);
                    }
                }
                acc = warp_sum(acc);
                if (simd_lane == 0) {
                    uint block_idx = row / 256;
                    float s = (float)embed_scales[block_idx];
                    slot_logits[row] = (half)(acc * s);
                }
            }
        }
        // Fill non-cluster logits with -inf
        for (uint row = tid; row < VOCAB_SIZE; row += tg_sz) {
            if (row < cstart || row >= cend) {
                slot_logits[row] = as_type<half>((unsigned short)0xFC00u);
            }
        }
        threadgroup_barrier(mem_flags::mem_device);

        // ── Top-5 token selection ──
        threadgroup float top5_vals[5];
        threadgroup uint top5_ids[5];
        if (tid == 0) {
            for (uint i = 0; i < 5; ++i) {
                top5_vals[i] = -1e10;
                top5_ids[i] = 0;
            }
            for (uint row = 0; row < VOCAB_SIZE; ++row) {
                float val = (float)slot_logits[row];
                if (val > top5_vals[4]) {
                    uint pos = 4;
                    while (pos > 0 && val > top5_vals[pos - 1]) --pos;
                    for (uint i = 4; i > pos; --i) {
                        top5_vals[i] = top5_vals[i - 1];
                        top5_ids[i] = top5_ids[i - 1];
                    }
                    top5_vals[pos] = val;
                    top5_ids[pos] = row;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Write top-5 candidates to draft_output buffer ──
        if (tid == 0) {
            draft_output[0] = 5;
            for (uint i = 0; i < 5; ++i) {
                draft_output[1 + i] = top5_ids[i];
                half logprob = (half)top5_vals[i];
                draft_output[6 + i] = as_type<uint>(logprob);
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
    }  // end else if (kind == 3)

                // --- After decode: signal COMPLETED -------------------
                threadgroup_barrier(mem_flags::mem_device);
                atomic_store_explicit(
                    (device atomic_uint*)entry, 3 | (kind << 2), memory_order_relaxed);  // COMPLETED
                atomic_fetch_add_explicit(completion_counter, 1, memory_order_relaxed);
                processed = true;
            }
        }

        if (!processed && tid == 0) {
            // Optional hint; on Apple GPUs this is a no-op
        }
    }  // end while(true)
}
"##;

// ====================================================================
//  INT4 Fused Ternary Variant (M5-optimized, 5-per-byte ternary)
// ====================================================================

pub const SHADER_SRC_INT4: &str = r##"#include <metal_stdlib>
using namespace metal;

constant uint HIDDEN_DIM      = 3840;
constant uint LAYERS          = 48;
constant uint NUM_Q_HEADS     = 16;
constant uint NUM_KV_HEADS    = 8;
constant uint HEAD_DIM        = 256;
constant uint GLOBAL_HEAD_DIM = 512;
constant uint FFN_INTER      = 15360;
constant uint VOCAB_SIZE      = 262144;
constant uint MAX_CTX         = 2048;
constant uint MAGIC_DIV3      = 2863311531u;
constant uint O_ROWS          = 4096;
constant uint DOWN_ROWS       = 15360;
constant uint NUM_CENTROIDS   = 256;

constant uint NUM_SINKS = 4;     // first 4 positions are permanent attention sinks (StreamingLLM)
constant uint KV_BLOCK           = 256;
constant uint KV_NIBBLES_U32     = 13;


// -- Work queue constants -------------------------------------------------
constant uint SLOT_WORDS = 4 + VOCAB_SIZE; // 262148
constant uint NUM_SLOTS  = 256;             // concurrent decode slots
constant uint RING_SIZE = 512;

// -- Tile dimensions ------------------------------------------------
constant uint LANES    = 32u;
constant uint PER_LANE = 20u;
constant uint TILE     = 640u;     // 32 × 20 weights per warp-coalesced wave

// Tile count per matrix (ceil(dim / 640))
constant uint Q_TILES     = (NUM_Q_HEADS * HEAD_DIM + TILE - 1) / TILE;  // 7
constant uint KV_TILES    = (NUM_KV_HEADS * HEAD_DIM + TILE - 1) / TILE; // 4
constant uint HID_TILES   = (HIDDEN_DIM + TILE - 1) / TILE;              // 6
constant uint FFN_TILES   = (FFN_INTER + TILE - 1) / TILE;              // 24
constant uint DOWN_TILES  __attribute__((unused)) = (FFN_INTER + TILE - 1) / TILE;              // 24
constant uint VOCAB_TILES __attribute__((unused)) = (VOCAB_SIZE + TILE - 1) / TILE;             // 410
constant uint NUM_MTP_HEADS  = 4;  // number of future-token predictors
constant uint MTP_HIDDEN     = 2048;
constant uint MTP_FFN_INTER  = 8192;
constant uint MTP_TILES      = (MTP_HIDDEN + TILE - 1) / TILE;  // 4
constant uint MTP_TILES_FFN  = (MTP_FFN_INTER + TILE - 1) / TILE; // 13
// ── Draft model architecture (100M params, lightweight speculative drafter) ──
constant uint DRAFT_LAYERS       = 8u;
constant uint DRAFT_HIDDEN       = 768u;
constant uint DRAFT_NUM_HEADS    = 8u;
constant uint DRAFT_NUM_KV_HEADS = 4u;  // GQA ratio 2:1
constant uint DRAFT_HEAD_DIM     = 96u;  // 768 / 8
constant uint DRAFT_FFN_INTER    = 2048u;
constant uint DRAFT_TILES        = (DRAFT_HIDDEN + TILE - 1) / TILE;   // 2
constant uint DRAFT_FFN_TILES    = (DRAFT_FFN_INTER + TILE - 1) / TILE; // 4
constant uint DRAFT_Q_TILES      = (DRAFT_NUM_HEADS * DRAFT_HEAD_DIM + TILE - 1) / TILE;   // 2
constant uint DRAFT_KV_TILES     = (DRAFT_NUM_KV_HEADS * DRAFT_HEAD_DIM + TILE - 1) / TILE; // 1
constant uint DRAFT_HID_TILES    = (DRAFT_HIDDEN + TILE - 1) / TILE;  // 2
// Per-layer nibble offsets for draft model weight layout
constant uint DRAFT_Q_OFF    = 0u;
constant uint DRAFT_K_OFF    = DRAFT_Q_OFF + DRAFT_HIDDEN * DRAFT_Q_TILES * LANES;
constant uint DRAFT_V_OFF    = DRAFT_K_OFF + DRAFT_HIDDEN * DRAFT_KV_TILES * LANES;
constant uint DRAFT_O_OFF    = DRAFT_V_OFF + DRAFT_HIDDEN * DRAFT_KV_TILES * LANES;
constant uint DRAFT_GATE_OFF = DRAFT_O_OFF + DRAFT_HIDDEN * DRAFT_HID_TILES * LANES;
constant uint DRAFT_UP_OFF   = DRAFT_GATE_OFF + DRAFT_HIDDEN * DRAFT_FFN_TILES * LANES;
constant uint DRAFT_DOWN_OFF = DRAFT_UP_OFF + DRAFT_HIDDEN * DRAFT_FFN_TILES * LANES;
constant uint DRAFT_LAYER_STRIDE = DRAFT_DOWN_OFF + DRAFT_FFN_INTER * DRAFT_HID_TILES * LANES;

// Per-layer nibble offsets (in u32 units) for each matrix.
// Computed from row × tile_count × LANES.
constant uint Q_OFF    = 0u;
constant uint K_OFF    = Q_OFF    + HIDDEN_DIM * Q_TILES * LANES;   // 3840×7×32
constant uint V_OFF    = K_OFF    + HIDDEN_DIM * KV_TILES * LANES;  // 3840×4×32
constant uint O_OFF    = V_OFF    + HIDDEN_DIM * KV_TILES * LANES;  // 3840×4×32
constant uint GATE_OFF = O_OFF    + O_ROWS     * HID_TILES * LANES; // 4096×6×32
constant uint UP_OFF   = GATE_OFF + HIDDEN_DIM * FFN_TILES * LANES; // 3840×24×32
constant uint DOWN_OFF = UP_OFF   + HIDDEN_DIM * FFN_TILES * LANES; // 3840×24×32
constant uint LAYER_STRIDE = DOWN_OFF + DOWN_ROWS * HID_TILES * LANES; // 15360×6×32

// Fused interleaved constants (ternary 5-per-byte)
constant uint SUB_TILE_BYTES = 180u;  // 20 TernaryBlock32 × 9 bytes
constant uint FUSED_TILE_STRIDE = 7u * SUB_TILE_BYTES;  // 1260 bytes
constant uint MAX_FUSED_TILES = 24u;
constant uint FUSED_LAYER_STRIDE = MAX_FUSED_TILES * FUSED_TILE_STRIDE;

// Matrix IDs within a fused tile
constant uint MAT_Q = 0u, MAT_K = 1u, MAT_V = 2u, MAT_O = 3u;
constant uint MAT_GATE = 4u, MAT_UP = 5u, MAT_DOWN = 6u;

// ---- Helpers -------------------------------------------------------

inline uint fast_div3(uint v) {
    return ((uint64_t)v * (uint64_t)MAGIC_DIV3) >> 33;
}
inline uint fast_mod3(uint v) {
    return v - fast_div3(v) * 3u;
}

/// GEMV from fused ternary 5-per-byte weights. Each lane reads one element position.
float compute_fused_ternary_gemv(
    device const uchar* fused_w, uint layer, uint matrix_id,
    uint row, uint ntiles,
    threadgroup const half* in_vec, uint lane,
    uint tile_start) {
    float acc = 0.0;
    uint row_offset = row * SUB_TILE_BYTES;
    for (uint b = 0; b < ntiles; ++b) {
        uint tile_offset = layer * FUSED_LAYER_STRIDE + (tile_start + b) * FUSED_TILE_STRIDE + matrix_id * SUB_TILE_BYTES + row_offset;
        // Each sub-tile = 20 TernaryBlock32 × 9 bytes = 180 bytes
        for (uint blk = 0; blk < TILE / 32; ++blk) {
            uint block_off = tile_offset + blk * 9;
            half scale = *(device const half*)(fused_w + block_off + 7);

            // Lane reads the ternary element at position `lane` within this block
            uchar byte_val = fused_w[block_off + lane / 5];
            uint trit;
            if (lane >= 30) {
                // Byte 6 holds 2 trits
                trit = (lane == 30) ? ((uint)byte_val % 3) : ((uint)byte_val / 3);
            } else {
                uint v = (uint)byte_val;
                for (uint i = 0; i < lane % 5; ++i) {
                    v = (v * 86u) >> 8;
                }
                uint q = (v * 86u) >> 8;
                trit = v - q * 3;
            }
            half w_val = (half)((int)trit - 1) * scale;
            uint act_idx = b * TILE + blk * 32 + lane;
            acc += (float)w_val * (float)in_vec[act_idx];
        }
    }
    return acc;
}

/// Warp reduction tree (5 shuffle steps, result on lane 0).
inline float warp_sum(float val) {
    val += simd_shuffle_xor(val, 1);
    val += simd_shuffle_xor(val, 2);
    val += simd_shuffle_xor(val, 4);
    val += simd_shuffle_xor(val, 8);
    val += simd_shuffle_xor(val, 16);
    return val;
}

// ---- RMSNorm -------------------------------------------------------

/// In-place RMSNorm on a 3840-d vector using all tg_size threads.
inline void fast_rmsnorm(threadgroup half* vec,
                         device const half* weight,
                         uint tid, uint tg_size,
                         threadgroup float* sums) {
    sums[tid] = 0.0;
    for (uint i = tid; i < HIDDEN_DIM; i += tg_size) {
        float v = (float)vec[i];
        sums[tid] += v * v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sums[tid] += sums[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rcp = rsqrt(sums[0] / (float)HIDDEN_DIM + 1e-6);
    for (uint i = tid; i < HIDDEN_DIM; i += tg_size) {
        vec[i] = (half)((float)vec[i] * rcp * (float)weight[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// ---- p-RoPE --------------------------------------------------------

inline void apply_rope(threadgroup half* qk, uint num_heads, uint h_dim,
                       uint seq_pos, uint tid, uint tg_size) {
    uint rope_dim = 64u; // partial factor 0.25 of 256
    float theta = 1e6;
    for (uint h = 0; h < num_heads; ++h) {
        uint base = h * h_dim;
        for (uint i = tid; i < rope_dim / 2; i += tg_size) {
            uint c = base + 2 * i;
            float freq = 1.0 / pow(theta, (float)(2 * i) / (float)rope_dim);
            float ang = (float)seq_pos * freq;
            float c0 = cos(ang), s0 = sin(ang);
            float x0 = (float)qk[c], x1 = (float)qk[c + 1];
            qk[c]     = (half)(x0 * c0 - x1 * s0);
            qk[c + 1] = (half)(x0 * s0 + x1 * c0);
        }
    }
}

// ---- SwiGLU --------------------------------------------------------

inline float swiglu(float g, float u) {
    return (g / (1.0 + exp(-g))) * u;
}

// ---- GQA Attention (inner loop over KV heads) ----------------------

/// Process one KV head group (2 query heads, 1 KV head).
/// Reads/writes q_chunk[k][d] via lane-level indexing.
/// Loads K_cache/V_cache from device memory for all past positions.
/// num_cached = number of valid KV cache positions (may be capped at MAX_CTX with eviction).
void gqa_group(device const half* kv_k, device const half* kv_v,
               threadgroup const half* q_buf, uint kv_h,
               uint num_cached, uint active_h_dim,
               uint tid, uint tg_size,
               threadgroup float* scores,   // [2 × MAX_CTX] float scratch
               threadgroup half* out_accum) // [2 × active_h_dim] output
{

    uint N = NUM_KV_HEADS * active_h_dim; // per-position KV stride

    // First pass: compute scores and global max
    float global_max = -1e10;
    for (uint p = tid; p < num_cached; p += tg_size) {
        uint kv_base = p * N + kv_h * active_h_dim;
        for (uint qh = 0; qh < 2; ++qh) {
            float s = 0.0;
            uint q_base = qh * active_h_dim;
            for (uint d = 0; d < active_h_dim; ++d) {
                s += (float)q_buf[q_base + d] * (float)kv_k[kv_base + d];
            }
            scores[qh * MAX_CTX + p] = s;
            if (s > global_max) global_max = s;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Second pass: exp(score - max) → per-head sum
    float sum_exp_q0 = 0.0;
    float sum_exp_q1 = 0.0;
    for (uint p = tid; p < num_cached; p += tg_size) {
        float e0 = exp(scores[0 * MAX_CTX + p] - global_max);
        float e1 = exp(scores[1 * MAX_CTX + p] - global_max);
        scores[0 * MAX_CTX + p] = e0;
        scores[1 * MAX_CTX + p] = e1;
        sum_exp_q0 += e0;
        sum_exp_q1 += e1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_sum_q0 = 1.0 / sum_exp_q0;
    float inv_sum_q1 = 1.0 / sum_exp_q1;

    // Third pass: weighted sum of V (per-head softmax)
    for (uint d = tid; d < active_h_dim; d += tg_size) {
        float v0 = 0.0, v1 = 0.0;
        for (uint p = 0; p < num_cached; ++p) {
            float s0 = scores[0 * MAX_CTX + p] * inv_sum_q0;
            float s1 = scores[1 * MAX_CTX + p] * inv_sum_q1;
            uint kv_base = p * N + kv_h * active_h_dim;
            float vv = (float)kv_v[kv_base + d];
            v0 += s0 * vv;
            v1 += s1 * vv;
        }
        out_accum[0 * active_h_dim + d] = (half)v0;
        out_accum[1 * active_h_dim + d] = (half)v1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}


// ---- Main Kernel ---------------------------------------------------

kernel void gemma4_full_decode_persistent(
    device const uchar*   int4_fused_w  [[buffer(0)]],
    device const half*    scales        [[buffer(1)]],
    device const half*    norms         [[buffer(2)]],  // aux: first part is norms
    device const uint*    embed_clust   [[buffer(3)]],  // ternary nibbles (reordered by cluster)
    device const uint*    centroids_ternary [[buffer(4)]],  // ternary-packed centroids (256 x 192 u32)
    device const uint*    cluster_map   [[buffer(5)]],  // VOCAB_SIZE x u32 cluster assignments
    device uint*          kv_k_nibbles  [[buffer(6)]],  // ternary-packed K nibbles
    device uint*          kv_v_nibbles  [[buffer(7)]],  // ternary-packed V nibbles
    device half*          kv_k_scales   [[buffer(8)]],  // FP16 block scales for K
    device half*          kv_v_scales   [[buffer(9)]],  // FP16 block scales for V
    device const half*    embed_scales  [[buffer(14)]],  // FP16 block scales for embed
    device const half*    centroid_scales   [[buffer(15)]], // FP16 block scales for centroids (256 x 15)
    device half*          centroid_scratch  [[buffer(16)]], // decompressed FP16 centroids (256 x HIDDEN_DIM)
    device atomic_uint*   centroid_decompress_progress [[buffer(17)]], // atomic progress counter
    device half*          kv_scratch_k  [[buffer(19)]], // decompressed K scratch (1 layer)
    device half*          kv_scratch_v  [[buffer(20)]], // decompressed V scratch (1 layer)
    device half*          entropy_map   [[buffer(21)]],
    device uint*          ring_entries  [[buffer(22)]],  // submission ring entries (WorkEntry[4 x RING_SIZE])
    device atomic_uint*   ring_tail     [[buffer(23)]],  // GPU-claimed tail offset
    device half*          slot_logits_base [[buffer(24)]], // per-slot logits (NUM_SLOTS x VOCAB_SIZE half)
    device atomic_uint*   completion_counter [[buffer(25)]], // incremented after COMPLETED
    device const uint*    mtp_ternary_w     [[buffer(26)]], // MTP head ternary weights
    device const uint*    draft_ternary_w   [[buffer(10)]],  // draft model ternary nibble weights
    device const half*    draft_scales      [[buffer(11)]],  // draft model block scales
    device const half*    draft_norms       [[buffer(12)]],  // draft model RMSNorm weights
    device uint*          draft_output      [[buffer(28)]],  // draft output: [N, tok_id0..4, logprob0..4]
    uint tid    [[thread_index_in_threadgroup]],
    uint tg_sz  [[threads_per_threadgroup]])
{
    // --- SRAM ----------------------------------------------------------
    // Budget: 7.5 + 7.5 + 2 + 1 + 1 + 0.008 = ~19 KB < 32 KB

    threadgroup half h_buf[HIDDEN_DIM];      // 7.5 KB --- residual stream
    threadgroup half n_buf[HIDDEN_DIM];      // 7.5 KB --- norm scratch
    threadgroup half q_chunk[1024];           // 2 KB  --- 2 Q-heads x 512 max
    threadgroup float shared_sums[256];       // 1 KB  --- tree reduction
    threadgroup float centroid_scores[256];   // 1 KB  --- centroid scout
    threadgroup uint cluster_bounds[2];       // 8 B   --- [cluster_start, cluster_end)
    threadgroup float entropy_acc[MAX_CTX];   // 8 KB  --- per-position entropy accumulator

    while (true) {
        // --- Idle work: centroid decompression --------------------------
        uint decomp_progress = atomic_load_explicit(
            centroid_decompress_progress, memory_order_relaxed);
        if (decomp_progress < NUM_CENTROIDS) {
            uint idx = NUM_CENTROIDS;
            if ((tid & 31) == 0) {
                idx = atomic_fetch_add_explicit(
                    centroid_decompress_progress, 1, memory_order_relaxed);
            }
            idx = simd_broadcast(idx, 0);
            if (idx < NUM_CENTROIDS) {
                device const uint* src = centroids_ternary + idx * HID_TILES * LANES;
                device half* dst = centroid_scratch + idx * HIDDEN_DIM;
                uint lane = tid & 31;
                for (uint b = 0; b < HID_TILES; ++b) {
                    uint val = src[b * LANES + lane];
                    uint act_base = b * TILE + lane * PER_LANE;
                    for (uint i = 0; i < PER_LANE; ++i) {
                        uint rem = fast_mod3(val);
                        int wgt = (int)rem - 1;
                        uint flat_idx = act_base + i;
                        uint block_idx = flat_idx / 256;
                        half s = centroid_scales[
                            idx * ((HIDDEN_DIM + 255) / 256) + block_idx];
                        dst[act_base + i] = (half)((float)wgt * (float)s);
                        val = fast_div3(val);
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_device);

        // --- Atomic ring dequeue from submission queue ------------------
        bool processed = false;
        uint my_tail = atomic_fetch_add_explicit(
            ring_tail, 1, memory_order_relaxed);
        uint idx = my_tail % RING_SIZE;
        device uint* entry = ring_entries + idx * 4;
        uint entry_state = atomic_load_explicit(
            (device atomic_uint*)entry, memory_order_relaxed);
        uint kind = entry_state >> 2;
        if ((entry_state & 3) == 1) {  // SUBMITTED (low 2 bits = state, upper = kind)
            uint expected = entry_state;
            if (atomic_compare_exchange_weak_explicit(
                (device atomic_uint*)entry, &expected, 2 | (kind << 2),  // CLAIMED
                memory_order_relaxed, memory_order_relaxed)) {
                uint current_token = entry[1];
                uint current_pos   = entry[2];
                uint kv_slot_id    = entry[3];

                // Number of valid KV cache positions
                uint num_cached = min(current_pos + 1, MAX_CTX);

                // KV cache offset for this partition
                uint slot_kv_offset = kv_slot_id * MAX_CTX * NUM_KV_HEADS * GLOBAL_HEAD_DIM * LAYERS;

                // Logits output goes to the slot's logits region
                device half* slot_logits = slot_logits_base + kv_slot_id * VOCAB_SIZE;

    // --- Stage 0: Embedding lookup from embed_clust via cluster_map ----------
    if (tid == 0) {
        uint c = cluster_map[current_token];
        uint cluster_start = 0;
        for (uint pos = 0; pos < VOCAB_SIZE; ++pos) {
            if (cluster_map[pos] < c) ++cluster_start;
        }
        uint rank = 0;
        for (uint pos = 0; pos < current_token; ++pos) {
            if (cluster_map[pos] == c) ++rank;
        }
        uint embed_row = cluster_start + rank;
        cluster_bounds[0] = embed_row;
    }

    // Tile-GEMV: decode ternary embed row into h_buf
    uint simd_lane = tid & 31;
    uint simd_id = tid / 32;
    uint sel_row = cluster_bounds[0];
    uint tile_base = sel_row * HID_TILES * LANES;
    for (uint b = simd_id; b < HID_TILES; b += tg_sz / 32) {
        uint val = embed_clust[tile_base + b * LANES + simd_lane];
        uint act_base = b * TILE + simd_lane * PER_LANE;
        for (uint i = 0; i < PER_LANE; ++i) {
            uint rem = fast_mod3(val);
            int wgt = (int)rem - 1;
            uint flat_idx = b * TILE + simd_lane * PER_LANE + i;
            uint block_idx = flat_idx / 256;
            float s = (float)embed_scales[block_idx];
            h_buf[act_base + i] = (half)((float)wgt * s);
            val = fast_div3(val);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (kind == 0) {
    // --- 48-layer loop -------------------------------------------------
    for (uint layer = 0; layer < LAYERS; ++layer) {
        bool shared = ((layer + 1) % 6 == 0);
        uint h_dim = shared ? GLOBAL_HEAD_DIM : HEAD_DIM;
        uint layer_base = layer * LAYER_STRIDE;

        // --- 1. Input RMSNorm ------------------------------------------
        device const half* in_norm_w = norms + layer * HIDDEN_DIM;
        for (uint i = tid; i < HIDDEN_DIM; i += tg_sz) n_buf[i] = h_buf[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        fast_rmsnorm(n_buf, in_norm_w, tid, tg_sz, shared_sums);

        // --- 2. Q projection -------------------------------------------
        uint qw_base = layer_base + Q_OFF;
        uint kw_base = layer_base + K_OFF;
        uint vw_base = layer_base + V_OFF;
        uint ow_base = layer_base + O_OFF;

        // Init attention-output accumulator in n_buf
        for (uint i = tid; i < HIDDEN_DIM; i += tg_sz) n_buf[i] = 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute cache position with StreamingLLM + cyclic FIFO eviction
        uint kv_cache_pos = current_pos;
        if (kv_cache_pos >= MAX_CTX) {
            kv_cache_pos = NUM_SINKS + (kv_cache_pos - NUM_SINKS) % (MAX_CTX - NUM_SINKS);
        }

        // ── Decompress K/V for this layer from ternary ──
        uint scratch_stride = NUM_KV_HEADS * GLOBAL_HEAD_DIM;
        for (uint i = tid; i < MAX_CTX * scratch_stride; i += tg_sz) {
            kv_scratch_k[i] = 0;
            kv_scratch_v[i] = 0;
        }
        threadgroup_barrier(mem_flags::mem_device);

        uint blocks_per_head = (h_dim + 255) / 256;
        uint bytes_per_kv_block = KV_NIBBLES_U32 * 4u;  // 52 bytes = 260 elements, 256 used
        for (uint p = 0; p < num_cached; ++p) {
            for (uint h = 0; h < NUM_KV_HEADS; ++h) {
                uint pos_head_vals = slot_kv_offset + layer * MAX_CTX * scratch_stride
                                   + p * scratch_stride + h * GLOBAL_HEAD_DIM;
                for (uint b = 0; b < blocks_per_head; ++b) {
                    uint val_offset = pos_head_vals + b * KV_BLOCK;
                    uint block_idx = val_offset / KV_BLOCK;
                    uint byte_offset = block_idx * bytes_per_kv_block;  // byte offset into packed cache

                    // ── Decompress K (5-per-byte ternary unpack) ──
                    half scale_k = kv_k_scales[block_idx];
                    for (uint t = (tid & 31u); t < bytes_per_kv_block; t += 32) {
                        uchar packed = ((device uchar*)kv_k_nibbles)[byte_offset + t];
                        uint dim_start = b * KV_BLOCK + t * 5u;
                        uint v = (uint)packed;
                        for (uint i = 0; i < 5; ++i) {
                            uint rem = v % 3u;
                            int wgt = (int)rem - 1;
                            uint dim = dim_start + i;
                            if (dim < h_dim) {
                                uint scratch_pos = p * scratch_stride + h * GLOBAL_HEAD_DIM + dim;
                                kv_scratch_k[scratch_pos] = (half)((float)wgt * (float)scale_k);
                            }
                            v /= 3u;
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_device);

                    // ── Decompress V (5-per-byte ternary unpack) ──
                    half scale_v = kv_v_scales[block_idx];
                    for (uint t = (tid & 31u); t < bytes_per_kv_block; t += 32) {
                        uchar packed = ((device uchar*)kv_v_nibbles)[byte_offset + t];
                        uint dim_start = b * KV_BLOCK + t * 5u;
                        uint v = (uint)packed;
                        for (uint i = 0; i < 5; ++i) {
                            uint rem = v % 3u;
                            int wgt = (int)rem - 1;
                            uint dim = dim_start + i;
                            if (dim < h_dim) {
                                uint scratch_pos = p * scratch_stride + h * GLOBAL_HEAD_DIM + dim;
                                kv_scratch_v[scratch_pos] = (half)((float)wgt * (float)scale_v);
                            }
                            v /= 3u;
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_device);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_device);

        // --- 2-6. KV-group loop: K/V scatter + Q GQA + accum attn ---
        for (uint kv_h = 0; kv_h < NUM_KV_HEADS; ++kv_h) {

            // --- K proj -> q_chunk[0..h_dim], scatter ---
            for (uint o = 0; o < h_dim; o += 32) {
                uint row = o + (tid & 31u);
                if (row < h_dim) {
                    uint flat_row = kv_h * h_dim + row;
                    float dk = compute_fused_ternary_gemv(int4_fused_w, layer, MAT_K, kv_h * h_dim + row, KV_TILES, n_buf, tid & 31u, 0u);
                    dk = warp_sum(dk);
                    if ((tid & 31u) == 0) q_chunk[row] = (half)dk;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            // --- V proj -> q_chunk[h_dim..2*h_dim], scatter ---
            for (uint o = 0; o < h_dim; o += 32) {
                uint row = o + (tid & 31u);
                if (row < h_dim) {
                    float dv = compute_fused_ternary_gemv(int4_fused_w, layer, MAT_V, kv_h * h_dim + row, KV_TILES, n_buf, tid & 31u, 0u);
                    dv = warp_sum(dv);
                    if ((tid & 31u) == 0) q_chunk[h_dim + row] = (half)dv;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            // --- RoPE on K ---
            apply_rope(q_chunk, 1, h_dim, current_pos, tid, tg_sz);

            // --- Scatter K, V to cache ---
            for (uint i = tid; i < h_dim; i += tg_sz) {
                uint scratch_pos = kv_cache_pos * scratch_stride + kv_h * GLOBAL_HEAD_DIM + i;
                kv_scratch_k[scratch_pos] = q_chunk[i];
                kv_scratch_v[scratch_pos] = q_chunk[h_dim + i];
            }
            threadgroup_barrier(mem_flags::mem_device);

            // --- Process 2 Q heads in this KV group ---
            for (uint q_pair = 0; q_pair < 2; ++q_pair) {
                uint qh = 2 * kv_h + q_pair;

                // --- Q proj -> q_chunk[0..h_dim] ---
                for (uint o = 0; o < h_dim; o += 32) {
                    uint row = o + (tid & 31u);
                    if (row < h_dim) {
                        uint flat_row = qh * h_dim + row;
                        float dp = compute_fused_ternary_gemv(int4_fused_w, layer, MAT_Q, flat_row, Q_TILES, n_buf, tid & 31u, 0u);
                        dp = warp_sum(dp);
                        if ((tid & 31u) == 0) q_chunk[row] = (half)dp;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }

                // --- RoPE on Q ---
                apply_rope(q_chunk, 1, h_dim, current_pos, tid, tg_sz);
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // --- GQA attention for this Q head ---
                // Pass 1: Q*K dot product for all cached positions
                float max_val = -1e10;
                for (uint p = tid; p < num_cached; p += tg_sz) {
                    float s = 0.0;
                    for (uint d = 0; d < h_dim; ++d)
                        s += (float)q_chunk[d] * (float)kv_scratch_k[p * scratch_stride + kv_h * GLOBAL_HEAD_DIM + d];
                    slot_logits[p] = (half)s;
                    if (s > max_val) max_val = s;
                }
                // Reduce max across threads
                shared_sums[tid] = max_val;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint stride = tg_sz / 2; stride > 0; stride >>= 1) {
                    if (tid < stride && shared_sums[tid + stride] > shared_sums[tid])
                        shared_sums[tid] = shared_sums[tid + stride];
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                float g_max = shared_sums[0];
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // Pass 2: exp(score - max), accumulate sum
                float sum_exp = 0.0;
                for (uint p = tid; p < num_cached; p += tg_sz) {
                    float e = exp((float)slot_logits[p] - g_max);
                    slot_logits[p] = (half)e;
                    sum_exp += e;
                }
                // Reduce sum_exp
                shared_sums[tid] = sum_exp;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint stride = tg_sz / 2; stride > 0; stride >>= 1) {
                    if (tid < stride) shared_sums[tid] += shared_sums[tid + stride];
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                threadgroup_barrier(mem_flags::mem_device);
                float inv_s = 1.0 / shared_sums[0];

                // Pass 3: weighted sum of V
                for (uint d = tid; d < h_dim; d += tg_sz) {
                    float acc = 0.0;
                    for (uint p = 0; p < num_cached; ++p) {
                        float s = (float)slot_logits[p] * inv_s;
                        acc += s * (float)kv_scratch_v[p * scratch_stride + kv_h * GLOBAL_HEAD_DIM + d];
                    }
                    uint write_pos = qh * h_dim + d;
                    if (write_pos < HIDDEN_DIM)
                        n_buf[write_pos] = (half)acc;
                }
                // Accumulate per-position entropy (-q * log2(q))
                for (uint p = tid; p < num_cached; p += tg_sz) {
                    float q = (float)slot_logits[p] * inv_s;
                    entropy_acc[p] += -q * log2(q + 1e-10);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Pack current position's K/V to ternary (5-per-byte) ──
        {
            uint blocks_per_head_pack = (h_dim + 255) / 256;
            uint bytes_per_kv_block = KV_NIBBLES_U32 * 4u;  // 52 bytes = 256 elements
            for (uint h = 0; h < NUM_KV_HEADS; ++h) {
                uint pos_head_vals = slot_kv_offset + layer * MAX_CTX * scratch_stride
                                   + kv_cache_pos * scratch_stride + h * GLOBAL_HEAD_DIM;
                for (uint b = 0; b < blocks_per_head_pack; ++b) {
                    uint block_idx = (pos_head_vals + b * KV_BLOCK) / KV_BLOCK;
                    uint byte_offset = block_idx * bytes_per_kv_block;

                    // ── Pack K block (5-per-byte ternary) ──
                    float max_mag_k = 0.0;
                    for (uint i = tid; i < KV_BLOCK && (b * KV_BLOCK + i) < h_dim; i += tg_sz) {
                        uint dim = b * KV_BLOCK + i;
                        float v = (float)kv_scratch_k[
                            kv_cache_pos * scratch_stride + h * GLOBAL_HEAD_DIM + dim];
                        if (v < 0) v = -v;
                        if (v > max_mag_k) max_mag_k = v;
                    }
                    shared_sums[tid] = max_mag_k;
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    for (uint stride = tg_sz / 2; stride > 0; stride >>= 1) {
                        if (tid < stride && shared_sums[tid + stride] > shared_sums[tid])
                            shared_sums[tid] = shared_sums[tid + stride];
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                    }
                    float scale_k = shared_sums[0];
                    float rcp_scale_k = (scale_k > 1e-12) ? 1.0 / scale_k : 1.0;
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    if (tid == 0) {
                        kv_k_scales[block_idx] = (half)scale_k;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    for (uint t = (tid & 31u); t < bytes_per_kv_block; t += 32) {
                        uint byte_val = 0;
                        uint dim_start = b * KV_BLOCK + t * 5u;
                        for (uint i = 0; i < 5; ++i) {
                            uint dim = dim_start + i;
                            float f = (dim < h_dim)
                                ? (float)kv_scratch_k[kv_cache_pos * scratch_stride
                                    + h * GLOBAL_HEAD_DIM + dim] * rcp_scale_k
                                : 0.0f;
                            int q = (int)round(f);
                            if (q > 1) q = 1;
                            if (q < -1) q = -1;
                            uint digit = (q == -1) ? 2u : (uint)q;
                            byte_val = byte_val * 3u + digit;
                        }
                        ((device uchar*)kv_k_nibbles)[byte_offset + t] = (uchar)byte_val;
                    }
                    threadgroup_barrier(mem_flags::mem_device);

                    // ── Pack V block (5-per-byte ternary) ──
                    float max_mag_v = 0.0;
                    for (uint i = tid; i < KV_BLOCK && (b * KV_BLOCK + i) < h_dim; i += tg_sz) {
                        uint dim = b * KV_BLOCK + i;
                        float v = (float)kv_scratch_v[
                            kv_cache_pos * scratch_stride + h * GLOBAL_HEAD_DIM + dim];
                        if (v < 0) v = -v;
                        if (v > max_mag_v) max_mag_v = v;
                    }
                    shared_sums[tid] = max_mag_v;
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    for (uint stride = tg_sz / 2; stride > 0; stride >>= 1) {
                        if (tid < stride && shared_sums[tid + stride] > shared_sums[tid])
                            shared_sums[tid] = shared_sums[tid + stride];
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                    }
                    float scale_v = shared_sums[0];
                    float rcp_scale_v = (scale_v > 1e-12) ? 1.0 / scale_v : 1.0;
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    if (tid == 0) {
                        kv_v_scales[block_idx] = (half)scale_v;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    for (uint t = (tid & 31u); t < bytes_per_kv_block; t += 32) {
                        uint byte_val = 0;
                        uint dim_start = b * KV_BLOCK + t * 5u;
                        for (uint i = 0; i < 5; ++i) {
                            uint dim = dim_start + i;
                            float f = (dim < h_dim)
                                ? (float)kv_scratch_v[kv_cache_pos * scratch_stride
                                    + h * GLOBAL_HEAD_DIM + dim] * rcp_scale_v
                                : 0.0f;
                            int q = (int)round(f);
                            if (q > 1) q = 1;
                            if (q < -1) q = -1;
                            uint digit = (q == -1) ? 2u : (uint)q;
                            byte_val = byte_val * 3u + digit;
                        }
                        ((device uchar*)kv_v_nibbles)[byte_offset + t] = (uchar)byte_val;
                    }
                    threadgroup_barrier(mem_flags::mem_device);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_device);

        // --- 7. Output projection GEMV + residual ---------------------
        for (uint o = 0; o < HIDDEN_DIM; o += 32) {
            uint row = o + (tid & 31u);
            if (row < HIDDEN_DIM) {
                float dp = compute_fused_ternary_gemv(int4_fused_w, layer, MAT_O, row, HID_TILES, n_buf, tid & 31u, 0u);

                dp = warp_sum(dp);
                if ((tid & 31u) == 0) h_buf[row] += (half)dp;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // --- 8. Post-Attention RMSNorm ------------------------------
        for (uint i = tid; i < HIDDEN_DIM; i += tg_sz) n_buf[i] = h_buf[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        fast_rmsnorm(n_buf, in_norm_w, tid, tg_sz, shared_sums);

        // --- 9. MLP Gate/Up projections -----------------------------
        for (uint o = 0; o < FFN_INTER; o += 32) {
            uint row = o + (tid & 31u);
            if (row < FFN_INTER) {
                float dp = compute_fused_ternary_gemv(int4_fused_w, layer, MAT_GATE, row, HID_TILES, n_buf, tid & 31u, 0u);

                dp = warp_sum(dp);
                if ((tid & 31u) == 0)
                    slot_logits[row] = (half)dp; // gate value
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint o = 0; o < FFN_INTER; o += 32) {
            uint row = o + (tid & 31u);
            if (row < FFN_INTER) {
                float dp = compute_fused_ternary_gemv(int4_fused_w, layer, MAT_UP, row, HID_TILES, n_buf, tid & 31u, 0u);

                dp = warp_sum(dp);
                if ((tid & 31u) == 0)
                    slot_logits[FFN_INTER + row] = (half)dp; // up value
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // --- 10. SwiGLU activation + Down projection ----------------
        for (uint o = 0; o < HIDDEN_DIM; o += 32) {
            uint row = o + (tid & 31u);
            float dp_total = 0.0;
            for (uint t = 0; t < FFN_TILES; ++t) {
                uint tile_offset = t * TILE;
                for (uint i = tid; i < TILE; i += tg_sz) {
                    float g = (float)slot_logits[tile_offset + i];
                    float u = (float)slot_logits[FFN_INTER + tile_offset + i];
                    float silu_g = g / (1.0 + exp(-g));
                    n_buf[i] = (half)(silu_g * u);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (row < HIDDEN_DIM) {
                    float dp = compute_fused_ternary_gemv(int4_fused_w, layer, MAT_DOWN, row, 1, n_buf, tid & 31u, t);
                    dp_total += warp_sum(dp);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (row < HIDDEN_DIM && (tid & 31u) == 0)
                h_buf[row] += (half)dp_total;
        }
    }
    // --- Entropy: normalize by total query heads and write to device buffer ---
    {
        uint total_queries = LAYERS * NUM_Q_HEADS;
        for (uint p = tid; p < num_cached; p += tg_sz) {
            float h = entropy_acc[p] / (float)total_queries;
            entropy_map[p] = (half)h;
        }
        for (uint p = tid; p < MAX_CTX; p += tg_sz) {
            entropy_acc[p] = 0.0;
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Stage: Centroid Scout + Targeted Unpack ------------------------
    // Step A: Compute dot product of h_buf against all 256 centroids.
    for (uint c = tid; c < NUM_CENTROIDS; c += tg_sz) {
        float score = 0.0;
        for (uint d = 0; d < HIDDEN_DIM; ++d) {
            score += (float)h_buf[d] * (float)centroid_scratch[c * HIDDEN_DIM + d];
        }
        centroid_scores[c] = score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step B: Find best cluster --- simple sequential scan by thread 0.
    threadgroup uint best_cluster = 0;
    if (tid == 0) {
        float best_val = -1e10;
        for (uint i = 0; i < NUM_CENTROIDS; ++i) {
            if (centroid_scores[i] > best_val) {
                best_val = centroid_scores[i];
                best_cluster = i;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step C: Find cluster [start, end) positions in the vocabulary.
    if (tid == 0) {
        uint start = VOCAB_SIZE;
        uint end = 0;
        for (uint pos = 0; pos < VOCAB_SIZE; ++pos) {
            if (cluster_map[pos] == best_cluster) {
                if (pos < start) start = pos;
                end = pos + 1;
            }
        }
        cluster_bounds[0] = start;
        cluster_bounds[1] = end;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step D: Compute logits only for tokens in the winning cluster.
    uint cstart = cluster_bounds[0];
    uint cend = cluster_bounds[1];
    for (uint row = cstart; row < cend; ++row) {
        uint simd_lane = tid & 31;
        uint simd_id = tid / 32;
        if ((row - cstart) % (tg_sz / 32) == simd_id) {
            uint tile_base = row * HID_TILES * LANES;
            float acc = 0.0;
            for (uint b = 0; b < HID_TILES; ++b) {
                uint val = embed_clust[tile_base + b * LANES + simd_lane];
                uint act_base = b * TILE + simd_lane * PER_LANE;
                for (uint i = 0; i < PER_LANE; ++i) {
                    uint rem = fast_mod3(val);
                    int wgt = (int)rem - 1;
                    if (wgt != 0) {
                        acc += (float)h_buf[act_base + i] * (float)wgt;
                    }
                    val = fast_div3(val);
                }
            }
            acc = warp_sum(acc);
            if (simd_lane == 0) {
                uint block_idx = row / 256;
                float s = (float)embed_scales[block_idx];
                slot_logits[row] = (half)(acc * s);
            }
        }
    }
    // Fill non-cluster logits with -inf (never selected by argmax).
    for (uint row = tid; row < VOCAB_SIZE; row += tg_sz) {
        if (row < cstart || row >= cend) {
            slot_logits[row] = as_type<half>((unsigned short)0xFC00u);
        }
    }
    threadgroup_barrier(mem_flags::mem_device);

    // ── MTP: Multi-Token Prediction heads ─────────────────────
    // Compute NUM_MTP_HEADS additional logit vectors by running the MTP
    // 2-layer MLP (up_proj + down_proj with residual) then centroid scout
    // on h_buf for each head. Weights are in buffer(26).
    {
        uint mtp_w_base = 0;  // buffer(26) starts at MTP weights
        uint per_head = MTP_HIDDEN * HID_TILES * LANES + HIDDEN_DIM * MTP_TILES * LANES;

        for (uint mtp_head = 0; mtp_head < NUM_MTP_HEADS; ++mtp_head) {
            device half* mtp_scratch = slot_logits + (mtp_head + 1) * VOCAB_SIZE;
            uint head_w_off = mtp_w_base + mtp_head * per_head;

            // Up projection: h_buf -> mtp_hidden (MTP_HIDDEN rows, HID_TILES tiles each)
            // 0. Save h_buf to device scratch (for residual after down-projection)
            for (uint i = tid; i < HIDDEN_DIM; i += tg_sz) mtp_scratch[i] = h_buf[i];
            threadgroup_barrier(mem_flags::mem_device);

            // 1. Up projection: h_buf -> mtp_scratch[HIDDEN_DIM ... HIDDEN_DIM+MTP_HIDDEN]
            device half* up_out = mtp_scratch + HIDDEN_DIM;
            for (uint row = 0; row < MTP_HIDDEN; row += 32) {
                uint r = row + (tid & 31u);
                if (r < MTP_HIDDEN) {
                    float acc = tile_gemv(mtp_ternary_w, head_w_off + r * HID_TILES * LANES,
                                         HID_TILES, tid & 31u, h_buf);
                    acc = warp_sum(acc);
                    if ((tid & 31u) == 0) up_out[r] = (half)acc;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            // 2. Copy up-out to n_buf with zero-padding to tile boundary
            for (uint i = tid; i < MTP_TILES * TILE; i += tg_sz) {
                n_buf[i] = (i < MTP_HIDDEN) ? up_out[i] : (half)0;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // 3. Down projection: n_buf (read) -> h_buf (write, overwriting input)
            uint down_base = head_w_off + MTP_HIDDEN * HID_TILES * LANES;
            for (uint row = 0; row < HIDDEN_DIM; row += 32) {
                uint r = row + (tid & 31u);
                if (r < HIDDEN_DIM) {
                    float acc = tile_gemv(mtp_ternary_w, down_base + r * MTP_TILES * LANES,
                                         MTP_TILES, tid & 31u, n_buf);
                    acc = warp_sum(acc);
                    if ((tid & 31u) == 0) h_buf[r] = (half)acc;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            // 4. Residual: h_buf (MLP output) += original h_buf from device backup
            for (uint i = tid; i < HIDDEN_DIM; i += tg_sz) h_buf[i] += mtp_scratch[i];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // RMS norm on h_buf
            for (uint i = tid; i < HIDDEN_DIM; i += tg_sz) n_buf[i] = h_buf[i];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            fast_rmsnorm(n_buf, norms + 0, tid, tg_sz, shared_sums);
            // Copy normalized result back to h_buf for centroid scout
            for (uint i = tid; i < HIDDEN_DIM; i += tg_sz) h_buf[i] = n_buf[i];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Centroid scout for this MTP head's logits
            device half* head_logits = slot_logits + (mtp_head + 1) * VOCAB_SIZE;

            // Step A: dot products against all centroids
            for (uint c = tid; c < NUM_CENTROIDS; c += tg_sz) {
                float score = 0.0;
                for (uint d = 0; d < HIDDEN_DIM; ++d) {
                    score += (float)h_buf[d] * (float)centroid_scratch[c * HIDDEN_DIM + d];
                }
                centroid_scores[c] = score;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Step B: find best cluster
            threadgroup uint head_best_cluster = 0;
            if (tid == 0) {
                float best_val = -1e10;
                for (uint i = 0; i < NUM_CENTROIDS; ++i) {
                    if (centroid_scores[i] > best_val) {
                        best_val = centroid_scores[i];
                        head_best_cluster = i;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Step C: find cluster bounds
            if (tid == 0) {
                uint start = VOCAB_SIZE;
                uint end = 0;
                for (uint pos = 0; pos < VOCAB_SIZE; ++pos) {
                    if (cluster_map[pos] == head_best_cluster) {
                        if (pos < start) start = pos;
                        end = pos + 1;
                    }
                }
                cluster_bounds[0] = start;
                cluster_bounds[1] = end;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Step D: compute logits for the winning cluster
            uint hcstart = cluster_bounds[0];
            uint hcend = cluster_bounds[1];
            for (uint row = hcstart; row < hcend; ++row) {
                uint simd_lane = tid & 31;
                uint simd_id = tid / 32;
                if ((row - hcstart) % (tg_sz / 32) == simd_id) {
                    uint tile_base = row * HID_TILES * LANES;
                    float acc = 0.0;
                    for (uint b = 0; b < HID_TILES; ++b) {
                        uint val = embed_clust[tile_base + b * LANES + simd_lane];
                        uint act_base = b * TILE + simd_lane * PER_LANE;
                        for (uint i = 0; i < PER_LANE; ++i) {
                            uint rem = fast_mod3(val);
                            int wgt = (int)rem - 1;
                            if (wgt != 0) {
                                acc += (float)h_buf[act_base + i] * (float)wgt;
                            }
                            val = fast_div3(val);
                        }
                    }
                    acc = warp_sum(acc);
                    if (simd_lane == 0) {
                        uint block_idx = row / 256;
                        float s = (float)embed_scales[block_idx];
                        head_logits[row] = (half)(acc * s);
                    }
                }
            }
            // Fill non-cluster logits with -inf
            for (uint row = tid; row < VOCAB_SIZE; row += tg_sz) {
                if (row < hcstart || row >= hcend) {
                    head_logits[row] = as_type<half>((unsigned short)0xFC00u);
                }
            }
            threadgroup_barrier(mem_flags::mem_device);
        }
    }

    }  // end if (kind == 0)
    else if (kind == 3) {
        // ── Draft model forward pass (sub-1ms speculative drafter) ──
        // h_buf already contains the embedded input from Stage 0
        // Reads draft_ternary_w (same ternary nibble format as main model)
        // Processes through DRAFT_LAYERS layers
        // Uses kv_scratch_k/v as FP16 KV cache for draft (small window)
        // No MTP heads, no centroid scout, no entropy accumulation

        uint draft_kv_stride = DRAFT_NUM_KV_HEADS * DRAFT_HEAD_DIM;
        uint draft_cache_pos = 0u;

        for (uint layer = 0; layer < DRAFT_LAYERS; ++layer) {
            uint h_dim = DRAFT_HEAD_DIM;
            uint layer_base = layer * DRAFT_LAYER_STRIDE;  // uses draft nibble weight layout stride
            uint scratch_layer_base = layer * MAX_CTX * draft_kv_stride;

            // --- 1. Input RMSNorm (inlined with DRAFT_HIDDEN dimension) ---
            device const half* in_norm_w = draft_norms + layer * DRAFT_HIDDEN;
            shared_sums[tid] = 0.0;
            for (uint i = tid; i < DRAFT_HIDDEN; i += tg_sz) {
                float v = (float)h_buf[i];
                shared_sums[tid] += v * v;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = tg_sz / 2; stride > 0; stride >>= 1) {
                if (tid < stride) shared_sums[tid] += shared_sums[tid + stride];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float rcp = rsqrt(shared_sums[0] / (float)DRAFT_HIDDEN + 1e-6);
            for (uint i = tid; i < DRAFT_HIDDEN; i += tg_sz) {
                n_buf[i] = (half)((float)h_buf[i] * rcp * (float)in_norm_w[i]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // --- 2. K/V projections via tile_gemv (nibble-based ternary) ---
            uint kw_base = layer_base + DRAFT_K_OFF;
            uint vw_base = layer_base + DRAFT_V_OFF;
            uint qw_base = layer_base + DRAFT_Q_OFF;
            uint ow_base = layer_base + DRAFT_O_OFF;

            // K projection: DRAFT_HIDDEN -> h_dim (per KV head)
            for (uint kv_h = 0; kv_h < DRAFT_NUM_KV_HEADS; ++kv_h) {
                for (uint o = 0; o < h_dim; o += 32) {
                    uint r = o + (tid & 31u);
                    if (r < h_dim) {
                        uint flat_row = kv_h * h_dim + r;
                        float dp = tile_gemv(draft_ternary_w, kw_base + flat_row * DRAFT_KV_TILES * LANES,
                                         DRAFT_KV_TILES, tid & 31u, n_buf);
                        dp = warp_sum(dp);
                        if ((tid & 31u) == 0) n_buf[DRAFT_HIDDEN + r] = (half)dp;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }

            // V projection: DRAFT_HIDDEN -> h_dim (per KV head)
            for (uint kv_h = 0; kv_h < DRAFT_NUM_KV_HEADS; ++kv_h) {
                for (uint o = 0; o < h_dim; o += 32) {
                    uint r = o + (tid & 31u);
                    if (r < h_dim) {
                        uint flat_row = kv_h * h_dim + r;
                        float dp = tile_gemv(draft_ternary_w, vw_base + flat_row * DRAFT_KV_TILES * LANES,
                                         DRAFT_KV_TILES, tid & 31u, n_buf);
                        dp = warp_sum(dp);
                        if ((tid & 31u) == 0) n_buf[DRAFT_HIDDEN + DRAFT_KV_TILES * TILE + r] = (half)dp;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }

            // Store K,V to FP16 device cache (kv_scratch_k/v, one position per layer)
            for (uint kv_h = 0; kv_h < DRAFT_NUM_KV_HEADS; ++kv_h) {
                for (uint i = tid; i < h_dim; i += tg_sz) {
                    kv_scratch_k[scratch_layer_base + draft_cache_pos * draft_kv_stride + kv_h * h_dim + i] =
                        n_buf[DRAFT_HIDDEN + kv_h * h_dim + i];
                    kv_scratch_v[scratch_layer_base + draft_cache_pos * draft_kv_stride + kv_h * h_dim + i] =
                        n_buf[DRAFT_HIDDEN + DRAFT_KV_TILES * TILE + kv_h * h_dim + i];
                }
            }
            threadgroup_barrier(mem_flags::mem_device);

            // --- 3. Full Q projection: DRAFT_HIDDEN -> DRAFT_HIDDEN (all 8 heads) ---
            for (uint qh = 0; qh < DRAFT_NUM_HEADS; ++qh) {
                for (uint o = 0; o < h_dim; o += 32) {
                    uint r = o + (tid & 31u);
                    if (r < h_dim) {
                        uint flat_row = qh * h_dim + r;
                        float dp = tile_gemv(draft_ternary_w, qw_base + flat_row * DRAFT_Q_TILES * LANES,
                                         DRAFT_Q_TILES, tid & 31u, n_buf);
                        dp = warp_sum(dp);
                        if ((tid & 31u) == 0) q_chunk[qh * h_dim + r] = (half)dp;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }

            // --- 4. RoPE on all Q heads ---
            apply_rope(q_chunk, DRAFT_NUM_HEADS, h_dim, draft_cache_pos, tid, tg_sz);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // --- 5. Attention init ---
            for (uint i = tid; i < DRAFT_HIDDEN; i += tg_sz) {
                n_buf[DRAFT_HIDDEN + 2 * h_dim + i] = 0;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // GQA: 8 Q heads, 4 KV heads (2:1). For each KV head, process 2 Q heads.
            for (uint kv_h = 0; kv_h < DRAFT_NUM_KV_HEADS; ++kv_h) {
                for (uint q_pair = 0; q_pair < 2; ++q_pair) {
                    uint qh = 2 * kv_h + q_pair;
                    threadgroup half* q_head = q_chunk + qh * h_dim;

                    // Q*K dot products for all cached positions (0..draft_cache_pos)
                    float max_val = -1e10;
                    for (uint p = tid; p <= draft_cache_pos; p += tg_sz) {
                        float s = 0.0;
                        device half* kv_k_ptr = kv_scratch_k + scratch_layer_base + p * draft_kv_stride + kv_h * h_dim;
                        for (uint d = 0; d < h_dim; ++d)
                            s += (float)q_head[d] * (float)kv_k_ptr[d];
                        slot_logits[p] = (half)s;
                        if (s > max_val) max_val = s;
                    }
                    shared_sums[tid] = max_val;
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    for (uint stride = tg_sz / 2; stride > 0; stride >>= 1) {
                        if (tid < stride && shared_sums[tid + stride] > shared_sums[tid])
                            shared_sums[tid] = shared_sums[tid + stride];
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                    }
                    float g_max = shared_sums[0];
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    // softmax
                    float sum_exp = 0.0;
                    for (uint p = tid; p <= draft_cache_pos; p += tg_sz) {
                        float e = exp((float)slot_logits[p] - g_max);
                        slot_logits[p] = (half)e;
                        sum_exp += e;
                    }
                    shared_sums[tid] = sum_exp;
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    for (uint stride = tg_sz / 2; stride > 0; stride >>= 1) {
                        if (tid < stride) shared_sums[tid] += shared_sums[tid + stride];
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                    }
                    threadgroup_barrier(mem_flags::mem_device);
                    float inv_s = 1.0 / shared_sums[0];

                    // Weighted sum of V
                    for (uint d = tid; d < h_dim; d += tg_sz) {
                        float acc = 0.0;
                        for (uint p = 0; p <= draft_cache_pos; ++p) {
                            float s = (float)slot_logits[p] * inv_s;
                            device half* kv_v_ptr = kv_scratch_v + scratch_layer_base + p * draft_kv_stride + kv_h * h_dim;
                            acc += s * (float)kv_v_ptr[d];
                        }
                        n_buf[DRAFT_HIDDEN + 2 * h_dim + qh * h_dim + d] = (half)acc;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }

            // --- 6. O projection + residual ---
            for (uint i = tid; i < DRAFT_HIDDEN; i += tg_sz) {
                n_buf[i] = n_buf[DRAFT_HIDDEN + 2 * h_dim + i];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // O projection: DRAFT_HIDDEN -> DRAFT_HIDDEN
            for (uint row = 0; row < DRAFT_HIDDEN; row += 32) {
                uint r = row + (tid & 31u);
                if (r < DRAFT_HIDDEN) {
                    float dp = tile_gemv(draft_ternary_w, ow_base + row * DRAFT_HID_TILES * LANES,
                                     DRAFT_HID_TILES, tid & 31u, n_buf);
                    dp = warp_sum(dp);
                    if ((tid & 31u) == 0) h_buf[r] += (half)dp;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            // --- 7. Post-Attention RMSNorm (inlined) ---
            shared_sums[tid] = 0.0;
            for (uint i = tid; i < DRAFT_HIDDEN; i += tg_sz) {
                float v = (float)h_buf[i];
                shared_sums[tid] += v * v;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = tg_sz / 2; stride > 0; stride >>= 1) {
                if (tid < stride) shared_sums[tid] += shared_sums[tid + stride];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float rcp2 = rsqrt(shared_sums[0] / (float)DRAFT_HIDDEN + 1e-6);
            for (uint i = tid; i < DRAFT_HIDDEN; i += tg_sz) {
                n_buf[i] = (half)((float)h_buf[i] * rcp2 * (float)in_norm_w[i]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // --- 8. MLP: Gate projection ---
            uint gate_base = layer_base + DRAFT_GATE_OFF;
            uint up_base   = layer_base + DRAFT_UP_OFF;
            for (uint row = 0; row < DRAFT_FFN_INTER; row += 32) {
                uint r = row + (tid & 31u);
                if (r < DRAFT_FFN_INTER) {
                    float dp = tile_gemv(draft_ternary_w, gate_base + row * DRAFT_HID_TILES * LANES,
                                     DRAFT_HID_TILES, tid & 31u, n_buf);
                    dp = warp_sum(dp);
                    if ((tid & 31u) == 0) slot_logits[r] = (half)dp;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            // --- 9. MLP: Up projection ---
            for (uint row = 0; row < DRAFT_FFN_INTER; row += 32) {
                uint r = row + (tid & 31u);
                if (r < DRAFT_FFN_INTER) {
                    float dp = tile_gemv(draft_ternary_w, up_base + row * DRAFT_HID_TILES * LANES,
                                     DRAFT_HID_TILES, tid & 31u, n_buf);
                    dp = warp_sum(dp);
                    if ((tid & 31u) == 0) slot_logits[DRAFT_FFN_INTER + r] = (half)dp;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            // --- 10. SwiGLU + Down projection ---
            uint down_base = layer_base + DRAFT_DOWN_OFF;
            for (uint row = 0; row < DRAFT_HIDDEN; row += 32) {
                float dp_total = 0.0;
                for (uint t = 0; t < DRAFT_FFN_TILES; ++t) {
                    uint tile_off = t * TILE;
                    uint n_off = t * TILE;
                    for (uint i = tid; i < TILE; i += tg_sz) {
                        float g = (float)slot_logits[tile_off + i];
                        float u = (float)slot_logits[DRAFT_FFN_INTER + tile_off + i];
                        n_buf[n_off + i] = (half)((g / (1.0 + exp(-g))) * u);
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    uint r = row + (tid & 31u);
                    if (r < DRAFT_HIDDEN) {
                        uint tile_base = down_base + row * DRAFT_FFN_TILES * LANES + t * LANES;
                        float dp = tile_gemv(draft_ternary_w, tile_base, 1, tid & 31u, n_buf);
                        dp_total += warp_sum(dp);
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                float result = warp_sum(dp_total);
                if ((tid & 31u) == 0) h_buf[row] += (half)result;
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }  // end for (uint layer = 0; layer < DRAFT_LAYERS; ++layer)

        // ── After all layers: output projection to vocab via centroid scout ──
        // Step A: dot products against all centroids.
        for (uint c = tid; c < NUM_CENTROIDS; c += tg_sz) {
            float score = 0.0;
            for (uint d = 0; d < DRAFT_HIDDEN; ++d) {
                score += (float)h_buf[d] * (float)centroid_scratch[c * HIDDEN_DIM + d];
            }
            centroid_scores[c] = score;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step B: Find best cluster
        threadgroup uint best_cluster = 0;
        if (tid == 0) {
            float best_val = -1e10;
            for (uint i = 0; i < NUM_CENTROIDS; ++i) {
                if (centroid_scores[i] > best_val) {
                    best_val = centroid_scores[i];
                    best_cluster = i;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step C: Find cluster [start, end) positions in the vocabulary.
        if (tid == 0) {
            uint start = VOCAB_SIZE;
            uint end = 0;
            for (uint pos = 0; pos < VOCAB_SIZE; ++pos) {
                if (cluster_map[pos] == best_cluster) {
                    if (pos < start) start = pos;
                    end = pos + 1;
                }
            }
            cluster_bounds[0] = start;
            cluster_bounds[1] = end;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step D: Compute logits for the winning cluster
        uint cstart = cluster_bounds[0];
        uint cend = cluster_bounds[1];
        for (uint row = cstart; row < cend; ++row) {
            uint simd_lane = tid & 31;
            uint simd_id = tid / 32;
            if ((row - cstart) % (tg_sz / 32) == simd_id) {
                uint tile_base = row * HID_TILES * LANES;
                float acc = 0.0;
                for (uint b = 0; b < HID_TILES; ++b) {
                    uint val = embed_clust[tile_base + b * LANES + simd_lane];
                    uint act_base = b * TILE + simd_lane * PER_LANE;
                    for (uint i = 0; i < PER_LANE; ++i) {
                        uint rem = fast_mod3(val);
                        int wgt = (int)rem - 1;
                        if (wgt != 0) {
                            acc += (float)h_buf[act_base + i] * (float)wgt;
                        }
                        val = fast_div3(val);
                    }
                }
                acc = warp_sum(acc);
                if (simd_lane == 0) {
                    uint block_idx = row / 256;
                    float s = (float)embed_scales[block_idx];
                    slot_logits[row] = (half)(acc * s);
                }
            }
        }
        // Fill non-cluster logits with -inf
        for (uint row = tid; row < VOCAB_SIZE; row += tg_sz) {
            if (row < cstart || row >= cend) {
                slot_logits[row] = as_type<half>((unsigned short)0xFC00u);
            }
        }
        threadgroup_barrier(mem_flags::mem_device);

        // ── Top-5 token selection ──
        threadgroup float top5_vals[5];
        threadgroup uint top5_ids[5];
        if (tid == 0) {
            for (uint i = 0; i < 5; ++i) {
                top5_vals[i] = -1e10;
                top5_ids[i] = 0;
            }
            for (uint row = 0; row < VOCAB_SIZE; ++row) {
                float val = (float)slot_logits[row];
                if (val > top5_vals[4]) {
                    uint pos = 4;
                    while (pos > 0 && val > top5_vals[pos - 1]) --pos;
                    for (uint i = 4; i > pos; --i) {
                        top5_vals[i] = top5_vals[i - 1];
                        top5_ids[i] = top5_ids[i - 1];
                    }
                    top5_vals[pos] = val;
                    top5_ids[pos] = row;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Write top-5 candidates to draft_output buffer ──
        if (tid == 0) {
            draft_output[0] = 5;
            for (uint i = 0; i < 5; ++i) {
                draft_output[1 + i] = top5_ids[i];
                half logprob = (half)top5_vals[i];
                draft_output[6 + i] = as_type<uint>(logprob);
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
    }  // end else if (kind == 3)

                // --- After decode: signal COMPLETED -------------------
                threadgroup_barrier(mem_flags::mem_device);
                atomic_store_explicit(
                    (device atomic_uint*)entry, 3 | (kind << 2), memory_order_relaxed);  // COMPLETED
                atomic_fetch_add_explicit(completion_counter, 1, memory_order_relaxed);
                processed = true;
            }
        }

        if (!processed && tid == 0) {
            // Optional hint; on Apple GPUs this is a no-op
        }
    }  // end while(true)
}
"##;

// ====================================================================
//  Compilation
// ====================================================================
pub(crate) fn compile_kernel(device: &Device, int4: bool) -> Result<ComputePipelineState, String> {
    let shader_src = if int4 { SHADER_SRC_INT4 } else { SHADER_SRC };
    let tmp = std::env::temp_dir().join("tribunus-full-transformer");
    let _ = std::fs::create_dir_all(&tmp);

    let src_path = tmp.join("gemma4_full.metal");
    let air_path = tmp.join("gemma4_full.air");
    let lib_path = tmp.join("gemma4_full.metallib");

    std::fs::write(&src_path, shader_src)
        .map_err(|e| format!("failed to write Metal source: {e}"))?;

    // Step 1: Compile .metal → .air via metal compiler
    let mut cmd = std::process::Command::new("xcrun");
    cmd.args(["-sdk", "macosx", "metal", "-std=metal3.2", "-O3", "-c"]);
    cmd.arg(src_path.to_str().unwrap())
        .arg("-o")
        .arg(air_path.to_str().unwrap());
    let status = cmd.status().map_err(|e| format!("xcrun metal: {e}"))?;
    if !status.success() {
        return Err("Metal source compilation failed".into());
    }

    // Step 2: Link .air → .metallib via metallib linker
    let mut cmd = std::process::Command::new("xcrun");
    cmd.args(["-sdk", "macosx", "metallib", "-o"]);
    cmd.arg(lib_path.to_str().unwrap())
        .arg(air_path.to_str().unwrap());
    let status = cmd.status().map_err(|e| format!("xcrun metallib: {e}"))?;
    if !status.success() {
        return Err("Metal library linking failed".into());
    }

    let lib_data = std::fs::read(&lib_path).map_err(|e| format!("read metallib: {e}"))?;
    let library = device
        .new_library_with_data(&lib_data)
        .map_err(|e| format!("new_library: {:?}", e))?;
    let function = library
        .get_function("gemma4_full_decode_persistent", None)
        .map_err(|e| format!("get_function: {:?}", e))?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| format!("pipeline state: {:?}", e))
}

/// Load a pre-compiled .metallib from bytes (alias for INT4 variant — same shader).
pub fn compile_kernel_from_metallib_int4(
    device: &Device,
    data: &[u8],
) -> Result<ComputePipelineState, String> {
    compile_kernel_from_metallib(device, data)
}

/// Load a pre-compiled .metallib from bytes and create a pipeline state.
pub fn compile_kernel_from_metallib(
    device: &Device,
    data: &[u8],
) -> Result<ComputePipelineState, String> {
    let library = device
        .new_library_with_data(data)
        .map_err(|e| format!("new_library_with_data: {:?}", e))?;
    let function = library
        .get_function("gemma4_full_decode_persistent", None)
        .map_err(|e| format!("get_function: {:?}", e))?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| format!("pipeline state: {:?}", e))
}
