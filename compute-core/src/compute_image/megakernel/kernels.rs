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

// ── KV Interleave ABI constants ─────────────────────────────────
#define CLAIM_UNOWNED 0
#define CLAIM_HELPER 1
#define CLAIM_DECODE_FALLBACK 2
#define CLAIM_DECODE_CONSUMER 3

#define OUTCOME_NONE 0
#define OUTCOME_READY_CONSUMABLE 1
#define OUTCOME_CANCELED 2
#define OUTCOME_POISONED 3
#define OUTCOME_BYPASSED 4

#define FAULT_NONE 0
#define FAULT_HANDOFF_INTEGRITY 1
#define FAULT_INVALID_READY_STATE 2
#define FAULT_GENERATION_MISMATCH 3
#define FAULT_UNRECOGNIZED_OUTCOME 4

#define KV_STATE_EMPTY 0
#define KV_STATE_QUEUED 1
#define KV_STATE_FILLING 2
#define KV_STATE_READY 3
#define KV_STATE_CONSUMING 7
#define KV_STATE_RECLAIMABLE 8
#define KV_STATE_POISONED 5
#define KV_STATE_CANCELED 6

struct KvScratchMetadataAbi {
    uint request_id, session_id, sequence_id, target_layer;
    uint token_epoch, kv_generation, page_table_generation, data_offset;
};

struct KvScratchDeviceControl {
    atomic_uint state;
    atomic_uint cancel_requested;
    atomic_uint payload_valid;
    atomic_uint request_generation;
    atomic_uint request_outcome;
    atomic_uint producer_claim;
    atomic_uint duplicate_write_detected;
    atomic_uint late_publish_rejection_count;
};

struct KvScratchHeader {
    KvScratchMetadataAbi metadata;
    KvScratchDeviceControl control;
};

struct EpochControl {
    atomic_uint epoch_close_requested;
    atomic_uint epoch_enqueue_limit;
    atomic_uint epoch_fatal_claim;
    atomic_uint epoch_fatal_fault;
    atomic_uint epoch_fatal_fault_generation;
    atomic_uint epoch_fatal_fault_request_id;
};

struct EpochReceiptData {
    atomic_uint requests_claimed;
    atomic_uint requests_ready_consumable;
    atomic_uint staging_consumptions;
    atomic_uint requests_canceled;
    atomic_uint requests_poisoned;
    atomic_uint requests_bypassed;
    atomic_uint late_ready_discarded_diagnostic;
    atomic_uint duplicate_write_detected;
    atomic_uint requests_unresolved;
};


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
    device const half*    head_gates        [[buffer(29)]],  // per-head attention gates (NUM_Q_HEADS × f16)
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
        // SIMD-group-parallel KV decompression: each SIMD group (32 lanes)
        // handles one KV head.  32-lane cooperation preserves coalesced
        // memory access.  0 barriers between heads — 1 final barrier.
        uint simd_group = tid / 32;
        uint lane = tid & 31;
        for (uint p = 0; p < num_cached; ++p) {
            uint h = simd_group;
            if (h < NUM_KV_HEADS) {
                for (uint b = 0; b < blocks_per_head; ++b) {
                uint pos_head_vals = slot_kv_offset + layer * MAX_CTX * scratch_stride
                                   + p * scratch_stride + h * GLOBAL_HEAD_DIM;
                    uint val_offset = pos_head_vals + b * KV_BLOCK;
                    uint block_idx = val_offset / KV_BLOCK;
                    uint nibble_idx = block_idx * KV_NIBBLES_U32;

                    // Clamp tile count to valid dimension range
                    uint t_limit = KV_NIBBLES_U32;
                    uint max_dim = b * KV_BLOCK + t_limit * PER_LANE;
                    if (max_dim > h_dim) {
                        t_limit = (h_dim - b * KV_BLOCK + PER_LANE - 1) / PER_LANE;
                    }

                    // Decompress K block — 32 lanes cooperate (coalesced access)
                    half scale_k = kv_k_scales[block_idx];
                    for (uint t = lane; t < t_limit; t += 32) {
                        uint val = kv_k_nibbles[nibble_idx + t];
                        uint dim_base = b * KV_BLOCK + t * PER_LANE;
                        uint rem_el = PER_LANE;
                        if (dim_base + rem_el > h_dim) rem_el = h_dim - dim_base;
                        for (uint i = 0; i < rem_el; ++i) {
                            uint rem = fast_mod3(val);
                            if (rem != 1) {
                                uint scratch_pos = p * scratch_stride + h * GLOBAL_HEAD_DIM + dim_base + i;
                                kv_scratch_k[scratch_pos] = (half)((int)(rem - 1) * (float)scale_k);
                            }
                            val = fast_div3(val);
                        }
                    }

                    // Decompress V block — same SIMD group
                    half scale_v = kv_v_scales[block_idx];
                    for (uint t = lane; t < t_limit; t += 32) {
                        uint val = kv_v_nibbles[nibble_idx + t];
                        uint dim_base = b * KV_BLOCK + t * PER_LANE;
                        uint rem_el = PER_LANE;
                        if (dim_base + rem_el > h_dim) rem_el = h_dim - dim_base;
                        for (uint i = 0; i < rem_el; ++i) {
                            uint rem = fast_mod3(val);
                            if (rem != 1) {
                                uint scratch_pos = p * scratch_stride + h * GLOBAL_HEAD_DIM + dim_base + i;
                                kv_scratch_v[scratch_pos] = (half)((int)(rem - 1) * (float)scale_v);
                            }
                            val = fast_div3(val);
                        }
                    }
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
                half gate = (half)(1.0 / (1.0 + exp(-(float)head_gates[qh])));
                // Pass 3: weighted sum of V (gated by per-head attention gate)
                for (uint d = tid; d < h_dim; d += tg_sz) {
                    float acc = 0.0;
                    for (uint p = 0; p < num_cached; ++p) {
                        float s = (float)slot_logits[p] * inv_s;
                        acc += s * (float)kv_scratch_v[p * scratch_stride + kv_h * GLOBAL_HEAD_DIM + d];
                    }
                    uint write_pos = qh * h_dim + d;
                    if (write_pos < HIDDEN_DIM)
                        n_buf[write_pos] = (half)((float)acc * (float)gate);
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
                draft_output[6 + i] = (uint)as_type<ushort>(logprob);
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


kernel void persistent_decode_worker(
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
    device const half*    head_gates        [[buffer(29)]],  // per-head attention gates (NUM_Q_HEADS × f16)
    device KvPrefetchQueueAbi* kv_prefetch_queue [[buffer(30)]],
    device KvScratchHeader* scratch_set_a [[buffer(31)]],
    device KvScratchHeader* scratch_set_b [[buffer(32)]],
    device half* scratch_k_a [[buffer(33)]],
    device half* scratch_v_a [[buffer(34)]],
    device half* scratch_k_b [[buffer(35)]],
    device half* scratch_v_b [[buffer(36)]],
    constant uint&        max_tokens_per_epoch [[buffer(37)]],
    device EpochControl*  epoch_control [[buffer(38)]],
    device EpochReceiptData* receipt [[buffer(39)]],
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
    threadgroup uint token_count;             // tokens consumed this epoch

    // ── Double-buffered KV scratch ──
    uint active_set = 0;
    device half* kv_scratch_k = scratch_k_a;
    device half* kv_scratch_v = scratch_v_a;

    token_count = 0;

    while (token_count < max_tokens_per_epoch) {
        device bool queue_active = (kv_prefetch_queue != nullptr);

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

        // ── Full-snapshot terminal execution router ─────────────────---
        // Phase 1: Claim a snapshot from the epoch control.
        //   CLAIM_UNOWNED=0 → CLAIM_HELPER=1 if epoch is closing (fallback)
        //                    → CLAIM_DECODE_CONSUMER=3 if primary consumer
        //                    → CLAIM_DECODE_FALLBACK=2 if epoch not closing but we lost primary

        uint claim = CLAIM_UNOWNED;
        uint retries = 0;
        uint max_retries = 8;
        uint my_request_id = 0;
        uint my_session_id = 0;
        uint my_sequence_id = 0;
        uint my_target_layer = 0;
        uint my_token_epoch = 0;
        uint my_kv_generation = 0;
        uint my_page_table_generation = 0;

        while (claim == CLAIM_UNOWNED && retries < max_retries) {
            // Fresh snapshot reloads
            threadgroup_barrier(mem_flags::mem_device);
            uint epoch_close = atomic_load_explicit(
                &epoch_control->epoch_close_requested, memory_order_acquire,
                memory_scope_device);
            uint epoch_limit = atomic_load_explicit(
                &epoch_control->epoch_enqueue_limit, memory_order_acquire,
                memory_scope_device);
            uint fatal_claim = atomic_load_explicit(
                &epoch_control->epoch_fatal_claim, memory_order_acquire,
                memory_scope_device);

            if (fatal_claim != CLAIM_UNOWNED) {
                // A previous worker already recorded a fatal fault;
                // exit the epoch for all workers.
                claim = CLAIM_HELPER;
                break;
            }

            // ── Ring dequeue ──
            bool dequeued = false;
            uint my_tail = atomic_fetch_add_explicit(
                ring_tail, 1, memory_order_relaxed);
            uint idx_ring = my_tail % RING_SIZE;
            device uint* entry = ring_entries + idx_ring * 4;
            uint entry_state = atomic_load_explicit(
                (device atomic_uint*)entry, memory_order_relaxed);
            uint kind = entry_state >> 2;
            if ((entry_state & 3) == 1) {  // SUBMITTED
                uint expected = entry_state;
                if (atomic_compare_exchange_weak_explicit(
                    (device atomic_uint*)entry, &expected, 2 | (kind << 2),  // CLAIMED
                    memory_order_relaxed, memory_order_relaxed)) {
                    my_request_id = entry[1];
                    my_session_id = entry[2];  // reusing; actually current_token
                    my_sequence_id = entry[3]; // actually current_pos
                    // my_token_epoch = current_pos (from entry)
                    // my_kv_generation = entry[...] - set during decode
                    dequeued = true;
                }
            }

            if (!dequeued) {
                // No work — yield for other warps and retry
                threadgroup_barrier(mem_flags::mem_threadgroup);
                ++retries;
                continue;
            }

            ++retries;
            my_token_epoch = my_sequence_id;   // current token's position
            my_kv_generation = epoch_limit;    // local generation tag

            // Determine claim type
            if (epoch_close != 0) {
                // Epoch is closing — we act as helper for cleanup
                claim = CLAIM_HELPER;
            } else {
                // Primary decode consumer
                claim = CLAIM_DECODE_CONSUMER;
            }
        }

        if (claim == CLAIM_UNOWNED) {
            // Bounded retry exhausted without finding work.
            // If epoch is closing, exit immediately (drain protocol).
            uint epoch_close = atomic_load_explicit(
                &epoch_control->epoch_close_requested, memory_order_acquire,
                memory_scope_device);
            if (epoch_close != 0) {
                break;
            }
            // Otherwise spin-yield and retry the main loop
            threadgroup_barrier(mem_flags::mem_threadgroup);
            continue;
        }

        if (claim == CLAIM_HELPER) {
            // ── Drain protocol at epoch close ──
            // Set epoch_enqueue_limit to final enqueue position, set
            // epoch_close_requested, issue cancellation for unresolved sets.
            if (tid == 0) {
                uint final_enq = atomic_load_explicit(
                    &receipt->requests_claimed, memory_order_acquire,
                    memory_scope_device);
                atomic_store_explicit(
                    &epoch_control->epoch_enqueue_limit, final_enq,
                    memory_order_release, memory_scope_device);
                atomic_store_explicit(
                    &epoch_control->epoch_close_requested, 1,
                    memory_order_release, memory_scope_device);

                // Issue cancellation for all unresolved sets
                // (Host-side will pick up the cancellation requests)
            }
            // EXIT IMMEDIATELY — no blocking on helper tasks
            break;
        }

        // ── CLAIM_DECODE_CONSUMER or CLAIM_DECODE_FALLBACK ──
        uint current_token = my_session_id;
        uint current_pos   = my_sequence_id;
        uint kv_slot_id    = 0;  // derived below

        // Reference: compute kv slot from current_token
        kv_slot_id = current_token % NUM_SLOTS;

        // Number of valid KV cache positions
        uint num_cached = min(current_pos + 1, MAX_CTX);

        // KV cache offset for this partition
        uint slot_kv_offset = kv_slot_id * MAX_CTX * NUM_KV_HEADS * GLOBAL_HEAD_DIM * LAYERS;

        // Logits output goes to the slot's logits region
        device half* slot_logits = slot_logits_base + kv_slot_id * VOCAB_SIZE;

        // Increment claim counter
        atomic_fetch_add_explicit(&receipt->requests_claimed, 1, memory_order_relaxed, memory_scope_device);

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

    // --- Compute generation tag for KV pages ---
    uint kv_generation = atomic_load_explicit(
        &epoch_control->epoch_fatal_fault_generation, memory_order_acquire,
        memory_scope_device);
    uint page_table_generation = kv_generation;

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


        // ── Bounded poll for prefetched KV ──
        bool kv_prefetched = false;
        if (layer > 0) {
            device KvScratchHeader* poll_header = (active_set == 0) ? scratch_set_a : scratch_set_b;
            for (uint spin = 0; spin < 64; ++spin) {
                uint st = atomic_load_explicit(&poll_header->control.state, memory_order_acquire, memory_scope_device);
                if (st == KV_STATE_READY) {
                    kv_prefetched = true;
                    // Staging consumption
                    atomic_store_explicit(&poll_header->control.state, KV_STATE_CONSUMING, memory_order_release, memory_scope_device);
                    atomic_fetch_add_explicit(&receipt->staging_consumptions, 1, memory_order_relaxed, memory_scope_device);
                    break;
                }
            }
        }

        if (!kv_prefetched) {
        // ── Decompress K/V for this layer from ternary ──
        uint scratch_stride = NUM_KV_HEADS * GLOBAL_HEAD_DIM;
        for (uint i = tid; i < MAX_CTX * scratch_stride; i += tg_sz) {
            kv_scratch_k[i] = 0;
            kv_scratch_v[i] = 0;
        }
        threadgroup_barrier(mem_flags::mem_device);
        uint blocks_per_head = (h_dim + 255) / 256;
        uint bytes_per_kv_block = KV_NIBBLES_U32 * 4u;  // 52 bytes = 260 elements, 256 used
        uint simd_group = tid / 32;
        uint lane = tid & 31;
        for (uint p = 0; p < num_cached; ++p) {
            uint h = simd_group;
            if (h < NUM_KV_HEADS) {
                for (uint b = 0; b < blocks_per_head; ++b) {
                uint pos_head_vals = slot_kv_offset + layer * MAX_CTX * scratch_stride
                                   + p * scratch_stride + h * GLOBAL_HEAD_DIM;
                    uint val_offset = pos_head_vals + b * KV_BLOCK;
                    uint block_idx = val_offset / KV_BLOCK;
                    uint nibble_idx = block_idx * KV_NIBBLES_U32;

                    uint t_limit = KV_NIBBLES_U32;
                    uint max_dim = b * KV_BLOCK + t_limit * PER_LANE;
                    if (max_dim > h_dim) {
                        t_limit = (h_dim - b * KV_BLOCK + PER_LANE - 1) / PER_LANE;
                    }

                    // Decompress K block
                    half scale_k = kv_k_scales[block_idx];
                    for (uint t = lane; t < t_limit; t += 32) {
                        uint val = kv_k_nibbles[nibble_idx + t];
                        uint dim_base = b * KV_BLOCK + t * PER_LANE;
                        uint rem_el = PER_LANE;
                        if (dim_base + rem_el > h_dim) rem_el = h_dim - dim_base;
                        for (uint i = 0; i < rem_el; ++i) {
                            uint rem = fast_mod3(val);
                            if (rem != 1) {
                                uint scratch_pos = p * scratch_stride + h * GLOBAL_HEAD_DIM + dim_base + i;
                                kv_scratch_k[scratch_pos] = (half)((int)(rem - 1) * (float)scale_k);
                            }
                            val = fast_div3(val);
                        }
                    }

                    // Decompress V block
                    half scale_v = kv_v_scales[block_idx];
                    for (uint t = lane; t < t_limit; t += 32) {
                        uint val = kv_v_nibbles[nibble_idx + t];
                        uint dim_base = b * KV_BLOCK + t * PER_LANE;
                        uint rem_el = PER_LANE;
                        if (dim_base + rem_el > h_dim) rem_el = h_dim - dim_base;
                        for (uint i = 0; i < rem_el; ++i) {
                            uint rem = fast_mod3(val);
                            if (rem != 1) {
                                uint scratch_pos = p * scratch_stride + h * GLOBAL_HEAD_DIM + dim_base + i;
                                kv_scratch_v[scratch_pos] = (half)((int)(rem - 1) * (float)scale_v);
                            }
                            val = fast_div3(val);
                        }
                    }
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
                float max_val = -1e10;
                for (uint p = tid; p < num_cached; p += tg_sz) {
                    float s = 0.0;
                    for (uint d = 0; d < h_dim; ++d)
                        s += (float)q_chunk[d] * (float)kv_scratch_k[p * scratch_stride + kv_h * GLOBAL_HEAD_DIM + d];
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

                float sum_exp = 0.0;
                for (uint p = tid; p < num_cached; p += tg_sz) {
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
                half gate = (half)(1.0 / (1.0 + exp(-(float)head_gates[qh])));
                for (uint d = tid; d < h_dim; d += tg_sz) {
                    float acc = 0.0;
                    for (uint p = 0; p < num_cached; ++p) {
                        float s = (float)slot_logits[p] * inv_s;
                        acc += s * (float)kv_scratch_v[p * scratch_stride + kv_h * GLOBAL_HEAD_DIM + d];
                    }
                    uint write_pos = qh * h_dim + d;
                    if (write_pos < HIDDEN_DIM)
                        n_buf[write_pos] = (half)((float)acc * (float)gate);
                }
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

        }  // end if (!kv_prefetched)

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

        // ── Enqueue KV prefetch request for next layer ──
        if (layer + 1 < LAYERS) {
            device KvScratchHeader* next_header = (active_set == 0) ? scratch_set_b : scratch_set_a;
            uint next_state = atomic_load_explicit(&next_header->control.state, memory_order_acquire, memory_scope_device);
            if (next_state == KV_STATE_EMPTY) {
                KvPrefetchRequest req;
                req.request_id = layer;
                req.session_id = current_token;
                req.target_layer = layer + 1;
                req.token_epoch = current_pos;
                req.scratch_set_index = 1 - active_set;
                uint enq = atomic_fetch_add_explicit(&kv_prefetch_queue->enqueue_pos.value, 1, memory_order_acq_rel, memory_scope_device);
                uint deq = atomic_load_explicit(&kv_prefetch_queue->dequeue_pos.value, memory_order_acquire, memory_scope_device);
                if (enq - deq < kv_prefetch_queue->capacity) {
                    uint idx = enq & kv_prefetch_queue->mask;
                    kv_prefetch_queue->entries[idx] = req;
                    atomic_store_explicit(&next_header->control.state, KV_STATE_QUEUED, memory_order_release, memory_scope_device);
                } else {
                    atomic_store_explicit(&kv_prefetch_queue->enqueue_pos.value, enq, memory_order_release, memory_scope_device);
                }
            }
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
    for (uint c = tid; c < NUM_CENTROIDS; c += tg_sz) {
        float score = 0.0;
        for (uint d = 0; d < HIDDEN_DIM; ++d) {
            score += (float)h_buf[d] * (float)centroid_scratch[c * HIDDEN_DIM + d];
        }
        centroid_scores[c] = score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

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
    for (uint row = tid; row < VOCAB_SIZE; row += tg_sz) {
        if (row < cstart || row >= cend) {
            slot_logits[row] = as_type<half>((unsigned short)0xFC00u);
        }
    }
    threadgroup_barrier(mem_flags::mem_device);

    // ── MTP: Multi-Token Prediction heads ─────────────────────
    {
        uint mtp_w_base = 0;
        uint per_head = MTP_HIDDEN * HID_TILES * LANES + HIDDEN_DIM * MTP_TILES * LANES;

        for (uint mtp_head = 0; mtp_head < NUM_MTP_HEADS; ++mtp_head) {
            device half* mtp_scratch = slot_logits + (mtp_head + 1) * VOCAB_SIZE;
            uint head_w_off = mtp_w_base + mtp_head * per_head;

            for (uint i = tid; i < HIDDEN_DIM; i += tg_sz) mtp_scratch[i] = h_buf[i];
            threadgroup_barrier(mem_flags::mem_device);

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

            for (uint i = tid; i < MTP_TILES * TILE; i += tg_sz) {
                n_buf[i] = (i < MTP_HIDDEN) ? up_out[i] : (half)0;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

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

            for (uint i = tid; i < HIDDEN_DIM; i += tg_sz) h_buf[i] += mtp_scratch[i];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint i = tid; i < HIDDEN_DIM; i += tg_sz) n_buf[i] = h_buf[i];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            fast_rmsnorm(n_buf, norms + 0, tid, tg_sz, shared_sums);
            for (uint i = tid; i < HIDDEN_DIM; i += tg_sz) h_buf[i] = n_buf[i];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            device half* head_logits = slot_logits + (mtp_head + 1) * VOCAB_SIZE;

            for (uint c = tid; c < NUM_CENTROIDS; c += tg_sz) {
                float score = 0.0;
                for (uint d = 0; d < HIDDEN_DIM; ++d) {
                    score += (float)h_buf[d] * (float)centroid_scratch[c * HIDDEN_DIM + d];
                }
                centroid_scores[c] = score;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

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
        uint draft_kv_stride = DRAFT_NUM_KV_HEADS * DRAFT_HEAD_DIM;
        uint draft_cache_pos = 0u;

        for (uint layer = 0; layer < DRAFT_LAYERS; ++layer) {
            uint h_dim = DRAFT_HEAD_DIM;
            uint layer_base = layer * DRAFT_LAYER_STRIDE;
            uint scratch_layer_base = layer * MAX_CTX * draft_kv_stride;

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

            uint kw_base = layer_base + DRAFT_K_OFF;
            uint vw_base = layer_base + DRAFT_V_OFF;
            uint qw_base = layer_base + DRAFT_Q_OFF;
            uint ow_base = layer_base + DRAFT_O_OFF;

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

            for (uint kv_h = 0; kv_h < DRAFT_NUM_KV_HEADS; ++kv_h) {
                for (uint i = tid; i < h_dim; i += tg_sz) {
                    kv_scratch_k[scratch_layer_base + draft_cache_pos * draft_kv_stride + kv_h * h_dim + i] =
                        n_buf[DRAFT_HIDDEN + kv_h * h_dim + i];
                    kv_scratch_v[scratch_layer_base + draft_cache_pos * draft_kv_stride + kv_h * h_dim + i] =
                        n_buf[DRAFT_HIDDEN + DRAFT_KV_TILES * TILE + kv_h * h_dim + i];
                }
            }
            threadgroup_barrier(mem_flags::mem_device);

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

            apply_rope(q_chunk, DRAFT_NUM_HEADS, h_dim, draft_cache_pos, tid, tg_sz);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint i = tid; i < DRAFT_HIDDEN; i += tg_sz) {
                n_buf[DRAFT_HIDDEN + 2 * h_dim + i] = 0;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint kv_h = 0; kv_h < DRAFT_NUM_KV_HEADS; ++kv_h) {
                for (uint q_pair = 0; q_pair < 2; ++q_pair) {
                    uint qh = 2 * kv_h + q_pair;
                    threadgroup half* q_head = q_chunk + qh * h_dim;

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

            for (uint i = tid; i < DRAFT_HIDDEN; i += tg_sz) {
                n_buf[i] = n_buf[DRAFT_HIDDEN + 2 * h_dim + i];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

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
        }

        // ── After all layers: output projection to vocab via centroid scout ──
        for (uint c = tid; c < NUM_CENTROIDS; c += tg_sz) {
            float score = 0.0;
            for (uint d = 0; d < DRAFT_HIDDEN; ++d) {
                score += (float)h_buf[d] * (float)centroid_scratch[c * HIDDEN_DIM + d];
            }
            centroid_scores[c] = score;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

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

        if (tid == 0) {
            draft_output[0] = 5;
            for (uint i = 0; i < 5; ++i) {
                draft_output[1 + i] = top5_ids[i];
                half logprob = (half)top5_vals[i];
                draft_output[6 + i] = (uint)as_type<ushort>(logprob);
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
    }  // end else if (kind == 3)

                // --- After decode: signal COMPLETED -------------------
                threadgroup_barrier(mem_flags::mem_device);
                atomic_store_explicit(
                    (device atomic_uint*)(ring_entries + idx_ring * 4), 3 | (kind << 2),
                    memory_order_relaxed);  // COMPLETED
                atomic_fetch_add_explicit(completion_counter, 1, memory_order_relaxed);
                // --- Record outcome ---
                atomic_fetch_add_explicit(&receipt->requests_ready_consumable, 1, memory_order_relaxed, memory_scope_device);
                // --- Local bypass: adjust enqueue limit for bypassed requests ---
                // (Handled by the epoch controller on the host; the worker GPU-local
                //  bookkeeping is complete.)

                ++token_count;
    }  // end while (token_count < max_tokens_per_epoch)
}


// ── Generation-Tagged Page Oracle ──────────────────────────────
// Writes a KV page with generation-tagged oracle using atomic_ulong
// packed [epoch:32|owner:32].  The COMMITTED_BIT (bit 63) marks the
// entry as definitively published.

#define COMMITTED_BIT (1ull << 63)

/// Atomically write a generation-tagged page descriptor.
///   page_table[t] holds atomic_ulong packed [committed|epoch|owner].
///   epoch_tag = packed (generation << 32) | worker_id.
///   After the payload is fully written, CAS the tag to set COMMITTED_BIT.
void write_kv_page_with_oracle(
    device atomic_ulong* page_table,
    uint page_index,
    uint generation,
    uint worker_id,
    device const half* payload_source,
    device half* payload_dest,
    uint payload_elements,
    uint tid,
    uint tg_sz)
{
    // 1. Write the payload data (scratch -> kv cache)
    for (uint i = tid; i < payload_elements; i += tg_sz) {
        payload_dest[i] = payload_source[i];
    }
    threadgroup_barrier(mem_flags::mem_device);

    // 2. Build the oracle tag: [committed=0 | generation(32) | worker_id(32)]
    ulong tag = ((ulong)generation << 32) | (ulong)worker_id;
    ulong committed_tag = tag | COMMITTED_BIT;

    // 3. CAS from 0 -> uncommitted tag (reserve)
    if (tid == 0) {
        ulong expected = 0;
        bool reserved = atomic_compare_exchange_weak_explicit(
            &page_table[page_index], &expected, tag,
            memory_order_acq_rel, memory_order_acquire,
            memory_scope_device, memory_scope_device);
        if (reserved) {
            // 4. Full commit barrier (payload already visible from step 1)
            //    Set COMMITTED_BIT to signal durability.
            atomic_store_explicit(&page_table[page_index], committed_tag,
                memory_order_release, memory_scope_device);
        } else {
            // Another worker already owns this slot; check if committed.
            ulong existing = expected;
            if (existing & COMMITTED_BIT) {
                // Doubled write detected — already committed by someone else.
                // (Epoch controller handles this as a duplicate-write event.)
            } else {
                // Another worker holds the reservation — the epoch controller
                // will fence this via generation mismatch detection.
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
}

// ---- KV Prefetch Worker -----------------------------------------------

kernel void persistent_kv_prefetch_worker(
    device const uint*    kv_k_nibbles  [[buffer(1)]],
    device const uint*    kv_v_nibbles  [[buffer(2)]],
    device const half*    kv_k_scales   [[buffer(3)]],
    device const half*    kv_v_scales   [[buffer(4)]],
    device half*          scratch_k     [[buffer(5)]],
    device half*          scratch_v     [[buffer(6)]],
    device KvScratchHeader* headers    [[buffer(7)]],
    constant uint&        slot_offset   [[buffer(8)]],
    constant uint&        max_positions [[buffer(9)]],
    constant uint&        max_tokens_per_epoch [[buffer(10)]],
    device EpochControl*  epoch_control [[buffer(11)]],
    uint tid    [[thread_index_in_threadgroup]],
    uint tg_sz  [[threads_per_threadgroup]])
{
    uint simd_group = tid / 32;
    uint lane = tid & 31;

    while (true) {
        // ── Exit conditions ──
        uint epoch_close = atomic_load_explicit(
            &epoch_control->epoch_close_requested, memory_order_acquire,
            memory_scope_device);
        uint epoch_limit = atomic_load_explicit(
            &epoch_control->epoch_enqueue_limit, memory_order_acquire,
            memory_scope_device);
        // (Exit is checked after each page group — fall through to poll)

        // 1. Poll for work
        uint deq = atomic_load_explicit(
            &epoch_control->epoch_fatal_claim, memory_order_acquire,
            memory_scope_device);
        if (deq != CLAIM_UNOWNED) {
            // ── Helper terminal cleanup: a fatal fault was recorded ──
            // Invalidate any in-flight scratch sets and bail.
            for (uint i = 0; i < 16; ++i) {
                atomic_store_explicit(&headers[i].control.payload_valid, 0,
                    memory_order_release, memory_scope_device);
                atomic_store_explicit(&headers[i].control.state, KV_STATE_RECLAIMABLE,
                    memory_order_release, memory_scope_device);
                atomic_store_explicit(&headers[i].control.producer_claim, CLAIM_UNOWNED,
                    memory_order_release, memory_scope_device);
            }
            return;
        }

        // ── Dequeue prefetch requests from the shared work queue ──
        // (Reuse KvPrefetchRequest ring from slot/kv_interleave)
        // In the FINAL Phase 1 design, prefetch requests are embedded in the
        // KvScratchHeader ring itself. We poll scratch headers directly.

        uint scratch_idx = 0;
        bool found = false;
        for (uint i = 0; i < 16; ++i) {
            uint st = atomic_load_explicit(
                &headers[i].control.state, memory_order_acquire,
                memory_scope_device);
            if (st == KV_STATE_QUEUED) {
                uint expected = KV_STATE_QUEUED;
                if (atomic_compare_exchange_weak_explicit(
                    &headers[i].control.state, &expected, KV_STATE_FILLING,
                    memory_order_acquire, memory_order_acquire,
                    memory_scope_device, memory_scope_device)) {
                    scratch_idx = i;
                    found = true;
                    break;
                }
            }
        }

        if (!found) {
            // ── Check exit condition ──
            if (epoch_close != 0) {
                return;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            continue;
        }

        device KvScratchHeader* hdr = headers + scratch_idx;
        uint target_layer = hdr->metadata.target_layer;
        uint token_epoch = hdr->metadata.token_epoch;
        uint kv_generation = hdr->metadata.kv_generation;
        uint data_offset = hdr->metadata.data_offset;

        // 2. Decompress historical KV for target_layer (all positions)
        uint num_cached = min(max_positions, MAX_CTX);
        uint h_dim = HEAD_DIM;
        uint scratch_stride = NUM_KV_HEADS * GLOBAL_HEAD_DIM;
        uint blocks_per_head = (h_dim + 255) / 256;
        device half* local_k = scratch_k + scratch_idx * MAX_CTX * scratch_stride;
        device half* local_v = scratch_v + scratch_idx * MAX_CTX * scratch_stride;

        for (uint i = tid; i < MAX_CTX * scratch_stride; i += tg_sz) {
            local_k[i] = 0;
            local_v[i] = 0;
        }
        threadgroup_barrier(mem_flags::mem_device);

        for (uint p = 0; p < num_cached; ++p) {
            uint h = simd_group;
            if (h < NUM_KV_HEADS) {
                for (uint b = 0; b < blocks_per_head; ++b) {
                    uint block_idx = p * scratch_stride + h * GLOBAL_HEAD_DIM;
                    block_idx = (block_idx + b * KV_BLOCK) / KV_BLOCK;
                    uint nibble_idx = block_idx * KV_NIBBLES_U32;

                    uint t_limit = KV_NIBBLES_U32;
                    uint max_dim = b * KV_BLOCK + t_limit * PER_LANE;
                    if (max_dim > h_dim) {
                        t_limit = (h_dim - b * KV_BLOCK + PER_LANE - 1) / PER_LANE;
                    }

                    half scale_k = kv_k_scales[block_idx];
                    for (uint t = lane; t < t_limit; t += 32) {
                        uint val = kv_k_nibbles[nibble_idx + t];
                        uint dim_base = b * KV_BLOCK + t * PER_LANE;
                        uint rem_el = PER_LANE;
                        if (dim_base + rem_el > h_dim) rem_el = h_dim - dim_base;
                        for (uint i = 0; i < rem_el; ++i) {
                            uint rem = fast_mod3(val);
                            if (rem != 1) {
                                uint scratch_pos = p * scratch_stride + h * GLOBAL_HEAD_DIM + dim_base + i;
                                local_k[scratch_pos] = (half)((int)(rem - 1) * (float)scale_k);
                            }
                            val = fast_div3(val);
                        }
                    }

                    half scale_v = kv_v_scales[block_idx];
                    for (uint t = lane; t < t_limit; t += 32) {
                        uint val = kv_v_nibbles[nibble_idx + t];
                        uint dim_base = b * KV_BLOCK + t * PER_LANE;
                        uint rem_el = PER_LANE;
                        if (dim_base + rem_el > h_dim) rem_el = h_dim - dim_base;
                        for (uint i = 0; i < rem_el; ++i) {
                            uint rem = fast_mod3(val);
                            if (rem != 1) {
                                uint scratch_pos = p * scratch_stride + h * GLOBAL_HEAD_DIM + dim_base + i;
                                local_v[scratch_pos] = (half)((int)(rem - 1) * (float)scale_v);
                            }
                            val = fast_div3(val);
                        }
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_device);

        // ── Cooperative cancellation: check after each page group ──
        uint cancel = atomic_load_explicit(
            &hdr->control.cancel_requested, memory_order_acquire,
            memory_scope_device);
        if (cancel != 0) {
            // CAS OUTCOME_NONE -> CANCELED
            uint expected_outcome = OUTCOME_NONE;
            uint outcome_won = atomic_compare_exchange_weak_explicit(
                &hdr->control.request_outcome, &expected_outcome, OUTCOME_CANCELED,
                memory_order_acq_rel, memory_order_acquire,
                memory_scope_device, memory_scope_device);
            if (outcome_won) {
                // Invalidate payload and mark canceled
                atomic_store_explicit(&hdr->control.payload_valid, 0,
                    memory_order_release, memory_scope_device);
                atomic_store_explicit(&hdr->control.state, KV_STATE_CANCELED,
                    memory_order_release, memory_scope_device);
                atomic_fetch_add_explicit(&epoch_control->epoch_fatal_fault_request_id, 1,
                    memory_order_relaxed, memory_scope_device); // diagnostic
            }  // else: preserve winner's outcome
            continue;
        }

        // ── Check exit condition between page groups ──
        uint et = atomic_load_explicit(&epoch_control->epoch_close_requested,
            memory_order_acquire, memory_scope_device);
        uint el = atomic_load_explicit(&epoch_control->epoch_enqueue_limit,
            memory_order_acquire, memory_scope_device);
        if (et != 0) {
            return;
        }

        // ── Ordered Readiness Publication ──
        // 1. Payload is already in scratch — set payload_valid=1 (release)
        atomic_store_explicit(&hdr->control.payload_valid, 1,
            memory_order_release, memory_scope_device);

        // 2. state = Ready (release) — makes it visible to consumers
        atomic_store_explicit(&hdr->control.state, KV_STATE_READY,
            memory_order_release, memory_scope_device);

        // 3. CAS request_outcome None -> ReadyConsumable
        uint expected_outcome = OUTCOME_NONE;
        uint outcome_won = atomic_compare_exchange_weak_explicit(
            &hdr->control.request_outcome, &expected_outcome, OUTCOME_READY_CONSUMABLE,
            memory_order_acq_rel, memory_order_acquire,
            memory_scope_device, memory_scope_device);

        if (outcome_won) {
            // Case A: we won the CAS — increment ready consumable counter
            atomic_fetch_add_explicit(&epoch_control->epoch_fatal_claim, 1,
                memory_order_relaxed, memory_scope_device);
        } else {
            // Case B: we lost — some other agent already set the outcome.
            // Invalidate our payload publication:
            //   - payload_valid = 0
            //   - state = Reclaimable
            //   - increment late_ready_discarded
            atomic_store_explicit(&hdr->control.payload_valid, 0,
                memory_order_release, memory_scope_device);
            atomic_store_explicit(&hdr->control.state, KV_STATE_RECLAIMABLE,
                memory_order_release, memory_scope_device);
            atomic_store_explicit(&hdr->control.producer_claim, CLAIM_UNOWNED,
                memory_order_release, memory_scope_device);
            uint discarded = atomic_fetch_add_explicit(
                &hdr->control.late_publish_rejection_count, 1,
                memory_order_relaxed, memory_scope_device);

            // Also increment epoch diagnostic
            uint diag = atomic_load_explicit(
                &headers[0].control.duplicate_write_detected,
                memory_order_relaxed, memory_scope_device);
            atomic_store_explicit(&headers[0].control.duplicate_write_detected,
                diag + 1, memory_order_relaxed, memory_scope_device);
        }

        // ── Exit on epoch close condition ──
        epoch_close = atomic_load_explicit(&epoch_control->epoch_close_requested,
            memory_order_acquire, memory_scope_device);
        if (epoch_close != 0) {
            return;
        }
    }
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
    device const half*    head_gates        [[buffer(29)]],  // per-head attention gates (NUM_Q_HEADS × f16)
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
        // SIMD-group-parallel KV decompression: each SIMD group (32 lanes)
        // handles one KV head.  32-lane cooperation preserves coalesced
        // memory access.  0 barriers between heads — 1 final barrier.
        uint simd_group = tid / 32;
        uint lane = tid & 31;
        for (uint p = 0; p < num_cached; ++p) {
            uint h = simd_group;
            if (h < NUM_KV_HEADS) {
                for (uint b = 0; b < blocks_per_head; ++b) {
                    uint pos_head_vals = slot_kv_offset + layer * MAX_CTX * scratch_stride
                                       + p * scratch_stride + h * GLOBAL_HEAD_DIM;
                    uint val_offset = pos_head_vals + b * KV_BLOCK;
                    uint block_idx = val_offset / KV_BLOCK;
                    uint byte_offset = block_idx * bytes_per_kv_block;

                    // Clamp byte range to valid dimensions
                    uint max_bytes = bytes_per_kv_block;
                    if (b * KV_BLOCK + bytes_per_kv_block * 5u > h_dim) {
                        max_bytes = (h_dim - b * KV_BLOCK + 4u) / 5u;
                    }

                    // Decompress K block — 32 lanes cooperate (coalesced access)
                    half scale_k = kv_k_scales[block_idx];
                    for (uint t = lane; t < max_bytes; t += 32) {
                        uchar packed = ((device uchar*)kv_k_nibbles)[byte_offset + t];
                        uint dim_base = b * KV_BLOCK + t * 5u;
                        uint v = (uint)packed;
                        uint rem_el = 5u;
                        if (dim_base + rem_el > h_dim) rem_el = h_dim - dim_base;
                        for (uint i = 0; i < rem_el; ++i) {
                            uint rem = v % 3u;
                            if (rem != 1) {
                                uint scratch_pos = p * scratch_stride + h * GLOBAL_HEAD_DIM + dim_base + i;
                                kv_scratch_k[scratch_pos] = (half)((int)(rem - 1) * (float)scale_k);
                            }
                            v /= 3u;
                        }
                    }

                    // Decompress V block — same SIMD group
                    half scale_v = kv_v_scales[block_idx];
                    for (uint t = lane; t < max_bytes; t += 32) {
                        uchar packed = ((device uchar*)kv_v_nibbles)[byte_offset + t];
                        uint dim_base = b * KV_BLOCK + t * 5u;
                        uint v = (uint)packed;
                        uint rem_el = 5u;
                        if (dim_base + rem_el > h_dim) rem_el = h_dim - dim_base;
                        for (uint i = 0; i < rem_el; ++i) {
                            uint rem = v % 3u;
                            if (rem != 1) {
                                uint scratch_pos = p * scratch_stride + h * GLOBAL_HEAD_DIM + dim_base + i;
                                kv_scratch_v[scratch_pos] = (half)((int)(rem - 1) * (float)scale_v);
                            }
                            v /= 3u;
                        }
                    }
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

                half gate = (half)(1.0 / (1.0 + exp(-(float)head_gates[qh])));
                // Pass 3: weighted sum of V (gated by per-head attention gate)
                for (uint d = tid; d < h_dim; d += tg_sz) {
                    float acc = 0.0;
                    for (uint p = 0; p < num_cached; ++p) {
                        float s = (float)slot_logits[p] * inv_s;
                        acc += s * (float)kv_scratch_v[p * scratch_stride + kv_h * GLOBAL_HEAD_DIM + d];
                    }
                    uint write_pos = qh * h_dim + d;
                    if (write_pos < HIDDEN_DIM)
                        n_buf[write_pos] = (half)((float)acc * (float)gate);
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
                draft_output[6 + i] = (uint)as_type<ushort>(logprob);
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

/// T32 coalesced uint4 GEMV production kernel.
/// 4 rows per TG, 128 threads (4 SIMD groups × 32 lanes).
/// Activation loaded once into SRAM, shared across all 4 rows.
/// Weights read via uint4 vector loads (32 threads read same block, broadcast).
/// Per-lane `/3` and `%3` trit extraction (no magic-division overflow).
/// SRAM-based reduction across SIMD group (no simd_sum issues).
pub const PERSISTENT_GEMV_SRC: &str = r##"#include <metal_stdlib>
using namespace metal;

constant uint HIDDEN_DIM  = 3840;
constant uint BLOCKS_PER_ROW = HIDDEN_DIM / 32; // 120

kernel void matvec_persistent_t32_coalesced(
    device const uint4*  weight_stream  [[buffer(0)]],
    device const half*   activation     [[buffer(1)]],
    device half*         output         [[buffer(2)]],
    uint ti                               [[thread_index_in_threadgroup]],
    uint tp                               [[threadgroup_position_in_grid]])
{
    // Cooperative load of all activations into SRAM
    threadgroup half sram[HIDDEN_DIM];
    for (uint i = ti; i < HIDDEN_DIM; i += 32) {
        sram[i] = activation[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint row = tp;
    const uint lane = ti % 32;
    float my_acc = 0.0f;
    uint row_base = row * BLOCKS_PER_ROW;
    half unpacked[32];
    half act_reg[32];

    for (uint bg = 0; bg < 128; bg += 32) {
        uint g = bg / 32;

        // Copy this iteration's 32 activation elements into registers
        uint act_base = g * 1024 + lane * 32;
        for (uint e = 0; e < 32; ++e) {
            act_reg[e] = sram[act_base + e];
        }

        uint logical_block = bg + lane;
        bool is_valid = logical_block < BLOCKS_PER_ROW;
        uint safe_block = is_valid ? logical_block : (BLOCKS_PER_ROW - 1);

        uint4 vec = weight_stream[row_base + safe_block];
        thread const uchar* raw = (thread const uchar*)&vec;
        ushort scale_bits = ((ushort)raw[7]) | ((ushort)raw[8] << 8);
        half scale = as_type<half>(scale_bits);

        // Full 32-element unpack from block
        for (uint i = 0; i < 7; ++i) {
            uchar bv = raw[i];
            uint v = (uint)bv;
            uint n = (i < 6) ? 5 : 2;
            for (uint j = 0; j < n; ++j) {
                unpacked[i * 5 + j] = (half)((int)(v % 3) - 1);
                v = v / 3;
            }
        }

        // Dot product from registers (zero SRAM bank conflicts)
        float local_sum = 0.0f;
        for (uint e = 0; e < 32; ++e) {
            local_sum += (float)(unpacked[e] * scale * act_reg[e]);
        }
        my_acc += is_valid ? local_sum : 0.0f;
    }

    float total = simd_sum(my_acc);
    if (lane == 0) {
        output[row] = (half)total;
    }
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
    cmd.args(["-sdk", "macosx", "metal", "-std=metal4.0", "-O3", "-c"]);
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

/// Extract a named function from an already-loaded library and return its pipeline state.
pub fn compile_function_from_lib(
    device: &Device,
    library: &LibraryRef,
    name: &str,
) -> Result<ComputePipelineState, String> {
    let function = library
        .get_function(name, None)
        .map_err(|e| format!("get_function({:?}): {:?}", name, e))?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| format!("pipeline state for {:?}: {:?}", name, e))
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
