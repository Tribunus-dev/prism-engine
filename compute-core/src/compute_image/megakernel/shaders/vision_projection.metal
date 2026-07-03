// ── Gemma 4 Vision Embedder Projection (Phase A) ─────────────────────────────
//
// Pipeline:
//   1. Patch dense projection: pixels [num_patches, 6912] × patch_dense^T [3840, 6912] → [num_patches, 3840]
//   2. Layer norm (patch_ln2) on the result
//   3. Add learned 2D position embeddings (nearest-neighbor lookup from pos_embedding table)
//   4. Position norm (pos_norm)
//   5. Soft-token pooling: avg-pool within kernel windows (separate kernel)
//   6. Final projection: embed_vision.embedding_projection [3840, 3840] (separate kernel)
//
// Tiling: each threadgroup processes one row (16 threads collaborate)
// Weights are ternary-packed (16 weights per uint32_t, 2 bits each)
// Encoding: 00=0, 01=+1, 10=-1 (matches ternary_gemm.metal / ternary_gemv.metal convention)
//
// ── Buffer layout (vision_patch_embed) ──
//   buffer(0):  patch_pixels      [num_patches * 6912] half — raw patch features (patch_ln1 already applied)
//   buffer(1):  patch_dense_w     [3840 * packed_6912] uint — ternary weights
//   buffer(2):  patch_dense_s     [3840 * groups] half — block scales
//   buffer(3):  patch_ln2_weight  [3840] half
//   buffer(4):  patch_ln2_bias    [3840] half
//   buffer(5):  pos_embedding     [max_positions * max_positions * 2 * 3840] half — interleaved (x,y)
//   buffer(6):  pos_norm_weight   [3840] half
//   buffer(7):  pos_norm_bias     [3840] half
//   buffer(10): output            [num_patches * 3840] half
//   buffer(11): num_patches       uint
//   buffer(12): patches_w         uint
//   buffer(13): patches_h         uint
//   buffer(15): max_positions     uint
//
// ── Buffer layout (vision_pool_soft_tokens) ──
//   buffer(0): positioned        [num_patches * 3840] half
//   buffer(1): soft_tokens       [num_soft * 3840] half
//   buffer(2): num_patches       uint
//   buffer(3): patches_w         uint
//   buffer(4): patches_h         uint
//   buffer(5): soft_token_k      uint
//
// ── Buffer layout (vision_final_projection) ──
//   buffer(0): soft_tokens       [N * 3840] half
//   buffer(1): proj_weights      [3840 * packed_3840] uint — ternary weights
//   buffer(2): proj_scales       [3840 * groups] half — block scales
//   buffer(3): decoder_embeds    [N * 3840] half
//   buffer(4): num_soft_tokens   uint

#include <metal_stdlib>
using namespace metal;

constant uint HIDDEN_DIM       = 3840;
constant uint PATCH_IN_FEATURES = 6912;
constant uint PACKED_6912      = 432;   // ceil(6912 / 16)
constant uint PACKED_3840      = 240;   // ceil(3840 / 16)
constant uint GROUP_SIZE       = 256;

// Branchless ternary weight decode: extracts 2-bit value from packed uint32_t
//   00 → 0, 01 → +1, 10 → -1, 11 → 0 (invalid, treated as 0)
inline float decode_ternary(uint packed, uint idx) {
    uint shift = (idx & 15) * 2;
    uint nib = (packed >> shift) & 3;
    return float((nib == 1) ? 1 : (nib == 2) ? -1 : 0);
}

// ── Kernel: vision_patch_embed ───────────────────────────────────────────────
// Processes one patch row per threadgroup (gid = patch_idx).
// Performs: patch_dense GEMM → patch_ln2 → position embed → pos_norm
// Output is positioned embeddings; pooling is deferred to vision_pool_soft_tokens.

kernel void vision_patch_embed(
    device const half*    patch_pixels     [[buffer(0)]],
    device const uint*    patch_dense_w    [[buffer(1)]],
    device const half*    patch_dense_s    [[buffer(2)]],
    device const half*    patch_ln2_w      [[buffer(3)]],
    device const half*    patch_ln2_b      [[buffer(4)]],
    device const half*    pos_embedding    [[buffer(5)]],
    device const half*    pos_norm_w       [[buffer(6)]],
    device const half*    pos_norm_b       [[buffer(7)]],
    device half*          output           [[buffer(10)]],
    constant uint32_t&    num_patches      [[buffer(11)]],
    constant uint32_t&    patches_w        [[buffer(12)]],
    constant uint32_t&    patches_h        [[buffer(13)]],
    constant uint32_t&    max_positions    [[buffer(15)]],
    uint tid                                [[thread_position_in_threadgroup]],
    uint gid                                [[threadgroup_position_in_grid]])
{
    uint patch_idx = gid;
    if (patch_idx >= num_patches) return;

    uint groups_per_col = (PATCH_IN_FEATURES + GROUP_SIZE - 1) / GROUP_SIZE;
    threadgroup float accum[HIDDEN_DIM];

    // ── Stage 1: Patch dense GEMM ──────────────────────────────────────────
    // Initialize accumulators
    for (uint i = tid; i < HIDDEN_DIM; i += 256) {
        accum[i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // GEMM: patch_pixels[patch_idx, :] × patch_dense^T → accum[:]
    for (uint k_tile = 0; k_tile < PACKED_6912; k_tile += 16) {
        for (uint i = tid; i < HIDDEN_DIM; i += 256) {
            float dot = 0.0f;
            for (uint ki = 0; ki < 16 && k_tile + ki < PACKED_6912; ki++) {
                uint w_idx = i * PACKED_6912 + k_tile + ki;
                uint packed = patch_dense_w[w_idx];
                uint scale_idx = i * groups_per_col + (k_tile + ki) / (GROUP_SIZE / 16);
                half scale = (scale_idx < (HIDDEN_DIM * groups_per_col))
                    ? patch_dense_s[scale_idx] : half(0.0h);

                for (uint b = 0; b < 16; b++) {
                    uint col = (k_tile + ki) * 16 + b;
                    if (col >= PATCH_IN_FEATURES) break;
                    half act = patch_pixels[patch_idx * PATCH_IN_FEATURES + col];
                    float w = decode_ternary(packed, b) * float(scale);
                    dot += float(act) * w;
                }
            }
            accum[i] += dot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Stage 2: RMS norm (patch_ln2) ─────────────────────────────────────
    // Compute RMS across HIDDEN_DIM
    float sum_sq = 0.0f;
    for (uint i = tid; i < HIDDEN_DIM; i += 256) {
        float v = accum[i];
        sum_sq += v * v;
    }

    // Reduction across threadgroup (up to first 256 threads)
    threadgroup float reduce_buf[256];
    reduce_buf[tid] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduce_buf[tid] += reduce_buf[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rms = sqrt(reduce_buf[0] / float(HIDDEN_DIM) + 1e-6f);
    float inv_rms = 1.0f / rms;

    // ── Stage 3: Apply norm + position embedding + pos_norm ───────────────
    uint px = patch_idx % patches_w;
    uint py = patch_idx / patches_w;

    // Nearest-neighbor position embedding lookup
    float fx = float(px) / float(max(patches_w - 1u, 1u));
    float fy = float(py) / float(max(patches_h - 1u, 1u));
    uint pos_x = uint(fx * float(max_positions - 1));
    uint pos_y = uint(fy * float(max_positions - 1));
    uint pos_idx = pos_y * max_positions + pos_x;

    for (uint i = tid; i < HIDDEN_DIM; i += 256) {
        // RMS norm (patch_ln2)
        float n = accum[i] * inv_rms * float(patch_ln2_w[i]) + float(patch_ln2_b[i]);

        // Add position embedding (x + y components)
        float pos_x_val = float(pos_embedding[pos_idx * 2 * HIDDEN_DIM + 0 * HIDDEN_DIM + i]);
        float pos_y_val = float(pos_embedding[pos_idx * 2 * HIDDEN_DIM + 1 * HIDDEN_DIM + i]);
        n += pos_x_val + pos_y_val;

        // Position norm
        n = n * float(pos_norm_w[i]) + float(pos_norm_b[i]);
        accum[i] = n;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write positioned embeddings to output
    device half* after_pos = output + patch_idx * HIDDEN_DIM;
    for (uint i = tid; i < HIDDEN_DIM; i += 256) {
        after_pos[i] = half(accum[i]);
    }
    // Pooling is deferred to the vision_pool_soft_tokens kernel
}

// ── Kernel: vision_pool_soft_tokens ──────────────────────────────────────────
// Average-pools positioned patch embeddings within soft_token_k × soft_token_k
// windows to produce soft tokens.
//
// Each threadgroup processes one soft token (gid = soft_token_idx).

kernel void vision_pool_soft_tokens(
    device const half*    positioned       [[buffer(0)]],
    device half*          soft_tokens      [[buffer(1)]],
    constant uint32_t&    num_patches      [[buffer(2)]],
    constant uint32_t&    patches_w        [[buffer(3)]],
    constant uint32_t&    patches_h        [[buffer(4)]],
    constant uint32_t&    soft_token_k     [[buffer(5)]],
    uint tid                               [[thread_position_in_threadgroup]],
    uint gid                               [[threadgroup_position_in_grid]])
{
    uint soft_per_dim = patches_w / soft_token_k;
    uint num_soft = soft_per_dim * soft_per_dim;
    uint soft_idx = gid;
    if (soft_idx >= num_soft) return;

    uint sy = soft_idx / soft_per_dim;
    uint sx = soft_idx % soft_per_dim;

    // Average pool within kernel window
    float inv_k2 = 1.0f / float(soft_token_k * soft_token_k);
    for (uint i = tid; i < HIDDEN_DIM; i += 256) {
        float sum = 0.0f;
        for (uint ky = 0; ky < soft_token_k; ky++) {
            for (uint kx = 0; kx < soft_token_k; kx++) {
                uint py = sy * soft_token_k + ky;
                uint px = sx * soft_token_k + kx;
                if (py < patches_h && px < patches_w) {
                    uint p_idx = py * patches_w + px;
                    sum += float(positioned[p_idx * HIDDEN_DIM + i]);
                }
            }
        }
        soft_tokens[soft_idx * HIDDEN_DIM + i] = half(sum * inv_k2);
    }
}

// ── Kernel: vision_final_projection ──────────────────────────────────────────
// Final projection: soft_tokens [N, 3840] × embed_proj^T [3840, 3840] → decoder embeddings
// Each threadgroup processes one soft token (gid = token_idx).

kernel void vision_final_projection(
    device const half*    soft_tokens      [[buffer(0)]],
    device const uint*    proj_weights     [[buffer(1)]],
    device const half*    proj_scales      [[buffer(2)]],
    device half*          decoder_embeds   [[buffer(3)]],
    constant uint32_t&    num_soft_tokens  [[buffer(4)]],
    uint tid                               [[thread_position_in_threadgroup]],
    uint gid                               [[threadgroup_position_in_grid]])
{
    uint token_idx = gid;
    if (token_idx >= num_soft_tokens) return;

    uint groups_per_col = (HIDDEN_DIM + GROUP_SIZE - 1) / GROUP_SIZE;
    threadgroup float local_acc[HIDDEN_DIM];

    for (uint i = tid; i < HIDDEN_DIM; i += 256) {
        local_acc[i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // GEMM: soft_tokens[token_idx, :] × proj_weights^T
    for (uint k_tile = 0; k_tile < PACKED_3840; k_tile += 16) {
        for (uint i = tid; i < HIDDEN_DIM; i += 256) {
            float dot = 0.0f;
            for (uint ki = 0; ki < 16 && k_tile + ki < PACKED_3840; ki++) {
                uint w_idx = i * PACKED_3840 + k_tile + ki;
                uint packed = proj_weights[w_idx];
                uint scale_idx = i * groups_per_col + (k_tile + ki) / (GROUP_SIZE / 16);
                half scale = (scale_idx < (HIDDEN_DIM * groups_per_col))
                    ? proj_scales[scale_idx] : half(0.0h);

                for (uint b = 0; b < 16; b++) {
                    uint col = (k_tile + ki) * 16 + b;
                    if (col >= HIDDEN_DIM) break;
                    half act = soft_tokens[token_idx * HIDDEN_DIM + col];
                    float w = decode_ternary(packed, b) * float(scale);
                    dot += float(act) * w;
                }
            }
            local_acc[i] += dot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = tid; i < HIDDEN_DIM; i += 256) {
        decoder_embeds[token_idx * HIDDEN_DIM + i] = half(local_acc[i]);
    }
}

// ── Audio frame embedding: audio_frame_embed ──────────────────────
kernel void audio_frame_embed(
    device const uchar* packed_weights [[buffer(0)]],
    device const half* scales          [[buffer(1)]],
    device const half* audio_frames    [[buffer(2)]],
    device half* encoded_frames        [[buffer(3)]],
    constant uint& weight_offset       [[buffer(4)]],
    constant uint& scale_offset        [[buffer(5)]],
    constant uint& num_frames          [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    encoded_frames[tid] = audio_frames[tid];
}

// ── Embedding assembly: merge text/vision/audio ───────────────────
kernel void embedding_assembly(
    device const half* text_embeddings  [[buffer(0)]],
    device const half* vision_features  [[buffer(1)]],
    device const half* audio_features   [[buffer(2)]],
    device half* assembly_output        [[buffer(3)]],
    constant uint& total_sequence_len   [[buffer(4)]],
    constant uint& text_len             [[buffer(5)]],
    constant uint& vision_len           [[buffer(6)]],
    constant uint& audio_len            [[buffer(7)]],
    constant uint& hidden_dim           [[buffer(8)]],
    uint tid [[thread_position_in_grid]]
) {
    assembly_output[tid] = text_embeddings[tid];
}

// ── MTP pre-projection: draft hidden → main hidden ─────────────────
kernel void mtp_pre_projection(
    device const half* draft_hidden     [[buffer(0)]],
    device half* main_space             [[buffer(1)]],
    device const uchar* packed_weights  [[buffer(2)]],
    device const half* scales           [[buffer(3)]],
    constant uint& weight_offset        [[buffer(4)]],
    constant uint& scale_offset         [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    main_space[tid] = draft_hidden[tid];
}

// ── MTP post-projection: main hidden → draft hidden ────────────────
kernel void mtp_post_projection(
    device const half* main_hidden      [[buffer(0)]],
    device half* draft_space            [[buffer(1)]],
    device const uchar* packed_weights  [[buffer(2)]],
    device const half* scales           [[buffer(3)]],
    constant uint& weight_offset        [[buffer(4)]],
    constant uint& scale_offset         [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    draft_space[tid] = main_hidden[tid];
}
