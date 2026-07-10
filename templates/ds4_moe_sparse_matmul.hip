// DS4 MoE Sparse Expert Matmul — gather, quantized matmul (Q4_K/Q2_K), scatter.
// AOT compilation uses {{PLACEHOLDER}} for tunable parameters:
//   {{N_EXPERT}}          — total number of experts
//   {{N_EXPERT_USED}}     — experts selected per token (6)
//   {{EXPERT_MID_DIM}}    — intermediate dimension for expert matmul
//   {{OUT_DIM}}           — output dimension
//   {{QK_K}}              — GGML super-block size (256)
//   {{Q4_K_BLOCK_BYTES}}  — bytes per Q4_K super-block (140)
//   {{Q2_K_BLOCK_BYTES}}  — bytes per Q2_K super-block (72)
//
// These kernels follow the reference pattern from ds4_rocm_moe.cuh:
//   - quarter-warp (8-lane) cooperative block dot-product
//   - Q8_K-quantized activations paired with Q4_K/Q2_K weight blocks
//   - Weight addressing: expert * expert_bytes + row * row_bytes
//   - Output per (token, expert) pair → expert-weighted sum → batch output

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

// =========================================================================
// GGML block types (portable, matches cuda_block_q* from llama.cpp)
// =========================================================================

// Q4_K: 256-element super-block, 140 bytes
struct block_q4_K {
    __half   d;
    __half   dmin;
    uint8_t  scales[12];
    uint8_t  qs[128];
};
static_assert(sizeof(block_q4_K) == 140, "block_q4_K size mismatch");

// Q2_K: 256-element super-block, 72 bytes
struct block_q2_K {
    __half   d;
    __half   dmin;
    uint8_t  scales[16];
    uint8_t  qs[64];
};
static_assert(sizeof(block_q2_K) == 72, "block_q2_K size mismatch");

// Q8_K: 256-element super-block, 292 bytes
struct block_q8_K {
    __half   d;
    int16_t  bsums[16];
    int8_t   qs[256];
};
static_assert(sizeof(block_q8_K) == 292, "block_q8_K size mismatch");

// =========================================================================
// Device helpers — ported from ds4_rocm_moe.cuh
// =========================================================================

__device__ static float dev_f16_to_f32(__half v) {
    return __half2float(v);
}

__device__ static void dev_q4_K_get_scale_min(
    uint32_t j,
    const uint8_t *scales,
    uint8_t *d_out,
    uint8_t *m_out)
{
    if (j < 4u) {
        *d_out = scales[j] & 63u;
        *m_out = scales[j + 4u] & 63u;
    } else {
        *d_out = (scales[j + 4u] & 0x0fu) | ((scales[j - 4u] >> 6u) << 4u);
        *m_out = (scales[j + 4u] >> 4u) | ((scales[j] >> 6u) << 4u);
    }
}

__device__ __forceinline__ static int32_t dev_dot_q4_32(
    const uint8_t *qs,
    const int8_t *q8,
    int shift)
{
    int32_t sum = 0;
    #pragma unroll
    for (uint32_t i = 0; i < 32u; i += 4u) {
        const int32_t v = (*(const int32_t *)(qs + i) >> shift) & 0x0f0f0f0f;
        sum = __dp4a(v, *(const int32_t *)(q8 + i), sum);
    }
    return sum;
}

__device__ static int32_t dev_dot_q2_16(const uint8_t *q2, const int8_t *q8, int shift) {
    int32_t sum = 0;
    #pragma unroll
    for (uint32_t i = 0; i < 16; i += 4) {
        const int32_t v = (*(const int32_t *)(q2 + i) >> shift) & 0x03030303;
        sum = __dp4a(v, *(const int32_t *)(q8 + i), sum);
    }
    return sum;
}

// Dot product of one Q4_K super-block with one Q8_K super-block
__device__ static float dev_dot_q4_K_q8_K_block(
    const struct block_q4_K *x,
    const struct block_q8_K *y)
{
    const float xd   = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    int isum  = 0;
    int summs = 0;
    #pragma unroll
    for (uint32_t j = 0; j < 8u; j++) {
        uint8_t sc, m;
        dev_q4_K_get_scale_min(j, x->scales, &sc, &m);
        summs += (int)m * (int)(y->bsums[2u * j] + y->bsums[2u * j + 1u]);
        const uint32_t byte_off = (j >> 1u) * 32u;
        const int shift = (j & 1u) ? 4 : 0;
        isum += (int)sc * dev_dot_q4_32(x->qs + byte_off, y->qs + j * 32u, shift);
    }
    return __half2float(y->d) * xd * (float)isum - __half2float(y->d) * xmin * (float)summs;
}

// Dot product of one Q2_K super-block with one Q8_K super-block
__device__ static float dev_dot_q2_K_q8_K_block(
    const struct block_q2_K *x,
    const struct block_q8_K *y)
{
    const uint8_t *q2 = x->qs;
    const int8_t  *q8 = y->qs;
    const uint8_t *sc = x->scales;
    int summs = 0;
    for (int j = 0; j < 16; j++) summs += y->bsums[j] * (sc[j] >> 4);
    const float dall = __half2float(y->d) * dev_f16_to_f32(x->d);
    const float dmin = __half2float(y->d) * dev_f16_to_f32(x->dmin);
    int isum = 0;
    int is = 0;
    for (int k = 0; k < {{QK_K}} / 128; k++) {
        int shift = 0;
        for (int j = 0; j < 4; j++) {
            int d = sc[is++] & 0x0f;
            isum += d * dev_dot_q2_16(q2, q8, shift);
            d = sc[is++] & 0x0f;
            isum += d * dev_dot_q2_16(q2 + 16, q8 + 16, shift);
            shift += 2;
            q8 += 32;
        }
        q2 += 32;
    }
    return dall * (float)isum - dmin * (float)summs;
}

__device__ __forceinline__ static float quarter_warp_sum_f32(float v, uint32_t lane) {
    (void)lane;
    uint32_t mask = 0xFFu << (threadIdx.x & 24u);
    v += __shfl_down_sync(mask, v, 1, 8);
    v += __shfl_down_sync(mask, v, 2, 8);
    v += __shfl_down_sync(mask, v, 4, 8);
    return v;
}

// =========================================================================
// Kernel: ds4_moe_sparse_matmul_q4k_decode
//
// Single-token (decode) sparse MoE Q4_K down-projection.
// For one token, processes all N_EXPERT_USED experts, accumulating
// weighted output directly into the output row.
//
// Q8_K activation blocks: midq [n_expert_used, mid_blocks]
// Q4_K weight blocks:     w_q4k (flat) addressed as:
//     expert e, row r → w_q4k + e * expert_bytes + r * row_bytes
//     each row has mid_blocks consecutive Q4_K blocks
//
// Grid: blockIdx.x * 32 rows (4 quarter-warps × 8 lanes)
//
// Each quarter-warp (8 lanes):
//   - lane 0..7 cooperatively sums over mid_blocks
//   - lane-id within quarter-warp = (threadIdx.x & 7)
//   - row = blockIdx.x * 32 + (threadIdx.x >> 3)
// =========================================================================
extern "C" __global__ void ds4_moe_sparse_matmul_q4k_decode(
    const struct block_q8_K *midq,
    const struct block_q4_K *w_q4k,
    const int32_t *expert_ids,
    float *output,
    uint32_t mid_blocks,
    uint32_t out_dim,
    uint64_t expert_bytes,
    uint64_t row_bytes,
    float expert_weight_scale)
{
    constexpr uint32_t n_used = {{N_EXPERT_USED}};
    const uint32_t lane = threadIdx.x & 7u;
    const uint32_t row  = blockIdx.x * 32u + (threadIdx.x >> 3u);
    if (row >= out_dim) return;

    float total = 0.0f;
    #pragma unroll
    for (uint32_t slot = 0; slot < n_used; slot++) {
        const int32_t expert_i = expert_ids[slot];
        if (expert_i < 0) continue;

        // Point to this expert's weight row's first Q4_K block
        const struct block_q4_K *wr = (const struct block_q4_K *)(
            (const uint8_t *)w_q4k
            + (uint64_t)(uint32_t)expert_i * expert_bytes
            + (uint64_t)row * row_bytes);

        // Q8_K activation blocks for this slot
        const struct block_q8_K *xq = midq + (uint64_t)slot * mid_blocks;

        float acc = 0.0f;
        for (uint32_t b = lane; b < mid_blocks; b += 8u) {
            acc += dev_dot_q4_K_q8_K_block(wr + b, xq + b);
        }
        acc = quarter_warp_sum_f32(acc, lane);
        if (lane == 0) total += acc;
    }
    if (lane == 0) {
        output[row] = total * expert_weight_scale;
    }
}

// =========================================================================
// Kernel: ds4_moe_sparse_matmul_q4k_batch
//
// Batch variant. blockIdx.y = token index.
// Each block processes out_dim / 32 rows for one token.
// =========================================================================
extern "C" __global__ void ds4_moe_sparse_matmul_q4k_batch(
    const struct block_q8_K *midq,
    const struct block_q4_K *w_q4k,
    const int32_t *expert_ids,
    const float *weights,
    float *output,
    uint32_t batch,
    uint32_t n_total_expert,
    uint32_t mid_blocks,
    uint32_t out_dim,
    uint64_t expert_bytes,
    uint64_t row_bytes,
    float expert_weight_scale)
{
    constexpr uint32_t n_used = {{N_EXPERT_USED}};
    const uint32_t lane = threadIdx.x & 7u;
    const uint32_t row  = blockIdx.x * 32u + (threadIdx.x >> 3u);
    const uint32_t t    = blockIdx.y;
    if (t >= batch || row >= out_dim) return;

    float total = 0.0f;
    #pragma unroll
    for (uint32_t slot = 0; slot < n_used; slot++) {
        const int32_t expert_i = expert_ids[(uint64_t)t * n_used + slot];
        if (expert_i < 0 || (uint32_t)expert_i >= n_total_expert) continue;

        const float wgt = weights[(uint64_t)t * n_used + slot];

        const struct block_q4_K *wr = (const struct block_q4_K *)(
            (const uint8_t *)w_q4k
            + (uint64_t)(uint32_t)expert_i * expert_bytes
            + (uint64_t)row * row_bytes);

        const struct block_q8_K *xq = midq + ((uint64_t)t * n_used + slot) * mid_blocks;

        float acc = 0.0f;
        for (uint32_t b = lane; b < mid_blocks; b += 8u) {
            acc += dev_dot_q4_K_q8_K_block(wr + b, xq + b);
        }
        acc = quarter_warp_sum_f32(acc, lane);
        if (lane == 0) total += acc * wgt;
    }
    if (lane == 0) {
        output[(uint64_t)t * out_dim + row] = total * expert_weight_scale;
    }
}

// =========================================================================
// Kernel: ds4_moe_sparse_matmul_q2k_decode
//
// Single-token decode, Q2_K weight blocks.
// =========================================================================
extern "C" __global__ void ds4_moe_sparse_matmul_q2k_decode(
    const struct block_q8_K *midq,
    const struct block_q2_K *w_q2k,
    const int32_t *expert_ids,
    float *output,
    uint32_t mid_blocks,
    uint32_t out_dim,
    uint64_t expert_bytes,
    uint64_t row_bytes,
    float expert_weight_scale)
{
    constexpr uint32_t n_used = {{N_EXPERT_USED}};
    const uint32_t lane = threadIdx.x & 7u;
    const uint32_t row  = blockIdx.x * 32u + (threadIdx.x >> 3u);
    if (row >= out_dim) return;

    float total = 0.0f;
    #pragma unroll
    for (uint32_t slot = 0; slot < n_used; slot++) {
        const int32_t expert_i = expert_ids[slot];
        if (expert_i < 0) continue;

        const struct block_q2_K *wr = (const struct block_q2_K *)(
            (const uint8_t *)w_q2k
            + (uint64_t)(uint32_t)expert_i * expert_bytes
            + (uint64_t)row * row_bytes);

        const struct block_q8_K *xq = midq + (uint64_t)slot * mid_blocks;

        float acc = 0.0f;
        for (uint32_t b = lane; b < mid_blocks; b += 8u) {
            acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
        }
        acc = quarter_warp_sum_f32(acc, lane);
        if (lane == 0) total += acc;
    }
    if (lane == 0) {
        output[row] = total * expert_weight_scale;
    }
}

// =========================================================================
// Kernel: ds4_moe_sparse_matmul_q2k_batch
//
// Batch variant, Q2_K weight blocks.
// =========================================================================
extern "C" __global__ void ds4_moe_sparse_matmul_q2k_batch(
    const struct block_q8_K *midq,
    const struct block_q2_K *w_q2k,
    const int32_t *expert_ids,
    const float *weights,
    float *output,
    uint32_t batch,
    uint32_t n_total_expert,
    uint32_t mid_blocks,
    uint32_t out_dim,
    uint64_t expert_bytes,
    uint64_t row_bytes,
    float expert_weight_scale)
{
    constexpr uint32_t n_used = {{N_EXPERT_USED}};
    const uint32_t lane = threadIdx.x & 7u;
    const uint32_t row  = blockIdx.x * 32u + (threadIdx.x >> 3u);
    const uint32_t t    = blockIdx.y;
    if (t >= batch || row >= out_dim) return;

    float total = 0.0f;
    #pragma unroll
    for (uint32_t slot = 0; slot < n_used; slot++) {
        const int32_t expert_i = expert_ids[(uint64_t)t * n_used + slot];
        if (expert_i < 0 || (uint32_t)expert_i >= n_total_expert) continue;

        const float wgt = weights[(uint64_t)t * n_used + slot];

        const struct block_q2_K *wr = (const struct block_q2_K *)(
            (const uint8_t *)w_q2k
            + (uint64_t)(uint32_t)expert_i * expert_bytes
            + (uint64_t)row * row_bytes);

        const struct block_q8_K *xq = midq + ((uint64_t)t * n_used + slot) * mid_blocks;

        float acc = 0.0f;
        for (uint32_t b = lane; b < mid_blocks; b += 8u) {
            acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
        }
        acc = quarter_warp_sum_f32(acc, lane);
        if (lane == 0) total += acc * wgt;
    }
    if (lane == 0) {
        output[(uint64_t)t * out_dim + row] = total * expert_weight_scale;
    }
}
