// Q8_0 GEMV — GGML block-f32 scale + int8, 32-element blocks.
//
// Block layout (36 bytes per block):
//   [0..3]   f32 scale
//   [4..35]  32 x int8 weights
//
// Each block holds 32 elements at 9 effective bits per element
// (8 bits data + 1 bit scale amortized across the block).
//
// Kernel: each thread block processes one output row,
// iterating over all blocks and accumulating f32 dot products.

#include <hip/hip_runtime.h>

extern "C" __global__ void q8_0_gemv(
    const float* __restrict__ input,
    const uint8_t* __restrict__ weights,  // packed blocks: [f32 scale, 32 x int8] x Nblocks
    const float* __restrict__ scales,     // per-block scales (reserved for API consistency)
    float* __restrict__ output,
    uint32_t in_dim,
    uint32_t out_dim)
{
    uint32_t row = blockIdx.x;
    if (row >= out_dim) return;

    const uint32_t blocks = (in_dim + 31u) / 32u;
    // Each block: 4 bytes f32 scale + 32 bytes int8 = 36 bytes
    const uint8_t* row_blocks = weights + (uint64_t)row * (uint64_t)blocks * 36u;

    float acc = 0.0f;

    for (uint32_t b = 0u; b < blocks; b++) {
        const uint8_t* blk = row_blocks + (uint64_t)b * 36u;

        // Read f32 scale
        float d;
        __builtin_memcpy(&d, blk, sizeof(float));

        // Read 32 int8 weights
        const int8_t* q = (const int8_t*)(blk + 4u);

        uint32_t i0 = b * 32u;
        uint32_t bn = (in_dim - i0 < 32u) ? (in_dim - i0) : 32u;

        float block_dot = 0.0f;
        for (uint32_t i = 0u; i < bn; i++) {
            block_dot += (float)q[i] * input[i0 + i];
        }
        acc += d * block_dot;
    }

    output[row] = acc;
}
