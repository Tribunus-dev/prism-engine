#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

// Q4_K GEMV: GGML K-quant 4-bit, 256-element super-blocks.
//
// Per super-block layout (140 bytes):
//   [0-1]   half d         — super-block scale
//   [2-3]   half dmin      — super-block minimum
//   [4-5]   half sc0       — sub-block 0 scale
//   [6-7]   half sc1       — sub-block 1 scale
//   [8-9]   half sc2       — sub-block 2 scale
//   [10-11] half sc3       — sub-block 3 scale
//   [12-139] uint8_t qs[128] — packed 4-bit weights (2 per byte)
//
// 4 sub-blocks × 64 elements = 256 elements per super-block.
// Each sub-block has its own f16 scale.
//
// Dequantized weight[i] = d * sc[sub] * (nibble - 8) + dmin
// where sub = i / 64, nibble extracted from packed bytes
// (low nibble for even i, high nibble for odd i).

#define Q4_K_BLOCK_SIZE 256u
#define Q4_K_SUB_BLOCK  64u
#define Q4_K_NIBBLES    4u
#define Q4_K_SCALE_BYTES 12u
#define Q4_K_QS_BYTES   128u
#define Q4_K_BYTES      140u

extern "C" __global__ void q4_k_gemv(
    const float* __restrict__ input,      // [in_dim] — single token input
    const uint8_t* __restrict__ weights,   // [out_dim * super_blocks * Q4_K_BYTES]
    float* __restrict__ output,            // [out_dim]
    uint32_t in_dim,
    uint32_t out_dim)
{
    uint32_t row = blockIdx.x;
    if (row >= out_dim) return;

    uint32_t tid = threadIdx.x;
    uint32_t super_blocks = (in_dim + Q4_K_BLOCK_SIZE - 1u) / Q4_K_BLOCK_SIZE;
    const uint8_t* w_row = weights + (uint64_t)row * (uint64_t)super_blocks * Q4_K_BYTES;
    float acc = 0.0f;

    for (uint32_t sb = 0; sb < super_blocks; sb++) {
        const uint8_t* blk = w_row + (uint64_t)sb * Q4_K_BYTES;

        // Load scales
        half d   = ((const half*)blk)[0];
        half dmin = ((const half*)blk)[1];
        half sc[4];
        sc[0] = ((const half*)blk)[2];
        sc[1] = ((const half*)blk)[3];
        sc[2] = ((const half*)blk)[4];
        sc[3] = ((const half*)blk)[5];

        float fd   = __half2float(d);
        float fdmin = __half2float(dmin);
        float fsc[4] = {
            __half2float(sc[0]),
            __half2float(sc[1]),
            __half2float(sc[2]),
            __half2float(sc[3]),
        };

        const uint8_t* qs = blk + Q4_K_SCALE_BYTES;
        uint32_t sb_start = sb * Q4_K_BLOCK_SIZE;
        uint32_t sb_end = min(sb_start + Q4_K_BLOCK_SIZE, in_dim);

        // Each thread handles elements spaced by blockDim.x within this super-block
        for (uint32_t i = tid; i < sb_end - sb_start; i += blockDim.x) {
            uint32_t abs_i = sb_start + i;
            uint32_t sub = i / Q4_K_SUB_BLOCK;
            uint32_t byte_off = i / 2u;
            uint32_t nibble_shift = (i & 1u) ? 4u : 0u;
            int q = (int)((qs[byte_off] >> nibble_shift) & 0x0Fu);
            float dequant = fd * fsc[sub] * (float)(q - 8) + fdmin;
            acc += dequant * input[abs_i];
        }
    }

    // Warp reduction
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 32);
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 16);
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 8);
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 4);
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 2);
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 1);

    // Cross-warp reduce via shared memory
    __shared__ float shared_acc[256];
    uint32_t warp_id = tid >> 5u;
    uint32_t lane = tid & 31u;
    if (lane == 0) shared_acc[warp_id] = acc;
    __syncthreads();
    if (tid == 0) {
        uint32_t num_warps = (blockDim.x + 31u) >> 5u;
        float total = shared_acc[0];
        for (uint32_t w = 1; w < num_warps; w++) total += shared_acc[w];
        output[row] = total;
    }
}

#undef Q4_K_BLOCK_SIZE
#undef Q4_K_SUB_BLOCK
#undef Q4_K_NIBBLES
#undef Q4_K_SCALE_BYTES
#undef Q4_K_QS_BYTES
#undef Q4_K_BYTES
