#include <cuda_fp16.h>
#include <cuda_runtime.h>

// CUDA translation of the Metal palettized GEMV kernel.
// One block (64 threads, 2 warps) per output channel.
//
// Input layout (same as Metal):
//   codebook_block: [out_dim * 16] half  — each row has a 16-entry FP16 codebook
//   indices_block:  [out_dim * in_dim/2] u8 — packed 4-bit nibbles, 8 per uint32
//
// Algorithm:
//   1. Collaborative codebook load into __shared__ memory
//   2. Vectorized nibble extraction + fused dequant dot-product
//   3. Warp-level shuffle reduction (intra-warp, 32 threads)
//   4. Cross-warp reduction via shared memory scratchpad (2 values)
//   5. Thread 0 writes final result to output_vector[row]

extern "C" __global__ void palettized_gemv(
    const half* __restrict__ input_vector,    // [in_dim]
    const half* __restrict__ codebook_block,  // [out_dim * 16]
    const uint8_t* __restrict__ indices_block,// [out_dim * in_dim/2]
    half* __restrict__ output_vector,         // [out_dim]
    uint32_t in_dim,
    uint32_t out_dim)
{
    uint32_t row     = blockIdx.x;                     // threadgroup_position_in_grid
    uint32_t tid     = threadIdx.x;                    // thread_position_in_threadgroup
    uint32_t lane    = threadIdx.x & 31;               // thread_index_in_simdgroup
    uint32_t warp_id = threadIdx.x >> 5;               // simdgroup_index_in_threadgroup

    const half*    row_cb  = codebook_block + (row * 16);
    const uint8_t* row_idx = indices_block  + (row * (in_dim / 2));

    // --- Collaborative codebook load into shared memory ---
    __shared__ half shared_cb[16];
    if (tid < 16) {
        shared_cb[tid] = row_cb[tid];
    }
    __syncthreads();

    // --- Vectorized index processing: 8 nibbles per uint32 read ---
    const uint32_t* idx_ptr  = reinterpret_cast<const uint32_t*>(row_idx);
    uint32_t        num_words = in_dim / 8;
    float           acc       = 0.0f;

    for (uint32_t i = tid; i < num_words; i += blockDim.x) {
        uint32_t packed = idx_ptr[i];
        uint32_t off    = i * 8;

        half v0 = shared_cb[packed & 0x0F];
        half v1 = shared_cb[(packed >> 4) & 0x0F];
        half v2 = shared_cb[(packed >> 8) & 0x0F];
        half v3 = shared_cb[(packed >> 12) & 0x0F];
        half v4 = shared_cb[(packed >> 16) & 0x0F];
        half v5 = shared_cb[(packed >> 20) & 0x0F];
        half v6 = shared_cb[(packed >> 24) & 0x0F];
        half v7 = shared_cb[(packed >> 28)];

        acc += __half2float(input_vector[off + 0]) * __half2float(v0)
            +  __half2float(input_vector[off + 1]) * __half2float(v1)
            +  __half2float(input_vector[off + 2]) * __half2float(v2)
            +  __half2float(input_vector[off + 3]) * __half2float(v3)
            +  __half2float(input_vector[off + 4]) * __half2float(v4)
            +  __half2float(input_vector[off + 5]) * __half2float(v5)
            +  __half2float(input_vector[off + 6]) * __half2float(v6)
            +  __half2float(input_vector[off + 7]) * __half2float(v7);
    }

    // --- Warp-level reduction via shuffle (SIMD-group reduction) ---
    // Equivalent to Metal's simd_sum(acc)
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 16);
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 8);
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 4);
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 2);
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 1);

    // --- Cross-warp reduction via shared memory scratchpad ---
    __shared__ half shared_reduction[2];
    if (lane == 0) {
        shared_reduction[warp_id] = __float2half(acc);
    }
    __syncthreads();

    // --- Thread 0 writes the final dot-product to output ---
    if (tid == 0) {
        output_vector[row] = __hadd(shared_reduction[0], shared_reduction[1]);
    }
}
