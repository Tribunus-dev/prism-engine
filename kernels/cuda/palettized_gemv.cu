// palettized_gemv.cu — NVIDIA CUDA port of the fused palettized (LUT) GEMV.
//
// Interleaved-fused: the 4-bit index is unpacked and dequantized through the
// per-row 16-entry codebook *in registers*, and the resulting weight is
// multiplied-accumulated immediately — the dequantized weights never touch
// global memory. This is the direct analogue of templates/palettized_gemv.metal.
//
// Layout & numerics: see kernels/common/palettized_abi.h (verified by the oracle).
//
// Build (PTX for the runtime loader, or an object for static link):
//   nvcc -arch=sm_80 -O3 --ptx  palettized_gemv.cu -o palettized_gemv.ptx
//   nvcc -arch=sm_80 -O3 -c     palettized_gemv.cu -o palettized_gemv.o
//
// Launch: one block per output row; 64–128 threads/block. Grid.x = out_dim.

#include <cuda_fp16.h>
#include <cstdint>
#include "../common/palettized_abi.h"

// ── batch = 1 decode GEMV ──────────────────────────────────────────────────
extern "C" __global__ void palettized_gemv(
    const __half*  __restrict__ input_vector,    // [in_dim]
    const __half*  __restrict__ codebook_block,   // [out_dim * 16]
    const uint8_t* __restrict__ indices_block,    // [out_dim * in_dim/2]
    __half*        __restrict__ output_vector,    // [out_dim]
    uint32_t in_dim,
    uint32_t out_dim)
{
    const uint32_t row = blockIdx.x;
    if (row >= out_dim) return;

    const uint32_t tid      = threadIdx.x;
    const uint32_t nthreads = blockDim.x;
    const uint32_t lane     = tid & (warpSize - 1);
    const uint32_t wid      = tid / warpSize;

    // Per-row codebook → shared memory, promoted to fp32 once (reused across all
    // index lookups by every thread in the block).
    __shared__ float sh_cb[PRISM_CODEBOOK_SIZE];
    if (tid < PRISM_CODEBOOK_SIZE)
        sh_cb[tid] = __half2float(codebook_block[row * PRISM_CODEBOOK_SIZE + tid]);
    __syncthreads();

    // Read indices 8-at-a-time as 32-bit words (coalesced across the warp).
    const uint32_t* __restrict__ idx_ptr =
        reinterpret_cast<const uint32_t*>(indices_block + row * (in_dim / 2));
    const uint32_t num_words = in_dim / PRISM_NIBBLES_PER_WORD;

    float acc = 0.0f;
    for (uint32_t w = tid; w < num_words; w += nthreads) {
        const uint32_t packed = idx_ptr[w];
        const uint32_t off = w * PRISM_NIBBLES_PER_WORD;
#pragma unroll
        for (int j = 0; j < PRISM_NIBBLES_PER_WORD; ++j) {
            const float wv = sh_cb[(packed >> (4 * j)) & 0xF];
            acc = __fmaf_rn(__half2float(input_vector[off + j]), wv, acc);
        }
    }

    // Intra-warp reduction via shuffle.
    for (int o = warpSize / 2; o > 0; o >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, o);

    // Inter-warp reduction via a tiny shared scratchpad.
    __shared__ float sh_part[32]; // max 32 warps/block
    if (lane == 0) sh_part[wid] = acc;
    __syncthreads();

    if (tid == 0) {
        const uint32_t nwarps = (nthreads + warpSize - 1) / warpSize;
        float total = 0.0f;
        for (uint32_t i = 0; i < nwarps; ++i) total += sh_part[i];
        output_vector[row] = __float2half(total);
    }
}

// ── batched prefill GEMV ────────────────────────────────────────────────────
// Reuses each row's codebook + indices across `batch` input vectors, amortizing
// the weight reads over the batch — the throughput path for prompt prefill on a
// discrete NVIDIA GPU. input/output are row-major [batch, dim].
// Grid: (out_dim, batch); block: 64–128 threads.
extern "C" __global__ void palettized_gemv_batched(
    const __half*  __restrict__ input_batch,      // [batch * in_dim]
    const __half*  __restrict__ codebook_block,    // [out_dim * 16]
    const uint8_t* __restrict__ indices_block,     // [out_dim * in_dim/2]
    __half*        __restrict__ output_batch,      // [batch * out_dim]
    uint32_t in_dim,
    uint32_t out_dim,
    uint32_t batch)
{
    const uint32_t row = blockIdx.x;
    const uint32_t b   = blockIdx.y;
    if (row >= out_dim || b >= batch) return;

    const uint32_t tid      = threadIdx.x;
    const uint32_t nthreads = blockDim.x;
    const uint32_t lane     = tid & (warpSize - 1);
    const uint32_t wid      = tid / warpSize;

    __shared__ float sh_cb[PRISM_CODEBOOK_SIZE];
    if (tid < PRISM_CODEBOOK_SIZE)
        sh_cb[tid] = __half2float(codebook_block[row * PRISM_CODEBOOK_SIZE + tid]);
    __syncthreads();

    const uint32_t* __restrict__ idx_ptr =
        reinterpret_cast<const uint32_t*>(indices_block + row * (in_dim / 2));
    const __half* __restrict__ x = input_batch + (size_t)b * in_dim;
    const uint32_t num_words = in_dim / PRISM_NIBBLES_PER_WORD;

    float acc = 0.0f;
    for (uint32_t w = tid; w < num_words; w += nthreads) {
        const uint32_t packed = idx_ptr[w];
        const uint32_t off = w * PRISM_NIBBLES_PER_WORD;
#pragma unroll
        for (int j = 0; j < PRISM_NIBBLES_PER_WORD; ++j) {
            const float wv = sh_cb[(packed >> (4 * j)) & 0xF];
            acc = __fmaf_rn(__half2float(x[off + j]), wv, acc);
        }
    }

    for (int o = warpSize / 2; o > 0; o >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, o);

    __shared__ float sh_part[32];
    if (lane == 0) sh_part[wid] = acc;
    __syncthreads();

    if (tid == 0) {
        const uint32_t nwarps = (nthreads + warpSize - 1) / warpSize;
        float total = 0.0f;
        for (uint32_t i = 0; i < nwarps; ++i) total += sh_part[i];
        output_batch[(size_t)b * out_dim + row] = __float2half(total);
    }
}

// ── host launchers (called from the Rust CudaBackend via FFI) ───────────────
#ifndef PRISM_KERNELS_NO_LAUNCHERS
extern "C" void prism_cuda_palettized_gemv(
    const __half* input, const __half* codebook, const uint8_t* indices,
    __half* output, uint32_t in_dim, uint32_t out_dim, cudaStream_t stream)
{
    const int threads = 128;
    palettized_gemv<<<out_dim, threads, 0, stream>>>(
        input, codebook, indices, output, in_dim, out_dim);
}

extern "C" void prism_cuda_palettized_gemv_batched(
    const __half* input, const __half* codebook, const uint8_t* indices,
    __half* output, uint32_t in_dim, uint32_t out_dim, uint32_t batch,
    cudaStream_t stream)
{
    const int threads = 128;
    dim3 grid(out_dim, batch);
    palettized_gemv_batched<<<grid, threads, 0, stream>>>(
        input, codebook, indices, output, in_dim, out_dim, batch);
}
#endif
