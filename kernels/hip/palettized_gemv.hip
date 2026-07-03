// palettized_gemv.hip — AMD (ROCm/HIP) port of the fused palettized GEMV.
//
// Targets discrete AMD GPUs: CDNA (MI2xx/MI3xx, wave64) and RDNA (Radeon,
// wave32). The reduction is wave-size-agnostic (loops over `warpSize`), so one
// source covers both. For an APU iGPU that shares memory with an XDNA NPU, use
// palettized_gemv_igpu_npu.hip instead — that variant is decode-tuned and
// assumes zero-copy unified memory.
//
// Layout & numerics: kernels/common/palettized_abi.h (verified by the oracle).
//
// Build:
//   hipcc -O3 --offload-arch=gfx942 -c palettized_gemv.hip -o palettized_gemv.o   # MI300
//   hipcc -O3 --offload-arch=gfx1100 -c palettized_gemv.hip -o palettized_gemv.o  # RDNA3

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include "../common/palettized_abi.h"

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
    const uint32_t lane     = tid % warpSize;
    const uint32_t wid      = tid / warpSize;

    __shared__ float sh_cb[PRISM_CODEBOOK_SIZE];
    if (tid < PRISM_CODEBOOK_SIZE)
        sh_cb[tid] = __half2float(codebook_block[row * PRISM_CODEBOOK_SIZE + tid]);
    __syncthreads();

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
            acc = fmaf(__half2float(input_vector[off + j]), wv, acc);
        }
    }

    // Wave-size-agnostic reduction (works for wave32 and wave64).
    for (int o = warpSize / 2; o > 0; o >>= 1)
        acc += __shfl_down(acc, o);

    __shared__ float sh_part[64]; // >= max waves per block
    if (lane == 0) sh_part[wid] = acc;
    __syncthreads();

    if (tid == 0) {
        const uint32_t nwaves = (nthreads + warpSize - 1) / warpSize;
        float total = 0.0f;
        for (uint32_t i = 0; i < nwaves; ++i) total += sh_part[i];
        output_vector[row] = __float2half(total);
    }
}

// Batched prefill path for discrete AMD GPUs. input/output row-major.
// Grid: (out_dim, batch).
extern "C" __global__ void palettized_gemv_batched(
    const __half*  __restrict__ input_batch,
    const __half*  __restrict__ codebook_block,
    const uint8_t* __restrict__ indices_block,
    __half*        __restrict__ output_batch,
    uint32_t in_dim,
    uint32_t out_dim,
    uint32_t batch)
{
    const uint32_t row = blockIdx.x;
    const uint32_t b   = blockIdx.y;
    if (row >= out_dim || b >= batch) return;

    const uint32_t tid      = threadIdx.x;
    const uint32_t nthreads = blockDim.x;
    const uint32_t lane     = tid % warpSize;
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
            acc = fmaf(__half2float(x[off + j]), wv, acc);
        }
    }

    for (int o = warpSize / 2; o > 0; o >>= 1)
        acc += __shfl_down(acc, o);

    __shared__ float sh_part[64];
    if (lane == 0) sh_part[wid] = acc;
    __syncthreads();

    if (tid == 0) {
        const uint32_t nwaves = (nthreads + warpSize - 1) / warpSize;
        float total = 0.0f;
        for (uint32_t i = 0; i < nwaves; ++i) total += sh_part[i];
        output_batch[(size_t)b * out_dim + row] = __float2half(total);
    }
}

#ifndef PRISM_KERNELS_NO_LAUNCHERS
extern "C" void prism_hip_palettized_gemv(
    const __half* input, const __half* codebook, const uint8_t* indices,
    __half* output, uint32_t in_dim, uint32_t out_dim, hipStream_t stream)
{
    const int threads = 128;
    palettized_gemv<<<dim3(out_dim), dim3(threads), 0, stream>>>(
        input, codebook, indices, output, in_dim, out_dim);
}

extern "C" void prism_hip_palettized_gemv_batched(
    const __half* input, const __half* codebook, const uint8_t* indices,
    __half* output, uint32_t in_dim, uint32_t out_dim, uint32_t batch,
    hipStream_t stream)
{
    const int threads = 128;
    palettized_gemv_batched<<<dim3(out_dim, batch), dim3(threads), 0, stream>>>(
        input, codebook, indices, output, in_dim, out_dim, batch);
}
#endif
