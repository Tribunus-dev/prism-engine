// ternary_gemv.cu — NVIDIA CUDA port of the branch-free 2-bit ternary GEMV.
//
// Ternary weights {-1, 0, +1} need no multiply — each 2-bit code selects
// add / skip / subtract of the input element. Port of compute-core's
// ternary_gemv.metal. Encoding: 00=0, 01=+1, 10=-1, 11=0 (see palettized_abi.h).
//
// Build: nvcc -arch=sm_80 -O3 --ptx ternary_gemv.cu -o ternary_gemv.ptx
// Launch: one warp per output row is efficient here; we use one block/row with
// a modest thread count and reduce, matching the palettized launcher style.

#include <cuda_fp16.h>
#include <cstdint>
#include "../common/palettized_abi.h"

extern "C" __global__ void ternary_gemv(
    const uint8_t* __restrict__ packed_weights,  // [out_dim * in_dim/4]
    const __half*  __restrict__ input,           // [in_dim]
    __half*        __restrict__ output,          // [out_dim]
    uint32_t in_dim,
    uint32_t out_dim)
{
    const uint32_t row = blockIdx.x;
    if (row >= out_dim) return;

    const uint32_t tid      = threadIdx.x;
    const uint32_t nthreads = blockDim.x;
    const uint32_t lane     = tid & (warpSize - 1);
    const uint32_t wid      = tid / warpSize;

    const uint32_t packed_cols = in_dim / PRISM_TERNARY_PER_BYTE; // 4 weights/byte
    const uint8_t* __restrict__ wrow = packed_weights + (size_t)row * packed_cols;

    float acc = 0.0f;
    for (uint32_t c = tid; c < packed_cols; c += nthreads) {
        const uint32_t byte = wrow[c];
        const uint32_t base = c * PRISM_TERNARY_PER_BYTE;
#pragma unroll
        for (int k = 0; k < PRISM_TERNARY_PER_BYTE; ++k) {
            const uint32_t code = (byte >> (2 * k)) & 0x3u;
            const float iv = __half2float(input[base + k]);
            // 00,11 -> 0 ; 01 -> +iv ; 10 -> -iv  (branch-free)
            acc += (code == 1u) ? iv : ((code == 2u) ? -iv : 0.0f);
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
        output[row] = __float2half(total);
    }
}

#ifndef PRISM_KERNELS_NO_LAUNCHERS
extern "C" void prism_cuda_ternary_gemv(
    const uint8_t* weights, const __half* input, __half* output,
    uint32_t in_dim, uint32_t out_dim, cudaStream_t stream)
{
    const int threads = 128;
    ternary_gemv<<<out_dim, threads, 0, stream>>>(
        weights, input, output, in_dim, out_dim);
}
#endif
