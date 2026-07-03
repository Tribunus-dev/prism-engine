// palettized_gemv_igpu_npu.hip — AMD APU variant: RDNA3.5 iGPU + XDNA NPU.
// Target: Ryzen AI / Strix Halo class parts (gfx1150/gfx1151) where a GPU and
// an NPU share one LPDDR pool.
//
// ── Why this is a DIFFERENT kernel from the discrete-GPU one ────────────────
//   1. ZERO-COPY UNIFIED MEMORY. All pointers are into the shared pool
//      (backend allocates via hipHostMalloc(...Coherent) or hipMallocManaged).
//      There is NO hipMemcpy H2D — the `.cimage` weights are mmap'd once and
//      read in place by both the iGPU and the NPU. Nothing crosses a bus.
//   2. DECODE-ONLY (batch = 1). Batched *prefill* runs on the XDNA NPU (matmul
//      units, compute-bound). This kernel is the latency-bound decode step, so
//      there is deliberately no batched variant here.
//   3. BANDWIDTH-FRUGAL. iGPU + NPU + CPU contend for the same DRAM bus, so we
//      cut memory transactions: indices are read 128 bits at a time (uint4 =
//      4 words = 32 nibbles) and the codebook stays resident in LDS.
//   4. wave32 sizing. RDNA3.5 is wave32 by default; a 64-thread block is two
//      waves — enough to hide latency without oversubscribing a small iGPU.
//
// ── NPU handoff contract ────────────────────────────────────────────────────
//   The XDNA NPU writes prefill KV / activations into a buffer in the SAME
//   unified pool. The scheduler passes that buffer's pointer straight in as
//   `input_vector` for the first decode step — no copy, no repack. Weights live
//   once (SegmentKind::TernaryWeights / palettized indices); the NPU reads them
//   for prefill and this kernel reads them for decode.
//
// Layout & numerics: kernels/common/palettized_abi.h.
// Build: hipcc -O3 --offload-arch=gfx1151 -c palettized_gemv_igpu_npu.hip

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include "../common/palettized_abi.h"

#ifndef PRISM_IGPU_BLOCK
#define PRISM_IGPU_BLOCK 64   // 2 × wave32 on RDNA3.5
#endif

extern "C" __global__ void palettized_gemv_igpu_npu(
    const __half*  __restrict__ input_vector,    // [in_dim]  (unified, from NPU)
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

    const uint32_t* __restrict__ idx_words =
        reinterpret_cast<const uint32_t*>(indices_block + row * (in_dim / 2));
    const uint32_t num_words = in_dim / PRISM_NIBBLES_PER_WORD;
    const uint32_t num_quads = num_words / 4;   // 4 words = 128-bit load

    float acc = 0.0f;

    // Main loop: one 128-bit load pulls 32 nibbles (32 input elems) at once,
    // minimizing memory transactions on the shared LPDDR bus.
    const uint4* __restrict__ idx_quads = reinterpret_cast<const uint4*>(idx_words);
    for (uint32_t q = tid; q < num_quads; q += nthreads) {
        const uint4 packed4 = idx_quads[q];
        const uint32_t words[4] = { packed4.x, packed4.y, packed4.z, packed4.w };
        const uint32_t base = q * 32u;
#pragma unroll
        for (int wsub = 0; wsub < 4; ++wsub) {
            const uint32_t packed = words[wsub];
            const uint32_t off = base + wsub * PRISM_NIBBLES_PER_WORD;
#pragma unroll
            for (int j = 0; j < PRISM_NIBBLES_PER_WORD; ++j) {
                const float wv = sh_cb[(packed >> (4 * j)) & 0xF];
                acc = fmaf(__half2float(input_vector[off + j]), wv, acc);
            }
        }
    }

    // Scalar tail for the words not covered by a full quad.
    for (uint32_t w = num_quads * 4 + tid; w < num_words; w += nthreads) {
        const uint32_t packed = idx_words[w];
        const uint32_t off = w * PRISM_NIBBLES_PER_WORD;
#pragma unroll
        for (int j = 0; j < PRISM_NIBBLES_PER_WORD; ++j) {
            const float wv = sh_cb[(packed >> (4 * j)) & 0xF];
            acc = fmaf(__half2float(input_vector[off + j]), wv, acc);
        }
    }

    for (int o = warpSize / 2; o > 0; o >>= 1)
        acc += __shfl_down(acc, o);

    __shared__ float sh_part[8]; // 64 threads / wave32 = 2 waves; headroom
    if (lane == 0) sh_part[wid] = acc;
    __syncthreads();

    if (tid == 0) {
        const uint32_t nwaves = (nthreads + warpSize - 1) / warpSize;
        float total = 0.0f;
        for (uint32_t i = 0; i < nwaves; ++i) total += sh_part[i];
        output_vector[row] = __float2half(total);
    }
}

#ifndef PRISM_KERNELS_NO_LAUNCHERS
// No H2D copy: `input`, `codebook`, `indices`, `output` are all pointers into
// the unified pool. The NPU produced `input` in that same pool during prefill.
extern "C" void prism_hip_igpu_palettized_gemv_decode(
    const __half* input, const __half* codebook, const uint8_t* indices,
    __half* output, uint32_t in_dim, uint32_t out_dim, hipStream_t stream)
{
    palettized_gemv_igpu_npu<<<dim3(out_dim), dim3(PRISM_IGPU_BLOCK), 0, stream>>>(
        input, codebook, indices, output, in_dim, out_dim);
}
#endif
