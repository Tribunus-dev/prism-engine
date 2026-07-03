// palettized_gemv_igpu_npu.cpp — Intel APU variant: Xe2-LPG iGPU + Intel NPU.
// Target: Lunar Lake / Core Ultra (Series 2) class parts where the Xe iGPU and
// the NPU share one LPDDR pool.
//
// ── Why this is a DIFFERENT kernel from the discrete-GPU one ────────────────
//   1. ZERO-COPY USM-SHARED. All pointers are sycl::malloc_shared allocations
//      in the pool both the iGPU and the NPU see. The `.cimage` weights are
//      mapped once; there is no explicit copy to a device buffer.
//   2. DECODE-ONLY (batch = 1). Batched *prefill* runs on the Intel NPU (its
//      matmul array is the compute-bound-prefill workhorse); this kernel is the
//      latency-bound decode step, so there is intentionally no batched form.
//   3. SIMD16 sub-groups + small (32-thread) work-groups — the right occupancy
//      for the modest Xe2-LPG iGPU, and it leaves EU headroom while the NPU and
//      CPU share the same power/bandwidth envelope.
//   4. Codebook kept in shared-local memory; one linear pass over the indices
//      to minimize traffic on the contended LPDDR bus.
//
// ── NPU handoff contract ────────────────────────────────────────────────────
//   The Intel NPU writes prefill activations / KV into a shared USM buffer; the
//   scheduler hands that pointer in as `input` for the first decode step — no
//   copy, no repack. Weights are resident once and read by both engines.
//
// Layout & numerics: kernels/common/palettized_abi.h.
// Build: icpx -fsycl -O3 -fsycl-targets=spir64 palettized_gemv_igpu_npu.cpp -c

#include <sycl/sycl.hpp>
#include <cstdint>
#include "../common/palettized_abi.h"

using sycl::half;

#ifndef PRISM_IGPU_WG
#define PRISM_IGPU_WG 32   // 2 × SIMD16 sub-groups
#endif
#ifndef PRISM_IGPU_SG
#define PRISM_IGPU_SG 16   // Xe-LPG favours SIMD16
#endif

// No explicit copy: `input`, `codebook`, `indices`, `output` are USM-shared
// pointers. `input` was produced by the NPU during prefill, in the same pool.
extern "C" void prism_sycl_igpu_palettized_gemv_decode(
    sycl::queue& q,
    const half* input, const half* codebook, const uint8_t* indices,
    half* output, uint32_t in_dim, uint32_t out_dim)
{
    constexpr uint32_t WG = PRISM_IGPU_WG;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sh_cb(sycl::range<1>(PRISM_CODEBOOK_SIZE), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t)out_dim * WG), sycl::range<1>(WG)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(PRISM_IGPU_SG)]] {
                const uint32_t row = it.get_group(0);
                const uint32_t tid = it.get_local_id(0);

                if (tid < PRISM_CODEBOOK_SIZE)
                    sh_cb[tid] = (float)codebook[row * PRISM_CODEBOOK_SIZE + tid];
                it.barrier(sycl::access::fence_space::local_space);

                const uint32_t* idx =
                    reinterpret_cast<const uint32_t*>(indices + (size_t)row * (in_dim / 2));
                const uint32_t num_words = in_dim / PRISM_NIBBLES_PER_WORD;

                float acc = 0.0f;
                for (uint32_t w = tid; w < num_words; w += WG) {
                    const uint32_t packed = idx[w];
                    const uint32_t off = w * PRISM_NIBBLES_PER_WORD;
#pragma unroll
                    for (int j = 0; j < PRISM_NIBBLES_PER_WORD; ++j)
                        acc += (float)input[off + j] * sh_cb[(packed >> (4 * j)) & 0xF];
                }

                const float total =
                    sycl::reduce_over_group(it.get_group(), acc, sycl::plus<float>());
                if (tid == 0) output[row] = (half)total;
            });
    });
}
