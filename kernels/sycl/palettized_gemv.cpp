// palettized_gemv.cpp — Intel (SYCL / oneAPI, Level Zero) port of the fused
// palettized GEMV. Targets discrete Intel GPUs (Arc, Xe-HPG/Xe2-HPG).
// For a Lunar Lake-class iGPU that shares memory with the Intel NPU, use
// palettized_gemv_igpu_npu.cpp (USM-shared, SIMD16, decode-tuned).
//
// Layout & numerics: kernels/common/palettized_abi.h (verified by the oracle).
// One work-group per output row; work-group reduction via reduce_over_group.
//
// Build: icpx -fsycl -O3 -fsycl-targets=spir64 palettized_gemv.cpp -c

#include <sycl/sycl.hpp>
#include <cstdint>
#include "../common/palettized_abi.h"

using sycl::half;

#ifndef PRISM_DGPU_WG
#define PRISM_DGPU_WG 64
#endif
#ifndef PRISM_DGPU_SG
#define PRISM_DGPU_SG 32   // Arc supports SIMD8/16/32; 32 is a good dGPU default
#endif

// batch = 1 decode GEMV
extern "C" void prism_sycl_palettized_gemv(
    sycl::queue& q,
    const half* input, const half* codebook, const uint8_t* indices,
    half* output, uint32_t in_dim, uint32_t out_dim)
{
    constexpr uint32_t WG = PRISM_DGPU_WG;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sh_cb(sycl::range<1>(PRISM_CODEBOOK_SIZE), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t)out_dim * WG), sycl::range<1>(WG)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(PRISM_DGPU_SG)]] {
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

// Batched prefill GEMV for a discrete Intel GPU. input/output row-major.
extern "C" void prism_sycl_palettized_gemv_batched(
    sycl::queue& q,
    const half* input_batch, const half* codebook, const uint8_t* indices,
    half* output_batch, uint32_t in_dim, uint32_t out_dim, uint32_t batch)
{
    constexpr uint32_t WG = PRISM_DGPU_WG;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sh_cb(sycl::range<1>(PRISM_CODEBOOK_SIZE), h);
        h.parallel_for(
            sycl::nd_range<2>(sycl::range<2>((size_t)out_dim * WG, batch),
                              sycl::range<2>(WG, 1)),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(PRISM_DGPU_SG)]] {
                const uint32_t row = it.get_group(0);
                const uint32_t b   = it.get_group(1);
                const uint32_t tid = it.get_local_id(0);

                if (tid < PRISM_CODEBOOK_SIZE)
                    sh_cb[tid] = (float)codebook[row * PRISM_CODEBOOK_SIZE + tid];
                it.barrier(sycl::access::fence_space::local_space);

                const uint32_t* idx =
                    reinterpret_cast<const uint32_t*>(indices + (size_t)row * (in_dim / 2));
                const half* x = input_batch + (size_t)b * in_dim;
                const uint32_t num_words = in_dim / PRISM_NIBBLES_PER_WORD;

                float acc = 0.0f;
                for (uint32_t w = tid; w < num_words; w += WG) {
                    const uint32_t packed = idx[w];
                    const uint32_t off = w * PRISM_NIBBLES_PER_WORD;
#pragma unroll
                    for (int j = 0; j < PRISM_NIBBLES_PER_WORD; ++j)
                        acc += (float)x[off + j] * sh_cb[(packed >> (4 * j)) & 0xF];
                }

                const float total =
                    sycl::reduce_over_group(it.get_group(), acc, sycl::plus<float>());
                if (tid == 0) output_batch[(size_t)b * out_dim + row] = (half)total;
            });
    });
}
