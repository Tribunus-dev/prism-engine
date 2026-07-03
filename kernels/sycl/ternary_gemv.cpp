// ternary_gemv.cpp — Intel (SYCL / oneAPI) port of the branch-free 2-bit
// ternary GEMV. Ternary {-1,0,+1} → no multiply. Encoding 00=0, 01=+1, 10=-1,
// 11=0 (see kernels/common/palettized_abi.h). One work-group per output row.
// Build: icpx -fsycl -O3 -fsycl-targets=spir64 ternary_gemv.cpp -c

#include <sycl/sycl.hpp>
#include <cstdint>
#include "../common/palettized_abi.h"

using sycl::half;

#ifndef PRISM_TERNARY_WG
#define PRISM_TERNARY_WG 64
#endif

extern "C" void prism_sycl_ternary_gemv(
    sycl::queue& q,
    const uint8_t* packed_weights, const half* input, half* output,
    uint32_t in_dim, uint32_t out_dim)
{
    constexpr uint32_t WG = PRISM_TERNARY_WG;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t)out_dim * WG), sycl::range<1>(WG)),
            [=](sycl::nd_item<1> it) {
                const uint32_t row = it.get_group(0);
                const uint32_t tid = it.get_local_id(0);

                const uint32_t packed_cols = in_dim / PRISM_TERNARY_PER_BYTE;
                const uint8_t* wrow = packed_weights + (size_t)row * packed_cols;

                float acc = 0.0f;
                for (uint32_t c = tid; c < packed_cols; c += WG) {
                    const uint32_t byte = wrow[c];
                    const uint32_t base = c * PRISM_TERNARY_PER_BYTE;
#pragma unroll
                    for (int k = 0; k < PRISM_TERNARY_PER_BYTE; ++k) {
                        const uint32_t code = (byte >> (2 * k)) & 0x3u;
                        const float iv = (float)input[base + k];
                        acc += (code == 1u) ? iv : ((code == 2u) ? -iv : 0.0f);
                    }
                }

                const float total =
                    sycl::reduce_over_group(it.get_group(), acc, sycl::plus<float>());
                if (tid == 0) output[row] = (half)total;
            });
    });
}
