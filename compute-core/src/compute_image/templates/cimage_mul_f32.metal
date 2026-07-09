// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Element-wise multiply: output[i] = a[i] * b[i]
// Each thread processes one element.
//
// Buffer layout:
//   [0] a        [n] f32
//   [1] b        [n] f32
//   [2] output   [n] f32
//   [3] constants (MlpConstants, only hidden_dim used as element count)

#include <metal_stdlib>
using namespace metal;

struct MlpConstants {
    uint32_t hidden_dim;
    uint32_t intermediate_dim;
    uint32_t group_size;
    uint32_t codec_id;
    float    epsilon;
    uint32_t _pad[3];
};

kernel void cimage_mul_f32(
    device const float*     a       [[buffer(0)]],
    device const float*     b       [[buffer(1)]],
    device float*           output  [[buffer(2)]],
    constant MlpConstants&  c       [[buffer(3)]],
    uint tid                         [[thread_position_in_grid]]
) {
    uint n = c.intermediate_dim;
    if (tid >= n) return;

    output[tid] = a[tid] * b[tid];
}
