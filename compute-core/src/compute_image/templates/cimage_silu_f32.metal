// SPDX-License-Identifier: MIT OR Apache-2.0
//
// SiLU (Sigmoid Linear Unit) activation: output[i] = input[i] / (1 + exp(-input[i]))
// Each thread processes one element.
//
// Buffer layout:
//   [0] input    [n] f32
//   [1] output   [n] f32
//   [2] constants (MlpConstants, only hidden_dim used as element count)

#include <metal_stdlib>
using namespace metal;

#ifndef MLP_CONSTANTS_DEFINED
#define MLP_CONSTANTS_DEFINED
struct MlpConstants {
    uint32_t hidden_dim;
    uint32_t intermediate_dim;
    uint32_t group_size;
    uint32_t codec_id;
    float    epsilon;
    uint32_t _pad[3];
};
#endif

kernel void cimage_silu_f32(
    device const float*     input   [[buffer(0)]],
    device float*           output  [[buffer(1)]],
    constant MlpConstants&  c       [[buffer(2)]],
    uint tid                         [[thread_position_in_grid]]
) {
    uint n = c.intermediate_dim;
    if (tid >= n) return;

    float x = input[tid];
    output[tid] = x / (1.0f + exp(-x));
}
