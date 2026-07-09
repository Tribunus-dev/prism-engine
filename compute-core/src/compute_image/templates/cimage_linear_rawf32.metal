// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Linear layer — FP32 weights (no quantization).
// Each thread computes one output element:
//   output[j] = Σ_i input[i] × weight[j * in_dim + i]
//
// Buffer layout:
//   [0] input    [in_dim] f32
//   [1] weight   [in_dim × out_dim] f32 (row-major)
//   [2] scales   unused (for API compatibility)
//   [3] biases   unused (for API compatibility)
//   [4] output   [out_dim] f32
//   [5] constants (MlpConstants)

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

kernel void cimage_linear_rawf32(
    device const float*     input   [[buffer(0)]],
    device const float*     weights [[buffer(1)]],
    device const float*     scales  [[buffer(2)]],
    device const float*     biases  [[buffer(3)]],
    device float*           output  [[buffer(4)]],
    constant MlpConstants&  c       [[buffer(5)]],
    uint tid                         [[thread_position_in_grid]]
) {
    uint in_dim = c.hidden_dim;
    uint out_dim = c.intermediate_dim;
    if (tid >= out_dim) return;

    float acc = 0.0f;
    uint row_offset = tid * in_dim;
    for (uint i = 0; i < in_dim; ++i) {
        acc = fma(weights[row_offset + i], input[i], acc);
    }
    output[tid] = acc;
}
