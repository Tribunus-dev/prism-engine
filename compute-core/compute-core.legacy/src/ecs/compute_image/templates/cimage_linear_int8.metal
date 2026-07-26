// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Linear layer — INT8 quantized weights with per-output-channel scale/bias.
// Each thread computes one output element:
//   output[j] = Σ_i input[i] × (int8_weight[i * out_dim + j] * scale[j] + bias[j])
//
// Physical storage: W_transposed[in_dim, out_dim] int8 row-major.
// scale[bias] are per logical output channel (tid).
// SIMD-coalesced access: adjacent lanes read adjacent addresses.
//
// Buffer layout:
//   [0] input    [in_dim] f32
//   [1] weight   [in_dim × padded_out_dim] int8 (row-major)
//   [2] scales   [out_dim] f32
//   [3] biases   [out_dim] f32
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

kernel void cimage_linear_int8(
    device const float*     input   [[buffer(0)]],
    device const char*      weights [[buffer(1)]],
    device const float*     scales  [[buffer(2)]],
    device const float*     biases  [[buffer(3)]],
    device float*           output  [[buffer(4)]],
    constant MlpConstants&  c       [[buffer(5)]],
    uint tid                         [[thread_position_in_grid]]
) {
    uint in_dim = c.hidden_dim;
    uint out_dim = c.intermediate_dim;
    if (tid >= out_dim) return;

    float scale = scales[tid];
    float bias  = biases[tid];

    float acc = 0.0f;
    // Physical storage: [in_dim, out_dim] int8 row-major.
    // weights[i * out_dim + tid] — coalesced across SIMD lanes.
    for (uint i = 0; i < in_dim; ++i) {
        float w = (float)weights[i * out_dim + tid] * scale + bias;
        acc = fma(w, input[i], acc);
    }
    output[tid] = acc;
}
