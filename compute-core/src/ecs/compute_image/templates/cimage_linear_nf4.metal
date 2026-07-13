// NF4 decode provided by canonical fragment: fragments/nf4_decode.metal
//
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Linear layer — NF4 quantized weights (packed 4-bit codes, 2 per byte).
// Each thread computes one output element by dequantizing codes on the fly:
//   output[j] = Σ_i input[i] × deq(code[j][i], tile, group)
//   where deq reads a 4-bit code, looks up nf4_codebook[code],
//   then applies per-group scale/bias.
//
// Packing: codes[byte_idx] = high_nibble<<4 | low_nibble.
//   i%2==0 → low nibble, i%2==1 → high nibble
//
// Buffer layout:
//   [0] input    [in_dim] f32
//   [1] codes    [in_dim × padded_out_dim / 2] uint8 (packed 4-bit)
//   [2] scales   [num_tiles × groups_per_tile] f32
//   [3] biases   [num_tiles × groups_per_tile] f32
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

kernel void cimage_linear_nf4(
    device const float*     input   [[buffer(0)]],
    device const uchar*     codes   [[buffer(1)]],
    device const float*     scales  [[buffer(2)]],
    device const float*     biases  [[buffer(3)]],
    device float*           output  [[buffer(4)]],
    constant MlpConstants&  c       [[buffer(5)]],
    uint tid                         [[thread_position_in_grid]]
) {
    uint in_dim     = c.hidden_dim;
    uint out_dim    = c.intermediate_dim;
    uint group_size = c.group_size;
    if (tid >= out_dim) return;

    uint groups_per_tile = 640 / group_size;
    uint tile            = tid / 640;

    float acc = 0.0f;
    uint code_row_offset = tid * (in_dim / 2);

    for (uint i = 0; i < in_dim; ++i) {
        uint group = i / group_size;
        float s = scales[tile * groups_per_tile + group];
        float b = biases[tile * groups_per_tile + group];
        float w = fma(unpack_nf4(codes, i), s, b);

        acc = fma(w, input[i], acc);
    }
    output[tid] = acc;
}
