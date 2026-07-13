// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Linear layer — NF4 quantized weights with per-group scale/bias.
// Each thread computes one output element:
//   output[j] = Σ_i input[i] × dequant_nf4(codes, i * TILE + j, scales, biases, group_size)
//
// Physical storage: Tile640 packed NF4. Each input row occupies one tile
// (640 outputs), padded with zeros if out_dim < 640. Each tile has 5 groups
// of 128 elements, each with its own scale and bias.
// SIMD-coalesced: adjacent lanes read adjacent nibble positions within the
// same tile — all 5 groups' data is contiguous and sequentially accessed.
//
// Buffer layout:
//   [0] input      [in_dim] f32
//   [1] codes      [in_dim × 320] packed uchar (2 nibbles per byte)
//   [2] scales     [in_dim × 5] f32
//   [3] biases     [in_dim × 5] f32
//   [4] output     [out_dim] f32
//   [5] constants  (MlpConstants)
//
// NF4 decode fragment is prepended at build time.

#include <metal_stdlib>
using namespace metal;

#define NF4_TILE_ELEMENTS 640
#define NF4_SCALES_PER_TILE 5
#define NF4_GROUP_SIZE 128

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

kernel void cimage_linear_nf4_f32(
    device const float*     input   [[buffer(0)]],
    device const uchar*     codes   [[buffer(1)]],
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
    uint out_group = tid / NF4_GROUP_SIZE;

    // Physical storage: tile-packed NF4 along output dimension.
    // Each input row occupies one tile (640 outputs).
    // codes[i * 320 + tid/2] gives the byte for element (i, tid).
    // scales[i * 5 + out_group] gives the scale.
    for (uint i = 0; i < in_dim; ++i) {
        float s = scales[i * NF4_SCALES_PER_TILE + out_group];
        float b = biases[i * NF4_SCALES_PER_TILE + out_group];
        float w = fma(unpack_nf4(codes, i * NF4_TILE_ELEMENTS + tid), s, b);
        acc = fma(w, input[i], acc);
    }
    output[tid] = acc;
}
