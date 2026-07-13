// NF4 decode provided by canonical fragment: fragments/nf4_decode.metal
//
#include <metal_stdlib>
using namespace metal;

//  ██████████████████████████████████████████████████████████████████████████
//  NF4 Tile640 GEMV kernel with per-column FP16 reduction-axis scaling
//  sidecar.  Does NOT materialise scaled_input into a separate buffer —
//  the FP16 scale is loaded inside the accumulation loop, converted to
//  float, and multiplied against the activation value before applying the
//  decoded NF4 weight.
//  ██████████████████████████████████████████████████████████████████████████
//
// Buffer ABI:
//   [0] packed_weights     device const uchar*   raw Tile640 packed bytes
//   [1] tile_scales        device const float*   FP32 per-group scales
//   [2] tile_biases        device const float*   FP32 per-group biases
//   [3] in_vector          device const float*   activation vector [in_dim]
//   [4] out_vector         device float*         result vector
//   [5] num_macro_tiles    constant uint         ceil(in_dim / 640)
//   [6] in_dim             constant uint         real (unpadded) input width
//   [7] reduction_scales   device const half*    FP16 column-scale sidecar
//
// Mathematical behavior:
//   scaled_input[col] = in_vector[col] * float(reduction_scales[col])
//   output[row] = sum over cols of dequant_nf4(weight[row,col]) * scaled_input[col]
//
// Threadgroup layout: 32 threads per threadgroup (one SIMD lane group),
// one threadgroup per output row.  Same loop structure as the base
// `fused_gemv_nf4_tile640_fp32` kernel, with reduction_scales multiplied
// inside the innermost loop.
//
// PARTIAL LAST TILE: when in_dim is not a multiple of 640 the packer
// zero-pads the tail (col >= in_dim → NF4 index 7 = 0.0), so those
// weights already contribute nothing.  But `in_vector` AND
// `reduction_scales` are only [in_dim] long, so we MUST guard both reads
// against `in_dim` to avoid out-of-bounds loads.
//
kernel void fused_gemv_nf4_scaled_reduction_tile640_fp32(
    device const uchar* packed_weights     [[buffer(0)]],
    device const float* tile_scales        [[buffer(1)]],
    device const float* tile_biases        [[buffer(2)]],
    device const float* in_vector          [[buffer(3)]],
    device float* out_vector               [[buffer(4)]],
    constant uint& num_macro_tiles         [[buffer(5)]],
    constant uint& in_dim                  [[buffer(6)]],
    device const half* reduction_scales    [[buffer(7)]],
    uint row                               [[threadgroup_position_in_grid]],
    uint simd_lane                         [[thread_index_in_threadgroup]]
) {
    constexpr uint TILE = 640;
    constexpr uint GROUP = 128;
    constexpr uint GROUPS_PER_TILE = 5;
    constexpr uint BYTES_PER_TILE = 320;
    constexpr uint LANE_VALUES = 4;

    float row_accumulator = 0.0f;
    uint row_weight_base = row * num_macro_tiles * BYTES_PER_TILE;
    uint row_meta_base = row * num_macro_tiles * GROUPS_PER_TILE;

    for (uint tile_idx = 0; tile_idx < num_macro_tiles; ++tile_idx) {
        uint meta_base = row_meta_base + tile_idx * GROUPS_PER_TILE;
        for (uint group = 0; group < GROUPS_PER_TILE; ++group) {
            float scale = tile_scales[meta_base + group];
            float bias = tile_biases[meta_base + group];
            uint src_base = tile_idx * TILE + group * GROUP + simd_lane * LANE_VALUES;

            #pragma unroll
            for (uint i = 0; i < LANE_VALUES; ++i) {
                uint col = src_base + i;
                if (col >= in_dim) {
                    continue; // zero-padded tail of a partial last tile
                }
                // row_weight_base offsets packed_weights to this row's data.
                float weight = fma(unpack_nf4(packed_weights + row_weight_base, col), scale, bias);
                // Load the FP16 reduction scale inline — no materialised buffer.
                float scaled_activation = in_vector[col] * float(reduction_scales[col]);
                row_accumulator += weight * scaled_activation;
            }
        }
    }

    row_accumulator = simd_sum(row_accumulator);
    if (simd_lane == 0) {
        out_vector[row] = row_accumulator;
    }
}
