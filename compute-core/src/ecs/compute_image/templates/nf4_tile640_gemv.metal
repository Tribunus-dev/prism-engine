// NF4 decode provided by canonical fragment: fragments/nf4_decode.metal
//
#include <metal_stdlib>
using namespace metal;

// Canonical NF4 Tile640 GEMV kernel.
//
// Buffer ABI:
//   [0] packed_weights  device const uchar*  raw Tile640 bytes
//   [1] scales          device const float*  FP32 group scales
//   [2] biases          device const float*  FP32 group biases
//   [3] in_vector       device const float*  activation vector [in_dim]
//   [4] out_vector      device float*        result vector
//   [5] num_macro_tiles constant uint        ceil(in_dim / 640)
//   [6] in_dim          constant uint        real (unpadded) input width
//
// Each threadgroup owns one output row and each SIMD lane reads one ushort
// from the 64-byte 128-element sub-tile payload.
//
// PARTIAL LAST TILE: when in_dim is not a multiple of 640 the packer zero-pads
// the tail (col >= in_dim → NF4 index 7 = 0.0), so those weights already
// contribute nothing. But `in_vector` is only [in_dim] long, so we MUST guard
// the activation read against `in_dim` to avoid an out-of-bounds load.
kernel void fused_gemv_nf4_tile640_fp32(
    device const uchar* packed_weights [[buffer(0)]],
    device const float* scales         [[buffer(1)]],
    device const float* biases         [[buffer(2)]],
    device const float* in_vector      [[buffer(3)]],
    device float* out_vector           [[buffer(4)]],
    constant uint& num_macro_tiles     [[buffer(5)]],
    constant uint& in_dim              [[buffer(6)]],
    uint row                           [[threadgroup_position_in_grid]],
    uint simd_lane                     [[thread_index_in_threadgroup]]
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
        uint weight_base = row_weight_base + tile_idx * BYTES_PER_TILE;

        for (uint group = 0; group < GROUPS_PER_TILE; ++group) {
            float scale = scales[meta_base + group];
            float bias = biases[meta_base + group];
            uint src_base = tile_idx * TILE + group * GROUP + simd_lane * LANE_VALUES;

            #pragma unroll
            for (uint i = 0; i < LANE_VALUES; ++i) {
                uint col = src_base + i;
                if (col >= in_dim) {
                    continue; // zero-padded tail of a partial last tile
                }
                float weight = fma(unpack_nf4(packed_weights, col), scale, bias);
                row_accumulator += weight * in_vector[col];
            }
        }
    }

    row_accumulator = simd_sum(row_accumulator);
    if (simd_lane == 0) {
        out_vector[row] = row_accumulator;
    }
}
