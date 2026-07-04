#include <metal_stdlib>
using namespace metal;

constant float nf4_table_fp32[16] = {
    -1.0f, -0.8480f, -0.5698f, -0.3940f,
    -0.2419f, -0.1057f, 0.0f, 0.1057f,
    0.2419f, 0.3940f, 0.5698f, 0.8480f,
    1.0f, 1.2588f, 1.5862f, 2.0f
};

// Canonical NF4 Tile640 GEMV kernel.
//
// Buffer ABI:
//   [0] packed_weights  device const uchar*  raw Tile640 bytes
//   [1] scales          device const float*  FP32 group scales
//   [2] biases          device const float*  FP32 group biases
//   [3] in_vector       device const float*  activation vector
//   [4] out_vector      device float*        result vector
//
// Each threadgroup owns one output row and each SIMD lane reads one ushort
// from the 64-byte 128-element sub-tile payload.
kernel void fused_gemv_nf4_tile640_fp32(
    device const uchar* packed_weights [[buffer(0)]],
    device const float* scales         [[buffer(1)]],
    device const float* biases         [[buffer(2)]],
    device const float* in_vector      [[buffer(3)]],
    device float* out_vector           [[buffer(4)]],
    constant uint& num_macro_tiles     [[buffer(5)]],
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
            device const ushort* packed_chunk =
                (device const ushort*)(packed_weights + weight_base + group * 64);
            ushort raw_bits = packed_chunk[simd_lane];
            uint src_base = tile_idx * TILE + group * GROUP + simd_lane * LANE_VALUES;

            #pragma unroll
            for (uint i = 0; i < LANE_VALUES; ++i) {
                uint nibble = (raw_bits >> (i * 4)) & 0x0Fu;
                float weight = scale * nf4_table_fp32[nibble] + bias;
                row_accumulator += weight * in_vector[src_base + i];
            }
        }
    }

    row_accumulator = simd_sum(row_accumulator);
    if (simd_lane == 0) {
        out_vector[row] = row_accumulator;
    }
}
