#include <metal_stdlib>
using namespace metal;

// INT8 Tile640 GEMV kernel.
//
// Buffer ABI (matches host-side Int8Tile640GEMVDispatcher):
//   [0] packed_weights   device const char*   raw Tile640 i8 bytes (640 bytes/tile)
//   [1] scales           device const float*  FP32 per-tile scales
//   [2] bias_padding     device const float*  FP32 per-tile biases (always 0 — INT8
//                                               stores zero entries here to keep the
//                                               tile-metadata stride identical to NF4)
//   [3] in_vector        device const float*  activation vector [in_dim]
//   [4] out_vector       device float*        result vector
//   [5] num_macro_tiles  constant uint        ceil(in_dim / 640)
//   [6] in_dim           constant uint        real (unpadded) input width
//   [7] reduction_scales device const half*   FP16 column-scale sidecar (optional,
//                                               null buffer = none)
//
// INT8: each tile is 640 bytes (one i8 per element).  One scale per tile;
// bias_padding entries are always zero so the tile-metadata segment stride
// matches NF4's (5 groups × 2 floats × 4 bytes = 40 bytes/tile for NF4, with
// bias_padding providing the dummy 4-byte entries to keep the same 40-byte
// stride — see pack_int8_weights for the buffer layout).
//
// Scale for row r tile t: scales[r * num_macro_tiles + t]
//
// With reduction-axis scales (reduction_scales != null):
//   acc += weight * (reduction_scales[col] * activation[col])
// Without:
//   acc += weight * activation[col]
//
// PARTIAL LAST TILE: zero-padded by the packer, but we still guard the
// activation and reduction_scales reads against in_dim to stay within bounds.
kernel void fused_gemv_int8_tile640_fp32(
    device const char* packed_weights        [[buffer(0)]],
    device const float* scales               [[buffer(1)]],
    device const float* bias_padding         [[buffer(2)]],
    device const float* in_vector            [[buffer(3)]],
    device float* out_vector                 [[buffer(4)]],
    constant uint& num_macro_tiles           [[buffer(5)]],
    constant uint& in_dim                    [[buffer(6)]],
    device const half* reduction_scales      [[buffer(7)]],
    uint row                                 [[threadgroup_position_in_grid]],
    uint tid                                 [[thread_position_in_threadgroup]]
) {
    constexpr uint TILE = 640;
    float acc = 0.0f;
    uint row_weight_base = row * num_macro_tiles * TILE; // 640 bytes per tile

    for (uint t = 0; t < num_macro_tiles; ++t) {
        float scale = scales[row * num_macro_tiles + t];
        device const char* tile = packed_weights + row_weight_base + t * TILE;

        for (uint i = tid; i < TILE; i += 32) {
            uint col = t * TILE + i;
            if (col >= in_dim) continue;
            float weight = scale * (float)tile[i];
            float act = reduction_scales
                ? (float)reduction_scales[col] * in_vector[col]
                : in_vector[col];
            acc += weight * act;
        }
    }

    float result = simd_sum(acc);
    if (tid == 0) {
        out_vector[row] = result;
    }
}
