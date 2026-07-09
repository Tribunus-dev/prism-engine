// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Ternary GEMV for CImage — unpacks 2-bit ternary codes (00=-1, 01=0, 10=+1,
// 11=reserved→0) and computes y = sum(W[i] * act[i]) * scale per group.
//
// Buffer layout:
//   [0] activations  [cols] half
//   [1] codes        [rows * groups_per_row * bytes_per_group] uchar
//   [2] scales       [rows * groups_per_row] half
//   [3] output       [rows] half
//   [4] constants    (TernaryGemvConstants)
//
// Thread-per-output-row dispatch.

#include <metal_stdlib>
using namespace metal;

struct TernaryGemvConstants {
    uint32_t rows;
    uint32_t cols;
    uint32_t group_size;
    uint32_t groups_per_row;
    uint32_t bytes_per_group;
    uint32_t output_dtype;
    uint32_t padding[3];
};

kernel void cimage_ternary_gemv_v1(
    device const half* activations           [[buffer(0)]],
    device const uchar* codes                [[buffer(1)]],
    device const half* scales                [[buffer(2)]],
    device half* output                      [[buffer(3)]],
    constant TernaryGemvConstants& c         [[buffer(4)]],
    uint row                                 [[thread_position_in_grid]]
) {
    if (row >= c.rows) return;

    // 00 = -1, 01 = 0, 10 = +1, 11 = reserved (treated as 0)
    float acc = 0.0;
    for (uint g = 0; g < c.groups_per_row; g++) {
        const float scale = float(scales[row * c.groups_per_row + g]);
        const uint group_byte_offset = row * c.groups_per_row * c.bytes_per_group
                                     + g * c.bytes_per_group;
        for (uint b = 0; b < c.bytes_per_group; b++) {
            const uchar byte = codes[group_byte_offset + b];
            // process 4 nibbles per byte
            for (uint n = 0; n < 4; n++) {
                const uint code = (byte >> (n * 2)) & 0x03;
                const uint weight_idx_in_group = b * 4 + n;
                if (weight_idx_in_group >= c.group_size) break;
                const uint col = g * c.group_size + weight_idx_in_group;
                if (col >= c.cols) break;
                const float act = float(activations[col]);
                float w;
                if (code == 0) w = -1.0;       // 00 = -1
                else if (code == 1) w = 0.0;    // 01 = 0
                else if (code == 2) w = 1.0;    // 10 = +1
                else w = 0.0;                    // 11 = reserved, treat as 0
                acc += w * act * scale;
            }
        }
    }
    output[row] = half(acc);
}
