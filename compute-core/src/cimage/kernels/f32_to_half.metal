// SPDX-License-Identifier: MIT OR Apache-2.0
//
// GPU-side f32 ↔ half conversion kernels.
// Each thread converts one element.
//
// Dispatch: grid = (N, 1, 1), threads = (1, 1, 1)

#include <metal_stdlib>
using namespace metal;

kernel void cimage_f32_to_half(
    device const float *in    [[buffer(0)]],
    device half       *out    [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = half(in[id]);
}

kernel void cimage_half_to_f32(
    device const half  *in    [[buffer(0)]],
    device float       *out   [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = float(in[id]);
}
