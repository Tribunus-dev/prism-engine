// Rotary position embedding — in-place sin/cos per head.
// Each thread handles one element.

#include <metal_stdlib>
using namespace metal;

kernel void rope_fp16(
    device half* x               [[buffer(0)]],  // [num_heads * head_dim]
    constant int64_t& pos        [[buffer(1)]],
    constant uint& head_dim      [[buffer(2)]],
    constant float& rope_theta   [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint i = gid.x;
    uint hd = head_dim;
    uint half_hd = hd / 2;
    uint elem_in_head = i % hd;

    if (elem_in_head >= half_hd) return;  // only process first half of each head

    uint head = i / hd;
    uint j = elem_in_head;
    float angle = float(pos) * pow(rope_theta, -2.0 * float(j) / float(hd));
    float sin_a, cos_a;
    sin_a = sin(angle);
    cos_a = cos(angle);

    float a = float(x[head * hd + j]);
    float b = float(x[head * hd + j + half_hd]);

    x[head * hd + j] = half(a * cos_a - b * sin_a);
    x[head * hd + j + half_hd] = half(a * sin_a + b * cos_a);
}
