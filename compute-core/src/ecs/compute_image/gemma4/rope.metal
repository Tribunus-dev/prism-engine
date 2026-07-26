#include <metal_stdlib>
using namespace metal;

/// Gemma 4 RoPE kernel.
kernel void gemma4_rope(
    device float *q          [[buffer(0)]],
    device float *k          [[buffer(1)]],
    constant uint &pos       [[buffer(2)]],
    constant uint &head_dim  [[buffer(3)]],
    uint tid                  [[thread_position_in_grid]]
) {
    if (tid >= head_dim) return;
    float theta = float(tid) * 0.5f;
    float cos_val = cos(float(pos) * exp2(-theta));
    float sin_val = sin(float(pos) * exp2(-theta));
    q[tid] = q[tid] * cos_val - q[tid ^ 1] * sin_val;
    k[tid] = k[tid] * cos_val - k[tid ^ 1] * sin_val;
}
