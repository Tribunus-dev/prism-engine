#include <metal_stdlib>
using namespace metal;

/// Gemma 4 SiLU activation kernel.
kernel void gemma4_silu(
    device const float *x    [[buffer(0)]],
    device float *out        [[buffer(1)]],
    uint tid                  [[thread_position_in_grid]]
) {
    float v = x[tid];
    out[tid] = v / (1.0 + exp(-v));
}
