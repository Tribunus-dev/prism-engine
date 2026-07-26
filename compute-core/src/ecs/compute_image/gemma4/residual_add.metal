#include <metal_stdlib>
using namespace metal;

/// Gemma 4 residual add kernel.
kernel void gemma4_residual_add(
    device const float *a    [[buffer(0)]],
    device const float *b    [[buffer(1)]],
    device float *out        [[buffer(2)]],
    uint tid                  [[thread_position_in_grid]]
) {
    out[tid] = a[tid] + b[tid];
}
