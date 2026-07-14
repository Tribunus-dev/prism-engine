#include <metal_stdlib>
using namespace metal;

/// Gemma 4 output projection (LM head) kernel.
kernel void gemma4_output_proj(
    device const float *in   [[buffer(0)]],
    device const float *w    [[buffer(1)]],
    device float *out        [[buffer(2)]],
    constant uint &d         [[buffer(3)]],
    uint tid                  [[thread_position_in_grid]]
) {
    float acc = 0.0;
    for (uint j = 0; j < d; ++j) acc += in[j] * w[tid * d + j];
    out[tid] = acc;
}
