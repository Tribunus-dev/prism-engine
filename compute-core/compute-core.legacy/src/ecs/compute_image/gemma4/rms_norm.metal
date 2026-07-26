#include <metal_stdlib>
using namespace metal;

/// Gemma 4 RMSNorm kernel.
kernel void gemma4_rms_norm(
    device const float *in  [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device float *out       [[buffer(2)]],
    constant uint &n         [[buffer(3)]],
    uint tid                 [[thread_position_in_grid]]
) {
    if (tid >= n) return;
    // Simplified RMSNorm: out = in * weight / rms(in)
    float sum_sq = 0.0;
    for (uint i = 0; i < n; ++i) sum_sq += in[i] * in[i];
    float rms = sqrt(sum_sq / float(n) + 1e-6);
    out[tid] = in[tid] * weight[tid % n] / rms;
}
