// RMS normalization — SIMD group reduction for sum of squares, then threadgroup broadcast.

#include <metal_stdlib>
using namespace metal;

kernel void rms_norm(
    device const half* input    [[buffer(0)]],  // [rows * dim]
    device half* output         [[buffer(1)]],  // [rows * dim]
    constant uint& dim          [[buffer(2)]],
    constant float& eps         [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tgid [[threadgroup_position_in_grid]]
) {
    uint row = tgid.x;
    uint tid = gid.x;
    uint d = dim;

    // Thread-local sum of squares
    float ssq = 0.0;
    for (uint i = tid; i < d; i += 32) {
        float v = float(input[row * d + i]);
        ssq += v * v;
    }

    // SIMD group sum (all 32 threads get the same result)
    ssq = simd_sum(ssq);

    // First thread writes to threadgroup memory for broadcast
    threadgroup float tg_ssq;
    if (tid == 0) {
        tg_ssq = ssq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float mean_sq = tg_ssq / float(d);
    float inv_rms = 1.0 / sqrt(mean_sq + eps);

    // Apply normalization
    for (uint i = tid; i < d; i += 32) {
        output[row * d + i] = half(float(input[row * d + i]) * inv_rms);
    }
}
