// RMS normalization — proper threadgroup binary-tree reduction.
// Eliminates dependence on Metal simd_sum (unreliable across Metal versions).

#include <metal_stdlib>
using namespace metal;

kernel void rms_norm(
    device const half* input    [[buffer(0)]],
    device half* output         [[buffer(1)]],
    constant uint& dim          [[buffer(2)]],
    constant float& eps         [[buffer(3)]],
    uint tid [[thread_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]]
) {
    // Thread-local sum of squares
    float ssq = 0.0;
    for (uint i = tid; i < dim; i += tg_size) {
        float v = float(input[i]);
        ssq += v * v;
    }

    // Threadgroup reduction: binary tree in shared memory
    threadgroup float tg_partial[256];
    tg_partial[tid] = ssq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            tg_partial[tid] += tg_partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float total_ssq = tg_partial[0];
    float inv_rms = 1.0 / sqrt(total_ssq / float(dim) + eps);

    // Apply normalization
    for (uint i = tid; i < dim; i += tg_size) {
        output[i] = half(float(input[i]) * inv_rms);
    }
}
