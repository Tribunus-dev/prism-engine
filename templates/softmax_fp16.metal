// Safe softmax — online softmax for arbitrary dimensions.
// Uses threadgroup memory for max and sum reduction.

#include <metal_stdlib>
using namespace metal;

kernel void softmax_fp16(
    device const half* input   [[buffer(0)]],  // [rows * dim]
    device half* output        [[buffer(1)]],  // [rows * dim]
    constant uint& dim         [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tgid [[threadgroup_position_in_grid]]
) {
    uint row = tgid.x;
    uint tid = gid.x;
    uint d = dim;

    // Online softmax: find max
    float max_val = -INFINITY;
    for (uint i = tid; i < d; i += 32) {
        max_val = max(max_val, float(input[row * d + i]));
    }
    // Reduction across SIMD group
    for (uint mask = 16; mask > 0; mask >>= 1) {
        max_val = max(max_val, simd_max(max_val));
    }
    threadgroup float tg_max;
    if (tid == 0) tg_max = max_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute sum of exp(x - max)
    float sum = 0.0;
    for (uint i = tid; i < d; i += 32) {
        sum += exp(float(input[row * d + i]) - tg_max);
    }
    for (uint mask = 16; mask > 0; mask >>= 1) {
        sum += simd_sum(sum);
    }
    threadgroup float tg_sum;
    if (tid == 0) tg_sum = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Normalize
    float inv_sum = 1.0 / (tg_sum + 1e-10);
    for (uint i = tid; i < d; i += 32) {
        output[row * d + i] = half(exp(float(input[row * d + i]) - tg_max) * inv_sum);
    }
}
