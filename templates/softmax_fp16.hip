// Safe softmax — online softmax for arbitrary dimensions.
// Uses shared memory for max and sum reduction.
// HIP port of softmax_fp16.metal.
// Note: Metal simd_max/simd_sum are translated to warp shuffle reduction.

#include <hip/hip_runtime.h>

extern "C" __global__ void softmax_fp16(
    const half* __restrict__ input,  // [rows * dim]
    half* __restrict__ output,       // [rows * dim]
    uint dim
) {
    // One block per row, 32 threads per block (warp-sized)
    uint row = blockIdx.x;
    uint tid = threadIdx.x;
    uint d = dim;

    // Online softmax: find max
    float max_val = -__int_as_float(0x7f800000u);  // -INFINITY
    for (uint i = tid; i < d; i += 32) {
        max_val = fmax(max_val, __half2float(input[row * d + i]));
    }
    // Warp reduction for max using shuffle
    for (uint mask = 16; mask > 0; mask >>= 1) {
        float other = __shfl_down_sync(0xFFFFFFFF, max_val, mask);
        max_val = fmax(max_val, other);
    }
    __shared__ float tg_max;
    if (tid == 0) tg_max = max_val;
    __syncthreads();

    // Compute sum of exp(x - max)
    float sum = 0.0f;
    for (uint i = tid; i < d; i += 32) {
        sum += expf(__half2float(input[row * d + i]) - tg_max);
    }
    // Warp reduction for sum using shuffle
    for (uint mask = 16; mask > 0; mask >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, mask);
    }
    __shared__ float tg_sum;
    if (tid == 0) tg_sum = sum;
    __syncthreads();

    // Normalize
    float inv_sum = 1.0f / (tg_sum + 1e-10f);
    for (uint i = tid; i < d; i += 32) {
        output[row * d + i] = __float2half(expf(__half2float(input[row * d + i]) - tg_max) * inv_sum);
    }
}
