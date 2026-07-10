// RMS normalization — proper threadgroup binary-tree reduction.
// HIP port of rms_norm.metal.

#include <hip/hip_runtime.h>

extern "C" __global__ void rms_norm(
    const half* __restrict__ input,
    half* __restrict__ output,
    uint dim,
    float eps
) {
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint tg_size = blockDim.x;

    // Thread-local sum of squares
    float ssq = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        float v = __half2float(input[i]);
        ssq += v * v;
    }

    // Threadgroup reduction: binary tree in shared memory
    __shared__ float tg_partial[256];
    tg_partial[threadIdx.x] = ssq;
    __syncthreads();

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            tg_partial[threadIdx.x] += tg_partial[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float total_ssq = tg_partial[0];
    float inv_rms = 1.0f / sqrt(total_ssq / float(dim) + eps);

    // Apply normalization
    for (uint i = tid; i < dim; i += tg_size) {
        output[i] = __float2half(__half2float(input[i]) * inv_rms);
    }
}
