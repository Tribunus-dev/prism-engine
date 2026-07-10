// SiLU activation and vector addition — GPU native.
// HIP port of silu_vec.metal.

#include <hip/hip_runtime.h>

/// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
extern "C" __global__ void silu_fp16(
    half* __restrict__ data,
    uint len
) {
    uint gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= len) return;
    float x = __half2float(data[gid]);
    data[gid] = __float2half(x / (1.0f + expf(-x)));
}

/// Vector add: a[i] = a[i] + b[i]
extern "C" __global__ void vec_add_fp16(
    half* __restrict__ a,
    const half* __restrict__ b,
    uint len
) {
    uint gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= len) return;
    a[gid] = __float2half(__half2float(a[gid]) + __half2float(b[gid]));
}
