// SiLU activation and vector addition — GPU native.
// Eliminates CPU round-trips from the MLP hot path.

#include <metal_stdlib>
using namespace metal;

/// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
kernel void silu_fp16(
    device half* data    [[buffer(0)]],
    constant uint& len   [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) return;
    float x = float(data[gid]);
    data[gid] = half(x / (1.0 + exp(-x)));
}

/// Vector add: a[i] = a[i] + b[i]
kernel void vec_add_fp16(
    device half* a         [[buffer(0)]],
    device const half* b   [[buffer(1)]],
    constant uint& len     [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) return;
    a[gid] = half(float(a[gid]) + float(b[gid]));
}
