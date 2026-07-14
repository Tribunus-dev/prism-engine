#include <metal_stdlib>
using namespace metal;

/// Gemma 4 QKV projection kernel.
kernel void gemma4_qkv_proj(
    device const float *in   [[buffer(0)]],
    device const float *wq   [[buffer(1)]],
    device const float *wk   [[buffer(2)]],
    device const float *wv   [[buffer(3)]],
    device float *q          [[buffer(4)]],
    device float *k          [[buffer(5)]],
    device float *v          [[buffer(6)]],
    constant uint &d         [[buffer(7)]],
    uint tid                  [[thread_position_in_grid]]
) {
    float acc = 0.0;
    for (uint j = 0; j < d; ++j) acc += in[j] * wq[tid * d + j];
    q[tid] = acc;
    acc = 0.0;
    for (uint j = 0; j < d; ++j) acc += in[j] * wk[tid * d + j];
    k[tid] = acc;
    acc = 0.0;
    for (uint j = 0; j < d; ++j) acc += in[j] * wv[tid * d + j];
    v[tid] = acc;
}
