#include <metal_stdlib>
using namespace metal;

/// Gemma 4 gate + up projection kernel.
kernel void gemma4_gate_up_proj(
    device const float *in   [[buffer(0)]],
    device const float *wg   [[buffer(1)]],
    device const float *wu   [[buffer(2)]],
    device float *gate       [[buffer(3)]],
    device float *up         [[buffer(4)]],
    constant uint &d         [[buffer(5)]],
    uint tid                  [[thread_position_in_grid]]
) {
    float g_acc = 0.0, u_acc = 0.0;
    for (uint j = 0; j < d; ++j) {
        g_acc += in[j] * wg[tid * d + j];
        u_acc += in[j] * wu[tid * d + j];
    }
    gate[tid] = g_acc;
    up[tid] = u_acc;
}
