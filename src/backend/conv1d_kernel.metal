#include <metal_stdlib>
using namespace metal;

// 1D convolution for audio models
template<int KERNEL_SIZE>
kernel void conv1d_kernel(
    device const half* input [[buffer(0)]],      // [N, C, T]
    device const half* weight [[buffer(1)]],     // [C_out, C_in, K]
    device const half* bias [[buffer(2)]],       // [C_out]
    device half* output [[buffer(3)]],           // [N, C_out, T_out]
    constant uint& N [[buffer(4)]],
    constant uint& C [[buffer(5)]],
    constant uint& T [[buffer(6)]],
    constant uint& C_out [[buffer(7)]],
    constant uint& stride [[buffer(8)]],
    constant uint& padding [[buffer(9)]],
    constant uint& dilation [[buffer(10)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint oc = gid.x;
    uint t_out = gid.y;
    uint n = gid.z;
    
    uint T_out = (T + 2 * padding - dilation * (KERNEL_SIZE - 1) - 1) / stride + 1;
    
    if (n >= N || oc >= C_out || t_out >= T_out) return;
    
    float sum = 0.0;
    
    for (uint c = 0; c < C; c++) {
        for (int k = 0; k < KERNEL_SIZE; k++) {
            int t_in = (int)(t_out * stride) - (int)padding + k * (int)dilation;
            if (t_in >= 0 && t_in < (int)T) {
                sum += (float)input[n * C * T + c * T + t_in] * (float)weight[oc * C * KERNEL_SIZE + c * KERNEL_SIZE + k];
            }
        }
    }
    
    output[n * C_out * T_out + oc * T_out + t_out] = (half)(sum + (bias ? (float)bias[oc] : 0.0f));
}
