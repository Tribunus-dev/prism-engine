#include <metal_stdlib>
using namespace metal;

// 3D convolution for video models
kernel void conv3d_kernel(
    device const half* input [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& N [[buffer(3)]],
    constant uint& C_in [[buffer(4)]],
    constant uint& F_in [[buffer(5)]],
    constant uint& H_in [[buffer(6)]],
    constant uint& W_in [[buffer(7)]],
    constant uint& C_out [[buffer(8)]],
    constant uint& F_out [[buffer(9)]],
    constant uint& H_out [[buffer(10)]],
    constant uint& W_out [[buffer(11)]],
    constant uint& KT [[buffer(12)]],
    constant uint& KH [[buffer(13)]],
    constant uint& KW [[buffer(14)]],
    constant uint& stride_t [[buffer(15)]],
    constant uint& stride_h [[buffer(16)]],
    constant uint& stride_w [[buffer(17)]],
    constant uint& pad_t [[buffer(18)]],
    constant uint& pad_h [[buffer(19)]],
    constant uint& pad_w [[buffer(20)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint w_out_idx = gid.x;
    uint h_out_idx = gid.y;
    uint f_out_idx = gid.z;

    if (w_out_idx >= W_out || h_out_idx >= H_out || f_out_idx >= F_out) return;

    for (uint n = 0; n < N; ++n) {
        for (uint c_out = 0; c_out < C_out; ++c_out) {
            float sum = 0.0;
            
            for (uint c_in = 0; c_in < C_in; ++c_in) {
                for (uint kt = 0; kt < KT; ++kt) {
                    for (uint kh = 0; kh < KH; ++kh) {
                        for (uint kw = 0; kw < KW; ++kw) {
                            int in_t = (int)(f_out_idx * stride_t + kt) - (int)pad_t;
                            int in_h = (int)(h_out_idx * stride_h + kh) - (int)pad_h;
                            int in_w = (int)(w_out_idx * stride_w + kw) - (int)pad_w;

                            if (in_t >= 0 && in_t < (int)F_in &&
                                in_h >= 0 && in_h < (int)H_in &&
                                in_w >= 0 && in_w < (int)W_in) {
                                
                                uint in_idx = n * (C_in * F_in * H_in * W_in)
                                            + c_in * (F_in * H_in * W_in)
                                            + (uint)in_t * (H_in * W_in)
                                            + (uint)in_h * W_in
                                            + (uint)in_w;
                                            
                                uint w_idx = c_out * (C_in * KT * KH * KW)
                                           + c_in * (KT * KH * KW)
                                           + kt * (KH * KW)
                                           + kh * KW
                                           + kw;
                                           
                                sum += (float)input[in_idx] * (float)weight[w_idx];
                            }
                        }
                    }
                }
            }
            
            uint out_idx = n * (C_out * F_out * H_out * W_out)
                         + c_out * (F_out * H_out * W_out)
                         + f_out_idx * (H_out * W_out)
                         + h_out_idx * W_out
                         + w_out_idx;
                         
            output[out_idx] = (half)sum;
        }
    }
}
