#include <cuda_fp16.h>

// 3D convolution for video models, tiled across temporal axis
template<int KT, int KH, int KW>
__global__ void conv3d_kernel(
    const half* __restrict__ input,     // [N, C_in, F_in, H_in, W_in]
    const half* __restrict__ weight,    // [C_out, C_in, KT, KH, KW]
    half* __restrict__ output,          // [N, C_out, F_out, H_out, W_out]
    int N, int C_in, int F_in, int H_in, int W_in,
    int C_out, int F_out, int H_out, int W_out,
    int stride_t, int stride_h, int stride_w,
    int pad_t, int pad_h, int pad_w
) {
    int w_out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int f_out_idx = blockIdx.z * blockDim.z + threadIdx.z;

    if (w_out_idx >= W_out || h_out_idx >= H_out || f_out_idx >= F_out) return;

    for (int n = 0; n < N; ++n) {
        for (int c_out = 0; c_out < C_out; ++c_out) {
            float sum = 0.0f;
            
            for (int c_in = 0; c_in < C_in; ++c_in) {
                for (int kt = 0; kt < KT; ++kt) {
                    for (int kh = 0; kh < KH; ++kh) {
                        for (int kw = 0; kw < KW; ++kw) {
                            int in_t = f_out_idx * stride_t + kt - pad_t;
                            int in_h = h_out_idx * stride_h + kh - pad_h;
                            int in_w = w_out_idx * stride_w + kw - pad_w;

                            if (in_t >= 0 && in_t < F_in &&
                                in_h >= 0 && in_h < H_in &&
                                in_w >= 0 && in_w < W_in) {
                                
                                int in_idx = n * (C_in * F_in * H_in * W_in)
                                           + c_in * (F_in * H_in * W_in)
                                           + in_t * (H_in * W_in)
                                           + in_h * W_in
                                           + in_w;
                                           
                                int w_idx = c_out * (C_in * KT * KH * KW)
                                          + c_in * (KT * KH * KW)
                                          + kt * (KH * KW)
                                          + kh * KW
                                          + kw;
                                          
                                sum += __half2float(input[in_idx]) * __half2float(weight[w_idx]);
                            }
                        }
                    }
                }
            }
            
            int out_idx = n * (C_out * F_out * H_out * W_out)
                        + c_out * (F_out * H_out * W_out)
                        + f_out_idx * (H_out * W_out)
                        + h_out_idx * W_out
                        + w_out_idx;
                        
            output[out_idx] = __float2half(sum);
        }
    }
}
