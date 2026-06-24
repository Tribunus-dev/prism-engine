#define WARP_SIZE 32
#define MAX_CHANNELS 1024

// 1D convolution for audio models, banked in shared memory
template<int KERNEL_SIZE>
__global__ void conv1d_kernel(
    const half* __restrict__ input,      // [N, C, T]
    const half* __restrict__ weight,     // [C_out, C_in, K]
    const half* __restrict__ bias,       // [C_out]
    half* __restrict__ output,           // [N, C_out, T_out]
    int N, int C, int T, int C_out, int stride, int padding, int dilation
) {
    int n = blockIdx.z;
    int oc = blockIdx.x;
    int t_out = blockIdx.y * blockDim.x + threadIdx.x;
    
    // Simplistic convolution implementation iterating over batch and time
    if (n < N && oc < C_out && t_out < (T + 2 * padding - dilation * (KERNEL_SIZE - 1) - 1) / stride + 1) {
        float sum = 0.0f;
        
        // Loop over input channels
        for (int c = 0; c < C; c++) {
            for (int k = 0; k < KERNEL_SIZE; k++) {
                int t_in = t_out * stride - padding + k * dilation;
                if (t_in >= 0 && t_in < T) {
                    float in_val = __half2float(input[n * C * T + c * T + t_in]);
                    float w_val = __half2float(weight[oc * C * KERNEL_SIZE + c * KERNEL_SIZE + k]);
                    sum += in_val * w_val;
                }
            }
        }
        
        output[n * C_out * ((T + 2 * padding - dilation * (KERNEL_SIZE - 1) - 1) / stride + 1) + oc * ((T + 2 * padding - dilation * (KERNEL_SIZE - 1) - 1) / stride + 1) + t_out] = __float2half(sum + (bias ? __half2float(bias[oc]) : 0.0f));
    }
}
