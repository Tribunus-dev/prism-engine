#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

extern "C" __global__ void palettized_gemv(
    const half* __restrict__ input_vector,
    const half* __restrict__ codebook_block,
    const uint8_t* __restrict__ indices_block,
    half* __restrict__ output_vector,
    uint32_t in_dim,
    uint32_t out_dim)
{
    uint32_t row = blockIdx.x;
    uint32_t tid = threadIdx.x;

    // Point to this row's codebook and indices within the split blocks
    const half*    row_cb  = codebook_block + (row * 16);
    const uint8_t* row_idx = indices_block  + (row * (in_dim / 2u));

    // Collaborative codebook load into shared memory
    __shared__ half shared_cb[16];
    if (tid < 16) {
        shared_cb[tid] = row_cb[tid];
    }
    __syncthreads();

    // Vectorized index processing: 8 nibbles per uint32 read
    const uint32_t* idx_ptr = reinterpret_cast<const uint32_t*>(row_idx);
    uint32_t num_words = in_dim / 8u;
    half acc = (half)0.0f;

    for (uint32_t i = tid; i < num_words; i += blockDim.x) {
        uint32_t packed = idx_ptr[i];
        uint32_t off = i * 8u;

        half v0 = shared_cb[packed & 0x0Fu];
        half v1 = shared_cb[(packed >> 4) & 0x0Fu];
        half v2 = shared_cb[(packed >> 8) & 0x0Fu];
        half v3 = shared_cb[(packed >> 12) & 0x0Fu];
        half v4 = shared_cb[(packed >> 16) & 0x0Fu];
        half v5 = shared_cb[(packed >> 20) & 0x0Fu];
        half v6 = shared_cb[(packed >> 24) & 0x0Fu];
        half v7 = shared_cb[(packed >> 28)];

        acc +=         input_vector[off + 0u] * v0
             +         input_vector[off + 1u] * v1
             +         input_vector[off + 2u] * v2
             +         input_vector[off + 3u] * v3
             +         input_vector[off + 4u] * v4
             +         input_vector[off + 5u] * v5
             +         input_vector[off + 6u] * v6
             +         input_vector[off + 7u] * v7;
    }

    // Single-wavefront reduction (64-wide) via shuffle
    // After step 1, threads 0-31 hold [t]+[t+32]; after step 2, threads 0-15 hold [t]+[t+16]+[t+32]+[t+48];
    // etc. Thread 0 accumulates the full sum.
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 32);
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 16);
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 8);
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 4);
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 2);
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 1);

    // Thread 0 writes the final dot-product to output
    if (tid == 0) {
        output_vector[row] = acc;
    }
}
