#include <metal_stdlib>
using namespace metal;

// [[kernel]] palettized_gemv — fused LUT dequant + dot product (split-block)
// One threadgroup (64 threads) per output channel.
//
// buffer(0): input_vector  [in_dim] half
// buffer(1): codebook_block [out_dim * 16] half    (all codebooks contiguous)
// buffer(2): indices_block  [out_dim * in_dim/2] u8 (all indices contiguous)
// buffer(3): output_vector [out_dim] half
// buffer(4): in_dim uint
// buffer(5): out_dim uint
kernel void palettized_gemv(
    device const half*    input_vector    [[buffer(0)]],
    device const half*    codebook_block  [[buffer(1)]],
    device const uint8_t* indices_block   [[buffer(2)]],
    device half*          output_vector   [[buffer(3)]],
    constant uint32_t&    in_dim          [[buffer(4)]],
    constant uint32_t&    out_dim         [[buffer(5)]],
    uint32_t row                          [[threadgroup_position_in_grid]],
    uint32_t tid                          [[thread_position_in_threadgroup]],
    uint32_t simd_lane                    [[thread_index_in_simdgroup]],
    uint32_t simd_id                      [[simdgroup_index_in_threadgroup]])
{
    // Point to this row's codebook and indices within the split blocks
    device const half*    row_cb  = codebook_block + (row * 16);
    device const uint8_t* row_idx = indices_block  + (row * (in_dim / 2));

    // Collaborative codebook load into threadgroup memory
    threadgroup half shared_cb[16];
    if (tid < 16) {
        shared_cb[tid] = row_cb[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Vectorized index processing: 8 nibbles per uint32 read
    device const uint32_t* idx_ptr =
        reinterpret_cast<device const uint32_t*>(row_idx);
    uint32_t num_words = in_dim / 8;
    half acc = 0.0h;

    for (uint32_t i = tid; i < num_words; i += 64) {
        uint32_t packed = idx_ptr[i];
        uint32_t off = i * 8;

        half v0 = shared_cb[packed & 0x0F];
        half v1 = shared_cb[(packed >> 4) & 0x0F];
        half v2 = shared_cb[(packed >> 8) & 0x0F];
        half v3 = shared_cb[(packed >> 12) & 0x0F];
        half v4 = shared_cb[(packed >> 16) & 0x0F];
        half v5 = shared_cb[(packed >> 20) & 0x0F];
        half v6 = shared_cb[(packed >> 24) & 0x0F];
        half v7 = shared_cb[(packed >> 28)];

        acc += input_vector[off + 0] * v0
            +  input_vector[off + 1] * v1
            +  input_vector[off + 2] * v2
            +  input_vector[off + 3] * v3
            +  input_vector[off + 4] * v4
            +  input_vector[off + 5] * v5
            +  input_vector[off + 6] * v6
            +  input_vector[off + 7] * v7;
    }

    // SIMD-group reduction (fast hardware shuffle)
    acc = simd_sum(acc);

    // Inter-SIMD reduction via shared memory scratchpad
    threadgroup half shared_reduction[2];
    if (simd_lane == 0) {
        shared_reduction[simd_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Thread 0 writes the final dot-product to output
    if (tid == 0) {
        output_vector[row] = shared_reduction[0] + shared_reduction[1];
    }
}
