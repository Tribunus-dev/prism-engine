#include <metal_stdlib>
using namespace metal;

// [[kernel]] palettized_gemv — fused LUT dequant + dot product (split-block).
// One threadgroup (64 threads) per output channel.
//
// PRECISION: fp16 I/O (bandwidth), fp32 dequant + fp32 accumulation + fp32
// reduction, fp16 store. Apple GPUs execute fp32 at full rate and this GEMV is
// memory-bandwidth-bound, so fp32 accumulation costs ~nothing and is strictly
// more accurate than the previous fp16 accumulator (which lost precision
// summing thousands of terms). ABI is unchanged — this is a drop-in replacement.
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
    device const half*    row_cb  = codebook_block + (row * 16);
    device const uint8_t* row_idx = indices_block  + (row * (in_dim / 2));

    // Codebook promoted to fp32 in threadgroup memory: the dequantized LUT
    // values feed an fp32 MAC, so we widen once here rather than per-lookup.
    threadgroup float shared_cb[16];
    if (tid < 16) {
        shared_cb[tid] = float(row_cb[tid]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Vectorized index processing: 8 nibbles per uint32 read.
    device const uint32_t* idx_ptr =
        reinterpret_cast<device const uint32_t*>(row_idx);
    uint32_t num_words = in_dim / 8;
    float acc = 0.0f;

    for (uint32_t i = tid; i < num_words; i += 64) {
        uint32_t packed = idx_ptr[i];
        uint32_t off = i * 8;

        // Interleaved fused dequant + MAC, accumulated in fp32.
        acc = fma(float(input_vector[off + 0]), shared_cb[ packed        & 0x0F], acc);
        acc = fma(float(input_vector[off + 1]), shared_cb[(packed >>  4)  & 0x0F], acc);
        acc = fma(float(input_vector[off + 2]), shared_cb[(packed >>  8)  & 0x0F], acc);
        acc = fma(float(input_vector[off + 3]), shared_cb[(packed >> 12)  & 0x0F], acc);
        acc = fma(float(input_vector[off + 4]), shared_cb[(packed >> 16)  & 0x0F], acc);
        acc = fma(float(input_vector[off + 5]), shared_cb[(packed >> 20)  & 0x0F], acc);
        acc = fma(float(input_vector[off + 6]), shared_cb[(packed >> 24)  & 0x0F], acc);
        acc = fma(float(input_vector[off + 7]), shared_cb[(packed >> 28)  & 0x0F], acc);
    }

    // fp32 SIMD-group reduction (fast hardware shuffle).
    acc = simd_sum(acc);

    // Inter-SIMD reduction via an fp32 threadgroup scratchpad.
    threadgroup float shared_reduction[32];
    if (simd_lane == 0) {
        shared_reduction[simd_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        uint32_t nsimd = (64 + 31) / 32; // threads per group / SIMD width
        float total = 0.0f;
        for (uint32_t s = 0; s < nsimd; ++s) total += shared_reduction[s];
        output_vector[row] = half(total); // store fp16 (switch to a float
                                           // buffer if you need fp32 output)
    }
}
