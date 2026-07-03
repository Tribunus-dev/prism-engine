#include <metal_stdlib>
using namespace metal;

// [[kernel]] ternary_tile640_gemv — fused 1.6-bit ternary GEMV.
//
// Consumes the compiler's space-optimal tile640 packing (20 base-3 trits per
// 32-bit word = 1.6 bits/weight, ≈ the log2(3)=1.585 information limit) and
// does PROPER fp32 dequantization: trit ∈ {0,+1,-1} × per-256-block scale,
// accumulated in fp32. This is the scaled counterpart to the legacy 2-bit
// ternary_gemv.metal (which is 2.0 bits/weight and scale-free).
//
// tile640 layout — matches repack_ternary_to_swizzled_u8 / decode_ternary_u32:
//   per row: nt = ceil(in_dim/640) tiles; per (tile t, lane 0..31) one u32
//   packs 20 trits (least-significant trit first); global column index is
//   col = t*640 + lane*20 + vi; trit_vi = (word / 3^vi) % 3, decoded
//   iteratively as rem%3 then rem/=3; digit 0->0, 1->+1, 2->-1.
//   Block scale is one fp16 per 256 flattened (row-major) weights.
//   (Verified by kernels/oracle/tile640_oracle.cpp: max|ref-kernel| ≈ 6e-7.)
//
// buffer(0): packed        [out_dim * nt * 32] uint   (nt = ceil(in_dim/640))
// buffer(1): input_vector  [in_dim] half
// buffer(2): block_scales  [ceil(out_dim*in_dim/256)] half
// buffer(3): output_vector [out_dim] half
// buffer(4): in_dim  uint
// buffer(5): out_dim uint
// One threadgroup (64 threads) per output row.
kernel void ternary_tile640_gemv(
    device const uint*  packed        [[buffer(0)]],
    device const half*  input_vector  [[buffer(1)]],
    device const half*  block_scales  [[buffer(2)]],
    device half*        output_vector [[buffer(3)]],
    constant uint&      in_dim        [[buffer(4)]],
    constant uint&      out_dim       [[buffer(5)]],
    uint row       [[threadgroup_position_in_grid]],
    uint tid       [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    if (row >= out_dim) return;

    const uint nt            = (in_dim + 639) / 640;
    const uint words_per_row = nt * 32;
    device const uint* row_pack = packed + row * words_per_row;
    const uint row_base = row * in_dim; // flattened base for scale indexing

    float acc = 0.0f;

    // Each thread strides over this row's (tile,lane) words.
    for (uint w = tid; w < words_per_row; w += 64) {
        const uint t    = w / 32;
        const uint lane = w % 32;
        const uint col0 = t * 640 + lane * 20;

        uint rem = row_pack[w];
        // Unpack 20 base-3 trits; interleaved fused dequant + MAC in fp32.
        for (uint vi = 0; vi < 20; ++vi) {
            const uint d = rem % 3u;   // 0,1,2  (least-significant trit first)
            rem /= 3u;
            const uint col = col0 + vi;
            if (col >= in_dim) break;  // padding trits past the row end

            if (d != 0u) {
                const float tv    = (d == 1u) ? 1.0f : -1.0f;      // +1 / -1
                const float scale = float(block_scales[(row_base + col) >> 8]); // /256
                acc = fma(float(input_vector[col]) * scale, tv, acc);
            }
        }
    }

    acc = simd_sum(acc); // fp32 reduction

    threadgroup float shared_reduction[32];
    if (simd_lane == 0) shared_reduction[simd_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        const uint nsimd = (64 + 31) / 32;
        float total = 0.0f;
        for (uint s = 0; s < nsimd; ++s) total += shared_reduction[s];
        output_vector[row] = half(total);
    }
}
