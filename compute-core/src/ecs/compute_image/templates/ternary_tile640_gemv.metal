// Ternary decode follows the canonical sequential-extraction pattern documented
// in fragments/ternary_decode.metal (uses built-in /3/%3, not fast_div3/fast_mod3).
//
#include <metal_stdlib>
using namespace metal;

// [[kernel]] ternary_tile640_gemv — fused 1.6-bit ternary GEMV, v7 two-level
// scales. Consumes the tile640 packing (20 base-3 trits/u32 = 1.6 bpw) and does
// fp32 dequant with TWO-LEVEL micro-scales: one bf16 page-max per 640-page plus
// one int8 relative scale per 20-weight lane (~2.0 bpw of scale overhead, and
// it neutralizes the fp16 scale overflow/underflow bug). One micro-scale is
// applied per unpacked word → zero register-shuffle overhead.
// (Layout + math verified by tools/quant_lab.rs and kernels/oracle.)
//
// buffer(0): packed       [out_dim * nt * 32] uint     (nt = ceil(in_dim/640))
// buffer(1): input_vector [in_dim] half
// buffer(2): page_scales  [out_dim * nt] ushort        (bf16 bits, per page)
// buffer(3): lane_scales  [out_dim * nt * 32] uchar     (int8 relative, per lane)
// buffer(4): output_vector[out_dim] half
// buffer(5): in_dim  uint
// buffer(6): out_dim uint
// One threadgroup (64 threads) per output row.
kernel void ternary_tile640_gemv(
    device const uint*   packed        [[buffer(0)]],
    device const half*   input_vector  [[buffer(1)]],
    device const ushort* page_scales   [[buffer(2)]],
    device const uchar*  lane_scales   [[buffer(3)]],
    device half*         output_vector [[buffer(4)]],
    constant uint&       in_dim        [[buffer(5)]],
    constant uint&       out_dim       [[buffer(6)]],
    uint row       [[threadgroup_position_in_grid]],
    uint tid       [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    if (row >= out_dim) return;

    const uint nt            = (in_dim + 639) / 640;
    const uint words_per_row = nt * 32;
    device const uint* row_pack = packed + row * words_per_row;

    float acc = 0.0f;
    for (uint wi = tid; wi < words_per_row; wi += 64) {
        const uint p    = wi / 32;             // page within row
        const uint lane = wi % 32;             // lane within page
        const uint col0 = p * 640 + lane * 20;

        // Two-level scale: bf16 page-max × (int8 lane / 127), reconstructed fp32.
        const float page_max = as_type<float>(uint(page_scales[row * nt + p]) << 16);
        const float scale    = page_max * (float(lane_scales[row * words_per_row + wi]) * (1.0f / 127.0f));

        uint rem = row_pack[wi];
        for (uint vi = 0; vi < 20; ++vi) {
            const uint d = rem % 3u; rem /= 3u;   // 0,1,2  (LSB trit first)
            const uint col = col0 + vi;
            if (col >= in_dim) break;
            if (d != 0u) {
                const float tv = (d == 1u) ? scale : -scale;  // +scale / -scale
                acc = fma(float(input_vector[col]), tv, acc);
            }
        }
    }

    acc = simd_sum(acc);
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

// [[kernel]] ternary_outlier_addback — adds the sparse bf16 outliers back into
// the dense result: output[row] += Σ input[col] * value, for each extracted
// weight. Outliers are <1% of weights, so this is a tiny second pass. Requires
// Metal 3 atomic<float> (Apple GPU family 7+). Launch: one thread per outlier.
//
// buffer(0): out_rows  [n_outliers] uint   (row of each outlier)
// buffer(1): out_cols  [n_outliers] uint   (col of each outlier)
// buffer(2): out_vals  [n_outliers] ushort (bf16 bits)
// buffer(3): input_vector [in_dim] half
// buffer(4): output_vector[out_dim] atomic<float>  (dense result, fp32)
// buffer(5): n_outliers uint
kernel void ternary_outlier_addback(
    device const uint*         out_rows [[buffer(0)]],
    device const uint*         out_cols [[buffer(1)]],
    device const ushort*       out_vals [[buffer(2)]],
    device const half*         input_vector  [[buffer(3)]],
    device atomic_float*       output_vector [[buffer(4)]],
    constant uint&             n_outliers    [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n_outliers) return;
    const float val = as_type<float>(uint(out_vals[gid]) << 16); // bf16 → fp32
    const float contrib = float(input_vector[out_cols[gid]]) * val;
    atomic_fetch_add_explicit(&output_vector[out_rows[gid]], contrib, memory_order_relaxed);
}
