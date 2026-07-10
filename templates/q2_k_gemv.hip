// Q2_K GEMV: 256-element super-blocks, 4xf16 scales, 64 bytes packed 2-bit weights.
// GGML K-quant 2-bit: each super-block holds 256 coefficients, a 2-bit value per
// coefficient (0-3), and four f16 scales that each cover 64 consecutive elements.
// Total payload: 72 bytes per super-block (2.25 bits/element).
//
// Template placeholders (compile-time):
//   TILE_WIDTH  — columns processed cooperatively
//   GROUP_SIZE  — rows per threadgroup (256 = one super-block)
//   THREADGROUP_SIZE — threads per group (typically 64/128/256)

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

// ── Q2_K super-block structure ───────────────────────────────────────────
//   Bytes    Contents
//   0-7      4 x f16 scales (each covering 64 elements)
//   8-71     64 bytes packed 2-bit weights (4 per byte)
//   =72      total

#define Q2K_SCALE_COUNT   4
#define Q2K_BLOCK_SIZE    256
#define Q2K_SCALES_BYTES  (Q2K_SCALE_COUNT * sizeof(half))   // 8
#define Q2K_QS_BYTES      64
#define Q2K_SUPER_BLOCK_SIZE (Q2K_SCALES_BYTES + Q2K_QS_BYTES) // 72

extern "C" __global__ void q2_k_gemv(
    const half* __restrict__ input_vector,   // [in_dim]
    const uint8_t* __restrict__ weights,     // [out_dim * Q2K_SUPER_BLOCK_SIZE / 4]
    const half* __restrict__ scales,         // [out_dim * Q2K_SCALE_COUNT]
    half* __restrict__ output_vector,         // [out_dim]
    uint32_t in_dim,
    uint32_t out_dim)
{
    uint32_t row = blockIdx.x;
    uint32_t tid = threadIdx.x;

    // ── Bail if row is out of bounds ──────────────────────────────────────
    if (row >= out_dim) return;

    // ── Locate this row's Q2_K super-block ────────────────────────────────
    uint32_t sb_stride = Q2K_SUPER_BLOCK_SIZE / 4;  // packed uint32 stride
    const uint32_t* sb_base = reinterpret_cast<const uint32_t*>(weights)
                            + row * sb_stride;

    // ── Load the 4 f16 scales ────────────────────────────────────────────
    half4 scale;
    const uint32_t* scale_base = reinterpret_cast<const uint32_t*>(&scales[row * Q2K_SCALE_COUNT]);
    reinterpret_cast<uint32_t*>(&scale)[0] = scale_base[0];
    reinterpret_cast<uint32_t*>(&scale)[1] = scale_base[1];

    // ── Dot product over packed uint32 words ──────────────────────────────
    //    Each uint32 holds 4 packed bytes = 16 coefficients (4 per byte, 2-bit each).
    //    256 coefficients / 16 per uint32 = 16 uint32 reads.
    //    With 4 scales, each scale covers 64 coefficients = 4 uint32.
    half acc = (half)0.0f;
    const uint32_t words_per_row = Q2K_QS_BYTES / sizeof(uint32_t);  // 16

    // Threads cooperatively cover the words
    for (uint32_t w = tid; w < words_per_row; w += blockDim.x) {
        uint32_t packed = sb_base[w];
        uint32_t base_coeff = w * 16u;  // 16 coefficients per uint32

        // Scale index: 0-3, each covering 64 coefficients
        uint32_t si = base_coeff / 64u;
        half s = (&scale.x)[si];

        // Unpack 16 coefficients: 4 bytes × 4 values per byte
        // Each byte holds 4 two-bit values at bit offsets 0,2,4,6.
        #pragma unroll
        for (uint32_t b = 0; b < 4; ++b) {
            uint32_t byte_val = (packed >> (b * 8u)) & 0xFFu;
            uint32_t byte_base = base_coeff + b * 4u;
            #pragma unroll
            for (uint32_t n = 0; n < 4; ++n) {
                uint32_t coeff_idx = byte_base + n;
                if (coeff_idx >= in_dim) break;
                uint32_t v = (byte_val >> (n * 2u)) & 0x3u;
                acc += input_vector[coeff_idx] * s * (half)v;
            }
        }
    }

    // ── Warp-level reduction via shuffle ──────────────────────────────────
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 32);
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 16);
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 8);
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 4);
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 2);
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 1);

    // ── Thread 0 writes final result ──────────────────────────────────────
    if (tid == 0) {
        output_vector[row] = acc;
    }
}
