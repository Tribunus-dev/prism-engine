// IQ2_XXS GEMV: importance-weighted 2-bit quant with codebook dequantization.
// 256-element super-blocks, each with a single f16 scale and 64 bytes of
// 2-bit codebook indices (4 indices per byte). The 2-bit index selects one
// of four distinct codebook values per group; group boundaries are
// aligned with the codebook structure.
//
// Template placeholders (compile-time):
//   TILE_WIDTH  — columns cooperatively processed
//   GROUP_SIZE  — rows per threadgroup (256 = one super-block)
//   THREADGROUP_SIZE — threads per group

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

// ── IQ2_XXS super-block structure ────────────────────────────────────────
//   Bytes    Contents
//   0-1      1 x f16 super-block scale
//   2-65     64 bytes packed 2-bit codebook indices (4 per byte)
//   =66      total (targeting 2.0625 bits/element)

#define IQ2XXS_BLOCK_SIZE    256
#define IQ2XXS_SCALE_BYTES   sizeof(half)     // 2
#define IQ2XXS_QS_BYTES      64
#define IQ2XXS_SUPER_BLOCK_SIZE (IQ2XXS_SCALE_BYTES + IQ2XXS_QS_BYTES)  // 66

// ── Codebook: 16-entry f16 lookup table ──────────────────────────────────
//    This is a SIMPLIFIED placeholder codebook for the template structure.
//    The full IQ2_XXS format uses a 256-entry uint64_t grid with a 128-entry
//    sign table (see ds4/ds4_iq2_tables_cuda.inc). The calibration pipeline
//    replaces this table with the actual trained codebook weights.
//
//    For this template: 16-entry centroid table indexed by 2-bit indices
//    (4 indices packed per byte). The compiler inlines these as constants
//    in the kernel's constant/texture cache.

__device__ __constant__ half iq2xxs_codebook[16] = {
    -1.000f, -0.750f, -0.500f, -0.250f,
     0.000f,  0.125f,  0.250f,  0.375f,
     0.500f,  0.625f,  0.750f,  0.875f,
     1.000f,  1.250f,  1.500f,  2.000f
};

extern "C" __global__ void iq2_xxs_gemv(
    const half* __restrict__ input_vector,   // [in_dim]
    const uint8_t* __restrict__ weights,     // [out_dim * IQ2XXS_SUPER_BLOCK_SIZE / 66]
    const half* __restrict__ scales,         // [out_dim]  — one f16 per super-block row
    half* __restrict__ output_vector,         // [out_dim]
    uint32_t in_dim,
    uint32_t out_dim)
{
    uint32_t row = blockIdx.x;
    uint32_t tid = threadIdx.x;

    if (row >= out_dim) return;

    // ── Load super-block scale ────────────────────────────────────────────
    half s = scales[row];

    // ── Locate this row's packed index data ───────────────────────────────
    //    IQ2_XXS uses a careful indexing scheme aligned to 256-element
    //    super-blocks.  For this kernel we store 64 bytes of packed 2-bit
    //    indices followed by the scale (handled separately above).
    //
    //    Each byte holds 4 indices (bits [1:0], [3:2], [5:4], [7:6]).
    //    64 bytes × 4 indices = 256 indices.
    const uint32_t* idx_ptr = reinterpret_cast<const uint32_t*>(
        weights + row * IQ2XXS_QS_BYTES);

    uint32_t num_dwords = IQ2XXS_QS_BYTES / sizeof(uint32_t);  // 16

    half acc = (half)0.0f;

    // ── Cooperative index processing ──────────────────────────────────────
    //    Each uint32 holds 16 indices (4 bytes × 4 indices/byte).
    //    Each index is 2 bits → 16 indices per uint32.

    for (uint32_t w = tid; w < num_dwords; w += blockDim.x) {
        uint32_t packed = idx_ptr[w];
        uint32_t base_idx = w * 16u;

        // Unpack 16 indices from this dword
        #pragma unroll
        for (uint32_t i = 0; i < 16; ++i) {
            uint32_t coeff_idx = base_idx + i;
            if (coeff_idx >= in_dim) break;

            // Each byte holds 4 indices; figure out which byte and which nibble
            uint32_t byte_offset = i / 4u;
            uint32_t nibble_within_byte = (i % 4u) * 2u;

            uint32_t idx = (packed >> (byte_offset * 8u + nibble_within_byte)) & 0x3u;

            // Lookup in codebook, scale, and accumulate
            half cb_val = iq2xxs_codebook[idx];
            acc += input_vector[coeff_idx] * s * cb_val;
        }
    }

    // ── Warp-level reduction ──────────────────────────────────────────────
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 32);
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 16);
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 8);
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 4);
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 2);
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 1);

    if (tid == 0) {
        output_vector[row] = acc;
    }
}
