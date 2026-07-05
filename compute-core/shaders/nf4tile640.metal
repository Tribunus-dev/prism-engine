#include <metal_stdlib>
using namespace metal;

// Symmetric [-1,1] NF4 codebook — MUST match nf4tile640.rs::NF4_CODEBOOK.
constant float nf4_table_fp32[16] = {
    -1.0f, -0.6961928f, -0.5250731f, -0.3949175f,
    -0.2844414f, -0.1847734f, -0.09105f, 0.0f,
    0.0795803f, 0.1609302f, 0.2461123f, 0.3379152f,
    0.4407099f, 0.562617f, 0.7229568f, 1.0f
};

// Fused dequantize + matrix multiply for nf4tile640 packed weights.
//
// Each thread computes one element of output[row, col] = Σ_k input[row,k] * W_deq[k,col].
// Weights are packed in nf4tile640 format: cache-line-aligned 512-byte tiles,
// each tile holding 640 values (10 groups × 64 elements, 10 FP16 scales).
//
// Buffer layout:
//   [0] packed_weights — nf4tile640-packed bytes, shape [K, ceil(N/640), 512]
//   [1] input          — activation matrix, row-major f32, shape [M, K]
//   [2] output         — result matrix, row-major f32, shape [M, N]
//   [3] M              — number of activation rows (constant uint)
//   [4] K              — inner dimension (constant uint)
//   [5] N              — output columns (constant uint)
kernel void dequant_mul_nf4tile640(
    device const uchar*  packed_weights [[buffer(0)]],
    device const float*  input          [[buffer(1)]],
    device float*        output         [[buffer(2)]],
    constant uint&       M              [[buffer(3)]],
    constant uint&       K              [[buffer(4)]],
    constant uint&       N              [[buffer(5)]],
    uint2                pos            [[thread_position_in_grid]]
) {
    uint m_idx = pos.y;
    uint n_idx = pos.x;
    if (m_idx >= M || n_idx >= N) { return; }

    constexpr uint TILE            = 640;
    constexpr uint GROUP           = 64;
    constexpr uint GROUPS_PER_TILE = 10;
    constexpr uint TILE_RECORD     = 512;
    constexpr uint SCALES_BYTES    = 20;  // 10 × 2-byte FP16 scales
    constexpr uint CODES_BYTES     = 320; // 640 values × 4 bits ÷ 8

    uint tile_in_row   = n_idx / TILE;
    uint tiles_per_row = (N + TILE - 1) / TILE;
    uint elem_in_tile  = n_idx % TILE;
    uint group         = elem_in_tile / GROUP;
    uint elem_in_group = elem_in_tile % GROUP;
    uint byte_in_group = elem_in_group / 2;
    uint nibble_shift  = (elem_in_group & 1) << 2; // 0 for low, 4 for high

    float accum = 0.0f;

    for (uint kr = 0; kr < K; ++kr) {
        uint tile_idx  = kr * tiles_per_row + tile_in_row;
        uint tile_base = tile_idx * TILE_RECORD;

        // Read FP16 scale for this group (embedded in tile).
        device const half* scales = (device const half*)(packed_weights + tile_base);
        float scale = float(scales[group]);

        // Extract the 4-bit NF4 code index.
        uint code_byte_offset = SCALES_BYTES + group * (GROUP / 2) + byte_in_group;
        uchar code_byte = packed_weights[tile_base + code_byte_offset];
        uint code = (code_byte >> nibble_shift) & 0xFu;

        float weight = scale * nf4_table_fp32[code];
        accum += weight * input[m_idx * K + kr];
    }

    output[m_idx * N + n_idx] = accum;
}
