#include <metal_stdlib>
using namespace metal;

// Default fallback NF4 codebook (canonical NF4 quantiles of N(0,1)).
// The runtime may override this via buffer[9] (ProfileDescriptor).
constant float fallback_nf4_table_fp32[16] = {
    -1.0f, -0.6961928f, -0.5250731f, -0.3949175f,
    -0.2844414f, -0.1847734f, -0.09105f, 0.0f,
    0.0795803f, 0.1609302f, 0.2461123f, 0.3379152f,
    0.4407099f, 0.562617f, 0.7229568f, 1.0f
};

// Profile descriptor for adaptive codebook.
// When profile_id != 0, codebook values override the fallback table.
struct ProfileDescriptor {
    uint  profile_id;         // 0 = use fallback (compat)
    uint  abi_version;        // must match PROFILE_ABI_VERSION
    uint  group_size;         // must be 128
    uint  tile_elements;      // must be 640
    float codebook[16];       // 16 reconstruction centroids
    uint  clipping_policy;    // 0=none, 1=percentile, 2=mse_optimal
    uint  bias_policy;        // 0=none, 1=affine
    uint  sidecar_policy;     // 0=none, 1=sparse_fp16, 2=protected
    uint  pad;                // align to 16 bytes
};

// Fused dequantize + matrix multiply for nf4tile640 packed weights.
//
// Each thread computes one element of output[row, col] = Σ_k input[row,k] * W_deq[k,col].
// Weights are packed in the compiler nf4tile640 format:
//   - group_size = 128 elements per quantization group
//   - groups_per_tile = 5 (640/128)
//   - Codes: 8×4-bit NF4 indices packed per u32 word, stored as LE bytes
//   - Scales and biases stored in separate buffers (one f32 per group)
//
// Buffer layout:
//   [0] packed_codes — nf4tile640-packed code bytes, shape [K, ceil(N/640), 320]
//   [1] scale_buffer — f32 scales, shape [K, ceil(N/640), 5]
//   [2] bias_buffer  — f32 biases, shape [K, ceil(N/640), 5]
//   [3] input        — activation matrix, row-major f32, shape [M, K]
//   [4] output       — result matrix, row-major f32, shape [M, N]
//   [5] M            — number of activation rows (constant uint)
//   [6] K_dim        — inner dimension (constant uint)
//   [7] N            — output columns (constant uint)
//   [8] group_size   — quantization group size (constant ushort, always 128)
kernel void dequant_mul_nf4tile640(
    device const uchar*  packed_codes   [[buffer(0)]],
    device const float*  scale_buffer   [[buffer(1)]],
    device const float*  bias_buffer    [[buffer(2)]],
    device const float*  input          [[buffer(3)]],
    device float*        output         [[buffer(4)]],
    constant uint&       M              [[buffer(5)]],
    constant uint&       K_dim          [[buffer(6)]],
    constant uint&       N              [[buffer(7)]],
    constant ushort&     group_size     [[buffer(8)]],
    constant const void*   profile_buffer   [[buffer(9)]],
    uint2                pos            [[thread_position_in_grid]]
) {
    uint m_idx = pos.y;
    uint n_idx = pos.x;
    if (m_idx >= M || n_idx >= N) { return; }

    // Select codebook: profile descriptor or fallback
    constant float* codebook = fallback_nf4_table_fp32;
    if (profile_buffer) {
        constant const ProfileDescriptor* desc = (constant const ProfileDescriptor*)profile_buffer;
        if (desc->profile_id > 0 && desc->abi_version == 1) {
            codebook = desc->codebook;
        }
    }

    constexpr uint TILE            = 640;
    constexpr uint GROUPS_PER_TILE = 5;      // 640 / 128
    constexpr uint CODES_PER_U32   = 8;
    constexpr uint U32S_PER_GROUP  = 16;     // 128 / 8
    constexpr uint CODES_BYTES     = 320;    // 640 / 2

    uint tile_in_col   = n_idx / TILE;
    uint tiles_per_col = (N + TILE - 1) / TILE;
    uint elem_in_tile  = n_idx % TILE;
    uint group         = elem_in_tile / group_size;
    uint elem_in_group = elem_in_tile % group_size;
    uint u32_index     = elem_in_group / CODES_PER_U32;
    uint nibble_shift  = (elem_in_group % CODES_PER_U32) << 2;  // * 4

    float accum = 0.0f;

    for (uint kr = 0; kr < K_dim; ++kr) {
        uint tile_idx        = kr * tiles_per_col + tile_in_col;
        uint tile_codes_base = tile_idx * CODES_BYTES;
        uint tile_meta_base  = tile_idx * GROUPS_PER_TILE;
        uint group_codes_ofs = group * U32S_PER_GROUP * 4;  // 16 u32s × 4 bytes per group

        // Read the u32 word containing the NF4 code from the packed codes buffer.
        device const uint* word_ptr = (device const uint*)(packed_codes + tile_codes_base + group_codes_ofs);
        uint word = word_ptr[u32_index];
        uint code = (word >> nibble_shift) & 0xFu;

        float scale = scale_buffer[tile_meta_base + group];
        float bias  = bias_buffer[tile_meta_base + group];

        float weight = codebook[code] * scale + bias;
        accum += weight * input[m_idx * K_dim + kr];
    }

    output[m_idx * N + n_idx] = accum;
}
