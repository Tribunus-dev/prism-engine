// [[kernel]] ternary_gemm — tiled branchless addition-only ternary GEMM
// Tiling: BM=16 × BN=16 × BK=32, 256 threads per tile (8 warps × 32 lanes)
//
// Ternary packing: 16 weights per uint32_t, 2 bits each
// Encoding: 00=0, 01=+1, 10=-1
//
// Same branchless addition-only approach as ternary_gemv, but tiled
// for batched matrix-matrix multiply (prefill/encode path).
//
// buffer(0): activation_matrix [M * K] half (row-major)
// buffer(1): weight_matrix    [N * packed_k] uint
// buffer(2): scale_matrix     [N * group_count] half
// buffer(3): output_matrix    [M * N] half
// buffer(4): M, K, N, group_size uints

#include <metal_stdlib>
using namespace metal;

constant uint BM = 16; // tile rows in activation
constant uint BN = 16; // tile cols in weight
constant uint BK = 32; // tile K-dim (256 ternary weights packed into 16 uint32s)

kernel void ternary_gemm(
    device const half*    act_matrix      [[buffer(0)]],
    device const uint*    weight_matrix   [[buffer(1)]],
    device const half*    scale_matrix    [[buffer(2)]],
    device half*          out_matrix      [[buffer(3)]],
    constant uint32_t&    M               [[buffer(4)]],
    constant uint32_t&    K               [[buffer(5)]],
    constant uint32_t&    N               [[buffer(6)]],
    constant uint32_t&    group_size      [[buffer(7)]],
    uint2 gid                             [[threadgroup_position_in_grid]],
    uint  tid                             [[thread_position_in_threadgroup]])
{
    uint packed_k = (K + 15) / 16;
    uint groups_per_col = (K + group_size - 1) / group_size;

    // Tile start coordinates
    uint tile_m = gid.x * BM;
    uint tile_n = gid.y * BN;

    // Thread position within tile
    uint tm = tid / BN;  // activation tile row (0..BM-1 mapped to 0..15)
    uint tn = tid % BN;  // weight tile col (0..BN-1)

    // Accumulator register
    half acc[16]; // BM=16
    for (uint i = 0; i < BM; i++) acc[i] = 0.0h;

    // Shared memory for activation tile (BM × BK)
    threadgroup half act_tile[BM * BK];

    // Iterate over K in BK-sized chunks
    for (uint k_start = 0; k_start < packed_k; k_start += BK / 16) {
        // ── Load activation tile into shared memory ──
        // Each thread loads BM/BK elements in a coalesced pattern
        for (uint i = 0; i < (BM * (BK / 16) + 255) / 256; i++) {
            uint flat_idx = tid + i * 256;
            uint act_row = flat_idx / (BK / 16);
            uint act_col = flat_idx % (BK / 16);
            uint global_k = (k_start + act_col) * 16;
            if (act_row < BM && global_k < K) {
                uint m_idx = min(tile_m + act_row, M - 1);
                act_tile[act_row * BK + act_col * 16] = act_matrix[m_idx * K + global_k + 0];
                // Load full uint32_t worth: 16 activations
                if (act_col * 16 + 15 < BK) {
                    // Contiguous 16 halfs from the activation row
                    for (uint j = 1; j < 16 && global_k + j < K; j++) {
                        act_tile[act_row * BK + act_col * 16 + j] = act_matrix[m_idx * K + global_k + j];
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Compute tile matmul ──
        for (uint kk = 0; kk < BK / 16; kk++) {
            uint global_k_pos = (k_start + kk) * 16;

            // Each thread processes one column of the weight tile
            uint weight_idx = (tile_n + tn) * packed_k + (k_start + kk);

            // Clamp to valid bounds
            if (tile_n + tn >= N || global_k_pos >= K) continue;

            uint packed = weight_matrix[weight_idx];

            // Extract 16 ternary weights
            uint4 w0 = uint4(
                (packed >> 0)  & 0x3, (packed >> 2)  & 0x3,
                (packed >> 4)  & 0x3, (packed >> 6)  & 0x3
            );
            uint4 w1 = uint4(
                (packed >> 8)  & 0x3, (packed >> 10) & 0x3,
                (packed >> 12) & 0x3, (packed >> 14) & 0x3
            );
            uint4 w2 = uint4(
                (packed >> 16) & 0x3, (packed >> 18) & 0x3,
                (packed >> 20) & 0x3, (packed >> 22) & 0x3
            );
            uint4 w3 = uint4(
                (packed >> 24) & 0x3, (packed >> 26) & 0x3,
                (packed >> 28) & 0x3, (packed >> 30) & 0x3
            );

            // Accumulate for each row in BM — branchless select
            for (uint ri = 0; ri < BM; ri++) {
                half row_act[16];
                // Gather 16 activations from shared memory
                uint act_base = (k_start + kk) * 16;
                for (uint j = 0; j < 16; j++) {
                    row_act[j] = act_tile[ri * BK + kk * 16 + j];
                }

                // ── Branchless select accumulation ──
                // select(0, +act, w==1) + select(0, -act, w==2)
                half4 a0 = half4(row_act[0], row_act[1], row_act[2], row_act[3]);
                half4 a1 = half4(row_act[4], row_act[5], row_act[6], row_act[7]);
                half4 a2 = half4(row_act[8], row_act[9], row_act[10], row_act[11]);
                half4 a3 = half4(row_act[12], row_act[13], row_act[14], row_act[15]);

                half4 v0 = select(0.0h, a0, w0 == 1) + select(0.0h, -a0, w0 == 2);
                half4 v1 = select(0.0h, a1, w1 == 1) + select(0.0h, -a1, w1 == 2);
                half4 v2 = select(0.0h, a2, w2 == 1) + select(0.0h, -a2, w2 == 2);
                half4 v3 = select(0.0h, a3, w3 == 1) + select(0.0h, -a3, w3 == 2);

                half block_sum = v0.x + v0.y + v0.z + v0.w
                               + v1.x + v1.y + v1.z + v1.w
                               + v2.x + v2.y + v2.z + v2.w
                               + v3.x + v3.y + v3.z + v3.w;

                acc[ri] += block_sum;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Apply scales and write output ──
    // Each thread writes one column tile_n+tn for all BM rows
    if (tile_n + tn < N) {
        uint group_id = 0;  // simplified: per-tensor scale
        half scale = scale_matrix[(tile_n + tn) * groups_per_col + group_id];
        for (uint ri = 0; ri < BM && tile_m + ri < M; ri++) {
            out_matrix[(tile_m + ri) * N + (tile_n + tn)] = acc[ri] * scale;
        }
    }
}
