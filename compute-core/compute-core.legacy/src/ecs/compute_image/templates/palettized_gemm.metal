#include <metal_stdlib>
using namespace metal;

// [[kernel]] palettized_gemm — tiled palettized matrix-matrix multiply
// Handles batched inference (M tokens), speculative decode validation,
// and chunked prefill segments on GPU.
//
// Tiling: BM=16 tokens, BN=16 output channels, BK=32 input features
// TG size: 16 × 16 = 256 threads, one TG per (token × channel) tile.
//
// Shared memory budget:
//   shared_X[16][32]   = 1,024 B  (input activations)
//   shared_W_idx[16][16]= 256 B    (packed weight indices)
//   shared_cb[16][16]  = 512 B     (codebook halves)
//   Total: 1,792 B (well under 32 KB)
kernel void palettized_gemm(
    device const uint8_t* weight_arena    [[buffer(0)]],
    device const half*    input_matrix    [[buffer(1)]],
    device half*          output_matrix   [[buffer(2)]],
    constant uint32_t&    M               [[buffer(3)]],
    constant uint32_t&    in_dim          [[buffer(4)]],
    constant uint32_t&    out_dim         [[buffer(5)]],
    uint2                 tg_pos          [[threadgroup_position_in_grid]],
    uint2                 local           [[thread_position_in_threadgroup]],
    uint                  tid             [[thread_index_in_threadgroup]])
{
    uint32_t row_stride = 32 + (in_dim / 2);
    uint32_t group_start_m = tg_pos.y * 16;
    uint32_t group_start_n = tg_pos.x * 16;

    // Shared memory tiles
    threadgroup half     shared_X[16][32];
    threadgroup uint8_t  shared_W_idx[16][16];
    threadgroup half     shared_cb[16][16];

    // Pre-load: 256 threads load 16 rows × 16 codebook entries = 256 values
    uint32_t cb_load_row = tid / 16;
    uint32_t cb_load_idx = tid % 16;
    uint32_t global_cb_row = group_start_n + cb_load_row;
    if (global_cb_row < out_dim) {
        device const half* row_cb =
            reinterpret_cast<device const half*>(weight_arena + (global_cb_row * row_stride));
        shared_cb[cb_load_row][cb_load_idx] = row_cb[cb_load_idx];
    } else {
        shared_cb[cb_load_row][cb_load_idx] = 0.0h;
    }

    half accumulator = 0.0h;
    uint32_t num_k_tiles = (in_dim + 31) / 32;

    for (uint32_t k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        uint32_t k_offset = k_tile * 32;

        // --- Load input tile: 256 threads load 16×32 half values ---
        // Each thread loads 2 elements (rows 0-7 and 8-15)
        uint32_t x_load_row_1 = tid / 32;
        uint32_t x_load_col   = tid % 32;
        uint32_t global_x_row_1 = group_start_m + x_load_row_1;
        uint32_t global_x_col   = k_offset + x_load_col;

        if (global_x_row_1 < M && global_x_col < in_dim) {
            shared_X[x_load_row_1][x_load_col] =
                input_matrix[global_x_row_1 * in_dim + global_x_col];
        } else {
            shared_X[x_load_row_1][x_load_col] = 0.0h;
        }

        uint32_t x_load_row_2 = x_load_row_1 + 8;
        uint32_t global_x_row_2 = group_start_m + x_load_row_2;
        if (global_x_row_2 < M && global_x_col < in_dim) {
            shared_X[x_load_row_2][x_load_col] =
                input_matrix[global_x_row_2 * in_dim + global_x_col];
        } else {
            shared_X[x_load_row_2][x_load_col] = 0.0h;
        }

        // --- Load weight indices: 256 threads load 16×16 bytes ---
        uint32_t w_load_row = tid / 16;
        uint32_t w_load_col = tid % 16;
        uint32_t global_w_row = group_start_n + w_load_row;
        uint32_t global_w_byte_col = (k_offset / 2) + w_load_col;

        if (global_w_row < out_dim && global_w_byte_col < (in_dim / 2)) {
            shared_W_idx[w_load_row][w_load_col] =
                (weight_arena + (global_w_row * row_stride) + 32)[global_w_byte_col];
        } else {
            shared_W_idx[w_load_row][w_load_col] = 0;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Inner loop: 16 iterations, 2 indices per iteration ---
        uint32_t local_row = local.y;
        uint32_t local_col = local.x;

        for (uint32_t k = 0; k < 16; ++k) {
            uint8_t packed_byte = shared_W_idx[local_col][k];
            uint8_t nibble_0 = packed_byte & 0x0F;
            uint8_t nibble_1 = packed_byte >> 4;

            half w0 = shared_cb[local_col][nibble_0];
            half w1 = shared_cb[local_col][nibble_1];

            accumulator += shared_X[local_row][k * 2]     * w0;
            accumulator += shared_X[local_row][k * 2 + 1] * w1;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // --- Write output ---
    uint32_t global_out_m = group_start_m + local.y;
    uint32_t global_out_n = group_start_n + local.x;
    if (global_out_m < M && global_out_n < out_dim) {
        output_matrix[global_out_m * out_dim + global_out_n] = accumulator;
    }
}
