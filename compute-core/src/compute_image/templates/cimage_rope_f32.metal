// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Rotary Position Embedding (RoPE) for CImage — applies position encoding
// to Q and K in parallel.
//
// Each thread processes one (head_index, dim_pair) where dim_pair selects
// consecutive dimensions (d, d+1) within a head. The standard RoPE rotation
// is applied:
//   rotated_x = x[d] * cos(theta) - x[d+1] * sin(theta)
//   rotated_y = x[d] * sin(theta) + x[d+1] * cos(theta)
//
// Buffer layout:
//   [0] q_input    [num_heads * head_dim] f32
//   [1] k_input    [num_kv_heads * head_dim] f32
//   [2] q_output   [num_heads * head_dim] f32
//   [3] k_output   [num_kv_heads * head_dim] f32
//   [4] position   [1] u32
//   [5] constants  (DecoderConstants)

#include <metal_stdlib>
using namespace metal;

#ifndef DECODER_CONSTANTS_DEFINED
#define DECODER_CONSTANTS_DEFINED
struct DecoderConstants {
    uint32_t hidden_dim;
    uint32_t num_heads;
    uint32_t num_kv_heads;
    uint32_t head_dim;
    uint32_t seq_len;
    uint32_t current_pos;
    float    epsilon;
    uint32_t _pad0;
};
#endif

kernel void cimage_rope_f32(
    device const float*     q_input   [[buffer(0)]],
    device const float*     k_input   [[buffer(1)]],
    device float*           q_output  [[buffer(2)]],
    device float*           k_output  [[buffer(3)]],
    device const uint*      position  [[buffer(4)]],
    constant DecoderConstants&  c     [[buffer(5)]],
    uint gid                         [[thread_position_in_grid]]
) {
    uint num_heads   = c.num_heads;
    uint num_kv_heads = c.num_kv_heads;
    uint head_dim    = c.head_dim;
    uint pos         = position[0];

    // Map thread to (head_index, dim_pair) — one thread per dim_pair
    uint pairs_per_head = head_dim / 2;
    uint head_index    = gid / pairs_per_head;
    uint dim_pair      = gid % pairs_per_head;

    // Guard: only process valid pairs (must hold for even head_dim)
    if (dim_pair >= pairs_per_head) return;

    uint d = dim_pair * 2;

    // Compute RoPE coefficients
    float exponent = 2.0f * (float)dim_pair / (float)head_dim;
    float theta    = (float)pos / pow(10000.0f, exponent);
    float cos_t    = cos(theta);
    float sin_t    = sin(theta);

    // Apply RoPE to Q
    if (head_index < num_heads) {
        uint q_base = head_index * head_dim;
        float x0 = q_input[q_base + d];
        float x1 = q_input[q_base + d + 1];
        q_output[q_base + d]     = x0 * cos_t - x1 * sin_t;
        q_output[q_base + d + 1] = x0 * sin_t + x1 * cos_t;
    }

    // Apply RoPE to K (shared kv_head mapping — each kv head maps 1:1)
    if (head_index < num_kv_heads) {
        uint k_base = head_index * head_dim;
        float x0 = k_input[k_base + d];
        float x1 = k_input[k_base + d + 1];
        k_output[k_base + d]     = x0 * cos_t - x1 * sin_t;
        k_output[k_base + d + 1] = x0 * sin_t + x1 * cos_t;
    }
}
