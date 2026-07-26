// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Attention Score Computation for CImage — computes S[q_head] = Q @ K^T / sqrt(head_dim)
// for one full query head per thread.
//
// Supports Grouped-Query Attention (GQA):
//   kv_head = head / (num_heads / num_kv_heads)
//
// Buffer layout:
//   [0] q_rope    [num_heads * head_dim] f32             (rotated Q)
//   [1] k_rope    [num_kv_heads * head_dim * seq_len] f32 (all cached K positions)
//   [2] scores    [num_heads * seq_len] f32              (output)
//   [3] constants (DecoderConstants)

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

kernel void cimage_attention_scores_f32(
    device const float*     q_rope  [[buffer(0)]],
    device const float*     k_rope  [[buffer(1)]],
    device float*           scores  [[buffer(2)]],
    constant DecoderConstants&  c   [[buffer(3)]],
    uint gid                        [[thread_position_in_grid]]
) {
    uint head        = gid;
    uint num_heads   = c.num_heads;
    uint num_kv_heads = c.num_kv_heads;
    uint head_dim    = c.head_dim;
    uint seq_len     = c.seq_len;

    if (head >= num_heads) return;

    // Grouped-Query Attention: map query head to kv head
    uint kv_head = head / (num_heads / num_kv_heads);

    // Pointer into Q for this head
    device const float* q_ptr = q_rope + head * head_dim;

    // Store scores for this head across all cached positions
    float rsqrt = 1.0f / sqrt((float)head_dim);

    for (uint pos = 0; pos < seq_len; pos++) {
        // K pointer: [kv_head * head_dim * seq_len + pos * head_dim]
        device const float* k_ptr = k_rope + kv_head * head_dim * seq_len + pos * head_dim;

        // Dot product
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            dot += q_ptr[d] * k_ptr[d];
        }

        scores[head * seq_len + pos] = dot * rsqrt;
    }
}
