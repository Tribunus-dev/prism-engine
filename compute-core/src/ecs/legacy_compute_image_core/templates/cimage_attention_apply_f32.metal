// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Apply attended scores to V for CImage — weighted sum over cached V positions.
//
// Each thread computes one output dimension for one head:
//   output[head * head_dim + d] = sum over pos of
//     scores[head * seq_len + pos] * v[kv_head * head_dim * seq_len + pos * head_dim + d]
//
// Supports Grouped-Query Attention (GQA):
//   kv_head = head / (num_heads / num_kv_heads)
//
// Buffer layout:
//   [0] scores    [num_heads * seq_len] f32                (softmaxed scores)
//   [1] v         [num_kv_heads * head_dim * seq_len] f32  (all cached V positions)
//   [2] output    [num_heads * head_dim] f32               (output)
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

kernel void cimage_attention_apply_f32(
    device const float*     scores  [[buffer(0)]],
    device const float*     v       [[buffer(1)]],
    device float*           output  [[buffer(2)]],
    constant DecoderConstants&  c   [[buffer(3)]],
    uint gid                        [[thread_position_in_grid]]
) {
    uint num_heads   = c.num_heads;
    uint num_kv_heads = c.num_kv_heads;
    uint head_dim    = c.head_dim;
    uint seq_len     = c.seq_len;

    // Each thread computes one output dimension for one head
    uint head = gid / head_dim;
    uint d    = gid % head_dim;

    if (head >= num_heads) return;
    if (d >= head_dim) return;

    // Grouped-Query Attention: map query head to kv head
    uint kv_head = head / (num_heads / num_kv_heads);

    // Accumulate weighted sum over cached positions
    float acc = 0.0f;
    device const float* head_scores = scores + head * seq_len;
    device const float* v_base = v + kv_head * head_dim * seq_len;

    for (uint pos = 0; pos < seq_len; pos++) {
        float score = head_scores[pos];
        float v_val = v_base[pos * head_dim + d];
        acc += score * v_val;
    }

    output[head * head_dim + d] = acc;
}
