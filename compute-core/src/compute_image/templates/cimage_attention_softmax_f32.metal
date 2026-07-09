// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Softmax over attention scores for CImage — one thread per head.
//
// Each thread applies numerically stable softmax across the seq_len dimension:
//   1. Find max over seq_len
//   2. Subtract max, exponentiate
//   3. Sum exp values
//   4. Divide each by sum
//
// Buffer layout:
//   [0] scores    [num_heads * seq_len] f32  (in/out)
//   [1] constants (DecoderConstants)

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

kernel void cimage_attention_softmax_f32(
    device float*           scores [[buffer(0)]],
    constant DecoderConstants&  c  [[buffer(1)]],
    uint gid                       [[thread_position_in_grid]]
) {
    uint head      = gid;
    uint num_heads = c.num_heads;
    uint seq_len   = c.seq_len;

    if (head >= num_heads) return;

    device float* head_scores = scores + head * seq_len;

    // Phase 1: find max
    float max_val = -INFINITY;
    for (uint pos = 0; pos < seq_len; pos++) {
        max_val = fmax(max_val, head_scores[pos]);
    }

    // Phase 2: subtract max and exponentiate, accumulating sum
    float sum = 0.0f;
    for (uint pos = 0; pos < seq_len; pos++) {
        float v = exp(head_scores[pos] - max_val);
        head_scores[pos] = v;
        sum += v;
    }

    // Phase 3: normalize
    float inv_sum = 1.0f / sum;
    for (uint pos = 0; pos < seq_len; pos++) {
        head_scores[pos] *= inv_sum;
    }
}
