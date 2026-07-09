// SPDX-License-Identifier: MIT OR Apache-2.0
//
// KV Cache Append for CImage — writes new K and V tokens into the cache buffers
// at the current position.
//
// Each thread handles one (kv_head, d) element.
//
// Buffer layout:
//   [0] k_new     [num_kv_heads * head_dim] f32                     (new token)
//   [1] v_new     [num_kv_heads * head_dim] f32                     (new token)
//   [2] k_cache   [num_kv_heads * head_dim * max_seq_len] f32       (in/out)
//   [3] v_cache   [num_kv_heads * head_dim * max_seq_len] f32       (in/out)
//   [4] constants (DecoderConstants)

#include <metal_stdlib>
using namespace metal;

#ifndef DECODER_CONSTANTS_DEFINED
#define DECODER_CONSTANTS_DEFINED
struct DecoderConstants {
    uint32_t hidden_dim;
    uint32_t num_heads;
    uint32_t num_kv_heads;
    uint32_t head_dim;
    uint32_t seq_len;       // max_seq_len (cache capacity)
    uint32_t current_pos;   // position to write at
    float    epsilon;
    uint32_t _pad0;
};
#endif

kernel void cimage_kv_append_f32(
    device const float*     k_new   [[buffer(0)]],
    device const float*     v_new   [[buffer(1)]],
    device float*           k_cache [[buffer(2)]],
    device float*           v_cache [[buffer(3)]],
    constant DecoderConstants&  c   [[buffer(4)]],
    uint gid                        [[thread_position_in_grid]]
) {
    uint num_kv_heads = c.num_kv_heads;
    uint head_dim     = c.head_dim;
    uint max_seq_len  = c.seq_len;
    uint cur_pos      = c.current_pos;

    // Each thread handles one (kv_head, d) element
    uint kv_head = gid / head_dim;
    uint d       = gid % head_dim;

    if (kv_head >= num_kv_heads) return;
    if (d >= head_dim) return;

    // Write K and V into cache at current position
    // Layout: [kv_head * head_dim * max_seq_len + cur_pos * head_dim + d]
    uint cache_slot  = kv_head * head_dim * max_seq_len + cur_pos * head_dim + d;
    uint input_base  = kv_head * head_dim + d;

    k_cache[cache_slot] = k_new[input_base];
    v_cache[cache_slot] = v_new[input_base];
}
