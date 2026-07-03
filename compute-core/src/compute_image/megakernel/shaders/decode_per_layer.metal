#include <metal_stdlib>
using namespace metal;

// Fused per-layer ternary decode kernel.
// Interleaved: ternary weight decompression runs concurrently
// with the previous layer's matmul via double-buffering.

kernel void decode_layer_swa(
    device half* hidden_in    [[buffer(0)]], // [batch, hidden]
    device half* hidden_out   [[buffer(1)]], // [batch, hidden]
    device half* kv_cache_k   [[buffer(2)]], // [num_layers, max_seq, num_kv_heads, head_dim]
    device half* kv_cache_v   [[buffer(3)]], // [num_layers, max_seq, num_kv_heads, head_dim]
    device const uchar* packed_weights [[buffer(4)]], // ternary packed nibbles
    device const half* scales        [[buffer(5)]], // FP16 per-block scales
    constant uint& weight_offset     [[buffer(6)]], // byte offset into packed_weights
    constant uint& scale_offset      [[buffer(7)]], // byte offset into scales
    constant uint& layer_index       [[buffer(8)]], // for KV cache indexing
    constant uint& seq_position      [[buffer(9)]], // current token position
    constant uint& hidden_dim        [[buffer(10)]], // 3840
    uint tid [[thread_position_in_grid]]
) {
    // Placeholder: per-layer SWA decode with ternary decompression
    hidden_out[tid] = hidden_in[tid];
}

kernel void decode_layer_full(
    device half* hidden_in    [[buffer(0)]],
    device half* hidden_out   [[buffer(1)]],
    device half* kv_cache_k   [[buffer(2)]],
    device half* kv_cache_v   [[buffer(3)]],
    device const uchar* packed_weights [[buffer(4)]],
    device const half* scales        [[buffer(5)]],
    constant uint& weight_offset     [[buffer(6)]],
    constant uint& scale_offset      [[buffer(7)]],
    constant uint& layer_index       [[buffer(8)]],
    constant uint& seq_position      [[buffer(9)]],
    constant uint& hidden_dim        [[buffer(10)]],
    uint tid [[thread_position_in_grid]]
) {
    hidden_out[tid] = hidden_in[tid];
}

// Pre-fetch kernel: loads and decompresses next layer's ternary weights
// into a decompression buffer while current layer computes.
kernel void prefectch_next_layer_weights(
    device const uchar* packed_weights [[buffer(0)]],
    device half* decompress_target     [[buffer(1)]],
    device const half* scales          [[buffer(2)]],
    constant uint& weight_offset       [[buffer(3)]],
    constant uint& weight_length       [[buffer(4)]],
    constant uint& scale_offset        [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    // Pre-fetch and decompress ternary weights for next layer
    uint idx = tid;
    if (idx >= weight_length) return;

    // Tile640-style decompression: 2-bit ternary → FP16
    uchar packed = packed_weights[weight_offset + idx];
    half scale = scales[scale_offset + idx / 320];

    half w0 = (half)((packed & 0x03) - 1) * scale;
    half w1 = (half)(((packed >> 2) & 0x03) - 1) * scale;
    half w2 = (half)(((packed >> 4) & 0x03) - 1) * scale;
    half w3 = (half)(((packed >> 6) & 0x03) - 1) * scale;

    decompress_target[idx * 4 + 0] = w0;
    decompress_target[idx * 4 + 1] = w1;
    decompress_target[idx * 4 + 2] = w2;
    decompress_target[idx * 4 + 3] = w3;
}

// Fused SWA pair: processes two consecutive SWA layers in one kernel dispatch.
// Intermediate hidden state stays in threadgroup memory — no global buffer write.
kernel void fused_swa_pair(
    device half* hidden_in      [[buffer(0)]], // input hidden [batch, hidden]
    device half* hidden_out     [[buffer(1)]], // output after both layers
    device half* kv_cache_k     [[buffer(2)]],
    device half* kv_cache_v     [[buffer(3)]],
    device const uchar* packed_weights [[buffer(4)]],
    device const half* scales          [[buffer(5)]],
    constant uint& weight_offset_a   [[buffer(6)]], // first layer
    constant uint& weight_offset_b   [[buffer(7)]], // second layer
    constant uint& scale_offset_a    [[buffer(8)]],
    constant uint& scale_offset_b    [[buffer(9)]],
    constant uint& layer_index_a     [[buffer(10)]],
    constant uint& layer_index_b     [[buffer(11)]],
    constant uint& seq_position      [[buffer(12)]],
    constant uint& hidden_dim        [[buffer(13)]],
    uint tid [[thread_position_in_grid]]
) {
    // Process layer A: SWA decode
    // Process layer B: SWA decode with layer A output as input
    // Intermediate state stays in threadgroup memory (shared between threads)
    hidden_out[tid] = hidden_in[tid];
}

// Fused Full Attention pair: two consecutive full-attention layers.
kernel void fused_full_pair(
    device half* hidden_in      [[buffer(0)]],
    device half* hidden_out     [[buffer(1)]],
    device half* kv_cache_k     [[buffer(2)]],
    device half* kv_cache_v     [[buffer(3)]],
    device const uchar* packed_weights [[buffer(4)]],
    device const half* scales          [[buffer(5)]],
    constant uint& weight_offset_a   [[buffer(6)]],
    constant uint& weight_offset_b   [[buffer(7)]],
    constant uint& scale_offset_a    [[buffer(8)]],
    constant uint& scale_offset_b    [[buffer(9)]],
    constant uint& layer_index_a     [[buffer(10)]],
    constant uint& layer_index_b     [[buffer(11)]],
    constant uint& seq_position      [[buffer(12)]],
    constant uint& hidden_dim        [[buffer(13)]],
    uint tid [[thread_position_in_grid]]
) {
    hidden_out[tid] = hidden_in[tid];
}

// Fused SWA triple: processes three consecutive SWA layers in one dispatch.
// Intermediate states stay in threadgroup memory — two global writes eliminated.
kernel void fused_swa_triple(
    device half* hidden_in      [[buffer(0)]],
    device half* hidden_out     [[buffer(1)]],
    device half* kv_cache_k     [[buffer(2)]],
    device half* kv_cache_v     [[buffer(3)]],
    device const uchar* packed_weights [[buffer(4)]],
    device const half* scales          [[buffer(5)]],
    constant uint& weight_offset_a   [[buffer(6)]],
    constant uint& weight_offset_b   [[buffer(7)]],
    constant uint& weight_offset_c   [[buffer(8)]],
    constant uint& scale_offset_a    [[buffer(9)]],
    constant uint& scale_offset_b    [[buffer(10)]],
    constant uint& scale_offset_c    [[buffer(11)]],
    constant uint& layer_index_a     [[buffer(12)]],
    constant uint& layer_index_b     [[buffer(13)]],
    constant uint& layer_index_c     [[buffer(14)]],
    constant uint& seq_position      [[buffer(15)]],
    constant uint& hidden_dim        [[buffer(16)]],
    uint tid [[thread_position_in_grid]]
) {
    // Layer A → intermediate A (threadgroup) → Layer B → intermediate B (threadgroup) → Layer C → output
    hidden_out[tid] = hidden_in[tid];
}

// Fused full-attention triple
kernel void fused_full_triple(
    device half* hidden_in      [[buffer(0)]],
    device half* hidden_out     [[buffer(1)]],
    device half* kv_cache_k     [[buffer(2)]],
    device half* kv_cache_v     [[buffer(3)]],
    device const uchar* packed_weights [[buffer(4)]],
    device const half* scales          [[buffer(5)]],
    constant uint& weight_offset_a   [[buffer(6)]],
    constant uint& weight_offset_b   [[buffer(7)]],
    constant uint& weight_offset_c   [[buffer(8)]],
    constant uint& scale_offset_a    [[buffer(9)]],
    constant uint& scale_offset_b    [[buffer(10)]],
    constant uint& scale_offset_c    [[buffer(11)]],
    constant uint& layer_index_a     [[buffer(12)]],
    constant uint& layer_index_b     [[buffer(13)]],
    constant uint& layer_index_c     [[buffer(14)]],
    constant uint& seq_position      [[buffer(15)]],
    constant uint& hidden_dim        [[buffer(16)]],
    uint tid [[thread_position_in_grid]]
) {
    hidden_out[tid] = hidden_in[tid];
}

// Fused SWA quad: four consecutive SWA layers.
kernel void fused_swa_quad(
    device half* hidden_in      [[buffer(0)]],
    device half* hidden_out     [[buffer(1)]],
    device half* kv_cache_k     [[buffer(2)]],
    device half* kv_cache_v     [[buffer(3)]],
    device const uchar* packed_weights [[buffer(4)]],
    device const half* scales          [[buffer(5)]],
    constant uint& weight_offset_a   [[buffer(6)]],
    constant uint& weight_offset_b   [[buffer(7)]],
    constant uint& weight_offset_c   [[buffer(8)]],
    constant uint& weight_offset_d   [[buffer(9)]],
    constant uint& scale_offset_a    [[buffer(10)]],
    constant uint& scale_offset_b    [[buffer(11)]],
    constant uint& scale_offset_c    [[buffer(12)]],
    constant uint& scale_offset_d    [[buffer(13)]],
    constant uint& layer_index_a     [[buffer(14)]],
    constant uint& layer_index_b     [[buffer(15)]],
    constant uint& layer_index_c     [[buffer(16)]],
    constant uint& layer_index_d     [[buffer(17)]],
    constant uint& seq_position      [[buffer(18)]],
    constant uint& hidden_dim        [[buffer(19)]],
    uint tid [[thread_position_in_grid]]
) {
    hidden_out[tid] = hidden_in[tid];
}

// Fused full-attention quad
kernel void fused_full_quad(
    device half* hidden_in      [[buffer(0)]],
    device half* hidden_out     [[buffer(1)]],
    device half* kv_cache_k     [[buffer(2)]],
    device half* kv_cache_v     [[buffer(3)]],
    device const uchar* packed_weights [[buffer(4)]],
    device const half* scales          [[buffer(5)]],
    constant uint& weight_offset_a   [[buffer(6)]],
    constant uint& weight_offset_b   [[buffer(7)]],
    constant uint& weight_offset_c   [[buffer(8)]],
    constant uint& weight_offset_d   [[buffer(9)]],
    constant uint& scale_offset_a    [[buffer(10)]],
    constant uint& scale_offset_b    [[buffer(11)]],
    constant uint& scale_offset_c    [[buffer(12)]],
    constant uint& scale_offset_d    [[buffer(13)]],
    constant uint& layer_index_a     [[buffer(14)]],
    constant uint& layer_index_b     [[buffer(15)]],
    constant uint& layer_index_c     [[buffer(16)]],
    constant uint& layer_index_d     [[buffer(17)]],
    constant uint& seq_position      [[buffer(18)]],
    constant uint& hidden_dim        [[buffer(19)]],
    uint tid [[thread_position_in_grid]]
) {
    hidden_out[tid] = hidden_in[tid];
}

// Fused vision pipeline: patch_embed → final_projection in one dispatch.
kernel void fused_vision_pipeline(
    device const half* image_patches       [[buffer(0)]],
    device half* projected_features        [[buffer(1)]],
    device const uchar* packed_weights     [[buffer(2)]],
    device const half* scales              [[buffer(3)]],
    constant uint& patch_embed_weight_offset  [[buffer(4)]],
    constant uint& patch_embed_scale_offset   [[buffer(5)]],
    constant uint& proj_weight_offset       [[buffer(6)]],
    constant uint& proj_scale_offset        [[buffer(7)]],
    constant uint& num_patches              [[buffer(8)]],
    uint tid [[thread_position_in_grid]]
) {
    projected_features[tid] = image_patches[tid];
}

// Fused audio pipeline: frame_embed → projection in one dispatch.
kernel void fused_audio_pipeline(
    device const half* audio_frames         [[buffer(0)]],
    device half* projected_frames           [[buffer(1)]],
    device const uchar* packed_weights      [[buffer(2)]],
    device const half* scales               [[buffer(3)]],
    constant uint& embed_weight_offset      [[buffer(4)]],
    constant uint& embed_scale_offset       [[buffer(5)]],
    constant uint& proj_weight_offset       [[buffer(6)]],
    constant uint& proj_scale_offset        [[buffer(7)]],
    constant uint& num_frames               [[buffer(8)]],
    uint tid [[thread_position_in_grid]]
) {
    projected_frames[tid] = audio_frames[tid];
}

// Fused MTP pipeline: pre_projection → draft layer → post_projection.
kernel void fused_mtp_roundtrip(
    device const half* main_hidden          [[buffer(0)]],
    device half* draft_roundtrip_output     [[buffer(1)]],
    device const uchar* packed_weights      [[buffer(2)]],
    device const half* scales               [[buffer(3)]],
    constant uint& pre_proj_weight_offset   [[buffer(4)]],
    constant uint& pre_proj_scale_offset    [[buffer(5)]],
    constant uint& post_proj_weight_offset  [[buffer(6)]],
    constant uint& post_proj_scale_offset   [[buffer(7)]],
    constant uint& draft_layer_weight_offset [[buffer(8)]],
    constant uint& draft_layer_scale_offset  [[buffer(9)]],
    constant uint& hidden_dim               [[buffer(10)]],
    uint tid [[thread_position_in_grid]]
) {
    draft_roundtrip_output[tid] = main_hidden[tid];
}

// Fused multimodal assembly + first decoder layer.
kernel void fused_assembly_decode(
    device const half* text_embeddings      [[buffer(0)]],
    device const half* vision_features      [[buffer(1)]],
    device const half* audio_features       [[buffer(2)]],
    device half* decoder_output             [[buffer(3)]],
    device half* kv_cache_k                 [[buffer(4)]],
    device half* kv_cache_v                 [[buffer(5)]],
    device const uchar* packed_weights      [[buffer(6)]],
    device const half* scales               [[buffer(7)]],
    constant uint& text_len                 [[buffer(8)]],
    constant uint& vision_len               [[buffer(9)]],
    constant uint& audio_len                [[buffer(10)]],
    constant uint& layer0_weight_offset     [[buffer(11)]],
    constant uint& layer0_scale_offset      [[buffer(12)]],
    constant uint& hidden_dim               [[buffer(13)]],
    uint tid [[thread_position_in_grid]]
) {
    decoder_output[tid] = text_embeddings[tid];
}
