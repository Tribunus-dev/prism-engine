#include <metal_stdlib>
using namespace metal;

// [[kernel]] fused_gate_up — reads input once, produces gate + up.
// Saves one full input re-read per MLP layer.
//
// buffer(0): input_vector    [hidden_dim] half
// buffer(1): gate_codebook   [intermediate * 16] half
// buffer(2): gate_indices    [intermediate * hidden_dim/2] u8
// buffer(3): up_codebook     [intermediate * 16] half
// buffer(4): up_indices      [intermediate * hidden_dim/2] u8
// buffer(5): output_gate     [intermediate] half
// buffer(6): output_up       [intermediate] half
// buffer(7): hidden_dim      uint
// buffer(8): intermediate     uint
kernel void fused_gate_up(
    device const half*    input_vector    [[buffer(0)]],
    device const half*    gate_codebook   [[buffer(1)]],
    device const uint8_t* gate_indices    [[buffer(2)]],
    device const half*    up_codebook     [[buffer(3)]],
    device const uint8_t* up_indices      [[buffer(4)]],
    device half*          output_gate     [[buffer(5)]],
    device half*          output_up       [[buffer(6)]],
    constant uint32_t&    hidden_dim      [[buffer(7)]],
    constant uint32_t&    intermediate    [[buffer(8)]],
    uint32_t row                          [[threadgroup_position_in_grid]],
    uint32_t tid                          [[thread_position_in_threadgroup]],
    uint32_t simd_lane                    [[thread_index_in_simdgroup]],
    uint32_t simd_id                      [[simdgroup_index_in_threadgroup]])
{
    if (row >= intermediate) return;

    uint32_t num_words = hidden_dim / 8;

    // Load gate and up codebooks into threadgroup memory
    threadgroup half gate_cb[16];
    device const half* gate_row_cb = gate_codebook + (row * 16);
    if (tid < 16) { gate_cb[tid] = gate_row_cb[tid]; }

    threadgroup half up_cb[16];
    device const half* up_row_cb = up_codebook + (row * 16);
    if (tid < 16) { up_cb[tid] = up_row_cb[tid]; }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const uint32_t* gidx = (device const uint32_t*)(gate_indices + (row * (hidden_dim / 2)));
    device const uint32_t* uidx = (device const uint32_t*)(up_indices   + (row * (hidden_dim / 2)));

    half gate_acc = 0.0h;
    half up_acc = 0.0h;

    for (uint32_t i = tid; i < num_words; i += 64) {
        uint32_t gp = gidx[i];
        uint32_t up = uidx[i];
        uint32_t off = i * 8;

        half x0 = input_vector[off + 0];
        half x1 = input_vector[off + 1];
        half x2 = input_vector[off + 2];
        half x3 = input_vector[off + 3];
        half x4 = input_vector[off + 4];
        half x5 = input_vector[off + 5];
        half x6 = input_vector[off + 6];
        half x7 = input_vector[off + 7];

        gate_acc += x0 * gate_cb[gp & 0x0F]
                  + x1 * gate_cb[(gp >> 4) & 0x0F]
                  + x2 * gate_cb[(gp >> 8) & 0x0F]
                  + x3 * gate_cb[(gp >> 12) & 0x0F]
                  + x4 * gate_cb[(gp >> 16) & 0x0F]
                  + x5 * gate_cb[(gp >> 20) & 0x0F]
                  + x6 * gate_cb[(gp >> 24) & 0x0F]
                  + x7 * gate_cb[(gp >> 28)];

        up_acc += x0 * up_cb[up & 0x0F]
                + x1 * up_cb[(up >> 4) & 0x0F]
                + x2 * up_cb[(up >> 8) & 0x0F]
                + x3 * up_cb[(up >> 12) & 0x0F]
                + x4 * up_cb[(up >> 16) & 0x0F]
                + x5 * up_cb[(up >> 20) & 0x0F]
                + x6 * up_cb[(up >> 24) & 0x0F]
                + x7 * up_cb[(up >> 28)];
    }

    gate_acc = simd_sum(gate_acc);
    up_acc = simd_sum(up_acc);

    threadgroup half shared_gate[2];
    threadgroup half shared_up[2];
    if (simd_lane == 0) { shared_gate[simd_id] = gate_acc; shared_up[simd_id] = up_acc; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        output_gate[row] = shared_gate[0] + shared_gate[1];
        output_up[row]   = shared_up[0]   + shared_up[1];
    }
}
