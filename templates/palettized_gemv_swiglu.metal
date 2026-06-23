#include <metal_stdlib>
using namespace metal;

// [[kernel]] palettized_gemv_swiglu — fused dual-palettized gate+up + SwiGLU
// Two palettized rows per output channel (gate and up), inline Swish activation.
//
// buffer(0): gate_weights [row_stride × out_dim]
// buffer(1): up_weights   [row_stride × out_dim]
// buffer(2): input_vector [in_dim] half
// buffer(3): output_vector [out_dim] half
// buffer(4): in_dim uint
// buffer(5): out_dim uint
kernel void palettized_gemv_swiglu(
    device const uint8_t* gate_weights    [[buffer(0)]],
    device const uint8_t* up_weights      [[buffer(1)]],
    device const half*    input_vector    [[buffer(2)]],
    device half*          output_vector   [[buffer(3)]],
    constant uint32_t&    in_dim          [[buffer(4)]],
    constant uint32_t&    out_dim         [[buffer(5)]],
    uint32_t row                          [[threadgroup_position_in_grid]],
    uint32_t tid                          [[thread_position_in_threadgroup]],
    uint32_t simd_lane                    [[thread_index_in_simdgroup]],
    uint32_t simd_id                      [[simdgroup_index_in_threadgroup]])
{
    uint32_t row_stride = 32 + (in_dim / 2);
    device const uint8_t* gate_base = gate_weights + (row * row_stride);
    device const uint8_t* up_base   = up_weights + (row * row_stride);

    // Load both codebooks into threadgroup memory
    threadgroup half shared_cb_gate[16];
    threadgroup half shared_cb_up[16];
    if (tid < 16) {
        shared_cb_gate[tid] = reinterpret_cast<device const half*>(gate_base)[tid];
        shared_cb_up[tid]   = reinterpret_cast<device const half*>(up_base)[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Vectorized index processing (8 nibbles per uint32 read)
    device const uint32_t* gate_idx =
        reinterpret_cast<device const uint32_t*>(gate_base + 32);
    device const uint32_t* up_idx =
        reinterpret_cast<device const uint32_t*>(up_base + 32);
    uint32_t num_words = in_dim / 8;

    half gate_acc = 0.0h;
    half up_acc   = 0.0h;

    for (uint32_t i = tid; i < num_words; i += 64) {
        uint32_t g_pack = gate_idx[i];
        uint32_t u_pack = up_idx[i];
        uint32_t off = i * 8;

        // Scalar input loads (half8 extended vector type removed in Metal 3+)
        half x0 = input_vector[off + 0];
        half x1 = input_vector[off + 1];
        half x2 = input_vector[off + 2];
        half x3 = input_vector[off + 3];
        half x4 = input_vector[off + 4];
        half x5 = input_vector[off + 5];
        half x6 = input_vector[off + 6];
        half x7 = input_vector[off + 7];

        // Dequantize gate codebook values
        half gw0 = shared_cb_gate[g_pack & 0x0F];
        half gw1 = shared_cb_gate[(g_pack >> 4) & 0x0F];
        half gw2 = shared_cb_gate[(g_pack >> 8) & 0x0F];
        half gw3 = shared_cb_gate[(g_pack >> 12) & 0x0F];
        half gw4 = shared_cb_gate[(g_pack >> 16) & 0x0F];
        half gw5 = shared_cb_gate[(g_pack >> 20) & 0x0F];
        half gw6 = shared_cb_gate[(g_pack >> 24) & 0x0F];
        half gw7 = shared_cb_gate[(g_pack >> 28)];

        // Dequantize up codebook values
        half uw0 = shared_cb_up[u_pack & 0x0F];
        half uw1 = shared_cb_up[(u_pack >> 4) & 0x0F];
        half uw2 = shared_cb_up[(u_pack >> 8) & 0x0F];
        half uw3 = shared_cb_up[(u_pack >> 12) & 0x0F];
        half uw4 = shared_cb_up[(u_pack >> 16) & 0x0F];
        half uw5 = shared_cb_up[(u_pack >> 20) & 0x0F];
        half uw6 = shared_cb_up[(u_pack >> 24) & 0x0F];
        half uw7 = shared_cb_up[(u_pack >> 28)];

        // Fused dot products (scalar)
        gate_acc += x0 * gw0 + x1 * gw1 + x2 * gw2 + x3 * gw3
                  + x4 * gw4 + x5 * gw5 + x6 * gw6 + x7 * gw7;
        up_acc   += x0 * uw0 + x1 * uw1 + x2 * uw2 + x3 * uw3
                  + x4 * uw4 + x5 * uw5 + x6 * uw6 + x7 * uw7;
    }

    // SIMD-group reductions (both accumulators via hardware shuffle)
    gate_acc = simd_sum(gate_acc);
    up_acc   = simd_sum(up_acc);

    // Inter-SIMD reduction via shared memory (2 SIMD-groups × 2 accumulators)
    threadgroup half shared_rg[2];
    threadgroup half shared_ru[2];
    if (simd_lane == 0) {
        shared_rg[simd_id] = gate_acc;
        shared_ru[simd_id] = up_acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Thread 0: finalize gate, compute Swish, apply GLU gating
    if (tid == 0) {
        half final_gate = shared_rg[0] + shared_rg[1];
        half final_up   = shared_ru[0] + shared_ru[1];

        // Swish activation: x * sigmoid(x) computed in f32
        float g_f32 = static_cast<float>(final_gate);
        half swish_gate = static_cast<half>(g_f32 / (1.0f + exp(-g_f32)));

        // GLU: swish(gate) * up
        output_vector[row] = swish_gate * final_up;
    }
}
