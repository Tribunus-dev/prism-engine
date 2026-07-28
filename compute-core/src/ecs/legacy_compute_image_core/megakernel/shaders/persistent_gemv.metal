#include <metal_stdlib>
using namespace metal;

constant uint HIDDEN_DIM  = 3840;
constant uint BLOCKS_PER_ROW = HIDDEN_DIM / 32; // 120

kernel void matvec_persistent_t32_coalesced(
    device const uint4*  weight_stream  [[buffer(0)]],
    device const half*   activation     [[buffer(1)]],
    device half*         output         [[buffer(2)]],
    uint ti                               [[thread_index_in_threadgroup]],
    uint tp                               [[threadgroup_position_in_grid]])
{
    // Cooperative load of all activations into SRAM
    threadgroup half sram[HIDDEN_DIM];
    for (uint i = ti; i < HIDDEN_DIM; i += 32) {
        sram[i] = activation[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint row = tp;
    const uint lane = ti % 32;
    float my_acc = 0.0f;
    uint row_base = row * BLOCKS_PER_ROW;
    half unpacked[32];
    half act_reg[32];

    for (uint bg = 0; bg < 128; bg += 32) {
        uint g = bg / 32;

        // Copy this iteration's 32 activation elements into registers
        uint act_base = g * 1024 + lane * 32;
        for (uint e = 0; e < 32; ++e) {
            act_reg[e] = sram[act_base + e];
        }

        uint logical_block = bg + lane;
        bool is_valid = logical_block < BLOCKS_PER_ROW;
        uint safe_block = is_valid ? logical_block : (BLOCKS_PER_ROW - 1);

        uint4 vec = weight_stream[row_base + safe_block];
        thread const uchar* raw = (thread const uchar*)&vec;
        ushort scale_bits = ((ushort)raw[7]) | ((ushort)raw[8] << 8);
        half scale = as_type<half>(scale_bits);

        // Full 32-element unpack from block
        for (uint i = 0; i < 7; ++i) {
            uchar bv = raw[i];
            uint v = (uint)bv;
            uint n = (i < 6) ? 5 : 2;
            for (uint j = 0; j < n; ++j) {
                unpacked[i * 5 + j] = (half)((int)(v % 3) - 1);
                v = v / 3;
            }
        }

        // Dot product from registers (zero SRAM bank conflicts)
        float local_sum = 0.0f;
        for (uint e = 0; e < 32; ++e) {
            local_sum += (float)(unpacked[e] * scale * act_reg[e]);
        }
        my_acc += is_valid ? local_sum : 0.0f;
    }

    float total = simd_sum(my_acc);
    if (lane == 0) {
        output[row] = (half)total;
    }
}
