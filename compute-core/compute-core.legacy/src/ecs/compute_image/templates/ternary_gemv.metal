// ── TERNARY GEMV (legacy 2-bit format) ──────────────────────────────────────
// Branch-free ternary-weight GEMV for 2-bit packed ternary weights.
//
// ⚠ FORMAT NOTE: this kernel consumes the LEGACY 2-bit packing (4 weights/byte,
// = 2.0 bits/weight) and applies NO block scale (unit scale). For the compiler's
// real weight format use `ternary_tile640_gemv.metal`, which consumes the
// space-optimal tile640 base-3 packing (20 trits/u32 = 1.6 bits/weight) AND
// applies the per-256 block scales. Prefer that kernel for production; this one
// is kept for the scale-free 2-bit path and legacy fixtures.
//
// PRECISION: this version accumulates in fp32 (was fp16). Ternary needs no
// multiply — each 2-bit code selects add / skip / subtract of the input element
// — but the running sum must be fp32 to avoid precision loss over long rows.
//
// Encoding (matches the compiler): 00 = 0, 01 = +1, 10 = -1, 11 = 0.
//
// Buffer layout:
//   [0] packed_weights [N * K/4] uint8_t  — packed ternary weights, row-major
//   [1] input          [K] half           — input vector (1D, one row)
//   [2] output         [N] half           — result vector
//   [3] in_dim         uint               — input dimension (K)
//   [4] out_dim        uint               — output dimension (N)
// K must be a multiple of 4 (4 weights per byte). Thread count: N (one/row).

#include <metal_stdlib>
using namespace metal;

kernel void ternary_gemv(
    device const uint8_t* packed_weights [[buffer(0)]],  // [N * K/4]
    device const half*    input          [[buffer(1)]],  // [K]
    device half*          output         [[buffer(2)]],  // [N]
    constant uint&        in_dim         [[buffer(3)]],  // K
    constant uint&        out_dim        [[buffer(4)]],  // N
    uint                  row            [[thread_position_in_grid]])
{
    if (row >= out_dim) return;

    uint packed_cols = in_dim / 4;  // 4 weights per byte
    uint offset      = row * packed_cols;

    float sum = 0.0f; // fp32 accumulation

    for (uint i = 0; i < packed_cols; ++i) {
        uint8_t byte = packed_weights[offset + i];
        float4  iv   = float4(*((device const half4*)(input + i * 4)));

        uint n0 =  uint(byte)       & 0x03u;
        uint n1 = (uint(byte) >> 2) & 0x03u;
        uint n2 = (uint(byte) >> 4) & 0x03u;
        uint n3 = (uint(byte) >> 6) & 0x03u;

        // 00,11 -> 0 ; 01 -> +iv ; 10 -> -iv   (branch-free select in fp32)
        sum += select(select(0.0f, iv.x, n0 == 1u), -iv.x, n0 == 2u);
        sum += select(select(0.0f, iv.y, n1 == 1u), -iv.y, n1 == 2u);
        sum += select(select(0.0f, iv.z, n2 == 1u), -iv.z, n2 == 2u);
        sum += select(select(0.0f, iv.w, n3 == 1u), -iv.w, n3 == 2u);
    }

    output[row] = half(sum);
}
