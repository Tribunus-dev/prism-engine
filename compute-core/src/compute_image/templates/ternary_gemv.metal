// ── TERNARY GEMV ───────────────────────────────────────────────────────────
// Branch-free ternary-weight GEMV kernel for 2-bit packed ternary weights.
//
// Performance characteristics (expected on Apple Silicon GPU):
//   - ~2× memory bandwidth reduction vs FP16 weights (2 bits vs 16 bits)
//   - Zero multiply instructions — all operations are conditional selects and
//     additions, maximizing ALU throughput on Apple GPUs
//   - One thread per output row → maximum parallelism for batch=1 matvec
//   - Half accumulator → matches input/output precision, sufficient for ternary
//
// Encoding:
//   2 bits per weight, 4 weights packed per UINT8 byte.
//   Compiler encoding (ternary_compile.rs): 0b00 = 0, 0b01 = +1, 0b10 = -1
//   This kernel decodes to match the compiler (00→0, 01→+iv, 10→-iv).
//   The ternary set {+1, 0, -1} means no multiplication is needed — just
//   conditional add, pass, or subtract of the input element.
//   All condition checks use select() — zero branching divergence.
//
// Buffer layout:
//   [0] packed_weights [N * K/4] uint8_t  — packed ternary weights, row-major
//   [1] input          [K] half             — input vector (1D, one row)
//   [2] output         [N] half             — result vector
//   [3] in_dim         uint                 — input dimension (K)
//   [4] out_dim        uint                 — output dimension (N)
//
// K must be a multiple of 4 (4 weights per byte).
//
// Thread count: N threads (one per output row).
//
// ── select() strategy ─────────────────────────────────────────────────────
// For each 2-bit nibble n:
//   n == 0 (00): zero              →   0.0h
//   n == 1 (01): keep input as-is  →   iv
//   n == 2 (10): negate input      →  -iv
//   n == 3 (11, reserved): zero    →   0.0h
//
// Two nested select() calls per weight produce the correct value:
//   inner: select(0.0h, iv, n == 1)    —  iv when n==1, else 0.0h
//   outer: select(inner, -iv, n == 2)   — -iv when n==2, else inner result
// Total: 8 select() calls per byte (2 per weight × 4 weights).

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

    half sum = 0.0h;

    for (uint i = 0; i < packed_cols; ++i) {
        uint8_t byte   = packed_weights[offset + i];
        half4   iv     = *((device const half4*)(input + i * 4));

        uint nibble0 = uint(byte)       & 0x03u;
        uint nibble1 = (uint(byte) >> 2) & 0x03u;
        uint nibble2 = (uint(byte) >> 4) & 0x03u;
        uint nibble3 = (uint(byte) >> 6) & 0x03u;

        // select(a, b, cond) → a if cond is false, b if cond is true
        half4 tmp;
        // Fixed: matches compiler encoding 00=0, 01=+1, 10=-1
        // Inner: when nibble==1 → +iv; outer: when nibble==2 → -iv
        tmp.x = select(select(0.0h, iv.x, nibble0 == 1u), -iv.x, nibble0 == 2u);
        tmp.y = select(select(0.0h, iv.y, nibble1 == 1u), -iv.y, nibble1 == 2u);
        tmp.z = select(select(0.0h, iv.z, nibble2 == 1u), -iv.z, nibble2 == 2u);
        tmp.w = select(select(0.0h, iv.w, nibble3 == 1u), -iv.w, nibble3 == 2u);

        sum += tmp.x + tmp.y + tmp.z + tmp.w;
    }

    output[row] = sum;
}
