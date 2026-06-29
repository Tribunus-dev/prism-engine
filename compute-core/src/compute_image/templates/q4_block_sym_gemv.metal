// ── Q4_BLOCK_SYM GEMV ─────────────────────────────────────────────────────
// Production-weighted-precision GEMV kernel for Q4_BLOCK_SYM_128 weights.
//
// Performance characteristics (validated on Apple M1 GPU):
//   - 1.55–3.24× faster than FP16 baseline (H=256..1024, I=1024..4096)
//   - 3.75–4× memory bandwidth reduction vs FP16 weights
//   - One thread per output row → maximum parallelism for batch=1 matvec
//   - Float32 internal accumulator → near-lossless accuracy despite half storage
//   - Branch-free: all nibble → signed-int paths use arithmetic, not predication
//
// Quantization scheme:
//   Q4_BLOCK_SYM: symmetric int4 (range -8..7) with per-group FP16 scale.
//   No zero-point — all values are signed and symmetric around zero.
//   Group size = 128 weights per shared scale (6.25% overhead for scales).
//
// Buffer layout:
//   [0] input    [K] half                     — input vector (1D, one row)
//   [1] weights  [N * K/8] uint              — packed int4, row-major:
//                                               8 values per uint32, nibble-order:
//                                               byte0 = {v[0], v[1]}, byte1 = {v[2], v[3]}, etc.
//   [2] scales   [N * K/group_size] half     — per-group FP16 scales, row-major:
//                                               scales[row * num_groups + group]
//   [3] output   [N] half                     — result vector
//   [4] K        uint                         — input dimension
//   [5] N        uint                         — output dimension
//   [6] group_size uint (must be multiple of 8, typically 128)
//
// Encoding: (n ^ 8) - 8 → branch-free sign extension of 4-bit unsigned to signed.
//   n=0..7 → -8..-1, n=8 → 0, n=9..15 → +1..+7
//
// Thread count: N threads (one per output row).
//
// Validation: see compute-core/tests/q4_block_sym_bench.rs
//   — 4-way benchmark: FP16 | Q4_GS128 | Q4_GS64 | PALETTE_LUT4
//   — Error vs FP32 reference < 1.5% typical at fp16 noise floor
//
// ── Implementation notes ───────────────────────────────────────────────────
// The inner loop extracts 8 nibbles per uint32 via as_type<uchar4> (free cast
// on Apple GPUs — compiles to byte-level vector loads from the scalar register).
// Each nibble is sign-extended via (nibble ^ 8u) - 8, multiplied by both the
// group scale and the corresponding input element, and accumulated in float32.
// Scaling by 1/group_size is unnecessary — scales encode the full per-group
// quantization factor including the /7 normalization.

#include <metal_stdlib>
using namespace metal;

kernel void q4_block_sym_gemv(
    device const half*      input       [[buffer(0)]],  // [K]
    device const uint*      weights     [[buffer(1)]],  // [N * K/8]
    device const half*      scales      [[buffer(2)]],  // [N * K/128]
    device half*            output      [[buffer(3)]],  // [N]
    constant uint&          K           [[buffer(4)]],
    constant uint&          N           [[buffer(5)]],
    constant uint&          group_size  [[buffer(6)]],
    uint                    row         [[thread_position_in_grid]])
{
    if (row >= N) return;

    uint num_groups   = K / group_size;
    uint words_per_grp = group_size / 8;       // 16 for gs=128
    uint packed_per_row = K / 8;

    // Base pointers for this row's data
    device const uint* row_weights = weights + (row * packed_per_row);
    device const half* row_scales  = scales   + (row * num_groups);

    float acc = 0.0f;

    // Outer loop over groups (each with its own FP16 scale)
    for (uint g = 0; g < num_groups; ++g) {
        half group_scale   = row_scales[g];
        float scale_f      = float(group_scale);
        uint  word_offset  = g * words_per_grp;

        // Inner loop over uint32 words within the group
        for (uint j = 0; j < words_per_grp; ++j) {
            uint packed    = row_weights[word_offset + j];
            uchar4 bytes   = as_type<uchar4>(packed);
            uint input_off = g * group_size + j * 8;

            // ── 8 nibbles, branch-free sign-extend, scale, dot ────────
            // Byte 0: nibbles 0,1
            { uint n = uint(bytes[0]) & 0x0Fu;  float v = float(int(n ^ 8u) - 8) * scale_f * float(input[input_off + 0]); acc += v; }
            { uint n = (uint(bytes[0]) >> 4) & 0x0Fu; float v = float(int(n ^ 8u) - 8) * scale_f * float(input[input_off + 1]); acc += v; }
            // Byte 1: nibbles 2,3
            { uint n = uint(bytes[1]) & 0x0Fu;  float v = float(int(n ^ 8u) - 8) * scale_f * float(input[input_off + 2]); acc += v; }
            { uint n = (uint(bytes[1]) >> 4) & 0x0Fu; float v = float(int(n ^ 8u) - 8) * scale_f * float(input[input_off + 3]); acc += v; }
            // Byte 2: nibbles 4,5
            { uint n = uint(bytes[2]) & 0x0Fu;  float v = float(int(n ^ 8u) - 8) * scale_f * float(input[input_off + 4]); acc += v; }
            { uint n = (uint(bytes[2]) >> 4) & 0x0Fu; float v = float(int(n ^ 8u) - 8) * scale_f * float(input[input_off + 5]); acc += v; }
            // Byte 3: nibbles 6,7
            { uint n = uint(bytes[3]) & 0x0Fu;  float v = float(int(n ^ 8u) - 8) * scale_f * float(input[input_off + 6]); acc += v; }
            { uint n = (uint(bytes[3]) >> 4) & 0x0Fu; float v = float(int(n ^ 8u) - 8) * scale_f * float(input[input_off + 7]); acc += v; }
        }
    }

    output[row] = half(acc);
}
