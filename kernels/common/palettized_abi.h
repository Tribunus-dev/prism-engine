/* palettized_abi.h — shared ABI for the fused palettized / ternary GEMV kernels.
 *
 * This is the single source of truth for the on-device weight layout that the
 * CUDA (NVIDIA), HIP (AMD), and SYCL (Intel) ports all implement identically.
 * It mirrors templates/palettized_gemv.metal and compute-core's ternary_gemv.metal
 * so a `.cimage` compiled once runs bit-compatibly on every backend.
 *
 * The layout is verified by kernels/oracle/palettized_oracle.cpp (run it in CI).
 *
 * ── Palettized (LUT) format ────────────────────────────────────────────────
 *   input_vector   [in_dim]              fp16
 *   codebook_block [out_dim * 16]        fp16   — 16-entry LUT per output row
 *   indices_block  [out_dim * in_dim/2]  u8     — 4-bit indices, 2 per byte
 *   output_vector  [out_dim]             fp16
 *
 *   Nibble order (LITTLE-ENDIAN, verified): reinterpret each row's indices as
 *   u32 words; 8 nibbles per word; element (8*w + j) uses nibble
 *   (word >> (4*j)) & 0xF, for j = 0..7. Equivalently the LSB nibble of byte b
 *   is the even element 2*b. `in_dim` MUST be a multiple of 8.
 *
 * ── Ternary format (2-bit) ─────────────────────────────────────────────────
 *   packed_weights [out_dim * in_dim/4] u8   — 4 weights per byte, 2 bits each
 *   Encoding (matches the Rust compiler): 00 = 0, 01 = +1, 10 = -1, 11 = 0.
 *   No multiply: conditional add / pass / subtract of the input element.
 *   `in_dim` MUST be a multiple of 4.
 *
 * ── Numerics ───────────────────────────────────────────────────────────────
 *   All ports accumulate in fp32 (more accurate than the reference Metal
 *   kernel's fp16 accumulator) and store fp16. If you need bit-parity with the
 *   Metal kernel instead of higher accuracy, switch the accumulator to fp16.
 */
#ifndef PRISM_PALETTIZED_ABI_H
#define PRISM_PALETTIZED_ABI_H

#define PRISM_CODEBOOK_SIZE     16   /* LUT entries per output row (4-bit index) */
#define PRISM_NIBBLES_PER_WORD  8    /* 4-bit indices packed per 32-bit word     */
#define PRISM_TERNARY_PER_BYTE  4    /* 2-bit ternary weights packed per byte     */

#endif /* PRISM_PALETTIZED_ABI_H */
