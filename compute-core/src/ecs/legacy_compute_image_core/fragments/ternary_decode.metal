// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Canonical ternary decode helpers — base-3 packed weights.
// Each u32 holds 20 ternary digits (trits), packed LSB-first:
//
//   val = d0 + d1*3 + d2*3^2 + ... + d19*3^19
//
// where each digit d ∈ {0, 1, 2} maps to weight w ∈ {-1, 0, +1}.
//
// The canonical digit-to-weight mapping is:
//   0 → -1  (negative)
//   1 →  0  (zero)
//   2 → +1  (positive)
//
// Include once per translation unit:
//   #include "ternary_decode.metal"

#ifndef __METAL_VERSION__
#error "Metal shader only"
#endif

constant uint TERNARY_MAGIC_DIV3 = 2863311531u;  // ceil(2^33/3) for fast uint ÷ 3

// Fast unsigned division by 3 using the multiplicative inverse.
inline uint fast_div3(uint v) {
    return ((uint64_t)v * (uint64_t)TERNARY_MAGIC_DIV3) >> 33;
}

// Fast unsigned modulo 3 using the division result.
inline uint fast_mod3(uint v) {
    return v - fast_div3(v) * 3u;
}

// Extract the i-th trit (0 ≤ i < 20) from a packed u32.
// i=0 returns the LSB trit (first weight).
inline uint unpack_trit(uint val, uint i) {
    // Shift right by i × log2(3) ≈ 1.585 bits per trit.
    // Using multiplication by the inverse avoids a loop over fast_mod3 per trit.
    // For i < 20 this is about 32 multiplications — on Apple GPU that's fine.
    // An alternative is iterative extraction (cheaper for sequential access).
    // We provide both; pick the right one for your access pattern.
    uint v = val;
    for (uint j = 0; j < i; ++j) {
        v = fast_div3(v);
    }
    return fast_mod3(v);
}

// Convert a trit {0, 1, 2} to a signed weight {-1, 0, +1}.
inline int trit_to_weight(uint trit) {
    return (int)trit - 1;
}

// Iterative trit extraction — best when consuming all 20 trits sequentially.
// Produces one `val = fast_div3(val)` per element, so the whole loop is just
// 20 mul+shifts and 20 mods.
// Example usage:
//   uint val = packed_word;
//   for (uint i = 0; i < 20; ++i) {
//       uint trit = val - fast_div3(val) * 3u;  // same as fast_mod3
//       int wgt = (int)trit - 1;
//       val = fast_div3(val);
//   }

// Dequantize one ternary weight with a scalar scale.
// trit 0 → -scale, trit 1 → 0, trit 2 → +scale.
inline float dequantize_ternary(uint trit, float scale) {
    // The canonical mapping: 0→-scale, 1→0, 2→+scale
    return (trit == 1) ? 0.0f : ((trit == 2) ? scale : -scale);
}
