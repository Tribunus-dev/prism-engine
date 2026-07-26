// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Canonical NF4 decode helpers — symmetric [-1,1] codebook from the QLoRA paper.
// Packed as 2 nibbles per byte: low nibble = even index, high nibble = odd index.
// Groups use per-group float scale (f32) and optional bias (f32).
//
// Include once per translation unit:
//   #include "nf4_decode.metal"
//
// (The #include guard is advisory; Metal concatenates at compile time and this
//  header includes neither types nor defines that conflict on re-inclusion.)

#ifndef __METAL_VERSION__
#error "Metal shader only"
#endif

#include <metal_stdlib>
using namespace metal;

// Symmetric [-1,1] NF4 codebook — MUST match compile/quantize.rs::NF4_CODEBOOK.
constant float nf4_codebook[16] = {
    -1.0f, -0.6961928f, -0.5250731f, -0.3949175f,
    -0.2844414f, -0.1847734f, -0.09105f, 0.0f,
     0.0795803f, 0.1609302f, 0.2461123f, 0.3379152f,
     0.4407099f, 0.562617f,  0.7229568f, 1.0f
};

// Extract the NF4 value at logical index `index` from a packed nibble array.
// Low nibble = even index, high nibble = odd index.
static float unpack_nf4(device const uchar* packed, uint index) {
    uchar byte = packed[index >> 1];
    uchar nibble = (index & 1) ? (byte >> 4) : (byte & 0x0Fu);
    return nf4_codebook[nibble];
}

// [[maybe_unused]] — some kernels use unpack_nf4 directly; this helper
// adds per-group scale+bias dequantization for those that need it.
// Dequantize one NF4 element:  val = codebook[nibble] * scale + bias.
// Groups are indexed as  (index / group_size).
// When `biases` is null the bias term is zero.
[[maybe_unused]] static float dequantize_nf4(
    device const uchar* packed,
    device const float* scales,
    device const float* biases,
    uint index,
    uint group_size) {
    uint group = index / group_size;
    float scale = scales[group];
    float bias = biases ? biases[group] : 0.0f;
    return fma(unpack_nf4(packed, index), scale, bias);
}
