// SPDX-License-Identifier: MIT OR Apache-2.0
//
// RMSNorm for CImage — single-vector normalization.
//
// All 64 threads in the single threadgroup process hidden_dim elements
// cooperatively via a strided loop:
//   1. Accumulate sum-of-squares across all elements (simd + threadgroup reduction)
//   2. Compute rms = sqrt(mean(sum_sq) + epsilon)
//   3. Normalize: output[i] = weight[i] * (input[i] / rms)
//
// Buffer layout:
//   [0] input    [hidden_dim] f32
//   [1] weight   [hidden_dim] f32
//   [2] output   [hidden_dim] f32
//   [3] constants (MlpConstants)

#include <metal_stdlib>
using namespace metal;

struct MlpConstants {
    uint32_t hidden_dim;
    uint32_t intermediate_dim;
    uint32_t group_size;
    uint32_t codec_id;
    float    epsilon;
    uint32_t _pad[3];
};

kernel void cimage_rmsnorm_f32(
    device const float*     input   [[buffer(0)]],
    device const float*     weight  [[buffer(1)]],
    device float*           output  [[buffer(2)]],
    constant MlpConstants&  c       [[buffer(3)]],
    uint tid                         [[thread_index_in_threadgroup]],
    uint gid                         [[threadgroup_position_in_grid]]
) {
    uint hidden_dim = c.hidden_dim;
    float eps = c.epsilon;

    // ── Phase 1: strided sum-of-squares ──
    float sum_sq = 0.0f;
    for (uint i = tid; i < hidden_dim; i += 64) {
        float x = input[i];
        sum_sq += x * x;
    }

    // SIMD-group reduction (32 lanes each).
    sum_sq = simd_sum(sum_sq);

    // Threadgroup-level reduction: two simdgroups per 64-thread threadgroup.
    threadgroup float shared[2];
    uint simd_lane = tid & 31;
    uint simd_id   = tid >> 5;
    if (simd_lane == 0) {
        shared[simd_id] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total_sum_sq = shared[0] + shared[1];
        float rms = sqrt(total_sum_sq / (float)hidden_dim + eps);
        shared[0] = rms;  // reuse: broadcast to all threads
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 2: normalize ──
    float rms = shared[0];
    for (uint i = tid; i < hidden_dim; i += 64) {
        output[i] = weight[i] * (input[i] / rms);
    }
}
