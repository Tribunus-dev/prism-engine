// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Batched FP32 GEMV on GPU for operator validation.
// Each thread computes one output element: output[vec_id * cols + col].
// Thread count = num_vectors × cols.
//
// This is the GPU accelerator for the admission-pipeline operator validation.
// All six matmuls (ref_stress, quant_stress, ref_promo, quant_promo,
// ref_holdout, quant_holdout) are dispatched as independent grid runs against
// the same weight matrix, sharing one MTLCommandBuffer commit.

#include <metal_stdlib>
using namespace metal;

/// Single-output-element GEMV: output[tid] = Σ_r input[row] × weight[r, col].
///
/// Grid dimensions:  { num_vectors * cols, 1, 1 }
/// One thread per output element — all threads independent.
kernel void batched_gemv_fp32(
    device const float* weights    [[buffer(0)]],  // [rows, cols]
    device const float* input_vecs [[buffer(1)]],  // [num_vectors, rows]
    device float* output_vecs      [[buffer(2)]],  // [num_vectors, cols]
    constant uint& rows            [[buffer(3)]],
    constant uint& cols            [[buffer(4)]],
    uint tid                       [[thread_position_in_grid]]
) {
    uint vec_id = tid / cols;
    uint col    = tid % cols;

    float acc = 0.0;
    for (uint r = 0; r < rows; ++r) {
        acc += input_vecs[vec_id * rows + r] * weights[r * cols + col];
    }
    output_vecs[tid] = acc;
}
