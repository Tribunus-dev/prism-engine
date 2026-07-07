// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Fused teacher-student GEMV.
//
// A single threadgrid computes both teacher (f32 reference) and student
// (quantized reconstruction) forward passes in one input load.
//
// Grid dimensions: { num_vectors * cols, 1, 1 }
// One thread per output element — all threads independent.
//
// When debug_outputs is nonzero:
//   Write per-element squared error to loss_per_vec for CPU-side reduction.
// Otherwise loss_per_vec is never accessed.

#include <metal_stdlib>
using namespace metal;

kernel void fused_teacher_student_gemv(
    device const float* teacher_weights [[buffer(0)]],  // [rows, cols]
    device const float* student_weights [[buffer(1)]],  // [rows, cols]
    device const float* input_vecs      [[buffer(2)]],  // [num_vectors, rows]
    device float* teacher_out          [[buffer(3)]],  // [num_vectors, cols]
    device float* student_out          [[buffer(4)]],  // [num_vectors, cols]
    device float* loss_per_vec         [[buffer(5)]],  // [num_vectors, cols] — debug only
    constant uint&  rows               [[buffer(6)]],
    constant uint&  cols               [[buffer(7)]],
    constant uint&  debug_outputs      [[buffer(8)]],
    uint tid                           [[thread_position_in_grid]]
) {
    uint vec_id = tid / cols;
    uint col    = tid % cols;

    float t_acc = 0.0;
    float s_acc = 0.0;
    for (uint r = 0; r < rows; ++r) {
        float x = input_vecs[vec_id * rows + r];
        t_acc += x * teacher_weights[r * cols + col];
        s_acc += x * student_weights[r * cols + col];
    }

    teacher_out[tid] = t_acc;
    student_out[tid] = s_acc;

    if (debug_outputs != 0) {
        float diff = t_acc - s_acc;
        loss_per_vec[tid] = diff * diff;
    }
}
