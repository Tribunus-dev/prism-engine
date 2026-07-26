// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Fused teacher-student GEMV.
//
// Loads teacher (f32 reference) and student (f32 reconstructed) weight
// matrices, computes both forward passes from the same input batch, and
// writes both output buffers — all in a single dispatch without intermediate
// memory round-trips.
//
// When debug is set (non-zero), per-element squared errors are written to
// loss_per_vec so the CPU can compute MSE without a separate comparison pass.
//
// Grid:  { num_vectors * cols, 1, 1 }
// One thread per output element — all threads independent.
//
// Buffer layout (matches FusedTeacherStudentDispatcher):
//   [[buffer(0)]]  teacher_weights — f32 reference      [rows, cols]
//   [[buffer(1)]]  student_weights — f32 reconstructed  [rows, cols]
//   [[buffer(2)]]  input_vecs      — f32                [num_vectors, rows]
//   [[buffer(3)]]  teacher_out     — f32                [num_vectors, cols]
//   [[buffer(4)]]  student_out     — f32                [num_vectors, cols]
//   [[buffer(5)]]  loss_per_vec    — f32                [num_vectors, cols] (debug)
//   [[buffer(6)]]  rows            — constant uint
//   [[buffer(7)]]  cols            — constant uint
//   [[buffer(8)]]  debug           — constant uint (0 = off, 1 = on)

#include <metal_stdlib>
using namespace metal;

kernel void fused_teacher_student_gemv(
    device const float* teacher_weights [[buffer(0)]],
    device const float* student_weights [[buffer(1)]],
    device const float* input_vecs      [[buffer(2)]],
    device float*       teacher_out     [[buffer(3)]],
    device float*       student_out     [[buffer(4)]],
    device float*       loss_per_vec    [[buffer(5)]],
    constant uint&      rows            [[buffer(6)]],
    constant uint&      cols            [[buffer(7)]],
    constant uint&      debug           [[buffer(8)]],
    uint tid                            [[thread_position_in_grid]]
) {
    // Decode thread ID into (vector index, column).
    uint vec_id = tid / cols;
    uint col    = tid % cols;

    // Input row stride and weight column base offset.
    uint vec_offset   = vec_id * rows;
    uint weight_base  = col * rows;

    float teacher_acc = 0.0f;
    float student_acc = 0.0f;

    // Unified loop: both teacher and student read the same input elements.
    for (uint r = 0; r < rows; ++r) {
        float x = input_vecs[vec_offset + r];
        teacher_acc = fma(teacher_weights[weight_base + r], x, teacher_acc);
        student_acc = fma(student_weights[weight_base + r], x, student_acc);
    }

    // Write teacher and student outputs — same thread, same index.
    teacher_out[tid] = teacher_acc;
    student_out[tid] = student_acc;

    // Debug: per-element squared error.
    if (debug != 0) {
        float diff = teacher_acc - student_acc;
        loss_per_vec[tid] = diff * diff;
    }
}
