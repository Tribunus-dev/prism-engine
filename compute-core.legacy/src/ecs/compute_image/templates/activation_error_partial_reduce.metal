#include <metal_stdlib>
using namespace metal;

// ── Struct definitions (must match kernel_types.rs #[repr(C)]) ──────────────

struct ProjectionParams {
    uint   in_dim;
    uint   out_dim;
    uint   page_count;
    uint   page_width;
    uint   mode_flags;
    uint   probe_seed;
    uint   reserved[5];
};

struct ErrorPartial {
    float  sum_sq_error;
    float  sum_abs_error;
    float  dot_teacher_student;
    float  sum_teacher_sq;
    float  sum_student_sq;
    float  max_abs_error;
    uint   element_count;
    uint   _pad;
};

// ── Function constants ──────────────────────────────────────────────────────

constant uint tile_size [[function_constant(0)]];

// ── Activation error partial reduction ───────────────────────────────────────
//
// Streaming elementwise comparison of teacher vs student buffers. Each
// threadgroup processes one tile of elements and writes one ErrorPartial
// record. The CPU or Accelerate lane reduces these records in canonical order.
//
// buffer(0): teacher  [total_elements] half
// buffer(1): student  [total_elements] half
// buffer(2): output   [ceil(total_elements / tile_size)] ErrorPartial
// buffer(3): params   ProjectionParams
//
// Grid: one threadgroup per tile. Threadgroup size = 64 threads.
kernel void activation_error_partial_reduce(
    device const half*         teacher    [[buffer(0)]],
    device const half*         student    [[buffer(1)]],
    device ErrorPartial*       output     [[buffer(2)]],
    constant ProjectionParams& params     [[buffer(3)]],
    uint gid                               [[threadgroup_position_in_grid]],
    uint tid                               [[thread_position_in_threadgroup]],
    uint simd_lane                          [[thread_index_in_simdgroup]],
    uint simd_id                            [[simdgroup_index_in_threadgroup]])
{
    const uint total_elements = params.in_dim;
    const uint start = gid * tile_size;
    const uint end   = min(start + tile_size, total_elements);

    if (start >= total_elements) return;

    // ── Threadgroup accumulator tile ───────────────────────────────────────
    // Two SIMD groups (64 threads / 32 lanes), one accumulator per group.
    threadgroup float tg_sum_sq_error[2];
    threadgroup float tg_sum_abs_error[2];
    threadgroup float tg_dot_teacher_student[2];
    threadgroup float tg_sum_teacher_sq[2];
    threadgroup float tg_sum_student_sq[2];
    threadgroup float tg_max_abs_error[2];
    threadgroup uint tg_element_count[2];

    // Thread 0 initialises the entire shared buffer.
    if (tid == 0) {
        tg_sum_sq_error[0]       = 0.0f;
        tg_sum_sq_error[1]       = 0.0f;
        tg_sum_abs_error[0]      = 0.0f;
        tg_sum_abs_error[1]      = 0.0f;
        tg_dot_teacher_student[0] = 0.0f;
        tg_dot_teacher_student[1] = 0.0f;
        tg_sum_teacher_sq[0]     = 0.0f;
        tg_sum_teacher_sq[1]     = 0.0f;
        tg_sum_student_sq[0]     = 0.0f;
        tg_sum_student_sq[1]     = 0.0f;
        tg_max_abs_error[0]      = 0.0f;
        tg_max_abs_error[1]      = 0.0f;
        tg_element_count[0]      = 0u;
        tg_element_count[1]      = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Per-thread accumulators ────────────────────────────────────────────
    float sum_sq_error       = 0.0f;
    float sum_abs_error      = 0.0f;
    float dot_teacher_student = 0.0f;
    float sum_teacher_sq     = 0.0f;
    float sum_student_sq     = 0.0f;
    float max_abs_error      = 0.0f;
    uint  element_count      = 0u;

    // ── Stream over elements assigned to this thread ───────────────────────
    for (uint i = start + tid; i < end; i += 64) {
        const float teacher_val = float(teacher[i]);
        const float student_val = float(student[i]);
        const float err         = teacher_val - student_val;

        sum_sq_error       = fma(err, err, sum_sq_error);
        sum_abs_error      += abs(err);
        dot_teacher_student = fma(teacher_val, student_val, dot_teacher_student);
        sum_teacher_sq     = fma(teacher_val, teacher_val, sum_teacher_sq);
        sum_student_sq     = fma(student_val, student_val, sum_student_sq);
        max_abs_error       = max(max_abs_error, abs(err));
        element_count      += 1u;
    }

    // ── SIMD-group reductions ──────────────────────────────────────────────
    sum_sq_error       = simd_sum(sum_sq_error);
    sum_abs_error      = simd_sum(sum_abs_error);
    dot_teacher_student = simd_sum(dot_teacher_student);
    sum_teacher_sq     = simd_sum(sum_teacher_sq);
    sum_student_sq     = simd_sum(sum_student_sq);
    max_abs_error       = simd_max(max_abs_error);
    element_count      = simd_sum(element_count);

    // ── Write SIMD results into threadgroup ────────────────────────────────
    if (simd_lane == 0) {
        tg_sum_sq_error[simd_id]       = sum_sq_error;
        tg_sum_abs_error[simd_id]      = sum_abs_error;
        tg_dot_teacher_student[simd_id] = dot_teacher_student;
        tg_sum_teacher_sq[simd_id]     = sum_teacher_sq;
        tg_sum_student_sq[simd_id]     = sum_student_sq;
        tg_max_abs_error[simd_id]      = max_abs_error;
        tg_element_count[simd_id]      = element_count;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Final cross-SIMD reduction ─────────────────────────────────────────
    if (tid == 0) {
        device ErrorPartial& out = output[gid];
        out.sum_sq_error        = tg_sum_sq_error[0]       + tg_sum_sq_error[1];
        out.sum_abs_error       = tg_sum_abs_error[0]      + tg_sum_abs_error[1];
        out.dot_teacher_student = tg_dot_teacher_student[0] + tg_dot_teacher_student[1];
        out.sum_teacher_sq      = tg_sum_teacher_sq[0]     + tg_sum_teacher_sq[1];
        out.sum_student_sq      = tg_sum_student_sq[0]     + tg_sum_student_sq[1];
        out.max_abs_error       = max(tg_max_abs_error[0],   tg_max_abs_error[1]);
        out.element_count       = tg_element_count[0]       + tg_element_count[1];
        out._pad                = 0u;
    }
}
