// ── RMSNorm Residual Probe ───────────────────────────────────────────────────
// Compares pre-norm (teacher) activations against post-norm * gain (student)
// for every element of every token. Writes one ErrorPartial record per token.
//
// The inputs capture the residual stream state before and after the RMSNorm
// linear transform:
//   pre_norm   = input to RMSNorm (residual stream activation)
//   post_norm  = output of RMSNorm (normalized) — before gain
//   gain       = learnable scale vector (per-hidden-dim, shared across tokens)
//
// Each threadgroup processes one token (row). 64 threads, fp32 accumulation.
//
// Buffer layout:
//   [0] pre_norm         [num_tokens * hidden_dim] half
//   [1] post_norm        [num_tokens * hidden_dim] half
//   [2] residual         [num_tokens * hidden_dim] half  (reserved/future use)
//   [3] gain             [hidden_dim]             half
//   [4] output           [num_tokens]             ErrorPartial
//   [5] params           ProjectionParams
//       → in_dim  = hidden_dim
//       → out_dim = num_tokens
//
// Function constants:
//   FCONST_EPSILON (index 0)  — small epsilon for RMS stability (default 1e-5)
//   FCONST_HIDDEN_DIM (index 1) — hidden dimension override (default 4096)

#include <metal_stdlib>
using namespace metal;

// ── Struct definitions (mirrors kernel_types.rs #[repr(C)]) ──────────────────

struct ProjectionParams {
    uint32_t in_dim;              // hidden dimension
    uint32_t out_dim;             // token count (number of rows)
    uint32_t page_count;          // unused
    uint32_t page_width;          // unused
    uint32_t mode_flags;          // unused
    uint32_t probe_seed;          // unused
    uint32_t reserved[5];         // pad to 16-byte alignment
};

struct ErrorPartial {
    float sum_sq_error;           // Σ(student - teacher)²
    float sum_abs_error;          // Σ|student - teacher|
    float dot_teacher_student;    // Σ teacher * student
    float sum_teacher_sq;         // Σ teacher² (pre-norm RMS sum)
    float sum_student_sq;         // Σ student² (post-norm RMS sum)
    float max_abs_error;          // max|student - teacher|
    uint32_t element_count;       // number of elements compared
    uint32_t _pad;                // 16-byte struct alignment
};

// ── Function constants ───────────────────────────────────────────────────────

constant float EPSILON [[function_constant(0)]];
constant uint HIDDEN_DIM [[function_constant(1)]];

// ── Kernel ───────────────────────────────────────────────────────────────────

kernel void rmsnorm_residual_probe(
    device const half*          pre_norm    [[buffer(0)]],
    device const half*          post_norm   [[buffer(1)]],
    device const half*          residual    [[buffer(2)]],
    device const half*          gain        [[buffer(3)]],
    device ErrorPartial*        output      [[buffer(4)]],
    constant ProjectionParams&  params      [[buffer(5)]],
    uint32_t gid                             [[threadgroup_position_in_grid]],
    uint32_t tid                             [[thread_position_in_threadgroup]],
    uint32_t simd_lane                       [[thread_index_in_simdgroup]],
    uint32_t simd_id                         [[simdgroup_index_in_threadgroup]])
{
    uint32_t hidden_dim = HIDDEN_DIM;
    if (params.in_dim != 0) { hidden_dim = params.in_dim; }
    uint32_t num_tokens = params.out_dim;

    if (gid >= num_tokens) return;

    // Row base pointers — gain is shared across all tokens
    device const half* row_pre  = pre_norm  + gid * hidden_dim;
    device const half* row_post = post_norm + gid * hidden_dim;
    device const half* row_gain = gain;

    // ── Per-thread accumulators ──────────────────────────────────────────────
    float local_sum_sq     = 0.0f;
    float local_sum_abs    = 0.0f;
    float local_dot        = 0.0f;
    float local_teacher_sq = 0.0f;
    float local_student_sq = 0.0f;
    float local_max_abs    = 0.0f;

    // ── Strided loop over hidden_dim ─────────────────────────────────────────
    for (uint32_t i = tid; i < hidden_dim; i += 64) {
        float teacher = float(row_pre[i]);
        float student = float(row_post[i]) * float(row_gain[i]);
        float drift   = student - teacher;
        float abs_drift = fabs(drift);

        local_sum_sq     += drift * drift;
        local_sum_abs    += abs_drift;
        local_dot        += teacher * student;
        local_teacher_sq += teacher * teacher;
        local_student_sq += student * student;
        local_max_abs     = fmax(local_max_abs, abs_drift);
    }

    // ── SIMD-group reduction ─────────────────────────────────────────────────
    float sum_sq     = simd_sum(local_sum_sq);
    float sum_abs    = simd_sum(local_sum_abs);
    float dot        = simd_sum(local_dot);
    float teacher_sq = simd_sum(local_teacher_sq);
    float student_sq = simd_sum(local_student_sq);
    float max_abs    = simd_max(local_max_abs);

    // ── Threadgroup reduction (2 simdgroups × 32 lanes) ──────────────────────
    threadgroup float shared_sum_sq[2];
    threadgroup float shared_sum_abs[2];
    threadgroup float shared_dot[2];
    threadgroup float shared_teacher_sq[2];
    threadgroup float shared_student_sq[2];
    threadgroup float shared_max_abs[2];

    if (simd_lane == 0) {
        shared_sum_sq[simd_id]      = sum_sq;
        shared_sum_abs[simd_id]     = sum_abs;
        shared_dot[simd_id]         = dot;
        shared_teacher_sq[simd_id]  = teacher_sq;
        shared_student_sq[simd_id]  = student_sq;
        shared_max_abs[simd_id]     = max_abs;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        output[gid].sum_sq_error         = shared_sum_sq[0] + shared_sum_sq[1];
        output[gid].sum_abs_error        = shared_sum_abs[0] + shared_sum_abs[1];
        output[gid].dot_teacher_student  = shared_dot[0] + shared_dot[1];
        output[gid].sum_teacher_sq       = shared_teacher_sq[0] + shared_teacher_sq[1];
        output[gid].sum_student_sq       = shared_student_sq[0] + shared_student_sq[1];
        output[gid].max_abs_error        = fmax(shared_max_abs[0], shared_max_abs[1]);
        output[gid].element_count        = hidden_dim;
        output[gid]._pad                 = 0;
    }
}
