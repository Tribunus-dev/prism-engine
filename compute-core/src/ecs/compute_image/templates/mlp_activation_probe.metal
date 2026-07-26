// ── MLP Activation Probe ────────────────────────────────────────────────────
// Records gate/up activation statistics for sampled MLP positions.
// Writes ErrorPartial records (32 B each, matching kernel_types.rs #[repr(C)]).
//
// Each threadgroup processes one sampled row of the intermediate dimension:
//   1. Load gate[i] and up[i] activations
//   2. Compute SiLU(gate) = gate * sigmoid(gate)
//   3. Accumulate per-field ErrorPartial: activation variance, correlation,
//      RMS² proxies, peak activation, element count
//   4. simd_sum all accumulators and write one ErrorPartial per sample
//
// Buffer layout:
//   [0] gate_activations  [num_samples = intermediate_dim]  half
//   [1] up_activations    [num_samples = intermediate_dim]  half
//   [2] down_output       [num_samples = hidden_dim]        half  (reference, unused)
//   [3] probe_records     [num_samples]                     ErrorPartial[]
//   [4] params            ProjectionParams
//
// Threadgroup dispatch: num_samples threadgroups = 64 threads.
//   in_dim    = intermediate dimension (gate/up length)
//   out_dim   = hidden dimension (down-projection output length)
//   page_count = num_samples
//   page_width = weight_factor bit pattern (reinterpreted as float)
//
// ErrorPartial field mapping:
//   sum_sq_error         = Σ SiLU(gate)²            — activation variance proxy
//   sum_abs_error        = Σ |SiLU(gate)|            — L1 activation magnitude
//   dot_teacher_student  = Σ gate * up               — gate-up correlation
//   sum_teacher_sq       = Σ gate²                   — gate RMS² proxy
//   sum_student_sq       = Σ up²                     — up RMS² proxy
//   max_abs_error        = max |SiLU(gate)|          — peak activation
//   element_count        = intermediate_dim
//   _pad                 = 0

#include <metal_stdlib>
using namespace metal;

// ── ABI structs ──────────────────────────────────────────────────────────────
// Binary layout matches compute_image::compile::kernel_types exactly.

struct ProjectionParams {
    uint  in_dim;          // intermediate dimension (gate/up activation length)
    uint  out_dim;         // hidden dimension (down-projection output length)
    uint  page_count;      // num_samples
    uint  page_width;      // weight_factor for down projection (float bit pattern)
    uint  mode_flags;      // mode bits: bit0 = record_stats (write ErrorPartial)
    uint  probe_seed;      // seed for deterministic probe sampling
    uint  reserved[5];     // future use
};

// ErrorPartial — fixed-order partial reduction record (matches Rust repr(C) exactly).
struct ErrorPartial {
    float sum_sq_error;          // Σ(student - teacher)²
    float sum_abs_error;         // Σ|student - teacher|
    float dot_teacher_student;   // Σ teacher * student
    float sum_teacher_sq;        // Σ teacher²
    float sum_student_sq;        // Σ student²
    float max_abs_error;         // max|student - teacher|
    uint  element_count;         // number of elements compared
    uint  _pad;                  // 8-byte struct tail alignment
};

// ── Helper: sigmoid via fp32, branch-free ───────────────────────────────────
static float sigmoid_f32(float x) {
    return 1.0f / (1.0f + exp(-x));
}

// ── Kernel ──────────────────────────────────────────────────────────────────

kernel void mlp_activation_probe(
    device const half*              gate_activations [[buffer(0)]],
    device const half*              up_activations   [[buffer(1)]],
    device const half*              down_output      [[buffer(2)]],
    device void*                    probe_records    [[buffer(3)]],
    constant ProjectionParams&      params           [[buffer(4)]],
    uint                            gid              [[threadgroup_position_in_grid]],
    uint                            tid              [[thread_position_in_threadgroup]],
    uint                            simd_lane        [[thread_index_in_simdgroup]],
    uint                            simd_id          [[simdgroup_index_in_threadgroup]])
{
    if (gid >= params.page_count) return;

    const uint  intermediate   = params.in_dim;
    const float weight_factor  = as_type<float>(params.page_width);
    const uint  row_offset     = gid * intermediate;

    device const half* gate_row = gate_activations + row_offset;
    device const half* up_row   = up_activations   + row_offset;
    (void)down_output;  // reserved for future error-comparison mode

    // ── Per-thread accumulators ─────────────────────────────────────────────
    float local_sq_error    = 0.0f;   // Σ SiLU(gate)²
    float local_abs_error   = 0.0f;   // Σ |SiLU(gate)|
    float local_dot         = 0.0f;   // Σ gate * up
    float local_gate_sq     = 0.0f;   // Σ gate²
    float local_up_sq       = 0.0f;   // Σ up²
    float local_max_abs     = 0.0f;   // max |SiLU(gate)|
    uint  local_count       = 0u;

    // ── Strided element loop ────────────────────────────────────────────────
    for (uint i = tid; i < intermediate; i += 64) {
        float gate_val = float(gate_row[i]);
        float up_val   = float(up_row[i]);

        // SiLU activation
        float silu = gate_val * sigmoid_f32(gate_val);
        float abs_silu = fabs(silu);

        // Accumulate ErrorPartial fields
        local_sq_error  += silu * silu;
        local_abs_error += abs_silu;
        local_dot       += gate_val * up_val;
        local_gate_sq   += gate_val * gate_val;
        local_up_sq     += up_val * up_val;
        local_max_abs    = fmax(local_max_abs, abs_silu);
        local_count++;
    }

    // ── SIMD-level reductions ───────────────────────────────────────────────
    float sum_sq_error   = simd_sum(local_sq_error);
    float sum_abs_error  = simd_sum(local_abs_error);
    float dot            = simd_sum(local_dot);
    float gate_sq        = simd_sum(local_gate_sq);
    float up_sq          = simd_sum(local_up_sq);
    float max_abs        = simd_max(local_max_abs);
    uint  cnt            = simd_sum(local_count);

    // Inter-SIMD reduction via threadgroup memory (2 SIMD groups of 32)
    threadgroup float tg_sq_err[2];
    threadgroup float tg_abs_err[2];
    threadgroup float tg_dot[2];
    threadgroup float tg_gate_sq[2];
    threadgroup float tg_up_sq[2];
    threadgroup float tg_max_abs[2];
    threadgroup uint  tg_count[2];

    if (simd_lane == 0) {
        tg_sq_err[simd_id]   = sum_sq_error;
        tg_abs_err[simd_id]  = sum_abs_error;
        tg_dot[simd_id]      = dot;
        tg_gate_sq[simd_id]  = gate_sq;
        tg_up_sq[simd_id]    = up_sq;
        tg_max_abs[simd_id]  = max_abs;
        tg_count[simd_id]    = cnt;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Thread 0: finalize and write ErrorPartial ───────────────────────────
    if (tid == 0) {
        device ErrorPartial& out =
            ((device ErrorPartial*)probe_records)[gid];

        out.sum_sq_error         = tg_sq_err[0] + tg_sq_err[1];
        out.sum_abs_error        = tg_abs_err[0] + tg_abs_err[1];
        out.dot_teacher_student  = tg_dot[0] + tg_dot[1];
        out.sum_teacher_sq       = tg_gate_sq[0] + tg_gate_sq[1];
        out.sum_student_sq       = tg_up_sq[0] + tg_up_sq[1];
        out.max_abs_error        = fmax(tg_max_abs[0], tg_max_abs[1]);
        out.element_count        = tg_count[0] + tg_count[1];
        out._pad                 = 0u;
    }
}
