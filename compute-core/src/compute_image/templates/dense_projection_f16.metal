// [[kernel]] dense_projection_f16 — standard dense fp16 GEMV with fused epilogue.
// One threadgroup (64 threads) per output row.
//
// buffer(0): weights          [out_dim * in_dim] half (row-major)
// buffer(1): input            [in_dim] half
// buffer(2): bias_or_gain     [out_dim or in_dim] half (optional — bias or rmsnorm gain)
// buffer(3): output           [out_dim] half
// buffer(4): ProjectionParams (constant)
// buffer(5): KernelReceipt    (device, instrumented path only)
//
// Fused epilogue (mode_flags bits):
//   1 = bias add
//   2 = rmsnorm (bias_or_gain buffer used as gain weights [out_dim])
//   4 = residual add (input[row % in_dim])
//   8 = SiLU activation
//  16 = instrumentation enabled (writes KernelReceipt)
//
// Conventions from existing templates: half I/O, fp32 accumulation + reduction,
// simd_sum + threadgroup scratchpad, threadgroup_per_output_row dispatch.

#include <metal_stdlib>
using namespace metal;

// ── Function constants ──────────────────────────────────────────────────────

constant float EPSILON [[function_constant(0)]];

// ── ABI structs (matching kernel_types.rs #[repr(C)]) ───────────────────────

struct ProjectionParams {
    uint32_t in_dim;
    uint32_t out_dim;
    uint32_t page_count;       // unused in dense kernel, kept for ABI compat
    uint32_t page_width;       // unused in dense kernel, kept for ABI compat
    uint32_t mode_flags;       // bit0=bias, bit1=rmsnorm, bit2=residual, bit3=silu, bit4=instrument
    uint32_t probe_seed;       // unused in dense kernel
    uint32_t reserved[5];      // pad to 16-byte alignment
};

struct KernelReceipt {
    uint32_t kernel_id;
    uint32_t phase_id;
    uint32_t page_count;
    uint32_t sidecar_hits;
    uint32_t sidecar_entries_read;
    uint32_t threadgroups;
    uint32_t threads_per_threadgroup;
    uint32_t output_elements;
    uint32_t flags;
    uint64_t logical_weight_bytes;
    uint64_t logical_sidecar_bytes;
    uint64_t logical_activation_bytes;
};

// ── Fused SiLU: x * sigmoid(x) ─────────────────────────────────────────────

static float silu_activation(float x) {
    return x / (1.0f + exp(-x));
}

// ── Kernel entry point ──────────────────────────────────────────────────────

kernel void dense_projection_f16(
    device const half*   weights       [[buffer(0)]],
    device const half*   input         [[buffer(1)]],
    device const half*   bias_or_gain  [[buffer(2)]],
    device half*         output        [[buffer(3)]],
    constant ProjectionParams&  params [[buffer(4)]],
    device KernelReceipt* receipt      [[buffer(5)]],
    uint32_t row                       [[threadgroup_position_in_grid]],
    uint32_t tid                       [[thread_position_in_threadgroup]],
    uint32_t simd_lane                 [[thread_index_in_simdgroup]],
    uint32_t simd_id                   [[simdgroup_index_in_threadgroup]])
{
    uint32_t in_dim  = params.in_dim;
    uint32_t out_dim = params.out_dim;
    uint32_t flags   = params.mode_flags;

    if (row >= out_dim) return;

    // ── Accumulate dot product in fp32 ──────────────────────────────────────
    float acc = 0.0f;
    uint32_t weights_offset = row * in_dim;

    for (uint32_t i = tid; i < in_dim; i += 64) {
        acc = fma(float(input[i]), float(weights[weights_offset + i]), acc);
    }

    // ── fp32 SIMD-group reduction ──────────────────────────────────────────
    acc = simd_sum(acc);

    // ── Inter-SIMD reduction via threadgroup scratchpad ────────────────────
    threadgroup float shared_reduction[2]; // 2 SIMD groups (64/32)
    if (simd_lane == 0) {
        shared_reduction[simd_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Fused epilogue (thread 0) ──────────────────────────────────────────
    if (tid == 0) {
        float result = shared_reduction[0] + shared_reduction[1];

        // bias add (mode bit 0)
        if (flags & 1) {
            result += float(bias_or_gain[row]);
        }

        // rmsnorm (mode bit 1): result = result / RMS * gain
        // NOTE: applied per-element (the only thing possible in one threadgroup).
        // RMS = sqrt(result² + ε), so this acts as a soft sign × gain:
        //   result / sqrt(result² + ε) ≈ sign(result) when |result| >> ε.
        // A true vector RMS-norm across the output requires a second kernel pass.
        if (flags & 2) {
            float rms = sqrt(result * result + EPSILON);
            float gain = float(bias_or_gain[row]);
            result = (result / rms) * gain;
        }

        // residual add (mode bit 2): add input[row] (maps to first or strided element)
        // NOTE: assumes in_dim == out_dim (typical for residual connections).
        // When row >= in_dim, we clamp to row % in_dim.
        if (flags & 4) {
            uint32_t residual_idx = row;
            if (residual_idx >= in_dim) {
                residual_idx = residual_idx % in_dim;
            }
            result += float(input[residual_idx]);
        }

        // SiLU activation (mode bit 3): x * sigmoid(x)
        if (flags & 8) {
            result = silu_activation(result);
        }

        output[row] = half(result);

        // Instrumentation (mode bit 4): write counters
        // Guard by row == 0 — only one writer; all threadgroups report the
        // same logical sizes, so the last writer is fine.
        if ((flags & 16) && row == 0) {
            receipt->kernel_id            = 0;           // set by host dispatcher
            receipt->phase_id             = 0;           // set by host dispatcher
            receipt->page_count           = 1;           // dense — no page structure
            receipt->sidecar_hits         = 0;
            receipt->sidecar_entries_read = 0;
            receipt->threadgroups         = out_dim;
            receipt->threads_per_threadgroup = 64;
            receipt->output_elements      = 1;
            receipt->flags                = flags;
            receipt->logical_weight_bytes = (uint64_t)in_dim * (uint64_t)out_dim * 2ULL;
            receipt->logical_sidecar_bytes   = 0;
            receipt->logical_activation_bytes = (uint64_t)in_dim * 2ULL;
        }
    }
}
