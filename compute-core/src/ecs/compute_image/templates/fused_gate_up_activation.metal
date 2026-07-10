// ── Fused RMSNorm + Gate/Up Ternary Projections + Activation ───────────────
//
// Fuses four operations into one GPU pass for the MLP gate/up path:
//   1. Load input into threadgroup memory
//   2. Compute RMSNorm (sum x² → sqrt(mean + ε) → reciprocal)
//   3. Gate projection: W_gate @ norm_input  (ternary page640 GEMV)
//   4. Up projection:   W_up   @ norm_input  (ternary page640 GEMV)
//   5. Activation: gate * SiLU(up) (SwiGLU) or GeGLU(configurable)
//   6. Write activation output to intermediate_dim buffer
//   7. Optional personality outputs
//
// One threadgroup (64 threads) processes one output position (one row of
// the intermediate dimension). Each threadgroup loads the full normalized
// input into threadgroup memory (shared across gate and up dot products).
//
// 3 personalities via mode_flags:
//   deployment: just compute activation output
//   diagnostic: write detailed probe records (histogram + stats)
//   fused_scoring: write compact ErrorPartial partials
//
// Buffer binding:
//   0 = input           half[]                   — input activations [in_dim]
//   1 = gate_pages      PackedTernaryPage640[]    — gate weight pages [intermediate * nt]
//   2 = up_pages        PackedTernaryPage640[]    — up weight pages [intermediate * nt]
//   3 = page_scales     half[]                    — per-page scale [2 * intermediate * nt]
//   4 = channel_scales  half[]                    — per-weight-position scales
//   5 = gain            half[]                    — RMSNorm gain [in_dim]
//   6 = output          half[]                    — activation output [intermediate]
//   7 = probe_records   void*                     — diagnostic or fused_scoring output
//   8 = params          ProjectionParams (constant)
//       → in_dim  = hidden_dim (input dimension)
//       → out_dim = intermediate_dim (gate/up output dimension)
//       → page_width = MODE_ACTIVATION bit pattern (bit0=SiLU/bit1=GeGLU)
//   9 = receipt         KernelReceipt             — instrumentation (optional)
//  10 = sidecar         half[]                    — sidecar entries (optional)
//  11 = sidecar_offsets uint[]                    — per-page sidecar byte base
//
// Function constants:
//   PAGE_WIDTH (index 0) — page width, default 640
//   EPSILON    (index 1) — RMSNorm epsilon, default 1e-5
//
// Activation style is selected via params.page_width bit 0:
//   0 = gate * SiLU(up)   (SwiGLU — default)
//   1 = SiLU(gate) * up   (alternative)
//   2+ reserved

#include <metal_stdlib>
using namespace metal;

// ── Mode flag constants ─────────────────────────────────────────────────────
constant uint MODE_SIDECAR       = 1u;    // bit 0
constant uint MODE_RECEIPT       = 2u;    // bit 1
constant uint MODE_DIAGNOSTIC    = 8u;    // bit 3 — detailed probe records
constant uint MODE_FUSED_SCORING = 16u;   // bit 4 — compact ErrorPartial

// ── Function constants ──────────────────────────────────────────────────────
constant uint  PAGE_WIDTH_FC [[function_constant(0)]];
constant float EPSILON_FC    [[function_constant(1)]];

// Max hidden dimension for threadgroup buffer.
constant uint TG_HD_MAX = 4096;

// ── Data structures (repr(C) matching Rust kernel_types.rs) ────────────────

struct PageHeader {
    uint   scale_index;
    uint   sidecar_start;
    uint   sidecar_end;
    ushort valid_tail_length;
    ushort flags;
};

struct PackedTernaryPage640 {
    uint       payload[40];
    PageHeader header;
};

struct PageSidecarHeader {
    uint   start_index;
    ushort count;
    ushort encoding;
    float  residual_scale;
    uint   flags;
};

struct ProjectionParams {
    uint  in_dim;        // hidden_dim
    uint  out_dim;       // intermediate_dim
    uint  page_count;    // total pages across gate+up
    uint  page_width;    // activation style (bit0=SiLU, bit1=GeGLU)
    uint  mode_flags;
    uint  probe_seed;
    uint  reserved[5];
};

struct KernelReceipt {
    uint     kernel_id;
    uint     phase_id;
    uint     page_count;
    uint     sidecar_hits;
    uint     sidecar_entries_read;
    uint     threadgroups;
    uint     threads_per_threadgroup;
    uint     output_elements;
    uint     flags;
    uint     _pad_receipt;
    uint64_t logical_weight_bytes;
    uint64_t logical_sidecar_bytes;
    uint64_t logical_activation_bytes;
};

struct ErrorPartial {
    float sum_sq_error;
    float sum_abs_error;
    float dot_teacher_student;
    float sum_teacher_sq;
    float sum_student_sq;
    float max_abs_error;
    uint  element_count;
    uint  _pad;
};

// ── Activation helpers ──────────────────────────────────────────────────────

static float sigmoid_f32(float x) {
    return 1.0f / (1.0f + exp(-x));
}

static float silu_f32(float x) {
    return x * sigmoid_f32(x);
}

static float gelu_f32(float x) {
    // Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanh(sqrt_2_over_pi * (x + coeff * x3)));
}

// ── Helper: ternary row dot product for fused gate/up ──────────────────────

METAL_FUNC float tern_row_dot_fused(
    device const PackedTernaryPage640* pages,
    uint                               row_pages,
    uint                               nt,
    threadgroup const half*            norm_input,
    device const half*                 page_scales,
    device const half*                 channel_scales,
    device const half*                 sidecar,
    device const uint*                 sidecar_offsets,
    uint                               gid,
    uint                               hd,
    uint                               page_width,
    uint                               flags,
    threadgroup uint*                  tg_sc_hits,
    threadgroup uint*                  tg_sc_reads,
    uint                               tid)
{
    float acc = 0.0f;

    for (uint p = 0; p < nt; ++p) {
        device const PackedTernaryPage640& page = pages[row_pages + p];
        const float page_scale_f = float(page_scales[page.header.scale_index]);

        for (uint wi = tid; wi < 32; wi += 64) {
            uint rem = page.payload[wi];
            const uint col0 = p * page_width + wi * 20;

            for (uint vi = 0; vi < 20; ++vi) {
                const uint d = rem % 3u;
                rem /= 3u;
                const uint col = col0 + vi;
                if (col >= hd) break;
                if (d != 0u) {
                    const float cs = float(channel_scales[(row_pages + p) * page_width + wi * 20 + vi]);
                    const float tv = (d == 1u) ? page_scale_f : -page_scale_f;
                    acc = fma(float(norm_input[col]), tv * cs, acc);
                }
            }
        }
    }

    // Sidecar processing
    if (flags & MODE_SIDECAR) {
        for (uint p = tid; p < nt; p += 64) {
            device const PackedTernaryPage640& page = pages[row_pages + p];
            if ((page.header.flags & 1u) == 0u) continue;

            const uint row_sc_base = sidecar_offsets[row_pages + p];
            device const char* sc_bytes =
                reinterpret_cast<device const char*>(sidecar);
            device const PageSidecarHeader* sc_hdr =
                reinterpret_cast<device const PageSidecarHeader*>(
                    sc_bytes + row_sc_base + page.header.sidecar_start);

            const uint sc_count = sc_hdr->count;
            const uint sc_start = sc_hdr->start_index;
            const float sc_scale = sc_hdr->residual_scale;
            device const half* sc_vals =
                reinterpret_cast<device const half*>(sc_hdr + 1);

            tg_sc_hits[tid] += 1;
            for (uint ei = tid; ei < sc_count; ei += 64) {
                const uint pos = sc_start + ei;
                const uint col = p * page_width + pos;
                if (col >= hd) break;
                acc = fma(float(norm_input[col]), float(sc_vals[ei]) * sc_scale, acc);
            }
            tg_sc_reads[tid] += sc_count;
        }
    }

    return acc;
}

// ── Kernel entry point ─────────────────────────────────────────────────────

kernel void fused_gate_up_activation(
    device const half*                 input         [[buffer(0)]],
    device const PackedTernaryPage640*  gate_pages   [[buffer(1)]],
    device const PackedTernaryPage640*  up_pages     [[buffer(2)]],
    device const half*                 page_scales   [[buffer(3)]],
    device const half*                 channel_scales[[buffer(4)]],
    device const half*                 gain          [[buffer(5)]],
    device half*                       output        [[buffer(6)]],
    device void*                       probe_records [[buffer(7)]],
    constant ProjectionParams&         params        [[buffer(8)]],
    device KernelReceipt*              receipt       [[buffer(9)]],
    device const half*                 sidecar       [[buffer(10)]],
    device const uint*                 sidecar_offsets[[buffer(11)]],
    uint gid                                         [[threadgroup_position_in_grid]],
    uint tid                                         [[thread_position_in_threadgroup]],
    uint simd_lane                                   [[thread_index_in_simdgroup]],
    uint simd_id                                     [[simdgroup_index_in_threadgroup]])
{
    const uint flags          = params.mode_flags;
    const uint hd             = params.in_dim;
    const uint intermediate   = params.out_dim;
    const uint page_width     = PAGE_WIDTH_FC > 0 ? PAGE_WIDTH_FC : params.page_width;
    const uint nt             = (hd + page_width - 1) / page_width;
    const uint act_style      = params.page_width & 1u;  // bit0 from page_width field

    // Activation style: 0 = gate*SiLU(up) (SwiGLU), 1 = SiLU(gate)*up
    const bool use_swiglu = (act_style == 0u);

    if (gid >= intermediate) return;

    // ── Sidecar instrumentation counters ─────────────────────────────────
    threadgroup uint tg_sc_hits[64];
    threadgroup uint tg_sc_reads[64];
    tg_sc_hits[tid]  = 0;
    tg_sc_reads[tid] = 0;

    // ── 1. Load input into threadgroup memory ────────────────────────────
    threadgroup half input_tg[TG_HD_MAX];
    device const half* token_input = input;  // single token (broadcast mode)

    for (uint i = tid; i < hd; i += 64) {
        input_tg[i] = token_input[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── 2. Compute RMSNorm ───────────────────────────────────────────────
    float local_sumsq = 0.0f;
    for (uint i = tid; i < hd; i += 64) {
        const float x = float(input_tg[i]);
        local_sumsq += x * x;
    }
    float group_sumsq = simd_sum(local_sumsq);

    threadgroup float shared_sumsq[2];
    if (simd_lane == 0) { shared_sumsq[simd_id] = group_sumsq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float shared_rms_recip = 0.0f;
    if (tid == 0) {
        const float epsilon = EPSILON_FC > 0.0f ? EPSILON_FC : 1e-5f;
        const float rms = sqrt((shared_sumsq[0] + shared_sumsq[1]) / float(hd) + epsilon);
        shared_rms_recip = 1.0f / rms;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float rms_recip = shared_rms_recip;

    // ── 3. Apply RMSNorm (with gain) ─────────────────────────────────────
    for (uint i = tid; i < hd; i += 64) {
        input_tg[i] = half(float(input_tg[i]) * rms_recip * float(gain[i]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── 4. Gate projection ───────────────────────────────────────────────
    const float gate_val = tern_row_dot_fused(
        gate_pages, gid * nt, nt, input_tg,
        page_scales, channel_scales, sidecar, sidecar_offsets,
        gid, hd, page_width, flags,
        tg_sc_hits, tg_sc_reads, tid);

    // ── 5. Up projection ─────────────────────────────────────────────────
    const float up_val = tern_row_dot_fused(
        up_pages, gid * nt, nt, input_tg,
        page_scales + intermediate * nt,    // up page scales follow gate
        channel_scales + intermediate * nt * page_width,  // up channel scales
        sidecar, sidecar_offsets,
        gid, hd, page_width, flags,
        tg_sc_hits, tg_sc_reads, tid);

    // ── 6. Apply activation ──────────────────────────────────────────────
    // Reductions: simd_sum the gate and up dot products across threads.
    float g = simd_sum(gate_val);
    float u = simd_sum(up_val);

    threadgroup float tg_g[2];
    threadgroup float tg_u[2];
    if (simd_lane == 0) {
        tg_g[simd_id] = g;
        tg_u[simd_id] = u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        g = tg_g[0] + tg_g[1];
        u = tg_u[0] + tg_u[1];

        // ── Activation ───────────────────────────────────────────────────
        float activation;
        if (use_swiglu) {
            activation = u * silu_f32(g);     // gate * SiLU(up) — SwiGLU
        } else {
            activation = g * silu_f32(u);     // SiLU(gate) * up
        }

        // ── Write output ─────────────────────────────────────────────────
        output[gid] = half(activation);

        // ── Personality: diagnostic (MODE_DIAGNOSTIC) ────────────────────
        // Write raw gate/up values for probe analysis.
        if ((flags & MODE_DIAGNOSTIC) && probe_records) {
            device float* probe = (device float*)probe_records + gid * 4;
            probe[0] = g;            // gate value
            probe[1] = u;            // up value
            probe[2] = activation;   // activation output
            probe[3] = 0.0f;         // reserved
        }

        // ── Personality: fused_scoring (MODE_FUSED_SCORING) ──────────────
        // Write ErrorPartial with gate/up statistics for this row.
        if ((flags & MODE_FUSED_SCORING) && probe_records) {
            device ErrorPartial* ep = (device ErrorPartial*)probe_records + gid;
            ep->sum_sq_error       = 0.0f;  // not applicable
            ep->sum_abs_error      = fabs(g) + fabs(u);
            ep->dot_teacher_student = g * u;
            ep->sum_teacher_sq     = g * g;
            ep->sum_student_sq     = u * u;
            ep->max_abs_error      = max(fabs(g), fabs(u));
            ep->element_count      = 1;
            ep->_pad               = 0;
        }

        // ── Instrumentation (MODE_RECEIPT) ───────────────────────────────
        if ((flags & MODE_RECEIPT) && gid == 0) {
            uint tg_hits  = 0;
            uint tg_reads = 0;
            for (uint i = 0; i < 64; ++i) {
                tg_hits  += tg_sc_hits[i];
                tg_reads += tg_sc_reads[i];
            }
            receipt->kernel_id              = 0;
            receipt->phase_id               = 0;
            receipt->page_count             = params.page_count;
            receipt->sidecar_hits           = tg_hits;
            receipt->sidecar_entries_read   = tg_reads;
            receipt->threadgroups           = intermediate;
            receipt->threads_per_threadgroup = 64;
            receipt->output_elements        = intermediate;
            receipt->flags                  = flags;
            receipt->_pad_receipt           = 0;
            receipt->logical_weight_bytes   = ulong(nt * 2) * ulong(sizeof(PackedTernaryPage640));
            receipt->logical_sidecar_bytes  = ulong(tg_reads) * ulong(sizeof(half));
            receipt->logical_activation_bytes = ulong(intermediate) * ulong(sizeof(half));
        }
    }
}
