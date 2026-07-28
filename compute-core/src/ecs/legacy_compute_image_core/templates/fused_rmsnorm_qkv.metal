// ── Fused RMSNorm + Q/K/V Ternary Projection ──────────────────────────────
//
// One threadgroup (64 threads) per token.
// Fuses three operations into one GPU pass:
//   1. Load input into threadgroup memory
//   2. Compute RMSNorm (sum x² → sqrt(mean + ε) → reciprocal)
//   3. Apply gain (learned RMSNorm scale vector)
//   4. For each of Q, K, V: read normalized input from registers, decode
//      tile640 packed ternary weights with per-weight-position channel scales,
//      accumulate into output values, apply sidecar overrides
//   5. Write Q/K/V to activation arena using ActivationView descriptors
//   6. Optional personality outputs: diagnostic (AttentionProbe probes) or
//      fused_scoring (compact ErrorPartial partials)
//   7. Optional instrumentation via KernelReceipt
//
// Each thread independently computes a contiguous block of output positions
// for each projection (no cross-thread reduction per output position).
//
// Buffer binding:
//   0 = input           half[]                   — [tokens × hidden_dim]
//   1 = q_pages         PackedTernaryPage640[]    — Q weight pages
//   2 = k_pages         PackedTernaryPage640[]    — K weight pages
//   3 = v_pages         PackedTernaryPage640[]    — V weight pages
//   4 = page_scales     half[]                   — per-page scale values
//   5 = channel_scales  half[]                   — per-weight-position scales [total_pages × PAGE_WIDTH]
//   6 = gain            half[]                   — RMSNorm gain vector [hidden_dim]
//   7 = sidecar         half[]                   — sidecar entries (interleaved headers + values)
//   8 = sidecar_offsets uint[]                   — per-page sidecar byte base
//   9 = arena           half[]                   — activation arena output
//  10 = arena_views     ActivationView[3]        — descriptors for Q, K, V slots
//  11 = params          ProjectionParams (constant)
//  12 = receipt         KernelReceipt            — instrumentation output (optional)
//  13 = error_partials  ErrorPartial[]           — fused_scoring output (optional)
//  14 = probe_output    AttentionProbe[]         — diagnostic output (optional)
//
// Dispatch: threadgroups = token count, threads_per_threadgroup = 64
//
// Function constants:
//   PAGE_WIDTH (index 0) — default 640
//   EPSILON    (index 1) — RMSNorm epsilon, default 1e-5
//   Q_DIM      (index 2) — Q output dimension override
//   K_DIM      (index 3) — K output dimension override
//   V_DIM      (index 4) — V output dimension override
//
// Mode bits (ProjectionParams.mode_flags):
//   0x01 = MODE_SIDECAR       — enable sidecar overrides
//   0x02 = MODE_RECEIPT       — write KernelReceipt instrumentation
//   0x08 = MODE_DIAGNOSTIC    — write detailed AttentionProbe probes
//   0x10 = MODE_FUSED_SCORING — write compact ErrorPartial partials

#include <metal_stdlib>
using namespace metal;

// ── Mode flag constants (matching kernel_types.rs ProjectionParams) ─────────
constant uint MODE_SIDECAR       = 1u;    // bit 0
constant uint MODE_RECEIPT       = 2u;    // bit 1
constant uint MODE_DIAGNOSTIC    = 8u;    // bit 3
constant uint MODE_FUSED_SCORING = 16u;   // bit 4

// ── Function constants ────────────────────────────────────────────────────
constant uint  PAGE_WIDTH_FC [[function_constant(0)]];
constant float EPSILON_FC    [[function_constant(1)]];
constant uint  Q_FC          [[function_constant(2)]];
constant uint  K_FC          [[function_constant(3)]];
constant uint  V_FC          [[function_constant(4)]];

// Max input dimension for threadgroup buffer (must be ≥ hidden_dim).
constant uint TG_HD_MAX = 4096;

// ── Data structures (repr(C) matching Rust kernel_types.rs) ──────────────

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

struct ActivationView {
    ulong  byte_offset;
    uint   row_stride;
    uint   col_stride;
    uint   dtype;
    uint   layout;
    uint   token_count;
    uint   hidden_dim;
};

struct ProjectionParams {
    uint  in_dim;
    uint  out_dim;
    uint  page_count;
    uint  page_width;
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

struct AttentionProbe {
    uint  head_id;
    uint  token_index;
    float teacher_max_logit;
    float student_max_logit;
    float teacher_entropy;
    float student_entropy;
    float sampled_probability_l1;
    float sampled_probability_kl;
};

// ── Helper: ternary tile640 row-dot product with sidecar support ──────────
//
// Returns the dot product for one output position.
// Applies per-weight-position channel_scales and optional sidecar overrides.

METAL_FUNC float ternary_row_dot(
    device const PackedTernaryPage640* pages,
    uint                               row_pages,
    uint                               nt,
    threadgroup const half*            norm_input,
    device const half*                 page_scales,
    device const half*                 channel_scales,
    device const half*                 sidecar,
    device const uint*                 sidecar_offsets,
    uint                               hd,
    uint                               page_width,
    uint                               mode_flags,
    threadgroup uint*                  tg_sc_hits,
    threadgroup uint*                  tg_sc_reads,
    uint                               tid)
{
    float acc = 0.0f;

    // ── Phase 1: packed ternary word processing ──────────────────────────
    for (uint p = 0; p < nt; ++p) {
        device const PackedTernaryPage640& page = pages[row_pages + p];
        const float page_scale_f = float(page_scales[page.header.scale_index]);

        for (uint wi = 0; wi < 32; ++wi) {
            uint rem = page.payload[wi];
            const uint col0 = p * page_width + wi * 20;

            device const half* cs_base = channel_scales
                + (row_pages + p) * page_width
                + wi * 20;

            for (uint vi = 0; vi < 20; ++vi) {
                const uint d = rem % 3u;
                rem /= 3u;
                const uint col = col0 + vi;
                if (col >= hd) break;
                if (d != 0u) {
                    const float cs = float(cs_base[vi]);
                    const float tv = (d == 1u) ? page_scale_f : -page_scale_f;
                    acc = fma(float(norm_input[col]), tv * cs, acc);
                }
            }
        }
    }

    // ── Phase 2: sidecar override processing ─────────────────────────────
    if (mode_flags & MODE_SIDECAR) {
        for (uint p = tid; p < nt; p += 64) {
            device const PackedTernaryPage640& page = pages[row_pages + p];
            if ((page.header.flags & 1u) == 0u) continue;

            const uint row_sidecar_base = sidecar_offsets[row_pages + p];
            device const char* sc_bytes =
                reinterpret_cast<device const char*>(sidecar);
            device const PageSidecarHeader* sc_hdr =
                reinterpret_cast<device const PageSidecarHeader*>(
                    sc_bytes + row_sidecar_base + page.header.sidecar_start);

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

// ── Main fused kernel ─────────────────────────────────────────────────────

kernel void fused_rmsnorm_qkv(
    device const half*                input          [[buffer(0)]],
    device const PackedTernaryPage640* q_pages       [[buffer(1)]],
    device const PackedTernaryPage640* k_pages       [[buffer(2)]],
    device const PackedTernaryPage640* v_pages       [[buffer(3)]],
    device const half*                page_scales    [[buffer(4)]],
    device const half*                channel_scales [[buffer(5)]],
    device const half*                gain           [[buffer(6)]],
    device const half*                sidecar        [[buffer(7)]],
    device const uint*                sidecar_offsets[[buffer(8)]],
    device half*                      arena          [[buffer(9)]],
    device const ActivationView*      arena_views    [[buffer(10)]],
    constant ProjectionParams&        params         [[buffer(11)]],
    device KernelReceipt*             receipt        [[buffer(12)]],
    device ErrorPartial*              error_partials [[buffer(13)]],
    device AttentionProbe*            probe_output   [[buffer(14)]],
    uint gid                                          [[threadgroup_position_in_grid]],
    uint tid                                          [[thread_position_in_threadgroup]],
    uint simd_lane                                    [[thread_index_in_simdgroup]],
    uint simd_id                                      [[simdgroup_index_in_threadgroup]])
{
    const uint flags        = params.mode_flags;
    const uint hd           = params.in_dim;
    const uint page_width   = PAGE_WIDTH_FC > 0 ? PAGE_WIDTH_FC : params.page_width;
    const uint nt           = (hd + page_width - 1) / page_width;
    const uint q_hd         = Q_FC > 0 ? Q_FC : 4096;
    const uint k_hd         = K_FC > 0 ? K_FC : q_hd;
    const uint v_hd         = V_FC > 0 ? V_FC : k_hd;
    const uint nq           = (q_hd + 63) / 64;
    const uint nk           = (k_hd + 63) / 64;
    const uint nv           = (v_hd + 63) / 64;

    if (gid >= arena_views[0].token_count) return;

    // ── Sidecar instrumentation counters ─────────────────────────────────
    threadgroup uint tg_sc_hits[64];
    threadgroup uint tg_sc_reads[64];
    tg_sc_hits[tid]  = 0;
    tg_sc_reads[tid] = 0;

    // ── 1. Load input into threadgroup memory ────────────────────────────
    threadgroup half input_tg[TG_HD_MAX];
    device const half* token_input = input + gid * hd;
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
        const float total_sumsq = shared_sumsq[0] + shared_sumsq[1];
        const float epsilon = EPSILON_FC > 0.0f ? EPSILON_FC : 1e-5f;
        const float rms = sqrt(total_sumsq / float(hd) + epsilon);
        shared_rms_recip = 1.0f / rms;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float rms_recip = shared_rms_recip;

    // ── 3. Apply RMSNorm gain to threadgroup buffer ──────────────────────
    // norm[i] = input[i] * rms_recip * gain[i]
    for (uint i = tid; i < hd; i += 64) {
        input_tg[i] = half(float(input_tg[i]) * rms_recip * float(gain[i]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── 4. Q projection ──────────────────────────────────────────────────
    for (uint i = 0; i < nq; ++i) {
        const uint out_idx = tid + i * 64;
        if (out_idx >= q_hd) break;

        const float dot = ternary_row_dot(
            q_pages, out_idx * nt, nt, input_tg,
            page_scales, channel_scales, sidecar, sidecar_offsets,
            hd, page_width, flags,
            tg_sc_hits, tg_sc_reads, tid);

        device half* q_base = arena + arena_views[0].byte_offset / sizeof(half)
                                     + gid * arena_views[0].row_stride;
        q_base[out_idx] = half(dot);
    }

    // ── 5. K projection ──────────────────────────────────────────────────
    for (uint i = 0; i < nk; ++i) {
        const uint out_idx = tid + i * 64;
        if (out_idx >= k_hd) break;

        const float dot = ternary_row_dot(
            k_pages, out_idx * nt, nt, input_tg,
            page_scales, channel_scales, sidecar, sidecar_offsets,
            hd, page_width, flags,
            tg_sc_hits, tg_sc_reads, tid);

        device half* k_base = arena + arena_views[1].byte_offset / sizeof(half)
                                     + gid * arena_views[1].row_stride;
        k_base[out_idx] = half(dot);
    }

    // ── 6. V projection ──────────────────────────────────────────────────
    for (uint i = 0; i < nv; ++i) {
        const uint out_idx = tid + i * 64;
        if (out_idx >= v_hd) break;

        const float dot = ternary_row_dot(
            v_pages, out_idx * nt, nt, input_tg,
            page_scales, channel_scales, sidecar, sidecar_offsets,
            hd, page_width, flags,
            tg_sc_hits, tg_sc_reads, tid);

        device half* v_base = arena + arena_views[2].byte_offset / sizeof(half)
                                     + gid * arena_views[2].row_stride;
        v_base[out_idx] = half(dot);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── 7. Personality: diagnostic output (MODE_DIAGNOSTIC) ──────────────
    // Writes AttentionProbe records for sampled token positions.
    // Sampling: (gid * 2654435761u) ^ probe_seed selects ~1/32.
    if ((flags & MODE_DIAGNOSTIC) && probe_output) {
        uint hash = gid * 2654435761u;
        hash ^= params.probe_seed;
        if (hash % 32u == 0u) {
            // Write one probe per projection slot (3 slots per sampled token).
            for (uint s = 0; s < 3; ++s) {
                const uint slot_dim = (s == 0) ? q_hd : ((s == 1) ? k_hd : v_hd);
                device half* slot_base = arena + arena_views[s].byte_offset / sizeof(half)
                                                 + gid * arena_views[s].row_stride;

                // Compute max logit across slot dimensions.
                float max_logit = -INFINITY;
                for (uint i = tid; i < slot_dim; i += 64) {
                    max_logit = max(max_logit, float(slot_base[i]));
                }
                float g_max = simd_max(max_logit);

                threadgroup float tg_max[2];
                if (simd_lane == 0) { tg_max[simd_id] = g_max; }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid == 0) {
                    const float final_max = max(tg_max[0], tg_max[1]);
                    device AttentionProbe& out = probe_output[gid * 3 + s];
                    out.head_id            = s;
                    out.token_index        = gid;
                    out.teacher_max_logit  = final_max;
                    out.student_max_logit  = 0.0f;
                    out.teacher_entropy    = 0.0f;
                    out.student_entropy    = 0.0f;
                    out.sampled_probability_l1 = 0.0f;
                    out.sampled_probability_kl = 0.0f;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }

    // ── 8. Personality: fused_scoring output (MODE_FUSED_SCORING) ───────
    // Writes ErrorPartial for each projection slot, comparing normalized
    // input (teacher) vs projection output (student) for the first token.
    if ((flags & MODE_FUSED_SCORING) && error_partials) {
        if (gid == 0) {
            const uint slot_dims[3] = {q_hd, k_hd, v_hd};
            for (uint s = 0; s < 3; ++s) {
                device half* slot_base = arena + arena_views[s].byte_offset / sizeof(half)
                                                 + gid * arena_views[s].row_stride;

                // Compare dims = min(hd, slot_dim)
                const uint cd = min(hd, slot_dims[s]);
                float sse = 0.0f, sae = 0.0f, tsq = 0.0f, max_abs = 0.0f;
                for (uint i = tid; i < cd; i += 64) {
                    float teacher = float(input_tg[i]);
                    float student = float(slot_base[i]);
                    float drift = student - teacher;
                    float abs_d = fabs(drift);
                    sse    += drift * drift;
                    sae    += abs_d;
                    tsq    += teacher * teacher;
                    max_abs = max(max_abs, abs_d);
                }

                // SIMD-group reduction — lane 0 writes to shared, tid==0 reads+sums.
                float r_sse = simd_sum(sse);
                float r_sae = simd_sum(sae);
                float r_tsq = simd_sum(tsq);
                float r_max = simd_max(max_abs);

                threadgroup float sh_sse[2], sh_sae[2], sh_tsq[2], sh_max[2];
                if (simd_lane == 0) {
                    sh_sse[simd_id] = r_sse;
                    sh_sae[simd_id] = r_sae;
                    sh_tsq[simd_id] = r_tsq;
                    sh_max[simd_id] = r_max;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid == 0) {
                    error_partials[s].sum_sq_error        = sh_sse[0] + sh_sse[1];
                    error_partials[s].sum_abs_error       = sh_sae[0] + sh_sae[1];
                    error_partials[s].dot_teacher_student = 0.0f;
                    error_partials[s].sum_teacher_sq      = sh_tsq[0] + sh_tsq[1];
                    error_partials[s].sum_student_sq      = 0.0f;
                    error_partials[s].max_abs_error       = max(sh_max[0], sh_max[1]);
                    error_partials[s].element_count       = cd;
                    error_partials[s]._pad                = 0;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }

    // ── 9. Instrumentation (MODE_RECEIPT) ─────────────────────────────────
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint tg_total_hits  = 0;
    uint tg_total_reads = 0;
    for (uint i = 0; i < 64; ++i) {
        tg_total_hits  += tg_sc_hits[i];
        tg_total_reads += tg_sc_reads[i];
    }

    if ((flags & MODE_RECEIPT) && gid == 0 && tid == 0) {
        const uint total_q_pages = q_hd * nt;
        const uint total_k_pages = k_hd * nt;
        const uint total_v_pages = v_hd * nt;
        const uint total_pages   = total_q_pages + total_k_pages + total_v_pages;

        receipt->kernel_id               = 0;
        receipt->phase_id                = 0;
        receipt->page_count              = total_pages;
        receipt->sidecar_hits            = tg_total_hits;
        receipt->sidecar_entries_read    = tg_total_reads;
        receipt->threadgroups            = arena_views[0].token_count;
        receipt->threads_per_threadgroup = 64;
        receipt->output_elements         = q_hd + k_hd + v_hd;
        receipt->flags                   = flags;
        receipt->_pad_receipt            = 0;
        receipt->logical_weight_bytes    = ulong(total_pages) * ulong(sizeof(PackedTernaryPage640));
        receipt->logical_sidecar_bytes   = ulong(tg_total_reads) * ulong(sizeof(half));
        receipt->logical_activation_bytes = ulong(q_hd + k_hd + v_hd) * ulong(sizeof(half));
    }
}
