// ── Fused O Projection + Residual ───────────────────────────────────────────
//
// Fuses O (output projection) ternary page-640 GEMV with residual addition.
// One threadgroup (64 threads) per output element.
//
//   out[gid] = Σ(W[gid][col] * attn_out[col]) + residual[gid]
//
// where W is a ternary-page640 packed weight matrix (20 base-3 trits/u32,
// 32 u32/page = 640 weights per page), with per-page and per-weight-position
// half scales, plus optional contiguous sidecar override spans.
//
// The residual input carries the pre-attention activation, so the fused
// output is the complete post-attention-plus-residual result.
//
// 3 personalities via mode_flags:
//   deployment (no MODE_DIAGNOSTIC or MODE_FUSED_SCORING): just compute + output
//   diagnostic (MODE_DIAGNOSTIC): write AttentionProbe records for sampled dims
//   fused_scoring (MODE_FUSED_SCORING): write ErrorPartial for pre/post-residual
//
// ABI:
//   0 = O packed pages          PackedTernaryPage640*   — [out_dim * nt]
//   1 = attention output        half*                   — [in_dim]
//   2 = residual input          half*                   — [out_dim]
//   3 = page scales             half*                   — [out_dim * nt]
//   4 = channel scales          half*                   — [out_dim * nt * PAGE_WIDTH]
//   5 = sidecar entries         half*                   — interleaved PageSidecarHeader + values
//   6 = sidecar offsets         uint*                   — [out_dim], byte base per row
//   7 = output activations      half*                   — [out_dim]
//   8 = params                  ProjectionParams (constant)
//   9 = receipt                 KernelReceipt*          — [1] (optional)
//  10 = probe output            AttentionProbe*         — [out_dim] (diagnostic mode)
//  11 = error partials          ErrorPartial*           — [1] (fused_scoring mode)
//
// Function constant:
//   PAGE_WIDTH (index 0) — default 640

#include <metal_stdlib>
using namespace metal;

// ── Mode flag constants (matching kernel_types.rs ProjectionParams) ─────────
constant uint MODE_SIDECAR       = 1u;    // bit 0
constant uint MODE_RECEIPT       = 2u;    // bit 1
constant uint MODE_DIAGNOSTIC    = 8u;    // bit 3
constant uint MODE_FUSED_SCORING = 16u;   // bit 4

// ── Function constants ──────────────────────────────────────────────────────
constant uint PAGE_WIDTH_FC [[function_constant(0)]];

// ── Data structures (repr(C) matching Rust kernel_types) ────────────────────

struct PageHeader {
    uint   scale_index;
    uint   sidecar_start;
    uint   sidecar_end;
    ushort valid_tail_length;
    ushort flags;
};

struct PackedTernaryPage640 {
    uint       payload[40];   // 32 used (20 trits/u32 * 32 = 640), 8 reserved
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

// ── Kernel ──────────────────────────────────────────────────────────────────

kernel void fused_o_proj_residual(
    device const PackedTernaryPage640* packed         [[buffer(0)]],
    device const half*                 attention      [[buffer(1)]],
    device const half*                 residual       [[buffer(2)]],
    device const half*                 page_scales    [[buffer(3)]],
    device const half*                 channel_scales [[buffer(4)]],
    device const half*                 sidecar        [[buffer(5)]],
    device const uint*                 sidecar_offsets[[buffer(6)]],
    device half*                       output         [[buffer(7)]],
    constant ProjectionParams&         params         [[buffer(8)]],
    device KernelReceipt*              receipt        [[buffer(9)]],
    device AttentionProbe*             probe          [[buffer(10)]],
    device ErrorPartial*               error_partials [[buffer(11)]],
    uint gid        [[threadgroup_position_in_grid]],
    uint tid        [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_id    [[simdgroup_index_in_threadgroup]])
{
    if (gid >= params.out_dim) return;

    const uint  flags       = params.mode_flags;
    const uint  in_dim      = params.in_dim;
    const uint  out_dim     = params.out_dim;
    const uint  page_width  = PAGE_WIDTH_FC > 0 ? PAGE_WIDTH_FC : params.page_width;
    const uint  nt          = (in_dim + page_width - 1) / page_width;
    const uint  words_per_page = 32u;
    const uint  weights_per_page = page_width;

    device const PackedTernaryPage640* row_pages = packed + (gid * nt);

    // ── Sidecar instrumentation counters ─────────────────────────────────
    threadgroup uint tg_sc_hits[64];
    threadgroup uint tg_sc_reads[64];
    uint local_hits  = 0;
    uint local_reads = 0;

    float acc = 0.0f;

    // ── Phase 1: packed ternary word processing ─────────────────────────────
    for (uint p = 0; p < nt; ++p) {
        const float page_scale_f = float(page_scales[gid * nt + p]);

        for (uint wi = tid; wi < words_per_page; wi += 64) {
            uint rem = row_pages[p].payload[wi];
            const uint col0 = p * page_width + wi * 20;

            device const half* cs_base = channel_scales
                + (gid * nt + p) * weights_per_page
                + wi * 20;

            for (uint vi = 0; vi < 20; ++vi) {
                const uint d = rem % 3u;
                rem /= 3u;
                const uint col = col0 + vi;
                if (col >= in_dim) break;
                if (d != 0u) {
                    const float cs = float(cs_base[vi]);
                    const float tv = (d == 1u) ? page_scale_f : -page_scale_f;
                    acc = fma(float(attention[col]), tv * cs, acc);
                }
            }
        }
    }

    // ── Phase 2: sidecar override processing ────────────────────────────────
    if ((flags & MODE_SIDECAR) && nt > 0) {
        const size_t row_sidecar_base = sidecar_offsets[gid];

        for (uint p = tid; p < nt; p += 64) {
            device const PackedTernaryPage640* page_ptr = row_pages + p;
            if ((page_ptr->header.flags & 1u) == 0u) continue;

            local_hits++;
            device const char* sidecar_bytes =
                reinterpret_cast<device const char*>(sidecar);
            device const PageSidecarHeader* sc_hdr =
                reinterpret_cast<device const PageSidecarHeader*>(
                    sidecar_bytes + row_sidecar_base + page_ptr->header.sidecar_start);

            const uint sc_count = sc_hdr->count;
            const uint sc_start = sc_hdr->start_index;
            const float sc_scale = sc_hdr->residual_scale;
            device const half* sc_vals =
                reinterpret_cast<device const half*>(sc_hdr + 1);

            for (uint ei = 0; ei < sc_count; ++ei) {
                const uint pos = sc_start + ei;
                const uint col = p * page_width + pos;
                if (col >= in_dim) break;
                acc = fma(float(attention[col]), float(sc_vals[ei]) * sc_scale, acc);
            }
            local_reads += sc_count;
        }
    }

    // Publish sidecar counters.
    tg_sc_hits[tid]  = local_hits;
    tg_sc_reads[tid] = local_reads;

    // ── Reduction ───────────────────────────────────────────────────────────
    acc = simd_sum(acc);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float shared_reduction[2];
    if (simd_lane == 0) {
        shared_reduction[simd_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total = shared_reduction[0] + shared_reduction[1];
        float pre_residual = total;

        // ── Add residual ───────────────────────────────────────────────────
        total += float(residual[gid]);

        // ── Write output ───────────────────────────────────────────────────
        output[gid] = half(total);

        // ── Personality: diagnostic (MODE_DIAGNOSTIC) ──────────────────────
        // Write AttentionProbe for deterministically sampled output dims.
        if ((flags & MODE_DIAGNOSTIC) && probe) {
            uint hash = gid * 2654435761u;
            hash ^= params.probe_seed;
            if (hash % 32u == 0u) {
                device AttentionProbe& out = probe[gid];
                out.head_id            = gid;
                out.token_index        = 0;
                out.teacher_max_logit  = pre_residual;
                out.student_max_logit  = total;
                out.teacher_entropy    = 0.0f;
                out.student_entropy    = 0.0f;
                out.sampled_probability_l1 = 0.0f;
                out.sampled_probability_kl = 0.0f;
            }
        }

        // ── Personality: fused_scoring (MODE_FUSED_SCORING) ────────────────
        // Write ErrorPartial with pre-residual vs post-residual comparison.
        if ((flags & MODE_FUSED_SCORING) && error_partials) {
            const float drift = total - pre_residual;
            error_partials[gid].sum_sq_error         = drift * drift;
            error_partials[gid].sum_abs_error        = fabs(drift);
            error_partials[gid].dot_teacher_student  = pre_residual * total;
            error_partials[gid].sum_teacher_sq       = pre_residual * pre_residual;
            error_partials[gid].sum_student_sq       = total * total;
            error_partials[gid].max_abs_error        = fabs(drift);
            error_partials[gid].element_count        = 1;
            error_partials[gid]._pad                 = 0;
        }

        // ── Instrumentation (MODE_RECEIPT) ─────────────────────────────────
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
            receipt->threadgroups           = out_dim;
            receipt->threads_per_threadgroup = 64;
            receipt->output_elements        = out_dim;
            receipt->flags                  = flags;
            receipt->_pad_receipt           = 0;
            receipt->logical_weight_bytes   =
                ulong(nt) * ulong(sizeof(PackedTernaryPage640));
            receipt->logical_sidecar_bytes  = ulong(tg_reads) * ulong(sizeof(half));
            receipt->logical_activation_bytes = ulong(out_dim) * ulong(sizeof(half));
        }
    }
}
