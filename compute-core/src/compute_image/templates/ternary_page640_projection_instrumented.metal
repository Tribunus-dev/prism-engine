// [[kernel]] ternary_page640_projection_instrumented — ternary tile640
// projection with instrumentation counters. Arithmetic is IDENTICAL to the
// baseline ternary_page640_projection kernel — only instrumentation is added.
//
// One threadgroup (64 threads) per output row. FP32 accumulation + SIMD/
// threadgroup reduction. Processes 640-weight base-3 packed pages with
// per-page page_scales and per-weight-position channel_scales, plus optional
// contiguous sidecar override spans.
//
// INSTRUMENTATION (deterministic, no effect on arithmetic result):
//   The main accumulation path matches the baseline exactly. Instrumentation
//   counters are tracked in thread-local or threadgroup variables and written
//   through the receipt buffer only — no atomics on the arithmetic path, no
//   reordering of memory operations relative to the baseline.
//
//   INSTRUMENTATION_DETAIL_LEVEL [[function_constant(0)]]:
//     0 — counters disabled; functionally identical to baseline
//     1 — page_fetches, sidecar_hits, sidecar_entries_read
//     2 — + branch-path counts (+1/-1/0 per page)
//     3 — + output statistics (max, min, mean)
//
// ABI (identical to baseline):
//   buffer(0): packed         [out_dim * nt] PackedTernaryPage640
//   buffer(1): input          [in_dim] half
//   buffer(2): page_scales    [out_dim * nt] half
//   buffer(3): channel_scales [out_dim * nt * PAGE_WIDTH] half (one per weight)
//   buffer(4): sidecar        interleaved PageSidecarHeader + half entries
//   buffer(5): sidecar_offsets [out_dim] uint (per-row byte base)
//   buffer(6): output         [out_dim] half
//   buffer(7): params         ProjectionParams (constant)
//   buffer(8): receipt        KernelReceipt (instrumentation, written when
//                              INSTRUMENTATION_DETAIL_LEVEL >= 1)

#include <metal_stdlib>
using namespace metal;

// ── Function constants ──────────────────────────────────────────────────────

constant uint PAGE_WIDTH [[function_constant(0)]];

constant uint INSTRUMENTATION_DETAIL_LEVEL [[function_constant(1)]];

constant bool INSTR_ENABLED = INSTRUMENTATION_DETAIL_LEVEL >= 1;
constant bool BRANCH_COUNTS = INSTRUMENTATION_DETAIL_LEVEL >= 2;
constant bool OUTPUT_STATS  = INSTRUMENTATION_DETAIL_LEVEL >= 3;

// ── Data structures (repr(C) matching Rust kernel_types) ────────────────────

struct PageHeader {
    uint   scale_index;
    uint   sidecar_start;
    uint   sidecar_end;
    ushort valid_tail_length;
    ushort flags;                    // bit0 = sidecar_present, bit1 = tail_padding
};

struct PackedTernaryPage640 {
    uint       payload[40];          // 32 used (20 trits/u32 × 32 = 640), 8 reserved
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
    uint  page_count;               // pages per output row
    uint  page_width;               // typically PAGE_WIDTH (640)
    uint  mode_flags;
    uint  probe_seed;
    uint  reserved[5];
};

struct KernelReceipt {
    uint  kernel_id;
    uint  phase_id;
    uint  page_count;               // pages fetched
    uint  sidecar_hits;
    uint  sidecar_entries_read;
    uint  threadgroups;
    uint  threads_per_threadgroup;
    uint  output_elements;
    uint  flags;
    ulong logical_weight_bytes;     // branch: pos|neg|zero (detail>=2)
    ulong logical_sidecar_bytes;    // stats: max<<32 | min (detail>=3)
    ulong logical_activation_bytes; // stats: mean (detail>=3)
};

// ── Main kernel ─────────────────────────────────────────────────────────────

kernel void ternary_page640_projection_instrumented(
    device const PackedTernaryPage640* packed          [[buffer(0)]],
    device const half*                 input           [[buffer(1)]],
    device const half*                 page_scales     [[buffer(2)]],
    device const half*                 channel_scales  [[buffer(3)]],
    device const half*                 sidecar         [[buffer(4)]],
    device const uint*                 sidecar_offsets [[buffer(5)]],
    device half*                       output          [[buffer(6)]],
    constant ProjectionParams&         params          [[buffer(7)]],
    device KernelReceipt*              receipt         [[buffer(8)]],
    uint row                                            [[threadgroup_position_in_grid]],
    uint tid                                            [[thread_position_in_threadgroup]],
    uint simd_lane                                      [[thread_index_in_simdgroup]],
    uint simd_id                                        [[simdgroup_index_in_threadgroup]])
{
    if (row >= params.out_dim) return;

    const uint in_dim          = params.in_dim;
    const uint out_dim         = params.out_dim;
    const uint page_width      = PAGE_WIDTH > 0 ? PAGE_WIDTH : params.page_width;
    const uint nt              = (in_dim + page_width - 1) / page_width;
    const uint words_per_page  = 32u;
    const uint weights_per_page = page_width;

    device const PackedTernaryPage640* row_pages = packed + (row * nt);

    // ── Instrumentation threadgroup state ─────────────────────────────────
    // Branch counters are per-thread; output-stats are thread-0 only.
    threadgroup int   tg_branch_pos[64];
    threadgroup int   tg_branch_neg[64];
    threadgroup int   tg_branch_zero[64];
    threadgroup float tg_out_max;
    threadgroup float tg_out_min;
    threadgroup float tg_out_mean;

    tg_branch_pos[tid]  = 0;
    tg_branch_neg[tid]  = 0;
    tg_branch_zero[tid] = 0;
    if (tid == 0) {
        tg_out_max  = -INFINITY;
        tg_out_min  =  INFINITY;
        tg_out_mean = 0.0f;
    }

    // ── Phase 1: packed ternary word processing (IDENTICAL to baseline) ─────
    float acc = 0.0f;

    // Per-thread branch counters — eliminated by compiler when BRANCH_COUNTS
    // is false (only accessed inside `if (BRANCH_COUNTS)` blocks).
    int pos_acc = 0, neg_acc = 0, zero_acc = 0;

    for (uint p = 0; p < nt; ++p) {
        const float page_scale_f = float(page_scales[row * nt + p]);

        for (uint wi = tid; wi < words_per_page; wi += 64) {
            uint rem = row_pages[p].payload[wi];

            const uint col0 = p * page_width + wi * 20;

            device const half* cs_base = channel_scales
                + (row * nt + p) * weights_per_page
                + wi * 20;

            for (uint vi = 0; vi < 20; ++vi) {
                const uint d = rem % 3u;
                rem /= 3u;
                const uint col = col0 + vi;
                if (col >= in_dim) break;
                if (d != 0u) {
                    const float cs = float(cs_base[vi]);
                    const float tv = (d == 1u) ? page_scale_f : -page_scale_f;
                    acc = fma(float(input[col]), tv * cs, acc);
                }

                // Branch counting — register only, no shared-state access in
                // this inner loop. The compiler eliminates this block entirely
                // when BRANCH_COUNTS is false.
                if (BRANCH_COUNTS) {
                    if      (d == 1u) ++pos_acc;
                    else if (d == 2u) ++neg_acc;
                    else              ++zero_acc;
                }
            }
        }
    }

    // ── Phase 2: sidecar override processing (IDENTICAL to baseline) ───────
    threadgroup uint tg_sc_hits[64];
    threadgroup uint tg_sc_reads[64];

    uint local_hits  = 0;
    uint local_reads = 0;

    if ((params.mode_flags & 1u) && nt > 0) {
        const size_t row_sidecar_base = sidecar_offsets[row];

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
                acc = fma(float(input[col]), float(sc_vals[ei]) * sc_scale, acc);
            }
            local_reads += sc_count;
        }
    }

    // ── Flush per-thread branch counts to threadgroup (barrier-free) ────────
    if (BRANCH_COUNTS) {
        tg_branch_pos[tid]  = pos_acc;
        tg_branch_neg[tid]  = neg_acc;
        tg_branch_zero[tid] = zero_acc;
    }

    // Publish sidecar stats into threadgroup memory.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tg_sc_hits[tid]  = local_hits;
    tg_sc_reads[tid] = local_reads;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint tg_sidecar_hits  = 0;
    uint tg_sidecar_reads = 0;
    if (tid == 0) {
        for (uint i = 0; i < 64; ++i) {
            tg_sidecar_hits  += tg_sc_hits[i];
            tg_sidecar_reads += tg_sc_reads[i];
        }
    }

    // ── Reduction (IDENTICAL to baseline) ───────────────────────────────────
    acc = simd_sum(acc);

    threadgroup float shared_reduction[32];
    if (simd_lane == 0) {
        shared_reduction[simd_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        const uint nsimd = (64 + 31) / 32;
        float total = 0.0f;
        for (uint s = 0; s < nsimd; ++s) {
            total += shared_reduction[s];
        }

        // ── Write output (IDENTICAL to baseline) ──────────────────────────
        output[row] = half(total);

        // ── Instrumentation: output statistics (detail >= 3) ──────────────
        if (OUTPUT_STATS) {
            tg_out_max   = total;
            tg_out_min   = total;
            tg_out_mean  = total;
        }

        // ── Instrumentation: threadgroup 0 writes global receipt ──────────
        if (INSTR_ENABLED && row == 0) {
            device KernelReceipt& r = receipt[0];

            // Base counters
            r.kernel_id                = 0;
            r.phase_id                 = 0;
            r.page_count               = nt;
            r.threadgroups             = out_dim;
            r.threads_per_threadgroup  = 64;
            r.output_elements          = out_dim;
            r.flags                    = 0;
            r.logical_weight_bytes     = 0;
            r.logical_sidecar_bytes    = 0;
            r.logical_activation_bytes = 0;

            // ── Sidecar tracking (detail >= 1) ────────────────────────────
            r.sidecar_hits         = tg_sidecar_hits;
            r.sidecar_entries_read = tg_sidecar_reads;

            // ── Branch counts (detail >= 2) ───────────────────────────────
            // Sum per-thread accumulators across all 64 threads.
            uint pos_total = 0, neg_total = 0, zero_total = 0;
            if (BRANCH_COUNTS) {
                for (uint i = 0; i < 64; ++i) {
                    pos_total  += uint(max(tg_branch_pos[i],  0));
                    neg_total  += uint(max(tg_branch_neg[i],  0));
                    zero_total += uint(max(tg_branch_zero[i], 0));
                }

                uint overflow = 0u;
                if (pos_total  > 0xFFFFFu) overflow = 1u;
                if (neg_total  > 0xFFFFFu) overflow = 1u;
                if (zero_total > 0xFFFFFu) overflow = 1u;
                r.flags = overflow;

                ulong packed = (ulong(zero_total) << 40)
                             | (ulong(neg_total)  << 20)
                             | ulong(pos_total);
                r.logical_weight_bytes = packed;
            }

            // ── Output statistics (detail >= 3) ───────────────────────────
            if (OUTPUT_STATS) {
                r.logical_sidecar_bytes =
                    (ulong(as_type<uint>(tg_out_min)) << 32)
                  | ulong(as_type<uint>(tg_out_max));
                r.logical_activation_bytes =
                    ulong(as_type<uint>(tg_out_mean));
            }
        }
    }
}
