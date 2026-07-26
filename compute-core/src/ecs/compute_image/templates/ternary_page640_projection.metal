// [[kernel]] ternary_page640_projection — ternary tile640 projection with
// sidecar overrides. One threadgroup (64 threads) per output row.
//
// FP32 accumulation + SIMD/threadgroup reduction. Processes 640-weight base-3
// packed pages with per-page and per-weight-position channel scales, plus
// optional contiguous sidecar override spans.
//
// Three personalities via mode_flags:
//   deployment  (bits 0=0, 1=0): bare compute, no sidecar, no instrumentation
//   diagnostic  (bit 1=1):       full sidecar + instrumentation + probe writes
//   fused-scoring (bit 0=1):     sidecar enabled, no instrumentation overhead
//
// ABI:
//   buffer(0): packed         [out_dim * nt] PackedTernaryPage640
//   buffer(1): input          [in_dim] half
//   buffer(2): page_scales    [out_dim * nt] half
//   buffer(3): channel_scales [out_dim * nt * PAGE_WIDTH] half  (one per weight pos)
//   buffer(4): sidecar        interleaved PageSidecarHeader + half entries
//   buffer(5): sidecar_offsets [out_dim] uint  (per-row byte base into sidecar buf)
//   buffer(6): output         [out_dim] half
//   buffer(7): params         ProjectionParams (constant)
//   buffer(8): receipt        KernelReceipt (instrumentation, written when mode_flags & 2)
//
// Page width is a function constant (PAGE_WIDTH, default 640).
//
// NOTE: channel_scales layout correction from spec pseudocode.
//   The spec wrote: channel_scales[row * words_per_row + wi*20 + vi]
//   where `wi` is page-local (0..31), making all pages share the same scale
//   region.  The correct indexing tiles per page:
//     channel_scales[(row * nt + p) * PAGE_WIDTH + wi * 20 + vi]

#include <metal_stdlib>
using namespace metal;

// ── Function constants ──────────────────────────────────────────────────────
constant uint PAGE_WIDTH [[function_constant(0)]];

// ── Data structures (repr(C) matching Rust kernel_types) ────────────────────

/// Page header: element-wise metadata for one packed page.
struct PageHeader {
    uint   scale_index;       // index into page_scales buffer
    uint   sidecar_start;     // byte offset to PageSidecarHeader in sidecar buffer
    uint   sidecar_end;       // byte offset (exclusive) end of sidecar span
    ushort valid_tail_length; // number of valid weight positions (<= PAGE_WIDTH)
    ushort flags;             // bit0 = sidecar_present, bit1 = tail_padding
};

/// A single packed ternary page (640 weights in base-3 tile640 encoding).
struct PackedTernaryPage640 {
    uint       payload[40];   // 32 used (20 trits/u32 = 32 = 640), 8 reserved
    PageHeader header;
};

/// Per-sidecar-span header: describes one contiguous override span within a page.
struct PageSidecarHeader {
    uint   start_index;    // first weight position overridden (0..PAGE_WIDTH-1)
    ushort count;          // number of consecutive overridden positions
    ushort encoding;       // entry storage format (0 = half values)
    float  residual_scale; // scale for sidecar entries (Rust f32 — matches repr(C))
    uint   flags;          // reserved
};

/// Per-dispatch projection parameters.
struct ProjectionParams {
    uint  in_dim;
    uint  out_dim;
    uint  page_count;    // total pages for this projection
    uint  page_width;    // typically PAGE_WIDTH (640)
    uint  mode_flags;    // bit0 = sidecar_enabled, bit1 = instrumentation_enabled
    uint  probe_seed;
    uint  reserved[5];
};

/// Instrumentation counters written by the kernel.
struct KernelReceipt {
    uint  kernel_id;
    uint  phase_id;
    uint  page_count;
    uint  sidecar_hits;
    uint  sidecar_entries_read;
    uint  threadgroups;
    uint  threads_per_threadgroup;
    uint  output_elements;
    uint  flags;
    ulong logical_weight_bytes;
    ulong logical_sidecar_bytes;
    ulong logical_activation_bytes;
};

// ── Kernel ──────────────────────────────────────────────────────────────────

kernel void ternary_page640_projection(
    device const PackedTernaryPage640* packed         [[buffer(0)]],
    device const half*                  input          [[buffer(1)]],
    device const half*                  page_scales    [[buffer(2)]],
    device const half*                  channel_scales [[buffer(3)]],
    device const half*                  sidecar        [[buffer(4)]],
    device const uint*                  sidecar_offsets[[buffer(5)]],
    device half*                        output         [[buffer(6)]],
    constant ProjectionParams&          params         [[buffer(7)]],
    device KernelReceipt*               receipt        [[buffer(8)]],
    uint row                                            [[threadgroup_position_in_grid]],
    uint tid                                            [[thread_position_in_threadgroup]],
    uint simd_lane                                      [[thread_index_in_simdgroup]],
    uint simd_id                                        [[simdgroup_index_in_threadgroup]])
{
    if (row >= params.out_dim) return;

    const uint in_dim       = params.in_dim;
    const uint out_dim      = params.out_dim;
    const uint page_width   = PAGE_WIDTH > 0 ? PAGE_WIDTH : params.page_width;
    const uint nt           = (in_dim + page_width - 1) / page_width;
    const uint words_per_page = 32u;     // 32 u32 words = 20 trits = 640
    const uint weights_per_page = page_width;

    // Base page pointer for this row.
    device const PackedTernaryPage640* row_pages = packed + (row * nt);

    float acc = 0.0f;

    // ── Phase 1: packed ternary word processing ─────────────────────────────
    // Outer page loop, inner word-lane loop at stride 64.
    for (uint p = 0; p < nt; ++p) {
        const float page_scale_f = float(page_scales[row * nt + p]);

        for (uint wi = tid; wi < words_per_page; wi += 64) {
            uint rem = row_pages[p].payload[wi];

            // Base column offset for this word lane within the page.
            const uint col0 = p * page_width + wi * 20;

            // Per-weight-position channel-scale base for this page.
            device const half* cs_base = channel_scales
                + (row * nt + p) * weights_per_page
                + wi * 20;

            // Unpack 20 base-3 trits (LSB first).
            for (uint vi = 0; vi < 20; ++vi) {
                const uint d = rem % 3u;    // 0=skip, 1=+scale, 2=-scale
                rem /= 3u;
                const uint col = col0 + vi;
                if (col >= in_dim) break;
                if (d != 0u) {
                    const float cs = float(cs_base[vi]);
                    const float tv = (d == 1u) ? page_scale_f : -page_scale_f;
                    acc = fma(float(input[col]), tv * cs, acc);
                }
            }
        }
    }

    // ── Phase 2: sidecar override processing ────────────────────────────────
    // Distribute pages across threads.  Accumulate local hit/read counters for
    // instrumentation.  Only threads with tid < nt participate.
    threadgroup uint tg_sc_hits[64];
    threadgroup uint tg_sc_reads[64];

    uint local_hits  = 0;
    uint local_reads = 0;

    if ((params.mode_flags & 1u) && nt > 0) {
        const size_t row_sidecar_base = sidecar_offsets[row];

        for (uint p = tid; p < nt; p += 64) {
            device const PackedTernaryPage640* page_ptr = row_pages + p;
            if ((page_ptr->header.flags & 1u) == 0u) continue;  // sidecar_present

            local_hits++;

            // sidecar_start is relative to the per-row base in sidecar_offsets.
            device const char* sidecar_bytes =
                reinterpret_cast<device const char*>(sidecar);
            device const PageSidecarHeader* sc_hdr =
                reinterpret_cast<device const PageSidecarHeader*>(
                    sidecar_bytes + row_sidecar_base + page_ptr->header.sidecar_start);

            const uint sc_count = sc_hdr->count;
            const uint sc_start = sc_hdr->start_index;
            const float sc_scale = sc_hdr->residual_scale;

            // Sidecar entry values follow the header.
            device const half* sc_vals =
                reinterpret_cast<device const half*>(sc_hdr + 1);

            for (uint ei = 0; ei < sc_count; ++ei) {
                const uint pos = sc_start + ei;  // position within this page
                const uint col = p * page_width + pos;
                if (col >= in_dim) break;
                // Override the ternary weight with the sidecar entry value
                // scaled by the sidecar's per-span residual scale.
                acc = fma(float(input[col]), float(sc_vals[ei]) * sc_scale, acc);
            }
            local_reads += sc_count;
        }
    }

    // Publish sidecar stats into threadgroup memory.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tg_sc_hits[tid]  = local_hits;
    tg_sc_reads[tid] = local_reads;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint tg_sidecar_hits  = 0;
    uint tg_sidecar_reads = 0;
    if (tid == 0) {
        // Single-thread sum across all 64 thread slots.
        for (uint i = 0; i < 64; ++i) {
            tg_sidecar_hits  += tg_sc_hits[i];
            tg_sidecar_reads += tg_sc_reads[i];
        }
    }

    // ── Reduction ───────────────────────────────────────────────────────────
    acc = simd_sum(acc);

    threadgroup float shared_reduction[32];
    if (simd_lane == 0) {
        shared_reduction[simd_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        const uint nsimd = (64 + 31) / 32;   // 64 threads / 32-lane SIMD = 2
        float total = 0.0f;
        for (uint s = 0; s < nsimd; ++s) {
            total += shared_reduction[s];
        }
        output[row] = half(total);
    }

    // ── Instrumentation (mode_flags bit 1 = 0x02) ───────────────────────────
    // Only row 0, tid 0 writes the receipt to avoid races.  Sidecar stats are
    // from this row only (representative sample for the dispatch).
    // Write all fields: zero the unused ones, fill the meaningful ones.
    if ((params.mode_flags & 2u) && row == 0 && tid == 0) {
        receipt->kernel_id               = 0;
        receipt->phase_id                = 0;
        receipt->page_count              = params.page_count;
        receipt->sidecar_hits            = tg_sidecar_hits;
        receipt->sidecar_entries_read    = tg_sidecar_reads;
        receipt->threadgroups            = out_dim;
        receipt->threads_per_threadgroup = 64;
        receipt->output_elements         = out_dim;
        receipt->flags                   = params.mode_flags;
        receipt->logical_weight_bytes    =
            ulong(nt) * ulong(sizeof(PackedTernaryPage640));
        receipt->logical_sidecar_bytes   = 0;
        receipt->logical_activation_bytes =
            ulong(in_dim) * ulong(sizeof(half));
    }
}
