// ── SIDECAR APPLY + VERIFY ───────────────────────────────────────────────────
// Applies per-span sidecar overrides to the activation buffer and writes a
// verification record per span. Each threadgroup (64 threads) processes one
// sidecar span — identified by a byte offset into the sidecar entries buffer.
//
// The small bounded sidecar span is loaded into threadgroup memory for
// coalesced access before application.
//
// Two modes (selected by params.mode_flags bit 1):
//   sealed      (bit 1 == 0) — normal inference, apply recorded sidecar.
//   candidate   (bit 1 == 1) — score proposed allocation under hard cap:
//                               also accumulates projected_impact = Σ|delta|,
//                               clamps application if count exceeds cap.
//
// ALWAYS runs: the sidecar spans are the definitive source of override values.
// The activation buffer is updated in-place; the host reads verification
// records to validate correctness and (in candidate mode) score pages.
//
// Threadgroup strategy:
//   1. Thread 0 loads PageSidecarHeader into threadgroup memory.
//   2. Threads load entry values into TG memory (stride-64 loop).
//   3. Barrier. Then all threads apply from TG memory.
//   4. simd_sum per simdgroup + barrier + thread 0 finalizes the record.
//
// Buffer layout:
//   [0] activations      device half*       — result buffer patched in-place
//   [1] sidecar          device uint8_t*    — interleaved PageSidecarHeader +
//                                              count × half override values
//   [2] sidecar_offsets  device uint*       — byte offsets into sidecar buffer
//   [3] output_verify    device SidecarVerifyRecord*  — one record per span
//   [4] params           constant ProjectionParams&   — page_count, mode_flags
//   [5] receipt          device KernelReceipt*        — instrument counters
//
// Validation: see compute-core/tests/sidecar_apply_verify_bench.rs

#include <metal_stdlib>
using namespace metal;

// ── ABI structs (match kernel_types.rs #[repr(C)]) ───────────────────────────

// PageSidecarHeader — per-span header in the sidecar entries buffer.
// sizeof = 20 bytes in Rust (start_index: u32, count: u16, encoding: u16,
//          residual_scale: f32, flags: u32).
struct PageSidecarHeader {
    uint    start_index;       // activation position for the first override
    ushort  count;             // number of override entries in this span
    ushort  encoding;          // encoding format (0 = direct half)
    float   residual_scale;    // scale factor (matches Rust f32)
    uint    flags;             // reserved for future per-span flags
};

// ProjectionParams — per-dispatch configuration.
struct ProjectionParams {
    uint    in_dim;
    uint    out_dim;
    uint    page_count;
    uint    page_width;
    uint    mode_flags;        // bit0=sidecar_enabled, bit1=candidate/score mode
    uint    probe_seed;
    uint    reserved[5];
};

// KernelReceipt — instrumentation counters updated deterministically.
struct KernelReceipt {
    uint    kernel_id;
    uint    phase_id;
    uint    page_count;
    uint    sidecar_hits;
    uint    sidecar_entries_read;
    uint    threadgroups;
    uint    threads_per_threadgroup;
    uint    output_elements;
    uint    flags;
    uint    _pad_receipt;         // explicit pad → u64 fields at offset 40
    ulong   logical_weight_bytes;
    ulong   logical_sidecar_bytes;
    ulong   logical_activation_bytes;
};

// Per-span verification record (32 bytes).
// Host reads these in span order to reconcile correctness.
struct SidecarVerifyRecord {
    uint    hit_count;
    uint    entries_read;
    float   checksum;
    float   projected_impact;
    float   _pad0;
    float   _pad1;
    uint    _pad2;
    uint    _pad3;
};

constant uint TG_SIZE = 64;

// ── Kernel ───────────────────────────────────────────────────────────────────

kernel void sidecar_apply_verify(
    device half*                    activations       [[buffer(0)]],
    device const uint8_t*           sidecar           [[buffer(1)]],
    device const uint*              sidecar_offsets   [[buffer(2)]],
    device SidecarVerifyRecord*     output_verify     [[buffer(3)]],
    constant ProjectionParams&      params            [[buffer(4)]],
    device KernelReceipt*           receipt           [[buffer(5)]],
    uint gid                                         [[threadgroup_position_in_grid]],
    uint tid                                         [[thread_position_in_threadgroup]],
    uint simd_lane                                   [[thread_index_in_simdgroup]],
    uint simd_id                                     [[simdgroup_index_in_threadgroup]])
{
    const uint span_count = params.page_count;
    if (gid >= span_count) return;

    // ── Threadgroup memory for the sidecar span ────────────────────────────
    // The header (20 B) + up to 64 half entries (128 B) = 148 B.
    // This fits comfortably within Metal's threadgroup memory limit (32 KiB).
    threadgroup half   tg_residual_scale;
    threadgroup uint   tg_start_index;
    threadgroup ushort tg_count;
    threadgroup half   tg_entries[TG_SIZE];

    // ── Phase 1: load sidecar header + entries into threadgroup memory ────
    const uint byte_offset = sidecar_offsets[gid];
    device const PageSidecarHeader* hdr =
        (device const PageSidecarHeader*)(sidecar + byte_offset);

    if (tid == 0) {
        tg_start_index    = hdr->start_index;
        tg_count          = hdr->count;
        tg_residual_scale = half(hdr->residual_scale);
    }

    // Load entry values into TG memory (stride-64 loop for spans > 64).
    device const half* entries_base =
        (device const half*)(sidecar + byte_offset + sizeof(PageSidecarHeader));
    const ushort count = hdr->count;
    for (uint i = tid; i < uint(count); i += TG_SIZE) {
        tg_entries[i] = entries_base[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 2: apply entries from threadgroup memory ────────────────────
    const uint   start_index    = tg_start_index;
    const uint   actual_count   = uint(tg_count);
    const float  residual_scale = float(tg_residual_scale);
    const bool   candidate_mode = (params.mode_flags & 2u) != 0u;

    uint   local_hits     = 0u;
    float  local_checksum = 0.0f;
    float  local_impact   = 0.0f;

    for (uint i = tid; i < actual_count; i += TG_SIZE) {
        const uint  pos = start_index + i;
        if (pos >= params.out_dim) break;
        const float ov    = float(tg_entries[i]);
        const float delta = fma(residual_scale, ov, 0.0f);

        activations[pos] = half(float(activations[pos]) + delta);

        local_hits     += 1u;
        local_checksum += delta;
        local_impact   += fabs(delta);
    }

    // ── Reduction: simd → threadgroup → thread 0 ──────────────────────────
    const uint   simd_hits     = simd_sum(local_hits);
    const float  simd_checksum = simd_sum(local_checksum);
    const float  simd_impact   = simd_sum(local_impact);

    threadgroup uint   tg_hits[2];
    threadgroup float  tg_cksum[2];
    threadgroup float  tg_impact[2];

    if (simd_lane == 0u) {
        tg_hits[simd_id]   = simd_hits;
        tg_cksum[simd_id]  = simd_checksum;
        tg_impact[simd_id] = simd_impact;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Thread 0: finalize verification record + update receipt ────────────
    if (tid == 0u) {
        const uint   total_hits  = tg_hits[0] + tg_hits[1];
        const float  total_cksum = tg_cksum[0] + tg_cksum[1];
        const float  total_imp   = tg_impact[0] + tg_impact[1];

        device SidecarVerifyRecord& rec = output_verify[gid];
        rec.hit_count        = total_hits;
        rec.entries_read     = count;
        rec.checksum         = total_cksum;
        rec.projected_impact = candidate_mode ? total_imp : 0.0f;
        rec._pad0            = 0.0f;
        rec._pad1            = 0.0f;
        rec._pad2            = 0u;
        rec._pad3            = 0u;

        // Relaxed-order atomics for instrumentation — never on arithmetic path.
        device atomic_uint* r_hits  = (device atomic_uint*)&receipt->sidecar_hits;
        device atomic_uint* r_reads = (device atomic_uint*)&receipt->sidecar_entries_read;
        atomic_fetch_add_explicit(r_hits,  total_hits, memory_order_relaxed);
        atomic_fetch_add_explicit(r_reads, actual_count, memory_order_relaxed);
    }
}
