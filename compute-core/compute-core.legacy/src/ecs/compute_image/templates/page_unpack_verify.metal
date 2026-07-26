// ── GPU-side Pack Verification (Debug Kernel) ────────────────────────────
//
// Each threadgroup unpacks one PackedTernaryPage640 (640 ternary weights at
// 1.6-bit base-3 encoding), writes decoded half values and a FNV-style
// checksum. Host side compares decoded values against reference to catch
// pack/unpack round-trip bugs.
//
// Buffer layout:
//   buffer(0): checksum          uint[]      — [page_count] FNV-style checksums
//   buffer(1): decoded_samples   half[]      — [page_count * 640] decoded values
//   buffer(2): packed_pages      PackedTernaryPage640[]
//   buffer(3): params            ProjectionParams
//
// Dispatch: threadgroups = page_count, threads_per_threadgroup = 32

#include <metal_stdlib>
using namespace metal;

constant uint TILE_SIZE = 640;   // weights per page
constant uint LANES     = 32;    // threads per page (one per u32 word)
constant uint PER_LANE  = 20;    // trits per word (TILE_SIZE / LANES)

// ── GPU-side structs: byte-for-byte match of Rust #[repr(C)] ────────────

struct PageHeader {
    uint   scale_index;         // index into page_scales buffer
    uint   sidecar_start;       // start offset in sidecar entries buffer
    uint   sidecar_end;         // end offset (exclusive) in sidecar entries
    ushort valid_tail_length;   // valid weight positions in this page (≤640)
    ushort flags;               // bit0 = sidecar_present, bit1 = tail_padding
};

struct PackedTernaryPage640 {
    uint       payload[40];     // 160 bytes (32 words used for base-3, 8 spare)
    PageHeader header;
};

struct ProjectionParams {
    uint in_dim;
    uint out_dim;
    uint page_count;
    uint page_width;
    uint mode_flags;
    uint probe_seed;
    uint reserved[5];           // pad to 16-byte alignment
};

// ── Kernel ──────────────────────────────────────────────────────────────

kernel void page_unpack_verify(
    device uint*                       checksum_output  [[buffer(0)]],
    device half*                       sample_output    [[buffer(1)]],
    device const PackedTernaryPage640* packed_pages     [[buffer(2)]],
    constant ProjectionParams&         params           [[buffer(3)]],
    uint gid  [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]])
{
    if (gid >= params.page_count) return;

    device const PackedTernaryPage640& page = packed_pages[gid];

    // ── Threadgroup memory for decoded values ──────────────────────────
    // 640 half × 2 B = 1280 B — well within Metal's threadgroup limit.
    threadgroup half decoded[TILE_SIZE];

    // ── Decode one u32 word (tid = 0..31) into 20 base-3 trits ────────
    uint word = page.payload[tid];
    uint local_checksum = 0;

    // Unit scale for round-trip verification — scale buffers are not
    // bound. The CPU comparator reconstructs the same unit-scaled values.
    const float scale = 1.0f;

    for (uint vi = 0; vi < PER_LANE; ++vi) {
        uint d = word % 3u;          // LSB trit first: 0=skip, 1=+scale, 2=-scale
        word /= 3u;

        uint col = tid * PER_LANE + vi;
        if (col >= params.page_width) break;

        half val;
        if (d == 0) {
            val = 0.0h;
        } else if (d == 1) {
            val = half(scale);
        } else {
            val = half(-scale);
        }
        decoded[col] = val;

        // FNV-like hash over trit digits (d + 1 in 1..3).
        local_checksum = local_checksum * 31u + uint(d + 1u);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Write sampled decoded values (half) to buffer(1) ───────────────
    const uint page_offset = gid * TILE_SIZE;
    for (uint vi = 0; vi < PER_LANE; ++vi) {
        uint col = tid * PER_LANE + vi;
        if (col >= params.page_width) break;
        sample_output[page_offset + col] = decoded[col];
    }

    // ── Combine checksums via threadgroup memory ──────────────────────
    threadgroup uint tg_checksums[LANES];
    tg_checksums[tid] = local_checksum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        // Roll up all 32 per-word checksums into one page checksum.
        // NOTE: ordered reduction — word 0 contributes 31^(31-i)× more than
        // word 31. Host-side verifier MUST replicate this exact reduction.
        uint page_checksum = tg_checksums[0];
        for (uint i = 1; i < LANES; ++i) {
            page_checksum = page_checksum * 31u + tg_checksums[i];
        }

        // Write checksum to buffer(0).
        checksum_output[gid] = page_checksum;
    }
}
