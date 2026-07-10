// ── Page Candidate Score ─────────────────────────────────────────────────────
// Scores candidate pages by estimating output impact from ternary weight
// placement. Uses a sparsity prior (current weight = 0) and unit-weight
// magnitude proxy (no scale buffer bound) — produces relative rankings.
//
// ## Algorithm (per threadgroup = one candidate page)
//   1. Load PackedTernaryPage640 at candidate_packed[gid]
//   2. lane (0..31) decodes payload[lane] into 20 base-3 trits
//   3. For each non-zero trit:
//      a) local_weighted_error += |gradient[position]|
//      b) predicted_activation_delta += |input[position]|
//   4. sidecar_cost = header.sidecar_end - header.sidecar_start
//   5. simd_sum across all 32 lanes
//   6. Write PageScore[gid]; gid == 0 also writes KernelReceipt
//
// ## ABI
//   buffer(0): candidate_packed  PackedTernaryPage640[] — candidate pages
//   buffer(1): input_activations half[]                  — layer input (in_dim)
//   buffer(2): error_gradient    half[]                  — [out_dim * in_dim]
//   buffer(3): output_scores     PageScore[]             — result per page
//   buffer(4): params            ProjectionParams
//   buffer(5): receipt           KernelReceipt
//
// ## Dispatch
//   threadgroups = page_count, threads_per_threadgroup = 32 (one SIMD group)
//
// ## Struct layouts (match kernel_types.rs #[repr(C)])
//
//   PageHeader (16 B)
//     scale_index       u32   4 B   — output row this page belongs to
//     sidecar_start     u32   4 B   — byte offset into sidecar entries
//     sidecar_end       u32   4 B   — exclusive end offset
//     valid_tail_length u16   2 B
//     flags             u16   2 B
//
//   PackedTernaryPage640 (176 B)
//     payload           u32[40] 160 B
//     header            PageHeader  16 B
//
//   ProjectionParams (44 B)
//     in_dim      u32
//     out_dim     u32
//     page_count  u32
//     page_width  u32
//     mode_flags  u32
//     probe_seed  u32
//     reserved    u32[5]
//
//   KernelReceipt (64 B)
//     9 u32 = 36 B + pad(u32) = 40 B + 3 u64 = 24 B = 64 B
//
//     kernel_id              u32
//     phase_id               u32
//     page_count             u32
//     sidecar_hits           u32
//     sidecar_entries_read   u32
//     threadgroups           u32
//     threads_per_threadgroup u32
//     output_elements        u32
//     flags                  u32
//     _pad_receipt           u32    ← explicit alignment pad
//     logical_weight_bytes   uint64_t    (8 B)
//     logical_sidecar_bytes  uint64_t    (8 B)
//     logical_activation_bytes uint64_t  (8 B)
//
//   PageScore (44 B)
//     page_id                  u32
//     _pad                     u32
//     local_weighted_error     float
//     predicted_activation_delta float
//     sidecar_cost             float
//     estimated_bytes          float
//     estimated_loads          float
//     accepted_score           float   (init 0)
//     challenger_score         float   (init 0)
//     flags                    u32
//     _pad2                    u32
//
//     11 fields x 4 B = 44 B
//
// ──────────────────────────────────────────────────────────────────────────────

#include <metal_stdlib>
using namespace metal;

constant uint LANES     = 32;    // threads per page
constant uint PER_LANE  = 20;    // trits per u32 word
constant uint PAGE_BYTES = 176;  // sizeof(PackedTernaryPage640)

// ── GPU-side structs: byte-for-byte match of Rust #[repr(C)] ────────────────

struct PageHeader {
    uint   scale_index;         // output row this page belongs to
    uint   sidecar_start;
    uint   sidecar_end;
    ushort valid_tail_length;
    ushort flags;
};

struct PackedTernaryPage640 {
    uint    payload[40];        // 160 bytes (32 words used in base-3, 8 spare)
    PageHeader header;
};

struct ProjectionParams {
    uint in_dim;
    uint out_dim;
    uint page_count;
    uint page_width;
    uint mode_flags;
    uint probe_seed;
    uint reserved[5];
};

struct PageScore {
    uint  page_id;
    uint  _pad;
    float local_weighted_error;
    float predicted_activation_delta;
    float sidecar_cost;
    float estimated_bytes;
    float estimated_loads;
    float accepted_score;
    float challenger_score;
    uint  flags;
    uint  _pad2;
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
    uint     _pad_receipt;         // explicit pad → u64 fields at offset 40
    uint64_t logical_weight_bytes;
    uint64_t logical_sidecar_bytes;
    uint64_t logical_activation_bytes;
};

// ── Page candidate score kernel ─────────────────────────────────────────────

kernel void page_candidate_score(
    device const PackedTernaryPage640* candidate_packed [[buffer(0)]],
    device const half*                 input_activation [[buffer(1)]],
    device const half*                 error_gradient   [[buffer(2)]],
    device PageScore*                  output_scores    [[buffer(3)]],
    constant ProjectionParams&         params           [[buffer(4)]],
    device KernelReceipt*              receipt          [[buffer(5)]],
    uint gid                                          [[threadgroup_position_in_grid]],
    uint tid                                          [[thread_position_in_threadgroup]])
{
    if (gid >= params.page_count) return;

    device const PackedTernaryPage640& page = candidate_packed[gid];

    // Page position within the weight matrix.
    // Pages are laid out row-by-row: every pages_per_row pages advances the
    // output row. Within a row, column slices are [page*page_width, (page+1)*page_width).
    const uint pages_per_row  = (params.in_dim + params.page_width - 1) / params.page_width;
    const uint output_row     = gid / pages_per_row;
    const uint page_in_row    = gid % pages_per_row;
    const uint col_start      = page_in_row * params.page_width;

    // Per-thread accumulators (lane 0..31)
    float local_weighted_error   = 0.0f;
    float predicted_activation_delta = 0.0f;
    uint  non_zero_count         = 0;

    // Decode one u32 word (tid = 0..31) into 20 base-3 trits.
    uint word = page.payload[tid];
    for (uint vi = 0; vi < PER_LANE; ++vi) {
        uint d = word % 3u;          // LSB trit first: 0=skip, 1=+scale, 2=-scale
        word /= 3u;

        if (d != 0u) {
            // Global column index in the weight matrix.
            uint col = col_start + tid * PER_LANE + vi;
            if (col >= params.in_dim) break;

            // |weight| = 1.0 proxy (no scale buffer bound for this kernel).
            // Error sensitivity: how much does a unit weight at (row, col)
            // affect the loss?
            float grad_abs = abs(float(error_gradient[output_row * params.in_dim + col]));
            local_weighted_error += grad_abs;

            // Activation delta estimate: |input[col]| for the new weight.
            predicted_activation_delta += abs(float(input_activation[col]));

            non_zero_count++;
        }
    }

    // SIMD reduction — 32 threads is one SIMD group, simd_sum covers all lanes.
    local_weighted_error        = simd_sum(local_weighted_error);
    predicted_activation_delta  = simd_sum(predicted_activation_delta);
    uint total_non_zero = simd_sum(non_zero_count);

    // Thread 0 writes the PageScore record.
    if (tid == 0) {
        uint sidecar_bytes = page.header.sidecar_end - page.header.sidecar_start;

        device PageScore& score = output_scores[gid];
        score.page_id                  = gid;
        score._pad                     = 0;
        score.local_weighted_error     = local_weighted_error;
        score.predicted_activation_delta = predicted_activation_delta;
        score.sidecar_cost             = float(sidecar_bytes);
        score.estimated_bytes          = float(PAGE_BYTES + sidecar_bytes);
        score.estimated_loads          = float(total_non_zero);
        score.accepted_score           = 0.0f;
        score.challenger_score         = 0.0f;
        score.flags                    = page.header.flags;
        score._pad2                    = 0;

        // gid == 0 also writes the KernelReceipt.
        if (gid == 0) {
            device KernelReceipt& rec = *receipt;
            rec.kernel_id                 = 0;
            rec.phase_id                  = 0;
            rec.page_count                = params.page_count;
            rec.sidecar_hits              = 0;
            rec.sidecar_entries_read      = 0;
            rec.threadgroups              = params.page_count;
            rec.threads_per_threadgroup   = LANES;
            rec.output_elements           = params.page_count;
            rec.flags                     = 0;
            rec._pad_receipt              = 0;
            rec.logical_weight_bytes      = uint64_t(total_non_zero) * 4;    // fp32 weight est.
            rec.logical_sidecar_bytes     = uint64_t(sidecar_bytes);
            rec.logical_activation_bytes  = uint64_t(total_non_zero) * 2;    // half input est.
        }
    }
}
