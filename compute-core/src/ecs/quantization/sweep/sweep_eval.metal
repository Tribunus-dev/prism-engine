#include <metal_stdlib>
using namespace metal;

// ──────────────────────────────────────────────────────────────────────────
// sweep_eval_nf4 — GPU-accelerated NF4 candidate evaluation for QuantSweep
//
// Each threadgroup (32 threads) evaluates one candidate on one tile of
// source weights. Grid is flattened 1D: gid = candidate_id * num_tiles + tile_id.
//
// Buffer layout:
//   [0] source_weights    — f32[N*K], row-major weight matrix, flat
//   [1] source_sq_sums    — f32[num_tiles], pre-computed sum-of-squares per tile
//   [2] candidate_params  — uint4[num_candidates]: (codebook_id, group_size, affine_mode, 0)
//   [3] codebook_bank     — f32[48] = 3 x 16-entry NF4 codebooks concatenated
//   [4] output_tile_results — f32[num_candidates x num_tiles x 3]: per tile
//                            [0] = squared_error_sum, [1] = max_abs_error, [2] = source_sq_sum
//   [5] constants         — uint4 (total_elements, num_tiles, num_candidates, 0)
//
// Threadgroup size: 32 threads.
// ──────────────────────────────────────────────────────────────────────────

constant uint TILE_SIZE = 640;

kernel void sweep_eval_nf4(
    device const float*  source_weights       [[buffer(0)]],
    device const float*  source_sq_sums       [[buffer(1)]],
    device const uint4*  candidate_params     [[buffer(2)]],
    device const float*  codebook_bank        [[buffer(3)]],
    device float*        output_tile_results  [[buffer(4)]],
    constant uint4&      constants            [[buffer(5)]],
    uint                 gid                  [[threadgroup_position_in_grid]],
    uint                 tid                  [[thread_index_in_threadgroup]]
) {
    uint total_elements = constants[0];
    uint num_tiles      = constants[1];
    uint num_candidates = constants[2];
    if (gid >= num_candidates * num_tiles) return;

    uint candidate_id = gid / num_tiles;
    uint tile_id      = gid % num_tiles;

    uint4 cand          = candidate_params[candidate_id];
    uint  codebook_id   = cand.x;
    uint  group_size    = cand.y;
    uint  affine_mode   = cand.z;

    device const float* codebook = codebook_bank + codebook_id * 16;
    float max_cb_abs = 0.0f;
    for (uint i = 0; i < 16; i++) { max_cb_abs = max(max_cb_abs, abs(codebook[i])); }
    if (max_cb_abs < 1e-30f) max_cb_abs = 1.0f;

    uint tile_start = tile_id * TILE_SIZE;
    uint tile_end   = min(tile_start + TILE_SIZE, total_elements);
    uint tile_elems = tile_end - tile_start;
    uint num_groups = (tile_elems + group_size - 1) / group_size;

    threadgroup float reduce_buf[32];
    float sq_err_accum = 0.0f;
    float max_err      = 0.0f;

    for (uint g = 0; g < num_groups; g++) {
        uint group_start = g * group_size;
        uint group_end   = min(group_start + group_size, tile_elems);
        uint group_elems = group_end - group_start;

        // Phase 1: per-thread max-abs
        float my_max_abs = 0.0f;
        for (uint e = tid; e < group_elems; e += 32) {
            my_max_abs = max(my_max_abs, abs(source_weights[tile_start + group_start + e]));
        }

        // Phase 2: tree reduction for group max-abs
        reduce_buf[tid] = my_max_abs;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 16) { reduce_buf[tid] = max(reduce_buf[tid], reduce_buf[tid + 16]); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 8)  { reduce_buf[tid] = max(reduce_buf[tid], reduce_buf[tid + 8]); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 4)  { reduce_buf[tid] = max(reduce_buf[tid], reduce_buf[tid + 4]); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 2)  { reduce_buf[tid] = max(reduce_buf[tid], reduce_buf[tid + 2]); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 1)  { reduce_buf[tid] = max(reduce_buf[tid], reduce_buf[tid + 1]); }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float scale = reduce_buf[0] / max_cb_abs;
        if (scale < 1e-30f) scale = 1.0f;

        // Phase 3: quantize & compute error
        for (uint e = tid; e < group_elems; e += 32) {
            float val   = source_weights[tile_start + group_start + e];
            float norm  = clamp(val / scale, -1.0f, 1.0f);
            float best_dist = INFINITY;
            uint8_t best_idx = 0;
            for (uint ci = 0; ci < 16; ci++) {
                float d = abs(norm - codebook[ci]);
                if (d < best_dist) { best_dist = d; best_idx = ci; }
            }
            float decoded = codebook[best_idx] * scale;
            float err     = val - decoded;
            sq_err_accum += err * err;
            max_err       = max(max_err, abs(err));
        }
    }

    // Final tree sum reduction for tile sq_err
    reduce_buf[tid] = sq_err_accum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 16) { reduce_buf[tid] = reduce_buf[tid] + reduce_buf[tid + 16]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 8)  { reduce_buf[tid] = reduce_buf[tid] + reduce_buf[tid + 8]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 4)  { reduce_buf[tid] = reduce_buf[tid] + reduce_buf[tid + 4]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 2)  { reduce_buf[tid] = reduce_buf[tid] + reduce_buf[tid + 2]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 1)  { reduce_buf[tid] = reduce_buf[tid] + reduce_buf[tid + 1]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tile_sq_err = reduce_buf[0];

    // Tree max reduction for tile max_err
    reduce_buf[tid] = max_err;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 16) { reduce_buf[tid] = max(reduce_buf[tid], reduce_buf[tid + 16]); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 8)  { reduce_buf[tid] = max(reduce_buf[tid], reduce_buf[tid + 8]); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 4)  { reduce_buf[tid] = max(reduce_buf[tid], reduce_buf[tid + 4]); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 2)  { reduce_buf[tid] = max(reduce_buf[tid], reduce_buf[tid + 2]); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 1)  { reduce_buf[tid] = max(reduce_buf[tid], reduce_buf[tid + 1]); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tile_max_err = reduce_buf[0];

    if (tid == 0) {
        uint out_base = candidate_id * num_tiles * 3 + tile_id * 3;
        output_tile_results[out_base + 0] = tile_sq_err;
        output_tile_results[out_base + 1] = tile_max_err;
        output_tile_results[out_base + 2] = source_sq_sums[tile_id];
    }
}
