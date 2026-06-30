// ── GPU-Accelerated TernaryTile640 Packer ────────────────────────────
//
// Compiles at compile-image-build time.  Each threadgroup processes one
// 640-weight tile of one row of the weight matrix.
//
// Grid layout:
//   threads  = rows × (cols / 640) × 32
//   Each SIMD lane processes 20 ternary weights, packs into 1 u32 via
//   Base-3 encoding (digit 0=0, 1=+1, 2=-1).
//
// Thread 0 in each threadgroup computes the tile's absmax scale via
// simd_reduce_max, then broadcasts to all lanes via threadgroup memory.
//
// Input:   BF16 row-major [N, K]
// Output:  packed_u32  [N × num_tiles × 32]  (u32)
//          scales_f32  [N × num_tiles]         (f32)

#include <metal_stdlib>
using namespace metal;

constant uint TILE_SIZE   = 640;   // weights per tile
constant uint LANES       = 32;    // threads per tile
constant uint PER_LANE    = 20;    // TILE_SIZE / LANES

kernel void tile640_pack(
    device const half*   input        [[buffer(0)]],  // [N, K] BF16 row-major
    device uint*         packed_out   [[buffer(1)]],  // [N × tiles × 32] u32
    device float*        scales_out   [[buffer(2)]],  // [N × tiles] f32
    constant uint&       K            [[buffer(3)]],  // input columns
    constant uint&       N            [[buffer(4)]],  // rows
    constant uint&       num_tiles    [[buffer(5)]],  // tiles per row
    uint                 tid          [[thread_position_in_grid]],
    uint                 lane         [[thread_index_in_simdgroup]])
{
    uint row    = tid / num_tiles;
    uint tile   = tid % num_tiles;
    if (row >= N || tile >= num_tiles) return;

    // ── Step 1: Load this tile's 640 BF16 weights into threadgroup memory ──
    // Each thread loads 20 weights (one lane's worth).
    threadgroup half  tile_weights[TILE_SIZE];
    uint tile_base = row * K + tile * TILE_SIZE;
    uint entry_idx = lane * PER_LANE;

    for (uint i = 0; i < PER_LANE; ++i) {
        uint src = tile_base + entry_idx + i;
        tile_weights[entry_idx + i] = src < row * K + K ? input[src] : 0.0h;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Step 2: Thread 0 computes absmax scale ─────────────────────────
    // Use simd_reduce_max for the per-thread chunk, then write to TG mem.
    threadgroup float tile_scale;
    if (lane == 0) {
        float absmax = 0.0f;
        for (uint i = 0; i < TILE_SIZE; ++i) {
            absmax = fmax(absmax, fabs((float)tile_weights[i]));
        }
        tile_scale = absmax > 1e-12f ? absmax : 1.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_scale = 1.0f / tile_scale;

    // Write scale output.
    uint scale_idx = row * num_tiles + tile;
    scales_out[scale_idx] = tile_scale;

    // ── Step 3: Pack 20 ternary weights into one u32 via Base-3 ────────
    uint packed = 0;
    uint mul    = 1;  // 3^0, 3^1, ..., 3^19
    for (uint i = 0; i < PER_LANE; ++i) {
        float val = (float)tile_weights[entry_idx + i] * inv_scale;
        uint digit;
        if (val > 0.5f)       digit = 1;  // +1
        else if (val < -0.5f) digit = 2;  // -1
        else                  digit = 0;  // 0
        packed += digit * mul;
        mul *= 3;
    }

    // ── Step 4: Write packed u32 to output ─────────────────────────────
    uint out_idx = row * num_tiles * LANES + tile * LANES + lane;
    packed_out[out_idx] = packed;
}

// ── Q8_0 to TernaryTile640 Pack ────────────────────────────────────
//
// Each threadgroup processes one 640-weight tile of one output row.
// Input is Q8_0 blocks in [N, K] row-major order (pre-transposed from
// GGUF's original [K, N] layout — the CPU reorders block indices)
// so each thread loads one contiguous Q8_0 block (34 bytes) via coalesced
// device memory access, dequantizes to f32 in threadgroup memory,
// then applies the same ternary-quantize + Base-3 pack as tile640_pack.
//
// Q8_0 block format: [2B f16 scale] [32B int8 values] = 34 bytes.
// 640 / 32 = 20 blocks per tile.

constant uint Q8_BLOCK_VALS = 32;   // values per Q8_0 block
constant uint Q8_BLOCK_BYTES = 34;  // total bytes per Q8_0 block
constant uint Q8_BLOCKS_PER_TILE = TILE_SIZE / Q8_BLOCK_VALS; // 20

kernel void q8_0_ternary_pack(
    device const uchar*  q8_input    [[buffer(0)]],  // [N, K] Q8_0 blocks, transposed
    device uint*         packed_out  [[buffer(1)]],
    device float*        scales_out  [[buffer(2)]],
    constant uint&       K           [[buffer(3)]],  // in_features
    constant uint&       N           [[buffer(4)]],  // out_features
    constant uint&       num_tiles   [[buffer(5)]],
    uint                 gid         [[threadgroup_position_in_grid]],
    uint                 lane        [[thread_index_in_simdgroup]])
{
    uint row  = gid / num_tiles;
    uint tile = gid % num_tiles;
    if (row >= N || tile >= num_tiles) return;

    threadgroup float tile_vals[TILE_SIZE];

    // ── Step 1: Load Q8_0 blocks via coalesced access ──────────────
    // Each of 32 lanes loads one block (only 20 needed; lanes 20..31 idle).
    uint b = lane;
    if (b < Q8_BLOCKS_PER_TILE) {
        uint block_base = (row * (K / Q8_BLOCK_VALS) + tile * Q8_BLOCKS_PER_TILE + b) * Q8_BLOCK_BYTES;
        half scale = *(device const half*)(q8_input + block_base);
        float fscale = (float)scale;
        uint remaining = K - (tile * TILE_SIZE + b * Q8_BLOCK_VALS);
        uint n_valid = remaining < Q8_BLOCK_VALS ? remaining : Q8_BLOCK_VALS;
        for (uint i = 0; i < n_valid; ++i) {
            tile_vals[b * Q8_BLOCK_VALS + i] = (float)((char)q8_input[block_base + 2 + i]) * fscale;
        }
        for (uint i = n_valid; i < Q8_BLOCK_VALS; ++i) {
            tile_vals[b * Q8_BLOCK_VALS + i] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Step 2: Compute tile absmax scale ──────────────────────────
    float local_max = 0.0f;
    uint start = lane * PER_LANE;
    for (uint i = 0; i < PER_LANE; ++i) {
        local_max = fmax(local_max, fabs(tile_vals[start + i]));
    }
    float tile_scale = simd_reduce_max(local_max);
    if (tile_scale < 1e-12f) tile_scale = 1.0f;
    float inv_scale = 1.0f / tile_scale;

    scales_out[row * num_tiles + tile] = tile_scale;

    // ── Step 3: Ternary quantize + Base-3 pack ─────────────────────
    uint packed = 0;
    uint mul = 1;
    for (uint i = 0; i < PER_LANE; ++i) {
        float val = tile_vals[start + i] * inv_scale;
        uint digit;
        if      (val >  0.5f) digit = 1;
        else if (val < -0.5f) digit = 2;
        else                  digit = 0;
        packed += digit * mul;
        mul *= 3;
    }

    uint out_idx = row * num_tiles * LANES + tile * LANES + lane;
    packed_out[out_idx] = packed;
}
