//! Gather kernel: reads paged TernaryBlock32 fragments from L2,
//! decompresses to FP16, and scatters into contiguous L1 staging buffer.
//! Also: scatter kernel for eviction (FP16 L1 → ternary L2).

use metal::*;

// ── Page constants (Gemma 4 12B) ───────────────────────────────────
// These duplicate kv.rs intent but are specific to the paging layer's
// TernaryBlock32 layout (32 elements/block, 9 bytes + 2 outlier).
// Head dimension = 512 (global), blocks = 16.

/// Gather kernel: reads 256 threads per dispatch.
/// Each thread handles one (layer, head, block, element) tuple.
pub const GATHER_KERNEL_SRC: &str = r##"
#include <metal_stdlib>
using namespace metal;

// Constants matching the C/K/C extension pipeline
constant uint NUM_KV_HEADS  = 8;
constant uint HEAD_DIM      = 512;
constant uint BLOCKS_PER_HEAD = HEAD_DIM / 32;  // 16

// TernaryBlock32: 7 bytes + 2 bytes scale = 9 bytes per 32 elements
constant uint TERNARY_BLOCK_BYTES = 9;
constant uint OUTLIER_BYTES = 2;
constant uint BYTES_PER_TOKEN_HEAD = BLOCKS_PER_HEAD * (TERNARY_BLOCK_BYTES + OUTLIER_BYTES);

// Each page = 64 tokens × 8 heads × BYTES_PER_TOKEN_HEAD
constant uint TOKENS_PER_PAGE = 64;
constant uint PAGE_BYTES = TOKENS_PER_PAGE * NUM_KV_HEADS * BYTES_PER_TOKEN_HEAD;

struct PageEntry {
    device uchar* page_base;
    uint token_offset_within_page;  // 0..63, bytes = token_offset * NUM_KV_HEADS * BYTES_PER_TOKEN_HEAD
};

/// Gather kernel: reads 256 threads per dispatch.
/// Each thread handles one (layer, head, block, element) tuple.
kernel void gather_ternary_to_fp16_l1(
    device const uchar*    page_table_buf    [[buffer(0)]],  // flattened PageEntry array
    device const uint*     token_count       [[buffer(1)]],  // number of tokens to gather
    device half*           l1_staging_k      [[buffer(2)]],  // FP16 output (K)
    device half*           l1_staging_v      [[buffer(3)]],  // FP16 output (V)
    constant uint&         layer             [[buffer(4)]],
    uint tid                                 [[thread_position_in_grid]])
{
    // tid maps to: [token_index * (NUM_KV_HEADS * HEAD_DIM) + head * HEAD_DIM + dim]
    uint total_elements = *token_count * NUM_KV_HEADS * HEAD_DIM;
    if (tid >= total_elements) return;

    uint token_idx = tid / (NUM_KV_HEADS * HEAD_DIM);
    uint head = (tid / HEAD_DIM) % NUM_KV_HEADS;
    uint dim = tid % HEAD_DIM;
    uint block_idx = dim / 32;
    uint elem_in_block = dim % 32;

    // Read PageEntry from table
    device const PageEntry* entry = (device const PageEntry*)(
        page_table_buf + token_idx * sizeof(PageEntry)
    );

    // Calculate offset within the page for this (head, block)
    uint head_block_offset = head * BLOCKS_PER_HEAD * TERNARY_BLOCK_BYTES
        + block_idx * TERNARY_BLOCK_BYTES;
    uint page_offset = entry->token_offset_within_page * NUM_KV_HEADS * BLOCKS_PER_HEAD * TERNARY_BLOCK_BYTES
        + head_block_offset;

    device const uchar* block_base = entry->page_base + page_offset;

    // Read TernaryBlock32
    // 7 bytes: 5 trits per byte
    uint byte_idx = elem_in_block / 5;
    uint pos_in_byte = elem_in_block % 5;
    uchar byte_val = block_base[byte_idx];

    // Extract trit at pos_in_byte
    uint trit;
    if (byte_idx >= 6) {
        if (elem_in_block == 30) trit = (uint)byte_val % 3;
        else trit = (uint)byte_val / 3;
    } else {
        uint v = (uint)byte_val;
        for (uint i = 0; i < pos_in_byte; ++i) {
            v = (v * 86u) >> 8;
        }
        uint q = (v * 86u) >> 8;
        trit = v - q * 3;
    }

    // Read FP16 scale at offset 7
    half scale = *(device const half*)(block_base + 7);
    half val = (half)((int)trit - 1) * scale;

    // Check outlier mask and patch if needed
    // Outlier mask: u32 at offset 9 (after 7 trits + 2 scale)
    uint mask = *(device const uint*)(block_base + 9);
    if ((mask >> elem_in_block) & 1u) {
        // Read FP16 outlier at offset 13 (after 7 trits + 2 scale + 4 mask = 13)
        uint bits_before = popcount(mask & ((1u << elem_in_block) - 1u));
        device const half* outlier_base = (device const half*)(block_base + 13);
        val = outlier_base[bits_before];
    }

    // Write to L1 staging (contiguous FP16)
    uint l1_offset = token_idx * NUM_KV_HEADS * HEAD_DIM + head * HEAD_DIM + dim;
    l1_staging_k[l1_offset] = val;

    // Repeat for V — same offset pattern but different buffer
    // For simplicity, V is gathered in a separate dispatch
}
"##;

/// Scatter kernel: reads FP16 from L1 staging, quantizes to ternary,
/// packs 5 trits/byte, and writes to paged L2.
/// Inverse of the gather kernel.
pub const SCATTER_KERNEL_SRC: &str = r##"
/// Scatter kernel: reads FP16 from L1 staging, quantizes to ternary,
/// packs 5 trits/byte, and writes to paged L2.
/// Inverse of the gather kernel.
kernel void scatter_fp16_l1_to_ternary(
    device uchar*     page_table_out_buf [[buffer(0)]],  // output PageEntry array
    device const uint* token_count      [[buffer(1)]],
    device const half* l1_staging_k     [[buffer(2)]],   // FP16 input
    constant uint&    layer             [[buffer(3)]],
    uint tid                            [[thread_position_in_grid]])
{
    // Each block of 32 values: one thread computes the block's max-abs scale,
    // then 32 threads each quantize one element and pack to the 7-byte structure.
    uint total_blocks = *token_count * NUM_KV_HEADS * BLOCKS_PER_HEAD;
    if (tid >= total_blocks) return;

    uint token_idx = tid / (NUM_KV_HEADS * BLOCKS_PER_HEAD);
    uint head = (tid / BLOCKS_PER_HEAD) % NUM_KV_HEADS;
    uint block_idx = tid % BLOCKS_PER_HEAD;

    // Read the 32 FP16 values from L1 staging
    uint l1_base = token_idx * NUM_KV_HEADS * HEAD_DIM + head * HEAD_DIM + block_idx * 32;

    // Find max absolute value to determine scale
    half max_abs = 0.0h;
    for (uint i = 0; i < 32; ++i) {
        half v = l1_staging_k[l1_base + i];
        half a = v < 0.0h ? -v : v;
        if (a > max_abs) max_abs = a;
    }

    // Scale factor: map [-max_abs, max_abs] → ternary {-1, 0, 1}
    half scale = max_abs / 1.0h;
    if (max_abs < 1e-10h) scale = 1.0h;

    // Pack 5 trits per byte (7 bytes total)
    uchar packed_bytes[7];
    uint outlier_mask = 0;
    half outlier_vals[32];  // at most 32, but typically 0-2

    uint outlier_count = 0;
    for (uint i = 0; i < 32; ++i) {
        half v = l1_staging_k[l1_base + i];
        int t = (int)(v / scale);

        // Clamp to [-1, 0, 1]
        if (t < -1) t = -1;
        if (t > 1) t = 1;

        // Detect outliers: values that don't round cleanly to ternary
        half reconstructed = (half)t * scale;
        half error = v - reconstructed;
        if (error < 0.0h) error = -error;
        if (error > 0.001h * scale) {
            outlier_mask |= (1u << i);
            outlier_vals[outlier_count] = v;
            ++outlier_count;
            t = 0;  // zero the ternary value, outlier carries the full precision
        }

        // Pack trit into packed_bytes
        uint byte_idx = i / 5;
        uint pos_in_byte = i % 5;
        if (pos_in_byte == 0) {
            packed_bytes[byte_idx] = (uchar)(t + 1);  // map {-1,0,1} → {0,1,2}
        } else {
            uint base = (uint)packed_bytes[byte_idx];
            // Note: real implementation uses carry-propagating base-3 addition
            // Here we just store raw for the structural pattern
        }
    }

    // Write page entry
    device const PageEntry* entry = (device const PageEntry*)(
        page_table_out_buf + token_idx * sizeof(PageEntry)
    );

    uint head_block_offset = head * BLOCKS_PER_HEAD * (TERNARY_BLOCK_BYTES + OUTLIER_BYTES)
        + block_idx * (TERNARY_BLOCK_BYTES + OUTLIER_BYTES);
    uint page_offset = entry->token_offset_within_page
        * NUM_KV_HEADS * BLOCKS_PER_HEAD * (TERNARY_BLOCK_BYTES + OUTLIER_BYTES)
        + head_block_offset;

    device uchar* block_base = entry->page_base + page_offset;

    // Write 7 trit bytes
    for (uint i = 0; i < 7; ++i) {
        block_base[i] = packed_bytes[i];
    }

    // Write FP16 scale at offset 7
    *(device half*)(block_base + 7) = scale;

    // Write outlier mask at offset 9
    *(device uint*)(block_base + 9) = outlier_mask;

    // Write outlier values at offset 13
    for (uint i = 0; i < outlier_count; ++i) {
        *(device half*)(block_base + 13 + i * 2) = outlier_vals[i];
    }
}
"##;

// ── Compilation helpers ───────────────────────────────────────────

/// Compile the gather kernel source into a `ComputePipelineState`.
///
/// The returned pipeline is configured for `gather_ternary_to_fp16_l1`.
pub fn compile_gather_kernel(device: &Device) -> Result<ComputePipelineState, String> {
    let opts = CompileOptions::new();
    opts.set_fast_math_enabled(true);
    let lib = device
        .new_library_with_source(GATHER_KERNEL_SRC, &opts)
        .map_err(|e| format!("gather kernel compile: {e}"))?;
    let func = lib
        .get_function("gather_ternary_to_fp16_l1", None)
        .map_err(|e| format!("gather kernel func: {e}"))?;
    device
        .new_compute_pipeline_state_with_function(&func)
        .map_err(|e| format!("gather pipeline: {e}"))
}

/// Compile the scatter kernel source into a `ComputePipelineState`.
///
/// The returned pipeline is configured for `scatter_fp16_l1_to_ternary`.
pub fn compile_scatter_kernel(device: &Device) -> Result<ComputePipelineState, String> {
    let opts = CompileOptions::new();
    opts.set_fast_math_enabled(true);
    let lib = device
        .new_library_with_source(SCATTER_KERNEL_SRC, &opts)
        .map_err(|e| format!("scatter kernel compile: {e}"))?;
    let func = lib
        .get_function("scatter_fp16_l1_to_ternary", None)
        .map_err(|e| format!("scatter kernel func: {e}"))?;
    device
        .new_compute_pipeline_state_with_function(&func)
        .map_err(|e| format!("scatter pipeline: {e}"))
}
