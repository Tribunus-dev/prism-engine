//! NPU weight pump trait — converts canonical ternary tile640 weights
//! into each NPU backend's native format.
//!
//! # Canonical weight format (tile640)
//!
//! Ternary weights are stored in the cimage as tile640 u32 packs:
//!   - Each u32 encodes 20 ternary digits (base-3): digit 0 → 0, 1 → +1, 2 → -1
//!   - Layout: `rows × ceil(cols / 640) × 32 × 4` bytes
//!   - Block scales: FP16, one per 256-element block (`ceil(rows * cols / 256) * 2` bytes)
//!
//! Each NPU backend implements `NpuWeightPump` to convert this canonical
//! format into its native weight layout at runtime.  The conversion runs
//! on the E-core (prefetch thread) and the output lands in memory that
//! the target NPU can directly read (SLC for ANE, SRAM scratchpad for
//! Intel NCE, AIE local tiles for XDNA, HVX registers for HTP, or
//! weight-staging SRAM for Google TPU).
//!
//! # No weight duplication
//!
//! Weights live once in `SegmentKind::TernaryWeights`.  Each NPU backend
//! materialises them into native format on-the-fly via its pump.  There
//! is never a duplicate weight segment for a second architecture.

use crate::compute_image::compile::ternary::{
    repack_ternary_to_swizzled_u8, swizzled_buffer_size, SegmentKind,
};

// ── Shared: tile640 decompression ──────────────────────────────────

/// Decompress tile640 u32-packed ternary weights to row-major f32 with
/// FP16 block scales applied.
///
/// `ternary_bytes` — tile640 u32 packs  (`rows × nt × 32 × 4` bytes).
/// `block_scales`  — FP16 scales, one per 256-element block.  Empty →
///                    unity scale (1.0) for all blocks.
/// `rows`, `cols`  — logical weight matrix dimensions.
///
/// Returns `Vec<f32>` of length `rows × cols` in row-major order.
fn decompress_tile640_to_f32(
    ternary_bytes: &[u8],
    block_scales: &[u8],
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let nt = (cols + 639) / 640; // tiles per row
    let n_vals = rows * cols;
    let n_blocks = (n_vals + 255) / 256;

    // Decode FP16 block scales
    let mut block_scales_f32 = vec![1.0f32; n_blocks];
    let n_scale_bytes = block_scales.len().min(n_blocks * 2);
    for bi in 0..n_scale_bytes / 2 {
        let bits = u16::from_le_bytes([block_scales[bi * 2], block_scales[bi * 2 + 1]]);
        block_scales_f32[bi] = half::f16::from_bits(bits).to_f32();
    }

    let mut out = vec![0.0f32; n_vals];

    // Iterate tile640 layout: row → tile → lane → vi
    for r in 0..rows {
        for t in 0..nt {
            for lane in 0..32 {
                // Byte offset of the u32 pack at (r, t, lane)
                let po = r * nt * 32 * 4 + t * 32 * 4 + lane * 4;
                if po + 4 > ternary_bytes.len() {
                    continue;
                }
                let packed = u32::from_le_bytes([
                    ternary_bytes[po],
                    ternary_bytes[po + 1],
                    ternary_bytes[po + 2],
                    ternary_bytes[po + 3],
                ]);

                // Decode 20 ternary digits from this u32
                let mut v = packed;
                for vi in 0..20 {
                    let col = t * 640 + lane * 20 + vi;
                    if col >= cols {
                        break;
                    }
                    // fast_mod3: v % 3 without hardware division
                    let rem = v - ((v as u64 * 2863311531u64) >> 33) as u32 * 3;
                    let digit = rem as u8; // 0, 1, or 2
                    v = (v as u64 * 2863311531u64 >> 33) as u32; // fast_div3

                    // Map digit to ternary: 0→-1, 1→0, 2→+1
                    let wgt = (digit as i32).wrapping_sub(1);

                    // Apply block scale
                    let val_idx = r * cols + col;
                    let block = val_idx / 256;
                    out[val_idx] = (wgt as f32) * block_scales_f32[block.min(n_blocks - 1)];
                }
            }
        }
    }
    out
}

/// Convert a row-major f32 slice to INT8 with quantization:
///   clamp(round(f32 * 127), -128, 127) → i8 → u8
fn quantize_f32_to_i8(src: &[f32], dst: &mut [u8]) {
    for (i, &v) in src.iter().enumerate() {
        if i >= dst.len() {
            break;
        }
        let q = (v * 127.0).round() as i32;
        let clamped = q.clamp(-128, 127) as i8;
        dst[i] = clamped as u8;
    }
}

/// Convert a single f32 to BF16 (truncate mantissa to 7 bits).
#[inline(always)]
fn f32_to_bf16(v: f32) -> u16 {
    let bits = v.to_bits();
    // Round to nearest even: add 0x7FFF + ((bits >> 16) & 1) then truncate
    let rounding_bias = 0x7FFFu32 + ((bits >> 16) & 1);
    ((bits + rounding_bias) >> 16) as u16
}

// ── Trait ─────────────────────────────────────────────────────────

/// Converts canonical ternary tile640 weights into an NPU-native format.
///
/// # Contract
///
/// - `repack` MUST be safe to call from any thread (the trait is `Send + Sync`).
/// - `output_buffer_size` MUST return the exact byte count needed.
/// - The implementation MUST NOT retain a reference to `ternary_bytes` or
///   `block_scales` after `repack` returns.
pub trait NpuWeightPump: Send + Sync {
    /// Which NPU backend this pump targets (maps to a `SegmentKind`).
    fn target_kind(&self) -> SegmentKind;

    /// Size of the NPU-native output buffer for `rows × cols` weight matrix.
    fn output_buffer_size(&self, rows: usize, cols: usize) -> usize;

    /// Repack tile640 ternary weights + FP16 block scales into NPU-native format.
    ///
    /// `ternary_bytes` — tile640 u32-packed ternary weights.
    /// `block_scales`  — FP16 block scales (1 per 256-element block).
    /// `rows`, `cols`  — logical weight matrix dimensions.
    /// `dst`           — pre-allocated output buffer.
    fn repack(
        &self,
        ternary_bytes: &[u8],
        block_scales: &[u8],
        rows: usize,
        cols: usize,
        dst: &mut [u8],
    );
}

// ── No-op pump (no NPU attached) ──────────────────────────────────

/// Pump that produces zero-length output.  Used when no NPU is present
/// or when weights should not be pumped (e.g. CPU-only execution).
pub struct NopWeightPump;

impl NpuWeightPump for NopWeightPump {
    fn target_kind(&self) -> SegmentKind {
        SegmentKind::TernaryWeights
    }

    fn output_buffer_size(&self, _rows: usize, _cols: usize) -> usize {
        0
    }

    fn repack(
        &self,
        _ternary_bytes: &[u8],
        _block_scales: &[u8],
        _rows: usize,
        _cols: usize,
        _dst: &mut [u8],
    ) {
        // no-op
    }
}

// ── Apple ANE ─────────────────────────────────────────────────────

/// ANE weight pump: tile640 → 16×16 swizzled u8 for the `gather` LUT.
///
/// The ANE's Planar Engine reads swizzled u8 from SLC and expands each
/// quartet of ternary digits (state byte → [81,4] LUT) to INT8 during
/// the matrix multiply.  The scale multiply also happens at gather time.
///
/// Block scales are **not** consumed here — they are embedded into the
/// gather LUT weights at compile time.
pub struct AneWeightPump;

impl NpuWeightPump for AneWeightPump {
    fn target_kind(&self) -> SegmentKind {
        SegmentKind::AneArchive
    }

    fn output_buffer_size(&self, rows: usize, cols: usize) -> usize {
        swizzled_buffer_size(rows, cols)
    }

    fn repack(
        &self,
        ternary_bytes: &[u8],
        block_scales: &[u8],
        rows: usize,
        cols: usize,
        dst: &mut [u8],
    ) {
        let _ = block_scales; // gather LUT handles scaling
        repack_ternary_to_swizzled_u8(ternary_bytes, rows, cols, dst, cols);
    }
}

// ── Intel NPU (NCE) ───────────────────────────────────────────────

/// Intel NCE weight pump: tile640 → INT8 linear buffer.
///
/// The Intel NPU (formerly VPU / NCE) uses INT8 weights in a linear MAC
/// array.  The pump decompresses tile640 ternary digits to INT8, applies
/// block-scale FP16 multipliers, and writes a flat row-major INT8 buffer
/// that the NCE reads via DMA into its 2 MB per-tile scratchpad SRAM.
///
/// Weight layout: linear row-major INT8, `rows × cols` bytes.
///
/// # Pump algorithm
///
/// 1. Decompress tile640 u32 packs → f32 (with block scales applied).
/// 2. Quantize f32 → INT8: `clamp(round(f32 × 127), -128, 127)`.
/// 3. Write flat row-major INT8 to dst.
pub struct IntelNpuWeightPump;

impl NpuWeightPump for IntelNpuWeightPump {
    fn target_kind(&self) -> SegmentKind {
        SegmentKind::IntelNpuBlob
    }

    fn output_buffer_size(&self, rows: usize, cols: usize) -> usize {
        rows * cols
    }

    fn repack(
        &self,
        ternary_bytes: &[u8],
        block_scales: &[u8],
        rows: usize,
        cols: usize,
        dst: &mut [u8],
    ) {
        let f32_vals = decompress_tile640_to_f32(ternary_bytes, block_scales, rows, cols);
        let n = rows * cols;
        let write_len = dst.len().min(n);
        quantize_f32_to_i8(&f32_vals[..write_len], &mut dst[..write_len]);
        // Zero-fill any trailing bytes if dst > rows*cols
        if dst.len() > write_len {
            dst[write_len..].fill(0);
        }
    }
}

// ── AMD XDNA ──────────────────────────────────────────────────────

/// AMD XDNA weight pump: tile640 → BF16 AIE tile layout.
///
/// The AMD XDNA array has AI Engine tiles with 64 KB local L1 memory,
/// sharing up to 4 MB L2 SRAM.  Weights are distributed across the tile
/// grid via DMA (S2MM / MM2S channels) and stored as BF16.
///
/// The pump decompresses tile640 ternary → BF16, applies block scales,
/// and writes a flat row-major BF16 buffer.  The DMA scatter (S2MM
/// channel) handles the AIE tile distribution on the device side.
///
/// Output layout: row-major BF16, `rows × cols × 2` bytes.  Each weight
/// occupies 2 bytes (BF16 = f32 upper 16 bits, round-to-nearest-even).
///
/// # Pump algorithm
///
/// 1. Decompress tile640 u32 packs → f32 (with block scales applied).
/// 2. Convert each f32 to BF16 via round-to-nearest-even truncation.
/// 3. Write flat row-major BF16 (2 bytes per weight) to dst.
pub struct AmdXdnaWeightPump;

impl NpuWeightPump for AmdXdnaWeightPump {
    fn target_kind(&self) -> SegmentKind {
        SegmentKind::AmdNpuBlob
    }

    fn output_buffer_size(&self, rows: usize, cols: usize) -> usize {
        rows * cols * 2
    }

    fn repack(
        &self,
        ternary_bytes: &[u8],
        block_scales: &[u8],
        rows: usize,
        cols: usize,
        dst: &mut [u8],
    ) {
        let f32_vals = decompress_tile640_to_f32(ternary_bytes, block_scales, rows, cols);
        let n = rows * cols;
        let max_out = (dst.len() / 2).min(n);
        for i in 0..max_out {
            let bf16 = f32_to_bf16(f32_vals[i]);
            dst[i * 2..i * 2 + 2].copy_from_slice(&bf16.to_le_bytes());
        }
        // Zero-fill remainder
        if dst.len() > max_out * 2 {
            dst[max_out * 2..].fill(0);
        }
    }
}

// ── Qualcomm Hexagon HTP ──────────────────────────────────────────

/// Qualcomm HTP weight pump: tile640 → INT8 HVX vector layout.
///
/// The Hexagon HTP uses 1024-bit HVX vector registers.  Weights are
/// INT8 and are loaded by the HVX DMA engine into vector-aligned
/// buffers.  The pump decompresses tile640 ternary → INT8, applies
/// block scales, and writes row-major INT8 with total buffer
/// 128-byte alignment (for HVX vector-aligned access).
///
/// Output layout: row-major INT8, aligned to 128 bytes.
///
/// # Pump algorithm
///
/// 1. Decompress tile640 u32 packs → f32 (with block scales applied).
/// 2. Quantize f32 → INT8: `clamp(round(f32 × 127), -128, 127)`.
/// 3. Write row-major INT8; fill padding beyond `rows × cols` with 0.
pub struct QualcommHtpWeightPump;

impl NpuWeightPump for QualcommHtpWeightPump {
    fn target_kind(&self) -> SegmentKind {
        SegmentKind::QualcommNpuBlob
    }

    fn output_buffer_size(&self, rows: usize, cols: usize) -> usize {
        // Total INT8 buffer padded to 128-byte HVX vector alignment
        let linear = rows * cols;
        (linear + 127) & !127
    }

    fn repack(
        &self,
        ternary_bytes: &[u8],
        block_scales: &[u8],
        rows: usize,
        cols: usize,
        dst: &mut [u8],
    ) {
        let f32_vals = decompress_tile640_to_f32(ternary_bytes, block_scales, rows, cols);
        let n = rows * cols;
        let write_len = dst.len().min(n);
        quantize_f32_to_i8(&f32_vals[..write_len], &mut dst[..write_len]);
        // Zero-fill HVX padding
        if dst.len() > write_len {
            dst[write_len..].fill(0);
        }
    }
}

// ── Google TPU ────────────────────────────────────────────────────

/// Google TPU weight pump: tile640 → INT8 systolic tile layout.
///
/// The Google Edge TPU uses a weight-stationary systolic array requiring
/// INT8 weights in 128×8 tile-aligned blocks.  The pump decompresses
/// tile640 ternary → INT8, applies block scales, and arranges the
/// result into 128×8 tiles suitable for the TPU's data-load engine.
///
/// Tile layout: `ceil(rows / 128) × ceil(cols / 8)` tiles.  Each tile
/// is 128 rows × 8 cols = 1024 bytes, stored row-major.
///
/// Within-tile order: row-major, i.e. for tile at (tr, tc) the 1024
/// bytes are [r0c0, r0c1, ..., r0c7, r1c0, ..., r127c7].  Tiles are
/// in row-major order: tile(0,0), tile(0,1), ..., tile(0,Ntc-1),
/// tile(1,0), ...
///
/// # Pump algorithm
///
/// 1. Decompress tile640 u32 packs → f32 (with block scales applied).
/// 2. Quantize f32 → INT8.
/// 3. Scatter into 128×8 tiles in row-major tile order.
pub struct GoogleTpuWeightPump;

impl NpuWeightPump for GoogleTpuWeightPump {
    fn target_kind(&self) -> SegmentKind {
        SegmentKind::GoogleTpuBlob
    }

    fn output_buffer_size(&self, rows: usize, cols: usize) -> usize {
        let tile_rows = (rows + 127) / 128;
        let tile_cols = (cols + 7) / 8;
        tile_rows * tile_cols * 128 * 8
    }

    fn repack(
        &self,
        ternary_bytes: &[u8],
        block_scales: &[u8],
        rows: usize,
        cols: usize,
        dst: &mut [u8],
    ) {
        let f32_vals = decompress_tile640_to_f32(ternary_bytes, block_scales, rows, cols);

        // Temporary INT8 buffer in row-major order
        let n = rows * cols;
        let mut i8_row_major = vec![0u8; n];
        quantize_f32_to_i8(&f32_vals, &mut i8_row_major);

        // Scatter into 128×8 systolic tiles
        let n_tr = (rows + 127) / 128;
        let n_tc = (cols + 7) / 8;
        let tile_stride = 128 * 8; // bytes per tile

        dst.fill(0);

        for tr in 0..n_tr {
            for tc in 0..n_tc {
                let tile_start = (tr * n_tc + tc) * tile_stride;
                let r0 = tr * 128;
                let c0 = tc * 8;
                let r_end = (r0 + 128).min(rows);
                let c_end = (c0 + 8).min(cols);

                // Fill this 128×8 tile
                for ri in r0..r_end {
                    let local_r = ri - r0;
                    for ci in c0..c_end {
                        let local_c = ci - c0;
                        let src_idx = ri * cols + ci;
                        let dst_idx = tile_start + local_r * 8 + local_c;
                        if dst_idx < dst.len() && src_idx < n {
                            dst[dst_idx] = i8_row_major[src_idx];
                        }
                    }
                }
            }
        }
    }
}

// ── Huawei Ascend NPU (DaVinci) ───────────────────────────────────

/// Huawei Ascend NPU weight pump: tile640 → FP16 DaVinci Cube layout.
///
/// The DaVinci Cube unit performs INT8 or FP16 matrix multiplication in
/// 16×16 or 32×32 tiles.  This pump decompresses tile640 ternary → FP16
/// (standard IEEE 16-bit float, not BF16) with block scales applied.
///
/// This is the **raw weight feed** path for direct Cube buffer submission
/// via CANN's `aclrtMalloc` + `aclrtMemcpy`.  For the more common path via
/// compiled OM files (the Huawei analogue of Core ML .mlmodelc), store the
/// .om blob in `SegmentKind::HuaweiAscendBlob` and load directly — no
/// pump needed.
///
/// Output layout: row-major FP16, `rows × cols × 2` bytes.
/// Each weight occupies 2 bytes (IEEE FP16 via `half::f16`).
pub struct HuaweiAscendWeightPump;

impl NpuWeightPump for HuaweiAscendWeightPump {
    fn target_kind(&self) -> SegmentKind {
        SegmentKind::HuaweiAscendBlob
    }

    fn output_buffer_size(&self, rows: usize, cols: usize) -> usize {
        rows * cols * 2
    }

    fn repack(
        &self,
        ternary_bytes: &[u8],
        block_scales: &[u8],
        rows: usize,
        cols: usize,
        dst: &mut [u8],
    ) {
        let f32_vals = decompress_tile640_to_f32(ternary_bytes, block_scales, rows, cols);
        let n = rows * cols;
        let max_out = (dst.len() / 2).min(n);
        for i in 0..max_out {
            let fp16 = half::f16::from_f32(f32_vals[i]);
            dst[i * 2..i * 2 + 2].copy_from_slice(&fp16.to_bits().to_le_bytes());
        }
        if dst.len() > max_out * 2 {
            dst[max_out * 2..].fill(0);
        }
    }
}

// ── Hailo NPU (Hailo-8 / Hailo-15) ────────────────────────────────

/// Hailo NPU weight pump: tile640 → INT8 dataflow buffer.
///
/// The Hailo-8 uses a structured dataflow architecture with dedicated
/// SRAM.  The primary integration path is via compiled HEF files
/// (Hailo Executable Format) produced by the DFC compiler — store the
/// .hef in `SegmentKind::HailoBlob` and load via `libhailort`.
///
/// This pump provides a raw INT8 weight feed for experimental direct
/// buffer submission (bypassing the HEF compiler).  It decompresses
/// tile640 ternary → INT8 with block scales, exactly like the Intel NCE
/// pump.
///
/// Output layout: row-major INT8, `rows × cols` bytes.
pub struct HailoWeightPump;

impl NpuWeightPump for HailoWeightPump {
    fn target_kind(&self) -> SegmentKind {
        SegmentKind::HailoBlob
    }

    fn output_buffer_size(&self, rows: usize, cols: usize) -> usize {
        rows * cols
    }

    fn repack(
        &self,
        ternary_bytes: &[u8],
        block_scales: &[u8],
        rows: usize,
        cols: usize,
        dst: &mut [u8],
    ) {
        let f32_vals = decompress_tile640_to_f32(ternary_bytes, block_scales, rows, cols);
        let n = rows * cols;
        let write_len = dst.len().min(n);
        quantize_f32_to_i8(&f32_vals[..write_len], &mut dst[..write_len]);
        if dst.len() > write_len {
            dst[write_len..].fill(0);
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build tile640 ternary data for a given shape.
    /// Fills each u32 with known digit patterns so the tensor can be
    /// verified after decompression.
    fn make_tile640_ternary(rows: usize, cols: usize) -> Vec<u8> {
        let nt = (cols + 639) / 640;
        let mut buf = vec![0u8; rows * nt * 32 * 4];
        for r in 0..rows {
            for t in 0..nt {
                for lane in 0..32 {
                    let po = r * nt * 32 * 4 + t * 32 * 4 + lane * 4;
                    let mut pk: u32 = 0;
                    for vi in 0..20 {
                        let col = t * 640 + lane * 20 + vi;
                        if col >= cols {
                            break;
                        }
                        // Digit = (row + col) % 3
                        let d: u32 = ((r + col) % 3) as u32;
                        pk += d * 3u32.pow(vi as u32);
                    }
                    buf[po..po + 4].copy_from_slice(&pk.to_le_bytes());
                }
            }
        }
        buf
    }

    /// Helper: build FP16 block scales for a given shape.
    /// Scale = 1.0 for all blocks (unity, identity test).
    fn make_unity_block_scales(rows: usize, cols: usize) -> Vec<u8> {
        let n_blocks = (rows * cols + 255) / 256;
        let mut buf = vec![0u8; n_blocks * 2];
        for bi in 0..n_blocks {
            let bits = half::f16::from_f32(1.0).to_bits();
            buf[bi * 2..bi * 2 + 2].copy_from_slice(&bits.to_le_bytes());
        }
        buf
    }

    /// Verify AneWeightPump output_buffer_size matches swizzled_buffer_size.
    #[test]
    fn test_ane_buffer_size() {
        let pump = AneWeightPump;
        for rows in [1, 16, 32, 64] {
            for cols in [640, 1280, 2560] {
                assert_eq!(
                    pump.output_buffer_size(rows, cols),
                    swizzled_buffer_size(rows, cols),
                    "AneWeightPump buffer size mismatch rows={rows} cols={cols}"
                );
            }
        }
    }

    /// Verify AneWeightPump.repack produces the same output as the free function.
    #[test]
    fn test_ane_pump_equivalence() {
        let rows = 32;
        let cols = 640;
        let ternary = make_tile640_ternary(rows, cols);

        let pump = AneWeightPump;
        let size = pump.output_buffer_size(rows, cols);
        let mut via_trait = vec![0u8; size];
        let mut via_direct = vec![0u8; size];

        pump.repack(&ternary, &[], rows, cols, &mut via_trait);
        repack_ternary_to_swizzled_u8(&ternary, rows, cols, &mut via_direct, cols);

        assert_eq!(
            via_trait, via_direct,
            "AneWeightPump repack must match free function"
        );
    }

    /// Verify Intel buffer size is linear INT8.
    #[test]
    fn test_intel_buffer_size() {
        let pump = IntelNpuWeightPump;
        assert_eq!(pump.output_buffer_size(32, 640), 32 * 640);
        assert_eq!(pump.output_buffer_size(256, 2560), 256 * 2560);
    }

    /// Verify Intel pump produces correct INT8 values.
    #[test]
    fn test_intel_pump_output() {
        let rows = 4;
        let cols = 640;
        let ternary = make_tile640_ternary(rows, cols);
        let scales = make_unity_block_scales(rows, cols);

        let pump = IntelNpuWeightPump;
        let size = pump.output_buffer_size(rows, cols);
        let mut dst = vec![0u8; size];
        pump.repack(&ternary, &scales, rows, cols, &mut dst);

        // With unity scales, digit 0→-1→i8=-1→u8=255, 1→0→0, 2→+1→1
        for r in 0..rows {
            for c in 0..cols {
                let expected_digit = ((r + c) % 3) as u8;
                let expected_i8 = ((expected_digit as i32).wrapping_sub(1)) as i8; // 0→-1, 1→0, 2→+1
                                                                                   // But with scale=1.0 and f32_to_i8: value × 127, round, clamp.
                                                                                   // expected_i8 = expected_f32 * 127
                let expected_f32 = expected_i8 as f32;
                let q = (expected_f32 * 127.0).round() as i32;
                let clamped = q.clamp(-128, 127) as i8;
                let expected = clamped as u8;
                assert_eq!(
                    dst[r * cols + c],
                    expected,
                    "IntelNpuWeightPump mismatch at ({r},{c}): digit={expected_digit}"
                );
            }
        }
    }

    /// Verify AMD buffer size is BF16 (2 bytes per weight).
    #[test]
    fn test_amd_buffer_size() {
        let pump = AmdXdnaWeightPump;
        assert_eq!(pump.output_buffer_size(32, 640), 32 * 640 * 2);
        assert_eq!(pump.output_buffer_size(128, 2560), 128 * 2560 * 2);
    }

    /// Verify AMD pump produces correct BF16 values.
    #[test]
    fn test_amd_pump_output() {
        let rows = 4;
        let cols = 640;
        let ternary = make_tile640_ternary(rows, cols);
        let scales = make_unity_block_scales(rows, cols);

        let pump = AmdXdnaWeightPump;
        let size = pump.output_buffer_size(rows, cols);
        let mut dst = vec![0u8; size];
        pump.repack(&ternary, &scales, rows, cols, &mut dst);

        // Verify each BF16 value matches the decompressed f32 truncated
        let f32_vals = decompress_tile640_to_f32(&ternary, &scales, rows, cols);
        for i in 0..rows * cols {
            let bf16_bits = u16::from_le_bytes([dst[i * 2], dst[i * 2 + 1]]);
            let expected_bf16 = f32_to_bf16(f32_vals[i]);
            assert_eq!(
                bf16_bits, expected_bf16,
                "AmdXdnaWeightPump BF16 mismatch at index {i}: f32={}",
                f32_vals[i]
            );
        }
    }

    /// Verify Qualcomm buffer size is INT8 with 128-byte alignment.
    #[test]
    fn test_qualcomm_buffer_size() {
        let pump = QualcommHtpWeightPump;
        let linear = 32 * 640;
        let aligned = (linear + 127) & !127;
        assert_eq!(pump.output_buffer_size(32, 640), aligned);
        assert_eq!(pump.output_buffer_size(33, 640) % 128, 0);
    }

    /// Verify Qualcomm pump produces correct INT8 values.
    #[test]
    fn test_qualcomm_pump_output() {
        let rows = 4;
        let cols = 640;
        let ternary = make_tile640_ternary(rows, cols);
        let scales = make_unity_block_scales(rows, cols);

        let pump = QualcommHtpWeightPump;
        let size = pump.output_buffer_size(rows, cols);
        let mut dst = vec![0u8; size];
        pump.repack(&ternary, &scales, rows, cols, &mut dst);

        // Verify first rows*cols values match Intel pump
        let intel_pump = IntelNpuWeightPump;
        let mut expected = vec![0u8; rows * cols];
        intel_pump.repack(&ternary, &scales, rows, cols, &mut expected);

        for i in 0..rows * cols {
            assert_eq!(dst[i], expected[i], "QualcommHtpWeightPump mismatch at {i}");
        }
        // Verify padding is zero
        for i in rows * cols..size {
            assert_eq!(dst[i], 0, "QualcommHtpWeightPump padding non-zero at {i}");
        }
    }

    /// Verify TPU buffer size matches 128×8 tile alignment.
    #[test]
    fn test_tpu_buffer_size() {
        let pump = GoogleTpuWeightPump;
        // Exact multiple of 128×8
        let rows = 256;
        let cols = 640;
        let expected = ((rows + 127) / 128) * ((cols + 7) / 8) * 128 * 8;
        assert_eq!(pump.output_buffer_size(rows, cols), expected);
        // Non-aligned
        // rows=100→1 tile, cols=100→13 tiles: 1 × 13 × 1024 = 13312
        assert_eq!(pump.output_buffer_size(100, 100), 1 * 13 * 128 * 8);
    }

    /// Verify TPU pump produces correct systolic tile layout.
    #[test]
    fn test_tpu_pump_output() {
        let rows = 10;
        let cols = 32;
        let ternary = make_tile640_ternary(rows, cols);
        let scales = make_unity_block_scales(rows, cols);

        let pump = GoogleTpuWeightPump;
        let size = pump.output_buffer_size(rows, cols);
        let mut dst = vec![0u8; size];
        pump.repack(&ternary, &scales, rows, cols, &mut dst);

        // Decompress reference row-major INT8
        let f32_vals = decompress_tile640_to_f32(&ternary, &scales, rows, cols);
        let mut ref_row_major = vec![0u8; rows * cols];
        for i in 0..rows * cols {
            let q = (f32_vals[i] * 127.0).round() as i32;
            let clamped = q.clamp(-128, 127) as i8;
            ref_row_major[i] = clamped as u8;
        }

        // Verify each tile location
        let n_tr = (rows + 127) / 128; // 1
        let n_tc = (cols + 7) / 8; // 4
        for tr in 0..n_tr {
            for tc in 0..n_tc {
                let tile_start = (tr * n_tc + tc) * 128 * 8;
                let r0 = tr * 128;
                let c0 = tc * 8;
                for ri in r0..(r0 + 128).min(rows) {
                    for ci in c0..(c0 + 8).min(cols) {
                        let local_r = ri - r0;
                        let local_c = ci - c0;
                        let tile_idx = tile_start + local_r * 8 + local_c;
                        let src_idx = ri * cols + ci;
                        assert_eq!(
                            dst[tile_idx],
                            ref_row_major[src_idx],
                            "TPU tile mismatch at raw ({ri},{ci}), tile ({tr},{tc}), local ({local_r},{local_c})"
                        );
                    }
                }
            }
        }
    }

    /// Verify decompress_tile640_to_f32 roundtrips known digit patterns.
    #[test]
    fn test_decompress_roundtrip() {
        for rows in [1, 4, 8] {
            for cols in [32, 64, 128, 640, 1280] {
                let ternary = make_tile640_ternary(rows, cols);
                let scales = make_unity_block_scales(rows, cols);
                let f32_vals = decompress_tile640_to_f32(&ternary, &scales, rows, cols);

                assert_eq!(f32_vals.len(), rows * cols);

                for r in 0..rows {
                    for c in 0..cols {
                        let expected_digit = ((r + c) % 3) as u8;
                        let expected_wgt = (expected_digit as i32).wrapping_sub(1) as f32;
                        assert!(
                            (f32_vals[r * cols + c] - expected_wgt).abs() < 1e-6,
                            "decompress mismatch at ({r},{c}): expected {expected_wgt}, got {}",
                            f32_vals[r * cols + c]
                        );
                    }
                }
            }
        }
    }

    /// Verify decompress applies block scales correctly.
    #[test]
    fn test_decompress_with_scales() {
        let rows = 2;
        let cols = 640;
        let ternary = make_tile640_ternary(rows, cols);

        // Non-unity block scales: scale[0] = 2.0, rest = 0.5
        let n_blocks = (rows * cols + 255) / 256;
        let mut scales = vec![0u8; n_blocks * 2];
        {
            let bits = half::f16::from_f32(2.0).to_bits();
            scales[0..2].copy_from_slice(&bits.to_le_bytes());
        }
        for bi in 1..n_blocks {
            let bits = half::f16::from_f32(0.5).to_bits();
            scales[bi * 2..bi * 2 + 2].copy_from_slice(&bits.to_le_bytes());
        }

        let f32_vals = decompress_tile640_to_f32(&ternary, &scales, rows, cols);

        for i in 0..rows * cols {
            let block = i / 256;
            let scale = if block == 0 { 2.0 } else { 0.5 };
            let r = i / cols;
            let c = i % cols;
            let digit = ((r + c) % 3) as u8;
            let expected_wgt = ((digit as i32).wrapping_sub(1)) as f32 * scale;
            assert!(
                (f32_vals[i] - expected_wgt).abs() < 1e-6,
                "scale mismatch at {i} (block {block}): expected {expected_wgt}, got {}",
                f32_vals[i]
            );
        }
    }
    /// Verify Huawei buffer size is FP16 (2 bytes per weight).
    #[test]
    fn test_huawei_buffer_size() {
        let pump = HuaweiAscendWeightPump;
        assert_eq!(pump.output_buffer_size(32, 640), 32 * 640 * 2);
        assert_eq!(pump.output_buffer_size(128, 2560), 128 * 2560 * 2);
    }

    /// Verify Hailo buffer size is INT8 linear.
    #[test]
    fn test_hailo_buffer_size() {
        let pump = HailoWeightPump;
        assert_eq!(pump.output_buffer_size(32, 640), 32 * 640);
        assert_eq!(pump.output_buffer_size(128, 2560), 128 * 2560);
    }

    /// Verify Huawei pump produces correct FP16 values.
    #[test]
    fn test_huawei_pump_output() {
        let rows = 4;
        let cols = 640;
        let ternary = make_tile640_ternary(rows, cols);
        let scales = make_unity_block_scales(rows, cols);
        let pump = HuaweiAscendWeightPump;
        let size = pump.output_buffer_size(rows, cols);
        let mut dst = vec![0u8; size];
        pump.repack(&ternary, &scales, rows, cols, &mut dst);
        let f32_vals = decompress_tile640_to_f32(&ternary, &scales, rows, cols);
        for i in 0..rows * cols {
            let fp16_bits = u16::from_le_bytes([dst[i * 2], dst[i * 2 + 1]]);
            let expected = half::f16::from_f32(f32_vals[i]);
            assert_eq!(
                fp16_bits,
                expected.to_bits(),
                "HuaweiAscend FP16 mismatch at {i}: f32={}",
                f32_vals[i]
            );
        }
    }

    /// Verify Hailo pump produces correct INT8 values.
    #[test]
    fn test_hailo_pump_output() {
        let rows = 4;
        let cols = 640;
        let ternary = make_tile640_ternary(rows, cols);
        let scales = make_unity_block_scales(rows, cols);
        let pump = HailoWeightPump;
        let size = pump.output_buffer_size(rows, cols);
        let mut dst = vec![0u8; size];
        pump.repack(&ternary, &scales, rows, cols, &mut dst);
        let intel_pump = IntelNpuWeightPump;
        let mut expected = vec![0u8; rows * cols];
        intel_pump.repack(&ternary, &scales, rows, cols, &mut expected);
        for i in 0..rows * cols {
            assert_eq!(dst[i], expected[i], "HailoWeightPump mismatch at {i}");
        }
    }
}
