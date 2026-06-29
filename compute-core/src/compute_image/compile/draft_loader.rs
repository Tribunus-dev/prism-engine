#![cfg_attr(not(target_os = "macos"), allow(dead_code))]

//! Draft model weight loader for DSpark checkpoints.
//!
//! Reads the 5-layer DSpark draft model from HuggingFace SafeTensors format
//! and packs into our ternary 5-per-byte fused interleaved buffer.
//!
//! The draft model is a small speculative decoder (100M params) that shares
//! Gemma 4's embedding and LM head.  We extract only the 5 transformer layers'
//! Q/K/V/O/Gate/Up/Down projections and pack them into `draft_fused_w` for
//! the Metal speculator kernel.

use crate::compute_image::compile::int4_pack;
use std::path::Path;
/// Log target for draft loader diagnostics.
const LOG_TARGET: &str = "draft_loader";

// ── Architecture constants matching DSpark's Gemma 4 12B draft model ──
// These mirror the MSL constants in megakernel/kernels.msl.

/// Number of draft transformer layers loaded from checkpoint.
pub const DRAFT_LAYERS: usize = 5;

/// Hidden dimension.
pub const DRAFT_HIDDEN: usize = 768;

/// Number of query heads (dense MHA).
pub const DRAFT_NUM_HEADS: usize = 8;

/// Number of KV heads (GQA 2:1).
pub const DRAFT_NUM_KV_HEADS: usize = 4;

/// Head dimension = HIDDEN / NUM_HEADS.
pub const DRAFT_HEAD_DIM: usize = DRAFT_HIDDEN / DRAFT_NUM_HEADS; // 96

/// FFN intermediate dimension (Swish-GLU, no expansion factor).
pub const DRAFT_FFN_INTER: usize = 2048;

// ── Per-layer element counts (number of f32 weights per matrix) ──

/// Q projection: (HIDDEN x HIDDEN) = 768x768.
pub const DRAFT_Q_SIZE: usize = DRAFT_HIDDEN * DRAFT_HIDDEN;

/// K projection: (HIDDEN x KV_HEADS * HEAD_DIM) = 768x384.
pub const DRAFT_K_SIZE: usize = DRAFT_HIDDEN * (DRAFT_NUM_KV_HEADS * DRAFT_HEAD_DIM);

/// V projection: same shape as K.
pub const DRAFT_V_SIZE: usize = DRAFT_K_SIZE;

/// O projection: (HEADS * HEAD_DIM x HIDDEN) = 768x768.
pub const DRAFT_O_SIZE: usize = (DRAFT_NUM_HEADS * DRAFT_HEAD_DIM) * DRAFT_HIDDEN;

/// Gate projection: (HIDDEN x FFN_INTER) = 768x2048.
pub const DRAFT_GATE_SIZE: usize = DRAFT_HIDDEN * DRAFT_FFN_INTER;

/// Up projection: same as gate.
pub const DRAFT_UP_SIZE: usize = DRAFT_GATE_SIZE;

/// Down projection: (FFN_INTER x HIDDEN) = 2048x768.
pub const DRAFT_DOWN_SIZE: usize = DRAFT_FFN_INTER * DRAFT_HIDDEN;

// ── Row counts for fused-interleave tile layout ──

/// Rows of Q = HIDDEN (out_features of q_proj).
pub const DRAFT_Q_ROWS: usize = DRAFT_HIDDEN;

/// Rows of K and V = NUM_KV_HEADS * HEAD_DIM.
pub const DRAFT_KV_ROWS: usize = DRAFT_NUM_KV_HEADS * DRAFT_HEAD_DIM;

/// Rows of O = HIDDEN.
pub const DRAFT_O_ROWS: usize = DRAFT_HIDDEN;

/// Rows of Gate and Up = FFN_INTER.
pub const DRAFT_HID_ROWS: usize = DRAFT_FFN_INTER;

/// Rows of Down = HIDDEN.
pub const DRAFT_FFN_ROWS: usize = DRAFT_HIDDEN;

/// Expected tensor name prefix for draft-transformer layers.
const DRAFT_TENSOR_PREFIX: &str = "draft_model.layers";

// ─── Public API ─────────────────────────────────────────────────────────

/// Load draft model weights from a directory containing HuggingFace checkpoint shards.
///
/// Expected files:
/// - `model.safetensors` (single file) or
/// - `model-00001-of-NNNNN.safetensors`, … (sharded)
/// - `config.json`
///
/// State dict keys follow the DFlash/DSpark naming convention:
/// - `draft_model.layers.{i}.self_attn.q_proj.weight`
/// - `draft_model.layers.{i}.self_attn.k_proj.weight`
/// - `draft_model.layers.{i}.self_attn.v_proj.weight`
/// - `draft_model.layers.{i}.self_attn.o_proj.weight`
/// - `draft_model.layers.{i}.mlp.gate_proj.weight`
/// - `draft_model.layers.{i}.mlp.up_proj.weight`
/// - `draft_model.layers.{i}.mlp.down_proj.weight`
///
/// Returns a single fused ternary buffer containing all 5 layers' packed weights,
/// suitable for uploading directly into `draft_fused_w` on the Metal side.
pub fn load_draft_weights(ckpt_dir: &Path) -> Result<Vec<u8>, String> {
    // 1. Discover safetensors shards.
    let shard_paths = collect_safetensors(ckpt_dir)?;
    if shard_paths.is_empty() {
        return Err(format!(
            "no safetensors files found in {}",
            ckpt_dir.display()
        ));
    }

    // 2. Load and pack each layer individually, then concatenate.
    let mut fused_all = Vec::new();

    for layer_idx in 0..DRAFT_LAYERS {
        let layer_fused = load_and_pack_layer(
            layer_idx,
            &shard_paths,
        )?;
        fused_all.extend_from_slice(&layer_fused);
    }

    Ok(fused_all)
}

/// Pack a single contiguous f32 weight matrix into fused-ternary-block format.
///
/// Arguments:
/// - `f32_weights`: flat f32 slice, row-major, `rows * cols` elements.
/// - `rows`: number of output rows (first dimension of the weight matrix).
/// - `cols`: number of input columns (second dimension of the weight matrix).
///
/// Returns serialized `TernaryBlock32` blocks: `rows * ceil(cols/32) * 9` bytes.
/// Caller must then fuse-interleave the 7 matrices for the layer via
/// `interleave_fused_ternary_layer`.
pub fn pack_matrix_to_fused_ternary(f32_weights: &[f32], rows: usize, cols: usize) -> Vec<u8> {
    let blocks_per_row = (cols + 31) / 32;
    let bytes_per_row = blocks_per_row * 9;
    let mut out = vec![0u8; rows * bytes_per_row];

    for r in 0..rows {
        for b in 0..blocks_per_row {
            let start = r * cols + b * 32;
            let mut block_weights = [0.0f32; 32];
            for i in 0..32 {
                let idx = start + i;
                block_weights[i] = if idx < f32_weights.len() {
                    f32_weights[idx]
                } else {
                    0.0
                };
            }
            let ternary = int4_pack::quantize_to_ternary_block32(&block_weights);
            let block_offset = r * bytes_per_row + b * 9;
            out[block_offset..block_offset + 7]
                .copy_from_slice(&ternary.packed_trits);
            out[block_offset + 7..block_offset + 9]
                .copy_from_slice(&ternary.block_scale.to_le_bytes());
        }
    }
    out
}

/// Load all 7 weight tensors for one draft layer, pack them to ternary, and
/// fuse-interleave into one contiguous buffer.
fn load_and_pack_layer(
    layer_idx: usize,
    shards: &[(std::path::PathBuf, Vec<u8>)],
) -> Result<Vec<u8>, String> {
    // All 7 projection names for this layer.
    let proj_names = [
        format!("{layer_idx}.self_attn.q_proj.weight"),
        format!("{layer_idx}.self_attn.k_proj.weight"),
        format!("{layer_idx}.self_attn.v_proj.weight"),
        format!("{layer_idx}.self_attn.o_proj.weight"),
        format!("{layer_idx}.mlp.gate_proj.weight"),
        format!("{layer_idx}.mlp.up_proj.weight"),
        format!("{layer_idx}.mlp.down_proj.weight"),
    ];

    // Row counts for each matrix (output dimension = rows).
    let row_counts: [usize; 7] = [
        DRAFT_Q_ROWS,    // Q
        DRAFT_KV_ROWS,   // K
        DRAFT_KV_ROWS,   // V
        DRAFT_O_ROWS,    // O
        DRAFT_HID_ROWS,  // Gate
        DRAFT_HID_ROWS,  // Up
        DRAFT_FFN_ROWS,  // Down
    ];

    // Column counts (input dimension = cols).
    let col_counts: [usize; 7] = [
        DRAFT_HIDDEN,           // Q
        DRAFT_HIDDEN,           // K
        DRAFT_HIDDEN,           // V
        DRAFT_HIDDEN,           // O
        DRAFT_HIDDEN,           // Gate
        DRAFT_HIDDEN,           // Up
        DRAFT_FFN_INTER,        // Down
    ];

    let mut packed_matrices: [Vec<u8>; 7] = Default::default();

    for (((name, rows), cols), dst) in proj_names
        .iter()
        .zip(row_counts.iter())
        .zip(col_counts.iter())
        .zip(packed_matrices.iter_mut())
    {
        let full_key = format!("{DRAFT_TENSOR_PREFIX}.{name}");
        let f32_weights = load_tensor_f32(&full_key, shards)?;

        // Validate element count matches expectation.
        let expected_len = rows * cols;
        if f32_weights.len() != expected_len {
            return Err(format!(
                "tensor '{full_key}' has {} elements, expected {expected_len}",
                f32_weights.len()
            ));
        }

        *dst = pack_matrix_to_fused_ternary(&f32_weights, *rows, *cols);
    }

    // Unpack the 7 packed matrices for the fuse-interleave call.
    let q = &packed_matrices[0];
    let k = &packed_matrices[1];
    let v = &packed_matrices[2];
    let o = &packed_matrices[3];
    let gate = &packed_matrices[4];
    let up = &packed_matrices[5];
    let down = &packed_matrices[6];

    let fused = int4_pack::interleave_fused_ternary_layer(
        q, k, v, o, gate, up, down,
        DRAFT_Q_ROWS,
        DRAFT_KV_ROWS,
        DRAFT_O_ROWS,
        DRAFT_HID_ROWS,
        DRAFT_FFN_ROWS,
    );

    Ok(fused)
}

// ─── SafeTensors helpers ────────────────────────────────────────────────

/// Collect all safetensors files in a directory, read them into memory, and
/// return sorted (path, buffer) pairs for deterministic ordering.
fn collect_safetensors(dir: &Path) -> Result<Vec<(std::path::PathBuf, Vec<u8>)>, String> {
    let mut shards = Vec::new();

    for entry in std::fs::read_dir(dir).map_err(|e| {
        format!("reading directory {}: {e}", dir.display())
    })? {
        let entry = entry.map_err(|e| format!("dir entry: {e}"))?;
        let path = entry.path();
        if path
            .extension()
            .map_or(false, |ext| ext == "safetensors")
        {
            let data = std::fs::read(&path)
                .map_err(|e| format!("reading {}: {e}", path.display()))?;
            shards.push((path, data));
        }
    }

    // Sort by filename for deterministic shard ordering.
    shards.sort_by(|(a, _), (b, _)| a.file_name().cmp(&b.file_name()));

    Ok(shards)
}

/// Load a single named tensor from the shards, convert to f32, and return.
///
/// Searches each shard in order; returns the first match.  Handles both
/// `safetensors::Dtype::F32` and `BF16` formats.
fn load_tensor_f32(
    key: &str,
    shards: &[(std::path::PathBuf, Vec<u8>)],
) -> Result<Vec<f32>, String> {
    for (shard_path, data) in shards {
        let tensors = safetensors::SafeTensors::deserialize(data)
            .map_err(|e| {
                format!("parsing safetensors {}: {e:?}", shard_path.display())
            })?;

        if let Ok(view) = tensors.tensor(key) {
            return Ok(tensor_data_to_f32(view.data(), view.dtype()));
        }
    }

    Err(format!(
        "tensor '{key}' not found in any of {} shard(s)",
        shards.len()
    ))
}

/// Convert raw safetensors buffer bytes to `Vec<f32>`, handling F32 and BF16.
fn tensor_data_to_f32(data: &[u8], dtype: safetensors::Dtype) -> Vec<f32> {
    match dtype {
        safetensors::Dtype::F32 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        safetensors::Dtype::BF16 => data
            .chunks_exact(2)
            .map(|c| {
                let u = u16::from_le_bytes([c[0], c[1]]);
                f32::from_bits((u as u32) << 16)
            })
            .collect(),
        other => {
            // Fallback: treat unknown as F32 (will panic if not 4-byte-aligned).
            eprintln!(
                "[{LOG_TARGET}] unsupported safetensors dtype {other:?}, treating as F32"
            );
            data.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        }
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Create synthetic row-major f32 weights for a matrix of given shape.
    /// Each weight is (row_idx * cols + col_idx + 1) mod 5 - 2 so values are
    /// in [-2, -1, 0, 1, 2], giving the ternary quantizer something to work
    /// with.
    fn synth_weights(rows: usize, cols: usize) -> Vec<f32> {
        let n = rows * cols;
        (0..n)
            .map(|i| (((i as i32) % 5) - 2) as f32)
            .collect()
    }

    #[test]
    fn test_pack_small_matrix_roundtrip() {
        // Small 4×16 matrix: 4 rows, 16 cols, 1 block per row.
        let rows = 4usize;
        let cols = 16usize;
        let weights = synth_weights(rows, cols);

        let packed = pack_matrix_to_fused_ternary(&weights, rows, cols);

        // 4 rows × 1 block × 9 bytes = 36 bytes.
        let blocks_per_row = (cols + 31) / 32; // 1
        let bytes_per_row = blocks_per_row * 9; // 9
        assert_eq!(
            packed.len(),
            rows * bytes_per_row,
            "packed size should be {rows} × {bytes_per_row}"
        );

        // Verify each block has a nonzero scale (indicates correct quantization).
        for r in 0..rows {
            let block_offset = r * bytes_per_row;
            let scale_bytes: [u8; 2] = [
                packed[block_offset + 7],
                packed[block_offset + 8],
            ];
            let scale = u16::from_le_bytes(scale_bytes);
            let scale_f32 = half::f16::from_bits(scale).to_f32();
            assert!(
                scale_f32 > 0.0,
                "block {r} scale should be positive, got {scale_f32}"
            );
        }

        // Re-extract the block scale and packed trits, then verify we can
        // reconstruct a reasonable approximation.  Since we synthesized
        // {-2..+2} weights and quantize to {-1,0,1}, the sign and relative
        // magnitude should be preserved.
        for r in 0..rows {
            let block_offset = r * bytes_per_row;

            // Read back the 7 packed bytes.
            let mut packed_trits = [0u8; 7];
            packed_trits.copy_from_slice(&packed[block_offset..block_offset + 7]);

            // Read back the scale.
            let scale_bytes: [u8; 2] = [
                packed[block_offset + 7],
                packed[block_offset + 8],
            ];
            let scale_bits = u16::from_le_bytes(scale_bytes);
            let scale = half::f16::from_bits(scale_bits).to_f32();

            // Unpack all 32 trits (we only wrote 16, trailing 16 are pad zeros).
            let mut all_trits = [0i8; 32];
            for byte_idx in 0..6 {
                let mut d5 = [0u8; 5];
                int4_pack::unpack_byte_5_trits(packed_trits[byte_idx], &mut d5);
                for (j, &digit) in d5.iter().enumerate() {
                    // digit 0→-1, 1→0, 2→+1
                    all_trits[byte_idx * 5 + j] = (digit as i8) - 1;
                }
            }
            // Last byte: 2 trits.
            {
                let last_byte = packed_trits[6];
                all_trits[30] = ((last_byte % 3) as i8) - 1;
                all_trits[31] = ((last_byte / 3) as i8) - 1;
            }

            // Verify sign agreement for the first 16 elements (original data).
            for i in 0..16 {
                let orig = weights[r * cols + i];
                let recon = (all_trits[i] as f32) * scale;
                // Sign must match (or weight is zero).
                if orig.abs() > 0.01 {
                    assert!(
                        (recon * orig) >= -1e-6,
                        "row {r}, col {i}: orig={orig}, recon={recon}, sign mismatch"
                    );
                }
            }
        }
    }

    #[test]
    fn test_fused_interleave_draft_layer() {
        // Produce a full layer worth of weights and verify the fused layout
        // has the expected byte counts and tile structure.
        let make_layer_weights = || -> [Vec<u8>; 7] {
            let q = pack_matrix_to_fused_ternary(
                &synth_weights(DRAFT_Q_ROWS, DRAFT_HIDDEN),
                DRAFT_Q_ROWS,
                DRAFT_HIDDEN,
            );
            let k = pack_matrix_to_fused_ternary(
                &synth_weights(DRAFT_KV_ROWS, DRAFT_HIDDEN),
                DRAFT_KV_ROWS,
                DRAFT_HIDDEN,
            );
            let v = pack_matrix_to_fused_ternary(
                &synth_weights(DRAFT_KV_ROWS, DRAFT_HIDDEN),
                DRAFT_KV_ROWS,
                DRAFT_HIDDEN,
            );
            let o = pack_matrix_to_fused_ternary(
                &synth_weights(DRAFT_O_ROWS, DRAFT_HIDDEN),
                DRAFT_O_ROWS,
                DRAFT_HIDDEN,
            );
            let gate = pack_matrix_to_fused_ternary(
                &synth_weights(DRAFT_HID_ROWS, DRAFT_HIDDEN),
                DRAFT_HID_ROWS,
                DRAFT_HIDDEN,
            );
            let up = pack_matrix_to_fused_ternary(
                &synth_weights(DRAFT_HID_ROWS, DRAFT_HIDDEN),
                DRAFT_HID_ROWS,
                DRAFT_HIDDEN,
            );
            let down = pack_matrix_to_fused_ternary(
                &synth_weights(DRAFT_FFN_ROWS, DRAFT_FFN_INTER),
                DRAFT_FFN_ROWS,
                DRAFT_FFN_INTER,
            );
            [q, k, v, o, gate, up, down]
        };

        let [q, k, v, o, gate, up, down] = make_layer_weights();

        let fused = int4_pack::interleave_fused_ternary_layer(
            &q, &k, &v, &o, &gate, &up, &down,
            DRAFT_Q_ROWS,
            DRAFT_KV_ROWS,
            DRAFT_O_ROWS,
            DRAFT_HID_ROWS,
            DRAFT_FFN_ROWS,
        );

        // Compute expected tile count.
        let q_tiles = (DRAFT_Q_ROWS + 31) / 32;    // 24
        let kv_tiles = (DRAFT_KV_ROWS + 31) / 32;  // 12
        let hid_tiles = (DRAFT_HID_ROWS + 31) / 32; // 64
        let ffn_tiles = (DRAFT_FFN_ROWS + 31) / 32; // 24
        let max_tiles = q_tiles.max(kv_tiles).max(hid_tiles).max(ffn_tiles); // 64

        let sub_tile: usize = 180; // 20 blocks × 9 bytes
        let fused_tile = 7 * sub_tile; // 1260

        assert_eq!(
            fused.len(),
            max_tiles * fused_tile,
            "fused buffer size: {max_tiles} tiles × {fused_tile} bytes"
        );

        // Spot-check: first bytes of each matrix slot in tile 0.
        // (These are packed trit bytes; just check they are non-zero since our
        // synthetic weights produce nonzero ternary quanta.)
        for m in 0..7 {
            let pos = m * sub_tile;
            // At least one of the first 7 bytes is nonzero if the matrix has data.
            let slice = &fused[pos..pos + 7];
            let has_any_nonzero = slice.iter().any(|&b| b != 0);
            assert!(
                has_any_nonzero,
                "matrix {m} in tile 0 has all-zero packed trits"
            );
        }

        // Verify tile positions beyond each matrix's row count are zero-filled.
        // K/V have only 12 tiles (384/32), so tile 13 should be zero for them.
        for t in kv_tiles..max_tiles {
            let base = t * fused_tile;
            let k_pos = base + sub_tile;   // matrix 1 = K
            let v_pos = base + 2 * sub_tile; // matrix 2 = V
            for pos in [k_pos, v_pos] {
                assert_eq!(
                    fused[pos..pos + sub_tile].iter().all(|&b| b == 0),
                    true,
                    "tile {t}: K/V beyond kv_tiles should be zero-padded"
                );
            }
        }
    }

    #[test]
    fn test_load_nonexistent_dir() {
        let result = load_draft_weights(Path::new("/nonexistent/path"));
        assert!(result.is_err(), "should error on missing directory");
    }

    #[test]
    fn test_pack_padding_handles_partial_block() {
        // Matrix with cols=33 (one full block + 1 element in second block).
        let rows = 2usize;
        let cols = 33usize;
        let weights: Vec<f32> = (0..rows * cols).map(|i| ((i as i32) % 7 - 3) as f32).collect();

        let packed = pack_matrix_to_fused_ternary(&weights, rows, cols);

        let blocks_per_row = (cols + 31) / 32; // 2
        let bytes_per_row = blocks_per_row * 9; // 18
        assert_eq!(packed.len(), rows * bytes_per_row);

        // The second block has one real element + 31 pad zeros.  Verify the
        // scale is nonzero (the one real element drives the quantizer).
        let scale_bytes: [u8; 2] = [packed[9 + 7], packed[9 + 8]];
        let scale = half::f16::from_bits(u16::from_le_bytes(scale_bytes)).to_f32();
        assert!(scale > 0.0, "partial block scale should be positive");
    }

    #[test]
    fn test_tensor_data_to_f32_f32() {
        let data: Vec<u8> = 1.0f32
            .to_le_bytes()
            .into_iter()
            .chain((-2.5f32).to_le_bytes())
            .chain(3.14f32.to_le_bytes())
            .collect();
        let result = tensor_data_to_f32(&data, safetensors::Dtype::F32);
        assert_eq!(result.len(), 3);
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] - (-2.5)).abs() < 1e-6);
        assert!((result[2] - 3.14).abs() < 1e-6);
    }

    #[test]
    fn test_tensor_data_to_f32_bf16() {
        // BF16(1.0) = u16(0x3F80), BF16(-2.0) = u16(0xC000)
        let mut data = Vec::new();
        data.extend_from_slice(&u16::to_le_bytes(0x3F80u16));
        data.extend_from_slice(&u16::to_le_bytes(0xC000u16));
        data.extend_from_slice(&u16::to_le_bytes(0x4049u16)); // ~3.14 BF16

        let result = tensor_data_to_f32(&data, safetensors::Dtype::BF16);
        assert_eq!(result.len(), 3);
        assert!((result[0] - 1.0).abs() < 0.01, "expected ~1.0, got {}", result[0]);
        assert!((result[1] - (-2.0)).abs() < 0.02, "expected -2.0, got {}", result[1]);
        assert!((result[2] - 3.14).abs() < 0.02, "expected ~3.14, got {}", result[2]);
    }

    #[test]
    fn test_constant_consistency() {
        // Verify derived constants are internally consistent.
        assert_eq!(DRAFT_HEAD_DIM, 96);
        assert_eq!(DRAFT_Q_SIZE, DRAFT_HIDDEN * DRAFT_HIDDEN);
        assert_eq!(
            DRAFT_K_SIZE,
            DRAFT_HIDDEN * (DRAFT_NUM_KV_HEADS * DRAFT_HEAD_DIM)
        );
        assert_eq!(DRAFT_V_SIZE, DRAFT_K_SIZE);
        assert_eq!(DRAFT_O_SIZE, DRAFT_HIDDEN * DRAFT_HIDDEN);
        assert_eq!(DRAFT_GATE_SIZE, DRAFT_HIDDEN * DRAFT_FFN_INTER);
        assert_eq!(DRAFT_UP_SIZE, DRAFT_GATE_SIZE);
        assert_eq!(DRAFT_DOWN_SIZE, DRAFT_FFN_INTER * DRAFT_HIDDEN);
    }
}
