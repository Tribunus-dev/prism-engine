//! CImage shard builder — constructs synthetic MLP shards for proof and testing.
//!
//! The shard builder generates deterministic pseudo-random tensors, packs them
//! according to the specified codec policy, and returns a `PendingCImageShard`
//! ready to be written to disk by `CImageWriter`.
//!
//! # Codec support
//!
//! - `RawF32` — pass-through: codes are flat f32 LE bytes, no metadata.
//! - `Nf4` — `pack_nf4_weights` from nf4tile640 (tile640 NF4 format).
//! - `Int8` — `pack_int8_weights` from nf4tile640 (per-tile symmetric INT8).
//! - Other codecs return an error.
//!
//! # Weight layout convention
//!
//! All projection weights (gate_proj, up_proj, down_proj) are **transposed**
//! from the conventional [out_features, in_features] layout to
//! [in_features, out_features] before storage. This is because both
//! `pack_nf4_weights` / `unpack_nf4_weights` operate in [in, out] layout
//! and `run_mlp_rawf32_reference` expects weights in the same transposed
//! form so that `matmul_f32(input, weights, 1, in, out)` computes
//! `input @ W^T` without an explicit transpose.
//!
//! | Tensor      | Conventional shape   | Stored shape            |
//! |-------------|----------------------|-------------------------|
//! | gate_proj   | [intermediate, dim]  | [dim, intermediate]     |
//! | up_proj     | [intermediate, dim]  | [dim, intermediate]     |
//! | down_proj   | [dim, intermediate]  | [intermediate, dim]     |
//! | rmsnorm_w   | [dim]                | [dim] (1-D, untouched)  |

use sha2::{Digest, Sha256};

use crate::cimage::*;
use crate::execution_plan::{CodecFamily, DType, HardwareProfileId};
use crate::nf4tile640::{
    pack_int8_weights, pack_nf4_weights, GROUPS_PER_TILE, GROUP_SIZE, PACKED_BYTES_PER_TILE,
    SCALES_F32_PER_TILE, TILE_ELEMENTS,
};
use crate::ternary::pack::pack_ternary_codes;
use half::f16;

// ─── Data structures ──────────────────────────────────────────────────────

/// Configuration for generating a synthetic MLP shard.
#[derive(Debug, Clone)]
pub struct SyntheticMlpShardConfig {
    /// Seed for the deterministic pseudo-random tensor generator.
    /// Each tensor uses a different seed: rmsnorm_weight = seed,
    /// gate_proj = seed+1, up_proj = seed+2, down_proj = seed+3.
    pub seed: u64,
    /// Hidden dimension (input/output channels for the MLP block).
    pub hidden_dim: usize,
    /// Intermediate dimension (gate/up projection output, down projection input).
    pub intermediate_dim: usize,
    /// Per-tensor codec policy.
    pub policy: SyntheticShardPolicy,
}

/// Per-tensor codec policy for a synthetic shard.
#[derive(Debug, Clone)]
pub struct SyntheticShardPolicy {
    pub gate_codec: CodecFamily,
    pub up_codec: CodecFamily,
    pub down_codec: CodecFamily,
    pub rmsnorm_codec: CodecFamily,
    pub allow_mixed_precision: bool,
}

/// A pending (unwritten) cimage shard, ready for `CImageWriter`.
pub struct PendingCImageShard {
    pub manifest: CImageManifestV0,
    pub payloads: Vec<PendingPayload>,
    pub receipts: Vec<PendingReceipt>,
}

/// Builder for constructing synthetic MLP cimage shards.
pub struct MlpShardBuilder;

// ─── Decoder layer data structures ────────────────────────────────────────

/// Configuration for generating a synthetic decoder layer shard.
#[derive(Debug, Clone)]
pub struct SyntheticDecoderLayerConfig {
    /// Seed for the deterministic pseudo-random tensor generator.
    /// Each tensor uses a different seed offset.
    pub seed: u64,
    /// Hidden dimension (input/output channels for the decoder layer).
    pub hidden_dim: usize,
    /// Number of query attention heads.
    pub num_heads: usize,
    /// Number of key/value attention heads (GQA).
    pub num_kv_heads: usize,
    /// Dimension per attention head.
    pub head_dim: usize,
    /// Intermediate dimension for the MLP sub-layer.
    pub intermediate_dim: usize,
    /// Sequence length (number of tokens).
    pub seq_len: usize,
    /// Per-tensor codec policy.
    pub policy: SyntheticDecoderPolicy,
}

/// Per-tensor codec policy for a synthetic decoder layer shard.
#[derive(Debug, Clone)]
pub struct SyntheticDecoderPolicy {
    /// Codec for Q/K/V/O projection weights.
    pub projection_codec: CodecFamily,
    /// Codec for MLP gate/up/down projection weights.
    pub mlp_codec: CodecFamily,
    /// Codec for layer norm weights.
    pub norm_codec: CodecFamily,
    /// Codec for attention projection weights.
    pub attention_codec: CodecFamily,
}

// ─── Tensor generation ────────────────────────────────────────────────────

/// Generate a deterministic pseudo-random f32 tensor using a simple LCG.
///
/// Values are uniformly distributed in [-1, 1). The LCG is the same as
/// MMIX (Knuth): `state = state * 6364136223846793005 + 1442695040888963407`.
fn deterministic_f32_tensor(seed: u64, shape: &[usize]) -> Vec<f32> {
    let n: usize = shape.iter().product();
    let mut state = seed;
    let mut data = Vec::with_capacity(n);
    for _ in 0..n {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let val = ((state >> 11) as f64) / (1u64 << 53) as f64;
        data.push((val * 2.0 - 1.0) as f32);
    }
    data
}

// ─── Helpers ──────────────────────────────────────────────────────────────

/// Compute SHA-256 hex digest of raw f32 values (little-endian bytes).
fn sha256_of_f32_slice(data: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for &v in data {
        hasher.update(v.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

/// Serialize a slice of f32 values into little-endian bytes.
fn f32_slice_to_le_bytes(data: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for &v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

/// Transpose a row-major matrix from [rows, cols] to [cols, rows].
fn transpose_matrix(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            out[j * rows + i] = data[i * cols + j];
        }
    }
    out
}

/// Build an NF4 PhysicalTileLayout for the given **pack** dimensions
/// (rows = in_features, cols = out_features — i.e. the stored shape).
fn nf4_tile_layout(pack_rows: usize, pack_cols: usize) -> PhysicalTileLayout {
    let tiles_per_row = pack_cols.div_ceil(TILE_ELEMENTS) as u32;
    let total_tiles = (pack_rows as u32) * tiles_per_row;
    let padded_cols = tiles_per_row * (TILE_ELEMENTS as u32);
    PhysicalTileLayout {
        tile_m: 1,
        tile_n: TILE_ELEMENTS as u32,
        tiles_per_row,
        total_tiles,
        padded_cols,
        group_size: GROUP_SIZE as u32,
        groups_per_tile: GROUPS_PER_TILE as u32,
        packed_bytes_per_tile: PACKED_BYTES_PER_TILE as u32,
        metadata_f32_per_tile: (SCALES_F32_PER_TILE * 2) as u32,
    }
}

/// Build an INT8 PhysicalTileLayout.
fn int8_tile_layout(pack_rows: usize, pack_cols: usize) -> PhysicalTileLayout {
    let tiles_per_row = pack_cols.div_ceil(TILE_ELEMENTS) as u32;
    let total_tiles = (pack_rows as u32) * tiles_per_row;
    let padded_cols = tiles_per_row * (TILE_ELEMENTS as u32);
    PhysicalTileLayout {
        tile_m: 1,
        tile_n: TILE_ELEMENTS as u32,
        tiles_per_row,
        total_tiles,
        padded_cols,
        group_size: TILE_ELEMENTS as u32,
        groups_per_tile: 1,
        packed_bytes_per_tile: TILE_ELEMENTS as u32,
        metadata_f32_per_tile: 2,
    }
}

/// Build a RawF32 PhysicalTileLayout — a single flat tile.
fn rawf32_tile_layout(tensor_len: usize) -> PhysicalTileLayout {
    PhysicalTileLayout {
        tile_m: 1,
        tile_n: tensor_len as u32,
        tiles_per_row: 1,
        total_tiles: 1,
        padded_cols: tensor_len as u32,
        group_size: 0,
        groups_per_tile: 0,
        packed_bytes_per_tile: (tensor_len * 4) as u32,
        metadata_f32_per_tile: 0,
    }
}

/// Tile layout for the Ternary1_58 codec — one tile per row, grouped by group_size.
fn ternary_grouped_layout(
    pack_rows: usize,
    pack_cols: usize,
    group_size: usize,
    dtype: DType,
) -> PhysicalTileLayout {
    let groups_per_row = pack_cols.div_ceil(group_size);
    let bytes_per_group = (group_size * 2 + 7) / 8;
    let scale_bytes_per_group = match dtype {
        DType::F16 => 2,
        _ => 4, // f32 fallback
    };
    PhysicalTileLayout {
        tile_m: 1,
        tile_n: pack_cols as u32,
        tiles_per_row: 1,
        total_tiles: pack_rows as u32,
        padded_cols: pack_cols as u32,
        group_size: group_size as u32,
        groups_per_tile: groups_per_row as u32,
        packed_bytes_per_tile: (groups_per_row * bytes_per_group) as u32,
        metadata_f32_per_tile: (groups_per_row * scale_bytes_per_group) as u32,
    }
}

// ─── Builder implementation ───────────────────────────────────────────────

impl MlpShardBuilder {
    /// Build a synthetic MLP shard from the given configuration.
    ///
    /// Generates four deterministic tensors (rmsnorm_weight, gate_proj,
    /// up_proj, down_proj), transposes them to [in_features, out_features]
    /// layout, packs each according to the per-tensor codec policy, and
    /// returns a `PendingCImageShard` with manifest, payloads, and empty
    /// receipts.
    pub fn build_synthetic_mlp_shard(
        config: SyntheticMlpShardConfig,
    ) -> CImageResult<PendingCImageShard> {
        let SyntheticMlpShardConfig {
            seed,
            hidden_dim,
            intermediate_dim,
            policy,
        } = config;

        // 1. Generate deterministic tensors in conventional [out, in] layout.
        let rmsnorm_weight = deterministic_f32_tensor(seed, &[hidden_dim]);
        let gate_proj =
            deterministic_f32_tensor(seed.wrapping_add(1), &[intermediate_dim, hidden_dim]);
        let up_proj =
            deterministic_f32_tensor(seed.wrapping_add(2), &[intermediate_dim, hidden_dim]);
        let down_proj =
            deterministic_f32_tensor(seed.wrapping_add(3), &[hidden_dim, intermediate_dim]);

        // 2. Transpose projections into [in_features, out_features] layout
        //    (the matmul_f32 in mlp_reference reads them transposed).
        let gate_proj_stored = transpose_matrix(&gate_proj, intermediate_dim, hidden_dim);
        let up_proj_stored = transpose_matrix(&up_proj, intermediate_dim, hidden_dim);
        let down_proj_stored = transpose_matrix(&down_proj, hidden_dim, intermediate_dim);

        // 3. Define each tensor: (key, stored_data, conventional_shape, codec, class).
        let tensor_defs: [(&str, &[f32], &[usize], CodecFamily, &str); 4] = [
            (
                "rmsnorm_weight",
                &rmsnorm_weight,
                &[hidden_dim],
                policy.rmsnorm_codec,
                "RmsNormWeight",
            ),
            (
                "gate_proj",
                &gate_proj_stored,
                &[intermediate_dim, hidden_dim],
                policy.gate_codec,
                "DecoderMlpProjection",
            ),
            (
                "up_proj",
                &up_proj_stored,
                &[intermediate_dim, hidden_dim],
                policy.up_codec,
                "DecoderMlpProjection",
            ),
            (
                "down_proj",
                &down_proj_stored,
                &[hidden_dim, intermediate_dim],
                policy.down_codec,
                "DecoderMlpProjection",
            ),
        ];

        let mut all_payloads: Vec<PendingPayload> = Vec::new();
        let mut tensor_entries: Vec<CImageTensorEntry> = Vec::with_capacity(4);

        for (idx, (tensor_key, stored, conv_shape, codec, tensor_class)) in
            tensor_defs.iter().enumerate()
        {
            let tensor_id = format!("t{}", idx);

            // Logical shape is the **conventional** [out_features, in_features].
            let logical_shape: Vec<u32> = if conv_shape.len() == 1 {
                // 1-D rmsnorm: store as [hidden_dim, 1] so that
                // validate_mlp_shard reads tensors[0].logical_shape[0] == hidden_dim.
                vec![conv_shape[0] as u32, 1u32]
            } else {
                conv_shape.iter().map(|&d| d as u32).collect()
            };

            // Pack dimensions: [in_features, out_features] = stored layout.
            let (pack_rows, pack_cols) = if conv_shape.len() == 1 {
                // rmsnorm_weight: [hidden_dim] → treat as [1, hidden_dim] for packing.
                // After transpose (no-op): [hidden_dim, 1] is the stored layout.
                (1usize, conv_shape[0])
            } else {
                // Stored as [in_features, out_features] = [shape[1], shape[0]].
                (conv_shape[1], conv_shape[0])
            };

            // SHA-256 over the **stored** (transposed) f32 values.
            let tensor_sha256 = sha256_of_f32_slice(stored);

            let (payload_ref, phy_layout, raw_ref) = match codec {
                CodecFamily::RawF32 => {
                    let raw_bytes = f32_slice_to_le_bytes(stored);
                    let codes_payload = PendingPayload {
                        payload_id: format!("p_{}_codes", tensor_key),
                        payload_kind: CImagePayloadKind::PackedTensorCodes,
                        codec: Some("RawF32".into()),
                        alignment_bytes: 64,
                        bytes: raw_bytes.clone(),
                    };
                    let raw_ref_payload = PendingPayload {
                        payload_id: format!("p_{}_rawf32", tensor_key),
                        payload_kind: CImagePayloadKind::RawF32Reference,
                        codec: Some("RawF32".into()),
                        alignment_bytes: 64,
                        bytes: raw_bytes,
                    };
                    all_payloads.push(codes_payload);
                    all_payloads.push(raw_ref_payload);

                    (
                        CImagePayloadRef::Single {
                            payload_id: format!("p_{}_codes", tensor_key),
                        },
                        rawf32_tile_layout(stored.len()),
                        Some(CImagePayloadRef::Single {
                            payload_id: format!("p_{}_rawf32", tensor_key),
                        }),
                    )
                }
                CodecFamily::Nf4 => {
                    let (codes, scales, biases, _packed_rows, _packed_cols) =
                        pack_nf4_weights(stored, pack_rows, pack_cols);

                    let codes_payload = PendingPayload {
                        payload_id: format!("p_{}_codes", tensor_key),
                        payload_kind: CImagePayloadKind::PackedTensorCodes,
                        codec: Some("NF4".into()),
                        alignment_bytes: 64,
                        bytes: codes,
                    };

                    // Metadata = scales concatenated with biases (both f32 LE).
                    let mut meta_bytes = Vec::with_capacity((scales.len() + biases.len()) * 4);
                    meta_bytes.extend_from_slice(&f32_slice_to_le_bytes(&scales));
                    meta_bytes.extend_from_slice(&f32_slice_to_le_bytes(&biases));

                    // NOTE: metadata ID must match what extract_packed_and_reconstruct
                    // builds: format!("{}_metadata", payload_id) where payload_id is
                    // "p_{key}_codes" → "p_{key}_codes_metadata".
                    let meta_payload = PendingPayload {
                        payload_id: format!("p_{}_codes_metadata", tensor_key),
                        payload_kind: CImagePayloadKind::TensorMetadata,
                        codec: Some("NF4".into()),
                        alignment_bytes: 64,
                        bytes: meta_bytes,
                    };

                    let raw_f32_payload = PendingPayload {
                        payload_id: format!("p_{}_rawf32", tensor_key),
                        payload_kind: CImagePayloadKind::RawF32Reference,
                        codec: Some("RawF32".into()),
                        alignment_bytes: 64,
                        bytes: f32_slice_to_le_bytes(stored),
                    };

                    all_payloads.push(codes_payload);
                    all_payloads.push(meta_payload);
                    all_payloads.push(raw_f32_payload);

                    let layout = nf4_tile_layout(pack_rows, pack_cols);
                    (
                        CImagePayloadRef::Single {
                            payload_id: format!("p_{}_codes", tensor_key),
                        },
                        layout,
                        Some(CImagePayloadRef::Single {
                            payload_id: format!("p_{}_rawf32", tensor_key),
                        }),
                    )
                }
                CodecFamily::Int8 => {
                    let (codes, scales, biases) = pack_int8_weights(stored, pack_rows, pack_cols);

                    let codes_payload = PendingPayload {
                        payload_id: format!("p_{}_codes", tensor_key),
                        payload_kind: CImagePayloadKind::PackedTensorCodes,
                        codec: Some("INT8".into()),
                        alignment_bytes: 64,
                        bytes: codes,
                    };

                    let mut meta_bytes = Vec::with_capacity((scales.len() + biases.len()) * 4);
                    meta_bytes.extend_from_slice(&f32_slice_to_le_bytes(&scales));
                    meta_bytes.extend_from_slice(&f32_slice_to_le_bytes(&biases));

                    let meta_payload = PendingPayload {
                        payload_id: format!("p_{}_codes_metadata", tensor_key),
                        payload_kind: CImagePayloadKind::TensorMetadata,
                        codec: Some("INT8".into()),
                        alignment_bytes: 64,
                        bytes: meta_bytes,
                    };

                    let raw_f32_payload = PendingPayload {
                        payload_id: format!("p_{}_rawf32", tensor_key),
                        payload_kind: CImagePayloadKind::RawF32Reference,
                        codec: Some("RawF32".into()),
                        alignment_bytes: 64,
                        bytes: f32_slice_to_le_bytes(stored),
                    };

                    all_payloads.push(codes_payload);
                    all_payloads.push(meta_payload);
                    all_payloads.push(raw_f32_payload);

                    let layout = int8_tile_layout(pack_rows, pack_cols);
                    (
                        CImagePayloadRef::Single {
                            payload_id: format!("p_{}_codes", tensor_key),
                        },
                        layout,
                        Some(CImagePayloadRef::Single {
                            payload_id: format!("p_{}_rawf32", tensor_key),
                        }),
                    )
                }
                CodecFamily::Ternary1_58 => {
                    let group_size: usize = 32;
                    let groups_per_row = pack_cols.div_ceil(group_size);

                    // 1. Quantize weights to {-1, 0, +1}.
                    let quantized: Vec<i8> = stored
                        .iter()
                        .map(|&w| {
                            if w > 0.01f32 {
                                1i8
                            } else if w < -0.01f32 {
                                -1i8
                            } else {
                                0i8
                            }
                        })
                        .collect();

                    // 2. Pack ternary codes.
                    let codes = pack_ternary_codes(&quantized)
                        .map_err(|e| CImageError::Other(format!("ternary pack failed: {e}")))?;

                    // 3. Compute per-group scales: sum(|w|) / group_size.
                    let mut scales_f32 = Vec::with_capacity(pack_rows * groups_per_row);
                    for r in 0..pack_rows {
                        for g in 0..groups_per_row {
                            let start = g * group_size;
                            let end = (start + group_size).min(pack_cols);
                            let mut sum_abs = 0.0f32;
                            for c in start..end {
                                sum_abs += stored[r * pack_cols + c].abs();
                            }
                            scales_f32.push(sum_abs / group_size as f32);
                        }
                    }
                    let scale_bytes: Vec<u8> = scales_f32
                        .iter()
                        .flat_map(|&s| f16::from_f32(s).to_le_bytes())
                        .collect();

                    // 4. Emit TernaryPackedCodes payload.
                    let codes_payload = PendingPayload {
                        payload_id: format!("p_{}_codes", tensor_key),
                        payload_kind: CImagePayloadKind::TernaryPackedCodes,
                        codec: Some("Ternary1_58".into()),
                        alignment_bytes: 64,
                        bytes: codes,
                    };

                    // 5. Emit TernaryScales payload.
                    let scales_payload = PendingPayload {
                        payload_id: format!("p_{}_scales", tensor_key),
                        payload_kind: CImagePayloadKind::TernaryScales,
                        codec: Some("Ternary1_58".into()),
                        alignment_bytes: 64,
                        bytes: scale_bytes,
                    };

                    // 6. Emit RawF32Reference alongside.
                    let raw_f32_payload = PendingPayload {
                        payload_id: format!("p_{}_rawf32", tensor_key),
                        payload_kind: CImagePayloadKind::RawF32Reference,
                        codec: Some("RawF32".into()),
                        alignment_bytes: 64,
                        bytes: f32_slice_to_le_bytes(stored),
                    };

                    all_payloads.push(codes_payload);
                    all_payloads.push(scales_payload);
                    all_payloads.push(raw_f32_payload);

                    let layout =
                        ternary_grouped_layout(pack_rows, pack_cols, group_size, DType::F16);

                    (
                        CImagePayloadRef::Single {
                            payload_id: format!("p_{}_codes", tensor_key),
                        },
                        layout,
                        Some(CImagePayloadRef::Single {
                            payload_id: format!("p_{}_rawf32", tensor_key),
                        }),
                    )
                }
                other => {
                    return Err(CImageError::Other(format!(
                        "unsupported codec {:?} for tensor {}",
                        other, tensor_key
                    )));
                }
            };

            let entry = CImageTensorEntry {
                tensor_id,
                tensor_key: tensor_key.to_string(),
                tensor_class: tensor_class.to_string(),
                logical_shape,
                source_dtype: DType::F32,
                codec: *codec,
                precision_plan: None,
                physical_layout: phy_layout,
                payload_ref,
                raw_f32_reference_ref: raw_ref,
                tensor_sha256,
                validation_digest: None,
            };

            tensor_entries.push(entry);
        }

        // 4. Build the execution plan summary.
        let plan_id = format!("synth_mlp_shard_{:016x}", seed);
        let execution_plan = ModelExecutionPlanSummary {
            plan_id: plan_id.clone(),
            region_count: 1,
            total_kernel_ops: 3,
            total_input_bytes: (hidden_dim * 4) as u64,
            total_output_bytes: (hidden_dim * 4) as u64,
            tensor_refs: vec!["t0".into(), "t1".into(), "t2".into(), "t3".into()],
        };

        // 5. Build manifest.
        let manifest = CImageManifestV0 {
            schema_version: 0,
            model_family: "SyntheticMLP".into(),
            artifact_kind: CImageArtifactKind::SyntheticShard,
            source_model_digest: None,
            compiler_policy_digest: "synthetic".into(),
            layout_profile: HardwareProfileId::AppleMProBalanced,
            tensors: tensor_entries,
            execution_plan,
            receipts: Vec::new(),
            assistant_graph: None,
            state_store_schema: None,
        };

        Ok(PendingCImageShard {
            manifest,
            payloads: all_payloads,
            receipts: Vec::new(),
        })
    }
    /// Build a single ternary linear layer cimage shard.
    ///
    /// Generates one weight matrix of shape [rows, cols], quantizes to
    /// Ternary1_58 with the given group_size, and returns a pending shard
    /// with a single tensor entry and its codes / scales / raw-f32 payloads.
    pub fn build_single_ternary_linear(
        rows: usize,
        cols: usize,
        group_size: usize,
        seed: u64,
    ) -> CImageResult<PendingCImageShard> {
        let tensor_key = "linear_weight";
        let tensor_id = "t0";

        // Generate as [cols, rows] then transpose to stored [rows, cols].
        let raw = deterministic_f32_tensor(seed, &[cols, rows]);
        let stored = transpose_matrix(&raw, cols, rows);

        let pack_rows = rows;
        let pack_cols = cols;
        let groups_per_row = pack_cols.div_ceil(group_size);

        // 1. Quantize weights to {-1, 0, +1}.
        let quantized: Vec<i8> = stored
            .iter()
            .map(|&w| {
                if w > 0.01f32 {
                    1i8
                } else if w < -0.01f32 {
                    -1i8
                } else {
                    0i8
                }
            })
            .collect();

        // 2. Pack ternary codes.
        let codes = pack_ternary_codes(&quantized)
            .map_err(|e| CImageError::Other(format!("ternary pack failed: {e}")))?;

        // 3. Compute per-group scales: sum(|w|) / group_size.
        let mut scales_f32 = Vec::with_capacity(pack_rows * groups_per_row);
        for r in 0..pack_rows {
            for g in 0..groups_per_row {
                let start = g * group_size;
                let end = (start + group_size).min(pack_cols);
                let mut sum_abs = 0.0f32;
                for c in start..end {
                    sum_abs += stored[r * pack_cols + c].abs();
                }
                scales_f32.push(sum_abs / group_size as f32);
            }
        }
        let scale_bytes: Vec<u8> = scales_f32
            .iter()
            .flat_map(|&s| f16::from_f32(s).to_le_bytes())
            .collect();

        let tensor_sha256 = sha256_of_f32_slice(&stored);

        // 4. Build payloads.
        let mut all_payloads: Vec<PendingPayload> = Vec::new();

        let codes_payload = PendingPayload {
            payload_id: format!("p_{}_codes", tensor_key),
            payload_kind: CImagePayloadKind::TernaryPackedCodes,
            codec: Some("Ternary1_58".into()),
            alignment_bytes: 64,
            bytes: codes,
        };
        let scales_payload = PendingPayload {
            payload_id: format!("p_{}_scales", tensor_key),
            payload_kind: CImagePayloadKind::TernaryScales,
            codec: Some("Ternary1_58".into()),
            alignment_bytes: 64,
            bytes: scale_bytes,
        };
        let raw_f32_payload = PendingPayload {
            payload_id: format!("p_{}_rawf32", tensor_key),
            payload_kind: CImagePayloadKind::RawF32Reference,
            codec: Some("RawF32".into()),
            alignment_bytes: 64,
            bytes: f32_slice_to_le_bytes(&stored),
        };
        all_payloads.push(codes_payload);
        all_payloads.push(scales_payload);
        all_payloads.push(raw_f32_payload);

        // 5. Build tensor entry.
        let layout = ternary_grouped_layout(pack_rows, pack_cols, group_size, DType::F16);

        let entry = CImageTensorEntry {
            tensor_id: tensor_id.to_string(),
            tensor_key: tensor_key.to_string(),
            tensor_class: "TernaryLinear".to_string(),
            logical_shape: vec![rows as u32, cols as u32],
            source_dtype: DType::F32,
            codec: CodecFamily::Ternary1_58,
            precision_plan: None,
            physical_layout: layout,
            payload_ref: CImagePayloadRef::Single {
                payload_id: format!("p_{}_codes", tensor_key),
            },
            raw_f32_reference_ref: Some(CImagePayloadRef::Single {
                payload_id: format!("p_{}_rawf32", tensor_key),
            }),
            tensor_sha256,
            validation_digest: None,
        };

        // 6. Build execution plan summary.
        let plan_id = format!("synth_ternary_linear_{:016x}", seed);
        let execution_plan = ModelExecutionPlanSummary {
            plan_id: plan_id.clone(),
            region_count: 1,
            total_kernel_ops: 1,
            total_input_bytes: (cols * 4) as u64,
            total_output_bytes: (rows * 4) as u64,
            tensor_refs: vec!["t0".into()],
        };

        // 7. Build manifest.
        let manifest = CImageManifestV0 {
            schema_version: 0,
            model_family: "SyntheticTernaryLinear".into(),
            artifact_kind: CImageArtifactKind::SyntheticShard,
            source_model_digest: None,
            compiler_policy_digest: "synthetic-ternary".into(),
            layout_profile: HardwareProfileId::AppleMProBalanced,
            tensors: vec![entry],
            execution_plan,
            receipts: Vec::new(),
            assistant_graph: None,
            state_store_schema: None,
        };

        Ok(PendingCImageShard {
            manifest,
            payloads: all_payloads,
            receipts: Vec::new(),
        })
    }


}
// ─── Decoder layer builder ─────────────────────────────────────────────────

/// Builder for constructing synthetic decoder layer cimage shards.
pub struct DecoderLayerShardBuilder;

impl DecoderLayerShardBuilder {
    /// Build a synthetic decoder layer shard from the given configuration.
    ///
    /// Generates 10 deterministic tensors (9 weights + position_ids),
    /// transposes projections to [in_features, out_features] storage layout,
    /// packs each according to the per-tensor codec policy, and returns a
    /// `PendingCImageShard` with manifest, payloads, and empty receipts.
    pub fn build_synthetic_decoder_layer(
        config: SyntheticDecoderLayerConfig,
    ) -> CImageResult<PendingCImageShard> {
        let SyntheticDecoderLayerConfig {
            seed,
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_dim,
            seq_len,
            policy,
        } = config;

        // Validate dimensions.
        assert!(
            hidden_dim == num_heads * head_dim,
            "hidden_dim ({}) must equal num_heads ({}) * head_dim ({})",
            hidden_dim,
            num_heads,
            head_dim
        );

        let kv_inner = head_dim * num_kv_heads;

        // 1. Generate deterministic tensors in conventional [out, in] layout.
        let input_layernorm = deterministic_f32_tensor(seed, &[hidden_dim]);
        let q_proj = deterministic_f32_tensor(seed.wrapping_add(1), &[hidden_dim, hidden_dim]);
        let k_proj = deterministic_f32_tensor(seed.wrapping_add(2), &[kv_inner, hidden_dim]);
        let v_proj = deterministic_f32_tensor(seed.wrapping_add(3), &[kv_inner, hidden_dim]);
        let o_proj = deterministic_f32_tensor(seed.wrapping_add(4), &[hidden_dim, hidden_dim]);
        let post_attention_layernorm =
            deterministic_f32_tensor(seed.wrapping_add(5), &[hidden_dim]);
        let gate_proj =
            deterministic_f32_tensor(seed.wrapping_add(6), &[intermediate_dim, hidden_dim]);
        let up_proj =
            deterministic_f32_tensor(seed.wrapping_add(7), &[intermediate_dim, hidden_dim]);
        let down_proj =
            deterministic_f32_tensor(seed.wrapping_add(8), &[hidden_dim, intermediate_dim]);

        // 2. Transpose projections into [in_features, out_features] layout.
        let q_proj_stored = transpose_matrix(&q_proj, hidden_dim, hidden_dim);
        let k_proj_stored = transpose_matrix(&k_proj, kv_inner, hidden_dim);
        let v_proj_stored = transpose_matrix(&v_proj, kv_inner, hidden_dim);
        let o_proj_stored = transpose_matrix(&o_proj, hidden_dim, hidden_dim);
        let gate_proj_stored = transpose_matrix(&gate_proj, intermediate_dim, hidden_dim);
        let up_proj_stored = transpose_matrix(&up_proj, intermediate_dim, hidden_dim);
        let down_proj_stored = transpose_matrix(&down_proj, hidden_dim, intermediate_dim);

        // 3. Position IDs: sequential values 0..seq_len stored as f32.
        let position_ids: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();

        // 4. Define each tensor.
        // Each tuple: (tensor_key, stored_data, conventional_shape, codec, class).
        struct TensorDef<'a> {
            key: &'a str,
            data: &'a [f32],
            conv_shape: &'a [usize],
            codec: CodecFamily,
            class: &'a str,
        }

        let tensor_defs = [
            TensorDef {
                key: "input_layernorm.weight",
                data: &input_layernorm,
                conv_shape: &[hidden_dim],
                codec: policy.norm_codec,
                class: "RmsNormWeight",
            },
            TensorDef {
                key: "q_proj.weight",
                data: &q_proj_stored,
                conv_shape: &[hidden_dim, hidden_dim],
                codec: policy.projection_codec,
                class: "AttentionProjection",
            },
            TensorDef {
                key: "k_proj.weight",
                data: &k_proj_stored,
                conv_shape: &[kv_inner, hidden_dim],
                codec: policy.projection_codec,
                class: "AttentionProjection",
            },
            TensorDef {
                key: "v_proj.weight",
                data: &v_proj_stored,
                conv_shape: &[kv_inner, hidden_dim],
                codec: policy.projection_codec,
                class: "AttentionProjection",
            },
            TensorDef {
                key: "o_proj.weight",
                data: &o_proj_stored,
                conv_shape: &[hidden_dim, hidden_dim],
                codec: policy.projection_codec,
                class: "AttentionProjection",
            },
            TensorDef {
                key: "post_attention_layernorm.weight",
                data: &post_attention_layernorm,
                conv_shape: &[hidden_dim],
                codec: policy.norm_codec,
                class: "RmsNormWeight",
            },
            TensorDef {
                key: "gate_proj.weight",
                data: &gate_proj_stored,
                conv_shape: &[intermediate_dim, hidden_dim],
                codec: policy.mlp_codec,
                class: "DecoderMlpProjection",
            },
            TensorDef {
                key: "up_proj.weight",
                data: &up_proj_stored,
                conv_shape: &[intermediate_dim, hidden_dim],
                codec: policy.mlp_codec,
                class: "DecoderMlpProjection",
            },
            TensorDef {
                key: "down_proj.weight",
                data: &down_proj_stored,
                conv_shape: &[hidden_dim, intermediate_dim],
                codec: policy.mlp_codec,
                class: "DecoderMlpProjection",
            },
            TensorDef {
                key: "position_ids",
                data: &position_ids,
                conv_shape: &[seq_len],
                codec: CodecFamily::RawF32,
                class: "PositionIds",
            },
        ];

        let mut all_payloads: Vec<PendingPayload> = Vec::new();
        let mut tensor_entries: Vec<CImageTensorEntry> = Vec::with_capacity(10);

        for (idx, tensor) in tensor_defs.iter().enumerate() {
            let tensor_id = format!("t{}", idx);

            // Logical shape is the conventional shape; for 1-D use [dim, 1].
            let logical_shape: Vec<u32> = if tensor.conv_shape.len() == 1 {
                vec![tensor.conv_shape[0] as u32, 1u32]
            } else {
                tensor.conv_shape.iter().map(|&d| d as u32).collect()
            };

            // Pack dimensions: [in_features, out_features] = stored layout.
            let (pack_rows, pack_cols) = if tensor.conv_shape.len() == 1 {
                (1usize, tensor.conv_shape[0])
            } else {
                (tensor.conv_shape[1], tensor.conv_shape[0])
            };

            // SHA-256 over the stored f32 values.
            let tensor_sha256 = sha256_of_f32_slice(tensor.data);

            let (payload_ref, phy_layout, raw_ref) = match tensor.codec {
                CodecFamily::RawF32 => {
                    let raw_bytes = f32_slice_to_le_bytes(tensor.data);
                    let codes_payload = PendingPayload {
                        payload_id: format!("p_{}_codes", tensor.key),
                        payload_kind: CImagePayloadKind::PackedTensorCodes,
                        codec: Some("RawF32".into()),
                        alignment_bytes: 64,
                        bytes: raw_bytes.clone(),
                    };
                    let raw_ref_payload = PendingPayload {
                        payload_id: format!("p_{}_rawf32", tensor.key),
                        payload_kind: CImagePayloadKind::RawF32Reference,
                        codec: Some("RawF32".into()),
                        alignment_bytes: 64,
                        bytes: raw_bytes,
                    };
                    all_payloads.push(codes_payload);
                    all_payloads.push(raw_ref_payload);

                    (
                        CImagePayloadRef::Single {
                            payload_id: format!("p_{}_codes", tensor.key),
                        },
                        rawf32_tile_layout(tensor.data.len()),
                        Some(CImagePayloadRef::Single {
                            payload_id: format!("p_{}_rawf32", tensor.key),
                        }),
                    )
                }
                CodecFamily::Nf4 => {
                    let (codes, scales, biases, _packed_rows, _packed_cols) =
                        pack_nf4_weights(tensor.data, pack_rows, pack_cols);

                    let codes_payload = PendingPayload {
                        payload_id: format!("p_{}_codes", tensor.key),
                        payload_kind: CImagePayloadKind::PackedTensorCodes,
                        codec: Some("NF4".into()),
                        alignment_bytes: 64,
                        bytes: codes,
                    };

                    let mut meta_bytes = Vec::with_capacity((scales.len() + biases.len()) * 4);
                    meta_bytes.extend_from_slice(&f32_slice_to_le_bytes(&scales));
                    meta_bytes.extend_from_slice(&f32_slice_to_le_bytes(&biases));

                    let meta_payload = PendingPayload {
                        payload_id: format!("p_{}_codes_metadata", tensor.key),
                        payload_kind: CImagePayloadKind::TensorMetadata,
                        codec: Some("NF4".into()),
                        alignment_bytes: 64,
                        bytes: meta_bytes,
                    };

                    let raw_f32_payload = PendingPayload {
                        payload_id: format!("p_{}_rawf32", tensor.key),
                        payload_kind: CImagePayloadKind::RawF32Reference,
                        codec: Some("RawF32".into()),
                        alignment_bytes: 64,
                        bytes: f32_slice_to_le_bytes(tensor.data),
                    };

                    all_payloads.push(codes_payload);
                    all_payloads.push(meta_payload);
                    all_payloads.push(raw_f32_payload);

                    let layout = nf4_tile_layout(pack_rows, pack_cols);
                    (
                        CImagePayloadRef::Single {
                            payload_id: format!("p_{}_codes", tensor.key),
                        },
                        layout,
                        Some(CImagePayloadRef::Single {
                            payload_id: format!("p_{}_rawf32", tensor.key),
                        }),
                    )
                }
                CodecFamily::Int8 => {
                    let (codes, scales, biases) =
                        pack_int8_weights(tensor.data, pack_rows, pack_cols);

                    let codes_payload = PendingPayload {
                        payload_id: format!("p_{}_codes", tensor.key),
                        payload_kind: CImagePayloadKind::PackedTensorCodes,
                        codec: Some("INT8".into()),
                        alignment_bytes: 64,
                        bytes: codes,
                    };

                    let mut meta_bytes = Vec::with_capacity((scales.len() + biases.len()) * 4);
                    meta_bytes.extend_from_slice(&f32_slice_to_le_bytes(&scales));
                    meta_bytes.extend_from_slice(&f32_slice_to_le_bytes(&biases));

                    let meta_payload = PendingPayload {
                        payload_id: format!("p_{}_codes_metadata", tensor.key),
                        payload_kind: CImagePayloadKind::TensorMetadata,
                        codec: Some("INT8".into()),
                        alignment_bytes: 64,
                        bytes: meta_bytes,
                    };

                    let raw_f32_payload = PendingPayload {
                        payload_id: format!("p_{}_rawf32", tensor.key),
                        payload_kind: CImagePayloadKind::RawF32Reference,
                        codec: Some("RawF32".into()),
                        alignment_bytes: 64,
                        bytes: f32_slice_to_le_bytes(tensor.data),
                    };

                    all_payloads.push(codes_payload);
                    all_payloads.push(meta_payload);
                    all_payloads.push(raw_f32_payload);

                    let layout = int8_tile_layout(pack_rows, pack_cols);
                    (
                        CImagePayloadRef::Single {
                            payload_id: format!("p_{}_codes", tensor.key),
                        },
                        layout,
                        Some(CImagePayloadRef::Single {
                            payload_id: format!("p_{}_rawf32", tensor.key),
                        }),
                    )
                }
                CodecFamily::Fp16
                | CodecFamily::SymInt4
                | CodecFamily::Ternary
                | CodecFamily::Ternary1_58
                | CodecFamily::Mixed => {
                    return Err(CImageError::Other(format!(
                        "unsupported codec {:?} for tensor {}",
                        tensor.codec, tensor.key
                    )));
                }
            };

            let entry = CImageTensorEntry {
                tensor_id,
                tensor_key: tensor.key.to_string(),
                tensor_class: tensor.class.to_string(),
                logical_shape,
                source_dtype: DType::F32,
                codec: tensor.codec,
                precision_plan: None,
                physical_layout: phy_layout,
                payload_ref,
                raw_f32_reference_ref: raw_ref,
                tensor_sha256,
                validation_digest: None,
            };

            tensor_entries.push(entry);
        }

        // 5. Build the execution plan summary.
        let plan_id = format!("synth_decoder_shard_{:016x}", seed);
        let execution_plan = ModelExecutionPlanSummary {
            plan_id: plan_id.clone(),
            region_count: 1,
            total_kernel_ops: 10,
            total_input_bytes: (hidden_dim * 4) as u64,
            total_output_bytes: (hidden_dim * 4) as u64,
            tensor_refs: (0..10u32).map(|i| format!("t{}", i)).collect(),
        };

        // 6. Build manifest.
        let manifest = CImageManifestV0 {
            schema_version: 0,
            model_family: "SyntheticDecoder".into(),
            artifact_kind: CImageArtifactKind::SyntheticShard,
            source_model_digest: None,
            compiler_policy_digest: "synthetic".into(),
            layout_profile: HardwareProfileId::AppleMProBalanced,
            tensors: tensor_entries,
            execution_plan,
            receipts: Vec::new(),
            assistant_graph: None,
            state_store_schema: None,
        };

        Ok(PendingCImageShard {
            manifest,
            payloads: all_payloads,
            receipts: Vec::new(),
        })
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── deterministic_f32_tensor ──────────────────────────────────────────

    #[test]
    fn test_deterministic_f32_tensor() {
        let t1 = deterministic_f32_tensor(42, &[10]);
        let t2 = deterministic_f32_tensor(42, &[10]);
        assert_eq!(t1.len(), 10);
        assert_eq!(t1, t2, "deterministic tensors must be identical");

        let t3 = deterministic_f32_tensor(43, &[10]);
        assert_ne!(t1, t3, "different seeds must produce different tensors");

        for &v in &t1 {
            assert!(v >= -1.0 && v < 1.0, "value {} out of range", v);
        }
    }

    // ── transpose_matrix ──────────────────────────────────────────────────

    #[test]
    fn test_transpose_matrix_identity() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let transposed = transpose_matrix(&data, 2, 3);
        // 2x3 → 3x2: [1,2,3,4,5,6] → [1,4,2,5,3,6]
        assert_eq!(transposed, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        // Round-trip
        let round = transpose_matrix(&transposed, 3, 2);
        assert_eq!(round, data);
    }

    // ── Default config ────────────────────────────────────────────────────

    fn rawf32_policy() -> SyntheticShardPolicy {
        SyntheticShardPolicy {
            gate_codec: CodecFamily::RawF32,
            up_codec: CodecFamily::RawF32,
            down_codec: CodecFamily::RawF32,
            rmsnorm_codec: CodecFamily::RawF32,
            allow_mixed_precision: false,
        }
    }

    fn default_config() -> SyntheticMlpShardConfig {
        SyntheticMlpShardConfig {
            seed: 42,
            hidden_dim: 64,
            intermediate_dim: 128,
            policy: rawf32_policy(),
        }
    }

    // ── All RawF32 ────────────────────────────────────────────────────────

    #[test]
    fn test_build_synthetic_mlp_shard_all_rawf32() {
        let config = default_config();
        let shard = MlpShardBuilder::build_synthetic_mlp_shard(config).unwrap();

        assert_eq!(shard.manifest.schema_version, 0);
        assert_eq!(shard.manifest.model_family, "SyntheticMLP");
        assert_eq!(
            shard.manifest.artifact_kind,
            CImageArtifactKind::SyntheticShard
        );
        assert_eq!(shard.manifest.tensors.len(), 4);

        // rmsnorm_weight — 1-D tensor, logical shape [1, hidden_dim]
        // rmsnorm_weight — 1-D tensor, logical shape [hidden_dim, 1]
        let e0 = &shard.manifest.tensors[0];
        assert_eq!(e0.tensor_id, "t0");
        assert_eq!(e0.tensor_key, "rmsnorm_weight");
        assert_eq!(e0.logical_shape, vec![64, 1]);
        assert_eq!(e0.codec, CodecFamily::RawF32);
        assert!(e0.raw_f32_reference_ref.is_some());

        // gate_proj — 2-D, logcial shape [intermediate_dim, hidden_dim]
        let e1 = &shard.manifest.tensors[1];
        assert_eq!(e1.tensor_id, "t1");
        assert_eq!(e1.tensor_key, "gate_proj");
        assert_eq!(e1.logical_shape, vec![128, 64]);
        assert_eq!(e1.codec, CodecFamily::RawF32);
        assert_eq!(e1.tensor_class, "DecoderMlpProjection");

        let e2 = &shard.manifest.tensors[2];
        assert_eq!(e2.tensor_id, "t2");
        assert_eq!(e2.tensor_key, "up_proj");
        assert_eq!(e2.logical_shape, vec![128, 64]);

        let e3 = &shard.manifest.tensors[3];
        assert_eq!(e3.tensor_id, "t3");
        assert_eq!(e3.tensor_key, "down_proj");
        assert_eq!(e3.logical_shape, vec![64, 128]);

        // Payloads: 4 tensors × 2 (PackedTensorCodes +
        // RawF32Reference) = 8
        assert_eq!(shard.payloads.len(), 8);

        // Execution plan
        assert_eq!(shard.manifest.execution_plan.region_count, 1);
        assert_eq!(shard.manifest.execution_plan.total_kernel_ops, 3);
        assert_eq!(shard.manifest.execution_plan.total_input_bytes, 256);
        assert_eq!(shard.manifest.execution_plan.total_output_bytes, 256);
        assert_eq!(
            shard.manifest.execution_plan.tensor_refs,
            vec!["t0", "t1", "t2", "t3"]
        );

        assert!(shard.receipts.is_empty());
    }

    // ── All NF4 ───────────────────────────────────────────────────────────

    #[test]
    fn test_build_synthetic_mlp_shard_nf4() {
        let config = SyntheticMlpShardConfig {
            seed: 42,
            hidden_dim: 128,
            intermediate_dim: 256,
            policy: SyntheticShardPolicy {
                gate_codec: CodecFamily::Nf4,
                up_codec: CodecFamily::Nf4,
                down_codec: CodecFamily::Nf4,
                rmsnorm_codec: CodecFamily::Nf4,
                allow_mixed_precision: false,
            },
        };
        let shard = MlpShardBuilder::build_synthetic_mlp_shard(config).unwrap();
        assert_eq!(shard.manifest.tensors.len(), 4);

        for entry in &shard.manifest.tensors {
            assert_eq!(entry.codec, CodecFamily::Nf4);
            let l = &entry.physical_layout;
            assert_eq!(l.tile_n, TILE_ELEMENTS as u32);
            assert_eq!(l.group_size, GROUP_SIZE as u32);
            assert_eq!(l.groups_per_tile, GROUPS_PER_TILE as u32);
            assert_eq!(l.packed_bytes_per_tile, PACKED_BYTES_PER_TILE as u32);
            assert_eq!(l.metadata_f32_per_tile, (SCALES_F32_PER_TILE * 2) as u32);
        }

        assert_eq!(shard.payloads.len(), 12); // 4 × 3 (codes + metadata + rawf32)

        // rmsnorm has logical shape [128, 1]
        assert_eq!(shard.manifest.tensors[0].logical_shape, vec![128, 1]);
    }

    // ── All INT8 ──────────────────────────────────────────────────────────

    #[test]
    fn test_build_synthetic_mlp_shard_int8() {
        let config = SyntheticMlpShardConfig {
            seed: 42,
            hidden_dim: 64,
            intermediate_dim: 128,
            policy: SyntheticShardPolicy {
                gate_codec: CodecFamily::Int8,
                up_codec: CodecFamily::Int8,
                down_codec: CodecFamily::Int8,
                rmsnorm_codec: CodecFamily::Int8,
                allow_mixed_precision: false,
            },
        };
        let shard = MlpShardBuilder::build_synthetic_mlp_shard(config).unwrap();
        assert_eq!(shard.manifest.tensors.len(), 4);

        for entry in &shard.manifest.tensors {
            assert_eq!(entry.codec, CodecFamily::Int8);
            let l = &entry.physical_layout;
            assert_eq!(l.tile_n, TILE_ELEMENTS as u32);
            assert_eq!(l.packed_bytes_per_tile, TILE_ELEMENTS as u32);
            assert_eq!(l.group_size, TILE_ELEMENTS as u32);
            assert_eq!(l.groups_per_tile, 1);
            assert_eq!(l.metadata_f32_per_tile, 2);
        }

        assert_eq!(shard.payloads.len(), 12);
    }

    // ── Mixed codec ───────────────────────────────────────────────────────

    #[test]
    fn test_build_synthetic_mlp_shard_mixed_codec() {
        let config = SyntheticMlpShardConfig {
            seed: 99,
            hidden_dim: 64,
            intermediate_dim: 128,
            policy: SyntheticShardPolicy {
                gate_codec: CodecFamily::Nf4,
                up_codec: CodecFamily::Int8,
                down_codec: CodecFamily::RawF32,
                rmsnorm_codec: CodecFamily::RawF32,
                allow_mixed_precision: true,
            },
        };
        let shard = MlpShardBuilder::build_synthetic_mlp_shard(config).unwrap();
        assert_eq!(shard.manifest.tensors.len(), 4);

        assert_eq!(shard.manifest.tensors[0].codec, CodecFamily::RawF32);
        assert_eq!(shard.manifest.tensors[1].codec, CodecFamily::Nf4);
        assert_eq!(shard.manifest.tensors[2].codec, CodecFamily::Int8);
        assert_eq!(shard.manifest.tensors[3].codec, CodecFamily::RawF32);

        // rmsnorm(RawF32)→2, gate(NF4)→3, up(INT8)→3, down(RawF32)→2 = 10
        assert_eq!(shard.payloads.len(), 10);
    }

    // ── Custom dimensions ─────────────────────────────────────────────────

    #[test]
    fn test_build_shard_custom_dimensions() {
        let config = SyntheticMlpShardConfig {
            seed: 0,
            hidden_dim: 32,
            intermediate_dim: 96,
            policy: rawf32_policy(),
        };
        let shard = MlpShardBuilder::build_synthetic_mlp_shard(config).unwrap();

        let tensors = &shard.manifest.tensors;
        assert_eq!(tensors[0].logical_shape, vec![32, 1]); // rmsnorm: [hidden_dim, 1]
        assert_eq!(tensors[1].logical_shape, vec![96, 32]); // gate: [intermediate, hidden]
        assert_eq!(tensors[2].logical_shape, vec![96, 32]); // up:   [intermediate, hidden]
        assert_eq!(tensors[3].logical_shape, vec![32, 96]); // down: [hidden, intermediate]

        assert_eq!(shard.manifest.execution_plan.total_input_bytes, 128);
        assert_eq!(shard.manifest.execution_plan.total_output_bytes, 128);
    }

    // ── Determinism ───────────────────────────────────────────────────────

    #[test]
    fn test_tensor_sha256_is_deterministic() {
        let s1 = MlpShardBuilder::build_synthetic_mlp_shard(default_config()).unwrap();
        let s2 = MlpShardBuilder::build_synthetic_mlp_shard(default_config()).unwrap();

        for (e1, e2) in s1.manifest.tensors.iter().zip(s2.manifest.tensors.iter()) {
            assert_eq!(
                e1.tensor_sha256, e2.tensor_sha256,
                "SHA-256 must be deterministic for {}",
                e1.tensor_key
            );
        }
    }

    #[test]
    fn test_build_shard_determinism() {
        let s1 = MlpShardBuilder::build_synthetic_mlp_shard(default_config()).unwrap();
        let s2 = MlpShardBuilder::build_synthetic_mlp_shard(default_config()).unwrap();
        assert_eq!(
            s1.manifest.execution_plan.plan_id,
            s2.manifest.execution_plan.plan_id
        );
        for (e1, e2) in s1.manifest.tensors.iter().zip(s2.manifest.tensors.iter()) {
            assert_eq!(e1.tensor_sha256, e2.tensor_sha256);
        }
    }

    // ── Payload invariants ────────────────────────────────────────────────

    #[test]
    fn test_payload_ids_are_unique() {
        let shard = MlpShardBuilder::build_synthetic_mlp_shard(default_config()).unwrap();
        let mut ids = std::collections::HashSet::new();
        for p in &shard.payloads {
            assert!(
                ids.insert(&p.payload_id),
                "duplicate payload ID: {}",
                p.payload_id
            );
        }
    }

    #[test]
    fn test_metadata_payload_id_convention() {
        // extract_packed_and_reconstruct builds metadata_id as
        // format!("{}_metadata", payload_id) where payload_id = "p_{tensor_key}_codes".
        // So the metadata payload must be "p_{tensor_key}_codes_metadata".
        let config = SyntheticMlpShardConfig {
            seed: 1,
            hidden_dim: 64,
            intermediate_dim: 128,
            policy: SyntheticShardPolicy {
                gate_codec: CodecFamily::Nf4,
                up_codec: CodecFamily::Int8,
                down_codec: CodecFamily::RawF32,
                rmsnorm_codec: CodecFamily::RawF32,
                allow_mixed_precision: false,
            },
        };
        let shard = MlpShardBuilder::build_synthetic_mlp_shard(config).unwrap();

        // NF4 gate_proj should have metadata "p_gate_proj_codes_metadata"
        let has_gate_meta = shard
            .payloads
            .iter()
            .any(|p| p.payload_id == "p_gate_proj_codes_metadata");
        assert!(
            has_gate_meta,
            "gate metadata must match extract_packed_and_reconstruct convention"
        );

        // INT8 up_proj should have metadata "p_up_proj_codes_metadata"
        let has_up_meta = shard
            .payloads
            .iter()
            .any(|p| p.payload_id == "p_up_proj_codes_metadata");
        assert!(
            has_up_meta,
            "up metadata must match extract_packed_and_reconstruct convention"
        );

        // RawF32 tensors should NOT have metadata payloads
        let meta_count = shard
            .payloads
            .iter()
            .filter(|p| p.payload_kind == CImagePayloadKind::TensorMetadata)
            .count();
        assert_eq!(meta_count, 2, "only NF4 and INT8 tensors have metadata");
    }

    // ── Physical layout ───────────────────────────────────────────────────

    #[test]
    fn test_rawf32_physical_layout() {
        let shard = MlpShardBuilder::build_synthetic_mlp_shard(default_config()).unwrap();
        for entry in &shard.manifest.tensors {
            let l = &entry.physical_layout;
            assert_eq!(l.group_size, 0);
            assert_eq!(l.groups_per_tile, 0);
            assert_eq!(l.tile_m, 1);
            assert_eq!(l.tiles_per_row, 1);
            assert_eq!(l.total_tiles, 1);

            // packed_bytes_per_tile = tensor_len * 4
            let tensor_len: usize = entry.logical_shape.iter().map(|&d| d as usize).product();
            assert_eq!(l.packed_bytes_per_tile, (tensor_len * 4) as u32);
        }
    }

    #[test]
    fn test_nf4_tile_layout_helpers() {
        let l = super::nf4_tile_layout(128, 640);
        assert_eq!(l.tile_n, 640);
        assert_eq!(l.tiles_per_row, 1);
        assert_eq!(l.total_tiles, 128);
        assert_eq!(l.padded_cols, 640);

        let l_small = super::nf4_tile_layout(128, 64);
        assert_eq!(l_small.tiles_per_row, 1);
        assert_eq!(l_small.padded_cols, 640);

        let l_multi = super::nf4_tile_layout(128, 1300);
        assert_eq!(l_multi.tiles_per_row, 3);
        assert_eq!(l_multi.padded_cols, 1920);
    }

    #[test]
    fn test_int8_tile_layout_helpers() {
        let l = super::int8_tile_layout(128, 640);
        assert_eq!(l.tiles_per_row, 1);
        assert_eq!(l.total_tiles, 128);
        assert_eq!(l.packed_bytes_per_tile, 640);
        assert_eq!(l.group_size, 640);
        assert_eq!(l.metadata_f32_per_tile, 2);
    }

    #[test]
    fn test_nf4_tiles_per_row_vs_logical_shape() {
        // When hidden_dim=640 every row fits one tile.
        let config = SyntheticMlpShardConfig {
            seed: 0,
            hidden_dim: 640,
            intermediate_dim: 128,
            policy: SyntheticShardPolicy {
                gate_codec: CodecFamily::Nf4,
                up_codec: CodecFamily::Nf4,
                down_codec: CodecFamily::Nf4,
                rmsnorm_codec: CodecFamily::Nf4,
                allow_mixed_precision: false,
            },
        };
        let shard = MlpShardBuilder::build_synthetic_mlp_shard(config).unwrap();
        // gate_proj logical_shape = [128, 640], stored as [640, 128]
        // pack_rows=640, pack_cols=128 → tiles_per_row = 128.div_ceil(640) = 1
        let gate = &shard.manifest.tensors[1];
        assert_eq!(gate.physical_layout.tiles_per_row, 1);

        // down_proj logical_shape = [640, 128], stored as [128, 640]
        // pack_rows=128, pack_cols=640 → tiles_per_row = 640.div_ceil(640) = 1
        let down = &shard.manifest.tensors[3];
        assert_eq!(down.physical_layout.tiles_per_row, 1);
    }

    // ── Error handling ────────────────────────────────────────────────────

    #[test]
    fn test_unsupported_codec_returns_error() {
        let config = SyntheticMlpShardConfig {
            seed: 0,
            hidden_dim: 64,
            intermediate_dim: 128,
            policy: SyntheticShardPolicy {
                gate_codec: CodecFamily::Fp16,
                up_codec: CodecFamily::RawF32,
                down_codec: CodecFamily::RawF32,
                rmsnorm_codec: CodecFamily::RawF32,
                allow_mixed_precision: false,
            },
        };
        let result = MlpShardBuilder::build_synthetic_mlp_shard(config);
        assert!(result.is_err(), "Fp16 codec should be rejected");
    }

    // ── Pack round-trip via raw f32 weights ───────────────────────────────

    #[test]
    fn test_nf4_roundtrip_weights() {
        // Generate a small tensor, pack NF4, unpack, compare dimensions.
        let hidden_dim = 64usize;
        let interm_dim = 128usize;
        let data = deterministic_f32_tensor(42, &[interm_dim, hidden_dim]);
        let stored = transpose_matrix(&data, interm_dim, hidden_dim);

        let (codes, scales, biases, _pr, _pc) = pack_nf4_weights(&stored, hidden_dim, interm_dim);

        let reconstructed = unpack_nf4_weights(&codes, &scales, &biases, hidden_dim, interm_dim);

        assert_eq!(reconstructed.len(), hidden_dim * interm_dim);
        // Reconstructed & original stored differ due to quantization loss,
        // but should be close.
        let max_diff = stored
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1.0,
            "NF4 round-trip max diff too large: {}",
            max_diff
        );
    }

    #[test]
    fn test_int8_roundtrip_weights() {
        let hidden_dim = 64usize;
        let interm_dim = 128usize;
        let data = deterministic_f32_tensor(42, &[interm_dim, hidden_dim]);
        let stored = transpose_matrix(&data, interm_dim, hidden_dim);

        let (codes, scales, biases) = pack_int8_weights(&stored, hidden_dim, interm_dim);

        let reconstructed = unpack_int8_weights(&codes, &scales, &biases, hidden_dim, interm_dim);

        assert_eq!(reconstructed.len(), hidden_dim * interm_dim);
        let max_diff = stored
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1.0,
            "INT8 round-trip max diff too large: {}",
            max_diff
        );
    }

    // ── Decoder layer shard builder tests ───────────────────────────────────

    fn decoder_rawf32_policy() -> SyntheticDecoderPolicy {
        SyntheticDecoderPolicy {
            projection_codec: CodecFamily::RawF32,
            mlp_codec: CodecFamily::RawF32,
            norm_codec: CodecFamily::RawF32,
            attention_codec: CodecFamily::RawF32,
        }
    }

    fn default_decoder_config() -> SyntheticDecoderLayerConfig {
        SyntheticDecoderLayerConfig {
            seed: 42,
            hidden_dim: 64,
            num_heads: 4,
            num_kv_heads: 4,
            head_dim: 16,
            intermediate_dim: 128,
            seq_len: 8,
            policy: decoder_rawf32_policy(),
        }
    }

    #[test]
    fn test_synthetic_decoder_cimage_builds_valid() {
        let config = default_decoder_config();
        let shard = DecoderLayerShardBuilder::build_synthetic_decoder_layer(config).unwrap();

        assert_eq!(shard.manifest.schema_version, 0);
        assert_eq!(shard.manifest.model_family, "SyntheticDecoder");
        assert_eq!(
            shard.manifest.artifact_kind,
            CImageArtifactKind::SyntheticShard
        );
        assert_eq!(shard.manifest.tensors.len(), 10);

        // Check each tensor entry.
        let tensors = &shard.manifest.tensors;

        // t0: input_layernorm.weight — 1-D
        assert_eq!(tensors[0].tensor_id, "t0");
        assert_eq!(tensors[0].tensor_key, "input_layernorm.weight");
        assert_eq!(tensors[0].logical_shape, vec![64, 1]);
        assert_eq!(tensors[0].codec, CodecFamily::RawF32);
        assert_eq!(tensors[0].tensor_class, "RmsNormWeight");

        // t1: q_proj.weight — [hidden_dim, hidden_dim]
        assert_eq!(tensors[1].tensor_id, "t1");
        assert_eq!(tensors[1].tensor_key, "q_proj.weight");
        assert_eq!(tensors[1].logical_shape, vec![64, 64]);
        assert_eq!(tensors[1].codec, CodecFamily::RawF32);
        assert_eq!(tensors[1].tensor_class, "AttentionProjection");

        // t2: k_proj.weight — [hidden_dim, head_dim * num_kv_heads]
        assert_eq!(tensors[2].tensor_key, "k_proj.weight");
        assert_eq!(tensors[2].logical_shape, vec![64, 64]);

        // t3: v_proj.weight — same shape as k
        assert_eq!(tensors[3].tensor_key, "v_proj.weight");
        assert_eq!(tensors[3].logical_shape, vec![64, 64]);

        // t4: o_proj.weight — [hidden_dim, hidden_dim]
        assert_eq!(tensors[4].tensor_key, "o_proj.weight");
        assert_eq!(tensors[4].logical_shape, vec![64, 64]);

        // t5: post_attention_layernorm.weight — 1-D
        assert_eq!(tensors[5].tensor_key, "post_attention_layernorm.weight");
        assert_eq!(tensors[5].logical_shape, vec![64, 1]);
        assert_eq!(tensors[5].tensor_class, "RmsNormWeight");

        // t6: gate_proj.weight — [intermediate_dim, hidden_dim]
        assert_eq!(tensors[6].tensor_key, "gate_proj.weight");
        assert_eq!(tensors[6].logical_shape, vec![128, 64]);
        assert_eq!(tensors[6].tensor_class, "DecoderMlpProjection");

        // t7: up_proj.weight — [intermediate_dim, hidden_dim]
        assert_eq!(tensors[7].tensor_key, "up_proj.weight");
        assert_eq!(tensors[7].logical_shape, vec![128, 64]);

        // t8: down_proj.weight — [hidden_dim, intermediate_dim]
        assert_eq!(tensors[8].tensor_key, "down_proj.weight");
        assert_eq!(tensors[8].logical_shape, vec![64, 128]);

        // t9: position_ids — [seq_len, 1]
        assert_eq!(tensors[9].tensor_key, "position_ids");
        assert_eq!(tensors[9].logical_shape, vec![8, 1]);
        assert_eq!(tensors[9].codec, CodecFamily::RawF32);
        assert_eq!(tensors[9].tensor_class, "PositionIds");

        // Each RawF32 tensor produces 2 payloads (codes + rawf32) → 10 * 2 = 20
        assert_eq!(shard.payloads.len(), 20);

        // Payload IDs must be unique.
        let mut ids = std::collections::HashSet::new();
        for p in &shard.payloads {
            assert!(
                ids.insert(&p.payload_id),
                "duplicate payload ID: {}",
                p.payload_id
            );
        }
    }

    #[test]
    fn test_decoder_shard_deterministic() {
        let s1 = DecoderLayerShardBuilder::build_synthetic_decoder_layer(default_decoder_config())
            .unwrap();
        let s2 = DecoderLayerShardBuilder::build_synthetic_decoder_layer(default_decoder_config())
            .unwrap();

        for (e1, e2) in s1.manifest.tensors.iter().zip(s2.manifest.tensors.iter()) {
            assert_eq!(
                e1.tensor_sha256, e2.tensor_sha256,
                "SHA-256 must be deterministic for {}",
                e1.tensor_key
            );
        }
    }

    #[test]
    fn test_decoder_shard_payload_count_mixed() {
        let config = SyntheticDecoderLayerConfig {
            seed: 99,
            hidden_dim: 64,
            num_heads: 4,
            num_kv_heads: 4,
            head_dim: 16,
            intermediate_dim: 128,
            seq_len: 1,
            policy: SyntheticDecoderPolicy {
                projection_codec: CodecFamily::Nf4,
                mlp_codec: CodecFamily::Int8,
                norm_codec: CodecFamily::RawF32,
                attention_codec: CodecFamily::Nf4,
            },
        };
        let shard = DecoderLayerShardBuilder::build_synthetic_decoder_layer(config).unwrap();
        assert_eq!(shard.manifest.tensors.len(), 10);

        // Count payloads: 2 norm weights (RawF32 → 2 each) + 4 projection (NF4 → 3 each) + 3 MLP (INT8 → 3 each) + 1 position_ids (RawF32 → 2)
        // = 2*2 + 4*3 + 3*3 + 1*2 = 4 + 12 + 9 + 2 = 27
        assert_eq!(shard.payloads.len(), 27);
    }

    use crate::nf4tile640::{unpack_int8_weights, unpack_nf4_weights};
}
