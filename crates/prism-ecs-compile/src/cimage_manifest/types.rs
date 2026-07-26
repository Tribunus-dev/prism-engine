//! `Manifest` schema and per-tensor types — the canonical on-disk shape of a
//! compiled CImage directory.
//!
//! This module owns the constitutional authority for the per-tensor
//! shape contract: the [`TensorEntry`] / [`QuantizationDesc`] /
//! [`AliasEntry`] types that populate the `tensor_table` and
//! `alias_table` of a [`super::Manifest`], plus the shared-lane
//! [`Nf4Tile640Layout`] / [`SharedWeightLayout`] descriptors that pin
//! the physical ABI for quantized weight triplets.
//!
//! The module does **not** own the manifest itself (see
//! [`super::header`]), the lease state machine (see
//! [`super::lease`]), or the kernel dispatch recipes (see
//! [`super::kernel`]). Those authorities each have their own file;
//! this file is the tensor-shape surface only.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

// ── Per-tensor table types ────────────────────────────────────────────────

/// One tensor entry in the global tensor table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorEntry {
    pub id: u32,
    pub name: String,
    pub role: String,
    pub layer: Option<u32>,
    pub segment: String,
    pub source_filename: String,
    pub source_sha256: String,
    pub source_offset: u64,
    pub offset: u64,
    pub byte_length: u64,
    pub logical_dtype: String,
    pub storage_dtype: String,
    pub logical_shape: Vec<u32>,
    pub physical_shape: Vec<u32>,
    pub mutability: String,
    pub quantization: Option<QuantizationDesc>,
    /// Per-tensor alignment in bytes for the mapped-no-copy backend (default 16).
    #[serde(default = "default_tensor_alignment_bytes")]
    pub tensor_alignment_bytes: u64,
    /// Layout version for the tensor-cache key computation (default 1).
    #[serde(default = "default_layout_version")]
    pub layout_version: u32,
    /// Per-backend artifact bindings for this tensor.
    /// Keyed by backend name ("mlx", "coreai", "accelerate", etc.).
    ///
    /// The re-implementation uses [`BTreeMap`] rather than [`std::collections::HashMap`]
    /// because the binding set is part of the manifest hash and the iteration
    /// order must be deterministic for replay and rebuild.
    #[serde(default)]
    pub artifact_bindings: BTreeMap<String, Vec<BackendWeightArtifact>>,
}

/// Per-tensor quantization descriptor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationDesc {
    pub bits: u32,
    pub group_size: u32,
    pub groups: u32,
    pub scale_tensor_id: u32,
    pub bias_tensor_id: u32,
    /// Explicit physical storage contract for shared-lane execution.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub storage_layout: Option<SharedWeightLayout>,
}

/// An alias mapping — resolves a logical tensor name to physical storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AliasEntry {
    pub logical_name: String,
    pub physical_tensor_id: u32,
    pub reason: String,
}

fn default_tensor_alignment_bytes() -> u64 {
    16
}
fn default_layout_version() -> u32 {
    1
}

// ── Source identity ───────────────────────────────────────────────────────

/// Cryptographic identity of the source checkpoint.
///
/// The re-implementation uses `Vec<ShardHash>` (with `BTreeMap`-style
/// sorted iteration in the builder) for the shard/tokenizer/auxiliary
/// hash lists. The struct is the **canonical on-disk shape** of the
/// source identity section of a manifest; changing it is a
/// constitutional change.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceIdentity {
    pub config_hash: String,
    pub shard_hashes: Vec<ShardHash>,
    pub tokenizer_hashes: Vec<ShardHash>,
    pub auxiliary_hashes: Vec<ShardHash>,
    pub model_type: String,
    pub quantization_bits: u32,
    pub quantization_group_size: u32,
    pub quantization_mode: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardHash {
    pub filename: String,
    pub sha256: String,
}

// ── Shared-lane ABI types ─────────────────────────────────────────────────

/// Physical shared-lane storage layout for quantized weight triplets.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SharedWeightLayout {
    Nf4Tile640(Nf4Tile640Layout),
}

/// Canonical NF4 Tile640 packed-weight + FP32 metadata ABI.
///
/// The weight tensor stores raw packed u8 bytes in 640-element macro-tiles.
/// Companion scale and bias tensors store FP32 metadata in the exact same
/// tile/group order. Both the Metal path and stateless ANE path consume this
/// descriptor so they agree on the same resident bytes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Nf4Tile640Layout {
    pub tile_elements: u32,
    pub quant_group_size: u32,
    pub groups_per_tile: u32,
    pub packed_weight_bytes_per_tile: u32,
    pub scale_values_per_tile: u32,
    pub bias_values_per_tile: u32,
    pub packed_weight_dtype: String,
    pub metadata_dtype: String,
    pub weight_lane_read_bytes: u32,
    /// Profile ID for adaptive codebook (None = canonical NF4).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile_id: Option<u32>,
}

impl Nf4Tile640Layout {
    pub const TILE_ELEMENTS: u32 = 640;
    pub const QUANT_GROUP_SIZE: u32 = 128;
    pub const GROUPS_PER_TILE: u32 = 5;
    pub const PACKED_WEIGHT_BYTES_PER_TILE: u32 = 320;
    pub const SCALE_VALUES_PER_TILE: u32 = 5;
    pub const BIAS_VALUES_PER_TILE: u32 = 5;
    pub const WEIGHT_LANE_READ_BYTES: u32 = 2;

    /// Return the canonical NF4 Tile640 layout used by the shared-lane
    /// ternary projections.
    pub fn canonical() -> Self {
        Self {
            tile_elements: Self::TILE_ELEMENTS,
            quant_group_size: Self::QUANT_GROUP_SIZE,
            groups_per_tile: Self::GROUPS_PER_TILE,
            packed_weight_bytes_per_tile: Self::PACKED_WEIGHT_BYTES_PER_TILE,
            scale_values_per_tile: Self::SCALE_VALUES_PER_TILE,
            bias_values_per_tile: Self::BIAS_VALUES_PER_TILE,
            packed_weight_dtype: "U8".into(),
            metadata_dtype: "F32".into(),
            weight_lane_read_bytes: Self::WEIGHT_LANE_READ_BYTES,
            profile_id: None,
        }
    }

    /// Number of 640-element tiles required to cover `cols` output channels.
    pub fn tiles_for_cols(&self, cols: u32) -> u32 {
        cols.div_ceil(self.tile_elements)
    }

    /// Per-row packed-weight byte count for an `N`-row weight.
    pub fn packed_row_bytes(&self, cols: u32) -> u32 {
        self.tiles_for_cols(cols) * self.packed_weight_bytes_per_tile
    }

    /// Per-row metadata (scale + bias) value count.
    pub fn metadata_row_values(&self, cols: u32) -> u32 {
        self.tiles_for_cols(cols) * self.groups_per_tile
    }
}

// ── Per-backend artifact types ───────────────────────────────────────────

/// Concrete packing scheme and target backend for a weight artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactKind {
    /// MLX NF4 — packed uint32 words (8 NF4 values per u32).
    MlxNf4U32,
    /// Shared-lane NF4 Tile640 — packed uint8 bytes with FP32 sidecar metadata.
    Nf4Tile640Shared,
    /// MLX 8-bit affine — packed uint32 words (4 u8 values per u32).
    MlxAf8U32,
    /// CPU fp16 — dequantized float16.
    CpuFp16,
    /// CPU quantized — block quantized bytes.
    CpuQuantized,
    /// Core ML fp16 external weight file.
    CoreAiFp16WeightFile,
    /// Intel Level Zero packed USM.
    IntelUsmPacked,
    /// Tenstorrent Tensix tiled.
    TensixTilePacked,
}

/// A concrete artifact for one backend execution lane.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendWeightArtifact {
    /// Logical tensor this artifact represents.
    pub logical_tensor_id: String,
    /// Target backend.
    pub backend: String,
    /// Packing/kernel format.
    pub artifact_kind: ArtifactKind,
    /// Logical (semantic) shape.
    pub logical_shape: Vec<u32>,
    /// Physical (storage) shape after packing.
    pub storage_shape: Vec<u32>,
    /// Source quantization before any dequantization transform.
    pub logical_quantization: Option<QuantizationDesc>,
    /// Storage dtype string ("U32", "F16", "U8", etc.).
    pub storage_dtype: String,
    /// How values are packed ("nf4_u32", "af8_u32", "af8_u8", "none_fp16").
    pub packing_scheme: String,
    /// Block quantization group size (0 = per-tensor).
    pub group_size: u32,
    /// Name of the companion scale tensor artifact binding.
    pub scale_binding: Option<String>,
    /// Name of the companion zero-point tensor artifact binding.
    pub zero_point_binding: Option<String>,
    /// Segment filename containing the raw bytes.
    pub segment_path: String,
    /// Byte offset within the segment.
    pub byte_offset: u64,
    /// Byte length of this artifact in the segment.
    pub byte_length: u64,
    /// SHA-256 checksum of the artifact bytes.
    pub checksum: String,
    /// Estimated numerical error introduced by quantization (0.0 for fp16).
    pub numerical_error: f64,
    /// Compiler version that produced this artifact.
    pub producer_version: String,
}

// ── Quantization profile + quality types ─────────────────────────────────

/// Qualification status for a quantized CImage.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuantizationQualityStatus {
    /// Passed all gates — safe for production.
    Qualified,
    /// Experimental — requires explicit opt-in at runtime.
    Experimental,
    /// Failed gates — must not load.
    Rejected,
    /// Not yet evaluated.
    #[default]
    Unknown,
}

/// One profile entry serialized into the manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationProfileEntry {
    /// Profile ID (matches ProfileId in the codebook crate).
    pub profile_id: u32,
    /// Profile name (e.g. "canonical_nf4_v1").
    pub name: String,
    /// ABI version.
    pub abi_version: u32,
    /// Codebook values (16 f32 values).
    pub codebook: Vec<f32>,
    /// Group size.
    pub group_size: u32,
    /// Tile elements.
    pub tile_elements: u32,
    /// Clipping policy string.
    #[serde(default)]
    pub clipping_policy: String,
    /// Bias policy string.
    #[serde(default)]
    pub bias_policy: String,
    /// Sidecar policy string ("none", "sparse_fp16_residual", "protected_channel").
    #[serde(default)]
    pub sidecar_policy: String,
    /// Training objective.
    #[serde(default)]
    pub training_objective: String,
    /// Training iterations.
    #[serde(default)]
    pub training_iterations: u32,
    /// Calibration corpus digest.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration_digest: Option<String>,
    /// Source model digest.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_model_digest: Option<String>,
    /// Compiler revision.
    #[serde(default)]
    pub compiler_revision: String,
}

/// Per-tensor quality evidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationQualityEntry {
    /// Tensor name.
    pub tensor_name: String,
    /// Matrix role.
    pub matrix_role: String,
    /// Profile ID used.
    pub profile_id: u32,
    /// Raw weight RMSE.
    pub weight_rmse: f32,
    /// Raw weight NRMSE.
    pub weight_nrmse: f32,
    /// Maximum absolute error.
    pub max_abs_error: f32,
    /// Activation-weighted output RMSE (0.0 if not calibrated).
    #[serde(default)]
    pub output_rmse: f32,
    /// SQNR in dB.
    #[serde(default)]
    pub sqnr_db: f32,
    /// Fraction of values clipped.
    #[serde(default)]
    pub clipped_fraction: f32,
    /// Effective bits per weight.
    #[serde(default)]
    pub effective_bpw: f32,
    /// Sidecar bytes.
    #[serde(default)]
    pub sidecar_bytes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_entry_uses_btreemap_for_artifact_bindings() {
        // Construct a TensorEntry and verify the artifact_bindings field is a
        // BTreeMap (the constitutional rule for canonical collections).
        let entry = TensorEntry {
            id: 0,
            name: "x".into(),
            role: "weight".into(),
            layer: Some(0),
            segment: "layer_0".into(),
            source_filename: "x.safetensors".into(),
            source_sha256: "deadbeef".into(),
            source_offset: 0,
            offset: 0,
            byte_length: 1024,
            logical_dtype: "F32".into(),
            storage_dtype: "U8".into(),
            logical_shape: vec![128, 128],
            physical_shape: vec![128, 128],
            mutability: "read_only".into(),
            quantization: None,
            tensor_alignment_bytes: 16,
            layout_version: 1,
            artifact_bindings: BTreeMap::new(),
        };
        assert!(entry.artifact_bindings.is_empty());
    }

    #[test]
    fn nf4_tile640_canonical_layout_uses_baseline_constants() {
        let layout = Nf4Tile640Layout::canonical();
        assert_eq!(layout.tile_elements, Nf4Tile640Layout::TILE_ELEMENTS);
        assert_eq!(layout.quant_group_size, Nf4Tile640Layout::QUANT_GROUP_SIZE);
        assert_eq!(layout.groups_per_tile, Nf4Tile640Layout::GROUPS_PER_TILE);
        assert_eq!(
            layout.packed_weight_bytes_per_tile,
            Nf4Tile640Layout::PACKED_WEIGHT_BYTES_PER_TILE
        );
    }

    #[test]
    fn nf4_tile640_tiles_for_cols_rounds_up() {
        let layout = Nf4Tile640Layout::canonical();
        // 640 cols exactly one tile.
        assert_eq!(layout.tiles_for_cols(640), 1);
        // 641 cols two tiles (rounded up).
        assert_eq!(layout.tiles_for_cols(641), 2);
        // 1 col one tile.
        assert_eq!(layout.tiles_for_cols(1), 1);
    }

    #[test]
    fn nf4_tile640_packed_row_bytes_scales_with_tiles() {
        let layout = Nf4Tile640Layout::canonical();
        // 1280 cols -> 2 tiles -> 2 * 320 = 640 bytes per row.
        assert_eq!(layout.packed_row_bytes(1280), 640);
    }

    #[test]
    fn nf4_tile640_metadata_row_values_scales_with_tiles() {
        let layout = Nf4Tile640Layout::canonical();
        // 1280 cols -> 2 tiles -> 2 * 5 = 10 metadata values per row.
        assert_eq!(layout.metadata_row_values(1280), 10);
    }

    #[test]
    fn quantization_quality_status_default_is_unknown() {
        assert_eq!(
            QuantizationQualityStatus::default(),
            QuantizationQualityStatus::Unknown
        );
    }

    #[test]
    fn shared_weight_layout_carries_nf4_tile640() {
        let layout = Nf4Tile640Layout::canonical();
        let shared = SharedWeightLayout::Nf4Tile640(layout.clone());
        match shared {
            SharedWeightLayout::Nf4Tile640(inner) => {
                assert_eq!(inner, layout);
            }
        }
    }

    #[test]
    fn quantization_profile_entry_carries_codebook() {
        let profile = QuantizationProfileEntry {
            profile_id: 7,
            name: "canonical_nf4_v1".into(),
            abi_version: 1,
            codebook: vec![0.0; 16],
            group_size: 128,
            tile_elements: 640,
            clipping_policy: "clamp".into(),
            bias_policy: "zero".into(),
            sidecar_policy: "none".into(),
            training_objective: "mse".into(),
            training_iterations: 0,
            calibration_digest: None,
            source_model_digest: None,
            compiler_revision: "v0".into(),
        };
        assert_eq!(profile.codebook.len(), 16);
        assert_eq!(profile.group_size, 128);
    }

    #[test]
    fn source_identity_round_trip_preserves_fields() {
        let src = SourceIdentity {
            config_hash: "abcd".into(),
            shard_hashes: vec![ShardHash {
                filename: "model.safetensors".into(),
                sha256: "00".into(),
            }],
            tokenizer_hashes: Vec::new(),
            auxiliary_hashes: Vec::new(),
            model_type: "qwen3".into(),
            quantization_bits: 4,
            quantization_group_size: 128,
            quantization_mode: "nf4".into(),
        };
        let json = serde_json::to_string(&src).unwrap();
        let parsed: SourceIdentity = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.config_hash, "abcd");
        assert_eq!(parsed.shard_hashes.len(), 1);
        assert_eq!(parsed.shard_hashes[0].filename, "model.safetensors");
        assert_eq!(parsed.quantization_bits, 4);
    }
}
