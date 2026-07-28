//! BitNet layer tensor resolver — maps layer-scoped tensor names to cimage
//! payload bytes using the naming convention established by the streaming
//! writer in [`prism_ecs_quantization::bitnet::phases`].
//!
//! The naming convention distinguishes two payload ID patterns:
//! - **RawF32 (norms / global tensors):** payload IDs use underscore separators:
//!   `p_layer_{layer}_{norm_name}_weight` (dots in the tensor key replaced with
//!   underscores).
//! - **Ternary (projection weights):** payload IDs preserve the dotted tensor key:
//!   `p_layer.{layer}.{proj_name}.weight_codes` / `…_scales`.

use crate::ecs::legacy_cimage::{CImageManifestV0, CImagePayloadDirectoryV0, CImageTensorEntry};
use crate::execution_plan::CodecFamily;

use super::error::{CImageRuntimeError, CImageRuntimeResult};

/// Resolves layer-scoped tensors from a loaded cimage's manifest and payload
/// directory into byte slices or decoded f32 vectors.
///
/// Each method builds the appropriate `tensor_key` and `payload_id` according
/// to the naming convention from the streaming writer, then looks up the
/// corresponding payload bytes in the cimage blob.
pub struct BitNetLayerTensorResolver<'a> {
    payload_dir: &'a CImagePayloadDirectoryV0,
    payload_blob: &'a [u8],
    manifest: &'a CImageManifestV0,
    layer: usize,
}

impl<'a> BitNetLayerTensorResolver<'a> {
    /// Create a new resolver for the given layer index.
    pub fn new(
        payload_dir: &'a CImagePayloadDirectoryV0,
        payload_blob: &'a [u8],
        manifest: &'a CImageManifestV0,
        layer: usize,
    ) -> Self {
        Self {
            payload_dir,
            payload_blob,
            manifest,
            layer,
        }
    }

    // ── Public resolution methods ────────────────────────────────────────

    /// Resolve a norm weight tensor (e.g. `input_layernorm`, `post_attention_layernorm`).
    ///
    /// `norm_name` is the short name without the `layer.{N}.` prefix or `.weight` suffix.
    ///
    /// Returns a flat `Vec<f32>` of the norm's weight values.
    ///
    /// # Codec handling
    ///
    /// - `CodecFamily::RawF32` — payload bytes are native f32 LE (the common case for
    ///   norms in the real cimage). Reads 4-byte chunks directly.
    /// - `CodecFamily::Ternary1_58` — unpack 2-bit ternary codes with a per-group f16
    ///   scale (same unpack logic as the existing runner at `region_runner.rs:1287-1318`).
    pub fn resolve_norm_weight(&self, norm_name: &str) -> CImageRuntimeResult<Vec<f32>> {
        let tensor_key = format!("layer.{}.{}.weight", self.layer, norm_name);
        let entry = self
            .find_tensor_entry(&tensor_key)
            .ok_or_else(|| CImageRuntimeError::MissingTensor(tensor_key.clone()))?;

        match entry.codec {
            CodecFamily::RawF32 => {
                // Payload ID uses underscore separators.
                let payload_id = format!("p_{}", tensor_key.replace('.', "_"));
                let data = self.find_payload_bytes(&payload_id)?;
                if data.len() % 4 != 0 {
                    return Err(CImageRuntimeError::InvalidTensorShape(format!(
                        "norm weight data for {} has length {} (not a multiple of 4)",
                        tensor_key,
                        data.len()
                    )));
                }
                Ok(data
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect())
            }
            CodecFamily::Ternary1_58 => {
                // Ternary payload IDs preserve the dotted tensor_key.
                let codes_id = format!("p_{}_codes", tensor_key);
                let scales_id = format!("p_{}_scales", tensor_key);
                let codes = self.find_payload_bytes(&codes_id)?;
                let scales = self.find_payload_bytes(&scales_id)?;

                let gs = entry.physical_layout.group_size as usize;
                let gpr = entry.physical_layout.groups_per_tile as usize;
                let n = entry.logical_shape.first().copied().unwrap_or(0) as usize;
                let bpg = (gs * 2 + 7) / 8; // bytes per group (4 codes per byte)

                let single_scale = if scales.len() >= 2 {
                    half::f16::from_le_bytes([scales[0], scales[1]]).to_f32()
                } else {
                    1.0
                };

                let mut fw = Vec::with_capacity(n);
                for c in 0..n {
                    let g = c / gs;
                    let wi = c % gs;
                    let bi = wi / 4;
                    let ni = wi % 4;
                    let b = if g < gpr && bi < bpg {
                        codes[g * bpg + bi]
                    } else {
                        0
                    };
                    let code = (b >> (ni * 2)) & 0x03;
                    let w: f32 = match code {
                        0 => -1.0,
                        1 => 0.0,
                        2 => 1.0,
                        _ => 0.0,
                    };
                    fw.push(w * single_scale);
                }
                Ok(fw)
            }
            other => Err(CImageRuntimeError::UnsupportedCodec(other)),
        }
    }

    /// Resolve a ternary projection's packed 2-bit codes (raw bytes).
    ///
    /// `proj_name` examples: `"q_proj"`, `"k_proj"`, `"v_proj"`, `"o_proj"`,
    /// `"gate_proj"`, `"up_proj"`, `"down_proj"`.
    ///
    /// Returns the raw bytes of the packed codes payload. The caller is expected
    /// to handle packing layout details via the tensor entry's `physical_layout`.
    pub fn resolve_ternary_codes(&self, proj_name: &str) -> CImageRuntimeResult<Vec<u8>> {
        let tensor_key = format!("layer.{}.{}.weight", self.layer, proj_name);
        let _entry = self
            .find_tensor_entry(&tensor_key)
            .ok_or_else(|| CImageRuntimeError::MissingTensor(tensor_key.clone()))?;

        let payload_id = format!("p_{}_codes", tensor_key);
        let data = self.find_payload_bytes(&payload_id)?;
        Ok(data.to_vec())
    }

    /// Resolve a ternary projection's f16 scales (raw bytes).
    ///
    /// `proj_name` examples same as [`resolve_ternary_codes`].
    ///
    /// Returns the raw bytes of the scales payload (2 bytes per f16, one per group).
    pub fn resolve_ternary_scales(&self, proj_name: &str) -> CImageRuntimeResult<Vec<u8>> {
        let tensor_key = format!("layer.{}.{}.weight", self.layer, proj_name);
        let _entry = self
            .find_tensor_entry(&tensor_key)
            .ok_or_else(|| CImageRuntimeError::MissingTensor(tensor_key.clone()))?;

        let payload_id = format!("p_{}_scales", tensor_key);
        let data = self.find_payload_bytes(&payload_id)?;
        Ok(data.to_vec())
    }

    /// Resolve the global `position_ids` tensor — a `Vec<f32>` of sequential
    /// position indices (one per position up to `SEQ_LEN`).
    ///
    /// The underlying payload is stored as `RawF32` bytes.
    pub fn resolve_position_ids(&self) -> CImageRuntimeResult<Vec<f32>> {
        let tensor_key = "position_ids";
        let _entry = self
            .find_tensor_entry(tensor_key)
            .ok_or_else(|| CImageRuntimeError::MissingTensor(tensor_key.into()))?;

        let payload_id = format!("p_{}", tensor_key.replace('.', "_"));
        let data = self.find_payload_bytes(&payload_id)?;
        if data.len() % 4 != 0 {
            return Err(CImageRuntimeError::InvalidTensorShape(format!(
                "position_ids data has length {} (not a multiple of 4)",
                data.len()
            )));
        }
        Ok(data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect())
    }

    // ── Internal helpers ──────────────────────────────────────────────

    /// Find a tensor entry by its `tensor_key`.
    pub(crate) fn find_tensor_entry(&self, key: &str) -> Option<&'a CImageTensorEntry> {
        self.manifest.tensors.iter().find(|t| t.tensor_key == key)
    }

    /// Look up a payload ID in the directory and return a slice into the
    /// payload blob.
    fn find_payload_bytes(&self, payload_id: &str) -> CImageRuntimeResult<&'a [u8]> {
        let entry = self
            .payload_dir
            .payloads
            .iter()
            .find(|e| e.payload_id == payload_id)
            .ok_or_else(|| CImageRuntimeError::MissingPayload(payload_id.into()))?;
        let start = entry.offset as usize;
        let end = start + entry.len as usize;
        Ok(&self.payload_blob[start..end])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::legacy_cimage::{
        CImageArtifactKind, CImagePayloadEntry, CImagePayloadKind, CImagePayloadRef,
        ModelExecutionPlanSummary, PhysicalTileLayout,
    };
    use crate::execution_plan::HardwareProfileId;

    fn empty_manifest() -> CImageManifestV0 {
        CImageManifestV0 {
            schema_version: 0,
            model_family: "BitNet".into(),
            artifact_kind: CImageArtifactKind::SyntheticShard,
            source_model_digest: None,
            compiler_policy_digest: "test".into(),
            layout_profile: HardwareProfileId::AppleMBaseMemoryBound,
            tensors: vec![],
            execution_plan: ModelExecutionPlanSummary {
                plan_id: "test".into(),
                region_count: 0,
                total_kernel_ops: 0,
                total_input_bytes: 0,
                total_output_bytes: 0,
                tensor_refs: vec![],
            },
            receipts: vec![],
            assistant_graph: None,
            state_store_schema: None,
        }
    }

    #[test]
    fn test_resolver_constructs() {
        let payload_dir = CImagePayloadDirectoryV0 { payloads: vec![] };
        let manifest = empty_manifest();
        let resolver = BitNetLayerTensorResolver::new(&payload_dir, &[], &manifest, 0);
        assert_eq!(resolver.layer, 0);
    }

    #[test]
    fn test_find_missing_tensor_returns_none() {
        let payload_dir = CImagePayloadDirectoryV0 { payloads: vec![] };
        let manifest = empty_manifest();
        let resolver = BitNetLayerTensorResolver::new(&payload_dir, &[], &manifest, 0);
        assert!(resolver
            .find_tensor_entry("layer.0.q_proj.weight")
            .is_none());
    }

    #[test]
    fn test_find_payload_bytes_missing() {
        let payload_dir = CImagePayloadDirectoryV0 { payloads: vec![] };
        let manifest = empty_manifest();
        let resolver = BitNetLayerTensorResolver::new(&payload_dir, &[], &manifest, 0);
        let err = resolver
            .find_payload_bytes("nonexistent")
            .expect_err("should error");
        match err {
            CImageRuntimeError::MissingPayload(id) => assert_eq!(id, "nonexistent"),
            _ => panic!("unexpected error: {err}"),
        }
    }

    #[test]
    fn test_resolve_norm_weight_rawf32() {
        let n = 16u32;
        let mut f32_data = Vec::with_capacity(n as usize * 4);
        for i in 0..n {
            f32_data.extend_from_slice(&(i as f32).to_le_bytes());
        }
        let tensor_key = "layer.0.input_layernorm.weight".to_string();
        let payload_id = "p_layer_0_input_layernorm_weight".to_string();

        let payload_dir = CImagePayloadDirectoryV0 {
            payloads: vec![CImagePayloadEntry {
                payload_id: payload_id.clone(),
                payload_kind: CImagePayloadKind::RawF32Reference,
                codec: None,
                offset: 0,
                len: f32_data.len() as u64,
                alignment_bytes: 64,
                sha256: "".into(),
            }],
        };
        let mut manifest = empty_manifest();
        manifest.tensors = vec![CImageTensorEntry {
            tensor_id: "t0".into(),
            tensor_key: tensor_key.clone(),
            tensor_class: "RmsNormWeight".into(),
            logical_shape: vec![n],
            source_dtype: crate::execution_plan::DType::F32,
            codec: CodecFamily::RawF32,
            precision_plan: None,
            physical_layout: PhysicalTileLayout {
                tile_m: 1,
                tile_n: n,
                tiles_per_row: 1,
                total_tiles: 1,
                padded_cols: n,
                group_size: n,
                groups_per_tile: 1,
                packed_bytes_per_tile: f32_data.len() as u32,
                metadata_f32_per_tile: 0,
            },
            payload_ref: CImagePayloadRef::Single {
                payload_id: payload_id.clone(),
            },
            raw_f32_reference_ref: None,
            tensor_sha256: "".into(),
            validation_digest: None,
        }];

        let resolver = BitNetLayerTensorResolver::new(&payload_dir, &f32_data, &manifest, 0);
        let result = resolver.resolve_norm_weight("input_layernorm").unwrap();
        assert_eq!(result.len(), n as usize);
        for i in 0..n as usize {
            assert!((result[i] - i as f32).abs() < 1e-6, "mismatch at {i}");
        }
    }

    #[test]
    fn test_resolve_position_ids() {
        let seq_len = 64u32;
        let mut pos_bytes = Vec::with_capacity(seq_len as usize * 4);
        for i in 0..seq_len {
            pos_bytes.extend_from_slice(&(i as f32).to_le_bytes());
        }
        let payload_id = "p_position_ids".to_string();

        let payload_dir = CImagePayloadDirectoryV0 {
            payloads: vec![CImagePayloadEntry {
                payload_id: payload_id.clone(),
                payload_kind: CImagePayloadKind::RawF32Reference,
                codec: None,
                offset: 0,
                len: pos_bytes.len() as u64,
                alignment_bytes: 64,
                sha256: "".into(),
            }],
        };
        let mut manifest = empty_manifest();
        manifest.tensors = vec![CImageTensorEntry {
            tensor_id: "t_pos".into(),
            tensor_key: "position_ids".into(),
            tensor_class: "PositionIds".into(),
            logical_shape: vec![seq_len],
            source_dtype: crate::execution_plan::DType::F32,
            codec: CodecFamily::RawF32,
            precision_plan: None,
            physical_layout: PhysicalTileLayout {
                tile_m: 1,
                tile_n: seq_len,
                tiles_per_row: 1,
                total_tiles: 1,
                padded_cols: seq_len,
                group_size: seq_len,
                groups_per_tile: 1,
                packed_bytes_per_tile: pos_bytes.len() as u32,
                metadata_f32_per_tile: 0,
            },
            payload_ref: CImagePayloadRef::Single {
                payload_id: payload_id.clone(),
            },
            raw_f32_reference_ref: None,
            tensor_sha256: "".into(),
            validation_digest: None,
        }];

        let resolver = BitNetLayerTensorResolver::new(&payload_dir, &pos_bytes, &manifest, 0);
        let result = resolver.resolve_position_ids().unwrap();
        assert_eq!(result.len(), seq_len as usize);
        for i in 0..seq_len as usize {
            assert!((result[i] - i as f32).abs() < 1e-6, "mismatch at {i}");
        }
    }

    #[test]
    fn test_resolve_ternary_codes_scales() {
        let tensor_key = "layer.3.q_proj.weight".to_string();
        let codes_data = vec![0x1Bu8, 0x2Cu8, 0x3Du8, 0x4Eu8];
        let scales_data = vec![0x00u8, 0x3Cu8]; // f16 = 1.0
        let codes_payload_id = "p_layer.3.q_proj.weight_codes".to_string();
        let scales_payload_id = "p_layer.3.q_proj.weight_scales".to_string();

        let payload_dir = CImagePayloadDirectoryV0 {
            payloads: vec![
                CImagePayloadEntry {
                    payload_id: codes_payload_id.clone(),
                    payload_kind: CImagePayloadKind::TernaryPackedCodes,
                    codec: Some("Ternary1_58".into()),
                    offset: 0,
                    len: codes_data.len() as u64,
                    alignment_bytes: 64,
                    sha256: "".into(),
                },
                CImagePayloadEntry {
                    payload_id: scales_payload_id.clone(),
                    payload_kind: CImagePayloadKind::TernaryScales,
                    codec: Some("Ternary1_58".into()),
                    offset: codes_data.len() as u64,
                    len: scales_data.len() as u64,
                    alignment_bytes: 64,
                    sha256: "".into(),
                },
            ],
        };
        let mut blob = Vec::new();
        blob.extend_from_slice(&codes_data);
        blob.extend_from_slice(&scales_data);

        let mut manifest = empty_manifest();
        manifest.tensors = vec![CImageTensorEntry {
            tensor_id: "t_q".into(),
            tensor_key: tensor_key.clone(),
            tensor_class: "AttentionProjection".into(),
            logical_shape: vec![4, 4],
            source_dtype: crate::execution_plan::DType::F32,
            codec: CodecFamily::Ternary1_58,
            precision_plan: None,
            physical_layout: PhysicalTileLayout {
                tile_m: 1,
                tile_n: 4,
                tiles_per_row: 1,
                total_tiles: 4,
                padded_cols: 4,
                group_size: 4,
                groups_per_tile: 1,
                packed_bytes_per_tile: 1,
                metadata_f32_per_tile: 0,
            },
            payload_ref: CImagePayloadRef::Single {
                payload_id: codes_payload_id.clone(),
            },
            raw_f32_reference_ref: None,
            tensor_sha256: "".into(),
            validation_digest: None,
        }];

        let resolver = BitNetLayerTensorResolver::new(&payload_dir, &blob, &manifest, 3);
        let result_codes = resolver.resolve_ternary_codes("q_proj").unwrap();
        assert_eq!(result_codes, codes_data);

        let result_scales = resolver.resolve_ternary_scales("q_proj").unwrap();
        assert_eq!(result_scales, scales_data);
    }

    #[test]
    fn test_tensor_key_format_conventions() {
        // Layer 7 norm — underscores in payload ID.
        assert_eq!(
            format!("p_{}", "layer.7.input_layernorm.weight".replace('.', "_")),
            "p_layer_7_input_layernorm_weight"
        );

        // Layer 7 ternary — dots preserved in payload ID.
        assert_eq!(
            format!("p_{}_codes", "layer.7.q_proj.weight"),
            "p_layer.7.q_proj.weight_codes"
        );
        assert_eq!(
            format!("p_{}_scales", "layer.7.q_proj.weight"),
            "p_layer.7.q_proj.weight_scales"
        );

        // Global position_ids — underscores.
        assert_eq!(
            format!("p_{}", "position_ids".replace('.', "_")),
            "p_position_ids"
        );
    }
}
