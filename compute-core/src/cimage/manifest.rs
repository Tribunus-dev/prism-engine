//! CImage manifest types — the semantic contract for a cimage artifact.
//!
//! The manifest binds logical tensor identities to physical byte ranges,
//! declares the execution plan summary, and references receipts.

use serde::{Deserialize, Serialize};

use crate::execution_plan::precision_plan::PrecisionPlan;
use crate::execution_plan::{CodecFamily, DType, HardwareProfileId};

/// V0 manifest: semantic contract for one cimage artifact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CImageManifestV0 {
    pub schema_version: u32,
    pub model_family: String,
    pub artifact_kind: CImageArtifactKind,
    pub source_model_digest: Option<String>,
    pub compiler_policy_digest: String,
    pub layout_profile: HardwareProfileId,
    pub tensors: Vec<CImageTensorEntry>,
    pub execution_plan: ModelExecutionPlanSummary,
    pub receipts: Vec<CImageReceiptRef>,
}

/// Classification of a cimage artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CImageArtifactKind {
    SyntheticShard,
    ModelShard,
    FullModel,
}

/// One tensor entry in the manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CImageTensorEntry {
    pub tensor_id: String,
    pub tensor_key: String,
    pub tensor_class: String,
    pub logical_shape: Vec<u32>,
    pub source_dtype: DType,
    pub codec: CodecFamily,
    pub precision_plan: Option<PrecisionPlan>,
    pub physical_layout: PhysicalTileLayout,
    pub payload_ref: CImagePayloadRef,
    pub raw_f32_reference_ref: Option<CImagePayloadRef>,
    pub tensor_sha256: String,
    pub validation_digest: Option<String>,
}

/// Reference into the payload directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CImagePayloadRef {
    Single {
        payload_id: String,
    },
    MixedPrecision {
        base_payload_id: String,
        override_table_payload_id: String,
        sidecar_payload_ids: Vec<String>,
    },
}

/// Physical tile layout description for a tensor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicalTileLayout {
    pub tile_m: u32,
    pub tile_n: u32,
    pub tiles_per_row: u32,
    pub total_tiles: u32,
    pub padded_cols: u32,
    pub group_size: u32,
    pub groups_per_tile: u32,
    pub packed_bytes_per_tile: u32,
    pub metadata_f32_per_tile: u32,
}

impl PhysicalTileLayout {
    /// Validate that the layout is self-consistent.
    pub fn is_valid(&self) -> bool {
        if self.tile_m == 0 {
            return false;
        }
        // For passthrough (RawF32), group_size == 0 means no grouping.
        if self.group_size > 0 {
            if self.tile_n % self.group_size != 0 {
                return false;
            }
            if self.groups_per_tile != self.tile_n / self.group_size {
                return false;
            }
        } else if self.groups_per_tile != 0 {
            return false;
        }
        true
    }
}

/// Summary of the execution plan (not the full plan — that is in the payload blob).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelExecutionPlanSummary {
    pub plan_id: String,
    pub region_count: u32,
    pub total_kernel_ops: u32,
    pub total_input_bytes: u64,
    pub total_output_bytes: u64,
    pub tensor_refs: Vec<String>,
}

/// Reference to a receipt stored in the receipt directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CImageReceiptRef {
    pub receipt_id: String,
    pub receipt_kind: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution_plan::CodecFamily;

    #[test]
    fn test_manifest_serde_roundtrip() {
        let manifest = CImageManifestV0 {
            schema_version: 0,
            model_family: "synthetic".into(),
            artifact_kind: CImageArtifactKind::SyntheticShard,
            source_model_digest: None,
            compiler_policy_digest: "deadbeef".into(),
            layout_profile: HardwareProfileId::AppleMBaseMemoryBound,
            tensors: vec![CImageTensorEntry {
                tensor_id: "t0".into(),
                tensor_key: "model.layers.0.mlp.gate_proj.weight".into(),
                tensor_class: "DecoderMlpProjection".into(),
                logical_shape: vec![128, 64],
                source_dtype: DType::F32,
                codec: CodecFamily::Nf4,
                precision_plan: None,
                physical_layout: PhysicalTileLayout {
                    tile_m: 1,
                    tile_n: 640,
                    tiles_per_row: 1,
                    total_tiles: 1,
                    padded_cols: 640,
                    group_size: 32,
                    groups_per_tile: 20,
                    packed_bytes_per_tile: 320,
                    metadata_f32_per_tile: 40,
                },
                payload_ref: CImagePayloadRef::Single {
                    payload_id: "p_t0_codes".into(),
                },
                raw_f32_reference_ref: Some(CImagePayloadRef::Single {
                    payload_id: "p_t0_rawf32".into(),
                }),
                tensor_sha256: "abc".into(),
                validation_digest: None,
            }],
            execution_plan: ModelExecutionPlanSummary {
                plan_id: "synth_mlp_000".into(),
                region_count: 1,
                total_kernel_ops: 3,
                total_input_bytes: 256,
                total_output_bytes: 512,
                tensor_refs: vec!["t0".into()],
            },
            receipts: vec![CImageReceiptRef {
                receipt_id: "r0".into(),
                receipt_kind: "LoadReceipt".into(),
            }],
        };
        let json = serde_json::to_string_pretty(&manifest).unwrap();
        let deserialized: CImageManifestV0 = serde_json::from_str(&json).unwrap();
        assert_eq!(
            deserialized.artifact_kind,
            CImageArtifactKind::SyntheticShard
        );
        assert_eq!(deserialized.tensors.len(), 1);
        assert_eq!(deserialized.tensors[0].tensor_id, "t0");
        assert_eq!(deserialized.tensors[0].codec, CodecFamily::Nf4);
    }

    #[test]
    fn test_physical_layout_validation() {
        let valid = PhysicalTileLayout {
            tile_m: 1,
            tile_n: 640,
            tiles_per_row: 1,
            total_tiles: 1,
            padded_cols: 640,
            group_size: 32,
            groups_per_tile: 20,
            packed_bytes_per_tile: 320,
            metadata_f32_per_tile: 40,
        };
        assert!(valid.is_valid());

        let invalid_group_size = PhysicalTileLayout {
            group_size: 0,
            ..valid.clone()
        };
        assert!(!invalid_group_size.is_valid());

        let mismatched_groups = PhysicalTileLayout {
            groups_per_tile: 99,
            ..valid
        };
        assert!(!mismatched_groups.is_valid());
    }

    #[test]
    fn test_artifact_kind_serde() {
        for kind in &[
            CImageArtifactKind::SyntheticShard,
            CImageArtifactKind::ModelShard,
            CImageArtifactKind::FullModel,
        ] {
            let json = serde_json::to_string(kind).unwrap();
            let back: CImageArtifactKind = serde_json::from_str(&json).unwrap();
            assert_eq!(*kind, back);
        }
    }

    #[test]
    fn test_payload_ref_variants() {
        let single = CImagePayloadRef::Single {
            payload_id: "p_0".into(),
        };
        let json = serde_json::to_string(&single).unwrap();
        let back: CImagePayloadRef = serde_json::from_str(&json).unwrap();
        assert!(matches!(back, CImagePayloadRef::Single { .. }));

        let mixed = CImagePayloadRef::MixedPrecision {
            base_payload_id: "base".into(),
            override_table_payload_id: "override".into(),
            sidecar_payload_ids: vec!["sc1".into(), "sc2".into()],
        };
        let json = serde_json::to_string(&mixed).unwrap();
        let back: CImagePayloadRef = serde_json::from_str(&json).unwrap();
        assert!(matches!(back, CImagePayloadRef::MixedPrecision { .. }));
    }
}
