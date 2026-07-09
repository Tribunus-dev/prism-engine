//! Phased BitNet cimage emission — linear, MLP block, and decoder layer.
//!
//! Each phase uses the existing cimage shard builder infrastructure to
//! emit a `PendingCImageShard` from BitNet-native ternary weights.

use crate::bitnet::importer::BitNetImporter;
use crate::cimage::shard_builder::PendingCImageShard;
use crate::cimage::*;
use crate::execution_plan::{CodecFamily, DType, HardwareProfileId};
use crate::ternary::codec::TernaryPackedTensor;
use sha2::{Digest, Sha256};

/// Compute SHA-256 hex digest of raw ternary codes (little-endian bytes).
fn sha256_of_bytes(data: &[u8]) -> String {
    format!("{:x}", Sha256::digest(data))
}

/// Build a `PhysicalTileLayout` for a ternary-grouped tensor.
fn ternary_layout(tensor: &TernaryPackedTensor) -> PhysicalTileLayout {
    PhysicalTileLayout {
        tile_m: 1,
        tile_n: tensor.cols as u32,
        tiles_per_row: 1,
        total_tiles: tensor.rows as u32,
        padded_cols: (tensor.groups_per_row * tensor.group_size) as u32,
        group_size: tensor.group_size as u32,
        groups_per_tile: tensor.groups_per_row as u32,
        packed_bytes_per_tile: (tensor.groups_per_row * tensor.bytes_per_group) as u32,
        metadata_f32_per_tile: ((tensor.groups_per_row + 1) / 2) as u32, // f16 -> 2 per u32
    }
}

/// Phase 1: Emit a single BitLinear weight matrix as a cimage.
pub fn emit_single_bitnet_linear(
    tensor_key: &str,
    tensor: &TernaryPackedTensor,
) -> CImageResult<PendingCImageShard> {
    let tensor_id = "t0".to_string();
    let codes_payload = PendingPayload {
        payload_id: format!("p_{}_codes", tensor_key),
        payload_kind: CImagePayloadKind::TernaryPackedCodes,
        codec: Some("Ternary1_58".into()),
        alignment_bytes: 64,
        bytes: tensor.codes.clone(),
    };

    let scale_bytes: Vec<u8> = tensor.scales.iter().flat_map(|s| s.to_le_bytes()).collect();
    let scales_payload = PendingPayload {
        payload_id: format!("p_{}_scales", tensor_key),
        payload_kind: CImagePayloadKind::TernaryScales,
        codec: Some("Ternary1_58".into()),
        alignment_bytes: 64,
        bytes: scale_bytes,
    };

    let payload_ref = CImagePayloadRef::Single {
        payload_id: format!("p_{}_codes", tensor_key),
    };

    let entry = CImageTensorEntry {
        tensor_id,
        tensor_key: tensor_key.to_string(),
        tensor_class: "BitNetLinear".into(),
        logical_shape: vec![tensor.rows as u32, tensor.cols as u32],
        source_dtype: DType::F32,
        codec: CodecFamily::Ternary1_58,
        precision_plan: None,
        physical_layout: ternary_layout(tensor),
        payload_ref,
        raw_f32_reference_ref: None,
        tensor_sha256: sha256_of_bytes(&tensor.codes),
        validation_digest: None,
    };

    let manifest = CImageManifestV0 {
        schema_version: 0,
        model_family: "BitNet-2B4T".into(),
        artifact_kind: CImageArtifactKind::SyntheticShard,
        source_model_digest: None,
        compiler_policy_digest: "bitnet-native-ternary".into(),
        layout_profile: HardwareProfileId::AppleMProBalanced,
        tensors: vec![entry],
        execution_plan: ModelExecutionPlanSummary {
            plan_id: format!("bitnet_linear_{}", tensor_key),
            region_count: 1,
            total_kernel_ops: 1,
            total_input_bytes: (tensor.cols * 4) as u64,
            total_output_bytes: (tensor.rows * 4) as u64,
            tensor_refs: vec!["t0".into()],
        },
        receipts: Vec::new(),
        assistant_graph: None,
        state_store_schema: None,
    };

    Ok(PendingCImageShard {
        manifest,
        payloads: vec![codes_payload, scales_payload],
        receipts: Vec::new(),
    })
}

/// Phase 2: Emit one BitNet MLP block (gate_proj + up_proj + down_proj).
pub fn emit_bitnet_mlp_block(
    seed: u64,
    hidden_dim: usize,
    intermediate_dim: usize,
    group_size: usize,
) -> CImageResult<PendingCImageShard> {
    let (gate, up, down) =
        BitNetImporter::import_mlp_block(seed, hidden_dim, intermediate_dim, group_size)
            .map_err(|e| CImageError::Other(format!("import MLP block: {e}")))?;

    let mut all_payloads = Vec::new();
    let mut entries = Vec::new();

    for (i, (key, tensor)) in [("gate_proj", &gate), ("up_proj", &up), ("down_proj", &down)]
        .iter()
        .enumerate()
    {
        let codes_payload = PendingPayload {
            payload_id: format!("p_{}_codes", key),
            payload_kind: CImagePayloadKind::TernaryPackedCodes,
            codec: Some("Ternary1_58".into()),
            alignment_bytes: 64,
            bytes: tensor.codes.clone(),
        };
        let scale_bytes: Vec<u8> = tensor.scales.iter().flat_map(|s| s.to_le_bytes()).collect();
        let scales_payload = PendingPayload {
            payload_id: format!("p_{}_scales", key),
            payload_kind: CImagePayloadKind::TernaryScales,
            codec: Some("Ternary1_58".into()),
            alignment_bytes: 64,
            bytes: scale_bytes,
        };
        all_payloads.push(codes_payload);
        all_payloads.push(scales_payload);

        let entry = CImageTensorEntry {
            tensor_id: format!("t{}", i),
            tensor_key: key.to_string(),
            tensor_class: "DecoderMlpProjection".into(),
            logical_shape: vec![tensor.rows as u32, tensor.cols as u32],
            source_dtype: DType::F32,
            codec: CodecFamily::Ternary1_58,
            precision_plan: None,
            physical_layout: ternary_layout(tensor),
            payload_ref: CImagePayloadRef::Single {
                payload_id: format!("p_{}_codes", key),
            },
            raw_f32_reference_ref: None,
            tensor_sha256: sha256_of_bytes(&tensor.codes),
            validation_digest: None,
        };
        entries.push(entry);
    }

    let manifest = CImageManifestV0 {
        schema_version: 0,
        model_family: "BitNet-2B4T".into(),
        artifact_kind: CImageArtifactKind::SyntheticShard,
        source_model_digest: None,
        compiler_policy_digest: "bitnet-native-ternary".into(),
        layout_profile: HardwareProfileId::AppleMProBalanced,
        tensors: entries,
        execution_plan: ModelExecutionPlanSummary {
            plan_id: format!("bitnet_mlp_{:016x}", seed),
            region_count: 1,
            total_kernel_ops: 3,
            total_input_bytes: (hidden_dim * 4) as u64,
            total_output_bytes: (hidden_dim * 4) as u64,
            tensor_refs: vec!["t0".into(), "t1".into(), "t2".into()],
        },
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

/// Phase 3: Emit one full BitNet decoder layer (attention + MLP).
///
/// For now this is a stub that delegates to `emit_bitnet_mlp_block` plus
/// placeholder attention tensor entries. Full attention+KV emission will
/// be added in a follow-up once the decoder layer format stabilises.
pub fn emit_bitnet_decoder_layer(
    seed: u64,
    hidden_dim: usize,
    intermediate_dim: usize,
    _num_heads: usize,
    _head_dim: usize,
    group_size: usize,
) -> CImageResult<PendingCImageShard> {
    // Start with the MLP block payloads.
    let mlp_shard = emit_bitnet_mlp_block(seed, hidden_dim, intermediate_dim, group_size)?;

    // Build a manifest that references both MLP and placeholder attention tensors.
    let plan_id = format!("bitnet_decoder_layer_{:016x}", seed);

    let manifest = CImageManifestV0 {
        tensors: mlp_shard.manifest.tensors,
        execution_plan: ModelExecutionPlanSummary {
            plan_id: plan_id.clone(),
            region_count: 2,
            total_kernel_ops: 6,
            total_input_bytes: (hidden_dim * 4) as u64,
            total_output_bytes: (hidden_dim * 4) as u64,
            tensor_refs: vec![
                "t0".into(),
                "t1".into(),
                "t2".into(),
                "t3".into(),
                "t4".into(),
                "t5".into(),
            ],
        },
        ..mlp_shard.manifest
    };

    Ok(PendingCImageShard {
        manifest,
        ..mlp_shard
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitnet_emit_single_linear() {
        let tensor = BitNetImporter::import_ternary_tensor(42, 8, 128, 32).unwrap();
        let shard = emit_single_bitnet_linear("test_linear", &tensor).unwrap();
        assert_eq!(shard.manifest.tensors.len(), 1);
        assert_eq!(shard.manifest.tensors[0].codec, CodecFamily::Ternary1_58);
        assert_eq!(shard.payloads.len(), 2); // codes + scales
    }

    #[test]
    fn test_bitnet_emit_mlp_block() {
        let shard = emit_bitnet_mlp_block(42, 256, 1024, 32).unwrap();
        assert_eq!(shard.manifest.tensors.len(), 3);
        assert_eq!(shard.payloads.len(), 6); // codes + scales for each of 3 tensors
    }

    #[test]
    fn test_bitnet_emit_decoder_layer() {
        let shard = emit_bitnet_decoder_layer(42, 256, 1024, 8, 32, 32).unwrap();
        assert_eq!(shard.manifest.tensors.len(), 3);
    }
}
