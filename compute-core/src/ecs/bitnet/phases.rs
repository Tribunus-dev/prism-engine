//! Phased BitNet cimage emission — linear, MLP block, and decoder layer.
//!
//! Each phase uses the existing cimage shard builder infrastructure to
//! emit a `PendingCImageShard` from BitNet-native ternary weights.

use crate::ecs::bitnet::checkpoint::BitNetCheckpoint;
use crate::ecs::bitnet::importer::BitNetImporter;
use crate::ecs::cimage::streaming_writer::StreamingCImageWriter;
use crate::ecs::cimage::*;
use crate::execution_plan::{CodecFamily, DType, HardwareProfileId};
use crate::ternary::codec::TernaryPackedTensor;
use crate::ternary::pack::unpack_ternary_codes;
use sha2::{Digest, Sha256};
use std::path::Path;

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
/// Configuration for emitting a BitNet decoder layer or full model cimage shard.
#[derive(Debug, Clone)]
pub struct BitNetDecoderLayerShardConfig {
    pub seed: u64,
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_dim: usize,
    pub seq_len: usize,
    pub group_size: usize,
    pub num_layers: usize,
}

/// Build a `PhysicalTileLayout` for a flat RawF32 tensor.
fn rawf32_tile_layout(data_len: usize) -> PhysicalTileLayout {
    PhysicalTileLayout {
        tile_m: 1,
        tile_n: data_len as u32,
        tiles_per_row: 1,
        total_tiles: 1,
        padded_cols: data_len as u32,
        group_size: 0,
        groups_per_tile: 0,
        packed_bytes_per_tile: (data_len * 4) as u32,
        metadata_f32_per_tile: 0,
    }
}

/// Helper: emit one ternary tensor with codes + scales + RawF32Reference payloads.
fn emit_ternary_decoder_tensor(
    all_payloads: &mut Vec<PendingPayload>,
    entries: &mut Vec<CImageTensorEntry>,
    tensor_idx: &mut usize,
    tensor_key: &str,
    tensor_class: &str,
    seed: u64,
    rows: usize,
    cols: usize,
    group_size: usize,
) -> CImageResult<()> {
    let tensor = BitNetImporter::import_ternary_tensor(seed, rows, cols, group_size)
        .map_err(|e| CImageError::Other(format!("import {tensor_key}: {e}")))?;

    // RawF32 reference: ternary {-1, 0, +1} values as f32.
    let weights = BitNetImporter::generate_ternary_weights(seed, rows * cols);
    let raw_f32_bytes: Vec<u8> = weights
        .iter()
        .flat_map(|&w| (w as f32).to_le_bytes())
        .collect();

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
    let raw_f32_payload = PendingPayload {
        payload_id: format!("p_{}_rawf32", tensor_key),
        payload_kind: CImagePayloadKind::RawF32Reference,
        codec: Some("RawF32".into()),
        alignment_bytes: 64,
        bytes: raw_f32_bytes,
    };

    all_payloads.push(codes_payload);
    all_payloads.push(scales_payload);
    all_payloads.push(raw_f32_payload);

    let entry = CImageTensorEntry {
        tensor_id: format!("t{}", *tensor_idx),
        tensor_key: tensor_key.to_string(),
        tensor_class: tensor_class.to_string(),
        logical_shape: vec![rows as u32, cols as u32],
        source_dtype: DType::F32,
        codec: CodecFamily::Ternary1_58,
        precision_plan: None,
        physical_layout: ternary_layout(&tensor),
        payload_ref: CImagePayloadRef::Single {
            payload_id: format!("p_{}_codes", tensor_key),
        },
        raw_f32_reference_ref: Some(CImagePayloadRef::Single {
            payload_id: format!("p_{}_rawf32", tensor_key),
        }),
        tensor_sha256: sha256_of_bytes(&tensor.codes),
        validation_digest: None,
    };
    entries.push(entry);
    *tensor_idx = tensor_idx.wrapping_add(1);

    Ok(())
}

/// Helper: emit one RawF32 tensor (e.g. position_ids) with codes + RawF32Reference.
fn emit_rawf32_tensor(
    all_payloads: &mut Vec<PendingPayload>,
    entries: &mut Vec<CImageTensorEntry>,
    tensor_idx: &mut usize,
    tensor_key: &str,
    tensor_class: &str,
    data: &[f32],
) -> CImageResult<()> {
    let raw_bytes: Vec<u8> = data.iter().flat_map(|&v| v.to_le_bytes()).collect();
    let raw_sha256 = sha256_of_bytes(&raw_bytes);

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

    let layout = rawf32_tile_layout(data.len());

    let entry = CImageTensorEntry {
        tensor_id: format!("t{}", *tensor_idx),
        tensor_key: tensor_key.to_string(),
        tensor_class: tensor_class.to_string(),
        logical_shape: vec![data.len() as u32, 1u32],
        source_dtype: DType::F32,
        codec: CodecFamily::RawF32,
        precision_plan: None,
        physical_layout: layout,
        payload_ref: CImagePayloadRef::Single {
            payload_id: format!("p_{}_codes", tensor_key),
        },
        raw_f32_reference_ref: Some(CImagePayloadRef::Single {
            payload_id: format!("p_{}_rawf32", tensor_key),
        }),
        tensor_sha256: raw_sha256,
        validation_digest: None,
    };
    entries.push(entry);
    *tensor_idx = tensor_idx.wrapping_add(1);

    Ok(())
}

/// Emit one full BitNet decoder layer (11 tensors: attention + MLP + norms + position_ids).
pub fn emit_bitnet_decoder_layer(
    config: &BitNetDecoderLayerShardConfig,
) -> CImageResult<PendingCImageShard> {
    let kv_inner = config.num_kv_heads * config.head_dim;

    let mut all_payloads: Vec<PendingPayload> = Vec::new();
    let mut entries: Vec<CImageTensorEntry> = Vec::with_capacity(11);
    let mut tensor_idx = 0usize;

    // 1. input_layernorm.weight — norm, single scale (group_size = hidden_dim)
    emit_ternary_decoder_tensor(
        &mut all_payloads,
        &mut entries,
        &mut tensor_idx,
        "input_layernorm.weight",
        "RmsNormWeight",
        config.seed,
        1,
        config.hidden_dim,
        config.hidden_dim,
    )?;

    // 2. q_proj.weight — attention Q projection
    emit_ternary_decoder_tensor(
        &mut all_payloads,
        &mut entries,
        &mut tensor_idx,
        "q_proj.weight",
        "AttentionProjection",
        config.seed.wrapping_add(1),
        config.hidden_dim,
        config.hidden_dim,
        config.group_size,
    )?;

    // 3. k_proj.weight — attention K projection
    emit_ternary_decoder_tensor(
        &mut all_payloads,
        &mut entries,
        &mut tensor_idx,
        "k_proj.weight",
        "AttentionProjection",
        config.seed.wrapping_add(2),
        config.hidden_dim,
        kv_inner,
        config.group_size,
    )?;

    // 4. v_proj.weight — attention V projection
    emit_ternary_decoder_tensor(
        &mut all_payloads,
        &mut entries,
        &mut tensor_idx,
        "v_proj.weight",
        "AttentionProjection",
        config.seed.wrapping_add(3),
        config.hidden_dim,
        kv_inner,
        config.group_size,
    )?;

    // 5. o_proj.weight — attention O projection
    emit_ternary_decoder_tensor(
        &mut all_payloads,
        &mut entries,
        &mut tensor_idx,
        "o_proj.weight",
        "AttentionProjection",
        config.seed.wrapping_add(4),
        config.hidden_dim,
        config.hidden_dim,
        config.group_size,
    )?;

    // 6. post_attention_layernorm.weight — norm
    emit_ternary_decoder_tensor(
        &mut all_payloads,
        &mut entries,
        &mut tensor_idx,
        "post_attention_layernorm.weight",
        "RmsNormWeight",
        config.seed.wrapping_add(5),
        1,
        config.hidden_dim,
        config.hidden_dim,
    )?;

    // 7. gate_proj.weight — MLP gate projection
    emit_ternary_decoder_tensor(
        &mut all_payloads,
        &mut entries,
        &mut tensor_idx,
        "gate_proj.weight",
        "DecoderMlpProjection",
        config.seed.wrapping_add(6),
        config.hidden_dim,
        config.intermediate_dim,
        config.group_size,
    )?;

    // 8. up_proj.weight — MLP up projection
    emit_ternary_decoder_tensor(
        &mut all_payloads,
        &mut entries,
        &mut tensor_idx,
        "up_proj.weight",
        "DecoderMlpProjection",
        config.seed.wrapping_add(7),
        config.hidden_dim,
        config.intermediate_dim,
        config.group_size,
    )?;

    // 9. down_proj.weight — MLP down projection
    emit_ternary_decoder_tensor(
        &mut all_payloads,
        &mut entries,
        &mut tensor_idx,
        "down_proj.weight",
        "DecoderMlpProjection",
        config.seed.wrapping_add(8),
        config.intermediate_dim,
        config.hidden_dim,
        config.group_size,
    )?;

    // 10. position_ids — sequential token positions, RawF32
    let pos_ids: Vec<f32> = (0..config.seq_len).map(|i| i as f32).collect();
    emit_rawf32_tensor(
        &mut all_payloads,
        &mut entries,
        &mut tensor_idx,
        "position_ids",
        "PositionIds",
        &pos_ids,
    )?;

    // 11. rmsnorm_w — same data as input_layernorm
    emit_ternary_decoder_tensor(
        &mut all_payloads,
        &mut entries,
        &mut tensor_idx,
        "rmsnorm_w",
        "RmsNormWeight",
        config.seed,
        1,
        config.hidden_dim,
        config.hidden_dim,
    )?;

    let plan_id = format!("bitnet_decoder_layer_{:016x}", config.seed);
    let manifest = CImageManifestV0 {
        schema_version: 0,
        model_family: "BitNet-b1.58-2B4T".into(),
        artifact_kind: CImageArtifactKind::SyntheticShard,
        source_model_digest: None,
        compiler_policy_digest: "bitnet-native-ternary".into(),
        layout_profile: HardwareProfileId::AppleMProBalanced,
        tensors: entries,
        execution_plan: ModelExecutionPlanSummary {
            plan_id,
            region_count: 2,
            total_kernel_ops: 11,
            total_input_bytes: (config.hidden_dim * 4) as u64,
            total_output_bytes: (config.hidden_dim * 4) as u64,
            tensor_refs: (0..11u32).map(|i| format!("t{}", i)).collect(),
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

/// Emit a full BitNet model by stacking `num_layers` decoder layers.
///
/// Each layer's tensors are prefixed with `"layer.{N}."`.
pub fn emit_bitnet_full_model(
    config: &BitNetDecoderLayerShardConfig,
) -> CImageResult<PendingCImageShard> {
    let mut all_payloads: Vec<PendingPayload> = Vec::new();
    let mut entries: Vec<CImageTensorEntry> = Vec::with_capacity(config.num_layers * 11);
    let mut tensor_idx = 0usize;

    for layer_idx in 0..config.num_layers {
        let layer_seed = config.seed.wrapping_add(layer_idx as u64);
        let prefix = format!("layer.{layer_idx}.");
        let kv_inner = config.num_kv_heads * config.head_dim;

        // 1. input_layernorm.weight
        emit_ternary_decoder_tensor(
            &mut all_payloads,
            &mut entries,
            &mut tensor_idx,
            &format!("{}input_layernorm.weight", prefix),
            "RmsNormWeight",
            layer_seed,
            1,
            config.hidden_dim,
            config.hidden_dim,
        )?;

        // 2. q_proj.weight
        emit_ternary_decoder_tensor(
            &mut all_payloads,
            &mut entries,
            &mut tensor_idx,
            &format!("{}q_proj.weight", prefix),
            "AttentionProjection",
            layer_seed.wrapping_add(1),
            config.hidden_dim,
            config.hidden_dim,
            config.group_size,
        )?;

        // 3. k_proj.weight
        emit_ternary_decoder_tensor(
            &mut all_payloads,
            &mut entries,
            &mut tensor_idx,
            &format!("{}k_proj.weight", prefix),
            "AttentionProjection",
            layer_seed.wrapping_add(2),
            config.hidden_dim,
            kv_inner,
            config.group_size,
        )?;

        // 4. v_proj.weight

        // 4. v_proj.weight
        emit_ternary_decoder_tensor(
            &mut all_payloads,
            &mut entries,
            &mut tensor_idx,
            &format!("{}v_proj.weight", prefix),
            "AttentionProjection",
            layer_seed.wrapping_add(3),
            config.hidden_dim,
            kv_inner,
            config.group_size,
        )?;
        emit_ternary_decoder_tensor(
            &mut all_payloads,
            &mut entries,
            &mut tensor_idx,
            &format!("{}o_proj.weight", prefix),
            "AttentionProjection",
            layer_seed.wrapping_add(4),
            config.hidden_dim,
            config.hidden_dim,
            config.group_size,
        )?;

        // 6. post_attention_layernorm.weight
        emit_ternary_decoder_tensor(
            &mut all_payloads,
            &mut entries,
            &mut tensor_idx,
            &format!("{}post_attention_layernorm.weight", prefix),
            "RmsNormWeight",
            layer_seed.wrapping_add(5),
            1,
            config.hidden_dim,
            config.hidden_dim,
        )?;

        // 7. gate_proj.weight
        emit_ternary_decoder_tensor(
            &mut all_payloads,
            &mut entries,
            &mut tensor_idx,
            &format!("{}gate_proj.weight", prefix),
            "DecoderMlpProjection",
            layer_seed.wrapping_add(6),
            config.hidden_dim,
            config.intermediate_dim,
            config.group_size,
        )?;

        // 8. up_proj.weight
        emit_ternary_decoder_tensor(
            &mut all_payloads,
            &mut entries,
            &mut tensor_idx,
            &format!("{}up_proj.weight", prefix),
            "DecoderMlpProjection",
            layer_seed.wrapping_add(7),
            config.hidden_dim,
            config.intermediate_dim,
            config.group_size,
        )?;

        // 9. down_proj.weight
        emit_ternary_decoder_tensor(
            &mut all_payloads,
            &mut entries,
            &mut tensor_idx,
            &format!("{}down_proj.weight", prefix),
            "DecoderMlpProjection",
            layer_seed.wrapping_add(8),
            config.intermediate_dim,
            config.hidden_dim,
            config.group_size,
        )?;

        // 10. position_ids (RawF32)
        let pos_ids: Vec<f32> = (0..config.seq_len).map(|i| i as f32).collect();
        emit_rawf32_tensor(
            &mut all_payloads,
            &mut entries,
            &mut tensor_idx,
            &format!("{}position_ids", prefix),
            "PositionIds",
            &pos_ids,
        )?;

        // 11. rmsnorm_w
        emit_ternary_decoder_tensor(
            &mut all_payloads,
            &mut entries,
            &mut tensor_idx,
            &format!("{}rmsnorm_w", prefix),
            "RmsNormWeight",
            layer_seed,
            1,
            config.hidden_dim,
            config.hidden_dim,
        )?;
    }

    let total_tensors = config.num_layers * 11;
    let plan_id = format!("bitnet_full_model_{:016x}", config.seed);
    let manifest = CImageManifestV0 {
        schema_version: 0,
        model_family: "BitNet-b1.58-2B4T".into(),
        artifact_kind: CImageArtifactKind::FullModel,
        source_model_digest: None,
        compiler_policy_digest: "bitnet-native-ternary".into(),
        layout_profile: HardwareProfileId::AppleMProBalanced,
        tensors: entries,
        execution_plan: ModelExecutionPlanSummary {
            plan_id,
            region_count: (config.num_layers as u32) * 2,
            total_kernel_ops: total_tensors as u32,
            total_input_bytes: (config.hidden_dim * 4) as u64,
            total_output_bytes: (config.hidden_dim * 4) as u64,
            tensor_refs: (0..total_tensors as u32)
                .map(|i| format!("t{}", i))
                .collect(),
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

/// Emit a full BitNet model cimage shard from a real safetensors checkpoint.
///
/// Loads the checkpoint, iterates all layers, extracts real codes + scales +
/// norms, and builds a single combined `PendingCImageShard`.
///
/// Per-layer tensors (11 per layer):
/// - 4 RawF32 norms: input_layernorm, post_attention_layernorm,
///   attn_sub_norm, ffn_sub_norm
/// - 7 ternary weights: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj,
///   down_proj
///
/// Global tensors: embed_tokens, position_ids, final_layernorm.
pub fn emit_bitnet_from_checkpoint(
    checkpoint_path: &Path,
    output_path: &Path,
    group_size: usize,
) -> CImageResult<CImageWriteReceipt> {
    let resolved_checkpoint = if checkpoint_path.is_dir() {
        checkpoint_path.join("model.safetensors")
    } else {
        checkpoint_path.to_path_buf()
    };
    let path_str = resolved_checkpoint.display();
    // Read the entire checkpoint into memory for fast random access
    let buffer = std::fs::read(&resolved_checkpoint)
        .map_err(|e| CImageError::Other(format!("failed to read {path_str}: {e}")))?;
    let ckpt = BitNetCheckpoint::from_buffer(buffer)
        .map_err(|e| CImageError::Other(format!("failed to parse checkpoint {path_str}: {e}")))?;

    let num_layers = ckpt.num_layers;
    let hidden_dim = ckpt.hidden_dim;
    let intermediate_dim = ckpt.intermediate_dim;
    let kv_inner = ckpt.num_kv_heads * ckpt.head_dim;

    // Create streaming writer — tensors go directly to disk, no memory accumulation
    let mut writer = StreamingCImageWriter::new(output_path)?;

    let mut entries: Vec<CImageTensorEntry> = Vec::with_capacity(num_layers * 11 + 3);
    let mut tensor_idx = 0usize;

    // ── Global tensors ─────────────────────────────────────────────────

    // embed_tokens.weight (RawF32 from BF16)
    let embed_bytes = ckpt
        .embed_tokens()
        .map_err(|e| CImageError::Other(format!("embed_tokens: {e}")))?;
    stream_rawf32_norm_tensor(
        &mut writer,
        &mut entries,
        &mut tensor_idx,
        "embed_tokens.weight",
        "EmbedTokens",
        &embed_bytes,
    )?;

    // ── position_ids ────────────────────────────────────────────
    const SEQ_LEN: usize = 4096;
    let pos_ids: Vec<f32> = (0..SEQ_LEN).map(|i| i as f32).collect();
    let mut pos_bytes = Vec::with_capacity(SEQ_LEN * 4);
    for f in &pos_ids {
        pos_bytes.extend_from_slice(&f.to_le_bytes());
    }
    stream_rawf32_norm_tensor(
        &mut writer,
        &mut entries,
        &mut tensor_idx,
        "position_ids",
        "PositionIds",
        &pos_bytes,
    )?;

    // ── Decoder layers ─────────────────────────────────────────────────

    for layer in 0..num_layers {
        let prefix = format!("layer.{layer}.");

        // 1. input_layernorm.weight — RawF32
        let ln_bytes = ckpt
            .layer_norm_weight(layer, "input_layernorm")
            .map_err(|e| CImageError::Other(format!("layer {layer} input_layernorm: {e}")))?;
        stream_rawf32_norm_tensor(
            &mut writer,
            &mut entries,
            &mut tensor_idx,
            &format!("{}input_layernorm.weight", prefix),
            "RmsNormWeight",
            &ln_bytes,
        )?;

        // 2-5. Attention projections (ternary)
        for (name, ckpt_key, rows, cols) in &[
            ("q_proj", "self_attn.q_proj", hidden_dim, hidden_dim),
            ("k_proj", "self_attn.k_proj", kv_inner, hidden_dim),
            ("v_proj", "self_attn.v_proj", kv_inner, hidden_dim),
            ("o_proj", "self_attn.o_proj", hidden_dim, hidden_dim),
        ] {
            stream_checkpoint_ternary_tensor(
                &mut writer,
                &mut entries,
                &mut tensor_idx,
                &ckpt,
                layer,
                &format!("{prefix}{name}.weight"),
                ckpt_key,
                "AttentionProjection",
                *rows,
                *cols,
                group_size,
            )?;
        }

        // 6. post_attention_layernorm.weight — RawF32
        let paln_bytes = ckpt
            .layer_norm_weight(layer, "post_attention_layernorm")
            .map_err(|e| {
                CImageError::Other(format!("layer {layer} post_attention_layernorm: {e}"))
            })?;
        stream_rawf32_norm_tensor(
            &mut writer,
            &mut entries,
            &mut tensor_idx,
            &format!("{}post_attention_layernorm.weight", prefix),
            "RmsNormWeight",
            &paln_bytes,
        )?;

        // 7-9. MLP projections (ternary)
        stream_checkpoint_ternary_tensor(
            &mut writer,
            &mut entries,
            &mut tensor_idx,
            &ckpt,
            layer,
            &format!("{}gate_proj.weight", prefix),
            "mlp.gate_proj",
            "DecoderMlpProjection",
            intermediate_dim,
            hidden_dim,
            group_size,
        )?;
        stream_checkpoint_ternary_tensor(
            &mut writer,
            &mut entries,
            &mut tensor_idx,
            &ckpt,
            layer,
            &format!("{}up_proj.weight", prefix),
            "mlp.up_proj",
            "DecoderMlpProjection",
            intermediate_dim,
            hidden_dim,
            group_size,
        )?;
        stream_checkpoint_ternary_tensor(
            &mut writer,
            &mut entries,
            &mut tensor_idx,
            &ckpt,
            layer,
            &format!("{}down_proj.weight", prefix),
            "mlp.down_proj",
            "DecoderMlpProjection",
            hidden_dim,
            intermediate_dim,
            group_size,
        )?;

        // 10. ffn_sub_norm.weight — RawF32
        let ffn_sub = ckpt
            .layer_ffn_sub_norm(layer)
            .map_err(|e| CImageError::Other(format!("layer {layer} ffn_sub_norm: {e}")))?;
        stream_rawf32_norm_tensor(
            &mut writer,
            &mut entries,
            &mut tensor_idx,
            &format!("{}ffn_sub_norm.weight", prefix),
            "RmsNormWeight",
            &ffn_sub,
        )?;

        // 11. attn_sub_norm.weight — RawF32
        let attn_sub = ckpt
            .layer_attn_sub_norm(layer)
            .map_err(|e| CImageError::Other(format!("layer {layer} attn_sub_norm: {e}")))?;
        stream_rawf32_norm_tensor(
            &mut writer,
            &mut entries,
            &mut tensor_idx,
            &format!("{}attn_sub_norm.weight", prefix),
            "RmsNormWeight",
            &attn_sub,
        )?;
    }

    // ── Global norm ────────────────────────────────────────────────────
    if let Ok(final_ln_bytes) = ckpt.final_layernorm() {
        stream_rawf32_norm_tensor(
            &mut writer,
            &mut entries,
            &mut tensor_idx,
            "final_layernorm.weight",
            "RmsNormWeight",
            &final_ln_bytes,
        )?;
    }

    // ── Manifest ───────────────────────────────────────────────────────
    let total_tensors = entries.len();
    let plan_id = format!("bitnet_from_checkpoint_{:016x}", rand_positive_u64());
    let manifest = CImageManifestV0 {
        schema_version: 0,
        model_family: "BitNet-b1.58-2B4T".into(),
        artifact_kind: CImageArtifactKind::FullModel,
        source_model_digest: None,
        compiler_policy_digest: "bitnet-native-ternary".into(),
        layout_profile: HardwareProfileId::AppleMProBalanced,
        tensors: entries,
        execution_plan: ModelExecutionPlanSummary {
            plan_id,
            region_count: (num_layers as u32) * 2,
            total_kernel_ops: total_tensors as u32,
            total_input_bytes: (hidden_dim * 4) as u64,
            total_output_bytes: (hidden_dim * 4) as u64,
            tensor_refs: (0..total_tensors as u32).map(|i| format!("t{i}")).collect(),
        },
        receipts: Vec::new(),
        assistant_graph: None,
        state_store_schema: None,
    };

    // Finalize: writes manifest + directories + footer, atomic rename
    writer.finalize(manifest)
}

#[allow(dead_code)]
/// Emit a RawF32 norm tensor from BF16 checkpoint data (already converted to
/// f32 LE bytes).
fn emit_rawf32_norm_tensor(
    all_payloads: &mut Vec<PendingPayload>,
    entries: &mut Vec<CImageTensorEntry>,
    tensor_idx: &mut usize,
    tensor_key: &str,
    tensor_class: &str,
    data: &[u8],
) -> CImageResult<()> {
    let n_elements = data.len() / 4;
    let raw_sha256 = sha256_of_bytes(data);

    let codes_payload = PendingPayload {
        payload_id: format!("p_{tensor_key}_codes"),
        payload_kind: CImagePayloadKind::PackedTensorCodes,
        codec: Some("RawF32".into()),
        alignment_bytes: 64,
        bytes: data.to_vec(),
    };
    let raw_ref_payload = PendingPayload {
        payload_id: format!("p_{tensor_key}_rawf32"),
        payload_kind: CImagePayloadKind::RawF32Reference,
        codec: Some("RawF32".into()),
        alignment_bytes: 64,
        bytes: data.to_vec(),
    };
    all_payloads.push(codes_payload);
    all_payloads.push(raw_ref_payload);

    let layout = rawf32_tile_layout(n_elements);
    let entry = CImageTensorEntry {
        tensor_id: format!("t{}", *tensor_idx),
        tensor_key: tensor_key.to_string(),
        tensor_class: tensor_class.to_string(),
        logical_shape: vec![n_elements as u32, 1u32],
        source_dtype: DType::F32,
        codec: CodecFamily::RawF32,
        precision_plan: None,
        physical_layout: layout,
        payload_ref: CImagePayloadRef::Single {
            payload_id: format!("p_{tensor_key}_codes"),
        },
        raw_f32_reference_ref: Some(CImagePayloadRef::Single {
            payload_id: format!("p_{tensor_key}_rawf32"),
        }),
        tensor_sha256: raw_sha256,
        validation_digest: None,
    };
    entries.push(entry);
    *tensor_idx = tensor_idx.wrapping_add(1);
    Ok(())
}

#[allow(dead_code)]
/// Emit a ternary weight tensor from checkpoint data.
///
/// Loads U8 codes + BF16 scale from the checkpoint for the given layer and
/// checkpoint tensor name, builds a `TernaryPackedTensor`, and emits three
/// payloads: codes (packed), scales (f16), and a RawF32 reference (unpacked
/// values × scale).
fn emit_checkpoint_ternary_tensor(
    ckpt: &BitNetCheckpoint,
    all_payloads: &mut Vec<PendingPayload>,
    entries: &mut Vec<CImageTensorEntry>,
    tensor_idx: &mut usize,
    layer: usize,
    tensor_key: &str,
    checkpoint_name: &str, // e.g. "self_attn.q_proj"
    tensor_class: &str,
    out_features: usize,
    in_features: usize,
    group_size: usize,
) -> CImageResult<()> {
    use crate::ecs::bitnet::checkpoint::make_ternary_from_checkpoint;

    let stored_rows = out_features / 4;
    let stored_cols = in_features;

    let codes = ckpt
        .layer_codes(layer, checkpoint_name)
        .map_err(|e| CImageError::Other(format!("{tensor_key} codes: {e}")))?;
    let scale = ckpt
        .layer_scale(layer, checkpoint_name)
        .map_err(|e| CImageError::Other(format!("{tensor_key} scale: {e}")))?;

    let tensor = make_ternary_from_checkpoint(codes, stored_rows, stored_cols, scale, group_size);

    // Codes payload.
    let codes_payload = PendingPayload {
        payload_id: format!("p_{tensor_key}_codes"),
        payload_kind: CImagePayloadKind::TernaryPackedCodes,
        codec: Some("Ternary1_58".into()),
        alignment_bytes: 64,
        bytes: tensor.codes.clone(),
    };
    // Scales payload (f16 LE bytes).
    let scale_bytes: Vec<u8> = tensor.scales.iter().flat_map(|s| s.to_le_bytes()).collect();
    let scales_payload = PendingPayload {
        payload_id: format!("p_{tensor_key}_scales"),
        payload_kind: CImagePayloadKind::TernaryScales,
        codec: Some("Ternary1_58".into()),
        alignment_bytes: 64,
        bytes: scale_bytes,
    };
    // RawF32 reference: unpack ternary codes × scale.
    let total_values = tensor.rows * tensor.cols;
    let unpacked = unpack_ternary_codes(&tensor.codes, total_values)
        .map_err(|e| CImageError::Other(format!("{tensor_key} unpack: {e}")))?;
    let raw_f32_bytes: Vec<u8> = unpacked
        .iter()
        .flat_map(|&v| ((v as f32) * scale).to_le_bytes())
        .collect();
    let raw_f32_payload = PendingPayload {
        payload_id: format!("p_{tensor_key}_rawf32"),
        payload_kind: CImagePayloadKind::RawF32Reference,
        codec: Some("RawF32".into()),
        alignment_bytes: 64,
        bytes: raw_f32_bytes,
    };
    all_payloads.push(codes_payload);
    all_payloads.push(scales_payload);
    all_payloads.push(raw_f32_payload);

    let entry = CImageTensorEntry {
        tensor_id: format!("t{}", *tensor_idx),
        tensor_key: tensor_key.to_string(),
        tensor_class: tensor_class.to_string(),
        logical_shape: vec![tensor.rows as u32, tensor.cols as u32],
        source_dtype: DType::F32,
        codec: CodecFamily::Ternary1_58,
        precision_plan: None,
        physical_layout: ternary_layout(&tensor),
        payload_ref: CImagePayloadRef::Single {
            payload_id: format!("p_{tensor_key}_codes"),
        },
        raw_f32_reference_ref: Some(CImagePayloadRef::Single {
            payload_id: format!("p_{tensor_key}_rawf32"),
        }),
        tensor_sha256: sha256_of_bytes(&tensor.codes),
        validation_digest: None,
    };
    entries.push(entry);
    *tensor_idx = tensor_idx.wrapping_add(1);
    Ok(())
}

/// Generate a random positive u64 seed from `/dev/urandom`.
/// Generate a random positive u64 seed from system time.
fn rand_positive_u64() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

/// Write a ternary tensor payload to a streaming writer directly to disk.
fn stream_checkpoint_ternary_tensor(
    writer: &mut StreamingCImageWriter,
    entries: &mut Vec<CImageTensorEntry>,
    tensor_idx: &mut usize,
    ckpt: &BitNetCheckpoint,
    layer: usize,
    tensor_key: &str,
    ckpt_proj_name: &str,
    tensor_class: &str,
    out_features: usize,
    in_features: usize,
    group_size: usize,
) -> CImageResult<()> {
    use crate::ecs::bitnet::checkpoint::make_ternary_from_checkpoint;

    let codes = ckpt
        .layer_codes(layer, ckpt_proj_name)
        .map_err(|e| CImageError::Other(format!("{tensor_key} codes: {e}")))?;
    let scale = ckpt
        .layer_scale(layer, ckpt_proj_name)
        .map_err(|e| CImageError::Other(format!("{tensor_key} scale: {e}")))?;

    let stored_rows = out_features / 4;
    let stored_cols = in_features;
    let tensor = make_ternary_from_checkpoint(codes, stored_rows, stored_cols, scale, group_size);

    // Write codes payload directly to disk
    let codes_payload_id = format!("p_{}_codes", tensor_key);
    writer.append_payload(
        codes_payload_id.clone(),
        CImagePayloadKind::TernaryPackedCodes,
        Some("Ternary1_58".into()),
        64,
        &tensor.codes,
    )?;

    // Write scales payload directly to disk
    let scales_payload_id = format!("p_{}_scales", tensor_key);
    let scale_bytes: Vec<u8> = tensor.scales.iter().flat_map(|s| s.to_le_bytes()).collect();
    writer.append_payload(
        scales_payload_id.clone(),
        CImagePayloadKind::TernaryScales,
        Some("Ternary1_58".into()),
        64,
        &scale_bytes,
    )?;

    let groups_per_row = in_features.div_ceil(group_size);
    let bytes_per_group = (group_size + 3) / 4;
    let tid = format!("t{}", *tensor_idx);
    let entry = CImageTensorEntry {
        tensor_id: tid,
        tensor_key: tensor_key.to_string(),
        tensor_class: tensor_class.to_string(),
        logical_shape: vec![out_features as u32, in_features as u32],
        source_dtype: DType::F32,
        codec: CodecFamily::Ternary1_58,
        precision_plan: None,
        physical_layout: PhysicalTileLayout {
            tile_m: 1,
            tile_n: in_features as u32,
            tiles_per_row: 1,
            total_tiles: out_features as u32,
            padded_cols: (groups_per_row * group_size) as u32,
            group_size: group_size as u32,
            groups_per_tile: groups_per_row as u32,
            packed_bytes_per_tile: (groups_per_row * bytes_per_group) as u32,
            metadata_f32_per_tile: groups_per_row as u32 / 2,
        },
        payload_ref: CImagePayloadRef::Single {
            payload_id: codes_payload_id,
        },
        raw_f32_reference_ref: None,
        tensor_sha256: sha256_of_bytes(&tensor.codes),
        validation_digest: None,
    };
    entries.push(entry);
    *tensor_idx = tensor_idx.wrapping_add(1);
    Ok(())
}

/// Write a RawF32 norm tensor payload to a streaming writer.
fn stream_rawf32_norm_tensor(
    writer: &mut StreamingCImageWriter,
    entries: &mut Vec<CImageTensorEntry>,
    tensor_idx: &mut usize,
    tensor_key: &str,
    tensor_class: &str,
    data: &[u8],
) -> CImageResult<()> {
    let elements = data.len() / 4;
    let payload_id = format!("p_{}", tensor_key.replace('.', "_"));
    writer.append_payload(
        payload_id.clone(),
        CImagePayloadKind::RawF32Reference,
        None,
        64,
        data,
    )?;
    let tid = format!("t{}", *tensor_idx);
    let entry = CImageTensorEntry {
        tensor_id: tid,
        tensor_key: tensor_key.to_string(),
        tensor_class: tensor_class.to_string(),
        logical_shape: vec![elements as u32],
        source_dtype: DType::F32,
        codec: CodecFamily::RawF32,
        precision_plan: None,
        physical_layout: PhysicalTileLayout {
            tile_m: 1,
            tile_n: elements as u32,
            tiles_per_row: 1,
            total_tiles: 1,
            padded_cols: elements as u32,
            group_size: elements as u32,
            groups_per_tile: 1,
            packed_bytes_per_tile: data.len() as u32,
            metadata_f32_per_tile: 0,
        },
        payload_ref: CImagePayloadRef::Single { payload_id },
        raw_f32_reference_ref: None,
        tensor_sha256: sha256_of_bytes(data),
        validation_digest: None,
    };
    entries.push(entry);
    *tensor_idx = tensor_idx.wrapping_add(1);
    Ok(())
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
        let config = BitNetDecoderLayerShardConfig {
            seed: 42,
            hidden_dim: 256,
            num_heads: 8,
            num_kv_heads: 4,
            head_dim: 32,
            intermediate_dim: 1024,
            seq_len: 64,
            group_size: 32,
            num_layers: 1,
        };
        let shard = emit_bitnet_decoder_layer(&config).unwrap();
        // 11 tensors per decoder layer
        assert_eq!(shard.manifest.tensors.len(), 11);
        // 10 ternary tensors × 3 payloads (codes + scales + raw_f32) + 1 RawF32 × 2 payloads = 32
        assert_eq!(shard.payloads.len(), 32);
    }

    #[test]
    fn test_bitnet_emit_full_model() {
        let config = BitNetDecoderLayerShardConfig {
            seed: 42,
            hidden_dim: 128,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 32,
            intermediate_dim: 512,
            seq_len: 32,
            group_size: 32,
            num_layers: 2,
        };
        let shard = emit_bitnet_full_model(&config).unwrap();
        assert_eq!(shard.manifest.tensors.len(), 22); // 2 layers × 11 tensors
                                                      // Verify layer-prefixed keys
        assert!(shard
            .manifest
            .tensors
            .iter()
            .any(|t| t.tensor_key == "layer.0.input_layernorm.weight"));
        assert!(shard
            .manifest
            .tensors
            .iter()
            .any(|t| t.tensor_key == "layer.1.input_layernorm.weight"));
        // Check manifest properties
        assert_eq!(shard.manifest.artifact_kind, CImageArtifactKind::FullModel);
        assert_eq!(shard.manifest.model_family, "BitNet-b1.58-2B4T");
    }
}
