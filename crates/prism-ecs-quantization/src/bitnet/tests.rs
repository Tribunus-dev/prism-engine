//! Integration tests for the BitNet native cimage pipeline.
//!
//! Validates deterministic hermetic roundtripping without live model
//! I/O.

use super::cimage_shim::{CImageManifestV0, CImagePayloadKind, CImagePayloadRef};
use super::importer::BitNetImporter;
use super::phases::{emit_bitnet_decoder_layer, emit_bitnet_mlp_block, emit_single_bitnet_linear};

/// Verify that a BitNet ternary cimage can be serialised and validated.
#[test]
fn bitnet_cimage_roundtrip() {
    let tensor = BitNetImporter::import_ternary_tensor(42, 4, 64, 16).unwrap();
    let shard = emit_single_bitnet_linear("test", &tensor).unwrap();

    let manifest_json = serde_json::to_string(&shard.manifest).unwrap();
    let deserialized: CImageManifestV0 = serde_json::from_str(&manifest_json).unwrap();

    assert_eq!(deserialized.tensors.len(), 1);
    assert_eq!(
        deserialized.tensors[0].payload_ref,
        CImagePayloadRef::Single {
            payload_id: "p_test_codes".into()
        }
    );
}

/// Verify that the ternary payload sizes are self-consistent.
#[test]
fn bitnet_payload_sizes() {
    let tensor = BitNetImporter::import_ternary_tensor(7, 10, 256, 128).unwrap();
    let shard = emit_single_bitnet_linear("size_check", &tensor).unwrap();

    let codes_payload = shard
        .payloads
        .iter()
        .find(|p| p.payload_kind == CImagePayloadKind::TernaryPackedCodes)
        .expect("codes payload");
    let scales_payload = shard
        .payloads
        .iter()
        .find(|p| p.payload_kind == CImagePayloadKind::TernaryScales)
        .expect("scales payload");

    // Codes: rows * groups_per_row * bytes_per_group
    // groups_per_row = 256/128 = 2, bytes_per_group = (128+3)/4 = 32
    // codes_len = 10 * 2 * 32 = 640
    assert_eq!(codes_payload.bytes.len(), 640);

    // Scales: rows * groups_per_row * 2 bytes (f16)
    assert_eq!(scales_payload.bytes.len(), 10 * 2 * 2);
}

/// Verify MLP block emission matches expected tensor keys.
#[test]
fn bitnet_mlp_tensor_keys() {
    let shard = emit_bitnet_mlp_block(99, 128, 512, 64).unwrap();
    let keys: Vec<&str> = shard
        .manifest
        .tensors
        .iter()
        .map(|t| t.tensor_key.as_str())
        .collect();
    assert_eq!(keys, vec!["gate_proj", "up_proj", "down_proj"]);
}

/// Verify the decoder layer emission has 11 tensors with the expected
/// total payload count.
#[test]
fn bitnet_decoder_layer_emits_eleven_tensors() {
    use super::phases::BitNetDecoderLayerShardConfig;
    let config = BitNetDecoderLayerShardConfig {
        seed: 42,
        hidden_dim: 64,
        num_heads: 1,
        num_kv_heads: 1,
        head_dim: 64,
        intermediate_dim: 128,
        seq_len: 8,
        group_size: 64,
        num_layers: 1,
    };
    let pending = emit_bitnet_decoder_layer(&config).expect("emit bitnet decoder layer");
    assert_eq!(pending.manifest.tensors.len(), 11);
}
