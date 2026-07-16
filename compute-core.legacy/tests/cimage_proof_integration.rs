//! Integration tests for CIMAGE-EMISSION-PROOF-0001.
//!
//! Tests the full roundtrip: build synthetic MLP shard → write cimage → load →
//! validate → reconstruct packed path → compare output against RawF32 reference.

#![cfg(test)]

use tribunus_compute_core::cimage::*;
use tribunus_compute_core::execution_plan::CodecFamily;

/// Test a full RawF32 MLP shard roundtrip — expected to produce near-exact output.
#[test]
fn test_integration_rawf32_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("rawf32_test.cimage");

    let config = SyntheticMlpShardConfig {
        seed: 42,
        hidden_dim: 64,
        intermediate_dim: 128,
        policy: SyntheticShardPolicy {
            gate_codec: CodecFamily::RawF32,
            up_codec: CodecFamily::RawF32,
            down_codec: CodecFamily::RawF32,
            rmsnorm_codec: CodecFamily::RawF32,
            allow_mixed_precision: false,
        },
    };

    let (write_rcpt, load_rcpt, shard_val) =
        emit_and_validate_synthetic_mlp(&path, config).expect("rawf32 roundtrip should succeed");

    assert_eq!(write_rcpt.tensor_count, 4);
    assert_eq!(load_rcpt.validation_status, CImageValidationStatus::Valid);
    assert!(
        shard_val.passed,
        "RawF32 shard should pass numerical validation: NRMSE={:.6} cosine={:.6}",
        shard_val.output_nrmse, shard_val.output_cosine
    );
    assert!(
        shard_val.max_abs_error < 1e-4,
        "RawF32 max_abs_error should be near zero: {}",
        shard_val.max_abs_error
    );
}

/// Test an INT8 MLP shard roundtrip — expected to pass with tight numerical gates.
#[test]
fn test_integration_int8_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("int8_test.cimage");

    let config = SyntheticMlpShardConfig {
        seed: 42,
        hidden_dim: 64,
        intermediate_dim: 128,
        policy: SyntheticShardPolicy {
            gate_codec: CodecFamily::Int8,
            up_codec: CodecFamily::Int8,
            down_codec: CodecFamily::Int8,
            rmsnorm_codec: CodecFamily::RawF32,
            allow_mixed_precision: false,
        },
    };

    let (write_rcpt, load_rcpt, shard_val) =
        emit_and_validate_synthetic_mlp(&path, config).expect("int8 roundtrip should succeed");

    assert_eq!(write_rcpt.tensor_count, 4);
    assert_eq!(load_rcpt.validation_status, CImageValidationStatus::Valid);
    assert!(
        shard_val.passed,
        "INT8 shard should pass: NRMSE={:.6} cosine={:.6}",
        shard_val.output_nrmse, shard_val.output_cosine
    );
}

/// Test an NF4 MLP shard roundtrip — expected to pass with relaxed numerical gates.
#[test]
fn test_integration_nf4_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("nf4_test.cimage");

    let config = SyntheticMlpShardConfig {
        seed: 42,
        hidden_dim: 64,
        intermediate_dim: 128,
        policy: SyntheticShardPolicy {
            gate_codec: CodecFamily::Nf4,
            up_codec: CodecFamily::Nf4,
            down_codec: CodecFamily::Nf4,
            rmsnorm_codec: CodecFamily::RawF32,
            allow_mixed_precision: false,
        },
    };

    let (write_rcpt, load_rcpt, shard_val) =
        emit_and_validate_synthetic_mlp(&path, config).expect("nf4 roundtrip should succeed");

    assert_eq!(write_rcpt.tensor_count, 4);
    assert_eq!(load_rcpt.validation_status, CImageValidationStatus::Valid);
    assert!(
        shard_val.passed,
        "NF4 shard should pass: NRMSE={:.6} cosine={:.6} max_abs_err={:.6}",
        shard_val.output_nrmse, shard_val.output_cosine, shard_val.max_abs_error
    );
}

/// Test mixed codecs: gate NF4, down INT8.
#[test]
fn test_integration_mixed_codec_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mixed_test.cimage");

    let config = SyntheticMlpShardConfig {
        seed: 42,
        hidden_dim: 64,
        intermediate_dim: 128,
        policy: SyntheticShardPolicy {
            gate_codec: CodecFamily::Nf4,
            up_codec: CodecFamily::Nf4,
            down_codec: CodecFamily::Int8,
            rmsnorm_codec: CodecFamily::RawF32,
            allow_mixed_precision: false,
        },
    };

    let (write_rcpt, load_rcpt, shard_val) = emit_and_validate_synthetic_mlp(&path, config)
        .expect("mixed codec roundtrip should succeed");

    assert_eq!(write_rcpt.tensor_count, 4);
    assert_eq!(load_rcpt.validation_status, CImageValidationStatus::Valid);
    assert!(
        shard_val.passed,
        "Mixed codec shard should pass: NRMSE={:.6} cosine={:.6}",
        shard_val.output_nrmse, shard_val.output_cosine
    );
}

/// Test that the loader rejects a corrupted cimage file.
#[test]
fn test_integration_rejects_corrupted_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("corrupted.cimage");

    // Write a valid cimage first
    let config = SyntheticMlpShardConfig {
        seed: 42,
        hidden_dim: 64,
        intermediate_dim: 128,
        policy: SyntheticShardPolicy {
            gate_codec: CodecFamily::RawF32,
            up_codec: CodecFamily::RawF32,
            down_codec: CodecFamily::RawF32,
            rmsnorm_codec: CodecFamily::RawF32,
            allow_mixed_precision: false,
        },
    };

    let pending = MlpShardBuilder::build_synthetic_mlp_shard(config).unwrap();
    CImageWriter::write_v0(&path, pending.manifest, pending.payloads, pending.receipts).unwrap();

    // Corrupt one byte in the payload blob
    let mut data = std::fs::read(&path).unwrap();
    let header_size = std::mem::size_of::<tribunus_compute_core::cimage::header::CImageHeaderV0>();
    // Find the footer offset from the header
    use std::io::Read;
    let mut file = std::fs::File::open(&path).unwrap();
    let mut header_bytes = vec![0u8; header_size];
    file.read_exact(&mut header_bytes).unwrap();
    let header: tribunus_compute_core::cimage::header::CImageHeaderV0 =
        bincode::deserialize(&header_bytes).unwrap();

    // Corrupt a byte in the payload blob
    let corrupt_offset = (header.payload_blob_offset + header.payload_blob_len / 2) as usize;
    if corrupt_offset < data.len() {
        data[corrupt_offset] ^= 0xFF;
        std::fs::write(&path, &data).unwrap();
    }

    // Loading should succeed (loader doesn't validate digests)
    let loaded = CImageLoader::load_v0(&path).unwrap();

    // Validation should detect the corruption
    let receipt = CImageValidator::validate_loaded(&loaded).unwrap();
    assert_eq!(
        receipt.validation_status,
        CImageValidationStatus::Invalid,
        "corrupted cimage should be invalid: errors: {:?}",
        receipt.errors
    );
    assert!(!receipt.errors.is_empty(), "should have at least one error");
}

/// Test that a cimage with bad magic is rejected.
#[test]
fn test_integration_rejects_bad_magic() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad_magic.cimage");

    // Write a valid cimage
    let config = SyntheticMlpShardConfig {
        seed: 42,
        hidden_dim: 64,
        intermediate_dim: 128,
        policy: SyntheticShardPolicy {
            gate_codec: CodecFamily::RawF32,
            up_codec: CodecFamily::RawF32,
            down_codec: CodecFamily::RawF32,
            rmsnorm_codec: CodecFamily::RawF32,
            allow_mixed_precision: false,
        },
    };

    let pending = MlpShardBuilder::build_synthetic_mlp_shard(config).unwrap();
    CImageWriter::write_v0(&path, pending.manifest, pending.payloads, pending.receipts).unwrap();

    // Corrupt the magic
    let mut data = std::fs::read(&path).unwrap();
    data[0] = 0x00;
    std::fs::write(&path, &data).unwrap();

    // Loading should fail on bad magic
    let result = CImageLoader::load_v0(&path);
    assert!(result.is_err(), "loading a file with bad magic should fail");
}

/// Test that we can manually load and inspect a cimage.
#[test]
fn test_integration_inspect_loaded_cimage() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("inspect_test.cimage");

    let config = SyntheticMlpShardConfig {
        seed: 42,
        hidden_dim: 64,
        intermediate_dim: 128,
        policy: SyntheticShardPolicy {
            gate_codec: CodecFamily::RawF32,
            up_codec: CodecFamily::RawF32,
            down_codec: CodecFamily::RawF32,
            rmsnorm_codec: CodecFamily::RawF32,
            allow_mixed_precision: false,
        },
    };

    let pending = MlpShardBuilder::build_synthetic_mlp_shard(config).unwrap();

    // Verify manifest structure before writing
    assert_eq!(pending.manifest.tensors.len(), 4);
    assert_eq!(pending.manifest.tensors[0].tensor_key, "rmsnorm_weight");
    assert_eq!(pending.manifest.tensors[1].tensor_key, "gate_proj");
    assert_eq!(pending.manifest.tensors[2].tensor_key, "up_proj");
    assert_eq!(pending.manifest.tensors[3].tensor_key, "down_proj");
    assert_eq!(pending.manifest.execution_plan.tensor_refs.len(), 4);
    assert_eq!(
        pending.manifest.artifact_kind,
        CImageArtifactKind::SyntheticShard
    );

    // Write and reload
    CImageWriter::write_v0(&path, pending.manifest, pending.payloads, pending.receipts).unwrap();

    let loaded = CImageLoader::load_v0(&path).unwrap();
    assert_eq!(loaded.manifest.tensors.len(), 4);
}
