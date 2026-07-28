//! CImage runtime resolver — converts loaded cimage artifacts into
//! runtime-resolvable tensors and reference bundles.

use sha2::{Digest, Sha256};

use crate::ecs::legacy_cimage::mlp_reference::{load_mlp_shard_tensors, run_mlp_reconstructed_reference};
use crate::ecs::legacy_cimage::{CImageManifestV0, CImagePayloadRef, CImageTensorEntry, LoadedCImageV0};
use crate::execution_plan::CodecFamily;

use super::error::{CImageRuntimeError, CImageRuntimeResult};
use super::tensor_store::{RuntimeTensor, RuntimeTensorPayload, RuntimeTensorStore};

/// The resolved runtime bundle for an MLP shard.
#[derive(Debug, Clone)]
pub struct ResolvedMlpShardRuntime {
    pub cimage_digest: String,
    pub manifest: CImageManifestV0,
    pub tensors: RuntimeTensorStore,
    pub cpu_reconstructed: Vec<f32>,
    pub cpu_rawf32_reference: Vec<f32>,
    pub hidden_dim: usize,
    pub intermediate_dim: usize,
    pub cpu_reference_bundle: CpuReferenceBundle,
}

/// Reference math results for comparison.
#[derive(Debug, Clone)]
pub struct CpuReferenceBundle {
    pub reconstructed_output: Vec<f32>,
    pub rawf32_output: Vec<f32>,
    pub reconstructed_digest: String,
    pub rawf32_digest: String,
    pub input_digest: String,
}

/// Runtime resolver — the bridge between cimage format and execution.
pub struct CImageRuntimeResolver;

impl CImageRuntimeResolver {
    /// Resolve an MLP shard cimage into runtime-resolvable tensors.
    ///
    /// This extracts tensor data from the cimage payload blob and decodes it
    /// into the RuntimeTensorStore format. It also computes the CPU reference
    /// outputs for numerical comparison.
    pub fn resolve_mlp_shard(
        image: &LoadedCImageV0,
    ) -> CImageRuntimeResult<ResolvedMlpShardRuntime> {
        let cimage_digest = compute_digest_for_cimage(image);
        let manifest = &image.manifest;

        // Build tensor store from manifest entries
        let mut store = RuntimeTensorStore::new();
        for entry in &manifest.tensors {
            let tensor = resolve_tensor_entry(image, entry)?;
            store.insert(tensor);
        }

        let hidden_dim = manifest.tensors[0].logical_shape[0] as usize;
        let intermediate_dim = manifest.tensors[1].logical_shape[0] as usize;

        // Compute CPU reference outputs
        let tensors = load_mlp_shard_tensors(image)?;
        let deterministic_input = generate_deterministic_input(42, hidden_dim);

        let input_digest = sha256_hex_f32(&deterministic_input);

        let reconstructed_output = run_mlp_reconstructed_reference(&deterministic_input, &tensors)
            .map_err(|e| CImageRuntimeError::CImage(e))?;
        let reconstructed_digest = sha256_hex_f32(&reconstructed_output);

        // Extract raw f32 reference
        let rmsnorm_raw = extract_rawf32(image, 0)?;
        let gate_raw = extract_rawf32(image, 1)?;
        let up_raw = extract_rawf32(image, 2)?;
        let down_raw = extract_rawf32(image, 3)?;
        let rawf32_output = crate::ecs::legacy_cimage::mlp_reference::run_mlp_rawf32_reference(
            &deterministic_input,
            &rmsnorm_raw,
            &gate_raw,
            &up_raw,
            &down_raw,
            hidden_dim,
            intermediate_dim,
        );
        let rawf32_digest = sha256_hex_f32(&rawf32_output);

        Ok(ResolvedMlpShardRuntime {
            cimage_digest,
            manifest: manifest.clone(),
            tensors: store,
            cpu_reconstructed: reconstructed_output.clone(),
            cpu_rawf32_reference: rawf32_output.clone(),
            hidden_dim,
            intermediate_dim,
            cpu_reference_bundle: CpuReferenceBundle {
                reconstructed_output,
                rawf32_output,
                reconstructed_digest,
                rawf32_digest,
                input_digest,
            },
        })
    }
}

/// Resolve one tensor entry into a RuntimeTensor.
fn resolve_tensor_entry(
    image: &LoadedCImageV0,
    entry: &CImageTensorEntry,
) -> CImageRuntimeResult<RuntimeTensor> {
    let payload = resolve_payload(image, entry)?;

    Ok(RuntimeTensor {
        tensor_id: entry.tensor_id.clone(),
        tensor_key: entry.tensor_key.clone(),
        tensor_class: entry.tensor_class.clone(),
        logical_shape: entry.logical_shape.clone(),
        codec: entry.codec,
        payload,
    })
}

/// Resolve the payload data for a single tensor entry.
fn resolve_payload(
    image: &LoadedCImageV0,
    entry: &CImageTensorEntry,
) -> CImageRuntimeResult<RuntimeTensorPayload> {
    match &entry.payload_ref {
        CImagePayloadRef::Single { payload_id } => {
            resolve_single_payload(image, payload_id, entry.codec)
        }
        CImagePayloadRef::MixedPrecision { .. } => {
            Err(CImageRuntimeError::UnsupportedCodec(CodecFamily::Mixed))
        }
    }
}

/// Resolve a single payload reference into a runtime payload.
fn resolve_single_payload(
    image: &LoadedCImageV0,
    payload_id: &str,
    codec: CodecFamily,
) -> CImageRuntimeResult<RuntimeTensorPayload> {
    let payload_entry = image
        .payload_directory
        .payloads
        .iter()
        .find(|e| e.payload_id == payload_id)
        .ok_or_else(|| CImageRuntimeError::MissingPayload(payload_id.to_string()))?;

    let start = payload_entry.offset as usize;
    let end = start + payload_entry.len as usize;
    let blob = &image.payload_blob[start..end];

    match codec {
        CodecFamily::RawF32 => {
            let f32s: Vec<f32> = blob
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            Ok(RuntimeTensorPayload::RawF32(f32s))
        }
        CodecFamily::Int8 | CodecFamily::Nf4 => {
            // Look for metadata payload
            let metadata_id = format!("{payload_id}_metadata");
            let meta_entry = image
                .payload_directory
                .payloads
                .iter()
                .find(|e| e.payload_id == metadata_id);

            let (scales, biases) = if let Some(meta) = meta_entry {
                let mstart = meta.offset as usize;
                let mend = mstart + meta.len as usize;
                let meta_bytes = &image.payload_blob[mstart..mend];
                let all_f32: Vec<f32> = meta_bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                let half = all_f32.len() / 2;
                (all_f32[..half].to_vec(), all_f32[half..].to_vec())
            } else {
                (vec![], vec![])
            };

            let codes = blob.to_vec();
            let group_size = if codec == CodecFamily::Nf4 { 32 } else { 640 };

            match codec {
                CodecFamily::Int8 => Ok(RuntimeTensorPayload::Int8Packed {
                    codes,
                    scales,
                    biases,
                }),
                CodecFamily::Nf4 => Ok(RuntimeTensorPayload::Nf4Packed {
                    codes,
                    scales,
                    biases,
                    group_size,
                }),
                _ => unreachable!(),
            }
        }
        _ => Err(CImageRuntimeError::UnsupportedCodec(codec)),
    }
}

/// Extract the RawF32 reference payload for a tensor by manifest index.
fn extract_rawf32(image: &LoadedCImageV0, tensor_idx: usize) -> CImageRuntimeResult<Vec<f32>> {
    let tensor = &image.manifest.tensors[tensor_idx];
    let Some(CImagePayloadRef::Single { payload_id }) = &tensor.raw_f32_reference_ref else {
        return Err(CImageRuntimeError::MissingPayload(format!(
            "tensor {} has no raw_f32_reference_ref",
            tensor.tensor_id
        )));
    };
    let entry = image
        .payload_directory
        .payloads
        .iter()
        .find(|e| e.payload_id == *payload_id)
        .ok_or_else(|| CImageRuntimeError::MissingPayload(payload_id.clone()))?;

    let start = entry.offset as usize;
    let end = start + entry.len as usize;
    let bytes = &image.payload_blob[start..end];
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

/// Compute a cimage digest from the loaded image.
fn compute_digest_for_cimage(image: &LoadedCImageV0) -> String {
    // Compute sha256 of the raw_file_bytes (excluding footer)
    let file_len = image.raw_file_bytes.len();
    let footer_size = std::mem::size_of::<crate::ecs::legacy_cimage::header::CImageFooterV0>();
    let digest_len = file_len.saturating_sub(footer_size);
    let bytes = &image.raw_file_bytes[..digest_len];
    format!("{:x}", Sha256::digest(bytes))
}

/// Generate deterministic input matching mlp_reference's approach.
fn generate_deterministic_input(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed;
    let mut data = Vec::with_capacity(n);
    for _ in 0..n {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let val = ((state >> 11) as f64) / (1u64 << 53) as f64;
        data.push((val * 2.0 - 1.0) as f32);
    }
    data
}

/// SHA-256 of an f32 slice.
fn sha256_hex_f32(data: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for &v in data {
        hasher.update(v.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::legacy_cimage::*;
    use crate::execution_plan::CodecFamily;

    fn build_test_cimage(codec: CodecFamily) -> (tempfile::TempDir, LoadedCImageV0) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.cimage");

        let config = SyntheticMlpShardConfig {
            seed: 42,
            hidden_dim: 64,
            intermediate_dim: 128,
            policy: SyntheticShardPolicy {
                gate_codec: codec,
                up_codec: codec,
                down_codec: codec,
                rmsnorm_codec: CodecFamily::RawF32,
                allow_mixed_precision: false,
            },
        };

        let pending = MlpShardBuilder::build_synthetic_mlp_shard(config).unwrap();
        CImageWriter::write_v0(&path, pending.manifest, pending.payloads, pending.receipts)
            .unwrap();
        let loaded = CImageLoader::load_v0(&path).unwrap();
        (dir, loaded)
    }

    #[test]
    fn test_resolve_mlp_shard_finds_four_tensors() {
        let (_dir, image) = build_test_cimage(CodecFamily::RawF32);
        let resolved = CImageRuntimeResolver::resolve_mlp_shard(&image).unwrap();
        assert_eq!(resolved.tensors.len(), 4);
        assert_eq!(resolved.hidden_dim, 64);
        assert_eq!(resolved.intermediate_dim, 128);
    }

    #[test]
    fn test_resolve_mlp_shard_tensor_keys() {
        let (_dir, image) = build_test_cimage(CodecFamily::Nf4);
        let resolved = CImageRuntimeResolver::resolve_mlp_shard(&image).unwrap();
        let ids = resolved.tensors.tensor_ids();
        assert!(ids.contains(&"t0"));
        assert!(ids.contains(&"t1"));
        assert!(ids.contains(&"t2"));
        assert!(ids.contains(&"t3"));
    }

    #[test]
    fn test_resolve_mlp_shard_rejects_missing_tensor() {
        let (_dir, image) = build_test_cimage(CodecFamily::RawF32);
        // Create a manifest with a tensor that has a bad payload ref
        let mut bad_image = image.clone();
        if let Some(tensor) = bad_image.manifest.tensors.first_mut() {
            tensor.payload_ref = CImagePayloadRef::Single {
                payload_id: "nonexistent".into(),
            };
        }
        let result = CImageRuntimeResolver::resolve_mlp_shard(&bad_image);
        assert!(result.is_err(), "should fail on missing payload");
    }

    #[test]
    fn test_resolve_int8_tensor_payload() {
        let (_dir, image) = build_test_cimage(CodecFamily::Int8);
        let resolved = CImageRuntimeResolver::resolve_mlp_shard(&image).unwrap();
        let gate = resolved.tensors.get("t1").unwrap();
        match &gate.payload {
            RuntimeTensorPayload::Int8Packed { codes, .. } => {
                assert!(!codes.is_empty());
            }
            _ => panic!("expected Int8Packed payload"),
        }
    }

    #[test]
    fn test_resolve_nf4_tensor_payload() {
        let (_dir, image) = build_test_cimage(CodecFamily::Nf4);
        let resolved = CImageRuntimeResolver::resolve_mlp_shard(&image).unwrap();
        let gate = resolved.tensors.get("t1").unwrap();
        match &gate.payload {
            RuntimeTensorPayload::Nf4Packed {
                codes, group_size, ..
            } => {
                assert!(!codes.is_empty());
                assert_eq!(*group_size, 32);
            }
            _ => panic!("expected Nf4Packed payload"),
        }
    }

    #[test]
    fn test_cpu_reference_computed() {
        let (_dir, image) = build_test_cimage(CodecFamily::RawF32);
        let resolved = CImageRuntimeResolver::resolve_mlp_shard(&image).unwrap();
        assert_eq!(resolved.cpu_reference_bundle.reconstructed_output.len(), 64);
        assert_eq!(resolved.cpu_reference_bundle.rawf32_output.len(), 64);
        assert!(!resolved
            .cpu_reference_bundle
            .reconstructed_digest
            .is_empty());
    }
}
