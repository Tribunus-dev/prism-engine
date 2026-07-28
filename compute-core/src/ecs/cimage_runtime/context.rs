//! Cimage runtime context — resolves generation payloads from ContentStore
//! into per-tensor byte buffers for Metal dispatch.
//!
//! Bridges the canonical generation model with the Metal runtime by loading
//! and organizing weight, scale, and kernel data from a ContentStore.

use half::f16;
use std::collections::BTreeMap;

use crate::ecs::canonical::generation::CimageGeneration;
use crate::ecs::canonical::kernel_abi::{ArtifactProvenance, KernelSemanticId};
use crate::ecs::legacy_cimage::generation_store::ContentStore;
use crate::ecs::cimage_runtime::tensor_store::{
    RuntimeTensor, RuntimeTensorPayload, RuntimeTensorStore,
};
use prism_ecs_ir::cimage_types::{
    CodecFamily, LogicalTensorId, PhysicalSegmentId, PhysicalTileLayout,
};

/// Runtime context built from a loaded generation via a ContentStore.
///
/// Holds the generation and all resolved payload bytes from the ContentStore,
/// organized by physical segment for direct Metal buffer creation.
/// Every weight, scale, residual, and bias segment referenced by the
/// generation's tensor bindings is loaded eagerly at construction.
///
/// Missing segments produce a clear error — no silent zeros or placeholders.
#[derive(Debug, Clone)]
pub struct CimageRuntimeContext {
    /// The generation being executed.
    pub generation: CimageGeneration,
    /// Pre-resolved runtime tensor store (tensor_key → payload).
    pub tensor_store: RuntimeTensorStore,
    /// Raw payload bytes per physical segment for direct Metal buffer creation.
    pub payloads: BTreeMap<PhysicalSegmentId, Vec<u8>>,
    /// Compiled kernel artifact provenance per semantic id.
    pub kernel_artifacts: BTreeMap<KernelSemanticId, ArtifactProvenance>,
}

impl CimageRuntimeContext {
    /// Build a runtime context by loading every payload segment referenced by
    /// `generation` from `store`.
    ///
    /// Iterates each tensor binding in the generation:
    /// 1. Retrieves the primary segment bytes from ContentStore
    /// 2. Retrieves scale and residual segments from ContentStore
    /// 3. Decodes raw segment bytes into a [`RuntimeTensorPayload`] using the
    ///    binding's [`CodecFamily`]
    /// 4. Populates the context's payload map and runtime tensor store
    ///
    /// Returns an error listing the first missing segment if any tensor
    /// binding's primary, scale, or residual segment is absent from the store.
    /// On success, the context is ready for Metal buffer creation or CPU
    /// fallback execution.
    pub fn load_from_generation(
        generation: CimageGeneration,
        store: &ContentStore,
    ) -> Result<Self, String> {
        let mut payloads: BTreeMap<PhysicalSegmentId, Vec<u8>> = BTreeMap::new();
        let mut tensor_store = RuntimeTensorStore::new();

        for (tensor_id, binding) in &generation.tensor_bindings {
            // ── Load primary segment ────────────────────────────────────
            let primary = store.get(&binding.primary_segment).ok_or_else(|| {
                format!(
                    "missing primary segment {:?} for tensor {:?}",
                    binding.primary_segment, tensor_id
                )
            })?;
            payloads.insert(binding.primary_segment.clone(), primary.to_vec());

            // ── Load scale segments ─────────────────────────────────────
            let mut scale_data: Vec<Vec<u8>> = Vec::with_capacity(binding.scale_segments.len());
            for seg_id in &binding.scale_segments {
                let data = store.get(seg_id).ok_or_else(|| {
                    format!(
                        "missing scale segment {:?} for tensor {:?}",
                        seg_id, tensor_id
                    )
                })?;
                payloads.insert(seg_id.clone(), data.to_vec());
                scale_data.push(data.to_vec());
            }

            // ── Load residual segments ──────────────────────────────────
            for seg_id in &binding.residual_segments {
                let data = store.get(seg_id).ok_or_else(|| {
                    format!(
                        "missing residual segment {:?} for tensor {:?}",
                        seg_id, tensor_id
                    )
                })?;
                payloads.insert(seg_id.clone(), data.to_vec());
            }

            // ── Decode segment bytes into runtime payload ───────────────
            let payload =
                decode_segment_to_payload(binding.codec, primary, &scale_data, &binding.layout)?;

            // Use logical_shape from the tile layout.
            let logical_shape = vec![
                binding.layout.logical_shape[0],
                binding.layout.logical_shape[1],
            ];

            let runtime_tensor = RuntimeTensor {
                tensor_id: tensor_id.0.clone(),
                tensor_key: tensor_id.0.clone(),
                tensor_class: "weight".into(),
                logical_shape,
                codec: binding.codec,
                payload,
            };
            tensor_store.insert(runtime_tensor);
        }

        Ok(Self {
            generation,
            tensor_store,
            payloads,
            kernel_artifacts: BTreeMap::new(),
        })
    }

    /// Return the raw weight bytes for a logical tensor, if its primary
    /// segment was loaded.
    pub fn get_weight_bytes(&self, tensor: &LogicalTensorId) -> Option<&[u8]> {
        let binding = self.generation.tensor_bindings.get(tensor)?;
        self.payloads
            .get(&binding.primary_segment)
            .map(|v| v.as_slice())
    }

    /// Return all scale segment bytes for a logical tensor, concatenated
    /// in binding order.
    pub fn get_scale_bytes(&self, tensor: &LogicalTensorId) -> Vec<&[u8]> {
        let binding = match self.generation.tensor_bindings.get(tensor) {
            Some(b) => b,
            None => return vec![],
        };
        binding
            .scale_segments
            .iter()
            .filter_map(|sid| self.payloads.get(sid).map(|v| v.as_slice()))
            .collect()
    }

    /// Return all residual segment bytes for a logical tensor.
    pub fn get_residual_bytes(&self, tensor: &LogicalTensorId) -> Vec<&[u8]> {
        let binding = match self.generation.tensor_bindings.get(tensor) {
            Some(b) => b,
            None => return vec![],
        };
        binding
            .residual_segments
            .iter()
            .filter_map(|sid| self.payloads.get(sid).map(|v| v.as_slice()))
            .collect()
    }

    /// Check whether all tensors referenced by the generation's bindings
    /// have their payloads present. Returns false if any segment is missing.
    pub fn is_complete(&self) -> bool {
        for binding in self.generation.tensor_bindings.values() {
            if !self.payloads.contains_key(&binding.primary_segment) {
                return false;
            }
            for sid in &binding.scale_segments {
                if !self.payloads.contains_key(sid) {
                    return false;
                }
            }
            for sid in &binding.residual_segments {
                if !self.payloads.contains_key(sid) {
                    return false;
                }
            }
        }
        true
    }
}

/// Decode raw segment bytes into a [`RuntimeTensorPayload`] according to
/// the specified codec family.
///
/// # Supported codecs
///
/// | Codec      | Primary segment            | Scale segments          |
/// |------------|----------------------------|-------------------------|
/// | `RawF32`   | f32 LE bytes               | ignored                 |
/// | `Fp16`     | u16 LE bytes               | ignored                 |
/// | `Int8Packed` | packed codes bytes      | f32 LE scale factors    |
/// | `Nf4Packed`  | packed codes bytes      | f32 LE scale factors    |
/// | `Ternary`  | 2-bit packed codes         | f16 LE scales           |
/// | `SymInt4`  | packed 4-bit codes         | f32 LE scale factors    |
///
/// Returns an error for unsupported codec families.
fn decode_segment_to_payload(
    codec: CodecFamily,
    primary: &[u8],
    scales: &[Vec<u8>],
    _layout: &PhysicalTileLayout,
) -> Result<RuntimeTensorPayload, String> {
    match codec {
        CodecFamily::RawF32 => {
            if primary.len() % 4 != 0 {
                return Err(format!(
                    "RawF32 segment length {} is not a multiple of 4",
                    primary.len()
                ));
            }
            let values: Vec<f32> = primary
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            Ok(RuntimeTensorPayload::RawF32(values))
        }

        CodecFamily::Fp16 => {
            if primary.len() % 2 != 0 {
                return Err(format!(
                    "Fp16 segment length {} is not a multiple of 2",
                    primary.len()
                ));
            }
            let values: Vec<u16> = primary
                .chunks_exact(2)
                .map(|b| u16::from_le_bytes([b[0], b[1]]))
                .collect();
            Ok(RuntimeTensorPayload::Fp16(values))
        }

        CodecFamily::Int8 => {
            // Int8 tile640 packed: codes are raw bytes, scales are f32.
            let scale_data = flatten_scale_segments(scales);
            let (scale_values, bias_values) = decode_f32_pair(&scale_data);
            Ok(RuntimeTensorPayload::Int8Packed {
                codes: primary.to_vec(),
                scales: scale_values,
                biases: bias_values,
            })
        }

        CodecFamily::Nf4 => {
            // NF4 packed: codes are raw bytes, scales are f32 with group_size
            // derived from the layout.
            let scale_data = flatten_scale_segments(scales);
            let (scale_values, bias_values) = decode_f32_pair(&scale_data);
            let group_size = _layout.group_size as usize;
            Ok(RuntimeTensorPayload::Nf4Packed {
                codes: primary.to_vec(),
                scales: scale_values,
                biases: bias_values,
                group_size,
            })
        }

        CodecFamily::Ternary | CodecFamily::Ternary1_58 => {
            // Ternary: 2-bit packed codes, scales are f16.
            let scale_data = flatten_scale_segments(scales);
            // For ternary, scale entries are f16 LE. We store them as
            // a raw byte vector since the runner handles them directly.
            Ok(RuntimeTensorPayload::RawF32(
                scale_data
                    .chunks_exact(2)
                    .map(|b| f32::from(f16::from_le_bytes([b[0], b[1]])))
                    .collect(),
            ))
        }

        CodecFamily::SymInt4 => {
            // Symmetric INT4: packed 4-bit codes, f32 scale factors.
            let scale_data = flatten_scale_segments(scales);
            let (scale_values, bias_values) = decode_f32_pair(&scale_data);
            Ok(RuntimeTensorPayload::Int8Packed {
                codes: primary.to_vec(),
                scales: scale_values,
                biases: bias_values,
            })
        }

        CodecFamily::Q8_0 => {
            // Q8_0: GGML-style block quantization. Store as raw bytes;
            // the Metal kernel handles dequantization.
            Ok(RuntimeTensorPayload::Int8Packed {
                codes: primary.to_vec(),
                scales: vec![],
                biases: vec![],
            })
        }

        CodecFamily::Q4_K | CodecFamily::Q2_K | CodecFamily::IQ2_XXS => {
            // GGML-style block quantization variants. Store as raw packed
            // bytes; the Metal kernel handles dequantization.
            Ok(RuntimeTensorPayload::Int8Packed {
                codes: primary.to_vec(),
                scales: vec![],
                biases: vec![],
            })
        }

        CodecFamily::Mixed => Err(
            "Mixed codec requires per-tensor decoding context; use the manifest path instead"
                .into(),
        ),
    }
}

/// Flatten multiple scale segment byte vectors into one contiguous vector.
fn flatten_scale_segments(segments: &[Vec<u8>]) -> Vec<u8> {
    let total: usize = segments.iter().map(|s| s.len()).sum();
    let mut out = Vec::with_capacity(total);
    for seg in segments {
        out.extend_from_slice(seg);
    }
    out
}

/// Decode interleaved f32 scale+pair data. Returns (scales, biases).
///
/// When the byte slice length is even, the first half is treated as scale
/// factors and the second half as biases. When there is no second half,
/// biases are empty.
fn decode_f32_pair(data: &[u8]) -> (Vec<f32>, Vec<f32>) {
    if data.len() < 4 {
        return (vec![], vec![]);
    }
    let count = data.len() / 4;
    let half = count / 2;
    let mut scales = Vec::with_capacity(half);
    let mut biases = Vec::with_capacity(half);

    for i in 0..half {
        let off = i * 4;
        let scale = f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
        scales.push(scale);
    }
    for i in 0..half {
        let off = (half + i) * 4;
        if off + 4 <= data.len() {
            let bias = f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
            biases.push(bias);
        }
    }
    (scales, biases)
}

/// Extract a layer index from a tensor identifier string.
///
/// Looks for a `layer.N` pattern (e.g. `"layer.0.q_proj.weight"`) and returns
/// the parsed index. Returns 0 when no layer pattern is found.
pub fn extract_layer_index(tensor_id: &str) -> usize {
    if let Some(dot) = tensor_id.find('.') {
        let rest = &tensor_id[dot + 1..];
        if rest.starts_with("layer.") {
            let after = &rest[6..]; // skip "layer."
            if let Some(dot2) = after.find('.') {
                if let Ok(idx) = after[..dot2].parse::<usize>() {
                    return idx;
                }
            }
        }
    }
    0
}
