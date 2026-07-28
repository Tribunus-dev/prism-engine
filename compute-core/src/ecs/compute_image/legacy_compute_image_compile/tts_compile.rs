//! TTS weight compilation — packs Qwen3-TTS weights as nf4tile640 cimage segments.
//!
//! Reads a HuggingFace-format `.safetensors` checkpoint, categorises tensors
//! by architectural role (Talker, Code Predictor, Mimi Codec), packs each 2D
//! weight matrix as nf4tile640, aggregates all weights for each component into
//! a single weight/scale/bias triplet, and writes the segment files to a staging
//! directory alongside segment-manifest entries.
//!
//! The Talker is a 28-layer decoder-only transformer (RMSNorm + GQA + SwiGLU).
//! The Code Predictor is a 5-layer transformer producing 16 RVQ codebook indices.
//! The Mimi Codec is a causal ConvNet converting codebook tokens → 24 kHz PCM.
//!
//! Well-known output filenames (one per component triplet):
//!
//! | File                          | Cimage SegmentKind     |
//! |-------------------------------|------------------------|
//! | `tts_talker_weight.bin`       | TtsTalkerWeight (30)   |
//! | `tts_talker_scale.bin`        | TtsTalkerScale   (31)  |
//! | `tts_talker_bias.bin`         | TtsTalkerBias    (32)  |
//! | `tts_code_predictor_weight.bin` | TtsCodePredictorWeight (33) |
//! | `tts_code_predictor_scale.bin`  | TtsCodePredictorScale  (34) |
//! | `tts_code_predictor_bias.bin`   | TtsCodePredictorBias   (35) |
//! | `tts_codec_weight.bin`        | TtsCodecWeight   (36)  |
//! | `tts_codebook.bin`            | TtsCodebook      (37)  |

use crate::ecs::compute_image::legacy_compute_image_compile::quantize::quantize_nf4_tile640_matrix_from_raw;
use crate::ecs::compute_image::manifest::TensorEntry;
use std::collections::HashMap;
use std::path::Path;

// ── TTS architecture constants ──────────────────────────────────────────────

/// Expected 2D weight suffixes in the HF checkpoint.
const WEIGHT_SUFFIX: &str = ".weight";

// ── Tensor categorisation ───────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TtsComponent {
    Talker,
    CodePredictor,
    Codec,
}

impl TtsComponent {
    /// Categorise a tensor name by its prefix.
    fn from_name(name: &str) -> Option<Self> {
        if name.starts_with("transformer.h.") || name.starts_with("model.layers.") {
            Some(TtsComponent::Talker)
        } else if name.starts_with("code_predictor.") {
            Some(TtsComponent::CodePredictor)
        } else if name.starts_with("codec.") || name.starts_with("mimi.") {
            Some(TtsComponent::Codec)
        } else {
            None
        }
    }
}

// ── Raw safetensors reader ──────────────────────────────────────────────────

/// Minimal parsed tensor from a safetensors file.
#[derive(Debug)]
struct RawTensor {
    name: String,
    data: Vec<u8>,
    shape: Vec<u64>,
    dtype: String,
}

/// Read a `.safetensors` file and return all tensor entries.
fn read_safetensors(path: &Path) -> Result<Vec<RawTensor>, String> {
    let raw =
        std::fs::read(path).map_err(|e| format!("failed to read '{}': {}", path.display(), e))?;

    if raw.len() < 8 {
        return Err("file too small for safetensors header".into());
    }

    let metadata_len = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
    let metadata_json: serde_json::Value = serde_json::from_slice(&raw[8..8 + metadata_len])
        .map_err(|e| format!("invalid safetensors metadata: {}", e))?;

    let tensors = metadata_json
        .as_object()
        .ok_or_else(|| "safetensors metadata root is not an object".to_string())?;

    let data_start = 8 + metadata_len;
    let mut results = Vec::new();

    for (name, info) in tensors {
        let obj = info
            .as_object()
            .ok_or_else(|| format!("tensor '{}' info is not an object", name))?;

        let dtype = obj
            .get("dtype")
            .and_then(|v| v.as_str())
            .ok_or_else(|| format!("tensor '{}' missing dtype", name))?
            .to_string();

        let shape: Vec<u64> = obj
            .get("shape")
            .and_then(|v| v.as_array())
            .ok_or_else(|| format!("tensor '{}' missing shape", name))?
            .iter()
            .map(|v| v.as_u64().unwrap_or(0))
            .collect();

        let offsets = obj
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .ok_or_else(|| format!("tensor '{}' missing data_offsets", name))?;

        let start: usize = offsets[0]
            .as_u64()
            .ok_or_else(|| format!("tensor '{}' invalid start offset", name))?
            as usize;
        let end: usize = offsets[1]
            .as_u64()
            .ok_or_else(|| format!("tensor '{}' invalid end offset", name))?
            as usize;

        if end < start || end > raw.len() - data_start {
            return Err(format!(
                "tensor '{}' data offsets [{}, {}] out of bounds (data_len={})",
                name,
                start,
                end,
                raw.len() - data_start
            ));
        }

        results.push(RawTensor {
            name: name.clone(),
            data: raw[data_start + start..data_start + end].to_vec(),
            shape,
            dtype,
        });
    }

    Ok(results)
}

// ── NF4 tile640 packing ─────────────────────────────────────────────────────

/// Pack a single 2D weight matrix as nf4tile640.
///
/// Returns (packed_weight, packed_scales, packed_biases, padded_in_dim).
fn pack_nf4_tile640_matrix(
    data: &[u8],
    dtype: &str,
    out_dim: u32,
    in_dim: u32,
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>), String> {
    quantize_nf4_tile640_matrix_from_raw(data, dtype, out_dim, in_dim)
        .map(|(w, s, b, _packed_in, _shape)| (w, s, b))
        .map_err(|e| format!("nf4 quantize {}x{}: {}", out_dim, in_dim, e))
}

// ── Aggregation accumulator ─────────────────────────────────────────────────

/// Accumulates packed NF4 weight/scale/bias bytes for one component.
#[derive(Default)]
struct ComponentAccumulator {
    /// Concat of all packed weight bytes (all rows × tiles × 320).
    weight_bytes: Vec<u8>,
    /// Concat of all packed scale bytes (all rows × tiles × 5 × 4).
    scale_bytes: Vec<u8>,
    /// Concat of all packed bias bytes (all rows × tiles × 5 × 4).
    bias_bytes: Vec<u8>,
    /// (out_dim, in_dim) pairs for each packed tensor — records the aggregate
    /// logical shape so the runtime can locate per-tensor boundaries.
    tensor_shapes: Vec<(u32, u32)>,
}

impl ComponentAccumulator {
    fn push(&mut self, weight: Vec<u8>, scale: Vec<u8>, bias: Vec<u8>, out_dim: u32, in_dim: u32) {
        self.weight_bytes.extend(weight);
        self.scale_bytes.extend(scale);
        self.bias_bytes.extend(bias);
        self.tensor_shapes.push((out_dim, in_dim));
    }

    fn is_empty(&self) -> bool {
        self.weight_bytes.is_empty()
    }
}

// ── Top-level pack function ─────────────────────────────────────────────────

/// Pack Qwen3-TTS weights from a `.safetensors` checkpoint into nf4tile640
/// cimage segment files written to `output_dir`.
///
/// Aggregate all 2D weight matrices for each major component (Talker, Code
/// Predictor, Codec) into a single weight/scale/bias triplet segment,
/// written as well-known filenames:
///
///   `tts_talker_weight.bin` / `_scale.bin` / `_bias.bin`
///   `tts_code_predictor_weight.bin` / `_scale.bin` / `_bias.bin`
///   `tts_codec_weight.bin`
///   `tts_codebook.bin`
///
/// Returns a `Vec<TensorEntry>` for every packed tensor (weight + scale + bias).
pub fn pack_tts_weights(
    safetensors_path: &Path,
    output_dir: &Path,
) -> Result<Vec<TensorEntry>, String> {
    let all_tensors = read_safetensors(safetensors_path)?;

    let mut talker = ComponentAccumulator::default();
    let mut code_predictor = ComponentAccumulator::default();
    let mut codec = ComponentAccumulator::default();
    let mut codebook_data: Option<(Vec<u8>, Vec<u64>, String)> = None;
    let mut codebook_name = String::new();

    for tensor in &all_tensors {
        let Some(component) = TtsComponent::from_name(&tensor.name) else {
            continue;
        };

        // Detect codebook: known embedding-like tensors (stored as-is, not NF4 packed).
        let is_codebook = tensor.name.contains("codebook")
            || tensor.name.contains("embed_tokens")
            || tensor.name.contains("embedding");

        if is_codebook {
            codebook_data = Some((
                tensor.data.clone(),
                tensor.shape.clone(),
                tensor.dtype.clone(),
            ));
            codebook_name = tensor.name.clone();
            continue;
        }

        // Skip 1D weights (biases, layer-norm gains) — they store verbatim in bias segment.
        if tensor.shape.len() != 2 || !tensor.name.ends_with(WEIGHT_SUFFIX) {
            continue;
        }

        let out_dim = tensor.shape[0] as u32;
        let in_dim = tensor.shape[1] as u32;

        let (packed_weight, packed_scales, packed_biases) =
            pack_nf4_tile640_matrix(&tensor.data, &tensor.dtype, out_dim, in_dim)?;

        match component {
            TtsComponent::Talker => {
                talker.push(packed_weight, packed_scales, packed_biases, out_dim, in_dim);
            }
            TtsComponent::CodePredictor => {
                code_predictor.push(packed_weight, packed_scales, packed_biases, out_dim, in_dim);
            }
            TtsComponent::Codec => {
                codec.push(packed_weight, packed_scales, packed_biases, out_dim, in_dim);
            }
        }
    }

    let mut all_entries: Vec<TensorEntry> = Vec::new();

    // Helper: write a triplet segment + build entries.
    let mut write_triplet =
        |prefix: &str, acc: &ComponentAccumulator, _tts_seg_kind_base: u32| -> Result<(), String> {
            if acc.is_empty() {
                return Ok(());
            }

            let weight_path = output_dir.join(format!("{}_weight.bin", prefix));
            let scale_path = output_dir.join(format!("{}_scale.bin", prefix));
            let bias_path = output_dir.join(format!("{}_bias.bin", prefix));

            std::fs::write(&weight_path, &acc.weight_bytes)
                .map_err(|e| format!("write {}: {}", weight_path.display(), e))?;
            std::fs::write(&scale_path, &acc.scale_bytes)
                .map_err(|e| format!("write {}: {}", scale_path.display(), e))?;
            std::fs::write(&bias_path, &acc.bias_bytes)
                .map_err(|e| format!("write {}: {}", bias_path.display(), e))?;

            // Aggregate logical shape: total out_dim across all tensors, padded tile640 dims.
            let total_out: u32 = acc.tensor_shapes.iter().map(|(o, _)| o).sum();
            let total_in: u32 = acc
                .tensor_shapes
                .iter()
                .map(|(_, i)| (i + 639) / 640 * 640)
                .sum();

            let weight_entry = TensorEntry {
                id: 0,
                name: format!("{}.weight", prefix),
                role: "tts_weight".to_string(),
                layer: None,
                segment: weight_path
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .to_string(),
                source_filename: String::new(),
                source_sha256: String::new(),
                source_offset: 0,
                offset: 0,
                byte_length: acc.weight_bytes.len() as u64,
                logical_dtype: "BF16".to_string(),
                storage_dtype: "U8".to_string(),
                logical_shape: vec![total_out, total_in],
                physical_shape: vec![total_out, total_in],
                mutability: "frozen".to_string(),
                quantization: Some(crate::ecs::compute_image::manifest::QuantizationDesc {
                    bits: 4,
                    group_size: 128,
                    groups: (acc.scale_bytes.len() / 4) as u32,
                    scale_tensor_id: 0,
                    bias_tensor_id: 0,
                    storage_layout: Some(
                        crate::ecs::compute_image::manifest::SharedWeightLayout::Nf4Tile640(
                            crate::ecs::compute_image::manifest::Nf4Tile640Layout {
                                tile_elements: 640,
                                quant_group_size: 128,
                                groups_per_tile: 5,
                                packed_weight_bytes_per_tile: 320,
                                scale_values_per_tile: 5,
                                bias_values_per_tile: 5,
                                packed_weight_dtype: "U8".to_string(),
                                metadata_dtype: "F32".to_string(),
                                weight_lane_read_bytes: 32,
                                profile_id: None,
                            },
                        ),
                    ),
                }),
                tensor_alignment_bytes: 16,
                layout_version: 1,
                artifact_bindings: HashMap::new(),
            };

            let scale_entry = TensorEntry {
                id: 0,
                name: format!("{}.scales", prefix),
                role: "tts_scale".to_string(),
                layer: None,
                segment: scale_path
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .to_string(),
                source_filename: String::new(),
                source_sha256: String::new(),
                source_offset: 0,
                offset: 0,
                byte_length: acc.scale_bytes.len() as u64,
                logical_dtype: "F32".to_string(),
                storage_dtype: "F32".to_string(),
                logical_shape: vec![total_out, (acc.scale_bytes.len() as u32) / total_out / 4],
                physical_shape: vec![total_out, (acc.scale_bytes.len() as u32) / total_out / 4],
                mutability: "frozen".to_string(),
                quantization: None,
                tensor_alignment_bytes: 16,
                layout_version: 1,
                artifact_bindings: HashMap::new(),
            };

            let bias_entry = TensorEntry {
                id: 0,
                name: format!("{}.biases", prefix),
                role: "tts_bias".to_string(),
                layer: None,
                segment: bias_path.file_name().unwrap().to_string_lossy().to_string(),
                source_filename: String::new(),
                source_sha256: String::new(),
                source_offset: 0,
                offset: 0,
                byte_length: acc.bias_bytes.len() as u64,
                logical_dtype: "F32".to_string(),
                storage_dtype: "F32".to_string(),
                logical_shape: vec![total_out, (acc.bias_bytes.len() as u32) / total_out / 4],
                physical_shape: vec![total_out, (acc.bias_bytes.len() as u32) / total_out / 4],
                mutability: "frozen".to_string(),
                quantization: None,
                tensor_alignment_bytes: 16,
                layout_version: 1,
                artifact_bindings: HashMap::new(),
            };

            all_entries.push(weight_entry);
            all_entries.push(scale_entry);
            all_entries.push(bias_entry);
            Ok(())
        };

    write_triplet("tts_talker", &talker, 30)?;
    write_triplet("tts_code_predictor", &code_predictor, 33)?;
    write_triplet("tts_codec", &codec, 36)?;

    // Write codebook segment (raw bytes, not nf4 packed).
    if let Some((codebook_bytes, shape, dtype)) = codebook_data {
        let codebook_path = output_dir.join("tts_codebook.bin");
        std::fs::write(&codebook_path, &codebook_bytes)
            .map_err(|e| format!("write {}: {}", codebook_path.display(), e))?;

        all_entries.push(TensorEntry {
            id: 0,
            name: codebook_name,
            role: "tts_codebook".to_string(),
            layer: None,
            segment: "tts_codebook.bin".to_string(),
            source_filename: String::new(),
            source_sha256: String::new(),
            source_offset: 0,
            offset: 0,
            byte_length: codebook_bytes.len() as u64,
            logical_dtype: dtype,
            storage_dtype: "BF16".to_string(),
            logical_shape: shape.iter().map(|&s| s as u32).collect(),
            physical_shape: shape.iter().map(|&s| s as u32).collect(),
            mutability: "frozen".to_string(),
            quantization: None,
            tensor_alignment_bytes: 16,
            layout_version: 1,
            artifact_bindings: HashMap::new(),
        });
    }

    Ok(all_entries)
}
