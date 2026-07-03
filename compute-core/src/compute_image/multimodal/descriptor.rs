//! Binary-layout multimodal descriptor types and supporting enums/structs.

#![allow(dead_code)]

use std::fmt;

// ──────────────────────────────────────────────
// 1. MultimodalInputDescriptorV1
// ──────────────────────────────────────────────

/// Magic bytes for a valid V1 multimodal descriptor: `b"PRMMOD01"`.
pub const MULTIMODAL_DESCRIPTOR_MAGIC: [u8; 8] = *b"PRMMOD01";

/// V1 binary-layout multimodal input descriptor.
///
/// This struct is loaded from a sealed binary image and validated against
/// `MULTIMODAL_DESCRIPTOR_MAGIC` and `version == 1`. Fields cover modality
/// capabilities, image/audio projector tables, position embeddings, and
/// cryptographic digests for processor contracts and tensor layouts.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MultimodalInputDescriptorV1 {
    pub magic: [u8; 8],
    pub version: u16,
    pub modality_mask: u16,
    pub flags: u32,
    pub decoder_hidden_size: u32,
    pub vocabulary_size: u32,
    pub image_patch_size: u16,
    pub image_pooling_kernel: u16,
    pub image_channels: u16,
    pub image_reserved: u16,
    pub image_min_soft_tokens: u32,
    pub image_default_soft_tokens: u32,
    pub image_max_soft_tokens: u32,
    pub image_position_table_height: u32,
    pub image_position_table_width: u32,
    pub image_position_embedding_width: u32,
    pub text_placeholder_token_id: u32,
    pub image_placeholder_token_id: u32,
    pub audio_placeholder_token_id: u32,
    pub image_projection_table_offset: u64,
    pub image_projection_count: u32,
    pub audio_projection_table_offset: u64,
    pub audio_projection_count: u32,
    pub position_embedding_segment_index: u16,
    pub projection_weight_segment_index: u16,
    pub projection_scale_segment_index: u16,
    pub auxiliary_weight_segment_index: u16,
    pub processor_contract_digest: [u8; 32],
    pub tensor_layout_digest: [u8; 32],
}

impl Default for MultimodalInputDescriptorV1 {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

impl MultimodalInputDescriptorV1 {
    /// Validate magic and version fields.
    pub fn validate(&self) -> Result<(), String> {
        if self.magic != MULTIMODAL_DESCRIPTOR_MAGIC {
            return Err(format!("bad multimodal descriptor magic: {:?}", self.magic));
        }
        if self.version != 1 {
            return Err(format!(
                "unsupported multimodal descriptor version: {}",
                self.version
            ));
        }
        Ok(())
    }
}

// ──────────────────────────────────────────────
// 2. ProjectionTensorRecord
// ──────────────────────────────────────────────

/// A single projection tensor entry in the multimodal weight table.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ProjectionTensorRecord {
    pub logical_name_hash: u64,
    pub role: u16,
    pub dtype: u16,
    pub weight_offset: u64,
    pub weight_length: u64,
    pub scale_offset: u64,
    pub scale_length: u64,
    pub input_width: u32,
    pub output_width: u32,
    pub rank: u8,
    pub layout: u8,
    pub quantization_kind: u8,
    pub flags: u8,
    pub dims: [u32; 4],
}

impl Default for ProjectionTensorRecord {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

// ──────────────────────────────────────────────
// 3. ProjectionRole
// ──────────────────────────────────────────────

/// Well-known roles for projection tensors.
#[repr(u16)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProjectionRole {
    ImagePatchEmbedding = 1,
    ImageProjection = 2,
    ImagePositionEmbedding = 3,
    ImagePooling = 4,
    AudioFrameEmbedding = 5,
    AudioProjection = 6,
    AudioPositionEmbedding = 7,
}

// ──────────────────────────────────────────────
// 4. ImageProcessorContractV1
// ──────────────────────────────────────────────

/// V1 binary-layout contract describing how images are pre-processed.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ImageProcessorContractV1 {
    pub resize_policy: u8,
    pub channel_order: u8,
    pub normalization_kind: u8,
    pub interpolation: u8,
    pub patch_size: u16,
    pub pooling_kernel: u16,
    pub min_soft_tokens: u32,
    pub default_soft_tokens: u32,
    pub max_soft_tokens: u32,
    pub max_patch_count: u32,
    pub width_divisibility: u16,
    pub height_divisibility: u16,
    pub placeholder_policy: u8,
    pub image_sequence_layout: u8,
    pub reserved: [u8; 6],
}

impl Default for ImageProcessorContractV1 {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

// ──────────────────────────────────────────────
// 5. AudioProcessorContractV1
// ──────────────────────────────────────────────

/// V1 binary-layout contract describing how audio is pre-processed.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct AudioProcessorContractV1 {
    pub sample_rate: u32,
    pub frame_size_ms: u16,
    pub hop_size_ms: u16,
    pub num_mel_bins: u16,
    pub num_features: u16,
    pub reserved: [u8; 12],
}

impl Default for AudioProcessorContractV1 {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

// ──────────────────────────────────────────────
// 6. InputModality
// ──────────────────────────────────────────────

/// An individual input modality used in the multimodal pipeline.
///
/// Variant values are powers of two so they can be OR'd into a bitmask.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputModality {
    Text = 1,
    Image = 2,
    Audio = 4,
}

impl InputModality {
    /// Return the single-bit mask value for this modality.
    pub fn as_mask_bit(&self) -> u16 {
        *self as u16
    }
}

// ──────────────────────────────────────────────
// 7. ProjectionBackend + ProjectionPrecision + MultimodalCapabilities
// ──────────────────────────────────────────────

/// Which compute backend runs a modality projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectionBackend {
    None,
    Metal,
    Cpu,
    Ane,
}

/// Precision mode for projection weights and computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectionPrecision {
    Fp16,
    Ternary,
    Hybrid,
    Unknown,
}

/// Runtime-advertised multimodal capabilities of the loaded image.
#[derive(Debug, Clone)]
pub struct MultimodalCapabilities {
    pub text: bool,
    pub image: bool,
    pub audio: bool,
    pub image_projection_backend: ProjectionBackend,
    pub audio_projection_backend: ProjectionBackend,
    pub max_images_per_prompt: u32,
    pub max_soft_tokens_per_image: u32,
    pub supports_mixed_embedding_prefill: bool,
}

impl Default for MultimodalCapabilities {
    fn default() -> Self {
        Self {
            text: true,
            image: false,
            audio: false,
            image_projection_backend: ProjectionBackend::None,
            audio_projection_backend: ProjectionBackend::None,
            max_images_per_prompt: 0,
            max_soft_tokens_per_image: 0,
            supports_mixed_embedding_prefill: false,
        }
    }
}

// ──────────────────────────────────────────────
// 8. MultimodalArtifactSummary
// ──────────────────────────────────────────────

/// A lightweight report summarising multimodal capabilities after loading.
pub struct MultimodalArtifactSummary {
    pub modalities: u32,
    pub image_soft_token_default: u32,
    pub image_soft_token_max: u32,
    pub projection_precision: ProjectionPrecision,
    pub processor_contract_digest: [u8; 32],
    pub tensor_layout_digest: [u8; 32],
}

// ──────────────────────────────────────────────
// 9. MultimodalAssemblyReceipt
// ──────────────────────────────────────────────

/// Receipt produced after assembling a multimodal prompt into GPU-ready
/// soft-token embeddings.
pub struct MultimodalAssemblyReceipt {
    pub session_id: u64,
    pub prompt_digest: [u8; 32],
    pub processor_contract_digest: [u8; 32],
    pub modality_mask: u32,
    pub image_count: u32,
    pub image_soft_token_counts: Vec<u32>,
    pub assembled_sequence_len: u32,
    pub embedding_digest: [u8; 32],
    pub projection_backend: ProjectionBackend,
    pub projection_precision: ProjectionPrecision,
    pub elapsed_ns: u64,
}

// ──────────────────────────────────────────────
// 10. ModalityError
// ──────────────────────────────────────────────

/// Errors that can arise during multimodal descriptor validation, projection,
/// or assembly.
#[derive(Debug, Clone)]
pub enum ModalityError {
    UnsupportedModality(InputModality),
    MissingDescriptor,
    DescriptorValidationFailed(String),
    SegmentNotFound(String),
    DimensionMismatch { expected: u32, actual: u32 },
    ProjectionFailed(String),
    AssemblyFailed(String),
    ContractDigestMismatch,
    FeatureGated(InputModality),
}

impl fmt::Display for ModalityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModalityError::UnsupportedModality(m) => {
                write!(f, "unsupported modality: {:?}", m)
            }
            ModalityError::MissingDescriptor => write!(f, "missing multimodal descriptor"),
            ModalityError::DescriptorValidationFailed(msg) => {
                write!(f, "descriptor validation failed: {}", msg)
            }
            ModalityError::SegmentNotFound(name) => {
                write!(f, "segment not found: {}", name)
            }
            ModalityError::DimensionMismatch { expected, actual } => {
                write!(f, "dimension mismatch: expected {}, got {}", expected, actual)
            }
            ModalityError::ProjectionFailed(msg) => {
                write!(f, "projection failed: {}", msg)
            }
            ModalityError::AssemblyFailed(msg) => {
                write!(f, "assembly failed: {}", msg)
            }
            ModalityError::ContractDigestMismatch => {
                write!(f, "processor contract digest mismatch")
            }
            ModalityError::FeatureGated(m) => {
                write!(f, "modality {:?} is feature-gated", m)
            }
        }
    }
}

impl std::error::Error for ModalityError {}
