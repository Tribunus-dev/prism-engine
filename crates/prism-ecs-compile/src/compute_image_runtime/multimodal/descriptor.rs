//! Multimodal descriptor types — pure data types for binary-layout
//! multimodal descriptors.
//!
//! The full binary-layout `MultimodalInputDescriptorV1` (with
//! `#[repr(C)]` and `unsafe { std::mem::zeroed() }` defaults) lives
//! engine-side at
//! `compute-core/src/ecs/compute_image/legacy_compute_image_runtime/multimodal/descriptor.rs`.
//! This file declares the canonical public taxonomy.

use serde::{Deserialize, Serialize};

/// Input modality kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InputModality {
    /// Text-only input.
    Text,
    /// Image input.
    Image,
    /// Audio input.
    Audio,
    /// Video input.
    Video,
}

/// Projection backend kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProjectionBackend {
    /// No projection (passthrough).
    None,
    /// Metal GPU projection.
    Metal,
    /// Apple Neural Engine via Core ML.
    CoreAi,
    /// CPU via Accelerate framework.
    Cpu,
}

/// Projection precision kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProjectionPrecision {
    /// F32 precision.
    F32,
    /// F16 precision.
    F16,
    /// BF16 precision.
    BF16,
    /// INT8 quantized.
    I8,
}

/// Role of a projection tensor in the multimodal pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProjectionRole {
    /// Image projection.
    Image,
    /// Audio projection.
    Audio,
    /// Video projection.
    Video,
    /// Auxiliary projection.
    Auxiliary,
}

/// Canonical identity for an image processor contract.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageProcessorContractV1 {
    /// Processor contract digest (hex).
    pub contract_digest: String,
    /// Image patch size.
    pub patch_size: u32,
    /// Number of image channels.
    pub channels: u32,
    /// Number of soft tokens produced.
    pub soft_token_count: u32,
}

/// Canonical identity for an audio processor contract.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioProcessorContractV1 {
    /// Processor contract digest (hex).
    pub contract_digest: String,
    /// Sample rate.
    pub sample_rate: u32,
    /// Number of mel bins.
    pub mel_bins: u32,
    /// Number of soft tokens produced.
    pub soft_token_count: u32,
}

/// A single projection tensor record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectionTensorRecord {
    /// Logical name hash.
    pub logical_name_hash: u64,
    /// Role of this projection.
    pub role: ProjectionRole,
    /// Projection backend.
    pub backend: ProjectionBackend,
    /// Projection precision.
    pub precision: ProjectionPrecision,
    /// Input width.
    pub input_width: u32,
    /// Output width.
    pub output_width: u32,
    /// Weight byte length.
    pub weight_length: u64,
    /// Scale byte length (for quantized).
    pub scale_length: u64,
}

/// Multimodal capabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalCapabilities {
    /// Supported modalities.
    pub supported_modalities: Vec<InputModality>,
    /// Maximum simultaneous modalities.
    pub max_concurrent_modalities: u32,
    /// Maximum image count per prompt.
    pub max_images: u32,
    /// Maximum audio length in seconds.
    pub max_audio_seconds: u32,
}

/// Multimodal artifact summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalArtifactSummary {
    /// Image processor contract digest.
    pub image_contract_digest: String,
    /// Audio processor contract digest.
    pub audio_contract_digest: String,
    /// Number of projection tensors.
    pub projection_count: u32,
}

/// Multimodal assembly receipt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalAssemblyReceipt {
    /// Session identifier.
    pub session_id: u64,
    /// Prompt content hash.
    pub prompt_digest: String,
    /// Processor contract digest.
    pub processor_contract_digest: String,
    /// Modality mask (bitfield).
    pub modality_mask: u32,
    /// Number of images.
    pub image_count: u32,
    /// Number of audio segments.
    pub audio_count: u32,
    /// Soft token counts per image.
    pub image_soft_token_counts: Vec<u32>,
    /// Assembled sequence length.
    pub assembled_sequence_len: u32,
    /// Embedding content hash.
    pub embedding_digest: String,
    /// Projection backend used.
    pub projection_backend: ProjectionBackend,
    /// Projection precision used.
    pub projection_precision: ProjectionPrecision,
    /// Elapsed time in nanoseconds.
    pub elapsed_ns: u64,
}

/// Errors raised during modality processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModalityError {
    /// The modality is not supported.
    UnsupportedModality,
    /// Too many concurrent modalities.
    TooManyModalities,
    /// Image dimensions out of range.
    ImageDimensionsOutOfRange,
    /// Audio length out of range.
    AudioLengthOutOfRange,
}

/// Placeholder binary-layout multimodal input descriptor (V1).
///
/// The full `#[repr(C)]` `MultimodalInputDescriptorV1` with magic,
/// version, and binary-layout fields lives engine-side at
/// `compute-core/src/ecs/compute_image/legacy_compute_image_runtime/multimodal/descriptor.rs`.
/// The engine-coupled definition participates in the binary-load path
/// and the engine re-exports it as the canonical type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalInputDescriptorV1 {
    /// Format version.
    pub version: u16,
    /// Modality mask (bitfield).
    pub modality_mask: u16,
    /// Decoder hidden size.
    pub decoder_hidden_size: u32,
    /// Vocabulary size.
    pub vocabulary_size: u32,
}
