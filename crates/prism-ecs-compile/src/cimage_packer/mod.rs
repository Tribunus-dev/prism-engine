//! CImage packer — V4 unified `.cimage` packing pipeline.
//!
//! This module owns the canonical authority for packing a finalized
//! CImage directory into a single V4 unified `.cimage` binary. The
//! pattern is the same as the engine's `cimage_packer::pipeline`:
//!
//! 1. Predict the total file size from the per-segment byte lengths.
//! 2. `ftruncate` + `mmap` at the total file size.
//! 3. Slice the mmap into per-segment regions using the
//!    [`AlignedMmapBuilder`].
//! 4. Write each segment to its slice, in execution order.
//! 5. Write the [`CimageHeader`] at offset 0.
//!
//! The packer is *pure effect*: it does not own canonical state, it
//! does not produce durable components, and it does not validate the
//! input (validation is the pipeline's job — see
//! `super::cimage_pipeline::diagnostics`). The packer's
//! responsibility is to lay out the binary; the pipeline's
//! responsibility is to decide what the binary means.
//!
//! # Module layout
//!
//! The packer surface is split by authority along the public / private
//! boundary:
//!
//! - [`pack_unified`] owns the legacy 5-segment packer (`pack_unified_cimage`).
//! - [`pack_from_dir`] owns the directory-aware packer
//!   (`pack_cimage_from_dir`).
//! - [`segment_writer`] owns the per-segment write helpers.
//! - [`multimodal`] owns the multimodal segment synthesis helpers.
//! - [`helpers`] owns the small private helpers shared by both packers.
//!
//! This file is the directory index and re-exports the public packer
//! entry points.

use std::io;

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub mod aligned_mmap;
pub mod helpers;
pub mod layout;
pub mod multimodal;
pub mod pack_from_dir;
pub mod pack_unified;
pub mod segment_writer;

pub use pack_from_dir::pack_cimage_from_dir;
pub use pack_unified::pack_unified_cimage;
pub use aligned_mmap::{AlignedMmapBuilder, AlignedMmapError};
pub use layout::{
    predict_tar_size, CImageLayoutPlan, CImageTopologyTable, QuantizationLayoutHint,
    SegmentDescriptor, StrideDescriptor,
};

/// Page size used by the CImage packer (16 KB Apple Silicon page).
pub const APPLE_PAGE_SIZE: usize = 16_384;

/// Per-crate error type for the cimage packer. Categorized as
/// `Rejected` (input validation, missing manifest), `Failed`
/// (I/O failure, header layout failure), or `Stale` (segment count
/// mismatch with header reservation).
#[derive(Debug, Error)]
pub enum CImagePackerError {
    #[error("rejected: {0}")]
    Rejected(String),

    #[error("failed: {0}")]
    Failed(String),

    #[error("stale: {0}")]
    Stale(String),
}

impl CImagePackerError {
    /// Construct a `Rejected` variant.
    pub fn rejected(message: impl Into<String>) -> Self {
        Self::Rejected(message.into())
    }

    /// Construct a `Failed` variant.
    pub fn failed(message: impl Into<String>) -> Self {
        Self::Failed(message.into())
    }

    /// Construct a `Stale` variant.
    pub fn stale(message: impl Into<String>) -> Self {
        Self::Stale(message.into())
    }
}

impl From<io::Error> for CImagePackerError {
    fn from(error: io::Error) -> Self {
        Self::Failed(format!("io: {error}"))
    }
}

/// Result alias for the cimage packer.
pub type CImagePackerResult<T> = Result<T, CImagePackerError>;

/// Segment kind discriminant.
///
/// Mirrors the engine's `SegmentKind` enum but is scoped to the
/// packer's per-segment write authority. The packer only needs to
/// distinguish the segments it writes; the broader `SegmentKind` used
/// by the manifest lives in the cimage data module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SegmentKind {
    /// Metal library (`model.metallib`).
    MetalLib,
    /// CUDA library (`model.cubin` / `model.fatbin`).
    CudaLib,
    /// ROCm library (`model.co` / `model.hsaco`).
    RocmLib,
    /// Level Zero library (`model_l0.spv`).
    LevelZeroLib,
    /// Vulkan library (`model_vulkan.spv`).
    VulkanLib,
    /// WebGPU library (`model_wgsl.spv`).
    WebGpuLib,
    /// Intel NPU blob (`npu_intel.bin`).
    IntelNpuBlob,
    /// AMD NPU blob (`npu_amdxdna.bin`).
    AmdNpuBlob,
    /// Qualcomm NPU blob (`npu_qualcomm.bin`).
    QualcommNpuBlob,
    /// Google TPU blob (`npu_google.bin`).
    GoogleTpuBlob,
    /// ANE archive (`npu_ane.tar` / `*.ane.tar`).
    AneArchive,
    /// Huawei Ascend blob (`npu_huawei.bin`).
    HuaweiAscendBlob,
    /// Hailo blob (`npu_hailo.hef`).
    HailoBlob,
    /// TTS talker weight segment.
    TtsTalkerWeight,
    /// TTS talker scale segment.
    TtsTalkerScale,
    /// TTS talker bias segment.
    TtsTalkerBias,
    /// TTS code predictor weight segment.
    TtsCodePredictorWeight,
    /// TTS code predictor scale segment.
    TtsCodePredictorScale,
    /// TTS code predictor bias segment.
    TtsCodePredictorBias,
    /// TTS codec weight segment.
    TtsCodecWeight,
    /// TTS codebook segment.
    TtsCodebook,
    /// Ternary weight segment.
    TernaryWeights,
    /// Persistent weight segment.
    Persistent,
    /// Per-layer weight segment.
    Layer(u32),
    /// Execution graph segment.
    ExecutionGraph,
    /// Model artifacts segment.
    ModelArtifacts,
    /// Multimodal descriptor segment.
    MultimodalDescriptor,
    /// Vision segment.
    Vision,
    /// Audio segment.
    Audio,
    /// Speculative draft segment.
    Draft,
}

/// Per-segment entry in the CImage header segment table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SegmentEntry {
    /// Segment kind discriminant.
    pub kind: u32,
    /// Byte offset in the .cimage file.
    pub offset: u64,
    /// Byte length in the .cimage file.
    pub length: u64,
}

impl SegmentEntry {
    /// Construct a new [`SegmentEntry`] from a [`SegmentKind`].
    pub fn new(kind: SegmentKind, offset: u64, length: u64) -> Self {
        Self {
            kind: kind.discriminant(),
            offset,
            length,
        }
    }
}

impl SegmentKind {
    /// Stable 32-bit discriminant for the segment kind.
    pub fn discriminant(&self) -> u32 {
        match self {
            Self::MetalLib => 1,
            Self::CudaLib => 2,
            Self::RocmLib => 3,
            Self::LevelZeroLib => 4,
            Self::VulkanLib => 5,
            Self::WebGpuLib => 6,
            Self::IntelNpuBlob => 7,
            Self::AmdNpuBlob => 8,
            Self::QualcommNpuBlob => 9,
            Self::GoogleTpuBlob => 10,
            Self::AneArchive => 11,
            Self::HuaweiAscendBlob => 12,
            Self::HailoBlob => 13,
            Self::TtsTalkerWeight => 14,
            Self::TtsTalkerScale => 15,
            Self::TtsTalkerBias => 16,
            Self::TtsCodePredictorWeight => 17,
            Self::TtsCodePredictorScale => 18,
            Self::TtsCodePredictorBias => 19,
            Self::TtsCodecWeight => 20,
            Self::TtsCodebook => 21,
            Self::TernaryWeights => 22,
            Self::Persistent => 23,
            Self::Layer(_) => 24,
            Self::ExecutionGraph => 25,
            Self::ModelArtifacts => 26,
            Self::MultimodalDescriptor => 27,
            Self::Vision => 28,
            Self::Audio => 29,
            Self::Draft => 30,
        }
    }
}

/// CImage header (the 256-byte on-disk header).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CimageHeader {
    /// Magic identifier (8 bytes).
    pub magic: [u8; 8],
    /// Version field.
    pub version: u32,
    /// Number of segments in the segment table.
    pub segment_count: u32,
    /// 32-byte payload hash.
    pub payload_hash: [u8; 32],
    /// Number of layers.
    pub num_layers: u32,
    /// Number of heads.
    pub num_heads: u32,
    /// Head dimension.
    pub head_dim: u32,
    /// Hidden dimension.
    pub hidden_dim: u32,
    /// Intermediate dimension.
    pub intermediate_dim: u32,
    /// Vocabulary size.
    pub vocab_size: u32,
    /// Quantization schema.
    pub quantization_schema: u32,
    /// Number of draft layers.
    pub draft_num_layers: u32,
    /// Per-segment entries.
    pub segments: Vec<SegmentEntry>,
    /// Trailing padding to align to 256 bytes.
    pub _pad: [u8; 8],
}

impl Default for CimageHeader {
    fn default() -> Self {
        Self {
            magic: *b"PRISM\0\0\0",
            version: 4,
            segment_count: 0,
            payload_hash: [0u8; 32],
            num_layers: 0,
            num_heads: 0,
            head_dim: 0,
            hidden_dim: 0,
            intermediate_dim: 0,
            vocab_size: 0,
            quantization_schema: 0,
            draft_num_layers: 0,
            segments: Vec::new(),
            _pad: [0u8; 8],
        }
    }
}

#[cfg(test)]
mod tests;
