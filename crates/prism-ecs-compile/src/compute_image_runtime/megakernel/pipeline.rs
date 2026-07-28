//! Megakernel pipeline — pure data types and pure algorithms.
//!
//! The actual `Megakernel` struct that owns the compiled Metal
//! compute pipeline state lives engine-side at
//! `compute-core/src/ecs/compute_image/legacy_compute_image_runtime/megakernel/pipeline.rs`.

use serde::{Deserialize, Serialize};

/// Megakernel configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MegakernelPipelineConfig {
    /// Hidden dimension.
    pub hidden_dim: u32,
    /// Number of layers.
    pub num_layers: u32,
    /// Number of KV heads.
    pub num_kv_heads: u32,
    /// Maximum context length.
    pub max_context: u32,
    /// Tap mode.
    pub tap_mode: super::kernels::TapMode,
}

/// Megakernel stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MegakernelStage {
    /// Decode one token.
    Decode,
    /// Prefill prompt.
    Prefill,
    /// Verify MTP draft.
    MtpVerify,
    /// Sample output token.
    Sample,
}

/// Megakernel runtime (engine-coupled stub).
#[derive(Debug, Clone, Default)]
pub struct Megakernel;

impl Megakernel {
    /// Create a new megakernel.
    pub fn new() -> Self {
        Self
    }

    /// Configure the megakernel.
    pub fn configure(&mut self, _config: MegakernelPipelineConfig) -> Result<(), String> {
        // Stub: the real implementation depends on Metal.
        Ok(())
    }
}
