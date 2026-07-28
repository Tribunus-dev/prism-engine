//! Megakernel kernels — pure data types and pure constants.
//!
//! The actual Metal kernel source strings and the runtime `KernelBuffers`
//! executor live engine-side at
//! `compute-core/src/ecs/compute_image/legacy_compute_image_runtime/megakernel/kernels.rs`.

/// Hidden dimension of the megakernel (Gemma 4 12B).
pub const HIDDEN_DIM: u32 = 3840;
/// Number of layers in the megakernel (Gemma 4 12B).
pub const LAYERS: u32 = 48;
/// Number of KV heads.
pub const NUM_KV_HEADS: u32 = 8;
/// Number of MTP (multi-token-prediction) heads.
pub const NUM_MTP_HEADS: u32 = 4;
/// Maximum context length (KV cache slots).
pub const MAX_CONTEXT: u32 = 2048;
/// Maximum MTP draft candidates.
pub const MAX_DRAFT_CANDIDATES: u32 = 5;

/// Tap mode for the megakernel's intermediate-state taps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum TapMode {
    /// No tap (disabled).
    None,
    /// Tap QKV projections.
    Qkv,
    /// Tap O projection.
    OProj,
    /// Tap FFN gate.
    FfnGate,
    /// Tap FFN up.
    FfnUp,
    /// Tap FFN down.
    FfnDown,
}

/// Kernel buffer layout for the megakernel.
#[derive(Debug, Clone)]
pub struct KernelBuffers {
    /// Hidden dimension.
    pub hidden_dim: u32,
    /// Number of layers.
    pub num_layers: u32,
    /// Number of KV heads.
    pub num_kv_heads: u32,
}

/// Compile statistics for a layer library.
#[derive(Debug, Clone, Default)]
pub struct CompileLayerLibraryStats {
    /// Compilation time in milliseconds.
    pub compile_time_ms: u64,
    /// Number of kernels compiled.
    pub num_kernels: u32,
    /// Pipeline identifier.
    pub pipeline_id: String,
}

/// Engine-side stub: the real implementation lives at the legacy path.
pub fn compile_layer_library(_layers: u32) -> Result<CompileLayerLibraryStats, String> {
    Err("compile_layer_library is engine-coupled; use the legacy path or call the engine binary directly".to_string())
}
