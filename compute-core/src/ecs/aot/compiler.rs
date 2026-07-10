//! AOT Metal compiler wrapper — compiles kernel source variants at CImage build time.
//!
//! Shells out to `xcrun metal` on the build machine to produce `.metallib`
//! payloads for each target profile. This runs only during CImage creation,
//! never on the end-user device.
//!
//! The actual compilation logic lives in `ecs::system::backend_compile::MetalCompiler`.
//! This module retains only the type definitions consumed by the AOT pipeline.

use serde::{Deserialize, Serialize};

use super::profile_id::AppleSiliconProfileId;

/// A successfully compiled kernel variant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledKernelVariant {
    /// Unique variant identifier (e.g. "gemv_nf4_m4max_t640_g128").
    pub variant_id: String,
    /// Target profile this was compiled for.
    pub target_profile: AppleSiliconProfileId,
    /// Entry point function name in the metallib.
    pub entry_point: String,
    /// Raw compiled metallib bytes.
    pub metallib_bytes: Vec<u8>,
    /// SHA-256 digest of the metallib bytes.
    pub digest: String,
    /// Compile timestamp.
    pub compiled_at: String,
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum CompileError {
    #[error("xcrun metal not found — Xcode CLI tools may not be installed")]
    MetalNotFound,
    #[error("compilation failed: {details}")]
    CompileFailed { details: String },
    #[error("metallib creation failed: {details}")]
    MetallibFailed { details: String },
    #[error("I/O error: {details}")]
    Io { details: String },
}
