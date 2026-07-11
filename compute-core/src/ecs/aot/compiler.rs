//! AOT (ahead-of-time) compiler types — transitional.
//!
//! These types are being superseded by the canonical types in
//! `ecs::canonical::kernel_abi` and `ecs::metal_backend`.
//!
//! - `CompiledKernelVariant` → `canonical::kernel_abi::CompiledKernelArtifact`
//! - `CompileError` → `metal_backend::BackendCompileError`
//! - Compilation logic → `metal_backend::MetalBackendCompiler`
//!
//! This module remains for backward compatibility during migration.
//! New code should use `ecs::metal_backend::MetalBackendCompiler`.

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
