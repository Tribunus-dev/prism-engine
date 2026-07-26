//! CImage compilation pipeline — authority-aware compilation, sequential and
//! differential compilation, receipt emission, diagnostics, and publishing.
//!
//! This module owns the canonical authority for the end-to-end CImage
//! compilation pipeline: the typed command entry points
//! ([`compile_with_authority`], [`compile_gguf_with_authority`], and the
//! speculative variant), the differential-compile path
//! ([`compile_differential`]), the canonical projection entry point
//! ([`compile_to_canonical`]), the receipt types and emission
//! ([`build_compile_receipt`], [`CompileReceipt`], [`StageProfile`]), the
//! post-emission diagnostics report ([`run_diagnostics`],
//! [`DiagnosticReport`]), and the publishing step ([`publish_image`]).
//!
//! # Module layout
//!
//! The pipeline surface is split by authority along the canonical change
//! flow:
//!
//! - [`admission`] owns the authority-aware preflight: fixture ceilings,
//!   image-build profile verification, compatibility validation, and the
//!   `CompilationAuthority` discriminant.
//! - [`receipts`] owns the [`CompileReceipt`] and [`StageProfile`] types and
//!   the emission path that produces a deterministic `receipt.json` next
//!   to the manifest.
//! - [`diagnostics`] owns the post-emission [`DiagnosticReport`] and the
//!   layer / global diagnostic records that surface compile outcomes to
//!   operators.
//! - [`differential`] owns the differential compile path: load an existing
//!   CImage, diff the new source tensors against the existing artifact,
//!   and emit only the changed tensors plus a structural receipt.
//! - [`publish`] owns the staging → destination copy that promotes a
//!   compile output into a canonical location.
//! - [`canonical`] owns the [`compile_to_canonical`] entry point that
//!   transforms a compile result into a `CanonicalModelIr` for the
//!   constitutional consumers.
//! - [`authority`] owns the [`CompilationAuthority`] discriminant and the
//!   `SourceIdentity` contract types.
//!
//! This file is the directory index and re-exports the public surface so
//! that callers using `cimage_pipeline::compile_with_authority` and
//! `cimage_pipeline::CompileReceipt` continue to compile unchanged after
//! the project-absorption decomposition.

use std::path::Path;

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub mod admission;
pub mod authority;
pub mod canonical;
pub mod diagnostics;
pub mod differential;
pub mod publish;
pub mod receipts;

pub use admission::{
    detect_validate_quant, extract_architecture_from_config, verify_fixture_ceiling,
    verify_image_build_profile, FixtureCeilingError, FixtureCeilingPolicy,
};
pub use authority::{CompilationAuthority, ImageBuildAttestation};
pub use canonical::{
    build_canonical_model_ir, build_canonical_model_ir_from_manifest, build_canonical_outcome,
    build_canonical_representation_plan, build_model_ir_from_config, compile_gguf_to_canonical,
    compile_to_canonical, CanonicalModelIr, CanonicalModelIrBuilder, CanonicalOutcome,
};
pub use diagnostics::{
    run_diagnostics, DiagnosticIssue, DiagnosticReport, GlobalDiagnostic, LayerDiagnostic,
};
pub use differential::compile_differential;
pub use publish::{publish_image, PublishError, PublishPolicy};
pub use receipts::{build_compile_receipt, CompileReceipt, StageProfile, StageTimings};

/// Per-crate error type for the cimage pipeline. Categorized as `Rejected`
/// (preflight, admission, fixture ceiling), `Failed` (effect, emit,
/// publish), or `Stale` (fencing mismatch, generation drift).
#[derive(Debug, Error)]
pub enum CImagePipelineError {
    #[error("rejected: {0}")]
    Rejected(String),

    #[error("failed: {0}")]
    Failed(String),

    #[error("stale: {0}")]
    Stale(String),
}

impl CImagePipelineError {
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

impl From<std::io::Error> for CImagePipelineError {
    fn from(error: std::io::Error) -> Self {
        Self::Failed(format!("io: {error}"))
    }
}

impl From<serde_json::Error> for CImagePipelineError {
    fn from(error: serde_json::Error) -> Self {
        Self::Failed(format!("json: {error}"))
    }
}

impl From<publish::PublishError> for CImagePipelineError {
    fn from(error: publish::PublishError) -> Self {
        Self::Failed(format!("publish: {error}"))
    }
}

/// Result alias for the cimage pipeline.
pub type CImagePipelineResult<T> = Result<T, CImagePipelineError>;

// ── Top-level entry points (re-exported from the submodules) ───────────────

/// Compile a source model directory into a CImage with authority checks.
pub fn compile_with_authority(
    source_dir: &str,
    output_dir: &str,
    authority: CompilationAuthority,
    skip_validation: bool,
    quantize_mode: Option<String>,
    target: Option<String>,
) -> CImagePipelineResult<CompiledArtifact> {
    admission::compile_with_authority(
        source_dir,
        output_dir,
        authority,
        skip_validation,
        quantize_mode,
        target,
    )
}

/// Compile a GGUF model directly into a CImage with authority checks.
pub fn compile_gguf_with_authority(
    gguf_path: &str,
    output_dir: &str,
    authority: CompilationAuthority,
    quantize_mode: Option<String>,
    target: Option<String>,
    ane_models_dir: Option<&str>,
    metallib_path: Option<&str>,
    mlx_capture_dir: Option<&str>,
) -> CImagePipelineResult<CompiledArtifact> {
    admission::compile_gguf_with_authority(
        gguf_path,
        output_dir,
        authority,
        quantize_mode,
        target,
        ane_models_dir,
        metallib_path,
        mlx_capture_dir,
    )
}

/// Compile a draft + target pair into a single speculative CImage.
pub fn compile_with_authority_speculative(
    target_dir: &str,
    draft_dir: &str,
    output_dir: &str,
    authority: CompilationAuthority,
    quantize_mode: Option<String>,
    target: Option<String>,
) -> CImagePipelineResult<CompiledArtifact> {
    admission::compile_with_authority_speculative(
        target_dir,
        draft_dir,
        output_dir,
        authority,
        quantize_mode,
        target,
    )
}

/// Read a finalized CImage directory and surface a typed reader handle.
pub fn read(image_dir: &str) -> CImagePipelineResult<CompiledImageReader> {
    differential::read(image_dir)
}

/// Verify a finalized CImage directory against the manifest and produce
/// a [`diagnostics::DiagnosticReport`].
pub fn verify(image_dir: &str) -> CImagePipelineResult<diagnostics::DiagnosticReport> {
    diagnostics::verify(image_dir)
}

// ── Artifact envelope (referenced by every entry point) ────────────────────

/// Compiled CImage artifact — manifest plus compile receipt.
///
/// This is the typed return type of the compile entry points. The
/// manifest is the canonical on-disk schema; the receipt is the
/// post-emission evidence record that participates in the canonical
/// change flow.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledArtifact {
    /// The manifest produced by the compile step.
    pub manifest: serde_json::Value,
    /// The receipt produced by the compile step.
    pub receipt: CompileReceipt,
}

/// Compiled CImage reader handle (returned by [`read`]).
#[derive(Debug, Clone)]
pub struct CompiledImageReader {
    /// Image directory path.
    pub image_dir: String,
    /// Parsed manifest, if present.
    pub manifest: Option<serde_json::Value>,
    /// Parsed receipt, if present.
    pub receipt: Option<CompileReceipt>,
}

#[cfg(test)]
mod tests;
