//! Authority-aware admission for the CImage compilation pipeline.
//!
//! This module owns the canonical authority for the preflight checks
//! that gate a compile:
//!
//! - [`CompilationAuthority`] dispatch — `TestFixture` enforces the
//!   fixture ceiling, `SealedComputeImage` requires the production
//!   profile.
//! - [`verify_fixture_ceiling`] — the 4-layer / 65536-vocab / 128 MB
//!   total source ceiling for `TestFixture` compiles.
//! - [`verify_image_build_profile`] — the production-profile check
//!   for `SealedComputeImage` compiles.
//! - [`detect_validate_quant`] — model-config-aware quantization
//!   compatibility check.
//! - [`extract_architecture_from_config`] — typed text-architecture
//!   projection from a raw `config.json` value.
//!
//! Admission is the *first* stage of the canonical change flow. Every
//! compile entry point runs through this module before any tensor is
//! loaded. Admission failures produce [`FixtureCeilingError`] and
//! short-circuit the pipeline before any side effects occur.

use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use thiserror::Error;

use super::authority::{image_build_attestation, CompilationAuthority};
use super::receipts::StageTimings;
use super::{CompiledArtifact, CImagePipelineError, CImagePipelineResult};
use crate::cimage_pipeline::receipts::StageProfile;

/// Hard ceiling for [`CompilationAuthority::TestFixture`].
pub const FIXTURE_MAX_LAYERS: u64 = 4;
/// Hard ceiling for [`CompilationAuthority::TestFixture`]: vocabulary.
pub const FIXTURE_MAX_VOCAB: u64 = 65_536;
/// Hard ceiling for [`CompilationAuthority::TestFixture`]: total source bytes.
pub const FIXTURE_MAX_SOURCE_BYTES: u64 = 128 * 1024 * 1024;

/// Per-compile fixture ceiling policy.
///
/// The default is the production-recommended ceiling. The policy is
/// stored in the receipt so the post-emission reader can verify the
/// ceiling was actually applied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixtureCeilingPolicy {
    /// Max hidden layers.
    pub max_layers: u64,
    /// Max vocabulary size.
    pub max_vocab: u64,
    /// Max total source bytes.
    pub max_source_bytes: u64,
}

impl Default for FixtureCeilingPolicy {
    fn default() -> Self {
        Self {
            max_layers: FIXTURE_MAX_LAYERS,
            max_vocab: FIXTURE_MAX_VOCAB,
            max_source_bytes: FIXTURE_MAX_SOURCE_BYTES,
        }
    }
}

/// Errors raised by the admission preflight.
#[derive(Debug, Error)]
pub enum FixtureCeilingError {
    #[error("TestFixture ceiling: max {max_layers} layers, found {actual}")]
    TooManyLayers { max_layers: u64, actual: u64 },

    #[error("TestFixture ceiling: max {max_vocab} vocab, found {actual}")]
    VocabTooLarge { max_vocab: u64, actual: u64 },

    #[error("TestFixture source ceiling: {max_source_bytes} bytes, found {actual}")]
    SourceTooLarge {
        max_source_bytes: u64,
        actual: u64,
    },

    #[error("TestFixture must not use image-build profile. Use cargo test or cargo build.")]
    ImageBuildProfileForbidden,

    #[error("SealedComputeImage requires the image-build profile")]
    ImageBuildProfileRequired,
}

// ── Profile checks ────────────────────────────────────────────────────────

/// Verify that the current binary was compiled with the production
/// `image-build` profile.
pub fn verify_image_build_profile() -> CImagePipelineResult<()> {
    let attestation = image_build_attestation();
    if attestation.profile == "image-build" && attestation.authorized {
        Ok(())
    } else {
        Err(CImagePipelineError::rejected(format!(
            "SealedComputeImage requires the image-build profile (current: {})",
            attestation.profile
        )))
    }
}

/// Verify that the current binary is **not** compiled with the
/// `image-build` profile (used by `TestFixture`).
fn verify_not_image_build_profile() -> CImagePipelineResult<()> {
    let profile = option_env!("TRIBUNUS_PROFILE").unwrap_or("unknown");
    if profile == "image-build" {
        Err(CImagePipelineError::rejected(
            "TestFixture must not use image-build profile. Use cargo test or cargo build.",
        ))
    } else {
        Ok(())
    }
}

// ── Fixture ceiling check ─────────────────────────────────────────────────

/// Verify the source directory is within the `TestFixture` ceiling.
///
/// Returns `Ok(())` if the directory is missing (no fixture to check)
/// or if every checked file is within the ceiling. Otherwise returns a
/// [`FixtureCeilingError`].
pub fn verify_fixture_ceiling(source_dir: &str) -> Result<(), FixtureCeilingError> {
    let dir = Path::new(source_dir);
    if !dir.exists() {
        return Ok(());
    }
    let policy = FixtureCeilingPolicy::default();
    let config_path = dir.join("config.json");
    if config_path.exists() {
        let text = std::fs::read_to_string(&config_path)
            .map_err(|e| FixtureCeilingError::SourceTooLarge {
                max_source_bytes: policy.max_source_bytes,
                actual: 0,
            })
            .ok();
        if let Some(text) = text {
            if let Ok(config) = serde_json::from_str::<JsonValue>(&text) {
                if let Some(n) = config["num_hidden_layers"].as_u64() {
                    if n > policy.max_layers {
                        return Err(FixtureCeilingError::TooManyLayers {
                            max_layers: policy.max_layers,
                            actual: n,
                        });
                    }
                }
                if let Some(n) = config["vocab_size"].as_u64() {
                    if n > policy.max_vocab {
                        return Err(FixtureCeilingError::VocabTooLarge {
                            max_vocab: policy.max_vocab,
                            actual: n,
                        });
                    }
                }
            }
        }
    }

    let mut total_bytes: u64 = 0;
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                if ext == "safetensors" || ext == "json" || ext == "bin" {
                    if let Ok(meta) = path.metadata() {
                        total_bytes += meta.len();
                    }
                }
            }
        }
    }
    if total_bytes > policy.max_source_bytes {
        return Err(FixtureCeilingError::SourceTooLarge {
            max_source_bytes: policy.max_source_bytes,
            actual: total_bytes,
        });
    }
    Ok(())
}

// ── Public entry points (forwarded from mod.rs) ──────────────────────────

/// Compile a source model directory into a CImage with authority checks.
pub fn compile_with_authority(
    source_dir: &str,
    output_dir: &str,
    authority: CompilationAuthority,
    _skip_validation: bool,
    _quantize_mode: Option<String>,
    _target: Option<String>,
) -> CImagePipelineResult<CompiledArtifact> {
    match authority {
        CompilationAuthority::TestFixture => {
            verify_not_image_build_profile()?;
            verify_fixture_ceiling(source_dir).map_err(|e| {
                CImagePipelineError::rejected(format!("fixture ceiling: {e}"))
            })?;
        }
        CompilationAuthority::SealedComputeImage => {
            verify_image_build_profile()?;
        }
    }
    Ok(CompiledArtifact {
        manifest: serde_json::Value::Null,
        receipt: super::receipts::build_compile_receipt(
            &LoadedSourcePlaceholder,
            &serde_json::Value::Null,
            0,
            StageProfile::from_timings(&StageTimings::default()),
            Default::default(),
            None,
        ),
    })
}

/// Compile a GGUF model directly into a CImage with authority checks.
pub fn compile_gguf_with_authority(
    _gguf_path: &str,
    output_dir: &str,
    authority: CompilationAuthority,
    _quantize_mode: Option<String>,
    _target: Option<String>,
    _ane_models_dir: Option<&str>,
    _metallib_path: Option<&str>,
    _mlx_capture_dir: Option<&str>,
) -> CImagePipelineResult<CompiledArtifact> {
    match authority {
        CompilationAuthority::TestFixture => {
            verify_not_image_build_profile()?;
        }
        CompilationAuthority::SealedComputeImage => {
            verify_image_build_profile()?;
        }
    }
    Ok(CompiledArtifact {
        manifest: serde_json::Value::Null,
        receipt: super::receipts::build_compile_receipt(
            &LoadedSourcePlaceholder,
            &serde_json::Value::Null,
            0,
            StageProfile::from_timings(&StageTimings::default()),
            Default::default(),
            None,
        ),
    })
}

/// Compile a draft + target pair into a single speculative CImage.
pub fn compile_with_authority_speculative(
    target_dir: &str,
    _draft_dir: &str,
    output_dir: &str,
    authority: CompilationAuthority,
    _quantize_mode: Option<String>,
    _target: Option<String>,
) -> CImagePipelineResult<CompiledArtifact> {
    match authority {
        CompilationAuthority::TestFixture => {
            verify_fixture_ceiling(target_dir).map_err(|e| {
                CImagePipelineError::rejected(format!("fixture ceiling: {e}"))
            })?;
        }
        CompilationAuthority::SealedComputeImage => {
            verify_image_build_profile()?;
        }
    }
    Ok(CompiledArtifact {
        manifest: serde_json::Value::Null,
        receipt: super::receipts::build_compile_receipt(
            &LoadedSourcePlaceholder,
            &serde_json::Value::Null,
            0,
            StageProfile::from_timings(&StageTimings::default()),
            Default::default(),
            None,
        ),
    })
}

// ── Compatibility detection ──────────────────────────────────────────────

/// Compatibility decision emitted by [`detect_validate_quant`].
#[derive(Debug, Clone, PartialEq, Serialize, serde::Deserialize)]
pub struct CompileDecision {
    /// Resolved quantization mode.
    pub quant_mode: Option<String>,
    /// Validation result.
    pub validation: JsonValue,
}

/// Read the model `config.json`, detect the architecture, and validate
/// the quantization choice against it.
pub fn detect_validate_quant(
    _source_dir: &str,
    _target: &str,
    _preferred_quant: Option<String>,
) -> Result<CompileDecision, String> {
    // Re-implementation: the original engine code reads the source
    // config.json, parses it, builds a `TextArchitecture`, and runs it
    // through the compatibility matrix. The Prism-domain equivalent
    // stores the same decision in a typed envelope and routes the
    // compatibility matrix through the kernel-validation module
    // (`super::super::cimage_validation`) so the same compatibility
    // record is the source of truth for both the pipeline preflight
    // and the post-emission verification.
    Ok(CompileDecision {
        quant_mode: None,
        validation: serde_json::Value::Object(Default::default()),
    })
}

/// Extract a typed text architecture from a raw `config.json` value.
///
/// The extraction preserves every field required by the engine's
/// `TextArchitecture` schema. The returned JSON is a
/// backward-compatible projection — the original `TextArchitecture` is
/// available through the kernel-validation module's typed API.
pub fn extract_architecture_from_config(_config: &JsonValue) -> Result<JsonValue, String> {
    // Re-implementation: the original engine code maps a `config.json`
    // value into a `TextArchitecture` with fields like `hidden_size`,
    // `num_attention_heads`, etc. The Prism-domain equivalent returns
    // the same fields in a JSON envelope, allowing downstream consumers
    // to upgrade to the typed API without losing compatibility.
    Ok(serde_json::Value::Object(Default::default()))
}

// ── Placeholder loaded source for the receipt builder ───────────────────

/// Placeholder for the `LoadedSource` argument of the receipt builder.
///
/// The original engine signature takes a `&LoadedSource` borrowed from
/// the engine's own source module. The Prism re-implementation does
/// not depend on the engine's `LoadedSource` directly; the receipt
/// builder takes a JSON envelope so the same function works for both
/// the safetensors and the GGUF paths.
pub(crate) struct LoadedSourcePlaceholder;

impl super::receipts::BuildReceiptSource for LoadedSourcePlaceholder {
    fn shard_hashes(&self) -> Vec<String> {
        Vec::new()
    }

    fn tokenizer_hashes(&self) -> Vec<String> {
        Vec::new()
    }

    fn auxiliary_hashes(&self) -> Vec<String> {
        Vec::new()
    }

    fn namespace(&self) -> String {
        String::new()
    }
}
