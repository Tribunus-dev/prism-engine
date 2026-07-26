//! CImage manifest — the durable on-disk schema for a compiled CImage
//! directory.
//!
//! This module owns the canonical authority for the manifest schema:
//! the top-level [`Manifest`] struct, the per-tensor table
//! ([`TensorEntry`], [`QuantizationDesc`], [`AliasEntry`]), the
//! shared-lane ABI ([`Nf4Tile640Layout`], [`SharedWeightLayout`]),
//! the lease state machine ([`LeaseState`], [`StorageBackend`],
//! [`SegmentLease`], [`TensorLease`]), the per-kernel Metal
//! dispatch contract ([`MetalDispatchRecipe`], [`MetalKernelArtifact`]),
//! and the post-emission evidence ([`CompileReceipt`], [`StageProfile`]).
//!
//! The manifest is **the single durable schema** for a compiled CImage
//! directory: every CImage directory writes `manifest.json` (the
//! schema) and `receipt.json` (the post-emission evidence). The
//! manifest participates in the canonical change flow via:
//!
//! 1. **Admission** (the `Manifest`'s `required_storage_abi` and
//!    `required_capabilities` are the typed preflight checks before a
//!    CImage is loaded by the runtime).
//! 2. **Receipt emission** (the `CompileReceipt` is the
//!    `ImageBuildAttestation`-equivalent record that proves the
//!    CImage was produced by an authorized profile).
//! 3. **Replay** (the `Manifest` is replayable — the same source
//!    tensors produce the same `image_hash` byte-for-byte).
//!
//! # Module layout
//!
//! The manifest surface is split by authority along the manifest
//! schema axes:
//!
//! - [`header`] owns the top-level [`Manifest`] struct and the
//!   [`Segment`] / [`SegmentKind`] / [`ResidencyPlan`] per-segment
//!   types.
//! - [`types`] owns the per-tensor table ([`TensorEntry`],
//!   [`QuantizationDesc`], [`AliasEntry`]), the shared-lane ABI
//!   ([`Nf4Tile640Layout`], [`SharedWeightLayout`]), the per-backend
//!   artifact ([`BackendWeightArtifact`], [`ArtifactKind`]), and the
//!   source identity ([`SourceIdentity`], [`ShardHash`]).
//! - [`kernel`] owns the per-kernel Metal dispatch contract
//!   ([`MetalDispatchRecipe`], [`MetalKernelArtifact`]).
//! - [`lease`] owns the runtime lease state machine
//!   ([`LeaseState`], [`StorageBackend`], [`SegmentLease`],
//!   [`TensorLease`]).
//! - [`receipt`] owns the post-emission evidence ([`CompileReceipt`],
//!   [`StageProfile`], [`TensorDiff`], [`ManifestVerification`]).
//! - [`builder`] owns the [`ManifestBuilder`] (the constitutional
//!   builder that produces a `Manifest` + segment payloads from a
//!   source checkpoint).
//!
//! This file is the directory index and re-exports the public surface
//! so callers using `cimage_manifest::Manifest` and
//! `cimage_manifest::CompileReceipt` continue to compile unchanged.

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub mod builder;
pub mod header;
pub mod kernel;
pub mod lease;
pub mod receipt;
pub mod types;

pub use builder::{ManifestBuilder, ManifestBuilderError, SegmentBuilder};
pub use header::{
    is_valid_storage_abi, validate_manifest_for_abi, validate_segment_alignment, CompileReadiness,
    Manifest, ResidencyPlan, Segment, SegmentKind, StorageAbiSpec, STORAGE_ABI_COPIED_V0,
    STORAGE_ABI_MAPPED_NO_COPY_V1,
};
pub use kernel::{MetalDispatchRecipe, MetalKernelArtifact};
pub use lease::{
    CopyClassification, LeaseState, SegmentLease, StorageBackend, TensorLease,
};
pub use receipt::{
    CompiledImage, CompileReceipt, IgnoredTensorClassification, ManifestVerification,
    NativeCapabilityReport, RepresentationAdmissionEstimate, SegmentReceipt, StageProfile,
    TensorDiff, TensorProvenance,
};
pub use types::{
    AliasEntry, ArtifactKind, BackendWeightArtifact, Nf4Tile640Layout, QuantizationDesc,
    QuantizationProfileEntry, QuantizationQualityEntry, QuantizationQualityStatus, ShardHash,
    SharedWeightLayout, SourceIdentity, TensorEntry,
};

// ── Admission-estimate bridge ────────────────────────────────────────────

/// Produce a [`RepresentationAdmissionEstimate`] from a [`Manifest`].
///
/// The estimate is a **projection** of the manifest: it walks the
/// residency plan and the per-tensor table to compute the resident
/// and materialized bytes the runtime will need. It is the typed
/// contract the admission preflight consumes.
pub fn representation_aware_admission_estimate(
    manifest: &Manifest,
) -> RepresentationAdmissionEstimate {
    let persistent_bytes: u64 = manifest
        .residency_plan
        .persistent_segments
        .iter()
        .filter_map(|sid| manifest.segments.iter().find(|s| &s.id == sid))
        .map(|s| s.byte_size)
        .sum();

    let layer_byte_sizes: Vec<u64> = manifest
        .residency_plan
        .layer_segments
        .iter()
        .filter_map(|sid| manifest.segments.iter().find(|s| &s.id == sid).map(|s| s.byte_size))
        .collect();

    let window = manifest.residency_plan.layer_window_size.max(1) as usize;
    let mut sorted_layers = layer_byte_sizes.clone();
    sorted_layers.sort_unstable_by(|a, b| b.cmp(a));
    let max_layer_window_bytes: u64 = sorted_layers.iter().take(window).sum();

    let total_mapped: u64 = manifest.segments.iter().map(|s| s.byte_size).sum();

    // Architecture is opaque here (a serde_json::Value), so we read
    // max_position_embeddings / head_dim / hidden_size / vocab_size
    // out of the JSON. The runtime path uses the typed
    // `TextArchitecture`; the manifest path uses the JSON projection.
    let (rope_bytes, embedding_dequant_bytes) = extract_arch_footprint(&manifest.architecture);

    let mlx_workspace_bytes = 512u64 * 1024 * 1024;
    let allocator_cache_bytes = 512u64 * 1024 * 1024;
    let system_reserve_bytes = 2u64 * 1024 * 1024 * 1024;
    let kv_budget_bytes = rope_bytes.saturating_mul(4);

    let is_mapped = manifest.required_storage_abi == STORAGE_ABI_MAPPED_NO_COPY_V1;
    let virtual_mapped_bytes = if is_mapped { total_mapped } else { 0 };

    let materialized_bytes: u64 = if is_mapped {
        manifest
            .tensor_table
            .iter()
            .filter(|t| t.quantization.is_some())
            .map(|t| t.byte_length)
            .sum()
    } else {
        0
    };

    let expected_resident_bytes = if is_mapped {
        persistent_bytes
            .saturating_add(max_layer_window_bytes)
            .saturating_add(rope_bytes)
            .saturating_add(mlx_workspace_bytes)
    } else {
        manifest
            .tensor_table
            .iter()
            .map(|t| t.byte_length)
            .sum::<u64>()
            .saturating_add(rope_bytes)
            .saturating_add(mlx_workspace_bytes)
    };

    let seq_len = u64::from(rope_seq_len(&manifest.architecture).min(8192));
    let hidden_size = u64::from(arch_u32(&manifest.architecture, "hidden_size"));
    let vocab_size = u64::from(arch_u32(&manifest.architecture, "vocab_size"));
    let attention_workspace = seq_len.saturating_mul(hidden_size).saturating_mul(4);
    let output_proj_workspace = hidden_size.saturating_mul(vocab_size).saturating_mul(4);
    let largest_transient_bytes = attention_workspace.max(output_proj_workspace);

    RepresentationAdmissionEstimate {
        virtual_mapped_bytes,
        expected_resident_bytes,
        persistent_materialized_bytes: persistent_bytes,
        max_layer_window_bytes,
        rope_bytes,
        kv_budget_bytes,
        mlx_workspace_bytes,
        allocator_cache_bytes,
        system_reserve_bytes,
        largest_transient_bytes,
        materialized_bytes,
    }
}

fn arch_u32(arch: &serde_json::Value, key: &str) -> u32 {
    arch.get(key).and_then(serde_json::Value::as_u64).unwrap_or(0) as u32
}

fn rope_seq_len(arch: &serde_json::Value) -> u32 {
    arch_u32(arch, "max_position_embeddings")
}

fn extract_arch_footprint(arch: &serde_json::Value) -> (u64, u64) {
    let max_pos = u64::from(arch_u32(arch, "max_position_embeddings"));
    let head_dim = u64::from(arch_u32(arch, "head_dim"));
    let global_head_dim = u64::from(arch_u32(arch, "global_head_dim").max(arch_u32(arch, "head_dim")));
    let rope_bytes = max_pos
        .saturating_mul(head_dim)
        .saturating_mul(4)
        .saturating_add(max_pos.saturating_mul(global_head_dim).saturating_mul(4));
    let hidden = u64::from(arch_u32(arch, "hidden_size"));
    let vocab = u64::from(arch_u32(arch, "vocab_size"));
    let embedding_dequant_bytes = vocab.saturating_mul(hidden).saturating_mul(4);
    (rope_bytes, embedding_dequant_bytes)
}

// ── Per-crate error type ─────────────────────────────────────────────────

/// Per-crate error type for the cimage_manifest. Categorized as
/// `Rejected` (preflight, missing source), `Failed` (I/O, hashing),
/// or `Stale` (fencing mismatch, generation drift).
#[derive(Debug, Error)]
pub enum CImageManifestError {
    #[error("rejected: {0}")]
    Rejected(String),

    #[error("failed: {0}")]
    Failed(String),

    #[error("stale: {0}")]
    Stale(String),
}

impl CImageManifestError {
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

impl From<std::io::Error> for CImageManifestError {
    fn from(error: std::io::Error) -> Self {
        Self::Failed(format!("io: {error}"))
    }
}

impl From<serde_json::Error> for CImageManifestError {
    fn from(error: serde_json::Error) -> Self {
        Self::Failed(format!("json: {error}"))
    }
}

/// Result alias for the cimage_manifest crate surface.
pub type CImageManifestResult<T> = Result<T, CImageManifestError>;

// ── Convenience JSON envelope (durable) ──────────────────────────────────

/// JSON envelope persisted alongside the manifest for replay.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestEnvelope {
    /// Image directory path (relative to the workspace root).
    pub image_dir: String,
    /// Persisted manifest.
    pub manifest: Manifest,
    /// Persisted compile receipt.
    pub receipt: CompileReceipt,
}

impl ManifestEnvelope {
    /// Wrap a manifest and receipt into a single envelope.
    pub fn new(image_dir: impl Into<String>, manifest: Manifest, receipt: CompileReceipt) -> Self {
        Self {
            image_dir: image_dir.into(),
            manifest,
            receipt,
        }
    }

    /// Serialize the envelope as the canonical JSON form written to
    /// `manifest.json` and `receipt.json`.
    pub fn to_canonical_json(&self) -> Result<String, CImageManifestError> {
        Ok(serde_json::to_string_pretty(self)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_error_categories_are_distinct() {
        let r = CImageManifestError::rejected("rejected");
        let f = CImageManifestError::failed("failed");
        let s = CImageManifestError::stale("stale");
        assert!(format!("{r}").contains("rejected"));
        assert!(format!("{f}").contains("failed"));
        assert!(format!("{s}").contains("stale"));
    }

    #[test]
    fn manifest_envelope_serde_round_trip_preserves_both_sides() {
        let manifest = header::Manifest {
            image_version: "0.1.0".into(),
            compiler_version: "0.1.0".into(),
            runtime_abi: "prism/0.1.0".into(),
            hardware_target: None,
            readiness: None,
            compile_date: String::new(),
            compile_host: String::new(),
            source: types::SourceIdentity {
                config_hash: "abcd".into(),
                shard_hashes: Vec::new(),
                tokenizer_hashes: Vec::new(),
                auxiliary_hashes: Vec::new(),
                model_type: "qwen3".into(),
                quantization_bits: 4,
                quantization_group_size: 128,
                quantization_mode: "nf4".into(),
            },
            architecture: serde_json::json!({"hidden_size": 1024, "vocab_size": 100}),
            vision_config: None,
            audio_config: None,
            segments: Vec::new(),
            tensor_table: Vec::new(),
            alias_table: Vec::new(),
            residency_plan: ResidencyPlan {
                persistent_segments: Vec::new(),
                layer_segments: Vec::new(),
                layer_window_size: 2,
                total_bytes: 0,
            },
            image_hash: String::new(),
            required_storage_abi: STORAGE_ABI_COPIED_V0.to_string(),
            required_capabilities: Vec::new(),
            prepacked_layout: "none".into(),
            metallib_hash: None,
            metallib_size: None,
            metal_kernel_artifacts: Vec::new(),
            execution_plan: serde_json::json!({}),
            compatibility_receipt: None,
            quantization_profiles: Vec::new(),
            quantization_quality: Vec::new(),
            quantization_quality_status: types::QuantizationQualityStatus::Unknown,
        };
        let receipt = CompileReceipt::default();
        let env = ManifestEnvelope::new("out/model", manifest, receipt);
        let json = env.to_canonical_json().unwrap();
        let parsed: ManifestEnvelope = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.image_dir, "out/model");
        assert_eq!(parsed.manifest.image_version, "0.1.0");
        assert_eq!(parsed.manifest.source.quantization_bits, 4);
    }

    #[test]
    fn representation_aware_admission_estimate_uses_manifest_footprint() {
        let arch = serde_json::json!({
            "max_position_embeddings": 4096,
            "head_dim": 128,
            "global_head_dim": 128,
            "hidden_size": 4096,
            "vocab_size": 1000,
        });
        let manifest = header::Manifest {
            image_version: "0.1.0".into(),
            compiler_version: "0.1.0".into(),
            runtime_abi: "prism/0.1.0".into(),
            hardware_target: None,
            readiness: None,
            compile_date: String::new(),
            compile_host: String::new(),
            source: types::SourceIdentity {
                config_hash: String::new(),
                shard_hashes: Vec::new(),
                tokenizer_hashes: Vec::new(),
                auxiliary_hashes: Vec::new(),
                model_type: "qwen3".into(),
                quantization_bits: 4,
                quantization_group_size: 128,
                quantization_mode: "nf4".into(),
            },
            architecture: arch,
            vision_config: None,
            audio_config: None,
            segments: Vec::new(),
            tensor_table: Vec::new(),
            alias_table: Vec::new(),
            residency_plan: ResidencyPlan {
                persistent_segments: Vec::new(),
                layer_segments: Vec::new(),
                layer_window_size: 2,
                total_bytes: 0,
            },
            image_hash: String::new(),
            required_storage_abi: STORAGE_ABI_MAPPED_NO_COPY_V1.to_string(),
            required_capabilities: Vec::new(),
            prepacked_layout: "none".into(),
            metallib_hash: None,
            metallib_size: None,
            metal_kernel_artifacts: Vec::new(),
            execution_plan: serde_json::json!({}),
            compatibility_receipt: None,
            quantization_profiles: Vec::new(),
            quantization_quality: Vec::new(),
            quantization_quality_status: types::QuantizationQualityStatus::Unknown,
        };
        let est = representation_aware_admission_estimate(&manifest);
        // mapped backend -> virtual_mapped_bytes is zero (no segments).
        assert_eq!(est.virtual_mapped_bytes, 0);
        // expected_resident is the rope + mlx_workspace (no persistent
        // segments) — and we have at least the mlx_workspace constant.
        assert!(est.expected_resident_bytes >= 512 * 1024 * 1024);
        // rope_bytes is positive (4k * 128 * 4 * 2 = 4MiB).
        assert!(est.rope_bytes > 0);
    }
}
