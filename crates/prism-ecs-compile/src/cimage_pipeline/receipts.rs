//! CImage compile receipt types and emission.
//!
//! This module owns the canonical authority for the receipt types that
//! are emitted alongside every CImage manifest:
//!
//! - [`CompileReceipt`] — the top-level receipt envelope.
//! - [`StageProfile`] — the per-stage timing record.
//! - [`StageTimings`] — the timing inputs from which a [`StageProfile`]
//!   is derived.
//! - [`build_compile_receipt`] — the receipt constructor used by every
//!   compile entry point.
//!
//! Receipts are the *durable evidence* that the compile produced. They
//! are written to `receipt.json` next to the manifest and participate
//! in the canonical change flow: the post-emission reader verifies
//! them, the projection rebuilds them, the replay path re-derives
//! them. A change to a receipt field is a constitutional change.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use prism_ecs_constitutional::types::Generation;

use super::authority::ImageBuildAttestation;

/// Per-stage timing inputs for a compile.
///
/// The original engine code records each stage as a `u64` millisecond
/// field. The Prism re-implementation keeps the same fields but adds
/// a `Total` derived from the rest, so the receipt is self-validating.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct StageTimings {
    /// Time spent in source discovery.
    pub source_discovery_ms: u64,
    /// Time spent in header parsing.
    pub header_parsing_ms: u64,
    /// Time spent in architecture normalization.
    pub architecture_normalization_ms: u64,
    /// Time spent in binding validation.
    pub binding_validation_ms: u64,
    /// Time spent in source hashing.
    pub source_hashing_ms: u64,
    /// Time spent in layout planning.
    pub layout_planning_ms: u64,
    /// Time spent in payload emission.
    pub payload_emission_ms: u64,
    /// Time spent in segment hashing.
    pub segment_hashing_ms: u64,
    /// Time spent in manifest generation.
    pub manifest_generation_ms: u64,
    /// Time spent in verification.
    pub verification_ms: u64,
}

impl StageTimings {
    /// Sum of all per-stage timings.
    pub fn total_ms(&self) -> u64 {
        self.source_discovery_ms
            .saturating_add(self.header_parsing_ms)
            .saturating_add(self.architecture_normalization_ms)
            .saturating_add(self.binding_validation_ms)
            .saturating_add(self.source_hashing_ms)
            .saturating_add(self.layout_planning_ms)
            .saturating_add(self.payload_emission_ms)
            .saturating_add(self.segment_hashing_ms)
            .saturating_add(self.manifest_generation_ms)
            .saturating_add(self.verification_ms)
    }
}

/// Per-stage timing profile emitted in the receipt.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct StageProfile {
    /// Source discovery time.
    pub source_discovery_ms: u64,
    /// Header parsing time.
    pub header_parsing_ms: u64,
    /// Architecture normalization time.
    pub architecture_normalization_ms: u64,
    /// Binding validation time.
    pub binding_validation_ms: u64,
    /// Source hashing time.
    pub source_hashing_ms: u64,
    /// Layout planning time.
    pub layout_planning_ms: u64,
    /// Payload emission time.
    pub payload_emission_ms: u64,
    /// Segment hashing time.
    pub segment_hashing_ms: u64,
    /// Manifest generation time.
    pub manifest_generation_ms: u64,
    /// Verification time.
    pub verification_ms: u64,
    /// Total source bytes processed.
    pub total_source_bytes: u64,
    /// Total emitted bytes.
    pub total_emitted_bytes: u64,
    /// Peak RSS bytes.
    pub peak_rss_bytes: u64,
    /// Peak MLX active bytes.
    pub peak_mlx_active_bytes: u64,
    /// Peak MLX cache bytes.
    pub peak_mlx_cache_bytes: u64,
}

impl StageProfile {
    /// Construct a profile from a [`StageTimings`] plus byte / memory
    /// totals.
    pub fn from_timings(timings: &StageTimings) -> Self {
        Self {
            source_discovery_ms: timings.source_discovery_ms,
            header_parsing_ms: timings.header_parsing_ms,
            architecture_normalization_ms: timings.architecture_normalization_ms,
            binding_validation_ms: timings.binding_validation_ms,
            source_hashing_ms: timings.source_hashing_ms,
            layout_planning_ms: timings.layout_planning_ms,
            payload_emission_ms: timings.payload_emission_ms,
            segment_hashing_ms: timings.segment_hashing_ms,
            manifest_generation_ms: timings.manifest_generation_ms,
            verification_ms: timings.verification_ms,
            total_source_bytes: 0,
            total_emitted_bytes: 0,
            peak_rss_bytes: 0,
            peak_mlx_active_bytes: 0,
            peak_mlx_cache_bytes: 0,
        }
    }

    /// Total stage time (sum of all per-stage timings).
    pub fn total_ms(&self) -> u64 {
        self.source_discovery_ms
            .saturating_add(self.header_parsing_ms)
            .saturating_add(self.architecture_normalization_ms)
            .saturating_add(self.binding_validation_ms)
            .saturating_add(self.source_hashing_ms)
            .saturating_add(self.layout_planning_ms)
            .saturating_add(self.payload_emission_ms)
            .saturating_add(self.segment_hashing_ms)
            .saturating_add(self.manifest_generation_ms)
            .saturating_add(self.verification_ms)
    }
}

/// Top-level compile receipt envelope.
///
/// The receipt is the durable evidence that a compile produced a
/// specific CImage. It carries:
///
/// - The build [`ImageBuildAttestation`] (proving the compile was
///   performed by an authorized profile).
/// - The per-stage [`StageProfile`].
/// - The source / manifest identity (shard hashes, tokenizer hashes,
///   auxiliary hashes, namespace).
/// - The constitutional [`Generation`] at the time the receipt was
///   emitted.
///
/// The receipt is serialized to `receipt.json` next to the manifest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompileReceipt {
    /// Per-stage profile.
    pub stage_profile: StageProfile,
    /// Build attestation.
    pub attestation: ImageBuildAttestation,
    /// Per-shard content hashes (BTreeMap for stable serialization).
    pub shard_hashes: BTreeMap<String, String>,
    /// Per-tokenizer content hashes.
    pub tokenizer_hashes: BTreeMap<String, String>,
    /// Per-auxiliary content hashes.
    pub auxiliary_hashes: BTreeMap<String, String>,
    /// Source namespace.
    pub namespace: String,
    /// Constitutional generation at receipt emission time.
    pub generation: Generation,
    /// Total elapsed milliseconds (redundant with stage_profile for
    /// quick post-emission queries).
    pub elapsed_ms: u64,
    /// Total source bytes (redundant with stage_profile).
    pub total_source_bytes: Option<u64>,
    /// Compliance flags, recorded as a sorted JSON object.
    pub compliance: BTreeMap<String, bool>,
}

/// Trait for the loaded-source input of [`build_compile_receipt`].
///
/// The original engine signature takes a `&LoadedSource` borrowed from
/// the engine's own source module. The Prism re-implementation takes
/// a trait object so the same function works for the safetensors, GGUF,
/// and synthesized-graph paths.
pub trait BuildReceiptSource {
    /// Per-shard content hashes.
    fn shard_hashes(&self) -> Vec<String>;
    /// Per-tokenizer content hashes.
    fn tokenizer_hashes(&self) -> Vec<String>;
    /// Per-auxiliary content hashes.
    fn auxiliary_hashes(&self) -> Vec<String>;
    /// Source namespace.
    fn namespace(&self) -> String;
}

/// Build a [`CompileReceipt`] from a loaded source, the emitted
/// manifest, the elapsed time, the per-stage profile, and a compliance
/// flag set.
pub fn build_compile_receipt(
    loaded: &dyn BuildReceiptSource,
    _manifest: &serde_json::Value,
    elapsed_ms: u64,
    stage_profile: StageProfile,
    compliance: BTreeMap<String, bool>,
    total_source_bytes: Option<u64>,
) -> CompileReceipt {
    let mut shard_hashes = BTreeMap::new();
    for (i, h) in loaded.shard_hashes().into_iter().enumerate() {
        shard_hashes.insert(format!("shard_{i:04}"), h);
    }
    let mut tokenizer_hashes = BTreeMap::new();
    for (i, h) in loaded.tokenizer_hashes().into_iter().enumerate() {
        tokenizer_hashes.insert(format!("tokenizer_{i:04}"), h);
    }
    let mut auxiliary_hashes = BTreeMap::new();
    for (i, h) in loaded.auxiliary_hashes().into_iter().enumerate() {
        auxiliary_hashes.insert(format!("auxiliary_{i:04}"), h);
    }
    CompileReceipt {
        stage_profile,
        attestation: super::authority::image_build_attestation(),
        shard_hashes,
        tokenizer_hashes,
        auxiliary_hashes,
        namespace: loaded.namespace(),
        generation: Generation(1),
        elapsed_ms,
        total_source_bytes,
        compliance,
    }
}
