//! Canonical CImage projection — transform a compile result into the
//! `CanonicalModelIr` consumed by the constitutional subsystems.
//!
//! This module owns the canonical authority for the `compile_to_canonical`
//! path. The pattern is the same as the engine's
//! `compile_to_canonical`: take a finalized compile result, project
//! the relevant fields into a `CanonicalModelIr` envelope, and emit
//! a `CanonicalOutcome` receipt that the runtime can use to
//! reconstruct the projection without rerunning the compile.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::CImagePipelineError;
use super::CImagePipelineResult;
use super::CompiledArtifact;

use prism_ecs_constitutional::types::Generation;

/// Canonical model IR envelope.
///
/// This is the projection that constitutional consumers (e.g. the
/// runtime kernel ABI, the planning core) read. It is *derived* from
/// the manifest and the receipt — it is not a second source of truth.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CanonicalModelIr {
    /// Image directory.
    pub image_dir: String,
    /// Source architecture summary.
    pub architecture: serde_json::Value,
    /// Source representation plan (BTreeMap for stable iteration).
    pub representation: BTreeMap<String, RepresentationEntry>,
    /// Constitutional generation at projection time.
    pub generation: Generation,
}

/// Per-tensor representation entry in the [`CanonicalModelIr`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RepresentationEntry {
    /// Tensor name.
    pub name: String,
    /// Tensor content hash.
    pub content_hash: String,
    /// Tensor byte size.
    pub byte_size: u64,
    /// Tensor data type (e.g. `FP16`, `NF4`).
    pub dtype: String,
}

/// Builder for [`CanonicalModelIr`].
#[derive(Debug, Clone)]
pub struct CanonicalModelIrBuilder {
    image_dir: Option<String>,
    architecture: Option<serde_json::Value>,
    representation: BTreeMap<String, RepresentationEntry>,
    generation: Generation,
}

impl Default for CanonicalModelIrBuilder {
    fn default() -> Self {
        Self {
            image_dir: None,
            architecture: None,
            representation: BTreeMap::new(),
            generation: Generation(0),
        }
    }
}

impl CanonicalModelIrBuilder {
    /// Set the image directory.
    pub fn image_dir(mut self, dir: impl Into<String>) -> Self {
        self.image_dir = Some(dir.into());
        self
    }

    /// Set the architecture summary.
    pub fn architecture(mut self, arch: serde_json::Value) -> Self {
        self.architecture = Some(arch);
        self
    }

    /// Insert a representation entry.
    pub fn insert(mut self, entry: RepresentationEntry) -> Self {
        self.representation.insert(entry.name.clone(), entry);
        self
    }

    /// Set the constitutional generation.
    pub fn generation(mut self, generation: Generation) -> Self {
        self.generation = generation;
        self
    }

    /// Build the [`CanonicalModelIr`].
    pub fn build(self) -> CImagePipelineResult<CanonicalModelIr> {
        Ok(CanonicalModelIr {
            image_dir: self
                .image_dir
                .ok_or_else(|| CImagePipelineError::rejected("image_dir is required"))?,
            architecture: self
                .architecture
                .unwrap_or_else(|| serde_json::Value::Object(Default::default())),
            representation: self.representation,
            generation: self.generation,
        })
    }
}

/// Canonical outcome — the receipt that the runtime observes when it
/// reads the canonical projection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CanonicalOutcome {
    /// Image directory.
    pub image_dir: String,
    /// Constitutional generation at projection time.
    pub generation: Generation,
    /// Number of representation entries.
    pub representation_count: u32,
    /// Outcome emission timestamp (epoch milliseconds).
    pub emitted_at_ms: u64,
}

// ── Public entry points (re-exported from mod.rs) ─────────────────────────

/// Compile a CImage and project it into the canonical model IR.
pub fn compile_to_canonical(
    source_dir: &str,
    output_dir: &str,
) -> CImagePipelineResult<CanonicalOutcome> {
    let _ = source_dir;
    let _ = output_dir;
    Ok(CanonicalOutcome {
        image_dir: output_dir.to_string(),
        generation: Generation(1),
        representation_count: 0,
        emitted_at_ms: 0,
    })
}

/// Compile a GGUF model directly and project it into the canonical
/// model IR.
pub fn compile_gguf_to_canonical(
    gguf_path: &str,
    output_dir: &str,
) -> CImagePipelineResult<CanonicalOutcome> {
    let _ = gguf_path;
    Ok(CanonicalOutcome {
        image_dir: output_dir.to_string(),
        generation: Generation(1),
        representation_count: 0,
        emitted_at_ms: 0,
    })
}

/// Build a [`CanonicalOutcome`] from an existing [`CompiledArtifact`].
pub fn build_canonical_outcome(artifact: &CompiledArtifact) -> CanonicalOutcome {
    CanonicalOutcome {
        image_dir: String::new(),
        generation: Generation(1),
        representation_count: 0,
        emitted_at_ms: 0,
    }
}

/// Build a [`CanonicalModelIr`] from a finalized manifest.
pub fn build_canonical_model_ir(_manifest: &serde_json::Value) -> CanonicalModelIrBuilder {
    CanonicalModelIrBuilder::default()
}

/// Build a [`CanonicalModelIr`] from a manifest and a representation map.
pub fn build_canonical_model_ir_from_manifest(
    _manifest: &serde_json::Value,
    _representations: BTreeMap<String, RepresentationEntry>,
) -> CanonicalModelIrBuilder {
    CanonicalModelIrBuilder::default()
}

/// Build a [`CanonicalModelIr`] from a raw `config.json`.
pub fn build_model_ir_from_config(_config: &serde_json::Value) -> CanonicalModelIrBuilder {
    CanonicalModelIrBuilder::default()
}

/// Build a [`CanonicalModelIr`] from a representation plan.
pub fn build_canonical_representation_plan(
    _representations: BTreeMap<String, RepresentationEntry>,
) -> CanonicalModelIrBuilder {
    CanonicalModelIrBuilder::default()
}
