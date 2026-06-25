//! PRISM-MODEL-TO-CIMAGE-0001 — Shared runtime vocabulary for the alpha release.
//!
//! These are the canonical types that every workstream uses: compute image
//! format, model identity, capability signature, import request, and
//! region execution plan.
//!
//! A compute image is immutable after sealing. Runtime state is mutable but
//! must always identify the image digest, artifact digest, slot generation,
//! and route origin that produced it.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::compilation::phase_ir::TensorDtype;
use crate::compilation::region_planner::RegionExecutionPlan;

// ── Compute image ────────────────────────────────────────────────────────

/// A sealed, immutable compute image produced by the model importer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrismComputeImage {
    /// Image format version.
    pub image_version: u32,
    /// Model identity digest.
    pub model_identity: ModelIdentity,
    /// Capability signature for this image.
    pub capability_signature: CapabilitySignature,
    /// Manifest of compiled artifacts (Core ML, Metal, CPU).
    pub artifact_manifest: PrismArtifactManifest,
    /// Region dependency and execution plan.
    pub region_plan: RegionExecutionPlan,
    /// Shared activation arena manifest.
    pub shared_arena_manifest: PrismArenaManifest,
    /// KV-cache buffer manifest.
    pub kv_manifest: PrismKvManifest,
    /// Installation policy.
    pub install_policy: InstallPolicy,
    /// SHA-256 hex digest of all preceding fields.
    pub image_digest: String,
}

// ── Model identity ───────────────────────────────────────────────────────

/// Identity and provenance of the imported model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelIdentity {
    /// Model family name, e.g. "qwen2".
    pub family: String,
    /// Full architecture description.
    pub architecture: String,
    /// Total parameter count.
    pub parameter_count: u64,
    /// SHA-256 hex digest of the tokenizer files.
    pub tokenizer_digest: String,
    /// SHA-256 hex digest of weight files.
    pub weight_digest: String,
    /// SHA-256 hex digest of configuration.
    pub config_digest: String,
}

// ── Capability signature ─────────────────────────────────────────────────

/// Declared capability requirements for this compute image.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilitySignature {
    /// Target platform, e.g. "apple-silicon".
    pub platform: String,
    /// SOC family, e.g. "M1+".
    pub soc_family: String,
    /// Minimum OS version.
    pub os_version: String,
    /// Core ML runtime version (if ANE route required).
    pub coreml_runtime_version: Option<String>,
    /// Metal feature set (if GPU route required).
    pub metal_feature_set: Option<String>,
    /// Supported dtypes for this image.
    pub supported_dtypes: Vec<TensorDtype>,
    /// Required hardware features.
    pub required_features: Vec<String>,
    /// SHA-256 hex digest of all preceding fields.
    pub signature_digest: String,
}

// ── Artifact manifest ────────────────────────────────────────────────────

/// References to compiled artifacts within a sealed compute image.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrismArtifactManifest {
    /// Core ML `.mlmodelc` artifacts, keyed by region name.
    pub coreml_artifacts: Vec<CoreMlArtifactEntry>,
    /// Metal shader libraries, keyed by kernel name.
    pub metal_artifacts: Vec<MetalArtifactEntry>,
    /// CPU fallback artifacts (if any).
    pub cpu_artifacts: Vec<CpuArtifactEntry>,
    /// Weight pack metadata.
    pub weight_packs: Vec<WeightPackEntry>,
}

/// A compiled Core ML artifact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreMlArtifactEntry {
    /// Region name matched to catalogue entry.
    pub region_name: String,
    /// Relative path within the image directory.
    pub path: String,
    /// SHA-256 hex digest of the artifact.
    pub artifact_digest: String,
}

/// A compiled Metal shader library.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetalArtifactEntry {
    /// Kernel function name.
    pub kernel_name: String,
    /// Relative path within the image directory.
    pub path: String,
    /// Metal feature set requirement.
    pub feature_set: String,
    /// SHA-256 hex digest.
    pub artifact_digest: String,
}

/// A CPU fallback artifact (e.g. reference kernel).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuArtifactEntry {
    /// Region name.
    pub region_name: String,
    /// Relative path.
    pub path: String,
    /// SHA-256 hex digest.
    pub artifact_digest: String,
}

/// Weight pack metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightPackEntry {
    /// Pack name, e.g. "layer_0_weights".
    pub pack_name: String,
    /// Tensor count.
    pub tensor_count: u32,
    /// Packed byte size.
    pub byte_size: u64,
    /// SHA-256 hex digest.
    pub pack_digest: String,
}

// ── Region execution plan ────────────────────────────────────────────────

/// Execution plan for all regions in one decoder layer iteration.

// ── Arena manifest ───────────────────────────────────────────────────────

/// Manifest for the shared activation arena.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrismArenaManifest {
    /// Total number of slots.
    pub slot_count: u32,
    /// Per-slot descriptors.
    pub slots: Vec<ArenaSlotDescriptor>,
}

/// Descriptor for one arena slot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArenaSlotDescriptor {
    /// Slot index.
    pub slot_index: u32,
    /// Dtype.
    pub dtype: TensorDtype,
    /// Logical shape.
    pub logical_shape: Vec<u32>,
    /// Physical shape (with padding).
    pub physical_shape: Vec<u32>,
    /// Stride in bytes.
    pub stride: u64,
}

// ── KV-cache manifest ────────────────────────────────────────────────────

/// Manifest for KV-cache buffers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrismKvManifest {
    /// Number of KV layers.
    pub layer_count: u32,
    /// Maximum sequence length.
    pub max_seq_length: u32,
    /// Head dimension.
    pub head_dim: u32,
    /// Number of KV heads.
    pub kv_head_count: u32,
    /// Dtype.
    pub dtype: TensorDtype,
    /// Total byte size per layer.
    pub bytes_per_layer: u64,
}

// ── Install policy ───────────────────────────────────────────────────────

/// Installation policy for this compute image.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstallPolicy {
    /// Whether installation requires Apple Silicon.
    pub requires_apple_silicon: bool,
    /// Minimum memory in bytes.
    pub min_memory_bytes: u64,
    /// Minimum free disk bytes.
    pub min_free_disk_bytes: u64,
    /// Whether to precreate Metal textures during install.
    pub precreate_metal_textures: bool,
    /// Whether to run Core ML warmup during install.
    pub run_coreml_warmup: bool,
}

// ── Import request ───────────────────────────────────────────────────────

/// Request to import a model into a sealed compute image.
#[derive(Debug, Clone)]
pub struct ModelImportRequest {
    /// Path to the local model directory.
    pub model_path: PathBuf,
    /// Target hardware profile.
    pub target: TargetProfile,
    /// Static context bucket for this image.
    pub context_bucket: u32,
    /// Requested activation dtype.
    pub requested_dtype: TensorDtype,
}

/// Target hardware profile.
#[derive(Debug, Clone)]
pub struct TargetProfile {
    /// Platform identifier.
    pub platform: String,
    /// Primary execution lane.
    pub primary_lane: ExecutionLane,
    /// Fallback execution lane.
    pub fallback_lane: ExecutionLane,
    /// Static batch size (1 for alpha).
    pub static_batch: u32,
    /// Static sequence bucket.
    pub static_sequence_bucket: u32,
}

// ── Import result ────────────────────────────────────────────────────────

/// Result of importing a model into a compute image.
#[derive(Debug)]
pub struct ModelImportResult {
    /// The sealed compute image.
    pub image: PrismComputeImage,
    /// Path to the image directory on disk.
    pub image_path: PathBuf,
}

use crate::backend::placement::ExecutionLane;
