//! `Manifest` top-level schema — the canonical on-disk header of a compiled
//! CImage directory.
//!
//! This module owns the constitutional authority for the [`Manifest`]
//! struct itself and the per-segment types ([`Segment`],
//! [`SegmentKind`]) that constitute the manifest's top-level shape. The
//! manifest is the **durable schema** that every CImage directory writes
//! to `manifest.json`; the receipt (see [`super::receipt`]) is the
//! **post-emission evidence** derived from it.
//!
//! The module does **not** own the per-tensor table (see
//! [`super::types`]), the lease state machine (see [`super::lease`]),
//! or the kernel dispatch recipes (see [`super::kernel`]). Those
//! authorities each have their own file; this file is the
//! manifest-top-level surface only.

use serde::{Deserialize, Serialize};

// Re-export the per-file SourceIdentity from the types module.
pub use super::types::SourceIdentity;

// ── Storage ABI constants ─────────────────────────────────────────────────

/// Storage ABI identifier for the baseline copied (CPU-allocated) path.
pub const STORAGE_ABI_COPIED_V0: &str = "copied-v0";
/// Storage ABI identifier for the mapped, no-copy (Metal-buffer) path.
pub const STORAGE_ABI_MAPPED_NO_COPY_V1: &str = "mapped-no-copy-v1";

/// Return true if `abi` is a recognised storage ABI identifier.
pub fn is_valid_storage_abi(abi: &str) -> bool {
    abi == STORAGE_ABI_COPIED_V0 || abi == STORAGE_ABI_MAPPED_NO_COPY_V1
}

// ── Storage ABI specification ─────────────────────────────────────────────

/// Specification for the mapped-no-copy-v1 storage ABI.
#[derive(Debug, Clone)]
pub struct StorageAbiSpec {
    pub abi_id: String,
    /// Minimum segment file alignment in bytes (must be a multiple of page size).
    pub segment_alignment_bytes: u64,
    /// Minimum tensor offset alignment within a segment.
    pub tensor_offset_alignment_bytes: u64,
    /// Supported physical dtypes in storage order.
    pub supported_physical_dtypes: Vec<String>,
    /// Byte order (always "le" for Apple Silicon).
    pub byte_order: String,
    /// Layout version for cache key stability.
    pub layout_version: u32,
    /// Weight tensor prepack layout identity.
    pub prepacked_layout: String,
}

impl StorageAbiSpec {
    /// Return the canonical mapped-no-copy-v1 ABI specification.
    pub fn mapped_no_copy_v1() -> Self {
        Self {
            abi_id: STORAGE_ABI_MAPPED_NO_COPY_V1.to_string(),
            segment_alignment_bytes: 4096,
            tensor_offset_alignment_bytes: 16,
            supported_physical_dtypes: vec![
                "U8".into(),
                "I8".into(),
                "F16".into(),
                "BF16".into(),
                "F32".into(),
                "U32".into(),
            ],
            byte_order: "le".into(),
            layout_version: 1,
            prepacked_layout: "none".into(),
        }
    }
}

// ── Top-level manifest ─────────────────────────────────────────────────────

/// Top-level ComputeImage manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub image_version: String,
    pub compiler_version: String,
    pub runtime_abi: String,
    /// Target hardware this image was compiled for (None = auto-detect at compile time).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hardware_target: Option<String>,
    /// Compilation readiness verdict after artifact audit.
    #[serde(default)]
    pub readiness: Option<CompileReadiness>,
    /// ISO 8601 timestamp of compilation.
    #[serde(default)]
    pub compile_date: String,
    /// Hostname of the machine that compiled this image.
    #[serde(default)]
    pub compile_host: String,
    /// Cryptographic identity of the source checkpoint.
    pub source: SourceIdentity,
    /// Architecture summary (free-form JSON value so we don't bind the
    /// manifest to a particular `TextArchitecture` shape — the manifest
    /// is the durable schema and the architecture is a projection).
    pub architecture: serde_json::Value,
    /// Vision encoder configuration (vision_config from model config.json).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vision_config: Option<serde_json::Value>,
    /// Audio encoder configuration (Gemma 4 Unified audio_config).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_config: Option<serde_json::Value>,
    pub segments: Vec<Segment>,
    pub tensor_table: Vec<super::types::TensorEntry>,
    pub alias_table: Vec<super::types::AliasEntry>,
    pub residency_plan: ResidencyPlan,
    pub image_hash: String,
    /// Storage ABI required by this image (e.g. "copied-v0", "mapped-no-copy-v1").
    #[serde(default = "default_storage_abi")]
    pub required_storage_abi: String,
    /// Capabilities the runtime must support to execute this image.
    #[serde(default)]
    pub required_capabilities: Vec<String>,
    /// Weight tensor prepack layout.
    /// "none" = source layout (int8 weights in standard [K,N] row-major).
    /// "prepacked-int8-v1" = transposed [N,K] with interleaved scale/bias per group.
    #[serde(default = "default_prepacked_layout")]
    pub prepacked_layout: String,
    /// SHA-256 of the precompiled Metal library bundle (.metallib) embedded in
    /// this image.  `None` means no metallib is available — the runtime MUST
    /// fall back to JIT-compiling Metal shaders at inference time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metallib_hash: Option<String>,
    /// Byte size of the precompiled Metal library bundle (.metallib) when
    /// `metallib_hash` is set.  `None` when no metallib is present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metallib_size: Option<u64>,
    /// Pre-compiled Metal kernel artifacts embedded in this image.
    #[serde(default)]
    pub metal_kernel_artifacts: Vec<super::kernel::MetalKernelArtifact>,
    /// Execution plan emitted by the compiler (prologue, layers, epilogue).
    #[serde(default)]
    pub execution_plan: serde_json::Value,
    /// CompatibilityMatrix validation receipt from compile time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compatibility_receipt: Option<serde_json::Value>,
    /// Quantization profile registry — profiles used by this image.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub quantization_profiles: Vec<super::types::QuantizationProfileEntry>,
    /// Per-tensor quantization quality evidence.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub quantization_quality: Vec<super::types::QuantizationQualityEntry>,
    /// Overall quantization quality status.
    #[serde(default)]
    pub quantization_quality_status: super::types::QuantizationQualityStatus,
}

/// Compilation readiness verdict after artifact audit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompileReadiness {
    /// Every required lane artifact passed validation.
    Ready,
    /// Image can serve through an approved fallback route but one optional
    /// lane failed to compile.
    ReadyDegraded { reason: Option<String> },
    /// No valid complete route exists.
    Rejected { reason: String },
}

impl Manifest {
    /// Check whether the manifest's `required_storage_abi` is compatible with
    /// the selected `StorageBackend`.
    pub fn storage_abi_matches(&self, backend: super::lease::StorageBackend) -> bool {
        match backend {
            super::lease::StorageBackend::Copied => {
                self.required_storage_abi == STORAGE_ABI_COPIED_V0
            }
            super::lease::StorageBackend::MappedNoCopy => {
                self.required_storage_abi == STORAGE_ABI_MAPPED_NO_COPY_V1
            }
        }
    }
}

// ── Per-segment types ─────────────────────────────────────────────────────

/// One binary segment containing tensors in execution order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    pub id: String,
    pub filename: String,
    pub byte_size: u64,
    pub sha256: String,
    pub tensor_ids: Vec<u32>,
    pub kind: SegmentKind,
    /// Alignment constraint in bytes for the mapped-no-copy backend (default 4096).
    #[serde(default = "default_alignment_bytes")]
    pub alignment_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SegmentKind {
    /// Always loaded (embeddings, final norm).
    Persistent,
    /// Per-layer, load/free per execution window.
    Layer(u32),
    /// Output projection (may alias Persistent).
    Final,
}

/// Runtime residency plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidencyPlan {
    /// Segments always loaded.
    pub persistent_segments: Vec<String>,
    /// Per-layer segments in execution order.
    pub layer_segments: Vec<String>,
    /// Max layers to keep resident simultaneously.
    pub layer_window_size: u32,
    /// Total image size in bytes.
    pub total_bytes: u64,
}

fn default_storage_abi() -> String {
    "copied-v0".to_string()
}
fn default_prepacked_layout() -> String {
    "none".to_string()
}
fn default_alignment_bytes() -> u64 {
    4096
}
pub(crate) fn alignment_default() -> u64 {
    default_alignment_bytes()
}

/// Validate that every segment in `segments` has `alignment_bytes` that is
/// a multiple of `min_alignment` (typically 4096 for the mapped-no-copy
/// backend).
pub fn validate_segment_alignment(
    segments: &[Segment],
    min_alignment: u64,
) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    for seg in segments {
        if seg.alignment_bytes % min_alignment != 0 {
            errors.push(format!(
                "segment {} alignment_bytes {} is not a multiple of {}",
                seg.id, seg.alignment_bytes, min_alignment,
            ));
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Validate the entire `Manifest` against a given `StorageAbiSpec`.
///
/// Checks:
/// - All segments have `alignment_bytes` that is a multiple of the ABI's
///   `segment_alignment_bytes`.
///
/// Returns `Err(Vec<String>)` with every violation; does not short-circuit.
pub fn validate_manifest_for_abi(
    manifest: &Manifest,
    spec: &StorageAbiSpec,
) -> Result<(), Vec<String>> {
    validate_segment_alignment(&manifest.segments, spec.segment_alignment_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_valid_storage_abi_recognises_both_abi_ids() {
        assert!(is_valid_storage_abi(STORAGE_ABI_COPIED_V0));
        assert!(is_valid_storage_abi(STORAGE_ABI_MAPPED_NO_COPY_V1));
        assert!(!is_valid_storage_abi("mystery-v9"));
    }

    #[test]
    fn mapped_no_copy_v1_spec_uses_baseline_constants() {
        let spec = StorageAbiSpec::mapped_no_copy_v1();
        assert_eq!(spec.abi_id, STORAGE_ABI_MAPPED_NO_COPY_V1);
        assert_eq!(spec.segment_alignment_bytes, 4096);
        assert_eq!(spec.tensor_offset_alignment_bytes, 16);
        assert_eq!(spec.byte_order, "le");
        assert_eq!(spec.layout_version, 1);
    }

    #[test]
    fn mapped_no_copy_v1_spec_supports_baseline_dtypes() {
        let spec = StorageAbiSpec::mapped_no_copy_v1();
        let mut dtypes = spec.supported_physical_dtypes.clone();
        dtypes.sort();
        assert_eq!(
            dtypes,
            vec![
                "BF16".to_string(),
                "F16".to_string(),
                "F32".to_string(),
                "I8".to_string(),
                "U32".to_string(),
                "U8".to_string(),
            ],
        );
    }

    #[test]
    fn segment_alignment_validator_reports_all_violations() {
        let segments = vec![
            Segment {
                id: "a".into(),
                filename: "segment_000.bin".into(),
                byte_size: 0,
                sha256: "00".into(),
                tensor_ids: Vec::new(),
                kind: SegmentKind::Persistent,
                alignment_bytes: 2048,
            },
            Segment {
                id: "b".into(),
                filename: "segment_001.bin".into(),
                byte_size: 0,
                sha256: "00".into(),
                tensor_ids: Vec::new(),
                kind: SegmentKind::Persistent,
                alignment_bytes: 4096,
            },
        ];
        let errors = validate_segment_alignment(&segments, 4096).unwrap_err();
        // Only segment "a" violates; segment "b" is fine.
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("a"));
    }

    #[test]
    fn manifest_storage_abi_matches_uses_canonical_strings() {
        let mut m = empty_manifest();
        m.required_storage_abi = STORAGE_ABI_COPIED_V0.to_string();
        assert!(m.storage_abi_matches(super::super::lease::StorageBackend::Copied));
        assert!(!m.storage_abi_matches(super::super::lease::StorageBackend::MappedNoCopy));

        m.required_storage_abi = STORAGE_ABI_MAPPED_NO_COPY_V1.to_string();
        assert!(!m.storage_abi_matches(super::super::lease::StorageBackend::Copied));
        assert!(m.storage_abi_matches(super::super::lease::StorageBackend::MappedNoCopy));
    }

    #[test]
    fn compile_readiness_serde_round_trip_preserves_reason() {
        let r = CompileReadiness::ReadyDegraded {
            reason: Some("ane lane missing".into()),
        };
        let json = serde_json::to_string(&r).unwrap();
        let parsed: CompileReadiness = serde_json::from_str(&json).unwrap();
        match parsed {
            CompileReadiness::ReadyDegraded { reason } => {
                assert_eq!(reason.as_deref(), Some("ane lane missing"));
            }
            other => panic!("unexpected variant: {other:?}"),
        }
    }

    #[test]
    fn validate_manifest_for_abi_propagates_segment_errors() {
        let mut m = empty_manifest();
        m.segments.push(Segment {
            id: "a".into(),
            filename: "segment_000.bin".into(),
            byte_size: 0,
            sha256: "00".into(),
            tensor_ids: Vec::new(),
            kind: SegmentKind::Persistent,
            alignment_bytes: 2048,
        });
        let spec = StorageAbiSpec::mapped_no_copy_v1();
        let errs = validate_manifest_for_abi(&m, &spec).unwrap_err();
        assert_eq!(errs.len(), 1);
    }

    fn empty_manifest() -> Manifest {
        Manifest {
            image_version: "0.1.0".into(),
            compiler_version: "0.1.0".into(),
            runtime_abi: "prism/0.1.0".into(),
            hardware_target: None,
            readiness: None,
            compile_date: String::new(),
            compile_host: String::new(),
            source: super::super::types::SourceIdentity {
                config_hash: String::new(),
                shard_hashes: Vec::new(),
                tokenizer_hashes: Vec::new(),
                auxiliary_hashes: Vec::new(),
                model_type: String::new(),
                quantization_bits: 0,
                quantization_group_size: 0,
                quantization_mode: String::new(),
            },
            architecture: serde_json::json!({}),
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
            quantization_quality_status: super::super::types::QuantizationQualityStatus::Unknown,
        }
    }
}
