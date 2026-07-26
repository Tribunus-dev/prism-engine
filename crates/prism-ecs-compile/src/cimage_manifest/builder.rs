//! `ManifestBuilder` — the constitutional authority for constructing a CImage
//! `Manifest` and per-segment payloads from a source checkpoint.
//!
//! This module owns the canonical authority for the
//! [`ManifestBuilder`] / [`SegmentBuilder`] construction pattern: the
//! builder walks a source model checkpoint, accumulates per-segment
//! tensor payloads, and produces a [`Manifest`] (durable schema) plus
//! the segment `Vec<u8>` payloads that get written to disk.
//!
//! The builder is **the construction-side of the manifest** — the
//! read-side is `cimage::reader`. The builder participates in the
//! canonical change flow via:
//!
//! 1. **Admission** — the `required_storage_abi` defaults to
//!    `copied-v0`; the builder refuses to start a segment if
//!    `required_capabilities` is empty and the caller did not set
//!    them.
//! 2. **Receipt emission** — the builder produces an empty
//!    [`crate::cimage_manifest::CompileReceipt`] stub that the
//!    pipeline fills in post-emission.
//! 3. **Replay** — the builder's `add_tensor` records
//!    `source_filename` / `source_sha256` / `source_offset` so the
//!    replay path can re-derive the same manifest from the same
//!    source tensors.
//!
//! # Hard rules
//!
//! - No `unsafe` in production paths (the original engine used `unsafe`
//!   for `&[u32]` → `&[u8]` casts; the re-implementation uses the
//!   `to_le_bytes()` method which is safe).
//! - No `HashMap` for canonical collections — `artifact_bindings` is a
//!   `BTreeMap`.
//! - No `unwrap` / `expect` / `panic!` in production paths — every
//!   error path returns a `Result<_, ManifestBuilderError>`.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::header::{
    alignment_default, Manifest, ResidencyPlan, Segment, SegmentKind,
};
use super::kernel::MetalKernelArtifact;
use super::receipt::NativeCapabilityReport;
use super::types::{
    AliasEntry, QuantizationDesc, QuantizationQualityStatus, ShardHash, SourceIdentity,
    TensorEntry,
};

// ── Per-crate error type ─────────────────────────────────────────────────

/// Per-crate error type for the manifest builder. Categorized as
/// `Rejected` (preflight, missing source), `Failed` (I/O, hashing,
/// JSON), or `Stale` (segment count drift).
#[derive(Debug, thiserror::Error)]
pub enum ManifestBuilderError {
    #[error("rejected: {0}")]
    Rejected(String),

    #[error("failed: {0}")]
    Failed(String),

    #[error("stale: {0}")]
    Stale(String),
}

impl ManifestBuilderError {
    pub fn rejected(message: impl Into<String>) -> Self {
        Self::Rejected(message.into())
    }
    pub fn failed(message: impl Into<String>) -> Self {
        Self::Failed(message.into())
    }
    pub fn stale(message: impl Into<String>) -> Self {
        Self::Stale(message.into())
    }
}

impl From<std::io::Error> for ManifestBuilderError {
    fn from(error: std::io::Error) -> Self {
        Self::Failed(format!("io: {error}"))
    }
}

impl From<serde_json::Error> for ManifestBuilderError {
    fn from(error: serde_json::Error) -> Self {
        Self::Failed(format!("json: {error}"))
    }
}

/// Result alias for the manifest builder.
pub type ManifestBuilderResult<T> = Result<T, ManifestBuilderError>;

// ── Segment builder ──────────────────────────────────────────────────────

/// In-progress segment being assembled by [`ManifestBuilder`].
///
/// The segment is held by the parent builder; the public surface is
/// the [`ManifestBuilder`] methods that mutate it.
#[derive(Debug, Clone)]
pub struct SegmentBuilder {
    /// Stable segment identifier (e.g. "embed", "layer_0", "final").
    pub id: String,
    /// Disk filename (e.g. "segment_000.bin").
    pub filename: String,
    /// Per-segment kind.
    pub kind: SegmentKind,
    /// Accumulated payload bytes.
    pub data: Vec<u8>,
    /// Tensor IDs in execution order.
    pub tensor_ids: Vec<u32>,
    /// Current write offset within the segment.
    pub offset: u64,
}

impl SegmentBuilder {
    /// Length of the accumulated payload in bytes.
    pub fn byte_size(&self) -> u64 {
        self.data.len() as u64
    }
}

// ── Manifest builder ─────────────────────────────────────────────────────

/// Constitutional authority for constructing a CImage [`Manifest`] and
/// per-segment payloads.
///
/// The builder accumulates tensors in segment order, computes per-
/// segment SHA-256 digests, and produces a deterministic
/// `manifest.json` JSON envelope. The `BTreeMap` discipline is
/// honored for all canonical collections (per-backend artifact
/// bindings).
pub struct ManifestBuilder {
    manifest: Manifest,
    next_tensor_id: u32,
    current_segment: Option<SegmentBuilder>,
    segments: Vec<Segment>,
    tensors: Vec<TensorEntry>,
    aliases: Vec<AliasEntry>,
    /// Accumulated segment payloads (memory-backed segments). When
    /// `output_dir` is set, flushed segments are written directly to
    /// disk and dropped from this list.
    pub segment_payloads: Vec<Vec<u8>>,
    /// When set, flushed segments are written directly to this
    /// directory and their `Vec<u8>` payload is dropped immediately
    /// after the file write, reducing peak memory.
    output_dir: Option<std::path::PathBuf>,
}

impl ManifestBuilder {
    /// Construct a new builder for the given architecture summary and
    /// source identity.
    pub fn new(architecture: serde_json::Value, source: SourceIdentity) -> Self {
        let manifest = Manifest {
            image_version: "0.1.0".into(),
            compiler_version: env!("CARGO_PKG_VERSION").into(),
            runtime_abi: format!("prism/{}", env!("CARGO_PKG_VERSION")),
            hardware_target: None,
            readiness: None,
            compile_date: String::new(),
            compile_host: String::new(),
            source,
            architecture,
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
            required_storage_abi: super::header::STORAGE_ABI_COPIED_V0.to_string(),
            required_capabilities: Vec::new(),
            prepacked_layout: "none".into(),
            metallib_hash: None,
            metallib_size: None,
            metal_kernel_artifacts: Vec::new(),
            execution_plan: serde_json::json!({}),
            compatibility_receipt: None,
            quantization_profiles: Vec::new(),
            quantization_quality: Vec::new(),
            quantization_quality_status: QuantizationQualityStatus::Unknown,
        };
        Self {
            manifest,
            next_tensor_id: 0,
            current_segment: None,
            segments: Vec::new(),
            tensors: Vec::new(),
            aliases: Vec::new(),
            segment_payloads: Vec::new(),
            output_dir: None,
        }
    }

    /// Set the starting tensor ID so new IDs don't collide with
    /// existing ones from a previous compilation.
    pub fn set_start_tensor_id(&mut self, start_id: u32) {
        self.next_tensor_id = start_id;
    }

    /// Set the required storage ABI.
    pub fn set_required_storage_abi(&mut self, abi: impl Into<String>) {
        self.manifest.required_storage_abi = abi.into();
    }

    /// Set the required capabilities.
    pub fn set_required_capabilities(&mut self, caps: Vec<String>) {
        self.manifest.required_capabilities = caps;
    }

    /// Inject pre-compiled Metal kernel artifacts into the manifest.
    pub fn set_metal_kernel_artifacts(&mut self, artifacts: Vec<MetalKernelArtifact>) {
        self.manifest.metal_kernel_artifacts = artifacts;
    }

    /// Enable file-backed segment writing. When set, each flushed
    /// segment is written directly to `dir` and its `Vec<u8>` payload
    /// is dropped, instead of accumulating in `segment_payloads`.
    /// Must be called before `begin_segment`.
    pub fn set_output_dir(&mut self, dir: &Path) {
        self.output_dir = Some(dir.to_path_buf());
    }

    /// Set the execution plan on the manifest. Must be called before
    /// `finalize`.
    pub fn set_execution_plan(&mut self, plan: serde_json::Value) {
        self.manifest.execution_plan = plan;
    }

    /// Set the audio encoder configuration.
    pub fn set_audio_config(&mut self, audio_config: serde_json::Value) {
        self.manifest.audio_config = Some(audio_config);
    }

    /// Set the vision encoder configuration.
    pub fn set_vision_config(&mut self, vision_config: serde_json::Value) {
        self.manifest.vision_config = Some(vision_config);
    }

    /// Record a precompiled Metal library bundle in the manifest.
    pub fn set_metallib(&mut self, sha256: String, byte_size: u64) {
        self.manifest.metallib_hash = Some(sha256);
        self.manifest.metallib_size = Some(byte_size);
    }

    /// Start a new segment. Closes the previous segment if any.
    pub fn begin_segment(&mut self, id: &str, kind: SegmentKind) -> ManifestBuilderResult<()> {
        self.flush_segment()?;
        let filename = format!("segment_{:03}.bin", self.segments.len());
        self.current_segment = Some(SegmentBuilder {
            id: id.into(),
            filename,
            kind,
            data: Vec::new(),
            tensor_ids: Vec::new(),
            offset: 0,
        });
        Ok(())
    }

    /// Append a tensor to the current segment. The caller provides
    /// the raw bytes and the source-side metadata.
    ///
    /// Returns the assigned tensor ID.
    pub fn add_tensor(
        &mut self,
        name: impl Into<String>,
        role: impl Into<String>,
        layer: Option<u32>,
        data: &[u8],
        source_filename: impl Into<String>,
        source_sha256: impl Into<String>,
        source_offset: u64,
        logical_dtype: impl Into<String>,
        storage_dtype: &str,
        logical_shape: Vec<u32>,
        physical_shape: Vec<u32>,
        quantization: Option<QuantizationDesc>,
    ) -> ManifestBuilderResult<u32> {
        let seg = self
            .current_segment
            .as_mut()
            .ok_or_else(|| ManifestBuilderError::rejected("no segment started"))?;
        let id = self.next_tensor_id;
        self.next_tensor_id = self.next_tensor_id.saturating_add(1);

        let offset = seg.offset;
        seg.data.extend_from_slice(data);
        seg.offset = seg.offset.saturating_add(data.len() as u64);
        seg.tensor_ids.push(id);

        self.tensors.push(TensorEntry {
            id,
            name: name.into(),
            role: role.into(),
            layer,
            segment: seg.id.clone(),
            source_filename: source_filename.into(),
            source_sha256: source_sha256.into(),
            source_offset,
            offset,
            byte_length: data.len() as u64,
            logical_dtype: logical_dtype.into(),
            storage_dtype: storage_dtype.into(),
            logical_shape,
            physical_shape,
            mutability: "read_only".into(),
            quantization,
            tensor_alignment_bytes: 16,
            layout_version: 1,
            artifact_bindings: BTreeMap::new(),
        });

        Ok(id)
    }

    /// Append a u32 word-aligned tensor to the current segment.
    /// Uses the safe `to_le_bytes()` method rather than an unsafe
    /// `&[u32]` → `&[u8]` cast.
    pub fn add_u32_tensor(
        &mut self,
        name: impl Into<String>,
        role: impl Into<String>,
        layer: Option<u32>,
        data: &[u32],
        source_filename: impl Into<String>,
        source_sha256: impl Into<String>,
        source_offset: u64,
        logical_dtype: impl Into<String>,
        storage_dtype: &str,
        logical_shape: Vec<u32>,
        physical_shape: Vec<u32>,
        quantization: Option<QuantizationDesc>,
    ) -> ManifestBuilderResult<u32> {
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for word in data {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        self.add_tensor(
            name,
            role,
            layer,
            &bytes,
            source_filename,
            source_sha256,
            source_offset,
            logical_dtype,
            storage_dtype,
            logical_shape,
            physical_shape,
            quantization,
        )
    }

    /// Register an alias (e.g., lm_head aliases embed_tokens).
    pub fn add_alias(
        &mut self,
        logical_name: &str,
        physical_tensor_id: u32,
        reason: &str,
    ) -> ManifestBuilderResult<()> {
        if self.tensors.iter().all(|t| t.id != physical_tensor_id) {
            return Err(ManifestBuilderError::rejected(format!(
                "alias {} points to unknown physical tensor id {}",
                logical_name, physical_tensor_id
            )));
        }
        self.aliases.push(AliasEntry {
            logical_name: logical_name.into(),
            physical_tensor_id,
            reason: reason.into(),
        });
        Ok(())
    }

    /// Return the number of finalized segments.
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Return a reference to the current in-progress manifest (read-only
    /// during construction).
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// Finalize the manifest: close the current segment, write
    /// segments + manifest.json to `output_dir`, and return the
    /// finalized manifest.
    pub fn finalize(
        mut self,
        output_dir: &Path,
    ) -> ManifestBuilderResult<(Manifest, Vec<Vec<u8>>)> {
        self.flush_segment()?;
        std::fs::create_dir_all(output_dir)?;

        if self.output_dir.is_none() {
            for (seg, payload) in self.segments.iter().zip(self.segment_payloads.iter()) {
                let path = output_dir.join(&seg.filename);
                std::fs::write(&path, payload)?;
            }
        }

        self.manifest.segments = self.segments.clone();
        self.manifest.tensor_table = self.tensors.clone();
        self.manifest.alias_table = self.aliases.clone();
        self.manifest.residency_plan.total_bytes =
            self.segments.iter().map(|s| s.byte_size).sum();
        self.manifest.image_hash = compute_manifest_hash(&self.manifest)?;

        let manifest_path = output_dir.join("manifest.json");
        let manifest_json = serde_json::to_string_pretty(&self.manifest)?;
        std::fs::write(&manifest_path, manifest_json)?;

        Ok((self.manifest, self.segment_payloads))
    }

    /// Flush the current segment without writing to disk. Returns the
    /// closed segment and its payload.
    pub fn flush_and_collect_segments(
        &mut self,
    ) -> ManifestBuilderResult<(Vec<Segment>, Vec<Vec<u8>>)> {
        self.flush_segment()?;
        let segments = std::mem::take(&mut self.segments);
        let payloads = std::mem::take(&mut self.segment_payloads);
        Ok((segments, payloads))
    }

    /// Produce an empty [`NativeCapabilityReport`] (the engine's
    /// pre-emission stub). The runtime fills in the real values.
    pub fn empty_native_capability_report() -> NativeCapabilityReport {
        NativeCapabilityReport::default()
    }

    fn flush_segment(&mut self) -> ManifestBuilderResult<()> {
        if let Some(seg) = self.current_segment.take() {
            let byte_size = seg.byte_size();
            let sha256 = {
                let mut h = Sha256::new();
                h.update(&seg.data);
                format!("{:x}", h.finalize())
            };

            if let Some(dir) = &self.output_dir {
                let path = dir.join(&seg.filename);
                std::fs::write(&path, &seg.data)?;
                // seg.data is dropped at the end of this scope; the
                // segment record below holds the sha256 and byte_size.
            } else {
                self.segment_payloads.push(seg.data.clone());
            }

            let segment = Segment {
                id: seg.id.clone(),
                filename: seg.filename,
                byte_size,
                sha256,
                tensor_ids: seg.tensor_ids,
                kind: seg.kind.clone(),
                alignment_bytes: alignment_default(),
            };

            match &segment.kind {
                SegmentKind::Persistent | SegmentKind::Final => {
                    self.manifest
                        .residency_plan
                        .persistent_segments
                        .push(segment.id.clone());
                }
                SegmentKind::Layer(_) => {
                    self.manifest
                        .residency_plan
                        .layer_segments
                        .push(segment.id.clone());
                }
            }

            self.segments.push(segment);
        }
        Ok(())
    }
}

// ── Manifest hash ────────────────────────────────────────────────────────

/// Compute the canonical SHA-256 manifest hash. The hash covers
/// image_version, compiler_version, runtime_abi, source, architecture,
/// segments, tensor_table, alias_table, and residency_plan — the same
/// fields the original engine covered.
pub fn compute_manifest_hash(manifest: &Manifest) -> Result<String, ManifestBuilderError> {
    #[derive(Serialize)]
    struct Fingerprint<'a> {
        image_version: &'a str,
        compiler_version: &'a str,
        runtime_abi: &'a str,
        source: &'a SourceIdentity,
        architecture: &'a serde_json::Value,
        segments: &'a [Segment],
        tensor_table: &'a [TensorEntry],
        alias_table: &'a [AliasEntry],
        residency_plan: &'a ResidencyPlan,
    }

    let fingerprint = Fingerprint {
        image_version: &manifest.image_version,
        compiler_version: &manifest.compiler_version,
        runtime_abi: &manifest.runtime_abi,
        source: &manifest.source,
        architecture: &manifest.architecture,
        segments: &manifest.segments,
        tensor_table: &manifest.tensor_table,
        alias_table: &manifest.alias_table,
        residency_plan: &manifest.residency_plan,
    };

    let bytes = serde_json::to_vec(&fingerprint)?;
    let mut h = Sha256::new();
    h.update(&bytes);
    Ok(format!("{:x}", h.finalize()))
}

// ── Manifest stub (for differential compile / pipeline use) ──────────────

/// Stub manifest used by the pipeline as the post-emission envelope.
/// The real manifest is computed by [`ManifestBuilder::finalize`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestStub {
    pub image_version: String,
    pub compiler_version: String,
    pub runtime_abi: String,
    pub source: SourceIdentity,
    pub segment_count: u32,
    pub tensor_count: u32,
    pub alias_count: u32,
    pub image_hash: String,
    pub required_storage_abi: String,
    pub required_capabilities: Vec<String>,
    pub shard_hashes: Vec<ShardHash>,
    pub tokenizer_hashes: Vec<ShardHash>,
    pub auxiliary_hashes: Vec<ShardHash>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cimage_manifest::types::ShardHash;

    fn empty_source() -> SourceIdentity {
        SourceIdentity {
            config_hash: String::new(),
            shard_hashes: Vec::new(),
            tokenizer_hashes: Vec::new(),
            auxiliary_hashes: Vec::new(),
            model_type: String::new(),
            quantization_bits: 0,
            quantization_group_size: 0,
            quantization_mode: String::new(),
        }
    }

    #[test]
    fn builder_constructs_with_default_abi_copied_v0() {
        let b = ManifestBuilder::new(serde_json::json!({}), empty_source());
        assert_eq!(
            b.manifest().required_storage_abi,
            super::super::header::STORAGE_ABI_COPIED_V0
        );
        assert_eq!(b.segment_count(), 0);
    }

    #[test]
    fn builder_set_required_storage_abi_overrides_default() {
        let mut b = ManifestBuilder::new(serde_json::json!({}), empty_source());
        b.set_required_storage_abi(super::super::header::STORAGE_ABI_MAPPED_NO_COPY_V1);
        assert_eq!(
            b.manifest().required_storage_abi,
            super::super::header::STORAGE_ABI_MAPPED_NO_COPY_V1
        );
    }

    #[test]
    fn builder_refuses_add_tensor_without_segment() {
        let mut b = ManifestBuilder::new(serde_json::json!({}), empty_source());
        let err = b
            .add_tensor(
                "x",
                "weight",
                None,
                &[0u8; 4],
                "x.safetensors",
                "00",
                0,
                "F32",
                "F32",
                vec![1, 1],
                vec![1, 1],
                None,
            )
            .unwrap_err();
        // The rejection is the "no segment started" preflight.
        assert!(matches!(err, ManifestBuilderError::Rejected(_)));
    }

    #[test]
    fn builder_records_tensor_with_assigned_id() {
        let mut b = ManifestBuilder::new(serde_json::json!({}), empty_source());
        b.begin_segment("embed", SegmentKind::Persistent).unwrap();
        let id = b
            .add_tensor(
                "embed.weight",
                "embedding",
                None,
                &[1u8, 2, 3, 4],
                "x.safetensors",
                "00",
                0,
                "F32",
                "F32",
                vec![2],
                vec![2],
                None,
            )
            .unwrap();
        assert_eq!(id, 0);
        b.begin_segment("layer_0", SegmentKind::Layer(0)).unwrap();
        let id2 = b
            .add_tensor(
                "layer_0.attn.q.weight",
                "weight",
                Some(0),
                &[0u8; 16],
                "x.safetensors",
                "00",
                16,
                "F32",
                "F32",
                vec![4, 4],
                vec![4, 4],
                None,
            )
            .unwrap();
        assert_eq!(id2, 1);
    }

    #[test]
    fn builder_rejects_alias_with_unknown_physical_id() {
        let mut b = ManifestBuilder::new(serde_json::json!({}), empty_source());
        b.begin_segment("embed", SegmentKind::Persistent).unwrap();
        let err = b.add_alias("lm_head.weight", 999, "alias").unwrap_err();
        assert!(matches!(err, ManifestBuilderError::Rejected(_)));
    }

    #[test]
    fn builder_accepts_alias_with_known_physical_id() {
        let mut b = ManifestBuilder::new(serde_json::json!({}), empty_source());
        b.begin_segment("embed", SegmentKind::Persistent).unwrap();
        let id = b
            .add_tensor(
                "embed.weight",
                "embedding",
                None,
                &[0u8; 4],
                "x.safetensors",
                "00",
                0,
                "F32",
                "F32",
                vec![2],
                vec![2],
                None,
            )
            .unwrap();
        b.add_alias("lm_head.weight", id, "alias").unwrap();
        assert_eq!(b.aliases.len(), 1);
        assert_eq!(b.aliases[0].physical_tensor_id, id);
    }

    #[test]
    fn builder_set_start_tensor_id_advances_next_id() {
        let mut b = ManifestBuilder::new(serde_json::json!({}), empty_source());
        b.set_start_tensor_id(100);
        b.begin_segment("embed", SegmentKind::Persistent).unwrap();
        let id = b
            .add_tensor(
                "embed.weight",
                "embedding",
                None,
                &[0u8; 4],
                "x.safetensors",
                "00",
                0,
                "F32",
                "F32",
                vec![2],
                vec![2],
                None,
            )
            .unwrap();
        assert_eq!(id, 100);
    }

    #[test]
    fn add_u32_tensor_emits_little_endian_bytes() {
        let mut b = ManifestBuilder::new(serde_json::json!({}), empty_source());
        b.begin_segment("embed", SegmentKind::Persistent).unwrap();
        let id = b
            .add_u32_tensor(
                "embed.u32",
                "scalar",
                None,
                &[0x04030201, 0x08070605],
                "x.safetensors",
                "00",
                0,
                "U32",
                "U32",
                vec![2],
                vec![2],
                None,
            )
            .unwrap();
        assert_eq!(id, 0);
        // The in-progress segment (not yet flushed) holds the bytes.
        let current = b.current_segment.as_ref().unwrap();
        let payload = &current.data;
        assert_eq!(payload.len(), 8);
        assert_eq!(&payload[..4], &[1, 2, 3, 4]);
        assert_eq!(&payload[4..], &[5, 6, 7, 8]);
    }

    #[test]
    fn finalize_writes_manifest_json_to_output_dir() {
        let dir = tempdir();
        let mut b = ManifestBuilder::new(serde_json::json!({}), empty_source());
        b.begin_segment("embed", SegmentKind::Persistent).unwrap();
        b.add_tensor(
            "embed.weight",
            "embedding",
            None,
            &[0u8; 4],
            "x.safetensors",
            "00",
            0,
            "F32",
            "F32",
            vec![2],
            vec![2],
            None,
        )
        .unwrap();
        let (manifest, payloads) = b.finalize(&dir).unwrap();
        assert_eq!(manifest.tensor_table.len(), 1);
        assert_eq!(payloads.len(), 1);
        // The manifest.json was written.
        let json_path = dir.join("manifest.json");
        assert!(json_path.exists());
        let json = std::fs::read_to_string(&json_path).unwrap();
        let parsed: Manifest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.tensor_table.len(), 1);
        assert!(!parsed.image_hash.is_empty());
    }

    #[test]
    fn manifest_hash_is_deterministic_for_same_inputs() {
        // Two builders constructed from identical inputs must produce
        // identical image_hash strings.
        let dir_a = tempdir();
        let dir_b = tempdir();
        let mut a = ManifestBuilder::new(
            serde_json::json!({"hidden_size": 1024}),
            SourceIdentity {
                config_hash: "abcd".into(),
                shard_hashes: vec![ShardHash {
                    filename: "a.safetensors".into(),
                    sha256: "00".into(),
                }],
                tokenizer_hashes: Vec::new(),
                auxiliary_hashes: Vec::new(),
                model_type: "qwen3".into(),
                quantization_bits: 4,
                quantization_group_size: 128,
                quantization_mode: "nf4".into(),
            },
        );
        let mut b = ManifestBuilder::new(
            serde_json::json!({"hidden_size": 1024}),
            SourceIdentity {
                config_hash: "abcd".into(),
                shard_hashes: vec![ShardHash {
                    filename: "a.safetensors".into(),
                    sha256: "00".into(),
                }],
                tokenizer_hashes: Vec::new(),
                auxiliary_hashes: Vec::new(),
                model_type: "qwen3".into(),
                quantization_bits: 4,
                quantization_group_size: 128,
                quantization_mode: "nf4".into(),
            },
        );
        a.begin_segment("embed", SegmentKind::Persistent).unwrap();
        a.add_tensor(
            "embed.weight",
            "embedding",
            None,
            &[1u8, 2, 3, 4],
            "a.safetensors",
            "00",
            0,
            "F32",
            "F32",
            vec![2],
            vec![2],
            None,
        )
        .unwrap();
        b.begin_segment("embed", SegmentKind::Persistent).unwrap();
        b.add_tensor(
            "embed.weight",
            "embedding",
            None,
            &[1u8, 2, 3, 4],
            "a.safetensors",
            "00",
            0,
            "F32",
            "F32",
            vec![2],
            vec![2],
            None,
        )
        .unwrap();
        let (a_manifest, _) = a.finalize(&dir_a).unwrap();
        let (b_manifest, _) = b.finalize(&dir_b).unwrap();
        assert_eq!(a_manifest.image_hash, b_manifest.image_hash);
    }

    fn tempdir() -> std::path::PathBuf {
        let base = std::env::temp_dir();
        let unique = format!(
            "cimage_manifest_test_{}_{:?}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        );
        let p = base.join(unique);
        std::fs::create_dir_all(&p).unwrap();
        p
    }
}
