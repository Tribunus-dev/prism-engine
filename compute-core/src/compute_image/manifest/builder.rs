//! ComputeImage builder — ImageBuilder and SegmentBuilder.
//!
//! Responsible for constructing manifests, assembling segment payloads,
//! and writing image artifacts to disk.

use super::types::{
    default_alignment_bytes, default_layout_version, default_tensor_alignment_bytes,
    compute_manifest_hash, AliasEntry, Manifest, MetalKernelArtifact, QuantizationDesc,
    ResidencyPlan, Segment, SegmentKind, SourceIdentity, TensorEntry,
};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::Path;

// ── Builder ────────────────────────────────────────────────────────────────

pub struct ImageBuilder {
    manifest: Manifest,
    next_tensor_id: u32,
    current_segment: Option<SegmentBuilder>,
    pub(crate) segments: Vec<Segment>,
    tensors: Vec<TensorEntry>,
    aliases: Vec<AliasEntry>,
    /// When set, flushed segments are written directly to this directory
    /// instead of accumulating in `segment_payloads`. The `Vec<u8>` data is
    /// dropped immediately after the file write, reducing peak memory.
    output_dir: Option<std::path::PathBuf>,
    /// Accumulated segment payloads (memory-backed segments).
    pub(crate) segment_payloads: Vec<Vec<u8>>,
}

struct SegmentBuilder {
    id: String,
    filename: String,
    kind: SegmentKind,
    data: Vec<u8>,
    tensor_ids: Vec<u32>,
    offset: u64,
}

impl ImageBuilder {
    pub fn new(arch: crate::config::TextArchitecture, source: SourceIdentity) -> Self {
        Self {
            manifest: Manifest {
                image_version: "0.1.0".into(),
                compiler_version: env!("CARGO_PKG_VERSION").into(),
                runtime_abi: format!(
                    "mlx-rs/0.21.0 core/{} safetensors/0.5.3",
                    env!("CARGO_PKG_VERSION")
                ),
                hardware_target: None,
                compile_date: String::new(),
                compile_host: String::new(),
                source,
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
                required_storage_abi: "copied-v0".to_string(),
                required_capabilities: Vec::new(),
                prepacked_layout: "none".into(),
                metallib_hash: None,
                metallib_size: None,
                metal_kernel_artifacts: Vec::new(),
                execution_plan: crate::config::ModelExecutionPlan::default(),
                readiness: None,
                phase_dag: None,
                compatibility_receipt: None,
            },
            next_tensor_id: 0,
            current_segment: None,
            segments: Vec::new(),
            tensors: Vec::new(),
            aliases: Vec::new(),
            output_dir: None,
            segment_payloads: Vec::new(),
        }
    }

    /// Set the starting tensor ID so new IDs don't collide with existing ones
    /// from a previous compilation.  Typically called right after `new()`.
    pub fn set_start_tensor_id(&mut self, start_id: u32) {
        self.next_tensor_id = start_id;
    }

    /// Inject pre-compiled Metal kernel artifacts into the manifest.
    pub fn set_metal_kernel_artifacts(&mut self, artifacts: Vec<MetalKernelArtifact>) {
        self.manifest.metal_kernel_artifacts = artifacts;
    }

    /// Enable file-backed segment writing. When set, each flushed segment
    /// is written directly to `dir` and its `Vec<u8>` payload is dropped,
    /// instead of accumulating in `segment_payloads`. This reduces the
    /// compiler's peak anonymous memory from ~2× output size to just the
    /// active segment buffer.
    /// Must be called before `begin_segment`. Does not flush the current
    /// segment if one is open.
    pub fn set_output_dir(&mut self, dir: &Path) {
        self.output_dir = Some(dir.to_path_buf());
    }

    /// Start a new segment. Closes the previous segment if any.
    pub fn begin_segment(&mut self, id: &str, kind: SegmentKind) {
        self.flush_segment();
        let filename = format!("segment_{:03}.bin", self.segments.len());
        self.current_segment = Some(SegmentBuilder {
            id: id.into(),
            filename,
            kind,
            data: Vec::new(),
            tensor_ids: Vec::new(),
            offset: 0,
        });
    }

    /// Append a tensor to the current segment. The caller provides the raw bytes.
    pub fn add_tensor(
        &mut self,
        name: String,
        role: String,
        layer: Option<u32>,
        data: &[u8],
        source_filename: String,
        source_sha256: String,
        source_offset: u64,
        logical_dtype: String,
        storage_dtype: &str,
        logical_shape: Vec<u32>,
        physical_shape: Vec<u32>,
        quantization: Option<QuantizationDesc>,
    ) -> u32 {
        let seg = self.current_segment.as_mut().expect("no segment started");
        let id = self.next_tensor_id;
        self.next_tensor_id += 1;

        let offset = seg.offset;
        seg.data.extend_from_slice(data);
        seg.offset += data.len() as u64;
        seg.tensor_ids.push(id);

        self.tensors.push(TensorEntry {
            id,
            name,
            role,
            layer,
            segment: seg.id.clone(),
            source_filename,
            source_sha256,
            source_offset,
            offset,
            byte_length: data.len() as u64,
            logical_dtype,
            storage_dtype: storage_dtype.into(),
            logical_shape,
            physical_shape,
            mutability: "read_only".into(),
            quantization,
            tensor_alignment_bytes: default_tensor_alignment_bytes(),
            layout_version: default_layout_version(),
            artifact_bindings: HashMap::new(),
        });

        id
    }

    /// Append a u32 word-aligned tensor to the current segment.
    /// Casts `&[u32]` to `&[u8]` in-place, avoiding per-word serialization.
    pub fn add_u32_tensor(
        &mut self,
        name: String,
        role: String,
        layer: Option<u32>,
        data: &[u32],
        source_filename: String,
        source_sha256: String,
        source_offset: u64,
        logical_dtype: String,
        storage_dtype: &str,
        logical_shape: Vec<u32>,
        physical_shape: Vec<u32>,
        quantization: Option<QuantizationDesc>,
    ) -> u32 {
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
        };
        self.add_tensor(
            name, role, layer, bytes,
            source_filename, source_sha256, source_offset,
            logical_dtype, storage_dtype, logical_shape, physical_shape, quantization,
        )
    }

    /// Register an alias (e.g., lm_head aliases embed_tokens).
    pub fn add_alias(&mut self, logical_name: &str, physical_tensor_id: u32, reason: &str) {
        self.aliases.push(AliasEntry {
            logical_name: logical_name.into(),
            physical_tensor_id,
            reason: reason.into(),
        });
    }

    /// Finalize and return the complete manifest.
    /// Set the compiler-emitted phase DAG.
    pub fn set_phase_graph(&mut self, dag: crate::compute_image::phase_dag::EmittedPhaseGraph) {
        self.manifest.phase_dag = Some(dag);
    }

    /// Return the number of segments.
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    pub fn finalize(mut self, output_dir: &Path) -> crate::Result<Manifest> {
        self.flush_segment();
        std::fs::create_dir_all(output_dir)
            .map_err(|e| crate::Error::from_reason(format!("mkdir: {}", e)))?;

        // Write segments to disk
        // When file-backed mode was used (output_dir set on ImageBuilder),
        // segments were already written by flush_segment — skip the loop.
        if self.output_dir.is_none() {
            for (seg, payload) in self.segments.iter().zip(self.segment_payloads.iter()) {
                let path = output_dir.join(&seg.filename);
                std::fs::write(&path, payload).map_err(|e| {
                    crate::Error::from_reason(format!("write segment {}: {}", seg.filename, e))
                })?;
            }
        }

        self.manifest.segments = self.segments;
        self.manifest.tensor_table = self.tensors;
        self.manifest.alias_table = self.aliases;
        self.manifest.compile_date = crate::now_iso8601();
        self.manifest.compile_host = crate::hostname_or_default();
        self.manifest.residency_plan.total_bytes =
            self.manifest.segments.iter().map(|s| s.byte_size).sum();
        self.manifest.image_hash = compute_manifest_hash(&self.manifest);

        // Write manifest
        let manifest_path = output_dir.join("manifest.json");
        let manifest_json = serde_json::to_string_pretty(&self.manifest)
            .map_err(|e| crate::Error::from_reason(format!("json: {}", e)))?;
        std::fs::write(&manifest_path, manifest_json)
            .map_err(|e| crate::Error::from_reason(format!("write manifest: {}", e)))?;

        Ok(self.manifest)
    }

    /// Flush the current segment and return everything needed to write new
    /// segment files + construct the manifest *without* writing to disk.
    /// Used by the differential compile path.
    pub fn flush_and_collect_segments(&mut self) -> (Vec<Segment>, Vec<Vec<u8>>, &Manifest) {
        self.flush_segment();
        let segments = std::mem::take(&mut self.segments);
        let payloads = std::mem::take(&mut self.segment_payloads);
        (segments, payloads, &self.manifest)
    }

    fn flush_segment(&mut self) {
        if let Some(seg) = self.current_segment.take() {
            let byte_size = seg.data.len() as u64;
            let sha256 = {
                let mut h = Sha256::new();
                h.update(&seg.data);
                format!("{:x}", h.finalize())
            };

            // File-backed or memory-backed segment storage
            if let Some(dir) = &self.output_dir {
                let path = dir.join(&seg.filename);
                std::fs::write(&path, &seg.data).expect("failed to write segment to disk");
                drop(seg.data);
            } else {
                self.segment_payloads.push(seg.data);
            }

            self.segments.push(Segment {
                id: seg.id,
                filename: seg.filename,
                byte_size,
                sha256,
                tensor_ids: seg.tensor_ids,
                kind: seg.kind,
                alignment_bytes: default_alignment_bytes(),
            });

            // Build residency plan
            let last_seg = self.segments.last().unwrap();
            match &last_seg.kind {
                SegmentKind::Persistent | SegmentKind::Final => {
                    self.manifest
                        .residency_plan
                        .persistent_segments
                        .push(last_seg.id.clone());
                }
                SegmentKind::Layer(_) => {
                    self.manifest
                        .residency_plan
                        .layer_segments
                        .push(last_seg.id.clone());
                }
            }
        }
    }

    /// Set the execution plan on the manifest. Must be called before finalize().
    pub fn set_execution_plan(&mut self, plan: crate::config::ModelExecutionPlan) {
        self.manifest.execution_plan = plan;
    }

    /// Set the audio encoder configuration on the manifest.
    pub fn set_audio_config(&mut self, audio_config: crate::config::AudioArchitecture) {
        self.manifest.audio_config = Some(audio_config);
    }

    /// Record a precompiled Metal library bundle in the manifest.
    ///
    /// `sha256` is the hex-encoded SHA-256 of the `.metallib` file; `byte_size`
    /// is its length in bytes.  The metallib file itself is expected to already
    /// have been placed in the output directory alongside the segment files.
    pub fn set_metallib(&mut self, sha256: String, byte_size: u64) {
        self.manifest.metallib_hash = Some(sha256);
        self.manifest.metallib_size = Some(byte_size);
    }

    /// Post-process: apply prepack-int8-v1 layout transform to all quantized
    /// weight tensors that have companion scale/bias tensors in the same segment.
    ///
    /// Walks the tensor table looking for weight tensors (naming convention:
    /// `*.weight`) that have corresponding `*.scales` and `*.biases` tensors in
    /// the same segment. For each triplet found, transposes [K,N] to [N,K],
    /// reorders by group, and interleaves scales/biases into one packed buffer.
    ///
    /// Updates tensor metadata and sets manifest.prepacked_layout.
    /// Must be called before finalize().
    pub fn prepack_quantized_weights(&mut self) -> crate::Result<()> {
        use crate::layout_transform;

        // Identify weight/scale/bias triplets.
        // A weight tensor named "X.weight" with dtype U8 is prepacked if
        // "X.scales" (F32) and "X.biases" (F32) exist in the same segment.
        let n_tensors = self.tensors.len();
        let mut prepack_count = 0u64;
        let mut prepack_bytes_before = 0u64;
        let mut prepack_bytes_after = 0u64;

        for i in 0..n_tensors {
            let t = &self.tensors[i];
            if !t.name.ends_with(".weight") || t.storage_dtype != "U8" {
                continue;
            }
            let base = &t.name[..t.name.len() - ".weight".len()];
            let scale_name = format!("{}.scales", base);
            let bias_name = format!("{}.biases", base);

            // Find companion tensors in the same segment
            let scale_idx = self
                .tensors
                .iter()
                .position(|e| e.name == scale_name && e.segment == t.segment);
            let bias_idx = self
                .tensors
                .iter()
                .position(|e| e.name == bias_name && e.segment == t.segment);
            let (si, bi) = match (scale_idx, bias_idx) {
                (Some(s), Some(b)) => (s, b),
                _ => continue,
            };

            // Determine dimensions from logical shape.
            // Weight shape is [K, N] (in_features, out_features).
            if t.logical_shape.len() != 2 {
                continue; // skip non-matrix weights (e.g., norms)
            }
            let k = t.logical_shape[0] as usize;

            // Determine group_size from quantization descriptor or default.
            let group_size = t
                .quantization
                .as_ref()
                .map(|q| q.group_size as usize)
                .unwrap_or(64);

            if k % group_size != 0 {
                continue; // must be divisible
            }

            // Mark these tensors. We'll rebuild the segment data after
            // collecting all triplets.
            prepack_count += 1;
            prepack_bytes_before +=
                t.byte_length + self.tensors[si].byte_length + self.tensors[bi].byte_length;
        }

        if prepack_count == 0 {
            return Ok(());
        }

        // Rebuild segment payloads with prepacked weights.
        // For each segment, we walk its tensor_ids in order, writing either
        // the original bytes or the prepacked bytes.
        let n_segments = self.segments.len();
        for seg_idx in 0..n_segments {
            let seg = &self.segments[seg_idx];
            let payload = &self.segment_payloads[seg_idx];
            let mut new_payload = Vec::with_capacity(payload.len());

            for &tid in &seg.tensor_ids {
                let ti = self
                    .tensors
                    .iter()
                    .position(|t| t.id == tid)
                    .expect("tensor_id in segment tensor_ids not found");
                let t = &self.tensors[ti];

                // Check if this tensor is part of a prepack triplet
                let is_prepacked = t.name.ends_with(".weight") && t.storage_dtype == "U8";
                if is_prepacked {
                    let base = &t.name[..t.name.len() - ".weight".len()];
                    let scale_name = format!("{}.scales", base);
                    let bias_name = format!("{}.biases", base);
                    let si = self
                        .tensors
                        .iter()
                        .position(|e| e.name == scale_name && e.segment == t.segment);
                    let bi = self
                        .tensors
                        .iter()
                        .position(|e| e.name == bias_name && e.segment == t.segment);

                    if let (Some(si), Some(bi)) = (si, bi) {
                        let k = t.logical_shape[0] as usize;
                        let n = t.logical_shape[1] as usize;
                        let group_size = t
                            .quantization
                            .as_ref()
                            .map(|q| q.group_size as usize)
                            .unwrap_or(64);

                        if k % group_size == 0 {
                            // Extract weight, scale, bias bytes from payload
                            let w_start = t.offset as usize;
                            let w_len = t.byte_length as usize;
                            let s_start = self.tensors[si].offset as usize;
                            let s_len = self.tensors[si].byte_length as usize;
                            let b_start = self.tensors[bi].offset as usize;
                            let b_len = self.tensors[bi].byte_length as usize;

                            let weight_bytes = &payload[w_start..w_start + w_len];
                            let scale_bytes = &payload[s_start..s_start + s_len];
                            let bias_bytes = &payload[b_start..b_start + b_len];

                            // Convert f32 slices
                            let scales: Vec<f32> = unsafe {
                                std::slice::from_raw_parts(
                                    scale_bytes.as_ptr() as *const f32,
                                    s_len / 4,
                                )
                            }
                            .to_vec();
                            let biases: Vec<f32> = unsafe {
                                std::slice::from_raw_parts(
                                    bias_bytes.as_ptr() as *const f32,
                                    b_len / 4,
                                )
                            }
                            .to_vec();

                            // Apply prepack
                            let (packed, _meta) = layout_transform::prepack_pipeline(
                                weight_bytes,
                                &scales,
                                &biases,
                                k,
                                n,
                                group_size,
                            );

                            // Write prepacked weight to new payload
                            let old_offset = new_payload.len();
                            new_payload.extend_from_slice(&packed);

                            // Update tensor metadata
                            let t_mut = &mut self.tensors[ti];
                            t_mut.offset = old_offset as u64;
                            t_mut.byte_length = packed.len() as u64;
                            t_mut.physical_shape = vec![
                                n as u32 * (k as u32 / group_size as u32) * (group_size as u32 + 2),
                            ];
                            t_mut.storage_dtype = "U8".into();
                            t_mut.layout_version = 2;

                            // Mark scale and bias as absorbed (zero-length)
                            self.tensors[si].byte_length = 0;
                            self.tensors[si].offset = old_offset as u64;
                            self.tensors[bi].byte_length = 0;
                            self.tensors[bi].offset = old_offset as u64;

                            prepack_bytes_after += packed.len() as u64;

                            continue; // skip original weight/scale/bias from new payload
                        }
                    }
                }

                // Skip zero-length tensors (absorbed scale/bias)
                if t.byte_length == 0 {
                    continue;
                }

                // Copy original tensor bytes unchanged
                let old_offset = new_payload.len();
                let start = t.offset as usize;
                let len = t.byte_length as usize;
                new_payload.extend_from_slice(&payload[start..start + len]);
                // Update offset if it changed (subsequent tensors shift)
                if old_offset != t.offset as usize {
                    let t_mut = &mut self.tensors[ti];
                    t_mut.offset = old_offset as u64;
                }
            }

            // Update segment byte size
            self.segments[seg_idx].byte_size = new_payload.len() as u64;
            self.segment_payloads[seg_idx] = new_payload;
        }

        self.manifest.prepacked_layout = "prepacked-int8-v1".into();
        let mb = |b: u64| format!("{:.1}MB", b as f64 / 1_048_576.0);
        eprintln!(
            "[compiler-prepack] tensors={} bytes_before={} bytes_after={}",
            prepack_count,
            mb(prepack_bytes_before),
            mb(prepack_bytes_after),
        );

        Ok(())
    }
}
