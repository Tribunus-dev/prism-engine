//! CImage writer — compile-time emitter for the `.cimage` format.
//!
//! This module owns the canonical authority for *writing* a `.cimage` file:
//! header reservation, page-aligned payload append, kernel / XDNA / ANE
//! embedding, and the high-level compilation-metadata wrapper used by the
//! compiler pipeline. The matching read authority lives in [`super::reader`].
//!
//! The cimage crate surface is split by authority along the read/write axis:
//! - [`super::reader`] owns the read path.
//! - This module owns the write path (both the low-level `CImageWriter` and
//!   the high-level `UniversalCImageWriter`).
//! - [`super`] (the parent module) keeps the data definitions that both sides
//!   consume (`TensorType`, `CImageHeader`, `TensorRecord`, descriptors),
//!   the standalone promotion helpers
//!   (`emit_int8_ane_program`, `promote_cimage_after_replay`,
//!   `promote_cimage_with_behavioral_evidence`), and the small data
//!   envelopes (`CImageError`, `CImageManifest`, `TensorPayloadEntry`).
//!
//! Further write-side extraction (e.g. a separate `CImageWriter::append_*`
//! per format family, or a `kernel_artifact` module) is possible future
//! work; this module is the single, typed entry point for `.cimage` writes
//! for now.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

use prism_ecs_quantization::ternarization::promotion::NativeTernaryPromotionEvidence;

use super::{
    AneProgramRecord, CImageHeader, HEADER_PAGES, KernelRecord, MAGIC, MoeTensorDescriptor,
    PAGE_SIZE, TensorPayloadEntry, TensorRecord, TensorType, TernaryDescriptor,
    UOpCaptureRecord, UOpWorkloadEvidence, VisionTensorDescriptor, XdnaArtifactRecord,
};

/// Classify a tensor name into a semantic family for downstream admission.
///
/// The classification is stable across the format boundary: the writer
/// records the family in `TensorRecord::semantic_family` and the runtime
/// loader reads it back to make admission decisions about expert routing,
/// vision components, etc. This helper is the single source of truth for
/// the name → family mapping.
fn semantic_family(name: &str) -> (&'static str, bool) {
    let lower = name.to_ascii_lowercase();
    if lower.contains("visual") || lower.contains("vision") || lower.contains("patch") {
        ("vision", false)
    } else if lower.contains("router") || lower.contains("gate") {
        ("router", true)
    } else if lower.contains("expert") {
        ("routed_expert", false)
    } else if lower.contains("embed") || lower.contains("lm_head") {
        ("embedding_or_head", false)
    } else if lower.contains("norm") {
        ("normalization", false)
    } else if lower.contains("attn") || lower.contains("attention") {
        ("attention", false)
    } else {
        ("shared", false)
    }
}

// ── CImageWriter (low-level) ────────────────────────────────────────────

/// Streaming writer for .cimage files.
///
/// Each `append_*` call:
/// 1. Pads the file to the next 16 KB boundary.
/// 2. Writes the payload at that aligned offset.
/// 3. Records the offset + size in the header.
///
/// `finalize()` seeks back to offset 0 and writes the header.
pub struct CImageWriter {
    file: File,
    header: CImageHeader,
}

impl CImageWriter {
    /// Create a new .cimage file at `path`.
    ///
    /// Reserves the first 128 KB for the header (configurable via HEADER_PAGES).
    pub fn new(path: &Path) -> Result<Self, String> {
        let mut file = File::create(path).map_err(|e| format!("create .cimage: {e}"))?;
        // Reserve first HEADER_PAGES × PAGE_SIZE for the header
        let header_bytes = (HEADER_PAGES * PAGE_SIZE) as usize;
        let zeros = vec![0u8; header_bytes];
        file.write_all(&zeros)
            .map_err(|e| format!("reserve header: {e}"))?;
        // Seek to end of header block so append starts at the next page
        file.seek(SeekFrom::Start(header_bytes as u64))
            .map_err(|e| format!("seek: {e}"))?;
        Ok(CImageWriter {
            file,
            header: CImageHeader::default(),
        })
    }

    /// Write a palettized split-block payload.
    ///
    /// Payload layout expected:
    ///   [codebook_block: dim_m × 16 × 2 bytes]
    ///   [indices_block:  dim_m × dim_n/2 bytes]
    pub fn append_palettized(
        &mut self,
        name: &str,
        payload: &[u8],
        dim_m: u32,
        dim_n: u32,
    ) -> Result<(), String> {
        self.append(name, payload, dim_m, dim_n, TensorType::Palettized4Bit)
    }

    /// Write a tensor payload with the given tensor type.
    ///
    /// The payload format is specific to the tensor type and must be
    /// understood by the runtime loader.
    pub fn append(
        &mut self,
        name: &str,
        payload: &[u8],
        dim_m: u32,
        dim_n: u32,
        tensor_type: TensorType,
    ) -> Result<(), String> {
        self.align_to_page()?;
        let offset = self.current_pos()?;
        self.file
            .write_all(payload)
            .map_err(|e| format!("write payload: {e}"))?;
        let (family, router_sensitive) = semantic_family(name);
        self.header.tensors.insert(
            name.to_string(),
            TensorRecord {
                tensor_type,
                offset,
                size: payload.len() as u64,
                dim_m,
                dim_n,
                scale_offset: None,
                scale_size: None,
                ternary: None,
                moe: None,
                vision: None,
                semantic_family: Some(family.into()),
                router_sensitive,
            },
        );
        Ok(())
    }

    /// Append an authoritative packed ternary payload. The caller must pass
    /// the exact descriptor used by the backend kernel; no FP16 expansion is
    /// performed here.
    pub fn append_native_ternary(
        &mut self,
        name: &str,
        payload: &[u8],
        dim_m: u32,
        dim_n: u32,
        tensor_type: TensorType,
        descriptor: TernaryDescriptor,
    ) -> Result<(), String> {
        if !matches!(
            tensor_type,
            TensorType::Ternary158 | TensorType::TernaryTile640
        ) {
            return Err("native ternary payload requires a ternary tensor type".into());
        }
        descriptor.validate()?;
        self.append(name, payload, dim_m, dim_n, tensor_type)?;
        self.header
            .tensors
            .get_mut(name)
            .expect("appended tensor must exist")
            .ternary = Some(descriptor);
        Ok(())
    }

    /// Append packed ternary codes and their authoritative per-group scales.
    /// The scale payload is kept as an internal record so replay can resolve
    /// it through offsets without reconstructing or expanding the weights.
    pub fn append_native_ternary_with_scales(
        &mut self,
        name: &str,
        payload: &[u8],
        scales: &[u8],
        dim_m: u32,
        dim_n: u32,
        tensor_type: TensorType,
        descriptor: TernaryDescriptor,
    ) -> Result<(), String> {
        self.append_native_ternary(name, payload, dim_m, dim_n, tensor_type, descriptor)?;
        let scale_name = format!("{name}.__scales");
        self.append(&scale_name, scales, 0, 0, TensorType::Blob)?;
        let scale_record = self
            .header
            .tensors
            .remove(&scale_name)
            .ok_or_else(|| "scale payload was not recorded".to_string())?;
        let record = self
            .header
            .tensors
            .get_mut(name)
            .ok_or_else(|| format!("native ternary tensor '{name}' was not recorded"))?;
        record.scale_offset = Some(scale_record.offset);
        record.scale_size = Some(scale_record.size);
        Ok(())
    }

    /// Write a standard FP16 tensor payload.
    pub fn append_fp16(
        &mut self,
        name: &str,
        payload: &[u8],
        dim_m: u32,
        dim_n: u32,
    ) -> Result<(), String> {
        self.align_to_page()?;
        let offset = self.current_pos()?;
        self.file
            .write_all(payload)
            .map_err(|e| format!("write payload: {e}"))?;
        let (family, router_sensitive) = semantic_family(name);
        self.header.tensors.insert(
            name.to_string(),
            TensorRecord {
                tensor_type: TensorType::StandardFP16,
                offset,
                size: payload.len() as u64,
                dim_m,
                dim_n,
                scale_offset: None,
                scale_size: None,
                ternary: None,
                moe: None,
                vision: None,
                semantic_family: Some(family.into()),
                router_sensitive,
            },
        );
        Ok(())
    }

    /// Finalize: write magic + header to the first 16 KB block.
    /// Set the execution plan JSON to embed in the CImage header.
    pub fn set_execution_plan(&mut self, plan_json: String) {
        self.header.execution_plan = Some(plan_json);
    }

    pub fn set_heterogeneous_workload_evidence(
        &mut self,
        evidence: crate::search::HeterogeneousScheduleEvidence,
    ) {
        self.header.heterogeneous_workload_evidence = Some(evidence);
    }

    /// Append a validated native XDNA artifact envelope and record its byte
    /// range in the CImage header. The payload remains opaque to the generic
    /// compiler but is owned by Prism's native XDNA codec at runtime.
    pub fn add_xdna_artifact(
        &mut self,
        name: &str,
        payload: &[u8],
        compiler_abi: impl Into<String>,
        generation: impl Into<String>,
    ) -> Result<(), String> {
        if name.is_empty() || payload.is_empty() {
            return Err("XDNA artifact name and payload must be nonempty".into());
        }
        self.align_to_page()?;
        let offset = self.current_pos()?;
        self.file
            .write_all(payload)
            .map_err(|e| format!("write XDNA artifact: {e}"))?;
        self.header.xdna_artifacts.insert(
            name.into(),
            XdnaArtifactRecord {
                offset,
                size: payload.len() as u64,
                compiler_abi: compiler_abi.into(),
                generation: generation.into(),
            },
        );
        Ok(())
    }

    pub fn set_format_plan(&mut self, plan_json: String) -> Result<(), String> {
        let _: prism_ecs_ir::evolution::compile_plan::FormatPlan =
            serde_json::from_str(&plan_json).map_err(|e| format!("invalid format plan: {e}"))?;
        self.header.format_plan = Some(plan_json);
        Ok(())
    }

    /// Attach the authoritative registry for the specialised models stored in
    /// this CImage. Validation happens before bytes are committed at finalize.
    pub fn set_model_manifest(
        &mut self,
        manifest: crate::model_manifest::MultiModelManifest,
    ) -> Result<(), String> {
        manifest.validate()?;
        self.header.model_manifest = Some(manifest);
        Ok(())
    }

    pub fn set_model_identity<T: serde::Serialize>(
        &mut self,
        family: impl Into<String>,
        config: &T,
    ) -> Result<(), String> {
        self.header.model_family = Some(family.into());
        self.header.model_config_json = Some(
            serde_json::to_string(config).map_err(|e| format!("serialize model config: {e}"))?,
        );
        Ok(())
    }

    pub fn set_model_capabilities<I, S>(&mut self, capabilities: I)
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.header.model_capabilities = capabilities.into_iter().map(Into::into).collect();
        self.header.model_capabilities.sort();
        self.header.model_capabilities.dedup();
    }

    pub fn require_model_capabilities<I, S>(&self, required: I) -> Result<(), String>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        for capability in required {
            if !self
                .header
                .model_capabilities
                .iter()
                .any(|value| value == capability.as_ref())
            {
                return Err(format!(
                    "CImage is missing verified model capability '{}'",
                    capability.as_ref()
                ));
            }
        }
        Ok(())
    }

    pub fn set_kv_compression_policy(
        &mut self,
        policy: &prism_ecs_quantization::kv_search::KvCompressionCandidate,
        max_error: f32,
    ) -> Result<(), String> {
        if !max_error.is_finite() || max_error < 0.0 {
            return Err("KV compression loss gate must be finite and nonnegative".into());
        }
        self.header.kv_compression_policy = Some(
            serde_json::to_string(&(policy, max_error))
                .map_err(|error| format!("serialize KV compression policy: {error}"))?,
        );
        Ok(())
    }

    pub fn set_qwen36_config(
        &mut self,
        config: crate::qwen3_6_moe::Qwen36Config,
    ) -> Result<(), String> {
        config.validate()?;
        self.header.qwen36_config = Some(config);
        Ok(())
    }

    pub fn set_moe_tensor(
        &mut self,
        name: &str,
        descriptor: MoeTensorDescriptor,
    ) -> Result<(), String> {
        let record = self
            .header
            .tensors
            .get_mut(name)
            .ok_or_else(|| format!("cannot annotate missing tensor '{name}'"))?;
        if descriptor.role != "router"
            && descriptor.role != "routed_expert"
            && descriptor.role != "routed_expert_bank"
            && descriptor.role != "shared_expert"
        {
            return Err(format!("invalid MoE tensor role '{}'", descriptor.role));
        }
        if descriptor.role == "routed_expert" && descriptor.expert.is_none() {
            return Err(format!(
                "routed expert tensor '{name}' is missing expert id"
            ));
        }
        if descriptor.role != "routed_expert" && descriptor.expert.is_some() {
            return Err(format!("non-routed tensor '{name}' has an expert id"));
        }
        if descriptor.role == "routed_expert_bank" && descriptor.expert_count.is_none() {
            return Err(format!(
                "routed expert bank '{name}' is missing expert count"
            ));
        }
        record.moe = Some(descriptor);
        Ok(())
    }

    pub fn set_vision_tensor(
        &mut self,
        name: &str,
        descriptor: VisionTensorDescriptor,
    ) -> Result<(), String> {
        if descriptor.component.is_empty() {
            return Err(format!("vision tensor '{name}' has an empty component"));
        }
        let record = self
            .header
            .tensors
            .get_mut(name)
            .ok_or_else(|| format!("cannot annotate missing tensor '{name}'"))?;
        record.vision = Some(descriptor);
        Ok(())
    }

    pub fn contains_native_ternary(&self) -> bool {
        self.header.tensors.values().any(|record| {
            matches!(
                record.tensor_type,
                TensorType::Ternary158 | TensorType::TernaryTile640
            )
        })
    }

    pub fn has_native_ternary_promotion(&self) -> bool {
        self.header
            .native_ternary_promotion
            .as_ref()
            .is_some_and(NativeTernaryPromotionEvidence::eligible)
    }

    pub fn set_native_ternary_promotion(
        &mut self,
        evidence: NativeTernaryPromotionEvidence,
    ) -> Result<(), String> {
        if !evidence.eligible() {
            return Err(format!(
                "native ternary promotion is not eligible: {}",
                evidence
                    .reject_reason()
                    .unwrap_or_else(|| "unknown promotion failure".to_string())
            ));
        }
        self.header.native_ternary_promotion = Some(evidence);
        Ok(())
    }

    /// Append a compiled Metal kernel payload to the .cimage file.
    ///
    /// The payload (compiled .metallib bytes) is written at the next 16 KB
    /// aligned offset and recorded in the header's `kernels` map.
    pub fn append_kernel(&mut self, name: &str, metallib_bytes: &[u8]) -> Result<(), String> {
        self.append_kernel_with_descriptor(name, metallib_bytes, None)
    }

    pub fn append_kernel_with_descriptor(
        &mut self,
        name: &str,
        metallib_bytes: &[u8],
        descriptor: Option<prism_ecs_kernel::KernelDescriptor>,
    ) -> Result<(), String> {
        self.align_to_page()?;
        let offset = self.current_pos()?;
        self.file
            .write_all(metallib_bytes)
            .map_err(|e| format!("write kernel '{}': {e}", name))?;
        self.header.kernels.insert(
            name.to_string(),
            KernelRecord {
                offset,
                size: metallib_bytes.len() as u64,
                name: name.to_string(),
                descriptor,
            },
        );
        Ok(())
    }

    /// Finalize: write magic + header to the first 16 KB block.
    pub fn finalize(mut self) -> Result<(), String> {
        let header_json =
            serde_json::to_string(&self.header).map_err(|e| format!("serialize header: {e}"))?;
        let header_bytes = header_json.as_bytes();
        let header_size = header_bytes.len() as u64;

        // Must fit in the reserved header block
        let reserved = HEADER_PAGES * PAGE_SIZE;
        assert!(
            16 + header_size <= reserved,
            "Header ({} B) exceeds reserved {} B",
            16 + header_size,
            reserved
        );

        self.file
            .seek(SeekFrom::Start(0))
            .map_err(|e| format!("seek to start: {e}"))?;
        self.file
            .write_all(MAGIC)
            .map_err(|e| format!("write magic: {e}"))?;
        self.file
            .write_all(&header_size.to_le_bytes())
            .map_err(|e| format!("write header size: {e}"))?;
        self.file
            .write_all(header_bytes)
            .map_err(|e| format!("write header: {e}"))?;
        self.file.flush().map_err(|e| format!("flush: {e}"))?;
        Ok(())
    }

    /// Pad file to the next 16 KB boundary.
    fn align_to_page(&mut self) -> Result<(), String> {
        let pos = self.current_pos()?;
        let remainder = pos % PAGE_SIZE;
        if remainder != 0 {
            let pad = (PAGE_SIZE - remainder) as usize;
            let zeros = vec![0u8; pad];
            self.file
                .write_all(&zeros)
                .map_err(|e| format!("align padding: {e}"))?;
        }
        Ok(())
    }

    fn current_pos(&mut self) -> Result<u64, String> {
        self.file
            .stream_position()
            .map_err(|e| format!("stream position: {e}"))
    }
}

// ── UniversalCImageWriter (high-level) ──────────────────────────────────

/// High-level CImage writer that wraps CImageWriter with compilation metadata.
pub struct UniversalCImageWriter {
    writer: CImageWriter,
}

impl UniversalCImageWriter {
    pub fn new(output_path: &Path) -> Self {
        Self {
            writer: CImageWriter::new(output_path).expect("failed to create CImage output"),
        }
    }

    pub fn set_source(&mut self, source: &prism_ecs_source::CanonicalSource) {
        self.writer.header.source_identity = Some(source.identity.clone());
        self.writer.header.source_catalog = Some(source.catalog.clone());
    }

    pub fn set_model_capabilities<I, S>(&mut self, capabilities: I)
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.writer.set_model_capabilities(capabilities);
    }

    pub fn set_kv_compression_policy(
        &mut self,
        policy: &prism_ecs_quantization::kv_search::KvCompressionCandidate,
        max_error: f32,
    ) -> Result<(), String> {
        self.writer.set_kv_compression_policy(policy, max_error)
    }

    pub fn set_execution_plan(&mut self, plan_json: String) {
        self.writer.header.execution_plan = Some(plan_json);
    }

    pub fn set_heterogeneous_workload_evidence(
        &mut self,
        evidence: crate::search::HeterogeneousScheduleEvidence,
    ) {
        self.writer.header.heterogeneous_workload_evidence = Some(evidence);
    }

    pub fn add_xdna_artifact(
        &mut self,
        name: &str,
        payload: &[u8],
        compiler_abi: impl Into<String>,
        generation: impl Into<String>,
    ) -> Result<(), String> {
        self.writer
            .add_xdna_artifact(name, payload, compiler_abi, generation)
    }

    /// Embed the tinygrad-inspired executable capture alongside the existing
    /// Prism execution plan. The CImage remains the ownership boundary: the
    /// compiler core produces a deterministic capture, while this writer
    /// seals its serialized plan into the artifact metadata.
    pub fn set_uop_capture(
        &mut self,
        capture: &prism_spatial_ir::CapturePlan,
    ) -> Result<(), String> {
        let envelope = serde_json::json!({
            "capture_digest": capture.digest(),
            "capture": capture,
        });
        let json = serde_json::to_string(&envelope)
            .map_err(|error| format!("serialize UOp capture: {error}"))?;
        self.set_execution_plan(json);
        Ok(())
    }

    /// Store additional executable UOp captures under stable strategy IDs.
    /// The captures are validated before entering the artifact header.
    pub fn set_uop_strategy_captures(
        &mut self,
        captures: &[(String, prism_spatial_ir::CapturePlan)],
    ) -> Result<(), String> {
        let mut records = HashMap::with_capacity(captures.len());
        for (strategy, capture) in captures {
            capture.validate()?;
            if records.contains_key(strategy) {
                return Err(format!("duplicate UOp strategy capture {strategy:?}"));
            }
            records.insert(
                strategy.clone(),
                UOpCaptureRecord {
                    capture_digest: capture.digest(),
                    capture: serde_json::to_string(capture)
                        .map_err(|error| format!("serialize UOp strategy capture: {error}"))?,
                    search_generation: None,
                },
            );
        }
        self.writer.header.uop_captures = records;
        Ok(())
    }

    /// Publish measured workload choices alongside the executable candidate
    /// set.  Every choice is recomputed through the shared selector so a
    /// caller cannot seal a stale or mismatched winner.
    pub fn set_uop_workload_evidence(
        &mut self,
        evaluations: &[crate::uop::UOpWorkloadMeasurement],
        strategies: &[prism_spatial_ir::FusionStrategy],
    ) -> Result<(), String> {
        if strategies.is_empty() {
            return Err("UOp workload evidence requires candidates".into());
        }
        let strategy_ids = strategies
            .iter()
            .map(|strategy| strategy.stable_id().to_string())
            .collect::<Vec<_>>();
        if strategy_ids.iter().any(String::is_empty)
            || strategy_ids.windows(2).any(|window| window[0] == window[1])
            || strategy_ids
                .iter()
                .enumerate()
                .any(|(index, strategy)| strategy_ids[..index].contains(strategy))
        {
            return Err("UOp workload evidence requires unique strategy IDs".into());
        }
        let mut evidence = Vec::with_capacity(evaluations.len());
        let mut scenarios = std::collections::HashSet::new();
        for evaluation in evaluations {
            evaluation.scenario.validate()?;
            if !scenarios.insert(evaluation.scenario) {
                return Err(format!(
                    "duplicate workload evidence for {:?}",
                    evaluation.scenario
                ));
            }
            let (selected_strategy, _) =
                crate::uop::select_measured_uop_strategy(strategies, &evaluation.measurements)?;
            evidence.push(UOpWorkloadEvidence {
                scenario: evaluation.scenario,
                strategies: strategy_ids.clone(),
                candidate_capture_digests: strategy_ids
                    .iter()
                    .map(|strategy| {
                        self.writer
                            .header
                            .uop_captures
                            .get(strategy)
                            .map(|record| record.capture_digest.clone())
                            .ok_or_else(|| {
                                format!(
                                    "UOp workload evidence references unembedded strategy {strategy:?}"
                                )
                            })
                    })
                    .collect::<Result<Vec<_>, String>>()?,
                measurements: evaluation.measurements.clone(),
                selected_strategy,
            });
        }
        self.writer.header.uop_workload_evidence = evidence;
        Ok(())
    }

    /// Add one measured workload table while ensuring the selected strategy
    /// is present in the sealed candidate set.
    pub fn add_uop_workload_evidence(
        &mut self,
        evaluations: &[crate::uop::UOpWorkloadMeasurement],
        strategies: &[prism_spatial_ir::FusionStrategy],
    ) -> Result<(), String> {
        self.set_uop_workload_evidence(evaluations, strategies)?;
        for entry in &self.writer.header.uop_workload_evidence {
            if !self
                .writer
                .header
                .uop_captures
                .contains_key(&entry.selected_strategy)
            {
                return Err(format!(
                    "workload evidence selects unembedded UOp strategy {:?}",
                    entry.selected_strategy
                ));
            }
        }
        Ok(())
    }

    /// Compile and embed strategy-indexed UOp captures. Kernel names are
    /// namespaced by strategy and ordinal so alternate layouts can coexist
    /// in one CImage without colliding with the legacy selected capture.
    pub fn add_uop_strategy_captures(
        &mut self,
        captures: &[(String, prism_spatial_ir::CapturePlan)],
    ) -> Result<(), String> {
        self.set_uop_strategy_captures(captures)?;
        for (strategy, capture) in captures {
            let prefix = crate::uop::strategy_kernel_prefix(strategy);
            let mut artifacts = crate::uop::compile_and_validate_uop_capture(capture)?;
            for artifact in &mut artifacts {
                for (index, payload) in artifact.payloads.iter_mut().enumerate() {
                    let name = format!("{prefix}{index}");
                    payload.descriptor.name = name.clone();
                    if let Some(descriptor) = artifact.manifest.kernels.get_mut(index) {
                        descriptor.name = name;
                    }
                }
                self.add_kernel_artifact(artifact.clone());
            }
        }
        Ok(())
    }

    /// Compile and publish a complete UOp strategy candidate set from one
    /// graph. The first candidate remains the legacy selected capture while
    /// every candidate is also available under its stable strategy ID.
    pub fn add_uop_strategy_candidate_set(
        &mut self,
        graph: &prism_spatial_ir::TinyGraph,
        target: prism_spatial_ir::LoweringTarget,
        strategies: &[prism_spatial_ir::FusionStrategy],
    ) -> Result<(), String> {
        let candidates = crate::uop::compile_uop_graph_strategies(graph, target, strategies)?;
        for (_, capture, _) in &candidates {
            // Candidate publication is still an admission boundary: an
            // unknown operation may be classified as Candidate, but it must
            // pass the same structural/backend validation as every other
            // executable capture before it enters the CImage.
            crate::uop::compile_and_validate_uop_capture(capture)?;
        }
        let Some((_, selected_capture, _)) = candidates.first() else {
            return Err("UOp strategy candidate set cannot be empty".into());
        };
        self.add_uop_capture(selected_capture)?;
        let search_generations = candidates
            .iter()
            .filter_map(|(strategy, _, _)| match strategy {
                prism_spatial_ir::FusionStrategy::PersistentMegakernel { search_generation } => {
                    Some((strategy.stable_id().to_string(), *search_generation))
                }
                _ => None,
            })
            .collect::<HashMap<_, _>>();
        let captures = candidates
            .into_iter()
            .map(|(strategy, capture, _)| (strategy.stable_id().into(), capture))
            .collect::<Vec<_>>();
        self.add_uop_strategy_captures(&captures)?;
        for (strategy, generation) in search_generations {
            if let Some(record) = self.writer.header.uop_captures.get_mut(&strategy) {
                record.search_generation = Some(generation);
            }
        }
        Ok(())
    }

    /// Publish executable strategy candidates and their workload evidence as
    /// one compiler operation.  Evidence is validated only after all
    /// candidate captures are embedded, making the resulting CImage an
    /// auditable executable set rather than two independently mutable tables.
    pub fn add_uop_strategy_candidate_set_with_evidence(
        &mut self,
        graph: &prism_spatial_ir::TinyGraph,
        target: prism_spatial_ir::LoweringTarget,
        strategies: &[prism_spatial_ir::FusionStrategy],
        evaluations: &[crate::uop::UOpWorkloadMeasurement],
    ) -> Result<(), String> {
        self.add_uop_strategy_candidate_set(graph, target, strategies)?;
        self.add_uop_workload_evidence(evaluations, strategies)
    }

    /// Compile and embed a UOp capture using the canonical kernel backend
    /// contract, while retaining the capture and replay receipt in metadata.
    pub fn add_uop_capture(
        &mut self,
        capture: &prism_spatial_ir::CapturePlan,
    ) -> Result<(), String> {
        self.set_uop_capture(capture)?;
        for artifact in crate::uop::compile_and_validate_uop_capture(capture)? {
            self.add_kernel_artifact(artifact);
        }
        Ok(())
    }

    /// Compile an edge-connected SpatialGraph and seal its executable UOp
    /// capture and kernel artifacts into this CImage.
    pub fn add_spatial_graph(
        &mut self,
        graph: &prism_spatial_ir::SpatialGraph,
        target: prism_spatial_ir::LoweringTarget,
    ) -> Result<(), String> {
        let (capture, artifacts) = crate::compile_spatial_graph(graph, target)?;
        self.set_uop_capture(&capture)?;
        for artifact in artifacts {
            self.add_kernel_artifact(artifact);
        }
        Ok(())
    }

    pub fn set_format_plan(&mut self, plan_json: String) -> Result<(), String> {
        self.writer.set_format_plan(plan_json)
    }

    pub fn set_model_manifest(
        &mut self,
        manifest: crate::model_manifest::MultiModelManifest,
    ) -> Result<(), String> {
        self.writer.set_model_manifest(manifest)
    }

    pub fn set_model_identity<T: serde::Serialize>(
        &mut self,
        family: impl Into<String>,
        config: &T,
    ) -> Result<(), String> {
        self.writer.set_model_identity(family, config)
    }

    pub fn set_qwen36_config(
        &mut self,
        config: crate::qwen3_6_moe::Qwen36Config,
    ) -> Result<(), String> {
        self.writer.set_qwen36_config(config)
    }

    pub fn set_moe_tensor(
        &mut self,
        name: &str,
        descriptor: MoeTensorDescriptor,
    ) -> Result<(), String> {
        self.writer.set_moe_tensor(name, descriptor)
    }

    pub fn set_vision_tensor(
        &mut self,
        name: &str,
        descriptor: VisionTensorDescriptor,
    ) -> Result<(), String> {
        self.writer.set_vision_tensor(name, descriptor)
    }

    /// Attach the measured promotion receipt produced by the evolutionary
    /// scheduler and backend validators.  The receipt is preserved verbatim
    /// in the CImage header and is checked again by the runtime loader.
    pub fn set_native_ternary_promotion(
        &mut self,
        evidence: NativeTernaryPromotionEvidence,
    ) -> Result<(), String> {
        if !evidence.eligible() {
            return Err(format!(
                "native ternary promotion is not eligible: {}",
                evidence
                    .reject_reason()
                    .unwrap_or_else(|| "unknown promotion failure".to_string())
            ));
        }
        self.writer.header.native_ternary_promotion = Some(evidence);
        Ok(())
    }

    pub fn set_joint_tiling_evidence(&mut self, evidence: crate::search::JointTilingEvidence) {
        self.writer.header.joint_tiling_evidence = Some(evidence);
    }

    pub fn add_tensor_payload(&mut self, entry: TensorPayloadEntry) -> Result<(), String> {
        self.writer.append(
            &entry.name,
            &entry.payload,
            entry.dim_m,
            entry.dim_n,
            entry.tensor_type,
        )
    }

    pub fn add_native_ternary_payload(
        &mut self,
        name: &str,
        payload: &[u8],
        dim_m: u32,
        dim_n: u32,
        tensor_type: TensorType,
        descriptor: TernaryDescriptor,
    ) -> Result<(), String> {
        self.writer
            .append_native_ternary(name, payload, dim_m, dim_n, tensor_type, descriptor)
    }

    pub fn add_native_ternary_payload_with_scales(
        &mut self,
        name: &str,
        payload: &[u8],
        scales: &[u8],
        dim_m: u32,
        dim_n: u32,
        tensor_type: TensorType,
        descriptor: TernaryDescriptor,
    ) -> Result<(), String> {
        self.writer.append_native_ternary_with_scales(
            name,
            payload,
            scales,
            dim_m,
            dim_n,
            tensor_type,
            descriptor,
        )
    }

    pub fn set_legalization_report(&mut self, report: crate::legalize::LegalizationReport) {
        if let Ok(json) = serde_json::to_string(&report) {
            self.writer.header.legalization_report = Some(json);
        }
    }

    pub fn set_events(&mut self, events: Vec<crate::CompilationEvent>) {
        if let Ok(json) = serde_json::to_string(&events) {
            self.writer.header.compilation_events = Some(json);
        }
    }

    pub fn set_search_trace(&mut self, trace: crate::SearchTrace) {
        if let Ok(json) = serde_json::to_string(&trace) {
            self.writer.header.search_trace = Some(json);
        }
    }

    pub fn set_uop_tuning_receipt(&mut self, receipt: crate::uop::UOpTuningReceipt) {
        self.writer.header.uop_tuning_receipt = Some(receipt);
    }

    pub fn set_selection_receipt(&mut self, receipt: crate::search::SearchSelectionReceipt) {
        self.writer.header.selection_receipt = Some(receipt);
    }

    pub fn add_kernel_artifact(&mut self, artifact: prism_ecs_kernel::KernelArtifact) {
        for payload in artifact.payloads {
            let name = payload.descriptor.name.clone();
            let descriptor = payload.descriptor;
            let _ =
                self.writer
                    .append_kernel_with_descriptor(&name, &payload.binary, Some(descriptor));
        }
    }

    /// Embed a compiled stateless ANE model and record its explicit int8 ABI.
    pub fn add_ane_program(
        &mut self,
        name: &str,
        modelc_payload: &[u8],
        activation_input: &str,
        weights_input: &str,
        output: &str,
    ) -> Result<(), String> {
        self.add_ane_program_typed(
            name,
            modelc_payload,
            activation_input,
            weights_input,
            output,
            "int8",
            "int8",
        )
    }

    pub fn add_ane_program_typed(
        &mut self,
        name: &str,
        modelc_payload: &[u8],
        activation_input: &str,
        weights_input: &str,
        output: &str,
        input_dtype: &str,
        output_dtype: &str,
    ) -> Result<(), String> {
        self.writer
            .append(name, modelc_payload, 0, 0, TensorType::Blob)?;
        let record = self
            .writer
            .header
            .tensors
            .get(name)
            .ok_or_else(|| format!("ANE program tensor '{name}' was not recorded"))?;
        let offset = record.offset;
        self.writer.header.ane_programs.insert(
            name.to_string(),
            AneProgramRecord {
                offset,
                size: modelc_payload.len() as u64,
                name: name.to_string(),
                activation_input: activation_input.to_string(),
                weights_input: weights_input.to_string(),
                output: output.to_string(),
                input_dtype: input_dtype.into(),
                output_dtype: output_dtype.into(),
            },
        );
        Ok(())
    }

    pub fn finalize(self) -> Result<(), String> {
        if self.writer.contains_native_ternary() && !self.writer.has_native_ternary_promotion() {
            return Err("native ternary CImage requires eligible promotion evidence".into());
        }
        self.writer.finalize()
    }

    /// Write the structural artifact without a promotion receipt. This is
    /// only for the first phase of [`super::promote_cimage_after_replay`];
    /// runtime admission still rejects the resulting file until promotion completes.
    pub fn finalize_unpromoted(self) -> Result<(), String> {
        self.writer.finalize()
    }
}
