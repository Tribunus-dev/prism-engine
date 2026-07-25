//! CImage reader — runtime loader for the `.cimage` format.
//!
//! This module owns the canonical authority for *reading* a `.cimage` file:
//! header parsing, payload location, format validation, evidence
//! verification, and the top-level file open + blob read helpers. The
//! matching write authority lives in [`super::writer`] (`CImageWriter` and
//! `UniversalCImageWriter`); the pure data definitions (`TensorType`,
//! `CImageHeader`, `TensorRecord`, descriptors) live in [`super`].
//!
//! The migration out of the monolithic `cimage.rs` is part of the
//! constitutional module-cohesion work (see
//! `references/module-discipline.md` §Concrete decomposition patterns for
//! Prism). Further reader-side extraction (e.g. a per-format validator
//! for UOp evidence, or a separate XDNA / ANE / Metal loader) is
//! possible future work; this module is the single, typed entry point
//! for `.cimage` reads for now.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use super::{CImageHeader, MAGIC, PAGE_SIZE, TensorRecord, TensorType, TernaryDescriptor, UOpWorkloadEvidence};

/// Loaded .cimage header (disk metadata read without payload).
pub struct CImageReader {
    pub header: CImageHeader,
    pub(crate) _file: File,
    /// Original file path, used when re-opening for payload reads.
    path: std::path::PathBuf,
}

impl CImageReader {
    /// Open a .cimage file and parse the header.
    pub fn open(path: &Path) -> Result<Self, String> {
        let mut file = File::open(path).map_err(|e| format!("open .cimage: {e}"))?;

        // Read magic
        let mut magic = [0u8; 8];
        file.read_exact(&mut magic)
            .map_err(|e| format!("read magic: {e}"))?;
        if &magic != MAGIC {
            return Err(format!(
                "Invalid magic: expected TRB_CIMG, got {:?}",
                &magic
            ));
        }

        // Read header size
        let mut hdr_size_bytes = [0u8; 8];
        file.read_exact(&mut hdr_size_bytes)
            .map_err(|e| format!("read header size: {e}"))?;
        let hdr_size = u64::from_le_bytes(hdr_size_bytes) as usize;

        // Read JSON header
        let mut hdr_buf = vec![0u8; hdr_size];
        file.read_exact(&mut hdr_buf)
            .map_err(|e| format!("read header: {e}"))?;
        let header: CImageHeader =
            serde_json::from_slice(&hdr_buf).map_err(|e| format!("parse header: {e}"))?;

        Ok(CImageReader {
            header,
            _file: file,
            path: path.to_path_buf(),
        })
    }

    /// Read and validate the typed UOp capture embedded by
    /// [`super::UniversalCImageWriter::add_uop_capture`].
    pub fn uop_capture(&self) -> Result<prism_spatial_ir::CapturePlan, String> {
        let plan = self
            .header
            .execution_plan
            .as_ref()
            .ok_or_else(|| "CImage has no execution plan".to_string())?;
        let envelope: serde_json::Value = serde_json::from_str(plan)
            .map_err(|error| format!("parse UOp capture envelope: {error}"))?;
        let expected_digest = envelope
            .get("capture_digest")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| "UOp capture envelope has no capture_digest".to_string())?;
        let capture: prism_spatial_ir::CapturePlan = serde_json::from_value(
            envelope
                .get("capture")
                .cloned()
                .ok_or_else(|| "UOp capture envelope has no capture".to_string())?,
        )
        .map_err(|error| format!("decode UOp capture: {error}"))?;
        if capture.digest() != expected_digest {
            return Err("UOp capture digest does not match CImage metadata".into());
        }
        capture.validate()?;
        Ok(capture)
    }

    /// Read one strategy-indexed UOp capture from a newer CImage. Legacy
    /// artifacts simply return a missing-key error and remain readable.
    pub fn uop_capture_for_strategy(
        &self,
        strategy: &str,
    ) -> Result<prism_spatial_ir::CapturePlan, String> {
        let record = self
            .header
            .uop_captures
            .get(strategy)
            .ok_or_else(|| format!("CImage has no UOp strategy capture {strategy:?}"))?;
        let capture: prism_spatial_ir::CapturePlan = serde_json::from_str(&record.capture)
            .map_err(|error| format!("decode UOp strategy capture: {error}"))?;
        if capture.digest() != record.capture_digest {
            return Err(format!("UOp strategy capture {strategy:?} digest mismatch"));
        }
        capture.validate()?;
        Ok(capture)
    }

    /// Return the evolutionary generation recorded for a strategy, when the
    /// compiler had one. Legacy strategy records return `None`.
    pub fn uop_strategy_search_generation(&self, strategy: &str) -> Option<u32> {
        self.header
            .uop_captures
            .get(strategy)
            .and_then(|record| record.search_generation)
    }

    /// Return the structured UOp tuning receipt after validating its digest
    /// and capture references. A non-production receipt remains readable for
    /// diagnostics but must not be used as runtime selection authority.
    pub fn uop_tuning_receipt(&self) -> Result<Option<&crate::uop::UOpTuningReceipt>, String> {
        let Some(receipt) = self.header.uop_tuning_receipt.as_ref() else {
            return Ok(None);
        };
        receipt.validate()?;
        for scenario in &receipt.scenarios {
            for candidate in &scenario.candidates {
                let Some(record) = self.header.uop_captures.get(&candidate.strategy_id) else {
                    return Err(format!(
                        "UOp tuning receipt references missing strategy {:?}",
                        candidate.strategy_id
                    ));
                };
                if record.capture_digest != candidate.capture_digest {
                    return Err(format!(
                        "UOp tuning receipt capture digest mismatch for {:?}",
                        candidate.strategy_id
                    ));
                }
            }
        }
        Ok(Some(receipt))
    }

    /// Return validated workload evidence embedded in the CImage.  The
    /// selected strategy is checked against the candidate capture table so
    /// metadata cannot direct a runtime to an absent executable.
    pub fn uop_workload_evidence(&self) -> Result<&[UOpWorkloadEvidence], String> {
        let mut scenarios = std::collections::HashSet::new();
        for entry in &self.header.uop_workload_evidence {
            entry.scenario.validate()?;
            if !scenarios.insert(entry.scenario) {
                return Err(format!(
                    "duplicate workload evidence for {:?}",
                    entry.scenario
                ));
            }
            if entry.strategies.is_empty()
                || entry.strategies.len() != entry.measurements.len()
                || (!entry.candidate_capture_digests.is_empty()
                    && entry.strategies.len() != entry.candidate_capture_digests.len())
            {
                return Err(format!(
                    "workload evidence for {:?} has invalid candidate dimensions",
                    entry.scenario
                ));
            }
            let mut strategy_ids = std::collections::HashSet::new();
            if entry
                .strategies
                .iter()
                .any(|strategy| strategy.is_empty() || !strategy_ids.insert(strategy))
            {
                return Err(format!(
                    "workload evidence for {:?} has duplicate or empty strategy IDs",
                    entry.scenario
                ));
            }
            if entry
                .measurements
                .iter()
                .enumerate()
                .any(|(index, measurement)| measurement.candidate_index != index)
            {
                return Err(format!(
                    "workload evidence for {:?} has noncanonical candidate indices",
                    entry.scenario
                ));
            }
            if entry
                .measurements
                .iter()
                .any(|measurement| measurement.latency_ns == 0)
            {
                return Err(format!(
                    "workload evidence for {:?} contains a zero-latency sample",
                    entry.scenario
                ));
            }
            for strategy in &entry.strategies {
                if !self.header.uop_captures.contains_key(strategy) {
                    return Err(format!(
                        "workload evidence references unembedded UOp strategy {strategy:?}"
                    ));
                }
                self.uop_capture_for_strategy(strategy).map_err(|error| {
                    format!(
                        "workload evidence references invalid UOp strategy {strategy:?}: {error}"
                    )
                })?;
            }
            if !entry.candidate_capture_digests.is_empty() {
                for (strategy, digest) in entry
                    .strategies
                    .iter()
                    .zip(&entry.candidate_capture_digests)
                {
                    if digest != &self.header.uop_captures[strategy].capture_digest {
                        return Err(format!(
                            "workload evidence digest mismatch for UOp strategy {strategy:?}"
                        ));
                    }
                }
            }
            if !self
                .header
                .uop_captures
                .contains_key(&entry.selected_strategy)
            {
                return Err(format!(
                    "workload evidence selects unembedded UOp strategy {:?}",
                    entry.selected_strategy
                ));
            }
            if !entry
                .strategies
                .iter()
                .any(|strategy| strategy == &entry.selected_strategy)
            {
                return Err(format!(
                    "workload evidence selects a strategy outside its candidate set for {:?}",
                    entry.scenario
                ));
            }
            let selected_index = entry
                .strategies
                .iter()
                .position(|strategy| strategy == &entry.selected_strategy)
                .expect("selected strategy was checked above");
            let best_index = entry
                .measurements
                .iter()
                .enumerate()
                .min_by_key(|(_, measurement)| {
                    measurement
                        .latency_ns
                        .saturating_add(measurement.materialized_bytes / 100)
                })
                .map(|(index, _)| index)
                .expect("nonempty measurements were checked above");
            if selected_index != best_index {
                return Err(format!(
                    "workload evidence selects a non-winning strategy for {:?}",
                    entry.scenario
                ));
            }
        }
        Ok(&self.header.uop_workload_evidence)
    }

    /// Return the validated evidence for one exact workload shape.
    pub fn uop_workload_evidence_for(
        &self,
        scenario: prism_spatial_ir::WorkloadScenario,
    ) -> Result<Option<&UOpWorkloadEvidence>, String> {
        Ok(self
            .uop_workload_evidence()?
            .iter()
            .find(|entry| entry.scenario == scenario))
    }

    /// Return the offset + size for a named tensor.
    pub fn tensor(&self, name: &str) -> Option<&TensorRecord> {
        self.header.tensors.get(name)
    }

    /// Read an embedded native XDNA artifact after validating its recorded
    /// range against the CImage file length.
    pub fn xdna_artifact(&self, name: &str) -> Result<Vec<u8>, String> {
        let record = self
            .header
            .xdna_artifacts
            .get(name)
            .ok_or_else(|| format!("XDNA artifact not found: {name}"))?;
        let file_len = self
            ._file
            .metadata()
            .map_err(|e| format!("stat CImage: {e}"))?
            .len();
        let end = record
            .offset
            .checked_add(record.size)
            .ok_or_else(|| format!("XDNA artifact {name} range overflows"))?;
        if record.size == 0 || end > file_len {
            return Err(format!("XDNA artifact {name} range is outside CImage"));
        }
        let mut file = File::open(&self.path).map_err(|e| format!("open XDNA payload: {e}"))?;
        file.seek(SeekFrom::Start(record.offset))
            .map_err(|e| format!("seek XDNA payload: {e}"))?;
        let mut payload = vec![0u8; record.size as usize];
        file.read_exact(&mut payload)
            .map_err(|e| format!("read XDNA payload: {e}"))?;
        Ok(payload)
    }

    pub fn validate_tensor(&self, name: &str) -> Result<TernaryDescriptor, String> {
        let record = self
            .tensor(name)
            .ok_or_else(|| format!("tensor not found: {name}"))?;
        if let Some(descriptor) = &record.ternary {
            descriptor.validate()?;
            return Ok(descriptor.clone());
        }
        TernaryDescriptor::legacy_for_type(&record.tensor_type)
            .ok_or_else(|| format!("tensor '{name}' is not a ternary tensor"))
    }

    /// Header-only runtime admission for tensor descriptors and file ranges.
    pub fn validate_payload_ranges(&self) -> Result<(), String> {
        self.validate_payload_ranges_with_promotion(true)
    }

    pub fn validate_format_plan(&self) -> Result<(), String> {
        if let Some(plan) = &self.header.format_plan {
            let _: prism_ecs_ir::evolution::compile_plan::FormatPlan =
                serde_json::from_str(plan)
                    .map_err(|e| format!("invalid CImage format plan: {e}"))?;
        }
        Ok(())
    }

    /// Validate an artifact for the hardware qualification phase before its
    /// promotion receipt exists. This still validates every range and
    /// descriptor, but deliberately leaves promotion eligibility to the
    /// caller after backend replay.
    pub fn validate_payload_ranges_for_validation(&self) -> Result<(), String> {
        self.validate_payload_ranges_with_promotion(false)
    }

    fn validate_payload_ranges_with_promotion(
        &self,
        require_promotion: bool,
    ) -> Result<(), String> {
        self.validate_format_plan()?;
        self.validate_model_identity()?;
        self.validate_qwen36_tensor_contract()?;
        if require_promotion {
            self.validate_native_ternary_promotion()?;
        }
        let file_len = self
            ._file
            .metadata()
            .map_err(|e| format!("stat CImage: {e}"))?
            .len();
        for (name, record) in &self.header.tensors {
            if record.offset % PAGE_SIZE != 0 {
                return Err(format!("tensor '{name}' is not page aligned"));
            }
            let end = record
                .offset
                .checked_add(record.size)
                .ok_or_else(|| format!("tensor '{name}' range overflows"))?;
            if end > file_len {
                return Err(format!("tensor '{name}' exceeds CImage file length"));
            }
            if record.ternary.is_some() {
                self.validate_tensor(name)?;
            }
            if let Some(family) = &record.semantic_family {
                if !matches!(
                    family.as_str(),
                    "vision"
                        | "router"
                        | "routed_expert"
                        | "embedding_or_head"
                        | "normalization"
                        | "attention"
                        | "shared"
                ) {
                    return Err(format!(
                        "tensor '{name}' has unknown semantic family '{family}'"
                    ));
                }
                if record.router_sensitive && family != "router" {
                    return Err(format!(
                        "tensor '{name}' is marked router-sensitive but belongs to '{family}'"
                    ));
                }
            }
            let is_native_ternary = matches!(
                record.tensor_type,
                TensorType::Ternary158 | TensorType::TernaryTile640
            );
            match (record.scale_offset, record.scale_size) {
                (Some(scale_offset), Some(scale_size)) => {
                    if !is_native_ternary {
                        return Err(format!("non-ternary tensor '{name}' has scale metadata"));
                    }
                    if scale_offset % PAGE_SIZE != 0 {
                        return Err(format!(
                            "scale payload for tensor '{name}' is not page aligned"
                        ));
                    }
                    if scale_size == 0 {
                        return Err(format!("scale payload for tensor '{name}' is empty"));
                    }
                    let scale_end = scale_offset.checked_add(scale_size).ok_or_else(|| {
                        format!("scale payload for tensor '{name}' range overflows")
                    })?;
                    if scale_end > file_len {
                        return Err(format!(
                            "scale payload for tensor '{name}' exceeds CImage file length"
                        ));
                    }
                }
                (None, None) if is_native_ternary => {
                    return Err(format!(
                        "native ternary tensor '{name}' is missing scale metadata"
                    ));
                }
                (Some(_), None) | (None, Some(_)) => {
                    return Err(format!("tensor '{name}' has incomplete scale metadata"));
                }
                (None, None) => {}
            }
        }
        for (name, record) in &self.header.kernels {
            let end = record
                .offset
                .checked_add(record.size)
                .ok_or_else(|| format!("kernel '{name}' range overflows"))?;
            if record.offset % PAGE_SIZE != 0 || end > file_len {
                return Err(format!("kernel '{name}' has an invalid file range"));
            }
        }
        for (name, record) in &self.header.ane_programs {
            let end = record
                .offset
                .checked_add(record.size)
                .ok_or_else(|| format!("ANE program '{name}' range overflows"))?;
            if record.offset % PAGE_SIZE != 0 || end > file_len {
                return Err(format!("ANE program '{name}' has an invalid file range"));
            }
        }
        for (name, record) in &self.header.xdna_artifacts {
            if record.offset % PAGE_SIZE != 0 {
                return Err(format!("XDNA artifact '{name}' is not page aligned"));
            }
            if record.size == 0 {
                return Err(format!("XDNA artifact '{name}' is empty"));
            }
            let end = record
                .offset
                .checked_add(record.size)
                .ok_or_else(|| format!("XDNA artifact '{name}' range overflows"))?;
            if end > file_len {
                return Err(format!("XDNA artifact '{name}' exceeds CImage file length"));
            }
            if record.compiler_abi.is_empty() || record.generation.is_empty() {
                return Err(format!("XDNA artifact '{name}' has incomplete metadata"));
            }
        }
        Ok(())
    }

    pub fn validate_model_identity(&self) -> Result<(), String> {
        match (&self.header.model_family, &self.header.model_config_json) {
            (None, None) => Ok(()),
            (Some(family), Some(config)) => {
                if family.trim().is_empty() {
                    return Err("model family is empty".into());
                }
                serde_json::from_str::<serde_json::Value>(config)
                    .map(|_| ())
                    .map_err(|e| format!("invalid model configuration for '{family}': {e}"))
            }
            (Some(_), None) => Err("model identity is missing model configuration".into()),
            (None, Some(_)) => Err("model configuration is missing model family".into()),
        }
    }

    /// Validate tensor-level Qwen3.6 MoE annotations against the embedded
    /// model configuration before runtime dispatch or replay.
    pub fn validate_qwen36_tensor_contract(&self) -> Result<(), String> {
        let Some(config) = &self.header.qwen36_config else {
            return Ok(());
        };
        config.validate()?;
        let mut router_layers = std::collections::BTreeSet::new();
        let mut expert_bank_layers = std::collections::BTreeSet::new();
        let mut vision_tensor_count = 0usize;
        for (name, record) in &self.header.tensors {
            if record.vision.is_some() && config.vision_config.is_none() {
                return Err(format!(
                    "vision tensor '{name}' has no vision configuration"
                ));
            }
            if let Some(vision) = &record.vision {
                vision_tensor_count += 1;
                if vision.component.is_empty() {
                    return Err(format!("vision tensor '{name}' has an empty component"));
                }
            }
            let Some(moe) = &record.moe else { continue };
            if moe.layer as usize >= config.num_hidden_layers {
                return Err(format!("tensor '{name}' MoE layer is out of range"));
            }
            match moe.role.as_str() {
                "router" if moe.expert.is_some() => {
                    return Err(format!("tensor '{name}' has an unexpected expert id"));
                }
                "router" => {
                    router_layers.insert(moe.layer as usize);
                }
                "routed_expert" => {
                    let expert = moe
                        .expert
                        .ok_or_else(|| format!("tensor '{name}' is missing expert id"))?;
                    if expert as usize >= config.num_experts {
                        return Err(format!("tensor '{name}' expert is out of range"));
                    }
                }
                "routed_expert_bank" => {
                    if moe.expert.is_some() {
                        return Err(format!("tensor '{name}' has an unexpected expert id"));
                    }
                    if moe.expert_count != Some(config.num_experts as u32) {
                        return Err(format!(
                            "tensor '{name}' fused expert bank declares {:?}; expected {}",
                            moe.expert_count, config.num_experts
                        ));
                    }
                    expert_bank_layers.insert(moe.layer as usize);
                }
                "shared_expert" => {}
                role => return Err(format!("tensor '{name}' has invalid MoE role '{role}'")),
            }
        }
        if router_layers.is_empty() && expert_bank_layers.is_empty() {
            if config.vision_config.is_some() && vision_tensor_count == 0 {
                return Err("Qwen3.6 CImage declares vision but has no vision tensors".into());
            }
            return Ok(());
        }
        for layer in 0..config.num_hidden_layers {
            if !router_layers.contains(&layer) {
                return Err(format!(
                    "Qwen3.6 CImage is missing a router tensor for layer {layer}"
                ));
            }
            if !expert_bank_layers.contains(&layer)
                && !self.header.tensors.values().any(|record| {
                    record.moe.as_ref().is_some_and(|moe| {
                        moe.layer as usize == layer && moe.role == "routed_expert"
                    })
                })
            {
                return Err(format!(
                    "Qwen3.6 CImage is missing routed experts for layer {layer}"
                ));
            }
        }
        if config.vision_config.is_some() && vision_tensor_count == 0 {
            return Err("Qwen3.6 CImage declares vision but has no vision tensors".into());
        }
        Ok(())
    }

    /// Validate the native ternary promotion receipt before any payload is
    /// mapped or copied.  Legacy artifacts without ternary payloads have no
    /// receipt and remain compatible; artifacts carrying a receipt must prove
    /// that it is complete and eligible.
    pub fn validate_native_ternary_promotion(&self) -> Result<(), String> {
        if self
            .header
            .model_capabilities
            .iter()
            .any(|capability| capability == "persistent-kv")
            && self.header.kv_compression_policy.is_none()
        {
            return Err(
                "persistent-KV CImage is missing an evolutionary KV compression policy".into(),
            );
        }
        let has_native_ternary = self.header.tensors.values().any(|record| {
            matches!(
                record.tensor_type,
                TensorType::Ternary158 | TensorType::TernaryTile640
            )
        });
        if has_native_ternary {
            let evidence = self
                .header
                .native_ternary_promotion
                .as_ref()
                .ok_or_else(|| "native ternary CImage is missing promotion evidence".to_string())?;
            if let Some(reason) = evidence.reject_reason() {
                return Err(format!("native ternary promotion rejected: {reason}"));
            }
        } else if let Some(evidence) = &self.header.native_ternary_promotion {
            if !evidence.eligible() {
                return Err("CImage contains ineligible native ternary promotion evidence".into());
            }
        }
        Ok(())
    }

    /// Load compiled kernel bytes from the .cimage file by kernel name.
    ///
    /// Reads the payload at the offset recorded in the header's `kernels` map.
    /// The returned bytes can be passed directly to
    /// `metal::Library::new_library_with_data` for GPU dispatch without
    /// recompilation from MSL source.
    pub fn load_kernel(&self, name: &str) -> Result<Vec<u8>, String> {
        let record = self
            .header
            .kernels
            .get(name)
            .ok_or_else(|| format!("kernel '{name}' not found in .cimage"))?;

        let mut file =
            File::open(&self.path).map_err(|e| format!("open .cimage for kernel '{name}': {e}"))?;
        file.seek(SeekFrom::Start(record.offset))
            .map_err(|e| format!("seek to kernel '{name}': {e}"))?;
        let mut buf = vec![0u8; record.size as usize];
        file.read_exact(&mut buf)
            .map_err(|e| format!("read kernel '{name}': {e}"))?;
        Ok(buf)
    }

    pub fn load_ane_program(&self, name: &str) -> Result<Vec<u8>, String> {
        let record = self
            .header
            .ane_programs
            .get(name)
            .ok_or_else(|| format!("ANE program '{name}' not found in .cimage"))?;
        let mut file = File::open(&self.path)
            .map_err(|e| format!("open .cimage for ANE program '{name}': {e}"))?;
        file.seek(SeekFrom::Start(record.offset))
            .map_err(|e| format!("seek to ANE program '{name}': {e}"))?;
        let mut buf = vec![0u8; record.size as usize];
        file.read_exact(&mut buf)
            .map_err(|e| format!("read ANE program '{name}': {e}"))?;
        Ok(buf)
    }
}

/// Read a named blob payload from a .cimage file.
///
/// Opens the file, locates the blob by name in the header, and returns
/// the raw bytes. Used to extract embedded metadata like compilation receipts.
pub fn cimage_read_blob(path: &Path, name: &str) -> Result<Vec<u8>, String> {
    let reader = CImageReader::open(path)?;
    let record = reader
        .header
        .tensors
        .get(name)
        .ok_or_else(|| format!("blob '{name}' not found in .cimage"))?;

    let mut file = File::open(path).map_err(|e| format!("open .cimage: {e}"))?;
    file.seek(SeekFrom::Start(record.offset))
        .map_err(|e| format!("seek to blob: {e}"))?;
    let mut buf = vec![0u8; record.size as usize];
    file.read_exact(&mut buf)
        .map_err(|e| format!("read blob: {e}"))?;
    Ok(buf)
}
