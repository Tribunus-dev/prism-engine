//! Runtime model — the loaded `RuntimeModel` entity and its CImage inspection.
//!
//! This module owns the canonical authority for loading a `.cimage` file
//! into an in-memory [`RuntimeModel`] (the eagerly-loaded tensor, kernel,
//! ANE program, and XDNA artifact set), for inspecting a CImage header
//! without loading payloads, and for the small load-state helpers
//! ([`tensor_format`], [`num_layers`]). Per-tensor / per-kernel / per-UOp
//! accessors live in sibling submodules by authority category:
//!
//! - [`tensor_accessors`] — per-tensor data, shape, representation
//! - [`kernel_accessors`] — per-kernel binary, descriptor, artifact, ANE
//! - [`uop_accessors`] — UOp capture, program, workload evidence
//! - [`evidence_accessors`] — compiler-sealed evidence envelopes
//! - [`registry_accessors`] — namespaced multi-model registry
//! - [`auxiliary`] — small load-state helpers
//!
//! The model is the input to every other runtime entity in this crate:
//! [`super::binding`], [`super::ane_backend`], [`super::kernel_dispatch`],
//! [`super::xdna_dispatch`], [`super::unified`], and
//! [`super::certification`] all take a `&RuntimeModel` (or a reference to
//! one of its fields) as their immutable input. Mutations of model state
//! belong here; downstream entities are immutable views over the loaded
//! model.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use prism_amd_npu_runtime::XdnaArtifact;
use prism_ecs_quantization::ternarization::promotion::NativeTernaryPromotionEvidence;
use prism_spatial_ir::CapturePlan;
use prism_spatial_ir::execution_plan::ExecutionPlan;

use crate::cimage::{CImageManifest, CImageReader};
use crate::uop::UOpCompiledProgram;

// ── Submodules by accessor category ─────────────────────────────────────

pub mod auxiliary;
pub mod evidence_accessors;
pub mod kernel_accessors;
pub mod registry_accessors;
pub mod tensor_accessors;
pub mod uop_accessors;

// ── Re-exports — the original `runtime::model::TypeName` public path ──
//
// `RuntimeModel::num_layers` is publicly accessible because
// `auxiliary` is a `pub mod` and the `impl RuntimeModel` block there
// declares a `pub fn`. We do not need a re-export; the method is
// reachable as `runtime::model::RuntimeModel::num_layers` directly.

// ── The `RuntimeModel` struct ────────────────────────────────────────────

/// A manifest-loaded model ready for execution.
///
/// Holds the parsed [`CImageManifest`], eagerly-loaded tensor and kernel
/// payloads, and a lazy offset map for on-demand tensor access. All of the
/// payload data is loaded into memory at construction time so dispatch paths
/// never touch the file system during inference.
#[derive(Debug)]
pub struct RuntimeModel {
    /// Path to the `.cimage` file.
    pub cimage_path: PathBuf,
    /// Sealed compiler provenance consumed by runtime admission and replay.
    pub source_identity: Option<prism_ecs_source::SourceIdentity>,
    pub source_catalog: Option<prism_ecs_source::TensorCatalog>,
    pub search_trace: Option<crate::SearchTrace>,
    pub legalization_report: Option<crate::legalize::LegalizationReport>,
    pub compilation_events: Option<Vec<crate::CompilationEvent>>,
    /// Parsed manifest metadata.
    pub manifest: CImageManifest,
    /// Loaded tensor payloads indexed by tensor name.
    pub tensors: HashMap<String, Vec<u8>>,
    /// Tensor records carrying shape and representation metadata.
    pub tensor_records: HashMap<String, crate::cimage::TensorRecord>,
    /// Per-group scale payloads linked from native ternary tensor records.
    /// These remain packed and are exposed separately so native kernels can
    /// bind codes and scales without reconstructing FP16/FP32 weights.
    pub tensor_scales: HashMap<String, Vec<u8>>,
    /// Loaded kernel payloads indexed by kernel name.
    pub kernels: HashMap<String, Vec<u8>>,
    /// Typed descriptors paired with kernel payloads in the CImage header.
    pub kernel_descriptors: HashMap<String, prism_ecs_kernel::KernelDescriptor>,
    /// Optional tinygrad-inspired UOp capture embedded by the compiler.
    ///
    /// This is validated while loading so callers can safely use it as the
    /// executable graph contract rather than treating the JSON envelope as
    /// untrusted metadata.
    pub uop_capture: Option<CapturePlan>,
    /// Validated executable UOp program retained for production dispatch.
    pub uop_program: Option<UOpCompiledProgram>,
    /// Additional executable UOp programs indexed by published strategy ID.
    pub uop_strategy_programs: HashMap<String, UOpCompiledProgram>,
    /// Sealed workload measurements and validated strategy choices.
    pub uop_workload_evidence: Vec<crate::cimage::UOpWorkloadEvidence>,
    /// Embedded stateless int8 ANE programs and their input/output contract.
    pub ane_programs: HashMap<String, (crate::cimage::AneProgramRecord, Vec<u8>)>,
    /// Validated native XDNA artifacts embedded in the CImage, indexed by
    /// artifact name. These are ready for handoff to the AMD NPU runtime.
    pub xdna_artifacts: HashMap<String, XdnaArtifact>,
    /// Compiler-selected progressive KV compression policy, retained as its
    /// canonical serialized manifest value for serving/runtime coordination.
    pub kv_compression_policy: Option<String>,
    /// Namespaced specialised-model registry embedded in the CImage header.
    pub model_manifest: Option<crate::model_manifest::MultiModelManifest>,
    /// Backend promotion evidence carried by native ternary CImages.
    pub native_ternary_promotion: Option<NativeTernaryPromotionEvidence>,
    /// Joint ANE/Metal tiling evidence retained by the CImage.
    pub joint_tiling_evidence: Option<crate::search::JointTilingEvidence>,
    pub heterogeneous_workload_evidence: Option<crate::search::HeterogeneousScheduleEvidence>,
    /// Validated per-tensor representation policy produced by evolutionary
    /// family selection, including fallback formats.
    pub format_plan: Option<prism_ecs_ir::evolution::compile_plan::FormatPlan>,
    /// AOT heterogeneous schedule and residency contract emitted by the
    /// compiler.
    pub execution_plan: Option<ExecutionPlan>,
    /// Realtime schedule paired with the batch schedule in a KernelManifest.
    /// Older artifacts leave this unset and use `execution_plan` for their
    /// single schedule.
    pub realtime_execution_plan: Option<ExecutionPlan>,
    /// Lazy tensor offset map: tensor name → (file offset, byte size).
    /// Populated during parsing so individual tensors can be loaded
    /// on demand without re-reading the manifest region.
    pub tensor_offsets: HashMap<String, (u64, u64)>,
    /// Read-only mapping of the CImage file for zero-copy backend views.
    pub mapped_cimage: Option<memmap2::Mmap>,
}

impl RuntimeModel {
    /// Resolve the compiler-selected representation for a tensor. Missing
    /// assignments are conservative FP16 rather than an accidental ternary
    /// fallback, which is important for family outliers.
    pub fn tensor_format(
        &self,
        tensor_name: &str,
    ) -> prism_ecs_ir::evolution::mutation_table::TensorFormat {
        self.format_plan
            .as_ref()
            .and_then(|plan| plan.get(tensor_name))
            .unwrap_or(prism_ecs_ir::evolution::mutation_table::TensorFormat::Fp16)
    }
}

/// Header-only summary of a `.cimage` file, returned by inspection paths
/// that must not allocate payload memory.
pub struct CImageInspection {
    /// Parsed manifest metadata.
    pub manifest: CImageManifest,
    /// File size on disk in bytes.
    pub file_bytes: u64,
    /// Sum of `TensorRecord::size` across all declared tensors.
    pub tensor_bytes: u64,
    /// True when the CImage header advertises the `native-xdna` model
    /// capability and embeds at least one XDNA artifact envelope.
    pub has_native_xdna: bool,
    /// Count of XDNA artifacts in the header.
    pub xdna_artifact_count: usize,
    /// Namespaced specialised-model registry embedded in the CImage header.
    pub model_manifest: Option<crate::model_manifest::MultiModelManifest>,
    /// Backend promotion evidence carried by native ternary CImages.
    pub native_ternary_promotion: Option<NativeTernaryPromotionEvidence>,
    /// Joint ANE/Metal tiling evidence retained by the CImage.
    pub joint_tiling_evidence: Option<crate::search::JointTilingEvidence>,
}

fn read_region(file: &mut File, offset: u64, size: u64) -> Result<Vec<u8>, super::RuntimeError> {
    file.seek(SeekFrom::Start(offset))
        .map_err(|e| super::RuntimeError::InvalidCImage(format!("seek payload: {e}")))?;
    let mut payload = vec![0u8; size as usize];
    file.read_exact(&mut payload)
        .map_err(|e| super::RuntimeError::InvalidCImage(format!("read payload: {e}")))?;
    Ok(payload)
}

impl RuntimeModel {
    /// Inspect only the CImage header and payload extents.
    ///
    /// This path is safe for multi-gigabyte artifacts: it never reads or
    /// allocates tensor, kernel, or Core ML payloads.  Use it for admission,
    /// recovery, inventory, and memory-budget checks before calling `load`.
    pub fn inspect(path: &Path) -> Result<CImageInspection, super::RuntimeError> {
        Self::inspect_with_promotion(path, true)
    }

    /// Inspect a freshly emitted artifact before backend promotion. This is
    /// intended only for the qualification workflow; production admission
    /// must use [`RuntimeModel::inspect`].
    pub fn inspect_for_validation(path: &Path) -> Result<CImageInspection, super::RuntimeError> {
        Self::inspect_with_promotion(path, false)
    }

    fn inspect_with_promotion(
        path: &Path,
        require_promotion: bool,
    ) -> Result<CImageInspection, super::RuntimeError> {
        let reader = CImageReader::open(path).map_err(super::RuntimeError::InvalidCImage)?;
        let validation = if require_promotion {
            reader.validate_payload_ranges()
        } else {
            reader.validate_payload_ranges_for_validation()
        };
        validation.map_err(super::RuntimeError::InvalidCImage)?;
        let file_bytes = std::fs::metadata(path)
            .map_err(|e| super::RuntimeError::FileNotFound(e.to_string()))?
            .len();
        let tensor_bytes = reader.header.tensors.values().map(|r| r.size).sum();
        if let Some(catalog) = reader.header.source_catalog.as_ref() {
            if catalog
                .tensors
                .iter()
                .any(|tensor| !reader.header.tensors.contains_key(&tensor.name))
            {
                return Err(super::RuntimeError::InvalidCImage(
                    "source catalog references a tensor without an embedded payload".into(),
                ));
            }
        }
        if let Some(trace) = reader.header.search_trace.as_deref() {
            serde_json::from_str::<crate::SearchTrace>(trace).map_err(|error| {
                super::RuntimeError::InvalidCImage(format!("invalid sealed search trace: {error}"))
            })?;
        }
        if let Some(report) = reader.header.legalization_report.as_deref() {
            serde_json::from_str::<crate::legalize::LegalizationReport>(report).map_err(
                |error| {
                    super::RuntimeError::InvalidCImage(format!(
                        "invalid sealed legalization report: {error}"
                    ))
                },
            )?;
        }
        let manifest = CImageManifest {
            schema_version: "TRB_CIMG/1".into(),
            source_digest: String::new(),
            tensor_count: reader.header.tensors.len(),
            kernel_count: reader.header.kernels.len(),
        };
        Ok(CImageInspection {
            manifest,
            file_bytes,
            tensor_bytes,
            has_native_xdna: !reader.header.xdna_artifacts.is_empty()
                && reader
                    .header
                    .model_capabilities
                    .iter()
                    .any(|capability| capability == "native-xdna"),
            xdna_artifact_count: reader.header.xdna_artifacts.len(),
            model_manifest: reader.header.model_manifest,
            native_ternary_promotion: reader.header.native_ternary_promotion,
            joint_tiling_evidence: reader.header.joint_tiling_evidence,
        })
    }

    /// Load a `.cimage` file into memory, validating the format and any
    /// embedded promotion evidence.
    pub fn load(path: &Path) -> Result<Self, super::RuntimeError> {
        Self::load_with_promotion(path, true)
    }

    /// Load a freshly emitted artifact before backend promotion. This is
    /// intended only for the qualification workflow; production admission
    /// must use [`RuntimeModel::load`].
    pub fn load_for_validation(path: &Path) -> Result<Self, super::RuntimeError> {
        Self::load_with_promotion(path, false)
    }

    fn load_with_promotion(
        path: &Path,
        require_promotion: bool,
    ) -> Result<Self, super::RuntimeError> {
        let reader = CImageReader::open(path).map_err(super::RuntimeError::InvalidCImage)?;
        let validation = if require_promotion {
            reader.validate_payload_ranges()
        } else {
            reader.validate_payload_ranges_for_validation()
        };
        validation.map_err(super::RuntimeError::InvalidCImage)?;
        let mut file = File::open(path).map_err(|e| super::RuntimeError::FileNotFound(e.to_string()))?;
        let mapped_cimage = unsafe { memmap2::MmapOptions::new().map(&file) }
            .map_err(|e| super::RuntimeError::InvalidCImage(format!("mmap CImage: {e}")))?;
        let cimage_len = file
            .metadata()
            .map_err(|e| super::RuntimeError::InvalidCImage(format!("stat CImage: {e}")))?
            .len();
        let mut tensors = HashMap::new();
        let mut tensor_records = HashMap::new();
        let mut tensor_offsets = HashMap::new();
        let mut tensor_scales = HashMap::new();
        for (name, record) in &reader.header.tensors {
            let payload = read_region(&mut file, record.offset, record.size)?;
            tensor_offsets.insert(name.clone(), (record.offset, record.size));
            tensor_records.insert(name.clone(), record.clone());
            tensors.insert(name.clone(), payload);
            if let (Some(offset), Some(size)) = (record.scale_offset, record.scale_size) {
                tensor_scales.insert(name.clone(), read_region(&mut file, offset, size)?);
            }
        }
        let mut kernels = HashMap::new();
        let mut kernel_descriptors = HashMap::new();
        for (name, record) in &reader.header.kernels {
            kernels.insert(
                name.clone(),
                read_region(&mut file, record.offset, record.size)?,
            );
            if let Some(descriptor) = &record.descriptor {
                kernel_descriptors.insert(name.clone(), descriptor.clone());
            }
        }
        let mut ane_programs = HashMap::new();
        for (name, record) in &reader.header.ane_programs {
            ane_programs.insert(
                name.clone(),
                (
                    record.clone(),
                    read_region(&mut file, record.offset, record.size)?,
                ),
            );
        }
        let mut xdna_artifacts = HashMap::new();
        for name in reader.header.xdna_artifacts.keys() {
            let payload = reader.xdna_artifact(name).map_err(|error| {
                super::RuntimeError::InvalidCImage(format!("read XDNA artifact {name}: {error}"))
            })?;
            let artifact =
                prism_amd_npu_runtime::XdnaArtifact::decode(&payload).map_err(|error| {
                    super::RuntimeError::InvalidCImage(format!("decode XDNA artifact {name}: {error}"))
                })?;
            artifact.validate().map_err(|error| {
                super::RuntimeError::InvalidCImage(format!("validate XDNA artifact {name}: {error}"))
            })?;
            xdna_artifacts.insert(name.clone(), artifact);
        }
        let (mut execution_plan, mut realtime_execution_plan) = reader
            .header
            .execution_plan
            .as_deref()
            .and_then(|json| {
                if let Ok(plan) = serde_json::from_str::<ExecutionPlan>(json) {
                    return Some((Some(plan), None));
                }
                let manifest = serde_json::from_str::<prism_spatial_ir::target::KernelManifest>(json)
                    .ok()?;
                Some((manifest.batch_plan, manifest.realtime_plan))
            })
            .unwrap_or((None, None));
        let uop_capture = reader
            .header
            .execution_plan
            .as_deref()
            .filter(|plan| {
                serde_json::from_str::<serde_json::Value>(plan)
                    .ok()
                    .and_then(|value| value.get("capture_digest").cloned())
                    .is_some()
            })
            .map(|_| {
                reader.uop_capture().map_err(|error| {
                    super::RuntimeError::InvalidCImage(format!("invalid UOp capture: {error}"))
                })
            })
            .transpose()?;
        let uop_program = uop_capture
            .as_ref()
            .map(|capture| {
                UOpCompiledProgram::compile(capture.clone()).map_err(|error| {
                    super::RuntimeError::InvalidCImage(format!("compile embedded UOp capture: {error}"))
                })
            })
            .transpose()?;
        let mut uop_strategy_programs = HashMap::new();
        for strategy in reader.header.uop_captures.keys() {
            let program = UOpCompiledProgram::from_cimage_strategy(&reader, strategy).map_err(
                |error| {
                    super::RuntimeError::InvalidCImage(format!(
                        "load UOp strategy program {strategy:?}: {error}"
                    ))
                },
            )?;
            uop_strategy_programs.insert(strategy.clone(), program);
        }
        let uop_workload_evidence = reader
            .uop_workload_evidence()
            .map_err(|error| {
                super::RuntimeError::InvalidCImage(format!("invalid UOp workload evidence: {error}"))
            })?
            .to_vec();
        let tuning_receipt = reader.uop_tuning_receipt().map_err(|error| {
            super::RuntimeError::InvalidCImage(format!("invalid UOp tuning receipt: {error}"))
        })?;
        let uop_workload_evidence =
            if tuning_receipt.is_some_and(|receipt| receipt.production_ready) {
                uop_workload_evidence
            } else {
                // Legacy timing tables and explicitly synthetic receipts remain
                // inspectable in CImage metadata, but cannot silently become
                // runtime selection authority.
                Vec::new()
            };
        let normalize_plan = |mut plan: ExecutionPlan| {
            for window in &mut plan.residency_windows {
                // A zero value in compiler output means "the complete
                // model". Resolve that contract against the actual
                // mapped CImage size at load time so every stream event
                // carries an explicit whole-model residency requirement.
                if window.model_bytes == 0 {
                    window.model_bytes = cimage_len;
                }
            }
            plan
        };
        execution_plan = execution_plan.map(normalize_plan);
        realtime_execution_plan = realtime_execution_plan.map(normalize_plan);
        if let Some(plan) = execution_plan.as_ref().or(realtime_execution_plan.as_ref()) {
            for step in &plan.fused_steps {
                if let Some(model_id) = &step.model_id {
                    let Some(model_manifest) = reader.header.model_manifest.as_ref() else {
                        return Err(super::RuntimeError::InvalidCImage(format!(
                            "execution plan references namespaced model {model_id:?} but CImage has no model manifest"
                        )));
                    };
                    if model_manifest.get(model_id).is_none() {
                        return Err(super::RuntimeError::InvalidCImage(format!(
                            "execution plan references unknown model {model_id:?}"
                        )));
                    }
                }
            }
        }
        if let Some(model_manifest) = &reader.header.model_manifest {
            for model in model_manifest.models.values() {
                for projector in &model.projectors {
                    let record = reader
                        .header
                        .tensors
                        .get(&projector.tensor_name)
                        .ok_or_else(|| {
                            super::RuntimeError::InvalidCImage(format!(
                                "model {:?} projector tensor {:?} is missing",
                                model.id, projector.tensor_name
                            ))
                        })?;
                    if record.dim_m as usize != projector.output_dim
                        || record.dim_n as usize != projector.input_dim
                    {
                        return Err(super::RuntimeError::InvalidCImage(format!(
                            "model {:?} projector {:?} has shape {}x{}, expected {}x{}",
                            model.id,
                            projector.tensor_name,
                            record.dim_m,
                            record.dim_n,
                            projector.output_dim,
                            projector.input_dim
                        )));
                    }
                }
            }
        }
        Ok(Self {
            cimage_path: path.to_path_buf(),
            source_identity: reader.header.source_identity,
            source_catalog: reader.header.source_catalog,
            search_trace: reader
                .header
                .search_trace
                .as_deref()
                .map(serde_json::from_str)
                .transpose()
                .map_err(|e| super::RuntimeError::InvalidCImage(format!("invalid search trace: {e}")))?,
            legalization_report: reader
                .header
                .legalization_report
                .as_deref()
                .map(serde_json::from_str)
                .transpose()
                .map_err(|e| {
                    super::RuntimeError::InvalidCImage(format!("invalid legalization report: {e}"))
                })?,
            compilation_events: reader
                .header
                .compilation_events
                .as_deref()
                .map(serde_json::from_str)
                .transpose()
                .map_err(|e| {
                    super::RuntimeError::InvalidCImage(format!("invalid compilation events: {e}"))
                })?,
            manifest: CImageManifest {
                schema_version: "TRB_CIMG/1".into(),
                source_digest: String::new(),
                tensor_count: tensors.len(),
                kernel_count: kernels.len(),
            },
            tensors,
            tensor_records,
            tensor_scales,
            kernels,
            kernel_descriptors,
            uop_capture,
            uop_program,
            uop_strategy_programs,
            uop_workload_evidence,
            ane_programs,
            xdna_artifacts,
            kv_compression_policy: reader.header.kv_compression_policy,
            format_plan: reader
                .header
                .format_plan
                .as_deref()
                .map(serde_json::from_str)
                .transpose()
                .map_err(|e| super::RuntimeError::InvalidCImage(format!("invalid format plan: {e}")))?,
            model_manifest: reader.header.model_manifest,
            native_ternary_promotion: reader.header.native_ternary_promotion,
            joint_tiling_evidence: reader.header.joint_tiling_evidence,
            heterogeneous_workload_evidence: reader.header.heterogeneous_workload_evidence,
            execution_plan,
            realtime_execution_plan,
            tensor_offsets,
            mapped_cimage: Some(mapped_cimage),
        })
    }
}
