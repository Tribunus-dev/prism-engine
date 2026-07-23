//! .cimage: Hardware-native memory dump format for Apple Silicon.
//!
//! Every tensor payload starts on a 16 KB page boundary, enabling
//! zero-copy `mmap` directly into an IOSurface arena.  No parsing or
//! deserialization of the payload body at load time.
//!
//! Layout per file:
//! ┌─ Magic: "TRB_CIMG" (8B)
//! ├─ Header size: u64 LE (8B)
//! ├─ JSON header (variable, padded to 16 KB)
//! ├─ Padding to 16 KB boundary
//! ├─ Tensor 0 payload (16 KB aligned)
//! ├─ Padding to next 16 KB boundary
//! ├─ Tensor 1 payload
//! └─ ...

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

use prism_ecs_quantization::ternarization::promotion::{
    BackendPass, NativeTernaryPromotionEvidence,
};

/// Apple Silicon page size — required for zero-copy IOSurface mmap.
const PAGE_SIZE: u64 = 16384;
/// Header reservation: enough for ~500 tensor entries (typical 12B model).
const HEADER_PAGES: u64 = 256; // 4 MB for Qwen3.6 tensor, MoE, and vision metadata

/// Magic identifier for .cimage files.
const MAGIC: &[u8; 8] = b"TRB_CIMG";

// ── Header types ────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum TensorType {
    StandardFP16,
    Palettized4Bit,
    Blob,
    /// Ternary (Ternary158) — 2-bit ternary encoding per value.
    Ternary158,
    /// Ternary Tile640 — 640-weight tile with two-level scales and outliers.
    TernaryTile640,
    /// Binary — 1-bit binary encoding per value.
    Binary1,
    /// NF4 — 4-bit NormalFloat with per-group scale/bias.
    NF4,
    /// Int4 — symmetric 4-bit quantization with per-group scale.
    Int4,
    /// FP8 — 8-bit floating point.
    FP8,
    /// BFloat16 — truncated FP32 as 16-bit.
    Bf16,
    /// Int8 — symmetric 8-bit integer with per-tensor scale.
    Int8,
    /// NormalFloat8 — 8-bit normal float with per-tensor scale.
    Nf8,
}

/// Versioned physical ternary descriptor. Optional metadata remains backward
/// compatible with legacy CImages that only carried `TensorType`.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct TernaryDescriptor {
    pub version: u16,
    #[serde(deserialize_with = "deserialize_descriptor_string")]
    pub codec: String,
    pub group_size: u32,
    #[serde(deserialize_with = "deserialize_descriptor_string")]
    pub scale_encoding: String,
    #[serde(deserialize_with = "deserialize_descriptor_string")]
    pub layout: String,
    #[serde(deserialize_with = "deserialize_descriptor_string")]
    pub packing: String,
    pub kernel_variant: String,
    #[serde(deserialize_with = "deserialize_descriptor_string")]
    pub residual: String,
}

fn deserialize_descriptor_string<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Error;
    let value = serde_json::Value::deserialize(deserializer)?;
    match value {
        serde_json::Value::String(value) => Ok(value),
        serde_json::Value::Object(object) => {
            let (name, payload) = object
                .into_iter()
                .next()
                .ok_or_else(|| D::Error::custom("empty ternary descriptor enum"))?;
            if name == "Tile640" {
                let width = payload
                    .get("tile_width")
                    .and_then(serde_json::Value::as_u64)
                    .ok_or_else(|| D::Error::custom("Tile640 descriptor missing tile_width"))?;
                Ok(format!("Tile640:{width}"))
            } else {
                Ok(name)
            }
        }
        other => Err(D::Error::custom(format!(
            "expected ternary descriptor string or enum, got {other}"
        ))),
    }
}

impl TernaryDescriptor {
    pub const VERSION: u16 = 1;
    pub fn validate(&self) -> Result<(), String> {
        if self.version != Self::VERSION {
            return Err(format!(
                "unsupported ternary descriptor version {}",
                self.version
            ));
        }
        if self.codec.is_empty() || self.layout.is_empty() || self.packing.is_empty() {
            return Err("ternary descriptor has empty codec, layout, or packing".into());
        }
        if !matches!(
            self.residual.to_ascii_lowercase().as_str(),
            "none" | "bf16" | "fp16" | "dense"
        ) {
            return Err(format!(
                "unsupported ternary residual encoding '{}'",
                self.residual
            ));
        }
        Ok(())
    }

    pub fn legacy_for_type(tensor_type: &TensorType) -> Option<Self> {
        let (codec, layout, packing, group_size, scale_encoding, kernel_variant) = match tensor_type
        {
            TensorType::Ternary158 => (
                "Ternary158",
                "RowMajor",
                "TwoBitLE",
                1,
                "F32",
                "ternary158_gemv",
            ),
            TensorType::TernaryTile640 => (
                "TernaryTile640",
                "Tile640:640",
                "Base3U32LE",
                640,
                "BF16",
                "ternary_tile640_gemv",
            ),
            _ => return None,
        };
        Some(Self {
            version: Self::VERSION,
            codec: codec.into(),
            group_size,
            scale_encoding: scale_encoding.into(),
            layout: layout.into(),
            packing: packing.into(),
            kernel_variant: kernel_variant.into(),
            residual: "None".into(),
        })
    }

    /// Mark a ternary tensor as mixed precision: the native ternary payload
    /// covers the lossless subset while the BF16 residual is retained for
    /// values that do not pass the reference gate.
    pub fn with_bf16_residual(mut self) -> Self {
        self.residual = "BF16".into();
        self
    }

    pub fn has_mixed_precision_residual(&self) -> bool {
        self.residual.eq_ignore_ascii_case("bf16") || self.residual.eq_ignore_ascii_case("fp16")
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TensorRecord {
    pub tensor_type: TensorType,
    /// Byte offset from start of file (always 16 KB aligned).
    pub offset: u64,
    /// Total payload byte size.
    pub size: u64,
    /// Output dimension (number of rows).
    pub dim_m: u32,
    /// Input dimension (number of columns).
    pub dim_n: u32,
    /// Optional page-aligned per-group scale payload for native ternary data.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale_offset: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale_size: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ternary: Option<TernaryDescriptor>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub moe: Option<MoeTensorDescriptor>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vision: Option<VisionTensorDescriptor>,
    /// Semantic family used by behavior-preserving search and promotion.
    /// This is intentionally metadata-only; payload decoding remains governed
    /// by `tensor_type` and `ternary`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub semantic_family: Option<String>,
    /// Router tensors require margin/order validation in addition to RMSE.
    #[serde(default)]
    pub router_sensitive: bool,
}

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

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct MoeTensorDescriptor {
    pub layer: u32,
    pub expert: Option<u32>,
    /// Number of experts represented by a fused routed-expert bank. `None`
    /// for scalar router/shared tensors and individually addressed experts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expert_count: Option<u32>,
    pub role: String,
    pub component: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct VisionTensorDescriptor {
    pub component: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct KernelRecord {
    /// Byte offset from start of file (16 KB aligned).
    pub offset: u64,
    /// Total payload byte size of the compiled .metallib.
    pub size: u64,
    /// Kernel name (e.g. "ternary_tile640_gemv", "matmul_fp16").
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub descriptor: Option<prism_ecs_kernel::KernelDescriptor>,
}

/// A compiled stateless Core ML program embedded in the CImage.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AneProgramRecord {
    pub offset: u64,
    pub size: u64,
    pub name: String,
    pub activation_input: String,
    pub weights_input: String,
    pub output: String,
    pub input_dtype: String,
    pub output_dtype: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct XdnaArtifactRecord {
    pub offset: u64,
    pub size: u64,
    pub compiler_abi: String,
    pub generation: String,
}

/// A strategy-indexed serialized UOp capture. The selected legacy capture is
/// still stored in `execution_plan`; this table lets newer runtimes retain
/// additional executable alternatives without breaking old readers.
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct UOpCaptureRecord {
    pub capture_digest: String,
    pub capture: String,
    /// Evolutionary generation that produced this strategy, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub search_generation: Option<u32>,
}

/// Measured strategy selection for one concrete workload shape.  This is
/// evidence, not an unconditional promotion: the selected ID is accepted by
/// a runtime only when the corresponding capture is embedded and validates.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct UOpWorkloadEvidence {
    pub scenario: prism_spatial_ir::WorkloadScenario,
    pub strategies: Vec<String>,
    /// Capture digests in the same order as `strategies`.
    #[serde(default)]
    pub candidate_capture_digests: Vec<String>,
    pub measurements: Vec<prism_spatial_ir::FusionMeasurement>,
    pub selected_strategy: String,
}

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct CImageHeader {
    pub tensors: HashMap<String, TensorRecord>,
    /// Canonical source identity and catalog used to produce this artifact.
    /// Keeping these in the sealed header makes payload completeness auditable
    /// without reopening the original model source.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_identity: Option<prism_ecs_source::SourceIdentity>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_catalog: Option<prism_ecs_source::TensorCatalog>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub legalization_report: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compilation_events: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub search_trace: Option<String>,
    /// Model-neutral identity/configuration. The legacy Qwen field below is
    /// retained so existing artifacts remain readable during migration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_family: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_config_json: Option<String>,
    /// Verified architecture capabilities required by the runtime. Keeping
    /// this separate from raw model config prevents a generic importer from
    /// claiming support for specialized attention or routing operators.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub model_capabilities: Vec<String>,
    /// Evolutionary KV-cache compression winner serialized with the artifact.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_compression_policy: Option<String>,
    /// First-class model registry for namespaced multimodal CImages.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_manifest: Option<crate::model_manifest::MultiModelManifest>,
    /// Validated Qwen3.6 model configuration used to build this artifact.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub qwen36_config: Option<crate::qwen3_6_moe::Qwen36Config>,
    /// Optional execution plan (serialized JSON) for heterogeneous routing.
    /// Contains per-layer OperationRoute assignments and ANE fused islands.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution_plan: Option<String>,
    /// Additional strategy-specific UOp captures keyed by stable policy ID.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub uop_captures: HashMap<String, UOpCaptureRecord>,
    /// Backend measurement evidence and the selected candidate for each
    /// workload shape.  Optional for compatibility with older CImages.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub uop_workload_evidence: Vec<UOpWorkloadEvidence>,
    /// Structured provenance for UOp tuning. Legacy workload evidence may be
    /// readable without this field, but it is not production admission
    /// evidence unless this receipt explicitly says so.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uop_tuning_receipt: Option<crate::uop::UOpTuningReceipt>,
    /// Search selection provenance, including explicit synthetic fallback.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selection_receipt: Option<crate::search::SearchSelectionReceipt>,
    /// The selected per-tensor representation plan produced by search.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format_plan: Option<String>,
    /// Embedded compiled Metal kernel payloads (kernel name -> record).
    /// Each kernel payload is a compiled .metallib stored at a page-aligned
    /// offset, loadable at runtime via MTLLibrary::new_library_with_data.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub kernels: HashMap<String, KernelRecord>,
    /// Embedded compiled Core ML `.mlmodelc` payloads for stateless ANE paths.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub ane_programs: HashMap<String, AneProgramRecord>,
    /// Embedded native XDNA artifact envelopes, loadable by the AMD runtime.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub xdna_artifacts: HashMap<String, XdnaArtifactRecord>,
    /// Measured joint ANE/Metal tiling evidence retained for validation and
    /// recovery.  This is provenance only and does not certify promotion.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub joint_tiling_evidence: Option<crate::search::JointTilingEvidence>,
    /// Promotion receipt for native ternary payloads.  The runtime preserves
    /// this evidence so an artifact cannot be admitted after a failed or
    /// incomplete backend validation pass.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub native_ternary_promotion: Option<NativeTernaryPromotionEvidence>,
}

// ── Writer ──────────────────────────────────────────────────────────────

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

// ── Reader (runtime loader) ─────────────────────────────────────────────

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
        use std::io::Read;

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
    /// [`UniversalCImageWriter::add_uop_capture`].
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
        use std::io::{Read, Seek, SeekFrom};
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
        use std::io::{Read, Seek, SeekFrom};

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
        use std::io::{Read, Seek, SeekFrom};
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
// ── TensorRecord helpers ─────────────────────────────────────────────────

impl TensorRecord {
    /// Compute expected payload section sizes for TernaryTile640.
    ///
    /// Returns `(packed_words_size, page_scales_size, lane_scales_size, n_pages)`.
    ///
    /// Layout per the emission code in `compiler.rs`:
    ///   packed_words:   u32 × N_packed = n_pages × 32
    ///   page_scales:    u16 × N_pages
    ///   lane_scales:    i8  × N_pages × 32
    ///   n_outliers:     u32
    ///   outlier_rows:   u32 × n_outliers
    ///   outlier_cols:   u32 × n_outliers
    ///   outlier_vals:   u16 × n_outliers (BF16 bits)
    pub fn ternary_tile640_layout(&self) -> Result<(usize, usize, usize, usize), String> {
        if self.tensor_type != TensorType::TernaryTile640 {
            return Err("ternary_tile640_layout called on non-TernaryTile640 tensor".to_string());
        }
        let out_dim = self.dim_m as usize;
        let in_dim = self.dim_n as usize;
        let tile_width = 640_usize;
        let pages_per_row = (in_dim + tile_width - 1) / tile_width;
        let n_pages = out_dim * pages_per_row;

        let packed_words_size = n_pages * 32 * 4; // 32 u32 words per page × 4 bytes each
        let page_scales_size = n_pages * 2; // 1 u16 BF16 per page
        let lane_scales_size = n_pages * 32; // 32 i8 per page

        Ok((
            packed_words_size,
            page_scales_size,
            lane_scales_size,
            n_pages,
        ))
    }
}

/// Append a blob payload to an already-finalized .cimage file.
pub fn cimage_append_blob(
    path: &std::path::Path,
    name: &str,
    payload: &[u8],
) -> Result<(), String> {
    use std::io::{Seek, Write};
    let reader = CImageReader::open(path)?;
    let end_offset = reader
        .header
        .tensors
        .values()
        .map(|r| r.offset + r.size)
        .max()
        .unwrap_or(HEADER_PAGES * PAGE_SIZE);
    let aligned = end_offset.div_ceil(PAGE_SIZE) * PAGE_SIZE;
    let mut file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .map_err(|e| format!("open: {e}"))?;
    file.seek(SeekFrom::Start(aligned))
        .map_err(|e| format!("seek: {e}"))?;
    file.write_all(payload)
        .map_err(|e| format!("write blob: {e}"))?;
    let mut header = reader.header;
    header.tensors.insert(
        name.to_string(),
        TensorRecord {
            tensor_type: TensorType::Blob,
            offset: aligned,
            size: payload.len() as u64,
            dim_m: 0,
            dim_n: 0,
            scale_offset: None,
            scale_size: None,
            ternary: None,
            moe: None,
            vision: None,
            semantic_family: None,
            router_sensitive: false,
        },
    );
    let hdr_json = serde_json::to_string(&header).map_err(|e| format!("serialize: {e}"))?;
    file.seek(SeekFrom::Start(0))
        .map_err(|e| format!("seek: {e}"))?;
    file.write_all(MAGIC)
        .map_err(|e| format!("write magic: {e}"))?;
    let hdr_size = hdr_json.len() as u64;
    file.write_all(&hdr_size.to_le_bytes())
        .map_err(|e| format!("write size: {e}"))?;
    file.write_all(hdr_json.as_bytes())
        .map_err(|e| format!("write json: {e}"))?;
    file.flush().map_err(|e| format!("flush: {e}"))?;
    Ok(())
}

/// Read a named blob payload from a .cimage file.
///
/// Opens the file, locates the blob by name in the header, and returns
/// the raw bytes. Used to extract embedded metadata like compilation receipts.
pub fn cimage_read_blob(path: &Path, name: &str) -> Result<Vec<u8>, String> {
    use std::io::{Read, Seek, SeekFrom};

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

// ---------------------------------------------------------------------------
// UniversalCImageWriter — High-level CImage emission wrapper

// ---------------------------------------------------------------------------
// CImageError type
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum CImageError {
    #[error("CImage write failed: {0}")]
    WriteFailed(String),
    #[error("CImage finalize failed: {0}")]
    FinalizeFailed(String),
}

// ---------------------------------------------------------------------------
// CImageManifest type
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CImageManifest {
    pub schema_version: String,
    pub source_digest: String,
    pub tensor_count: usize,
    pub kernel_count: usize,
}

/// Payload entry for tensor data in CImage format.
#[derive(Debug, Clone)]
pub struct TensorPayloadEntry {
    pub name: String,
    pub payload: Vec<u8>,
    pub representation: String,
    pub effective_bpp: f32,
    pub dim_m: u32,
    pub dim_n: u32,
    pub tensor_type: TensorType,
}

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
        self.writer
            .append(&entry.name, &entry.payload, entry.dim_m, entry.dim_n, entry.tensor_type)
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
    /// only for the first phase of [`promote_cimage_after_replay`]; runtime
    /// admission still rejects the resulting file until promotion completes.
    pub fn finalize_unpromoted(self) -> Result<(), String> {
        self.writer.finalize()
    }
}

/// Emit a minimal CImage containing one stateless int8 ANE program. Full
/// compiler pipelines can use [`UniversalCImageWriter::add_ane_program`]
/// while this entry point is useful for tile-specialized artifacts produced
/// by evolutionary-search candidates.
pub fn emit_int8_ane_program(
    output_path: &Path,
    name: &str,
    modelc_payload: &[u8],
    activation_input: &str,
    weights_input: &str,
    output: &str,
) -> Result<(), String> {
    let mut writer = UniversalCImageWriter::new(output_path);
    writer.add_ane_program(
        name,
        modelc_payload,
        activation_input,
        weights_input,
        output,
    )?;
    writer.finalize()
}

/// Complete native ternary promotion after the artifact has been emitted.
///
/// The first pass validates the unpromoted artifact's ranges, descriptors, MoE
/// annotations, and payload digests. Only after that succeeds is the replay
/// bit attached to the fixed-size header. The rewritten artifact is then
/// reopened and replay-validated, so the returned evidence refers to the
/// bytes actually present on disk.
pub fn promote_cimage_after_replay(
    path: &Path,
    mut evidence: NativeTernaryPromotionEvidence,
) -> Result<NativeTernaryPromotionEvidence, String> {
    let mut reader = CImageReader::open(path)?;
    reader.validate_payload_ranges_for_validation()?;
    evidence.cimage_replay = BackendPass::passed();
    if !evidence.eligible() {
        return Err(format!(
            "cannot promote CImage before replay: {}",
            evidence
                .reject_reason()
                .unwrap_or_else(|| "promotion evidence is incomplete".to_string())
        ));
    }
    reader.header.native_ternary_promotion = Some(evidence.clone());
    let header_json = serde_json::to_vec(&reader.header)
        .map_err(|e| format!("serialize promotion header: {e}"))?;
    let reserved = (HEADER_PAGES * PAGE_SIZE) as usize;
    if 16usize.saturating_add(header_json.len()) > reserved {
        return Err("promotion header exceeds reserved CImage header space".into());
    }
    let mut file = OpenOptions::new()
        .write(true)
        .open(path)
        .map_err(|e| format!("open CImage for promotion: {e}"))?;
    file.seek(SeekFrom::Start(0))
        .map_err(|e| format!("seek CImage promotion header: {e}"))?;
    file.write_all(MAGIC)
        .map_err(|e| format!("write CImage promotion magic: {e}"))?;
    file.write_all(&(header_json.len() as u64).to_le_bytes())
        .map_err(|e| format!("write CImage promotion header size: {e}"))?;
    file.write_all(&header_json)
        .map_err(|e| format!("write CImage promotion header: {e}"))?;
    file.flush()
        .map_err(|e| format!("flush CImage promotion header: {e}"))?;

    let promoted = CImageReader::open(path)?;
    promoted.validate_payload_ranges()?;
    Ok(evidence)
}

/// Promotion entry point for search callers that have structured reference
/// evidence. It converts the measured behavioral gates into the promotion
/// receipt instead of allowing a caller to hand-author a passing boolean.
pub fn promote_cimage_with_behavioral_evidence(
    path: &Path,
    mut evidence: NativeTernaryPromotionEvidence,
    behavioral: prism_ecs_ir::evolution::TernaryObjectiveEvidence,
    limits: &prism_ecs_ir::evolution::TernaryAdmissionLimits,
) -> Result<NativeTernaryPromotionEvidence, String> {
    let behavioral_passed = behavioral.behavioral_passes(limits);
    evidence.behavioral_reference = BackendPass {
        attempted: true,
        passed: behavioral_passed,
    };
    evidence.activation_error = Some(behavioral.activation_error);
    evidence.logit_divergence = Some(behavioral.logit_divergence);
    evidence.task_loss = Some(behavioral.task_loss);
    evidence.router_agreement = Some(behavioral.router_agreement);
    evidence.router_margin_error = Some(behavioral.router_margin_error);
    evidence.logit_cross_entropy = Some(behavioral.logit_cross_entropy);
    evidence.generation_loss = Some(behavioral.generation_loss);
    evidence.expert_balance_error = Some(behavioral.expert_balance_error);
    if let Some(reason) = evidence.behavioral_reject_reason(limits) {
        return Err(format!("cannot promote CImage: {reason}"));
    }
    promote_cimage_after_replay(path, evidence)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_model_uses_generic_identity_and_native_replay_boundary() {
        let path = std::env::temp_dir().join(format!(
            "prism_compile_dense_generic_{}.cimage",
            std::process::id()
        ));
        let mut writer = UniversalCImageWriter::new(&path);
        writer
            .set_model_identity("llama", &serde_json::json!({"num_hidden_layers": 1}))
            .unwrap();
        writer
            .add_native_ternary_payload_with_scales(
                "model.layers.0.self_attn.q_proj.weight",
                &[0, 1, 2, 0],
                &[0, 0, 128, 63],
                1,
                4,
                TensorType::Ternary158,
                TernaryDescriptor::legacy_for_type(&TensorType::Ternary158).unwrap(),
            )
            .unwrap();
        writer.finalize_unpromoted().unwrap();
        let reader = CImageReader::open(&path).unwrap();
        reader.validate_payload_ranges_for_validation().unwrap();
        assert_eq!(reader.header.model_family.as_deref(), Some("llama"));
        assert!(reader
            .tensor("model.layers.0.self_attn.q_proj.weight")
            .is_some());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn uop_capture_compiles_and_round_trips_through_cimage() {
        let path = std::env::temp_dir().join(format!(
            "prism_compile_uop_capture_{}.cimage",
            std::process::id()
        ));
        let mut graph = prism_spatial_ir::TinyGraph::default();
        let input = graph.add(
            prism_spatial_ir::UOpKind::Input { name: "x".into() },
            vec![],
            vec![2],
        );
        let relu = graph.add(prism_spatial_ir::UOpKind::Relu, vec![input], vec![2]);
        graph.add(
            prism_spatial_ir::UOpKind::Output { name: "y".into() },
            vec![relu],
            vec![2],
        );
        let capture = graph
            .lower(prism_spatial_ir::LoweringTarget::Portable)
            .unwrap();
        let mut writer = UniversalCImageWriter::new(&path);
        writer.add_uop_capture(&capture).unwrap();
        writer
            .add_uop_strategy_captures(&[("standard_fused".into(), capture.clone())])
            .unwrap();
        writer.finalize().unwrap();
        let reader = CImageReader::open(&path).unwrap();
        assert_eq!(reader.header.kernels.len(), 2);
        let execution_plan = reader.header.execution_plan.as_ref().unwrap();
        let envelope: serde_json::Value = serde_json::from_str(execution_plan).unwrap();
        assert_eq!(envelope["capture_digest"], capture.digest());
        assert_eq!(reader.uop_capture().unwrap().digest(), capture.digest());
        assert_eq!(
            reader
                .uop_capture_for_strategy("standard_fused")
                .unwrap()
                .digest(),
            capture.digest()
        );
        let strategy_program =
            crate::UOpCompiledProgram::from_cimage_strategy(&reader, "standard_fused").unwrap();
        assert_eq!(strategy_program.artifacts.len(), capture.kernels.len());
        let loaded = crate::UOpCompiledProgram::from_cimage(&reader).unwrap();
        assert_eq!(loaded.artifacts.len(), capture.kernels.len());
        let runtime = crate::runtime::RuntimeModel::load_for_validation(&path).unwrap();
        assert_eq!(runtime.uop_capture().unwrap().digest(), capture.digest());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn uop_strategy_candidate_set_compiles_and_seals_all_strategies() {
        let path = std::env::temp_dir().join(format!(
            "prism_compile_uop_candidate_set_{}.cimage",
            std::process::id()
        ));
        let mut graph = prism_spatial_ir::TinyGraph::default();
        let input = graph.add(
            prism_spatial_ir::UOpKind::Input { name: "x".into() },
            vec![],
            vec![4],
        );
        let value = graph.add(prism_spatial_ir::UOpKind::Relu, vec![input], vec![4]);
        graph.add(
            prism_spatial_ir::UOpKind::Output { name: "y".into() },
            vec![value],
            vec![4],
        );
        let strategies = [
            prism_spatial_ir::FusionStrategy::StandardFused,
            prism_spatial_ir::FusionStrategy::InterleavedFused { stages: vec![] },
            prism_spatial_ir::FusionStrategy::PerOperation,
            prism_spatial_ir::FusionStrategy::PersistentMegakernel {
                search_generation: 0,
            },
        ];
        let mut writer = UniversalCImageWriter::new(&path);
        writer
            .add_uop_strategy_candidate_set(
                &graph,
                prism_spatial_ir::LoweringTarget::Portable,
                &strategies,
            )
            .unwrap();
        writer.finalize().unwrap();
        let reader = CImageReader::open(&path).unwrap();
        assert_eq!(reader.header.uop_captures.len(), strategies.len());
        for strategy in strategies {
            assert!(reader
                .uop_capture_for_strategy(strategy.stable_id())
                .is_ok());
        }
        assert_eq!(
            reader.uop_strategy_search_generation("persistent_megakernel"),
            Some(0)
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn uop_workload_evidence_round_trips_and_requires_embedded_winner() {
        let path = std::env::temp_dir().join(format!(
            "prism_compile_uop_workload_evidence_{}.cimage",
            std::process::id()
        ));
        let mut graph = prism_spatial_ir::TinyGraph::default();
        let input = graph.add(
            prism_spatial_ir::UOpKind::Input { name: "x".into() },
            vec![],
            vec![4],
        );
        let value = graph.add(prism_spatial_ir::UOpKind::Relu, vec![input], vec![4]);
        graph.add(
            prism_spatial_ir::UOpKind::Output { name: "y".into() },
            vec![value],
            vec![4],
        );
        let strategies = [
            prism_spatial_ir::FusionStrategy::StandardFused,
            prism_spatial_ir::FusionStrategy::InterleavedFused { stages: vec![] },
            prism_spatial_ir::FusionStrategy::PerOperation,
            prism_spatial_ir::FusionStrategy::PersistentMegakernel {
                search_generation: 3,
            },
        ];
        let mut writer = UniversalCImageWriter::new(&path);
        writer
            .add_uop_strategy_candidate_set(
                &graph,
                prism_spatial_ir::LoweringTarget::Portable,
                &strategies,
            )
            .unwrap();
        let scenario = prism_spatial_ir::WorkloadScenario {
            realtime: false,
            batch_size: 4,
            sequence_length: 16,
        };
        writer
            .add_uop_workload_evidence(
                &[crate::uop::UOpWorkloadMeasurement {
                    scenario,
                    measurements: vec![
                        prism_spatial_ir::FusionMeasurement {
                            candidate_index: 0,
                            latency_ns: 40,
                            materialized_bytes: 10,
                        },
                        prism_spatial_ir::FusionMeasurement {
                            candidate_index: 1,
                            latency_ns: 30,
                            materialized_bytes: 10,
                        },
                        prism_spatial_ir::FusionMeasurement {
                            candidate_index: 2,
                            latency_ns: 50,
                            materialized_bytes: 10,
                        },
                        prism_spatial_ir::FusionMeasurement {
                            candidate_index: 3,
                            latency_ns: 20,
                            materialized_bytes: 10,
                        },
                    ],
                }],
                &strategies,
            )
            .unwrap();
        writer.set_uop_tuning_receipt(
            crate::uop::UOpTuningReceipt::explicit_fallback(
                "graph-digest",
                prism_spatial_ir::LoweringTarget::Portable,
                "test fallback; legacy workload timings are diagnostic only",
            )
            .unwrap(),
        );
        writer.finalize().unwrap();
        let mut reader = CImageReader::open(&path).unwrap();
        let tuning = reader.uop_tuning_receipt().unwrap().unwrap();
        assert!(!tuning.production_ready);
        assert_eq!(tuning.source, crate::uop::UOpMeasurementSource::SyntheticFallback);
        let evidence = reader.uop_workload_evidence().unwrap();
        assert_eq!(evidence.len(), 1);
        assert_eq!(evidence[0].scenario, scenario);
        assert_eq!(evidence[0].selected_strategy, "persistent_megakernel");
        assert_eq!(
            reader
                .uop_workload_evidence_for(scenario)
                .unwrap()
                .unwrap()
                .selected_strategy,
            "persistent_megakernel"
        );
        assert!(reader
            .uop_workload_evidence_for(prism_spatial_ir::WorkloadScenario {
                realtime: true,
                batch_size: 1,
                sequence_length: 1,
            })
            .unwrap()
            .is_none());
        reader.header.uop_workload_evidence[0].selected_strategy = "standard_fused".into();
        assert!(reader.uop_workload_evidence().is_err());
        reader.header.uop_workload_evidence[0].selected_strategy = "persistent_megakernel".into();
        reader.header.uop_workload_evidence[0].measurements[0].latency_ns = 0;
        assert!(reader.uop_workload_evidence().is_err());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn layer_norm_uop_capture_round_trips_through_cimage() {
        let path = std::env::temp_dir().join(format!(
            "prism_compile_layer_norm_{}.cimage",
            std::process::id()
        ));
        let mut graph = prism_spatial_ir::TinyGraph::default();
        let x = graph.add(
            prism_spatial_ir::UOpKind::Input { name: "x".into() },
            vec![],
            vec![1, 2],
        );
        let weight = graph.add(
            prism_spatial_ir::UOpKind::Input {
                name: "weight".into(),
            },
            vec![],
            vec![2],
        );
        let bias = graph.add(
            prism_spatial_ir::UOpKind::Input {
                name: "bias".into(),
            },
            vec![],
            vec![2],
        );
        let norm = graph.add(
            prism_spatial_ir::UOpKind::LayerNorm {
                rows: 1,
                features: 2,
                epsilon: 1e-5,
            },
            vec![x, weight, bias],
            vec![1, 2],
        );
        graph.add(
            prism_spatial_ir::UOpKind::Output { name: "out".into() },
            vec![norm],
            vec![1, 2],
        );
        let capture = graph
            .lower(prism_spatial_ir::LoweringTarget::Portable)
            .unwrap();
        let mut writer = UniversalCImageWriter::new(&path);
        writer.add_uop_capture(&capture).unwrap();
        writer.finalize().unwrap();
        let reader = CImageReader::open(&path).unwrap();
        let loaded = crate::UOpCompiledProgram::from_cimage(&reader).unwrap();
        assert_eq!(loaded.capture.digest(), capture.digest());
        assert_eq!(
            loaded.artifacts[0].manifest.kernels[0]
                .binding_signature
                .len(),
            4
        );
        let runtime = crate::runtime::RuntimeModel::load_for_validation(&path).unwrap();
        assert_eq!(runtime.uop_capture().unwrap().digest(), capture.digest());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn gather_uop_capture_round_trips_through_cimage() {
        let path = std::env::temp_dir().join(format!(
            "prism_compile_gather_{}.cimage",
            std::process::id()
        ));
        let mut graph = prism_spatial_ir::TinyGraph::default();
        let weight = graph.add(
            prism_spatial_ir::UOpKind::Input {
                name: "weight".into(),
            },
            vec![],
            vec![3, 2],
        );
        let indices = graph.add(
            prism_spatial_ir::UOpKind::Input {
                name: "indices".into(),
            },
            vec![],
            vec![2],
        );
        let gather = graph.add(
            prism_spatial_ir::UOpKind::Gather {
                rows: 2,
                vocab: 3,
                features: 2,
            },
            vec![weight, indices],
            vec![2, 2],
        );
        graph.add(
            prism_spatial_ir::UOpKind::Output { name: "out".into() },
            vec![gather],
            vec![2, 2],
        );
        let capture = graph
            .lower(prism_spatial_ir::LoweringTarget::Portable)
            .unwrap();
        let mut writer = UniversalCImageWriter::new(&path);
        writer.add_uop_capture(&capture).unwrap();
        writer.finalize().unwrap();
        let reader = CImageReader::open(&path).unwrap();
        let loaded = crate::UOpCompiledProgram::from_cimage(&reader).unwrap();
        assert_eq!(loaded.capture.digest(), capture.digest());
        assert_eq!(
            loaded.artifacts[0].manifest.kernels[0]
                .binding_signature
                .len(),
            3
        );
        let runtime = crate::runtime::RuntimeModel::load_for_validation(&path).unwrap();
        assert_eq!(runtime.uop_capture().unwrap().digest(), capture.digest());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn scatter_uop_capture_round_trips_through_cimage() {
        let path = std::env::temp_dir().join(format!(
            "prism_compile_scatter_{}.cimage",
            std::process::id()
        ));
        let mut graph = prism_spatial_ir::TinyGraph::default();
        let base = graph.add(
            prism_spatial_ir::UOpKind::Input {
                name: "base".into(),
            },
            vec![],
            vec![3, 2],
        );
        let indices = graph.add(
            prism_spatial_ir::UOpKind::Input {
                name: "indices".into(),
            },
            vec![],
            vec![2],
        );
        let updates = graph.add(
            prism_spatial_ir::UOpKind::Input {
                name: "updates".into(),
            },
            vec![],
            vec![2, 2],
        );
        let scatter = graph.add(
            prism_spatial_ir::UOpKind::Scatter {
                rows: 3,
                updates: 2,
                features: 2,
            },
            vec![base, indices, updates],
            vec![3, 2],
        );
        graph.add(
            prism_spatial_ir::UOpKind::Output { name: "out".into() },
            vec![scatter],
            vec![3, 2],
        );
        let capture = graph
            .lower(prism_spatial_ir::LoweringTarget::Portable)
            .unwrap();
        let mut writer = UniversalCImageWriter::new(&path);
        writer.add_uop_capture(&capture).unwrap();
        writer.finalize().unwrap();
        let reader = CImageReader::open(&path).unwrap();
        let loaded = crate::UOpCompiledProgram::from_cimage(&reader).unwrap();
        assert_eq!(loaded.capture.digest(), capture.digest());
        assert_eq!(
            loaded.artifacts[0].manifest.kernels[0]
                .binding_signature
                .len(),
            4
        );
        let runtime = crate::runtime::RuntimeModel::load_for_validation(&path).unwrap();
        assert_eq!(runtime.uop_capture().unwrap().digest(), capture.digest());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn spatial_graph_capture_round_trips_through_cimage_runtime() {
        use prism_ecs_ir::cimage_types::TensorShape;
        use prism_spatial_ir::graph::{
            ComputeIntensity, ComputeKind, EdgeDirection, MemoryKind, MemoryRegion, ShapeContract,
            SpatialEdge, SpatialEdgeId, SpatialNode,
        };
        let path = std::env::temp_dir().join(format!(
            "prism_compile_spatial_graph_{}.cimage",
            std::process::id()
        ));
        let mut graph = prism_spatial_ir::SpatialGraph::new();
        let a = graph.add_node(SpatialNode::Memory {
            id: prism_spatial_ir::SpatialNodeId(1),
            kind: MemoryKind::InputBuffer,
            region: MemoryRegion {
                shape: TensorShape { dims: vec![2, 3] },
                element_size: 4,
                strides: vec![],
            },
        });
        let b = graph.add_node(SpatialNode::Memory {
            id: prism_spatial_ir::SpatialNodeId(2),
            kind: MemoryKind::InputBuffer,
            region: MemoryRegion {
                shape: TensorShape { dims: vec![3, 2] },
                element_size: 4,
                strides: vec![],
            },
        });
        let matmul = graph.add_node(SpatialNode::Compute {
            id: prism_spatial_ir::SpatialNodeId(3),
            kind: ComputeKind::MatMul,
            shape: ShapeContract::new(
                vec![
                    TensorShape { dims: vec![2, 3] },
                    TensorShape { dims: vec![3, 2] },
                ],
                vec![TensorShape { dims: vec![2, 2] }],
            ),
            intensity: ComputeIntensity::ComputeBound,
        });
        graph.add_edge(SpatialEdge {
            id: SpatialEdgeId(1),
            source: a,
            sink: matmul,
            direction: EdgeDirection::Forward,
            source_output_idx: 0,
            sink_input_idx: 0,
            shape: Some(TensorShape { dims: vec![2, 3] }),
        });
        graph.add_edge(SpatialEdge {
            id: SpatialEdgeId(2),
            source: b,
            sink: matmul,
            direction: EdgeDirection::Forward,
            source_output_idx: 0,
            sink_input_idx: 1,
            shape: Some(TensorShape { dims: vec![3, 2] }),
        });
        let mut writer = UniversalCImageWriter::new(&path);
        writer
            .add_spatial_graph(&graph, prism_spatial_ir::LoweringTarget::Portable)
            .unwrap();
        writer.finalize().unwrap();
        let reader = CImageReader::open(&path).unwrap();
        let capture = reader.uop_capture().unwrap();
        assert_eq!(capture.graph_op_count, 4);
        let runtime = crate::runtime::RuntimeModel::load_for_validation(&path).unwrap();
        assert_eq!(runtime.uop_capture().unwrap().digest(), capture.digest());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn native_tensor_records_preserve_semantic_family() {
        let path = std::env::temp_dir().join(format!(
            "prism_compile_semantic_family_{}.cimage",
            std::process::id()
        ));
        let mut writer = CImageWriter::new(&path).expect("create CImage");
        writer
            .append_native_ternary_with_scales(
                "layers.0.router.weight",
                &[0, 1, 2, 0],
                &[0, 0, 128, 63],
                1,
                4,
                TensorType::Ternary158,
                TernaryDescriptor::legacy_for_type(&TensorType::Ternary158).unwrap(),
            )
            .expect("append router payload");
        writer.finalize().expect("finalize CImage");
        let reader = CImageReader::open(&path).expect("open CImage");
        let record = reader.tensor("layers.0.router.weight").unwrap();
        assert_eq!(record.semantic_family.as_deref(), Some("router"));
        assert!(record.router_sensitive);
        reader.validate_payload_ranges_for_validation().unwrap();
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn runtime_rejects_native_ternary_without_promotion_receipt() {
        let path = std::env::temp_dir().join(format!(
            "prism_compile_native_ternary_{}.cimage",
            std::process::id()
        ));
        let mut writer = CImageWriter::new(&path).expect("create CImage");
        writer
            .append_native_ternary_with_scales(
                "weights",
                &[0, 1, 2, 0],
                &[0, 0, 128, 63],
                1,
                4,
                TensorType::Ternary158,
                TernaryDescriptor::legacy_for_type(&TensorType::Ternary158).unwrap(),
            )
            .expect("append ternary payload");
        writer.finalize().expect("finalize CImage");

        let reader = CImageReader::open(&path).expect("open CImage");
        let error = reader
            .validate_native_ternary_promotion()
            .expect_err("missing receipt must reject native ternary");
        assert!(error.contains("missing promotion evidence"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn post_emission_promotion_reopens_and_validates_artifact() {
        let path = std::env::temp_dir().join(format!(
            "prism_compile_post_emission_promotion_{}.cimage",
            std::process::id()
        ));
        let mut writer = CImageWriter::new(&path).expect("create CImage");
        writer
            .append_native_ternary_with_scales(
                "weights",
                &[0, 1, 2, 0],
                &[0, 0, 128, 63],
                1,
                4,
                TensorType::Ternary158,
                TernaryDescriptor::legacy_for_type(&TensorType::Ternary158).unwrap(),
            )
            .expect("append ternary payload");
        writer.finalize().expect("write unpromoted artifact");

        let evidence = NativeTernaryPromotionEvidence {
            cpu_canary: BackendPass::passed(),
            accelerate_reconstruction: BackendPass::passed(),
            metal_packed: BackendPass::passed(),
            ane_static: BackendPass::unavailable(),
            cimage_replay: BackendPass::unavailable(),
            behavioral_reference: BackendPass::passed(),
            activation_error: Some(0.0),
            logit_divergence: Some(0.0),
            task_loss: Some(0.0),
            router_agreement: Some(1.0),
            router_margin_error: Some(0.0),
            logit_cross_entropy: Some(0.0),
            generation_loss: Some(0.0),
            expert_balance_error: Some(0.0),
            ane_selected: false,
            packed_abi_digest: "abi-digest".into(),
            reference_digest: "reference-digest".into(),
        };
        let promoted = promote_cimage_after_replay(&path, evidence).expect("promote CImage");
        assert!(promoted.cimage_replay.passed);
        let reader = CImageReader::open(&path).expect("reopen promoted CImage");
        reader.validate_payload_ranges().expect("validate replay");
        let _ = std::fs::remove_file(path);
    }

    fn write_unpromoted_native_test_image(path: &std::path::Path) {
        let mut writer = CImageWriter::new(path).expect("create CImage");
        writer
            .append_native_ternary_with_scales(
                "weights",
                &[0, 1, 2, 0],
                &[0, 0, 128, 63],
                1,
                4,
                TensorType::Ternary158,
                TernaryDescriptor::legacy_for_type(&TensorType::Ternary158).unwrap(),
            )
            .expect("append ternary payload");
        writer.finalize().expect("write unpromoted CImage");
    }

    fn promotion_evidence_without_behavioral_measurements() -> NativeTernaryPromotionEvidence {
        NativeTernaryPromotionEvidence {
            cpu_canary: BackendPass::passed(),
            accelerate_reconstruction: BackendPass::passed(),
            metal_packed: BackendPass::passed(),
            ane_static: BackendPass::unavailable(),
            cimage_replay: BackendPass::unavailable(),
            behavioral_reference: BackendPass::passed(),
            activation_error: None,
            logit_divergence: None,
            task_loss: None,
            router_agreement: None,
            router_margin_error: None,
            logit_cross_entropy: None,
            generation_loss: None,
            expert_balance_error: None,
            ane_selected: false,
            packed_abi_digest: "abi".into(),
            reference_digest: "reference".into(),
        }
    }

    fn measured_behavioral_test_evidence() -> prism_ecs_ir::evolution::TernaryObjectiveEvidence {
        prism_ecs_ir::evolution::TernaryObjectiveEvidence {
            quality: 0.9,
            activation_error: 0.012,
            logit_divergence: 0.023,
            task_loss: 0.034,
            router_agreement: 0.97,
            router_margin_error: 0.014,
            logit_cross_entropy: 0.025,
            generation_loss: 0.036,
            expert_balance_error: 0.017,
            latency_ms: 2.0,
            native_ternary_fraction: 1.0,
            energy: 1.0,
            ..prism_ecs_ir::evolution::TernaryObjectiveEvidence::missing()
        }
    }

    #[test]
    fn promotion_rejects_missing_behavioral_measurement() {
        let path = std::env::temp_dir().join(format!(
            "prism_compile_missing_behavioral_{}_{}.cimage",
            std::process::id(),
            line!()
        ));
        write_unpromoted_native_test_image(&path);
        let error = promote_cimage_after_replay(&path, promotion_evidence_without_behavioral_measurements())
            .expect_err("missing behavioral measurements must reject promotion");
        assert!(error.contains("activation_error measurement is missing"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn promotion_rejects_failed_behavioral_threshold() {
        let path = std::env::temp_dir().join(format!(
            "prism_compile_failed_behavioral_{}_{}.cimage",
            std::process::id(),
            line!()
        ));
        write_unpromoted_native_test_image(&path);
        let mut behavioral = measured_behavioral_test_evidence();
        behavioral.activation_error = 0.2;
        let error = promote_cimage_with_behavioral_evidence(
            &path,
            promotion_evidence_without_behavioral_measurements(),
            behavioral,
            &prism_ecs_ir::evolution::TernaryAdmissionLimits::default(),
        )
        .expect_err("failed behavioral threshold must reject promotion");
        assert!(error.contains("activation_error 0.2 exceeds maximum"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn promotion_preserves_measured_behavioral_provenance_in_cimage() {
        let path = std::env::temp_dir().join(format!(
            "prism_compile_behavioral_provenance_{}_{}.cimage",
            std::process::id(),
            line!()
        ));
        write_unpromoted_native_test_image(&path);
        let promoted = promote_cimage_with_behavioral_evidence(
            &path,
            promotion_evidence_without_behavioral_measurements(),
            measured_behavioral_test_evidence(),
            &prism_ecs_ir::evolution::TernaryAdmissionLimits::default(),
        )
        .expect("measured behavioral evidence should promote");
        assert_eq!(promoted.activation_error, Some(0.012));
        assert_eq!(promoted.logit_divergence, Some(0.023));
        assert_eq!(promoted.task_loss, Some(0.034));
        assert_eq!(promoted.expert_balance_error, Some(0.017));

        let reader = CImageReader::open(&path).expect("reopen promoted CImage");
        let stored = reader
            .header
            .native_ternary_promotion
            .as_ref()
            .expect("promotion evidence should be stored");
        assert_eq!(stored, &promoted);
        reader.validate_payload_ranges().expect("CImage admission");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn qwen36_vision_descriptor_replays_and_validates() {
        let path = std::env::temp_dir().join(format!(
            "prism_compile_qwen36_vision_{}.cimage",
            std::process::id()
        ));
        let config = crate::qwen3_6_moe::Qwen36Config::from_json_str(
            r#"{"model_type":"qwen3_6_moe","hidden_size":8,"num_hidden_layers":1,"num_experts":2,"num_experts_per_tok":1,"vocab_size":16,"vision_config":{"depth":2}}"#,
        ).unwrap();
        let mut writer = CImageWriter::new(&path).unwrap();
        writer.set_qwen36_config(config).unwrap();
        writer
            .append(
                "vision.patch_embed.weight",
                &[1, 2, 3, 4],
                1,
                4,
                TensorType::Blob,
            )
            .unwrap();
        writer
            .set_vision_tensor(
                "vision.patch_embed.weight",
                VisionTensorDescriptor {
                    component: "patch_embed".into(),
                },
            )
            .unwrap();
        writer
            .append(
                "model.layers.0.mlp.router.weight",
                &[1, 2, 3, 4],
                1,
                4,
                TensorType::Blob,
            )
            .unwrap();
        writer
            .set_moe_tensor(
                "model.layers.0.mlp.router.weight",
                MoeTensorDescriptor {
                    layer: 0,
                    expert: None,
                    expert_count: None,
                    role: "router".into(),
                    component: Some("router".into()),
                },
            )
            .unwrap();
        writer
            .append(
                "model.layers.0.mlp.experts.gate_up_proj",
                &[1, 2, 3, 4],
                2,
                2,
                TensorType::Blob,
            )
            .unwrap();
        writer
            .set_moe_tensor(
                "model.layers.0.mlp.experts.gate_up_proj",
                MoeTensorDescriptor {
                    layer: 0,
                    expert: None,
                    expert_count: Some(2),
                    role: "routed_expert_bank".into(),
                    component: Some("gate_up_proj".into()),
                },
            )
            .unwrap();
        writer.finalize().unwrap();
        let reader = CImageReader::open(&path).unwrap();
        reader.validate_qwen36_tensor_contract().unwrap();
        assert_eq!(
            reader
                .tensor("vision.patch_embed.weight")
                .unwrap()
                .vision
                .as_ref()
                .unwrap()
                .component,
            "patch_embed"
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn embedded_xdna_artifact_round_trips_by_range() {
        let path = std::env::temp_dir().join(format!(
            "prism_compile_xdna_artifact_{}.cimage",
            std::process::id()
        ));
        let mut writer = CImageWriter::new(&path).unwrap();
        writer
            .add_xdna_artifact("prefill", b"PXDA-native", "prism-xdna-v1", "Aie2p")
            .unwrap();
        writer.finalize().unwrap();
        let reader = CImageReader::open(&path).unwrap();
        assert_eq!(reader.xdna_artifact("prefill").unwrap(), b"PXDA-native");
        assert_eq!(reader.header.xdna_artifacts["prefill"].generation, "Aie2p");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn format_plan_round_trips_through_runtime_validation() {
        let path = std::env::temp_dir().join(format!(
            "prism_compile_format_plan_{}.cimage",
            std::process::id()
        ));
        let mut writer = CImageWriter::new(&path).unwrap();
        let plan = prism_ecs_ir::evolution::compile_plan::FormatPlan::new();
        writer
            .set_format_plan(serde_json::to_string(&plan).unwrap())
            .unwrap();
        writer.finalize().unwrap();
        let reader = CImageReader::open(&path).unwrap();
        reader.validate_payload_ranges_for_validation().unwrap();
        assert!(reader.header.format_plan.is_some());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn persistent_kv_requires_and_round_trips_evolutionary_policy() {
        let path = std::env::temp_dir().join(format!(
            "prism_compile_kv_policy_{}.cimage",
            std::process::id()
        ));
        let mut writer = UniversalCImageWriter::new(&path);
        writer.set_model_capabilities(["persistent-kv"]);
        let candidate = prism_ecs_quantization::kv_search::KvCompressionCandidate {
            mode: prism_ecs_quantization::kv_search::AsymmetricQuantModeId::KeyLightValueHeavy,
            key_bits: 2,
            value_bits: 4,
            group_size: 64,
            qjl_bits: 0,
        };
        writer.set_kv_compression_policy(&candidate, 0.01).unwrap();
        writer.finalize().unwrap();
        let reader = CImageReader::open(&path).unwrap();
        reader.validate_payload_ranges_for_validation().unwrap();
        assert!(reader.header.kv_compression_policy.is_some());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn joint_tiling_provenance_round_trips_through_cimage_header() {
        let path = std::env::temp_dir().join(format!(
            "prism_compile_joint_tiling_{}.cimage",
            std::process::id()
        ));
        let evidence = crate::search::JointTilingEvidence {
            selected_configuration: Some(crate::search::JointTilingConfiguration {
                ane_unit: prism_ecs_ir::evolution::foundation::AneUnitAxis::Auto,
                ane_tile_m: 64,
                ane_tile_n: 64,
                ane_tile_k: 32,
                metal_tile_m: 128,
                metal_tile_n: 64,
                metal_tile_k: 32,
                metal_threadgroup_width: 8,
                metal_threadgroup_height: 8,
            }),
            selected_score: Some(0.75),
            both_backends_feasible: true,
            both_backends_measured: true,
            profiles_evaluated: Vec::new(),
        };
        let mut writer = UniversalCImageWriter::new(&path);
        writer.set_joint_tiling_evidence(evidence.clone());
        writer.finalize().expect("finalize CImage");

        let reader = CImageReader::open(&path).expect("open CImage");
        assert_eq!(reader.header.joint_tiling_evidence, Some(evidence));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn eligible_native_ternary_receipt_admits_cimage() {
        use prism_ecs_quantization::ternarization::promotion::{
            BackendPass, NativeTernaryPromotionEvidence,
        };
        let path = std::env::temp_dir().join(format!(
            "prism_compile_promoted_{}.cimage",
            std::process::id()
        ));
        let mut writer = CImageWriter::new(&path).expect("create CImage");
        writer
            .append_native_ternary_with_scales(
                "weights",
                &[0, 1, 2, 0],
                &[0, 0, 128, 63],
                1,
                4,
                TensorType::Ternary158,
                TernaryDescriptor::legacy_for_type(&TensorType::Ternary158).unwrap(),
            )
            .expect("append ternary payload");
        writer
            .set_native_ternary_promotion(NativeTernaryPromotionEvidence {
                cpu_canary: BackendPass::passed(),
                accelerate_reconstruction: BackendPass::passed(),
                metal_packed: BackendPass::passed(),
                ane_static: BackendPass::passed(),
                cimage_replay: BackendPass::passed(),
                behavioral_reference: BackendPass::passed(),
                activation_error: Some(0.0),
                logit_divergence: Some(0.0),
                task_loss: Some(0.0),
                router_agreement: Some(1.0),
                router_margin_error: Some(0.0),
                logit_cross_entropy: Some(0.0),
                generation_loss: Some(0.0),
                expert_balance_error: Some(0.0),
                ane_selected: true,
                packed_abi_digest: "abi".into(),
                reference_digest: "reference".into(),
            })
            .expect("set promotion receipt");
        writer.finalize().expect("finalize promoted CImage");

        let reader = CImageReader::open(&path).expect("open promoted CImage");
        reader
            .validate_payload_ranges()
            .expect("promoted CImage must be admitted");
        let _ = std::fs::remove_file(path);
    }
}
