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
//!
//! # Module layout
//!
//! The cimage crate surface is split by authority along the read/write
//! axis:
//! - [`reader`] owns the read path (file open, header parse, payload
//!   location, format validation, evidence verification). It is the only
//!   module that reads `.cimage` files at runtime.
//! - [`writer`] owns the write path — both the low-level `CImageWriter`
//!   and the high-level `UniversalCImageWriter` that adds compilation
//!   metadata on top of it.
//! - This module owns the data definitions that both sides consume
//!   (`TensorType`, `CImageHeader`, `TensorRecord`, descriptors), the
//!   standalone promotion helpers (`emit_int8_ane_program`,
//!   `promote_cimage_after_replay`, `promote_cimage_with_behavioral_evidence`),
//!   and the small data envelopes (`CImageError`, `CImageManifest`,
//!   `TensorPayloadEntry`).
//!
//! The split-out of `reader` and `writer` out of the original monolithic
//! `cimage.rs` is the constitutional module-cohesion work called out in
//! `references/module-discipline.md` §Concrete decomposition patterns
//! for Prism.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

use prism_ecs_quantization::ternarization::promotion::{
    BackendPass, NativeTernaryPromotionEvidence,
};

// Reader module — owns the read authority. See module doc and `reader.rs`.
pub mod reader;
// Re-export the reader's public surface so callers using `cimage::CImageReader`
// and `cimage::cimage_read_blob` keep working. The reader module is the new
// home; the path `cimage::reader::CImageReader` is also valid.
pub use reader::{cimage_read_blob, CImageReader};

// Writer module — owns the write authority (CImageWriter +
// UniversalCImageWriter). See module doc and `writer.rs`.
pub mod writer;
// Re-export the writers so callers using `cimage::CImageWriter` and
// `cimage::UniversalCImageWriter` keep working. The writer module is the
// new home; the path `cimage::writer::CImageWriter` is also valid.
pub use writer::{CImageWriter, UniversalCImageWriter};

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

impl TensorType {
    /// Stable single-byte discriminant for content addressing.
    ///
    /// **The wire format is the `prism_ecs_quantization::cimage::TensorType`
    /// discriminant order** — the canonical order used by the
    /// constitutional `QuantizationResultComponent`. The Rust enum
    /// order above is preserved for backward compatibility with
    /// v1-serialized CImage headers; the discriminant method is the
    /// only thing that crosses the wire.
    ///
    /// Do not change the mapping table without bumping the
    /// `QuantizationResultComponent` `schema_version`.
    pub fn discriminant_byte(&self) -> [u8; 1] {
        // Mapping table from the canonical (quantization-crate) order
        // to this enum. Same byte values as
        // `prism_ecs_quantization::cimage::TensorType::discriminant_byte`.
        let n: u8 = match self {
            TensorType::Bf16 => 0,
            TensorType::Int8 => 1,
            TensorType::Nf8 => 2,
            TensorType::StandardFP16 => 3,
            TensorType::Palettized4Bit => 4,
            TensorType::Blob => 5,
            TensorType::Ternary158 => 6,
            TensorType::Binary1 => 7,
            TensorType::NF4 => 8,
            TensorType::Int4 => 9,
            TensorType::TernaryTile640 => 10,
            TensorType::FP8 => 11,
        };
        [n]
    }

    /// Reverse mapping for `discriminant_byte`. Returns `None` for an
    /// unknown byte so callers can detect stale or corrupted schema
    /// versions without panicking.
    pub fn from_discriminant_byte(byte: u8) -> Option<TensorType> {
        match byte {
            0 => Some(TensorType::Bf16),
            1 => Some(TensorType::Int8),
            2 => Some(TensorType::Nf8),
            3 => Some(TensorType::StandardFP16),
            4 => Some(TensorType::Palettized4Bit),
            5 => Some(TensorType::Blob),
            6 => Some(TensorType::Ternary158),
            7 => Some(TensorType::Binary1),
            8 => Some(TensorType::NF4),
            9 => Some(TensorType::Int4),
            10 => Some(TensorType::TernaryTile640),
            11 => Some(TensorType::FP8),
            _ => None,
        }
    }
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
    /// Workload profile grid and heterogeneous route evidence used for
    /// realtime/batch execution selection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub heterogeneous_workload_evidence: Option<crate::search::HeterogeneousScheduleEvidence>,
    /// Promotion receipt for native ternary payloads.  The runtime preserves
    /// this evidence so an artifact cannot be admitted after a failed or
    /// incomplete backend validation pass.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub native_ternary_promotion: Option<NativeTernaryPromotionEvidence>,
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
        assert_eq!(
            tuning.source,
            crate::uop::UOpMeasurementSource::SyntheticFallback
        );
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
        let error = promote_cimage_after_replay(
            &path,
            promotion_evidence_without_behavioral_measurements(),
        )
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
    fn heterogeneous_workload_evidence_round_trips_through_cimage_runtime() {
        let path = std::env::temp_dir().join(format!(
            "prism_compile_heterogeneous_workload_{}.cimage",
            std::process::id()
        ));
        let profile = crate::workload_search::default_profile_grid()[0];
        let evidence = crate::workload_search::WorkloadThroughputEvidence {
            profile,
            representation: "Ternary158".into(),
            tiling_digest: "tile-test".into(),
            tokens_per_second: 123.0,
            latency_ms: 8.1,
            measured: true,
            evidence_source: "test-native".into(),
            execution_fingerprint: "test-graph".into(),
            projected: true,
            projection_basis: "test".into(),
            mixed_precision_graph: "ternary-expert-int8-attention".into(),
            ..crate::workload_search::WorkloadThroughputEvidence::default()
        };
        let schedule = crate::search::HeterogeneousScheduleEvidence {
            steps: 2,
            route_sequence: vec!["ane".into(), "metal".into()],
            zero_copy_steps: 2,
            estimated_latency_ns: 8100000,
            residency_windows: 1,
            supports_realtime_text: true,
            supports_batched_text: true,
            supports_batched_audio: false,
            workload_profiles: vec![profile],
            throughput_evidence: vec![evidence],
            selected_execution_graph: crate::workload_search::SelectedExecutionGraph::default(),
        };
        let mut writer = CImageWriter::new(&path).unwrap();
        writer.set_heterogeneous_workload_evidence(schedule.clone());
        writer.finalize().unwrap();
        let reader = CImageReader::open(&path).unwrap();
        let stored = reader
            .header
            .heterogeneous_workload_evidence
            .as_ref()
            .unwrap();
        assert_eq!(
            stored.throughput_evidence[0].mixed_precision_graph,
            "ternary-expert-int8-attention"
        );
        let runtime = crate::runtime::RuntimeModel::load_for_validation(&path).unwrap();
        assert_eq!(runtime.heterogeneous_workload_evidence.unwrap().steps, 2);
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
            all_backends_feasible: false,
            all_backends_measured: false,
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
