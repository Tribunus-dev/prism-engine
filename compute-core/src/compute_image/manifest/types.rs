//! ComputeImage manifest data types.
//!
//! Pure data types — no builder, runtime I/O, or telemetry logic.

use crate::compute_image::hw_assessment::AssessmentReceipt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

// ── Storage ABI constants ──────────────────────────────────────────────────

/// Storage ABI identifier for the baseline copied (CPU-allocated) path.
pub const STORAGE_ABI_COPIED_V0: &str = "copied-v0";
/// Storage ABI identifier for the mapped, no-copy (Metal-buffer) path.
pub const STORAGE_ABI_MAPPED_NO_COPY_V1: &str = "mapped-no-copy-v1";

/// Return true if `abi` is a recognised storage ABI identifier.
pub fn is_valid_storage_abi(abi: &str) -> bool {
    abi == STORAGE_ABI_COPIED_V0 || abi == STORAGE_ABI_MAPPED_NO_COPY_V1
}

/// Magic identifier for the legacy V1 .cimage binary format.
pub const CIMAGE_MAGIC: u32 = 0x43494D47;

/// Legacy V1 on-disk header for the ternary .cimage format (128 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CImageHeader {
    pub magic: u32,
    pub version: u32,
    pub quantization_schema: u32,
    pub payload_hash: [u8; 32],
    pub phase_count: u32,
    pub layout_offset: u64,
    pub phase_offset: u64,
    pub ane_hidden_dim_limit: u32,
    pub ane_ffn_dim_limit: u32,
    pub ane_max_batch: u32,
    pub ane_keepalive_interval_us: u64,
    pub lane_isolation: bool,
    pub(crate) _pad: [u8; 43],
}

impl Default for CImageHeader {
    fn default() -> Self {
        Self {
            magic: 0,
            version: 0,
            quantization_schema: 0,
            payload_hash: [0u8; 32],
            phase_count: 0,
            layout_offset: 0,
            phase_offset: 0,
            ane_hidden_dim_limit: 0,
            ane_ffn_dim_limit: 0,
            ane_max_batch: 0,
            ane_keepalive_interval_us: 0,
            lane_isolation: false,
            _pad: [0u8; 43],
        }
    }
}

// ── Compilation authority ──────────────────────────────────────────────────

/// Who is asking to compile a model, and under what authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompilationAuthority {
    /// Unit-test fixtures only. Small ceiling enforced.
    TestFixture,
    /// Production sealed ComputeImage. Requires image-build profile.
    SealedComputeImage,
}

impl std::fmt::Display for CompilationAuthority {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompilationAuthority::TestFixture => write!(f, "TestFixture"),
            CompilationAuthority::SealedComputeImage => write!(f, "SealedComputeImage"),
        }
    }
}

// ── Top-level manifest ─────────────────────────────────────────────────────

fn default_storage_abi() -> String {
    "copied-v0".to_string()
}
fn default_prepacked_layout() -> String {
    "none".to_string()
}
pub(crate) fn default_alignment_bytes() -> u64 {
    4096
}
pub(crate) fn default_tensor_alignment_bytes() -> u64 {
    16
}
pub(crate) fn default_layout_version() -> u32 {
    1
}

/// Top-level ComputeImage manifest.
#[derive(Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub image_version: String,
    pub compiler_version: String,
    pub runtime_abi: String,
    /// Target hardware this image was compiled for (None = auto-detect at compile time).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hardware_target: Option<crate::config::HardwareTarget>,
    /// Compilation readiness verdict after artifact audit.
    #[serde(default)]
    pub readiness: Option<CompileReadiness>,
    /// ISO 8601 timestamp of compilation.
    #[serde(default)]
    pub compile_date: String,
    /// Hostname of the machine that compiled this image.
    #[serde(default)]
    pub compile_host: String,
    pub source: SourceIdentity,
    pub architecture: crate::config::TextArchitecture,
    /// Vision encoder configuration (vision_config from model config.json).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vision_config: Option<crate::config::VisionArchitecture>,
    /// Audio encoder configuration (Gemma 4 Unified audio_config).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_config: Option<crate::config::AudioArchitecture>,
    pub segments: Vec<Segment>,
    pub tensor_table: Vec<TensorEntry>,
    pub alias_table: Vec<AliasEntry>,
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
    pub metal_kernel_artifacts: Vec<MetalKernelArtifact>,
    /// Execution plan emitted by the compiler (prologue, layers, epilogue).
    #[serde(default)]
    pub execution_plan: crate::config::ModelExecutionPlan,
    /// Compiler-emitted phase DAG for PhaseEngine dispatch (optional).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase_dag: Option<crate::compute_image::phase_dag::EmittedPhaseGraph>,
    /// CompatibilityMatrix validation receipt from compile time.
    /// Contains model family, fallback chain, and any warnings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compatibility_receipt: Option<serde_json::Value>,
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
    pub fn storage_abi_matches(&self, backend: &StorageBackend) -> bool {
        match backend {
            StorageBackend::Copied => self.required_storage_abi == STORAGE_ABI_COPIED_V0,
            StorageBackend::MappedNoCopy => {
                self.required_storage_abi == STORAGE_ABI_MAPPED_NO_COPY_V1
            }
        }
    }
}

// ── Storage ABI specification ──────────────────────────────────────────────

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
    /// "none" for source layout, "prepacked-int8-v1" for transposed+interleaved.
    pub prepacked_layout: String,
}

impl StorageAbiSpec {
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

/// Validate a single `TensorEntry` against the mapped-no-copy-v1 ABI.
///
/// Checks:
/// - Offset must be aligned to `tensor_offset_alignment_bytes`.
/// - `storage_dtype` must be in `supported_physical_dtypes`.
/// - Quantized tensors with scale/bias side-tensors must have group sizes
///   compatible with the declared shape (groups × group_size must not overflow
///   the flattened logical element count).
///
/// Collects all violations into the returned `Vec`; does not short-circuit.
pub fn validate_tensor_for_mapped_abi(
    entry: &TensorEntry,
    spec: &StorageAbiSpec,
) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();

    // Offset alignment check
    if entry.offset % spec.tensor_offset_alignment_bytes != 0 {
        errors.push(format!(
            "tensor {} offset {} is not aligned to {} bytes",
            entry.name, entry.offset, spec.tensor_offset_alignment_bytes,
        ));
    }

    // Storage dtype in supported list
    let dtype_upper = entry.storage_dtype.to_uppercase();
    if !spec
        .supported_physical_dtypes
        .iter()
        .any(|d| d.to_uppercase() == dtype_upper)
    {
        errors.push(format!(
            "tensor {} storage_dtype {} is not in supported dtypes {:?}",
            entry.name, entry.storage_dtype, spec.supported_physical_dtypes,
        ));
    }

    // Quantized tensor validation
    if let Some(qdesc) = &entry.quantization {
        let log_prod: u64 = entry.logical_shape.iter().copied().map(u64::from).product();
        let groups = u64::from(qdesc.groups);
        let group_size = u64::from(qdesc.group_size);
        let packed = groups.saturating_mul(group_size);
        if packed > log_prod {
            errors.push(format!(
                "tensor {} quantized groups {} × group_size {} = {} > logical elements {}",
                entry.name, qdesc.groups, qdesc.group_size, packed, log_prod,
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
/// - All tensors pass `validate_tensor_for_mapped_abi`.
///
/// Returns `Err(Vec<String>)` with every violation; does not short-circuit.
pub fn validate_manifest_for_abi(
    manifest: &Manifest,
    spec: &StorageAbiSpec,
) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();

    // Segment alignment validation
    for seg in &manifest.segments {
        if seg.alignment_bytes % spec.segment_alignment_bytes != 0 {
            errors.push(format!(
                "segment {} alignment_bytes {} is not a multiple of {} (ABI segment alignment)",
                seg.id, seg.alignment_bytes, spec.segment_alignment_bytes,
            ));
        }
    }

    // Tensor validation against ABI
    for entry in &manifest.tensor_table {
        if let Err(tensor_errors) = validate_tensor_for_mapped_abi(entry, spec) {
            errors.extend(tensor_errors);
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

// ── Physical dtype & layout validation ─────────────────────────────────────

/// Validate that `dtype` is a recognised physical storage dtype and return
/// the expected byte count for the given shape.  Handles unpacked dtypes
/// (f32 4b, bf16 2b, f16 2b, u8 1b, i8 1b, u32 4b) and quantized packed
/// dtypes where the caller accounts for group-size packing separately.
///
/// Quantized packed types ("U8", "I8" with quantization context) have the
/// same per-element byte count as their unpacked counterpart (1×prod), so
/// this function returns `prod` for both unpacked and quantized u8/i8.
pub fn validate_physical_dtype(
    dtype: &str,
    byte_length: u64,
    shape: &[u32],
) -> Result<u64, String> {
    let prod: u64 = shape.iter().copied().map(u64::from).product();
    let element_bytes = match dtype {
        "f32" | "F32" | "Float32" => 4u64,
        "bf16" | "BF16" | "BFloat16" => 2,
        "f16" | "F16" | "Float16" => 2,
        "u8" | "U8" | "Uint8" => 1,
        "i8" | "I8" | "Int8" => 1,
        "u32" | "U32" | "Uint32" => 4,
        other => return Err(format!("unsupported physical dtype: {}", other)),
    };
    let expected = prod.saturating_mul(element_bytes);
    if byte_length != expected {
        return Err(format!(
            "dtype {} with shape {:?}: expected {} bytes ({}×{}), got {}",
            dtype, shape, expected, prod, element_bytes, byte_length,
        ));
    }
    Ok(expected)
}

/// Validate physical tensor layout constraints for a single `TensorEntry`
/// within a segment of `segment_byte_size` bytes.
///
/// Checks: byte_length > 0, offset + byte_length <= segment_byte_size,
/// shape-based byte count matches byte_length, and when the entry declares
/// a `QuantizationDesc` the scale/bias entries are dimensionally consistent.
pub fn validate_tensor_layout(entry: &TensorEntry, segment_byte_size: u64) -> Result<(), String> {
    if entry.byte_length == 0 {
        return Err(format!("tensor {} has zero byte_length", entry.name));
    }
    let end = entry.offset.saturating_add(entry.byte_length);
    if end > segment_byte_size {
        return Err(format!(
            "tensor {} offset {} + byte_length {} exceeds segment size {}",
            entry.name, entry.offset, entry.byte_length, segment_byte_size,
        ));
    }

    // Validate that physical_shape × dtype bytes matches byte_length.
    // Allow quantization packing where byte_length may differ from
    // the unpacked product (e.g. packed weights smaller than logical).
    if entry.quantization.is_some() {
        // For quantized tensors, the byte_length is the packed payload;
        // logical validation is ownership of the caller.  We only check
        // that it is non-zero (already done above) and that the physical
        // shape is not degenerate.
        if entry.physical_shape.is_empty() || entry.physical_shape.iter().any(|&d| d == 0) {
            return Err(format!(
                "tensor {} has degenerate quantized physical shape {:?}",
                entry.name, entry.physical_shape,
            ));
        }
    } else {
        // Unquantized: validate dtype byte count matches.
        validate_physical_dtype(
            &entry.storage_dtype,
            entry.byte_length,
            &entry.physical_shape,
        )?;
    }

    Ok(())
}

// ── Source identity ────────────────────────────────────────────────────────

/// Cryptographic identity of the source checkpoint.
#[derive(Clone, Serialize, Deserialize)]
pub struct SourceIdentity {
    pub config_hash: String,
    pub shard_hashes: Vec<ShardHash>,
    pub tokenizer_hashes: Vec<ShardHash>,
    pub auxiliary_hashes: Vec<ShardHash>,
    pub model_type: String,
    pub quantization_bits: u32,
    pub quantization_group_size: u32,
    pub quantization_mode: String,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ShardHash {
    pub filename: String,
    pub sha256: String,
}

// ── Segment / tensor / alias types ─────────────────────────────────────────

/// One binary segment containing tensors in execution order.
#[derive(Clone, Serialize, Deserialize)]
pub struct Segment {
    pub id: String,       // "embed", "layer_0", "layer_5", "final"
    pub filename: String, // "segment_000.bin"
    pub byte_size: u64,
    pub sha256: String,
    pub tensor_ids: Vec<u32>, // ordered tensor references
    pub kind: SegmentKind,
    /// Alignment constraint in bytes for the mapped-no-copy backend (default 4096).
    #[serde(default = "default_alignment_bytes")]
    pub alignment_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SegmentKind {
    Persistent, // always loaded (embeddings, final norm)
    Layer(u32), // per-layer, load/free per execution window
    Final,      // output projection (may alias Persistent)
}

/// One tensor entry in the global table.
#[derive(Clone, Serialize, Deserialize)]
pub struct TensorEntry {
    pub id: u32,
    pub name: String,
    pub role: String,
    pub layer: Option<u32>,
    pub segment: String,
    pub source_filename: String,
    pub source_sha256: String,
    pub source_offset: u64,
    pub offset: u64,
    pub byte_length: u64,
    pub logical_dtype: String,
    pub storage_dtype: String,
    pub logical_shape: Vec<u32>,
    pub physical_shape: Vec<u32>,
    pub mutability: String,
    pub quantization: Option<QuantizationDesc>,
    /// Per-tensor alignment in bytes for the mapped-no-copy backend (default 16).
    #[serde(default = "default_tensor_alignment_bytes")]
    pub tensor_alignment_bytes: u64,
    /// Layout version for the tensor-cache key computation (default 1).
    #[serde(default = "default_layout_version")]
    pub layout_version: u32,
    /// Per-backend artifact bindings for this tensor.
    /// Keyed by backend name ("mlx", "coreml", "accelerate", etc.).
    #[serde(default)]
    pub artifact_bindings: HashMap<String, Vec<BackendWeightArtifact>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationDesc {
    pub bits: u32,
    pub group_size: u32,
    pub groups: u32,
    pub scale_tensor_id: u32,
    pub bias_tensor_id: u32,
}

/// An alias mapping — resolves a logical tensor name to physical storage.
#[derive(Clone, Serialize, Deserialize)]
pub struct AliasEntry {
    pub logical_name: String,
    pub physical_tensor_id: u32,
    pub reason: String,
}

/// Concrete packing scheme and target backend for a weight artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactKind {
    /// MLX NF4 — packed uint32 words (8 NF4 values per u32).
    MlxNf4U32,
    /// MLX 8-bit affine — packed uint32 words (4 u8 values per u32).
    MlxAf8U32,
    /// CPU fp16 — dequantized float16.
    CpuFp16,
    /// CPU quantized — block quantized bytes.
    CpuQuantized,
    /// Core ML fp16 external weight file.
    CoreMlFp16WeightFile,
    /// Intel Level Zero packed USM.
    IntelUsmPacked,
    /// Tenstorrent Tensix tiled.
    TensixTilePacked,
}

/// A concrete artifact for one backend execution lane.
/// Describes exactly where and how a weight tensor is stored for a specific lane.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendWeightArtifact {
    /// Logical tensor this artifact represents.
    pub logical_tensor_id: String,
    /// Target backend.
    pub backend: String,
    /// Packing/kernel format.
    pub artifact_kind: ArtifactKind,
    /// Logical (semantic) shape.
    pub logical_shape: Vec<u32>,
    /// Physical (storage) shape after packing.
    pub storage_shape: Vec<u32>,
    /// Source quantization before any dequantization transform.
    pub logical_quantization: Option<QuantizationDesc>,
    /// Storage dtype string ("U32", "F16", "U8", etc.).
    pub storage_dtype: String,
    /// How values are packed ("nf4_u32", "af8_u32", "af8_u8", "none_fp16").
    pub packing_scheme: String,
    /// Block quantization group size (0 = per-tensor).
    pub group_size: u32,
    /// Name of the companion scale tensor artifact binding.
    pub scale_binding: Option<String>,
    /// Name of the companion zero-point tensor artifact binding.
    pub zero_point_binding: Option<String>,
    /// Segment filename containing the raw bytes.
    pub segment_path: String,
    /// Byte offset within the segment.
    pub byte_offset: u64,
    /// Byte length of this artifact in the segment.
    pub byte_length: u64,
    /// SHA-256 checksum of the artifact bytes.
    pub checksum: String,
    /// Estimated numerical error introduced by quantization (0.0 for fp16).
    pub numerical_error: f64,
    /// Compiler version that produced this artifact.
    pub producer_version: String,
}

/// Dispatch configuration for a compiled Metal kernel.
/// Describes how to bind buffers and dispatch the compute shader.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetalDispatchRecipe {
    /// Entry point function name within the compiled Metal library.
    pub entry_point: String,
    /// Human-readable kernel name for identification.
    pub kernel_name: String,
    /// Threadgroup size (threads per threadgroup).
    pub threads_per_threadgroup: [u32; 3],
    /// Grid size (number of threadgroups).
    pub threadgroups_per_grid: [u32; 3],
    /// Metal buffer indices for each logical binding, keyed by binding name.
    pub buffer_slot_map: HashMap<String, u32>,
    /// Scalar binding indices with their Metal type string, keyed by binding name.
    pub scalar_index_map: HashMap<String, (u32, String)>,
    /// K (input channel) dimension from the export.
    pub k: u64,
    /// N (output channel) dimension from the export.
    pub n: u64,
    /// Block quantization group size.
    pub group_size: u32,
    /// Quantization bits.
    pub bits: u8,
    /// Kernel ABI version — must match between compiler and runtime.
    pub kernel_abi_version: u32,
}

/// A pre-compiled Metal kernel embedded in the ComputeImage.
/// The .metallib is stored under `metal/kernels/<artifact_id>.metallib`
/// in the image directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetalKernelArtifact {
    /// Unique identifier for this artifact (e.g., "q_proj_nf4_layer0").
    pub artifact_id: String,
    /// Which logical operation this artifact implements.
    pub logical_operation: String,
    /// Target artifact kind (MlxNf4U32, MlxAf8U32, etc.).
    pub kind: ArtifactKind,
    /// Path to the .metallib relative to the image root.
    pub metallib_relpath: String,
    /// BLAKE3 hash of the .metallib for integrity.
    pub metallib_blake3: String,
    /// Byte size of the .metallib.
    pub metallib_byte_length: u64,
    /// Dispatch recipe.
    pub dispatch: MetalDispatchRecipe,
    /// Logical shape of the weight tensor (e.g., [896, 896]).
    pub logical_shape: Vec<u32>,
    /// Storage shape of the packed weight (e.g., [896, 112]).
    pub storage_shape: Vec<u32>,
    /// Quantization bits (4 for NF4, 8 for AF8).
    pub bits: u8,
    /// Block quantization group size.
    pub group_size: u32,
    /// Name of the companion scale tensor.
    pub scale_tensor: String,
    /// Name of the companion bias tensor.
    pub bias_tensor: String,
    /// GPU family this artifact was compiled for.
    pub gpu_family: String,
    /// SHA-256 checksum of the entire artifact descriptor.
    pub checksum: String,
}

// ── Compilation receipts ───────────────────────────────────────────────────

/// Result of diffing current source tensors against a previous compilation
/// manifest.
#[derive(Default, Debug)]
pub struct TensorDiff {
    /// Tensor names whose hash matches the previous compile.
    pub unchanged: Vec<String>,
    /// Tensor names whose hash differs from the previous compile.
    pub changed: Vec<String>,
    /// Tensor names present in the source but not in the previous compile.
    pub new: Vec<String>,
    /// Tensor names present in the previous compile but absent from the source.
    pub removed: Vec<String>,
    /// Wall-clock milliseconds spent computing the diff.
    pub elapsed_ms: u128,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct TensorProvenance {
    pub tensor_name: String,
    pub source_sha256: String,
    pub emitted_sha256: String,
    pub preserved_byte_for_byte: bool,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct IgnoredTensorClassification {
    pub name: String,
    pub classification: String,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct SegmentReceipt {
    pub id: String,
    pub filename: String,
    pub sha256: String,
    pub byte_size: u64,
}

#[derive(Clone, Serialize, Deserialize, Default)]
pub struct CompileReceipt {
    pub source_config_hash: String,
    pub source_shard_hashes: Vec<ShardHash>,
    pub compiler_version: String,
    pub runtime_abi: String,
    pub normalized_architecture_hash: String,
    pub execution_plan_hash: String,
    pub complete_image_hash: String,
    pub segment_hashes: Vec<SegmentReceipt>,
    pub tensor_count: usize,
    pub alias_count: usize,
    pub segment_count: usize,
    pub ignored_tensor_classifications: Vec<IgnoredTensorClassification>,
    pub total_source_bytes: u64,
    pub total_emitted_bytes: u64,
    pub elapsed_ms: u128,
    pub transformed_payloads: Vec<String>,
    pub byte_provenance: Vec<TensorProvenance>,
    pub structural_verification: bool,
    /// Native dependency identity captured at compile time.
    pub native_dependency_report: NativeCapabilityReport,
    /// Hardware assessment receipt from compile-time kernel profiling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hw_assessment: Option<AssessmentReceipt>,
    pub stage_profile: StageProfile,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StageProfile {
    pub source_discovery_ms: u64,
    pub source_hashing_ms: u64,
    pub header_parsing_ms: u64,
    pub architecture_normalization_ms: u64,
    pub binding_validation_ms: u64,
    pub layout_planning_ms: u64,
    pub payload_emission_ms: u64,
    pub segment_hashing_ms: u64,
    pub manifest_generation_ms: u64,
    pub verification_ms: u64,
    pub total_source_bytes: u64,
    pub total_emitted_bytes: u64,
    pub peak_rss_bytes: u64,
    pub peak_mlx_active_bytes: u64,
    pub peak_mlx_cache_bytes: u64,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct CompiledImage {
    pub manifest: Manifest,
    pub receipt: CompileReceipt,
}

// ── Verification ───────────────────────────────────────────────────────────

#[derive(Clone, Serialize, Deserialize)]
pub struct ManifestVerification {
    pub manifest_hash_matches: bool,
    pub segment_hashes_match: bool,
    pub verified_segment_count: usize,
    pub total_bytes: u64,
}

/// How tensor bytes were moved from storage into MLX.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CopyClassification {
    /// Direct mmap view, no application copy. MLX may still copy internally.
    MappedNoCopy,
    /// Copied from mmap into an application-side buffer before MLX construction.
    CopiedFallback,
    /// MLX created a contiguous temporary (reshape, transpose, dtype cast, repeat).
    MaterializedContiguous,
    /// BF16 -> F32 or other dtype promotion.
    MaterializedDtypeConversion,
    /// K/V physically repeated for grouped-query attention.
    MaterializedRepeat,
}

/// How tensor bytes were moved from storage into MLX.
#[derive(Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum StorageBackend {
    Copied,
    MappedNoCopy,
}

#[derive(Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum LeaseState {
    Opened,
    Bound,
    Active,
    Retiring,
    Released,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct SegmentLease {
    pub segment_id: String,
    pub filename: String,
    pub backend: StorageBackend,
    pub state: LeaseState,
    pub tensor_handles: Vec<u64>,
    pub byte_size: u64,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct TensorLease {
    pub name: String,
    pub handle: u64,
    pub segment_id: String,
    pub state: LeaseState,
}

/// Native dependency identity and capability report.
/// Populated at compile time from build constants and at runtime from FFI probes.
#[derive(Clone, Serialize, Deserialize, Default)]
pub struct NativeCapabilityReport {
    pub mlx_core_version: String,
    pub mlx_c_version: String,
    pub mlx_rs_version: String,
    pub mlx_sys_version: String,
    pub compute_native_version: String,
    pub supports_quantized_matmul: bool,
    pub supports_dequantize: bool,
    pub supports_memory_telemetry: bool,
    pub supports_cache_control: bool,
    pub supports_external_array: bool,
    pub supports_multithreaded_execution: bool,
    pub metal_available: bool,
    pub accelerate_available: bool,
}

/// Runtime residency plan.
#[derive(Clone, Serialize, Deserialize)]
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

/// Admission-estimate for representation-aware memory budgeting.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct RepresentationAdmissionEstimate {
    pub virtual_mapped_bytes: u64,
    pub expected_resident_bytes: u64,
    pub persistent_materialized_bytes: u64,
    pub max_layer_window_bytes: u64,
    pub rope_bytes: u64,
    pub kv_budget_bytes: u64,
    pub mlx_workspace_bytes: u64,
    pub allocator_cache_bytes: u64,
    pub system_reserve_bytes: u64,
    /// Maximum single transient allocation during inference
    /// (attention workspace, output projection buffer, etc.).
    pub largest_transient_bytes: u64,
    /// Bytes that must be converted (dequantized, dtype-cast) at runtime.
    pub materialized_bytes: u64,
}

// ── Manifest hashing ───────────────────────────────────────────────────────

pub(crate) fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

pub(crate) fn compute_manifest_hash(manifest: &Manifest) -> String {
    #[derive(Serialize)]
    struct Fingerprint<'a> {
        image_version: &'a str,
        compiler_version: &'a str,
        runtime_abi: &'a str,
        source: &'a SourceIdentity,
        architecture: &'a crate::config::TextArchitecture,
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

    let bytes = serde_json::to_vec(&fingerprint).expect("manifest fingerprint serialization");
    sha256_bytes(&bytes)
}
