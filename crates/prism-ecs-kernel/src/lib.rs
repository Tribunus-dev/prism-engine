//! Kernel backend interface — target-independent kernel contract types and traits.
//!
//! This crate defines the canonical interfaces that every kernel backend
//! (Metal, CPU, ANE, CUDA, Vulkan) must implement. Types are serializable
//! at crate boundaries for cross-process artifact exchange.
//!
//! # Hierarchy
//!
//! - [`KernelDescriptor`] — immutable description of a single compiled kernel.
//! - [`KernelPayload`] — a compiled kernel binary paired with its descriptor.
//! - [`KernelArtifact`] — one or more payloads plus a manifest.
//! - [`KernelManifest`] — per-kernel descriptors and an optional fusion plan.
//! - [`FusionPlan`] / [`FusedStep`] — kernel fusion scheduling.
//! - [`KernelBackend`] trait — each backend implements validate/compile/dispatch/measure.
//!
//! See ADR-032 (Kernel Backend Architecture) for the full design.

use serde::{Deserialize, Serialize};

// Re-export for convenience — widely used in downstream signatures.
pub use prism_ecs_core::canonical::kernel_abi::KernelSemanticId;

pub mod metal_backend;
pub use metal_backend::{
    MetalBackend, BUILTIN_KERNELS, FP16_GEMV_MSL, FP16_MATMUL_MSL, INT8_GEMV_MSL,
    NF4_TILE640_GEMV_MSL, TERNARY_TILE640_GEMV_MSL,
};
pub mod cpu_backend;
pub use cpu_backend::CpuBackend;
pub mod accelerate_backend;
pub use accelerate_backend::{AccelerateBackend, TernaryParityReport};
pub mod moe;
pub use moe::{request_from_router_logits, weighted_aggregate, MoeDispatchRequest};

#[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
pub mod metal_dispatch;

// Re-export TensorFormat for compatibility during unification.
pub use prism_ecs_ir::evolution::mutation_table::TensorFormat;
// ---------------------------------------------------------------------------
// Kernel variant
// ---------------------------------------------------------------------------

/// Identifies the compute strategy implemented by a kernel.
///
/// Each variant maps to a specific codec family and tile geometry convention.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KernelVariant {
    /// Standard FP16 matrix multiply (tiled GEMM).
    FP16Matmul,
    /// Standard FP16 matrix-vector multiply (GEMV).
    FP16GEMV,
    /// Ternary (ternary-weighted) Tile640 kernel with explicit ABI parameters.
    TernaryTile640(TernaryKernelAbi),
    /// NF4 quantized Tile640 kernel.
    NF4Tile640,
    /// INT8 quantized Tile640 kernel.
    INT8Tile640,
    /// Generic quantized GEMV (codec-agnostic).
    QuantizedGEMV,
    /// Backend-specific or experimental variant.
    Custom(String),
}

/// Explicit ternary kernel ABI parameters.
///
/// These are extracted from the calibration admission pipeline and determine
/// the exact memory layout used by the ternary tile codec.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TernaryKernelAbi {
    /// Page size in bytes for ternary tile storage.
    pub page_size: u32,
    /// Lane size in elements.
    pub lane_size: u32,
    /// Number of ternary words per page.
    pub words_per_page: u32,
    /// Number of scale factor bits per group.
    pub scale_bits: u8,
    /// Maximum number of outlier entries the format can hold.
    pub outlier_capacity: u32,
    /// Pack format discriminator (0 = packed, 1 = sparse, etc.).
    pub pack_format: u8,
}

// ---------------------------------------------------------------------------
// Backend identity
// ---------------------------------------------------------------------------

/// Target backend for kernel compilation and dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BackendKind {
    /// Apple Metal (GPU on Apple Silicon).
    Metal,
    /// CPU backend (reference / fallback).
    CPU,
    /// Apple Neural Engine.
    ANE,
    /// NVIDIA CUDA.
    CUDA,
    /// Vulkan (cross-platform GPU).
    Vulkan,
    /// AMD Ryzen AI XDNA NPU, lowered through Prism's native spatial runtime.
    AmdNpu,
}

// ---------------------------------------------------------------------------
// Binding description
// ---------------------------------------------------------------------------

/// The role a buffer plays in a kernel invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BufferRole {
    /// Data flows into the kernel.
    Input,
    /// Data flows out of the kernel.
    Output,
    /// Constant / read-only uniform data.
    Constant,
    /// Intermediate scratch buffer (no cross-invocation persistence).
    Intermediate,
}

/// Element type for a bound buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BindingDataType {
    /// 32-bit IEEE 754 float.
    Float32,
    /// 16-bit IEEE 754 float.
    Float16,
    /// Unsigned 8-bit integer.
    UInt8,
    /// Unsigned 16-bit integer.
    UInt16,
    /// Unsigned 32-bit integer.
    UInt32,
    /// Signed 8-bit integer.
    Int8,
    /// Signed 16-bit integer.
    Int16,
    /// Signed 32-bit integer.
    Int32,
}

/// A single buffer binding slot in a kernel's argument list.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BindingSlot {
    /// Buffer argument index (matches the Metal/GPU `[[buffer(N)]]` attribute).
    pub index: u32,
    /// Role of this buffer (input, output, constant, intermediate).
    pub role: BufferRole,
    /// Element type of the bound buffer.
    pub data_type: BindingDataType,
}

// ---------------------------------------------------------------------------
// Dispatch geometry
// ---------------------------------------------------------------------------

/// Thread dispatch geometry for grid-based compute dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DispatchGeometry {
    /// Threads per threadgroup (local work-group size).
    pub threads_per_threadgroup: [u32; 3],
    /// Number of threadgroups per grid dimension.
    pub threadgroups_per_grid: [u32; 3],
    /// Total threads per grid dimension (often `threads_per_threadgroup * threadgroups_per_grid`).
    pub threads_per_grid: [u32; 3],
}

// ---------------------------------------------------------------------------
// Kernel descriptor
// ---------------------------------------------------------------------------

/// Immutable description of a single compiled kernel.
///
/// Carries everything needed to identify, validate, and dispatch a kernel:
/// its source identity, variant, backend, compiled binary digest, binding
/// signature, and dispatch geometry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KernelDescriptor {
    /// Human- or tool-readable kernel name (e.g. `"ternary_matmul_tile640"`).
    pub name: String,
    /// Compute strategy variant.
    pub variant: KernelVariant,
    /// Target backend.
    pub backend: BackendKind,
    /// SHA-256 hex digest of the kernel source / MLIR / MSL.
    pub source_digest: String,
    /// SHA-256 hex digest of the compiled binary.
    pub binary_digest: String,
    /// Ordered binding signature that the kernel expects.
    pub binding_signature: Vec<BindingSlot>,
    /// Thread dispatch geometry.
    pub dispatch_geometry: DispatchGeometry,
}

// ---------------------------------------------------------------------------
// Kernel payload & artifact
// ---------------------------------------------------------------------------

/// A compiled kernel binary paired with its descriptor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KernelPayload {
    /// Compiled kernel binary bytes (e.g. metallib, compiled SPIR-V, etc.).
    pub binary: Vec<u8>,
    /// Descriptor for this single kernel.
    pub descriptor: KernelDescriptor,
}

/// The complete artifact produced by a backend's compilation step.
///
/// Contains one or more kernel payloads (multi-kernel artifacts — e.g. a
/// fusion bundle), a manifest, and a top-level artifact digest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KernelArtifact {
    /// All kernel payloads in this artifact.
    pub payloads: Vec<KernelPayload>,
    /// Manifest describing the artifact contents.
    pub manifest: KernelManifest,
    /// SHA-256 hex digest of the entire artifact.
    pub artifact_digest: String,
}

// ---------------------------------------------------------------------------
// Kernel manifest & fusion
// ---------------------------------------------------------------------------

/// Describes the contents of a compiled kernel artifact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KernelManifest {
    /// Descriptors for every kernel in the artifact.
    pub kernels: Vec<KernelDescriptor>,
    /// Optional fusion plan that combines kernels into fused invocations.
    pub fusion_plan: Option<FusionPlan>,
    /// SHA-256 hex digest of the manifest itself.
    pub manifest_digest: String,
}

/// A plan for fusing multiple kernels into a single dispatch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FusionPlan {
    /// Names of kernels to fuse, in fusion order.
    pub fused_kernels: Vec<String>,
    /// Ordered list of fused execution steps.
    pub schedule: Vec<FusedStep>,
    /// SHA-256 hex digest of the fusion plan.
    pub fusion_digest: String,
}

/// A single step within a fused-kernel schedule.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FusedStep {
    /// Name of the kernel to dispatch in this step.
    pub kernel_name: String,
    /// Input binding names or symbols for this step.
    pub input_bindings: Vec<String>,
    /// Output binding names or symbols for this step.
    pub output_bindings: Vec<String>,
}

// ---------------------------------------------------------------------------
// Compile / dispatch / measurement request types
// ---------------------------------------------------------------------------

/// Request to compile kernel source bytes into a [`KernelArtifact`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KernelCompileRequest {
    /// Raw kernel source bytes (MSL, SPIR-V, MLIR, etc.).
    pub source: Vec<u8>,
    /// Descriptor describing the expected kernel shape and variant.
    pub descriptor: KernelDescriptor,
    /// Optional path to the source file for diagnostics.
    pub source_path: Option<String>,
}

/// Request to dispatch a compiled kernel artifact with concrete input data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelDispatchRequest {
    /// Compiled artifact to dispatch.
    pub artifact: KernelArtifact,
    /// Raw input buffers (one per expected input binding).
    pub inputs: Vec<Vec<u8>>,
    /// Binding slots that map inputs to kernel argument indices.
    pub bindings: Vec<BindingSlot>,
}

/// Output from a single kernel dispatch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelOutput {
    /// Raw output buffers (one per output binding).
    pub outputs: Vec<Vec<u8>>,
    /// Wall-clock dispatch time in nanoseconds.
    pub dispatch_time_ns: u64,
}

/// Request to benchmark a compiled kernel artifact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KernelMeasurementRequest {
    /// Compiled artifact to benchmark.
    pub artifact: KernelArtifact,
    /// Input buffers used for each measured dispatch.
    pub inputs: Vec<Vec<u8>>,
    /// Number of iterations to run for measurement.
    pub iterations: u32,
}

/// Performance measurement result for a kernel dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct KernelMeasurement {
    /// Average dispatch time across iterations, in nanoseconds.
    pub avg_time_ns: f64,
    /// Minimum observed dispatch time, in nanoseconds.
    pub min_time_ns: f64,
    /// Maximum observed dispatch time, in nanoseconds.
    pub max_time_ns: f64,
    /// Estimated memory bandwidth achieved, in GB/s.
    pub bandwidth_gbps: f64,
}

// ---------------------------------------------------------------------------
// KernelError
// ---------------------------------------------------------------------------

/// Errors that can occur during kernel lifecycle operations.
///
/// Backend implementations return these errors from the [`KernelBackend`] trait
/// methods to provide structured diagnostics up the stack.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, thiserror::Error)]
pub enum KernelError {
    /// The requested backend is not available on this system.
    #[error("Unsupported backend: {0}")]
    UnsupportedBackend(String),

    /// Kernel compilation failed with a backend-specific error.
    #[error("Kernel compilation failed: {0}")]
    CompilationFailed(String),

    /// Kernel descriptor or binary validation failed.
    #[error("Kernel validation failed: {0}")]
    ValidationFailed(String),

    /// Kernel dispatch failed at runtime.
    #[error("Kernel dispatch failed: {0}")]
    DispatchFailed(String),

    /// Kernel measurement / benchmarking failed.
    #[error("Kernel measurement failed: {0}")]
    MeasurementFailed(String),

    /// The named kernel was not found in the artifact or registry.
    #[error("Kernel not found: {0}")]
    KernelNotFound(String),

    /// The provided binding signature does not match what the kernel expects.
    #[error("Binding mismatch: {0}")]
    BindingMismatch(String),
}

// ---------------------------------------------------------------------------
// KernelBackend trait
// ---------------------------------------------------------------------------

/// Trait implemented by every kernel compilation/dispatch backend.
///
/// # Lifecycle
///
/// 1. **`validate`** — check that a descriptor is self-consistent and
///    compatible with this backend before starting a potentially expensive
///    compilation.
/// 2. **`compile`** — compile source bytes into a [`KernelArtifact`].
/// 3. **`dispatch`** — execute a compiled artifact with concrete inputs.
/// 4. **`measure`** — benchmark a compiled artifact over N iterations.
///
/// All methods are `Send + Sync` so backends can be shared across threads
/// (e.g. sitting behind an `Arc<KernelBackend>` in the daemon schedule).
pub trait KernelBackend: Send + Sync {
    /// Validate that a kernel descriptor is well-formed and compatible with
    /// this backend *before* attempting compilation.
    ///
    /// Returns `Ok(())` if the descriptor passes backend-specific checks,
    /// or [`KernelError::ValidationFailed`] with a diagnostic message.
    fn validate(&self, descriptor: &KernelDescriptor) -> Result<(), KernelError>;

    /// Compile raw kernel source into a [`KernelArtifact`].
    ///
    /// The backend is responsible for invoking the appropriate toolchain
    /// (e.g. `xcrun metal` for Metal backends, `nvcc` for CUDA) and
    /// producing a portable, cached [`KernelArtifact`].
    fn compile(&self, request: &KernelCompileRequest) -> Result<KernelArtifact, KernelError>;

    /// Dispatch a compiled kernel artifact with the provided input bindings.
    ///
    /// Returns the output buffers and wall-clock dispatch time.
    fn dispatch(&self, request: &KernelDispatchRequest) -> Result<KernelOutput, KernelError>;

    /// Benchmark a compiled kernel artifact over the specified number of
    /// iterations.
    ///
    /// Returns aggregate timing statistics and estimated bandwidth.
    fn measure(&self, request: &KernelMeasurementRequest)
        -> Result<KernelMeasurement, KernelError>;

    /// Human-readable name for this backend (e.g. `"metal"`, `"cpu"`, `"ane"`).
    fn name(&self) -> &str;
}
