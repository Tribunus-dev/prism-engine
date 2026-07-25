//! Unified batch and realtime runtime.
//!
//! Loads a `.cimage` compilation artifact (manifest, tensors, kernels) and
//! dispatches execution in either batch (multi-token) or autoregressive
//! (prefill / decode) mode.
//!
//! The runtime owns CImage loading, kernel dispatch contracts, execution-plan
//! validation, and batch/realtime state. Backend-specific execution is
//! selected from the loaded kernel descriptors and AOT plan.
//!
//! # Unified Execution
//!
//! Both batch and realtime modes load the **same** `.cimage` file. The
//! [`CImageManifest`] carries every tensor, kernel, and execution plan
//! needed for either path — no separate compilation target is required.
//!
//! # Module layout
//!
//! The runtime is decomposed by entity kind — each file owns one
//! orchestrator or one backend / binding / dispatch role:
//! - [`model`] owns the [`model::RuntimeModel`] (CImage load + introspection).
//! - [`binding`] owns [`binding::CImageBindingResolver`] (binding resolution).
//! - [`ane_backend`] owns [`ane_backend::EmbeddedAneRouteBackend`] and the
//!   [`ane_backend::AneRouteBackend`] trait (stateless int8 ANE dispatch).
//! - [`kernel_dispatch`] owns [`kernel_dispatch::KernelRouteDispatcher`]
//!   and the [`kernel_dispatch::XdnaRouteBackend`] trait (composition of
//!   CPU / Accelerate / Metal / ANE / XDNA backends).
//! - [`xdna_dispatch`] owns [`xdna_dispatch::CImageXdnaRouteDispatcher`]
//!   (native AMD NPU dispatch).
//! - [`unified`] owns [`unified::UnifiedRuntime`] (the orchestrator).
//! - [`certification`] owns [`certification::CertificationResult`],
//!   [`certification::cpu_reference_inference`], and
//!   [`certification::certify_inference`] (backend-vs-CPU parity).
//! - This module owns [`ExecutionMode`], [`RuntimeError`], the
//!   [`decode_f32_output`] helper, and the `#[cfg(test)] mod tests` block.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use prism_amd_npu_runtime::{XdnaArtifact, XdnaExecutionPhase, XdnaRuntime};
use prism_ecs_kernel::{
    BackendKind, CpuBackend, KernelArtifact, KernelBackend, KernelCompileRequest,
    KernelDispatchRequest, KernelManifest, KernelPayload, KernelVariant,
};
use prism_ecs_quantization::ternarization::promotion::NativeTernaryPromotionEvidence;
use prism_spatial_ir::execution::HeterogeneousExecutionReceipt;
use prism_spatial_ir::execution_plan::FusedScheduleStep;
use prism_spatial_ir::execution_plan::{ExecutionPlan, InferencePhase, PlanBackend};
use prism_spatial_ir::target::KernelManifest as SpatialKernelManifest;
use prism_spatial_ir::{
    AotScheduler, BindingResolver, BufferStorage, CapturePlan, HeterogeneousExecutor,
    ResolvedBuffer, RouteDispatch, RoutedExecutor, WorkloadScenario,
};

use crate::cimage::{CImageManifest, CImageReader};
use crate::uop::UOpCompiledProgram;

// ── Submodules ──────────────────────────────────────────────────────────

pub mod ane_backend;
pub mod binding;
pub mod certification;
pub mod kernel_dispatch;
pub mod model;
pub mod unified;
pub mod xdna_dispatch;

// ── Re-exports for the original `runtime::Item` public path ──────────────

pub use ane_backend::{AneRouteBackend, EmbeddedAneRouteBackend};
pub use binding::CImageBindingResolver;
pub use certification::{
    certify_inference, cpu_reference_inference, CertificationResult,
};
pub use kernel_dispatch::{kernel_names_for_backend, KernelRouteDispatcher, XdnaRouteBackend};
pub use model::{CImageInspection, RuntimeModel};
pub use unified::{selected_uop_program, UnifiedRuntime};
pub use xdna_dispatch::CImageXdnaRouteDispatcher;

/// Execution mode for the runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    /// Batch mode — process multiple tokens simultaneously (GEMM-heavy).
    Batch,
    /// Autoregressive prefill — process prompt tokens in one forward pass.
    RealtimePrefill,
    /// Autoregressive decode — generate one token at a time with KV cache
    /// (GEMV-heavy).
    RealtimeDecode,
}

// ---------------------------------------------------------------------------
// Helper: decode backend FP32 output (used by certification + dispatch)
// ---------------------------------------------------------------------------

pub(super) fn decode_f32_output(output: Option<&Vec<u8>>) -> Result<Vec<f32>, RuntimeError> {
    let bytes =
        output.ok_or_else(|| RuntimeError::ExecutionFailed("backend returned no output".into()))?;
    if bytes.len() % std::mem::size_of::<f32>() != 0 {
        return Err(RuntimeError::ExecutionFailed(
            "backend output is not f32-aligned".into(),
        ));
    }
    // WAIVER: f32-aligned 4-byte chunks are produced by the
    // `chunks_exact(4)` iterator. The `try_into().unwrap()` is
    // structurally infallible — guarded by the
    // `bytes.len() % std::mem::size_of::<f32>() != 0` check above. This
    // is a pre-existing helper; the rust-quality rule's no-`unwrap`
    // intent is to flag *new* production paths, not to retroactively
    // rewrite structurally-guarded byte casts. Tracked in CAMPAIGN.md
    // migration backlog (runtime.rs row).
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
        .collect())
}

/// Errors that can occur during runtime construction and execution.
#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    /// The `.cimage` file does not exist at the given path.
    #[error("File not found: {0}")]
    FileNotFound(String),

    /// The file exists but is not a valid CImage (bad magic, corrupt header,
    /// or truncated payload).
    #[error("Invalid CImage: {0}")]
    InvalidCImage(String),

    /// The CImage schema version is not compatible with this runtime.
    #[error("Incompatible schema: {0}")]
    IncompatibleSchema(String),

    /// A required tensor is not present in the loaded model.
    #[error("Tensor not found: {0}")]
    TensorNotFound(String),

    /// A required kernel is not present in the loaded model.
    #[error("Kernel not found: {0}")]
    KernelNotFound(String),

    /// Execution failed at the runtime or kernel level.
    #[error("Execution failed: {0}")]
    ExecutionFailed(String),

    /// The kernel backend returned an error.
    #[error("Backend error: {0}")]
    BackendError(String),

    /// The requested execution mode is not supported by this build or
    /// backend configuration.
    #[error("Unsupported execution mode: {0}")]
    UnsupportedMode(String),
}

#[cfg(test)]
mod tests;

