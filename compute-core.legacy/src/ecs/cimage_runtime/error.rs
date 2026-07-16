//! CImage runtime bridge — errors for artifact-driven execution.

use crate::ecs::cimage::CImageError;
use crate::execution_plan::CodecFamily;

/// Errors that can occur during cimage runtime execution.
#[derive(Debug, Clone)]
pub enum CImageRuntimeError {
    /// Wraps a cimage format error.
    CImage(CImageError),
    /// The artifact kind is not supported by this runtime path.
    UnsupportedArtifactKind,
    /// A required tensor entry is missing from the manifest.
    MissingTensor(String),
    /// A required payload is missing from the payload directory.
    MissingPayload(String),
    /// The codec family is not supported by this runtime path.
    UnsupportedCodec(CodecFamily),
    /// The tensor shape is invalid for the requested operation.
    InvalidTensorShape(String),
    /// Metal runtime is unavailable (not macOS or metal-dispatch not enabled).
    MetalUnavailable,
    /// Metal library compilation failed.
    MetalLibraryCompileFailed(String),
    /// PSO creation failed for a specific kernel.
    PipelineCreationFailed(String),
    /// A Metal buffer allocation failed.
    BufferAllocationFailed(String),
    /// A kernel binding is missing for a required slot.
    KernelBindingMissing(String),
    /// The hazard checker rejected the execution region.
    HazardViolation(String),
    /// Metal command buffer execution failed.
    ExecutionFailed(String),
    /// Numerical validation of the output failed.
    ValidationFailed(String),
    /// The low-level lowering produced an invalid op.
    LoweringFailed(String),
}

impl std::fmt::Display for CImageRuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CImage(e) => write!(f, "cimage error: {e}"),
            Self::UnsupportedArtifactKind => write!(f, "unsupported artifact kind"),
            Self::MissingTensor(id) => write!(f, "missing tensor: {id}"),
            Self::MissingPayload(id) => write!(f, "missing payload: {id}"),
            Self::UnsupportedCodec(c) => write!(f, "unsupported codec: {c:?}"),
            Self::InvalidTensorShape(s) => write!(f, "invalid tensor shape: {s}"),
            Self::MetalUnavailable => write!(
                f,
                "Metal runtime is unavailable (requires macOS + metal-dispatch)"
            ),
            Self::MetalLibraryCompileFailed(s) => write!(f, "Metal library compile failed: {s}"),
            Self::PipelineCreationFailed(s) => write!(f, "pipeline creation failed: {s}"),
            Self::BufferAllocationFailed(s) => write!(f, "buffer allocation failed: {s}"),
            Self::KernelBindingMissing(s) => write!(f, "kernel binding missing: {s}"),
            Self::HazardViolation(s) => write!(f, "hazard violation: {s}"),
            Self::ExecutionFailed(s) => write!(f, "execution failed: {s}"),
            Self::ValidationFailed(s) => write!(f, "validation failed: {s}"),
            Self::LoweringFailed(s) => write!(f, "lowering failed: {s}"),
        }
    }
}

impl std::error::Error for CImageRuntimeError {}

impl From<CImageError> for CImageRuntimeError {
    fn from(e: CImageError) -> Self {
        Self::CImage(e)
    }
}

impl From<String> for CImageRuntimeError {
    fn from(s: String) -> Self {
        Self::LoweringFailed(s)
    }
}

/// Convenience result alias for cimage runtime operations.
pub type CImageRuntimeResult<T> = Result<T, CImageRuntimeError>;
