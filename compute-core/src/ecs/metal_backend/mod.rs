//! MetalBackend — unified Metal compilation backend.
//!
//! Implements the BackendCompiler trait for Apple GPU targets.
//! All existing Metal compilation paths (megakernel, per-layer, fused,
//! primitive) register through the MetalImplementationCatalogue.
//!
//! PR D — Unified Metal backend. Register implementations here; the
//! actual dispatch routing moves to KernelPlan-based dispatch in PR E.

pub mod catalogue;
pub mod compiler;
pub mod toolchain;

pub use catalogue::*;
pub use compiler::*;
pub use toolchain::*;

use prism_ecs_constitutional::canonical::kernel_abi::{
    CompiledKernelArtifact, KernelAbi, KernelGroup, KernelSemanticId,
};

/// Identifies the target for a BackendCompiler.
#[derive(Debug, Clone, PartialEq)]
pub enum BackendTarget {
    AppleGpu,
    AppleNeuralEngine,
}

/// Context passed to the backend during lowering.
#[derive(Debug, Clone)]
pub struct LoweringContext {
    pub target: BackendTarget,
    pub metal_language_version: Option<String>,
}

/// Context for the toolchain invocation.
#[derive(Debug, Clone)]
pub struct ToolchainContext {
    pub sdk: String,
    pub metal_std: String,
    pub optimization: String,
}

impl Default for ToolchainContext {
    fn default() -> Self {
        Self {
            sdk: "macosx".into(),
            metal_std: "metal4.0".into(),
            optimization: "-O3".into(),
        }
    }
}

/// Error from a backend compilation step.
#[derive(Debug, Clone, thiserror::Error)]
pub enum BackendCompileError {
    #[error("lowering failed: {0}")]
    LoweringFailed(String),
    #[error("compilation failed: {0}")]
    CompilationFailed(String),
    #[error("toolchain not found: {0}")]
    ToolchainNotFound(String),
}

/// Intermediate representation produced by backend lowering.
#[derive(Debug, Clone)]
pub struct BackendKernelIr {
    pub semantic_id: KernelSemanticId,
    pub source: String,
    pub entry_point: String,
    pub abi: KernelAbi,
}

/// A compiler backend — lowers KernelGroup to compiled artifacts.
pub trait BackendCompiler: Send + Sync {
    /// The target this backend compiles for.
    fn target(&self) -> BackendTarget;

    /// Lower a KernelGroup to backend-specific IR.
    fn lower(
        &self,
        group: &KernelGroup,
        context: &LoweringContext,
    ) -> Result<BackendKernelIr, BackendCompileError>;

    /// Compile backend IR into a sealed artifact.
    fn compile(
        &self,
        kernel: &BackendKernelIr,
        toolchain: &ToolchainContext,
    ) -> Result<CompiledKernelArtifact, BackendCompileError>;
}
