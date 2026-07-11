//! MetalBackendCompiler — the unified Metal compilation backend.
//!
//! Implements BackendCompiler for Apple GPU targets. Delegates to the
//! MetalImplementationCatalogue for registered implementations and
//! MetalToolchain for xcrun-based compilation.

use super::catalogue::MetalImplementationCatalogue;
use super::toolchain::MetalToolchain;
use super::{
    BackendCompileError, BackendCompiler, BackendKernelIr, BackendTarget, LoweringContext,
    ToolchainContext,
};
use crate::ecs::canonical::kernel_abi::{
    CompiledKernelArtifact, KernelAbi, KernelGroup, KernelImplementationId, KernelSemanticId,
};

/// The unified Metal backend compiler.
///
/// Owns the implementation catalogue and toolchain. All Metal kernel
/// compilation (megakernel, per-layer, fused, primitive, runtime,
/// AOT) goes through this single struct.
pub struct MetalBackendCompiler {
    pub catalogue: MetalImplementationCatalogue,
    pub toolchain: MetalToolchain,
}

impl MetalBackendCompiler {
    /// Create a new Metal backend with default catalogue and toolchain.
    pub fn new() -> Self {
        Self {
            catalogue: MetalImplementationCatalogue::default(),
            toolchain: MetalToolchain::default(),
        }
    }

    /// Create a Metal backend with a custom toolchain.
    pub fn with_toolchain(toolchain: MetalToolchain) -> Self {
        Self {
            catalogue: MetalImplementationCatalogue::default(),
            toolchain,
        }
    }

    /// Compile a Metal source string to a CompiledKernelArtifact.
    /// Convenience wrapper for quick compilation without the full
    /// lower+compile pipeline.
    pub fn compile_source(
        &self,
        name: &str,
        source: &str,
        entry_point: &str,
        semantic_id: &str,
        abi: KernelAbi,
    ) -> Result<CompiledKernelArtifact, BackendCompileError> {
        let output = self
            .toolchain
            .compile_source(name, source)
            .map_err(|e| BackendCompileError::CompilationFailed(e))?;

        Ok(CompiledKernelArtifact {
            implementation_id: KernelImplementationId(format!("metal.compiled.{name}")),
            semantic_id: KernelSemanticId(semantic_id.into()),
            compiled_bytes: output.metallib_bytes,
            sha256: output.sha256,
            entry_point: entry_point.into(),
            abi,
        })
    }

    /// Check whether the Metal toolchain is available.
    pub fn is_available(&self) -> bool {
        self.toolchain.is_available()
    }

    /// Return an iterator over all registered implementations.
    pub fn implementations(
        &self,
    ) -> impl Iterator<Item = &crate::ecs::canonical::kernel_abi::MetalImplementationRegistration>
    {
        self.catalogue.iter()
    }
}

impl BackendCompiler for MetalBackendCompiler {
    fn target(&self) -> BackendTarget {
        BackendTarget::AppleGpu
    }

    fn lower(
        &self,
        group: &KernelGroup,
        _context: &LoweringContext,
    ) -> Result<BackendKernelIr, BackendCompileError> {
        // Look up the semantic ID in the catalogue to find the implementation.
        let _registration = self
            .catalogue
            .for_semantic(&group.semantic_id)
            .into_iter()
            .next()
            .ok_or_else(|| {
                BackendCompileError::LoweringFailed(format!(
                    "no implementation for {}",
                    group.semantic_id.0
                ))
            })?;

        // Construct backend IR from the KernelGroup.
        // Actual source assembly happens in PR G (source consolidation).
        // For now, the IR carries the semantic identity and ABI so the
        // compile step can produce the artifact.
        Ok(BackendKernelIr {
            semantic_id: group.semantic_id.clone(),
            source: String::new(), // populated by source provider in PR G
            entry_point: format!("{}_kernel", group.semantic_id.0.replace('.', "_")),
            abi: group.abi.clone(),
        })
    }

    fn compile(
        &self,
        kernel: &BackendKernelIr,
        toolchain: &ToolchainContext,
    ) -> Result<CompiledKernelArtifact, BackendCompileError> {
        if kernel.source.is_empty() {
            // No source yet — return a structural artifact for pipeline compat.
            return Ok(CompiledKernelArtifact {
                implementation_id: KernelImplementationId(format!(
                    "metal.structural.{}",
                    kernel.semantic_id.0
                )),
                semantic_id: kernel.semantic_id.clone(),
                compiled_bytes: Vec::new(),
                sha256: String::new(),
                entry_point: kernel.entry_point.clone(),
                abi: kernel.abi.clone(),
            });
        }

        let tc = MetalToolchain::new(
            &toolchain.sdk,
            &toolchain.metal_std,
            &toolchain.optimization,
        );
        let output = tc
            .compile_source(&kernel.semantic_id.0, &kernel.source)
            .map_err(|e| BackendCompileError::CompilationFailed(e))?;

        Ok(CompiledKernelArtifact {
            implementation_id: KernelImplementationId(format!(
                "metal.compiled.{}",
                kernel.semantic_id.0
            )),
            semantic_id: kernel.semantic_id.clone(),
            compiled_bytes: output.metallib_bytes,
            sha256: output.sha256,
            entry_point: kernel.entry_point.clone(),
            abi: kernel.abi.clone(),
        })
    }
}

impl Default for MetalBackendCompiler {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for MetalBackendCompiler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalBackendCompiler")
            .field("catalogue_len", &self.catalogue.len())
            .field("toolchain_available", &self.toolchain.is_available())
            .finish()
    }
}
