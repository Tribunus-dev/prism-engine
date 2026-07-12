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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::canonical::execution_graph::{ExecutionLane, RegionId};
    use crate::ecs::canonical::kernel_abi::{
        DispatchGeometryPolicy, KernelImplementationClass, SpecializationParameters,
    };

    /// Verifies that lowering a KernelGroup whose semantic ID exists in the
    /// default catalogue produces BackendKernelIr with empty source.
    /// Documents the structural gap: source assembly is deferred (PR G).
    #[test]
    fn test_lower_produces_empty_source() {
        let compiler = MetalBackendCompiler::default();
        let group = KernelGroup {
            semantic_id: KernelSemanticId("prism.transformer.gemma4.decode.v1".into()),
            implementation_class: KernelImplementationClass::PersistentTransformer,
            operations: vec![],
            specialization: SpecializationParameters {
                tile_m: None,
                tile_k: None,
                tile_n: None,
                group_size: None,
                metadata_layout: None,
            },
            abi: KernelAbi {
                version: 1,
                buffers: vec![],
                constants: vec![],
                threadgroup_memory: vec![],
                dispatch_geometry: DispatchGeometryPolicy::FromConstant,
                threads_per_threadgroup: (256, 1, 1),
            },
            source_region: RegionId(0),
            target_lane: ExecutionLane::MetalGpu,
        };
        let context = LoweringContext {
            target: BackendTarget::AppleGpu,
            metal_language_version: None,
        };
        let ir = compiler
            .lower(&group, &context)
            .expect("lower should succeed for known semantic ID");
        assert!(
            ir.source.is_empty(),
            "lowered IR should have empty source, got {} bytes",
            ir.source.len()
        );
    }

    /// Verifies that compile() accepts a BackendKernelIr with empty source
    /// and returns a structural artifact with empty compiled_bytes.
    /// Documents the structural gap: real Metal compilation does not run
    /// until source is populated.
    #[test]
    fn test_compile_accepts_empty_source() {
        let compiler = MetalBackendCompiler::default();
        let kernel = BackendKernelIr {
            semantic_id: KernelSemanticId("prism.transformer.gemma4.decode.v1".into()),
            source: String::new(),
            entry_point: "gemma4_decode_kernel".into(),
            abi: KernelAbi {
                version: 1,
                buffers: vec![],
                constants: vec![],
                threadgroup_memory: vec![],
                dispatch_geometry: DispatchGeometryPolicy::FromConstant,
                threads_per_threadgroup: (256, 1, 1),
            },
        };
        let toolchain = ToolchainContext::default();
        let artifact = compiler
            .compile(&kernel, &toolchain)
            .expect("compile should succeed with empty source");
        assert!(
            artifact.compiled_bytes.is_empty(),
            "compiled artifact should have empty compiled_bytes, got {}",
            artifact.compiled_bytes.len()
        );
    }
}
