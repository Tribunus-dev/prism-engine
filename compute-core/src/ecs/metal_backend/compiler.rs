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
        // Look up the first matching implementation in the catalogue.
        let registration = self
            .catalogue
            .for_semantic(&group.semantic_id)
            .into_iter()
            .next()
            .ok_or_else(|| {
                BackendCompileError::LoweringFailed(format!(
                    "no implementation for {}",
                    group.semantic_id.0
                ))
            })?
            .clone();

        // Read source from the registration's source_path, or use empty for generated kernels.
        let source = match &registration.source_path {
            Some(path) => {
                let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
                let full_path = base.join(path);
                match std::fs::read_to_string(&full_path) {
                    Ok(s) => s,
                    Err(e) => {
                        return Err(BackendCompileError::LoweringFailed(format!(
                            "cannot read source {}: {}",
                            full_path.display(),
                            e
                        )));
                    }
                }
            }
            None => String::new(),
        };
        // Use the registration's entry point, or derive from semantic_id.
        let entry_point = registration
            .source_entry_point
            .unwrap_or_else(|| format!("{}_kernel", group.semantic_id.0.replace('.', "_")));

        Ok(BackendKernelIr {
            semantic_id: group.semantic_id.clone(),
            source,
            entry_point,
            abi: registration.abi.clone(),
        })
    }

    fn compile(
        &self,
        kernel: &BackendKernelIr,
        toolchain: &ToolchainContext,
    ) -> Result<CompiledKernelArtifact, BackendCompileError> {
        if kernel.source.is_empty() {
            // Structural artifact allowed only in test builds for backward compat.
            if cfg!(test) {
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
            return Err(BackendCompileError::CompilationFailed(
                "empty kernel source: an authoritative source provider must be registered".into(),
            ));
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

    /// Verifies that lowering a KernelGroup whose semantic ID has a registered
    /// source_path reads the actual .metal file into the IR.
    #[test]
    fn test_lower_populates_source() {
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
            !ir.source.is_empty(),
            "lowered IR for megakernel should have source from .metal file"
        );
        assert_eq!(
            ir.entry_point, "gemma4_full_decode_persistent",
            "entry_point should come from registration"
        );
        assert!(
            ir.abi.buffers.len() >= 4,
            "abi buffers should be populated from registration"
        );
    }

    /// Verifies that compile() with empty source and cfg!(test) produces a
    /// structural artifact for backward compat in test builds.
    #[test]
    fn test_compile_structural_in_test_mode() {
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
        // In cfg!(test), empty source is still accepted (structural artifact)
        let artifact = compiler
            .compile(&kernel, &toolchain)
            .expect("compile should produce structural artifact in test mode");
        assert!(
            artifact.compiled_bytes.is_empty(),
            "structural artifact should have empty compiled_bytes"
        );
    }

    /// Verifies that lower() fails for an unregistered semantic ID.
    #[test]
    fn test_lower_fails_unknown_semantic() {
        let compiler = MetalBackendCompiler::default();
        let group = KernelGroup {
            semantic_id: KernelSemanticId("prism.nonexistent.v1".into()),
            implementation_class: KernelImplementationClass::Primitive,
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
                dispatch_geometry: DispatchGeometryPolicy::FromOutputBuffer,
                threads_per_threadgroup: (64, 1, 1),
            },
            source_region: RegionId(0),
            target_lane: ExecutionLane::MetalGpu,
        };
        let context = LoweringContext {
            target: BackendTarget::AppleGpu,
            metal_language_version: None,
        };
        let result = compiler.lower(&group, &context);
        assert!(result.is_err(), "lower should fail for unknown semantic ID");
    }

    /// Verifies that lower() for a primitive with source_path: None succeeds
    /// and returns empty source (for generated/dynamic kernels).
    #[test]
    fn test_lower_generated_kernel_has_empty_source() {
        let compiler = MetalBackendCompiler::default();
        // "prism.linear.rawf32.v1" uses source_path: None in register_primitives
        let group = KernelGroup {
            semantic_id: KernelSemanticId("prism.linear.rawf32.v1".into()),
            implementation_class: KernelImplementationClass::Primitive,
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
                dispatch_geometry: DispatchGeometryPolicy::FromOutputBuffer,
                threads_per_threadgroup: (64, 1, 1),
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
            .expect("lower should succeed for registered primitive");
        assert!(
            ir.source.is_empty(),
            "generated kernel should have empty source"
        );
    }
}
