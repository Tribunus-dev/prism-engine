//! MetalImplementationCatalogue — registry of all Metal kernel implementations.
//!
//! Each implementation (megakernel, per-layer, fused, primitive) registers
//! against a semantic contract with its ABI and supported configurations.

use crate::ecs::canonical::execution_graph::ExecutionLane;
use crate::ecs::canonical::kernel_abi::{
    KernelAbi, KernelGroup, KernelImplementationClass, KernelImplementationId, KernelSemanticId,
    MetalImplementationRegistration,
};
use crate::ecs::canonical::model_ir::ArchitectureId;
use crate::ecs::canonical::representation::TensorRepresentation;

/// Catalogue of all Metal kernel implementations.
#[derive(Debug, Clone)]
pub struct MetalImplementationCatalogue {
    implementations: Vec<MetalImplementationRegistration>,
}

impl MetalImplementationCatalogue {
    /// Create an empty catalogue.
    pub fn new() -> Self {
        Self {
            implementations: Vec::new(),
        }
    }

    /// Register a Metal kernel implementation.
    pub fn register(&mut self, registration: MetalImplementationRegistration) {
        self.implementations.push(registration);
    }

    /// Find all implementations for a given semantic ID.
    pub fn for_semantic(
        &self,
        semantic_id: &KernelSemanticId,
    ) -> Vec<&MetalImplementationRegistration> {
        self.implementations
            .iter()
            .filter(|r| r.semantic_id == *semantic_id)
            .collect()
    }

    /// Find an implementation by its implementation ID.
    pub fn by_id(&self, id: &KernelImplementationId) -> Option<&MetalImplementationRegistration> {
        self.implementations
            .iter()
            .find(|r| r.implementation_id == *id)
    }

    /// Number of registered implementations.
    pub fn len(&self) -> usize {
        self.implementations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.implementations.is_empty()
    }

    /// Iterate over all registered implementations.
    pub fn iter(&self) -> impl Iterator<Item = &MetalImplementationRegistration> {
        self.implementations.iter()
    }

    /// Register the built-in megakernel implementation.
    pub fn register_megakernel(&mut self) {
        self.register(MetalImplementationRegistration {
            semantic_id: KernelSemanticId("prism.transformer.gemma4.decode.v1".into()),
            implementation_id: KernelImplementationId("metal.megakernel.gemma4.decode.v1".into()),
            supported_architectures: vec![ArchitectureId("gemma4".into())],
            supported_representations: vec![
                TensorRepresentation::Nf4Tile640(128),
                TensorRepresentation::TernaryTile640,
            ],
            abi: KernelAbi {
                version: 1,
                buffers: vec![],
                constants: vec![],
                threadgroup_memory: vec![],
                dispatch_geometry:
                    crate::ecs::canonical::kernel_abi::DispatchGeometryPolicy::FromConstant,
                threads_per_threadgroup: (256, 1, 1),
            },
        });
    }

    /// Register the per-layer decoder kernel implementation.
    pub fn register_per_layer(&mut self) {
        self.register(MetalImplementationRegistration {
            semantic_id: KernelSemanticId("prism.decoder.per_layer.v1".into()),
            implementation_id: KernelImplementationId("metal.per_layer.decode.v1".into()),
            supported_architectures: vec![
                ArchitectureId("gemma4".into()),
                ArchitectureId("llama".into()),
            ],
            supported_representations: vec![TensorRepresentation::Fp32],
            abi: KernelAbi {
                version: 1,
                buffers: vec![],
                constants: vec![],
                threadgroup_memory: vec![],
                dispatch_geometry:
                    crate::ecs::canonical::kernel_abi::DispatchGeometryPolicy::FromOutputBuffer,
                threads_per_threadgroup: (64, 1, 1),
            },
        });
    }

    /// Register primitive projection kernels (linear, RMSNorm, RoPE, etc.).
    pub fn register_primitives(&mut self) {
        for (name, semantic) in &[
            ("linear_nf4", "prism.linear.nf4.v1"),
            ("linear_rawf32", "prism.linear.rawf32.v1"),
            ("rmsnorm", "prism.rmsnorm.v1"),
            ("silu", "prism.silu.v1"),
            ("rope", "prism.rope.partial.v1"),
            ("attention_scores", "prism.attention.scores.v1"),
            ("attention_softmax", "prism.attention.softmax.v1"),
            ("attention_apply", "prism.attention.apply.v1"),
            ("residual_add", "prism.residual_add.v1"),
        ] {
            self.register(MetalImplementationRegistration {
                semantic_id: KernelSemanticId(semantic.to_string()),
                implementation_id: KernelImplementationId(format!("metal.primitive.{}.v1", name)),
                supported_architectures: vec![],
                supported_representations: vec![],
                abi: KernelAbi {
                    version: 1,
                    buffers: vec![],
                    constants: vec![],
                    threadgroup_memory: vec![],
                    dispatch_geometry:
                        crate::ecs::canonical::kernel_abi::DispatchGeometryPolicy::FromOutputBuffer,
                    threads_per_threadgroup: (64, 1, 1),
                },
            });
        }
    }
}

impl Default for MetalImplementationCatalogue {
    fn default() -> Self {
        let mut cat = Self::new();
        cat.register_megakernel();
        cat.register_per_layer();
        cat.register_primitives();
        cat
    }
}
