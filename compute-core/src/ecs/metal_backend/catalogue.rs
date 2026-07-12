//! MetalImplementationCatalogue — registry of all Metal kernel implementations.
//!
//! Each implementation (megakernel, per-layer, fused, primitive) registers
//! against a semantic contract with its ABI and supported configurations.

use crate::ecs::canonical::kernel_abi::{
    BufferBinding, ConstantBinding, DispatchGeometryPolicy, KernelAbi, KernelImplementationId,
    KernelSemanticId, MetalImplementationRegistration,
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
            source_path: Some("src/ecs/compute_image/megakernel/shaders/gemma4_full.metal".into()),
            source_entry_point: Some("gemma4_full_decode_persistent".into()),
            abi: KernelAbi {
                version: 1,
                buffers: vec![
                    BufferBinding {
                        slot: 0,
                        name: "weights".into(),
                        byte_size: 0,
                        optional: false,
                    },
                    BufferBinding {
                        slot: 1,
                        name: "activations".into(),
                        byte_size: 0,
                        optional: false,
                    },
                    BufferBinding {
                        slot: 2,
                        name: "kv_cache".into(),
                        byte_size: 0,
                        optional: false,
                    },
                    BufferBinding {
                        slot: 3,
                        name: "constants".into(),
                        byte_size: 0,
                        optional: false,
                    },
                ],
                constants: vec![
                    ConstantBinding {
                        index: 0,
                        name: "hidden_size".into(),
                        default_value: None,
                    },
                    ConstantBinding {
                        index: 1,
                        name: "num_heads".into(),
                        default_value: None,
                    },
                    ConstantBinding {
                        index: 2,
                        name: "head_dim".into(),
                        default_value: None,
                    },
                    ConstantBinding {
                        index: 3,
                        name: "num_layers".into(),
                        default_value: None,
                    },
                    ConstantBinding {
                        index: 4,
                        name: "seq_len".into(),
                        default_value: None,
                    },
                ],
                threadgroup_memory: vec![],
                dispatch_geometry: DispatchGeometryPolicy::FromConstant,
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
            source_path: Some(
                "src/ecs/compute_image/megakernel/shaders/decode_per_layer.metal".into(),
            ),
            source_entry_point: Some("decode_per_layer".into()),
            abi: KernelAbi {
                version: 1,
                buffers: vec![
                    BufferBinding {
                        slot: 0,
                        name: "weights".into(),
                        byte_size: 0,
                        optional: false,
                    },
                    BufferBinding {
                        slot: 1,
                        name: "activations".into(),
                        byte_size: 0,
                        optional: false,
                    },
                    BufferBinding {
                        slot: 2,
                        name: "kv_cache".into(),
                        byte_size: 0,
                        optional: false,
                    },
                ],
                constants: vec![
                    ConstantBinding {
                        index: 0,
                        name: "hidden_size".into(),
                        default_value: None,
                    },
                    ConstantBinding {
                        index: 1,
                        name: "num_heads".into(),
                        default_value: None,
                    },
                    ConstantBinding {
                        index: 2,
                        name: "head_dim".into(),
                        default_value: None,
                    },
                ],
                threadgroup_memory: vec![],
                dispatch_geometry: DispatchGeometryPolicy::FromOutputBuffer,
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
                source_path: None,
                source_entry_point: None,
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

    /// Register NF4 linear primitive kernel implementation.
    pub fn register_linear_nf4(&mut self) {
        self.register(MetalImplementationRegistration {
            semantic_id: KernelSemanticId("prism.linear.nf4.v1".into()),
            implementation_id: KernelImplementationId("metal.primitive.linear_nf4.v1".into()),
            supported_architectures: vec![],
            supported_representations: vec![TensorRepresentation::Nf4Tile640(128)],
            source_path: Some("src/ecs/compute_image/templates/cimage_linear_nf4.metal".into()),
            source_entry_point: Some("cimage_linear_nf4".into()),
            abi: KernelAbi {
                version: 1,
                buffers: vec![
                    BufferBinding {
                        slot: 0,
                        name: "weights_packed".into(),
                        byte_size: 0,
                        optional: false,
                    },
                    BufferBinding {
                        slot: 1,
                        name: "scales".into(),
                        byte_size: 0,
                        optional: false,
                    },
                    BufferBinding {
                        slot: 2,
                        name: "input".into(),
                        byte_size: 0,
                        optional: false,
                    },
                    BufferBinding {
                        slot: 3,
                        name: "output".into(),
                        byte_size: 0,
                        optional: false,
                    },
                ],
                constants: vec![
                    ConstantBinding {
                        index: 0,
                        name: "in_features".into(),
                        default_value: None,
                    },
                    ConstantBinding {
                        index: 1,
                        name: "out_features".into(),
                        default_value: None,
                    },
                    ConstantBinding {
                        index: 2,
                        name: "group_size".into(),
                        default_value: None,
                    },
                ],
                threadgroup_memory: vec![],
                dispatch_geometry: DispatchGeometryPolicy::FromOutputBuffer,
                threads_per_threadgroup: (64, 1, 1),
            },
        });
    }

    /// Register ternary GEMV primitive kernel implementation.
    pub fn register_ternary_gemv(&mut self) {
        self.register(MetalImplementationRegistration {
            semantic_id: KernelSemanticId("prism.ternary.gemv.v1".into()),
            implementation_id: KernelImplementationId("metal.primitive.ternary_gemv.v1".into()),
            supported_architectures: vec![],
            supported_representations: vec![TensorRepresentation::TernaryTile640],
            source_path: Some("src/ecs/compute_image/templates/ternary_tile640_gemv.metal".into()),
            source_entry_point: Some("ternary_tile640_gemv".into()),
            abi: KernelAbi {
                version: 1,
                buffers: vec![
                    BufferBinding {
                        slot: 0,
                        name: "ternary_weights".into(),
                        byte_size: 0,
                        optional: false,
                    },
                    BufferBinding {
                        slot: 1,
                        name: "scales".into(),
                        byte_size: 0,
                        optional: false,
                    },
                    BufferBinding {
                        slot: 2,
                        name: "input".into(),
                        byte_size: 0,
                        optional: false,
                    },
                    BufferBinding {
                        slot: 3,
                        name: "output".into(),
                        byte_size: 0,
                        optional: false,
                    },
                ],
                constants: vec![],
                threadgroup_memory: vec![],
                dispatch_geometry: DispatchGeometryPolicy::FromOutputBuffer,
                threads_per_threadgroup: (64, 1, 1),
            },
        });
    }

    /// Register RMSNorm primitive kernel implementation.
    pub fn register_rmsnorm(&mut self) {
        self.register(MetalImplementationRegistration {
            semantic_id: KernelSemanticId("prism.rmsnorm.v1".into()),
            implementation_id: KernelImplementationId("metal.primitive.rmsnorm.v1".into()),
            supported_architectures: vec![],
            supported_representations: vec![TensorRepresentation::Fp32],
            source_path: Some("src/ecs/compute_image/templates/cimage_rmsnorm_f32.metal".into()),
            source_entry_point: Some("cimage_rmsnorm_f32".into()),
            abi: KernelAbi {
                version: 1,
                buffers: vec![
                    BufferBinding {
                        slot: 0,
                        name: "input".into(),
                        byte_size: 0,
                        optional: false,
                    },
                    BufferBinding {
                        slot: 1,
                        name: "weight".into(),
                        byte_size: 0,
                        optional: false,
                    },
                    BufferBinding {
                        slot: 2,
                        name: "output".into(),
                        byte_size: 0,
                        optional: false,
                    },
                ],
                constants: vec![
                    ConstantBinding {
                        index: 0,
                        name: "hidden_size".into(),
                        default_value: None,
                    },
                    ConstantBinding {
                        index: 1,
                        name: "epsilon_f".into(),
                        default_value: None,
                    },
                ],
                threadgroup_memory: vec![],
                dispatch_geometry: DispatchGeometryPolicy::FromOutputBuffer,
                threads_per_threadgroup: (64, 1, 1),
            },
        });
    }
}

impl Default for MetalImplementationCatalogue {
    fn default() -> Self {
        let mut cat = Self::new();
        cat.register_megakernel();
        cat.register_per_layer();
        cat.register_linear_nf4();
        cat.register_ternary_gemv();
        cat.register_rmsnorm();
        cat.register_primitives();
        cat
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_catalogue_default_has_all_registrations() {
        let catalogue = MetalImplementationCatalogue::default();
        assert!(
            catalogue.len() >= 12,
            "expected >=12 registrations, got {}",
            catalogue.len()
        );

        // Verify individual registrations exist
        assert!(
            catalogue
                .for_semantic(&KernelSemanticId(
                    "prism.transformer.gemma4.decode.v1".into()
                ))
                .len()
                > 0,
            "megakernel should be registered"
        );
        assert!(
            catalogue
                .for_semantic(&KernelSemanticId("prism.linear.nf4.v1".into()))
                .len()
                >= 1,
            "linear_nf4 should be registered"
        );
        assert!(
            catalogue
                .for_semantic(&KernelSemanticId("prism.rmsnorm.v1".into()))
                .len()
                >= 1,
            "rmsnorm should be registered"
        );
    }

    #[test]
    fn test_linear_nf4_has_source_path() {
        let catalogue = MetalImplementationCatalogue::default();
        let registrations = catalogue.for_semantic(&KernelSemanticId("prism.linear.nf4.v1".into()));
        let nf4_reg = registrations
            .iter()
            .find(|r| r.implementation_id.0 == "metal.primitive.linear_nf4.v1")
            .expect("linear_nf4 registration should exist");
        assert!(
            nf4_reg.source_path.is_some(),
            "linear_nf4 should have source_path"
        );
        assert!(
            nf4_reg.source_entry_point.is_some(),
            "linear_nf4 should have entry_point"
        );
        assert!(
            nf4_reg.abi.buffers.len() >= 4,
            "linear_nf4 abi should have >=4 buffer bindings"
        );
        assert!(
            nf4_reg.abi.constants.len() >= 3,
            "linear_nf4 abi should have >=3 constants"
        );
    }

    #[test]
    fn test_ternary_gemv_has_source_path() {
        let catalogue = MetalImplementationCatalogue::default();
        let registrations =
            catalogue.for_semantic(&KernelSemanticId("prism.ternary.gemv.v1".into()));
        let t_reg = registrations
            .iter()
            .find(|r| r.implementation_id.0 == "metal.primitive.ternary_gemv.v1")
            .expect("ternary_gemv registration should exist");
        assert!(t_reg.source_path.is_some());
        assert!(t_reg.source_entry_point.is_some());
    }

    #[test]
    fn test_rmsnorm_has_source_path() {
        let catalogue = MetalImplementationCatalogue::default();
        let registrations = catalogue.for_semantic(&KernelSemanticId("prism.rmsnorm.v1".into()));
        let r_reg = registrations
            .iter()
            .find(|r| r.implementation_id.0 == "metal.primitive.rmsnorm.v1")
            .expect("rmsnorm registration should exist");
        assert!(r_reg.source_path.is_some());
        assert!(r_reg.source_entry_point.is_some());
        assert!(
            r_reg.abi.constants.len() >= 2,
            "rmsnorm abi should have >=2 constants"
        );
    }

    #[test]
    fn test_megakernel_registration_has_source_and_abi() {
        let catalogue = MetalImplementationCatalogue::default();
        let m = catalogue
            .for_semantic(&KernelSemanticId(
                "prism.transformer.gemma4.decode.v1".into(),
            ))
            .into_iter()
            .find(|r| r.implementation_id.0 == "metal.megakernel.gemma4.decode.v1")
            .expect("megakernel registration should exist");
        assert!(
            m.source_path.is_some(),
            "megakernel should have source_path"
        );
        assert!(
            m.source_entry_point.is_some(),
            "megakernel should have entry_point"
        );
        assert!(
            m.abi.buffers.len() >= 4,
            "megakernel abi should have >=4 buffers"
        );
        assert!(
            m.abi.constants.len() >= 5,
            "megakernel abi should have >=5 constants"
        );
    }
}
