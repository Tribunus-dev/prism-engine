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
use std::sync::LazyLock;

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
        self.register(MetalImplementationRegistration {
            semantic_id: KernelSemanticId("prism.linear.rawf32.v1".into()),
            implementation_id: KernelImplementationId("metal.primitive.linear_rawf32.v1".into()),
            supported_architectures: vec![],
            supported_representations: vec![],
            source_path: Some("src/ecs/compute_image/templates/cimage_linear_rawf32.metal".into()),
            source_entry_point: Some("cimage_linear_rawf32".into()),
            abi: KernelAbi {
                version: 1,
                buffers: vec![],
                constants: vec![],
                threadgroup_memory: vec![],
                dispatch_geometry: DispatchGeometryPolicy::FromOutputBuffer,
                threads_per_threadgroup: (64, 1, 1),
            },
        });
        self.register(MetalImplementationRegistration {
            semantic_id: KernelSemanticId("prism.silu.v1".into()),
            implementation_id: KernelImplementationId("metal.primitive.silu.v1".into()),
            supported_architectures: vec![],
            supported_representations: vec![],
            source_path: Some("src/ecs/compute_image/templates/cimage_silu_f32.metal".into()),
            source_entry_point: Some("cimage_silu_f32".into()),
            abi: KernelAbi {
                version: 1,
                buffers: vec![],
                constants: vec![],
                threadgroup_memory: vec![],
                dispatch_geometry: DispatchGeometryPolicy::FromOutputBuffer,
                threads_per_threadgroup: (64, 1, 1),
            },
        });
        self.register(MetalImplementationRegistration {
            semantic_id: KernelSemanticId("prism.rope.partial.v1".into()),
            implementation_id: KernelImplementationId("metal.primitive.rope.v1".into()),
            supported_architectures: vec![],
            supported_representations: vec![],
            source_path: Some("src/ecs/compute_image/templates/cimage_rope_f32.metal".into()),
            source_entry_point: Some("cimage_rope_f32".into()),
            abi: KernelAbi {
                version: 1,
                buffers: vec![],
                constants: vec![],
                threadgroup_memory: vec![],
                dispatch_geometry: DispatchGeometryPolicy::FromOutputBuffer,
                threads_per_threadgroup: (64, 1, 1),
            },
        });
        self.register(MetalImplementationRegistration {
            semantic_id: KernelSemanticId("prism.attention.scores.v1".into()),
            implementation_id: KernelImplementationId("metal.primitive.attention_scores.v1".into()),
            supported_architectures: vec![],
            supported_representations: vec![],
            source_path: Some(
                "src/ecs/compute_image/templates/cimage_attention_scores_f32.metal".into(),
            ),
            source_entry_point: Some("cimage_attention_scores_f32".into()),
            abi: KernelAbi {
                version: 1,
                buffers: vec![],
                constants: vec![],
                threadgroup_memory: vec![],
                dispatch_geometry: DispatchGeometryPolicy::FromOutputBuffer,
                threads_per_threadgroup: (64, 1, 1),
            },
        });
        self.register(MetalImplementationRegistration {
            semantic_id: KernelSemanticId("prism.attention.softmax.v1".into()),
            implementation_id: KernelImplementationId(
                "metal.primitive.attention_softmax.v1".into(),
            ),
            supported_architectures: vec![],
            supported_representations: vec![],
            source_path: Some(
                "src/ecs/compute_image/templates/cimage_attention_softmax_f32.metal".into(),
            ),
            source_entry_point: Some("cimage_attention_softmax_f32".into()),
            abi: KernelAbi {
                version: 1,
                buffers: vec![],
                constants: vec![],
                threadgroup_memory: vec![],
                dispatch_geometry: DispatchGeometryPolicy::FromOutputBuffer,
                threads_per_threadgroup: (64, 1, 1),
            },
        });
        self.register(MetalImplementationRegistration {
            semantic_id: KernelSemanticId("prism.attention.apply.v1".into()),
            implementation_id: KernelImplementationId("metal.primitive.attention_apply.v1".into()),
            supported_architectures: vec![],
            supported_representations: vec![],
            source_path: Some(
                "src/ecs/compute_image/templates/cimage_attention_apply_f32.metal".into(),
            ),
            source_entry_point: Some("cimage_attention_apply_f32".into()),
            abi: KernelAbi {
                version: 1,
                buffers: vec![],
                constants: vec![],
                threadgroup_memory: vec![],
                dispatch_geometry: DispatchGeometryPolicy::FromOutputBuffer,
                threads_per_threadgroup: (64, 1, 1),
            },
        });
        self.register(MetalImplementationRegistration {
            semantic_id: KernelSemanticId("prism.residual_add.v1".into()),
            implementation_id: KernelImplementationId("metal.primitive.residual_add.v1".into()),
            supported_architectures: vec![],
            supported_representations: vec![],
            source_path: Some(
                "src/ecs/compute_image/templates/cimage_residual_add_f32.metal".into(),
            ),
            source_entry_point: Some("cimage_residual_add_f32".into()),
            abi: KernelAbi {
                version: 1,
                buffers: vec![],
                constants: vec![],
                threadgroup_memory: vec![],
                dispatch_geometry: DispatchGeometryPolicy::FromOutputBuffer,
                threads_per_threadgroup: (64, 1, 1),
            },
        });
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
                        name: "input".into(),
                        byte_size: 0,
                        optional: false,
                    },
                    BufferBinding {
                        slot: 1,
                        name: "codes".into(),
                        byte_size: 0,
                        optional: false,
                    },
                    BufferBinding {
                        slot: 2,
                        name: "scales".into(),
                        byte_size: 0,
                        optional: false,
                    },
                    BufferBinding {
                        slot: 3,
                        name: "biases".into(),
                        byte_size: 0,
                        optional: false,
                    },
                    BufferBinding {
                        slot: 4,
                        name: "output".into(),
                        byte_size: 0,
                        optional: false,
                    },
                    BufferBinding {
                        slot: 5,
                        name: "constants".into(),
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

    /// Register palettized GEMV primitive kernel.
    pub fn register_palettized_gemv(&mut self) {
        self.register(MetalImplementationRegistration {
            semantic_id: KernelSemanticId("prism.palettized.gemv.v1".into()),
            implementation_id: KernelImplementationId("metal.primitive.palettized_gemv.v1".into()),
            supported_architectures: vec![],
            supported_representations: vec![],
            source_path: Some("src/ecs/compute_image/templates/palettized_gemv.metal".into()),
            source_entry_point: Some("palettized_gemv".into()),
            abi: KernelAbi {
                version: 1,
                buffers: vec![],
                constants: vec![],
                threadgroup_memory: vec![],
                dispatch_geometry: DispatchGeometryPolicy::FromOutputBuffer,
                threads_per_threadgroup: (64, 1, 1),
            },
        });
    }

    /// Register palettized SWIGLU primitive kernel.
    pub fn register_palettized_swiglu(&mut self) {
        self.register(MetalImplementationRegistration {
            semantic_id: KernelSemanticId("prism.palettized.swiglu.v1".into()),
            implementation_id: KernelImplementationId("metal.primitive.palettized_swiglu.v1".into()),
            supported_architectures: vec![],
            supported_representations: vec![],
            source_path: Some("src/ecs/compute_image/templates/palettized_gemv_swiglu.metal".into()),
            source_entry_point: Some("palettized_gemv_swiglu".into()),
            abi: KernelAbi {
                version: 1,
                buffers: vec![],
                constants: vec![],
                threadgroup_memory: vec![],
                dispatch_geometry: DispatchGeometryPolicy::FromOutputBuffer,
                threads_per_threadgroup: (64, 1, 1),
            },
        });
    }

    /// Register palettized GEMM primitive kernel.
    pub fn register_palettized_gemm(&mut self) {
        self.register(MetalImplementationRegistration {
            semantic_id: KernelSemanticId("prism.palettized.gemm.v1".into()),
            implementation_id: KernelImplementationId("metal.primitive.palettized_gemm.v1".into()),
            supported_architectures: vec![],
            supported_representations: vec![],
            source_path: Some("src/ecs/compute_image/templates/palettized_gemm.metal".into()),
            source_entry_point: Some("palettized_gemm".into()),
            abi: KernelAbi {
                version: 1,
                buffers: vec![],
                constants: vec![],
                threadgroup_memory: vec![],
                dispatch_geometry: DispatchGeometryPolicy::FromOutputBuffer,
                threads_per_threadgroup: (64, 1, 1),
            },
        });
    }

    /// Register NF4 tile640 GEMV primitive kernel.
    pub fn register_nf4_tile640_gemv(&mut self) {
        self.register(MetalImplementationRegistration {
            semantic_id: KernelSemanticId("prism.nf4.tile640.gemv.v1".into()),
            implementation_id: KernelImplementationId("metal.primitive.nf4_tile640_gemv.v1".into()),
            supported_architectures: vec![],
            supported_representations: vec![],
            source_path: Some("src/ecs/compute_image/templates/nf4_tile640_gemv.metal".into()),
            source_entry_point: Some("nf4_tile640_gemv".into()),
            abi: KernelAbi {
                version: 1,
                buffers: vec![],
                constants: vec![],
                threadgroup_memory: vec![],
                dispatch_geometry: DispatchGeometryPolicy::FromOutputBuffer,
                threads_per_threadgroup: (64, 1, 1),
            },
        });
    }

    /// Register ternary GEMV standard (v2) primitive kernel.
    pub fn register_ternary_gemv_standard(&mut self) {
        self.register(MetalImplementationRegistration {
            semantic_id: KernelSemanticId("prism.ternary.gemv.v2".into()),
            implementation_id: KernelImplementationId("metal.primitive.ternary_gemv.v2".into()),
            supported_architectures: vec![],
            supported_representations: vec![],
            source_path: Some("src/ecs/compute_image/templates/ternary_gemv.metal".into()),
            source_entry_point: Some("ternary_gemv".into()),
            abi: KernelAbi {
                version: 1,
                buffers: vec![],
                constants: vec![],
                threadgroup_memory: vec![],
                dispatch_geometry: DispatchGeometryPolicy::FromOutputBuffer,
                threads_per_threadgroup: (64, 1, 1),
            },
        });
    }

    /// Register ternary GEMM primitive kernel.
    pub fn register_ternary_gemm(&mut self) {
        self.register(MetalImplementationRegistration {
            semantic_id: KernelSemanticId("prism.ternary.gemm.v1".into()),
            implementation_id: KernelImplementationId("metal.primitive.ternary_gemm.v1".into()),
            supported_architectures: vec![],
            supported_representations: vec![],
            source_path: Some("src/ecs/compute_image/templates/ternary_gemm.metal".into()),
            source_entry_point: Some("ternary_gemm".into()),
            abi: KernelAbi {
                version: 1,
                buffers: vec![],
                constants: vec![],
                threadgroup_memory: vec![],
                dispatch_geometry: DispatchGeometryPolicy::FromOutputBuffer,
                threads_per_threadgroup: (64, 1, 1),
            },
        });
    }

    /// Register Q4 block-sym GEMV primitive kernel.
    pub fn register_q4_block_sym_gemv(&mut self) {
        self.register(MetalImplementationRegistration {
            semantic_id: KernelSemanticId("prism.q4.block_sym.gemv.v1".into()),
            implementation_id: KernelImplementationId("metal.primitive.q4_block_sym_gemv.v1".into()),
            supported_architectures: vec![],
            supported_representations: vec![],
            source_path: Some("src/ecs/compute_image/templates/q4_block_sym_gemv.metal".into()),
            source_entry_point: Some("q4_block_sym_gemv".into()),
            abi: KernelAbi {
                version: 1,
                buffers: vec![],
                constants: vec![],
                threadgroup_memory: vec![],
                dispatch_geometry: DispatchGeometryPolicy::FromOutputBuffer,
                threads_per_threadgroup: (64, 1, 1),
            },
        });
    }

    /// Register KV mixed primitive kernel.
    pub fn register_kv_mixed(&mut self) {
        self.register(MetalImplementationRegistration {
            semantic_id: KernelSemanticId("prism.kv.mixed.v1".into()),
            implementation_id: KernelImplementationId("metal.primitive.kv_mixed.v1".into()),
            supported_architectures: vec![],
            supported_representations: vec![],
            source_path: Some("src/ecs/compute_image/templates/kv_mixed.metal".into()),
            source_entry_point: Some("kv_mixed".into()),
            abi: KernelAbi {
                version: 1,
                buffers: vec![],
                constants: vec![],
                threadgroup_memory: vec![],
                dispatch_geometry: DispatchGeometryPolicy::FromOutputBuffer,
                threads_per_threadgroup: (64, 1, 1),
            },
        });
    }

    /// Register fused gate-up projection kernel.
    pub fn register_fused_gate_up(&mut self) {
        self.register(MetalImplementationRegistration {
            semantic_id: KernelSemanticId("prism.fused.gate_up.v1".into()),
            implementation_id: KernelImplementationId("metal.primitive.fused_gate_up.v1".into()),
            supported_architectures: vec![],
            supported_representations: vec![],
            source_path: Some("src/ecs/compute_image/templates/fused_gate_up.metal".into()),
            source_entry_point: Some("fused_gate_up".into()),
            abi: KernelAbi {
                version: 1,
                buffers: vec![],
                constants: vec![],
                threadgroup_memory: vec![],
                dispatch_geometry: DispatchGeometryPolicy::FromOutputBuffer,
                threads_per_threadgroup: (64, 1, 1),
            },
        });
    }

    /// Register tile640 pack kernels (pack, q8_0_ternary_pack, nf4_tile640_pack).
    pub fn register_tile640_pack(&mut self) {
        self.register(MetalImplementationRegistration {
            semantic_id: KernelSemanticId("prism.tile640.pack.v1".into()),
            implementation_id: KernelImplementationId("metal.primitive.tile640_pack.v1".into()),
            supported_architectures: vec![],
            supported_representations: vec![],
            source_path: Some("src/ecs/compute_image/templates/tile640_pack.metal".into()),
            source_entry_point: Some("tile640_pack".into()),
            abi: KernelAbi {
                version: 1,
                buffers: vec![],
                constants: vec![],
                threadgroup_memory: vec![],
                dispatch_geometry: DispatchGeometryPolicy::FromOutputBuffer,
                threads_per_threadgroup: (64, 1, 1),
            },
        });
        self.register(MetalImplementationRegistration {
            semantic_id: KernelSemanticId("prism.tile640.pack.q8_0_ternary.v1".into()),
            implementation_id: KernelImplementationId(
                "metal.primitive.tile640_pack_q8_0_ternary.v1".into(),
            ),
            supported_architectures: vec![],
            supported_representations: vec![],
            source_path: Some("src/ecs/compute_image/templates/tile640_pack.metal".into()),
            source_entry_point: Some("q8_0_ternary_pack".into()),
            abi: KernelAbi {
                version: 1,
                buffers: vec![],
                constants: vec![],
                threadgroup_memory: vec![],
                dispatch_geometry: DispatchGeometryPolicy::FromOutputBuffer,
                threads_per_threadgroup: (64, 1, 1),
            },
        });
        self.register(MetalImplementationRegistration {
            semantic_id: KernelSemanticId("prism.tile640.pack.nf4.v1".into()),
            implementation_id: KernelImplementationId("metal.primitive.tile640_pack_nf4.v1".into()),
            supported_architectures: vec![],
            supported_representations: vec![],
            source_path: Some("src/ecs/compute_image/templates/tile640_pack.metal".into()),
            source_entry_point: Some("nf4_tile640_pack".into()),
            abi: KernelAbi {
                version: 1,
                buffers: vec![],
                constants: vec![],
                threadgroup_memory: vec![],
                dispatch_geometry: DispatchGeometryPolicy::FromOutputBuffer,
                threads_per_threadgroup: (64, 1, 1),
            },
        });
    }

    /// Register megakernel INT4 decode kernel.
    pub fn register_megakernel_int4(&mut self) {
        self.register(MetalImplementationRegistration {
            semantic_id: KernelSemanticId("prism.transformer.gemma4.decode.int4.v1".into()),
            implementation_id: KernelImplementationId("metal.megakernel.gemma4.decode.int4.v1".into()),
            supported_architectures: vec![ArchitectureId("gemma4".into())],
            supported_representations: vec![TensorRepresentation::TernaryTile640],
            source_path: Some("src/ecs/compute_image/megakernel/shaders/gemma4_full_int4.metal".into()),
            source_entry_point: Some("gemma4_full_decode_int4".into()),
            abi: KernelAbi {
                version: 1,
                buffers: vec![],
                constants: vec![],
                threadgroup_memory: vec![],
                dispatch_geometry: DispatchGeometryPolicy::FromOutputBuffer,
                threads_per_threadgroup: (64, 1, 1),
            },
        });
    }

    /// Register persistent GEMV kernel.
    pub fn register_persistent_gemv(&mut self) {
        self.register(MetalImplementationRegistration {
            semantic_id: KernelSemanticId("prism.persistent.gemv.v1".into()),
            implementation_id: KernelImplementationId("metal.primitive.persistent_gemv.v1".into()),
            supported_architectures: vec![],
            supported_representations: vec![],
            source_path: Some("src/ecs/compute_image/megakernel/shaders/persistent_gemv.metal".into()),
            source_entry_point: Some("persistent_gemv".into()),
            abi: KernelAbi {
                version: 1,
                buffers: vec![],
                constants: vec![],
                threadgroup_memory: vec![],
                dispatch_geometry: DispatchGeometryPolicy::FromOutputBuffer,
                threads_per_threadgroup: (64, 1, 1),
            },
        });
    }

    /// Register vision projection kernel.
    pub fn register_vision_projection(&mut self) {
        self.register(MetalImplementationRegistration {
            semantic_id: KernelSemanticId("prism.vision.projection.v1".into()),
            implementation_id: KernelImplementationId("metal.primitive.vision_projection.v1".into()),
            supported_architectures: vec![],
            supported_representations: vec![],
            source_path: Some(
                "src/ecs/compute_image/megakernel/shaders/vision_projection.metal".into(),
            ),
            source_entry_point: Some("vision_patch_embed".into()),
            abi: KernelAbi {
                version: 1,
                buffers: vec![],
                constants: vec![],
                threadgroup_memory: vec![],
                dispatch_geometry: DispatchGeometryPolicy::FromOutputBuffer,
                threads_per_threadgroup: (64, 1, 1),
            },
        });
    }

    /// Register TTS codec kernel.
    pub fn register_tts_codec(&mut self) {
        self.register(MetalImplementationRegistration {
            semantic_id: KernelSemanticId("prism.tts.codec.v1".into()),
            implementation_id: KernelImplementationId("metal.primitive.tts_codec.v1".into()),
            supported_architectures: vec![],
            supported_representations: vec![],
            source_path: Some("src/ecs/compute_image/shaders/tts_codec.metal".into()),
            source_entry_point: Some("codebook_gather".into()),
            abi: KernelAbi {
                version: 1,
                buffers: vec![],
                constants: vec![],
                threadgroup_memory: vec![],
                dispatch_geometry: DispatchGeometryPolicy::FromOutputBuffer,
                threads_per_threadgroup: (64, 1, 1),
            },
        });
    }

    /// Register NF4 tile640 dequant-mul kernel.
    pub fn register_nf4tile640(&mut self) {
        self.register(MetalImplementationRegistration {
            semantic_id: KernelSemanticId("prism.nf4tile640.dequant_mul.v1".into()),
            implementation_id: KernelImplementationId("metal.primitive.nf4tile640.v1".into()),
            supported_architectures: vec![],
            supported_representations: vec![],
            source_path: Some("src/ecs/compute_image/shaders/nf4tile640.metal".into()),
            source_entry_point: Some("dequant_mul_nf4tile640".into()),
            abi: KernelAbi {
                version: 1,
                buffers: vec![],
                constants: vec![],
                threadgroup_memory: vec![],
                dispatch_geometry: DispatchGeometryPolicy::FromOutputBuffer,
                threads_per_threadgroup: (64, 1, 1),
            },
        });
    }

    /// Register linear INT8 primitive kernel.
    pub fn register_linear_int8(&mut self) {
        self.register(MetalImplementationRegistration {
            semantic_id: KernelSemanticId("prism.linear.int8.v1".into()),
            implementation_id: KernelImplementationId("metal.primitive.linear_int8.v1".into()),
            supported_architectures: vec![],
            supported_representations: vec![],
            source_path: Some("src/ecs/compute_image/templates/cimage_linear_int8.metal".into()),
            source_entry_point: Some("cimage_linear_int8".into()),
            abi: KernelAbi {
                version: 1,
                buffers: vec![],
                constants: vec![],
                threadgroup_memory: vec![],
                dispatch_geometry: DispatchGeometryPolicy::FromOutputBuffer,
                threads_per_threadgroup: (64, 1, 1),
            },
        });
    }

    /// Register mul primitive kernel.
    pub fn register_mul(&mut self) {
        self.register(MetalImplementationRegistration {
            semantic_id: KernelSemanticId("prism.mul.v1".into()),
            implementation_id: KernelImplementationId("metal.primitive.mul.v1".into()),
            supported_architectures: vec![],
            supported_representations: vec![],
            source_path: Some("src/ecs/compute_image/templates/cimage_mul_f32.metal".into()),
            source_entry_point: Some("cimage_mul_f32".into()),
            abi: KernelAbi {
                version: 1,
                buffers: vec![],
                constants: vec![],
                threadgroup_memory: vec![],
                dispatch_geometry: DispatchGeometryPolicy::FromOutputBuffer,
                threads_per_threadgroup: (64, 1, 1),
            },
        });
    }

    /// Register KV append primitive kernel.
    pub fn register_kv_append(&mut self) {
        self.register(MetalImplementationRegistration {
            semantic_id: KernelSemanticId("prism.kv.append.v1".into()),
            implementation_id: KernelImplementationId("metal.primitive.kv_append.v1".into()),
            supported_architectures: vec![],
            supported_representations: vec![],
            source_path: Some("src/ecs/compute_image/templates/cimage_kv_append_f32.metal".into()),
            source_entry_point: Some("cimage_kv_append_f32".into()),
            abi: KernelAbi {
                version: 1,
                buffers: vec![],
                constants: vec![],
                threadgroup_memory: vec![],
                dispatch_geometry: DispatchGeometryPolicy::FromOutputBuffer,
                threads_per_threadgroup: (64, 1, 1),
            },
        });
    }

    /// Register ternary CImage GEMV primitive kernel.
    pub fn register_ternary_cimage_gemv(&mut self) {
        self.register(MetalImplementationRegistration {
            semantic_id: KernelSemanticId("prism.ternary.cimage.gemv.v1".into()),
            implementation_id: KernelImplementationId("metal.primitive.ternary_cimage_gemv.v1".into()),
            supported_architectures: vec![],
            supported_representations: vec![],
            source_path: Some("src/ecs/compute_image/templates/cimage_ternary_gemv_v1.metal".into()),
            source_entry_point: Some("cimage_ternary_gemv_v1".into()),
            abi: KernelAbi {
                version: 1,
                buffers: vec![],
                constants: vec![],
                threadgroup_memory: vec![],
                dispatch_geometry: DispatchGeometryPolicy::FromOutputBuffer,
                threads_per_threadgroup: (64, 1, 1),
            },
        });
    }

    /// Register f32-to-half conversion kernel.
    pub fn register_f32_to_half(&mut self) {
        self.register(MetalImplementationRegistration {
            semantic_id: KernelSemanticId("prism.convert.f32_to_half.v1".into()),
            implementation_id: KernelImplementationId("metal.primitive.f32_to_half.v1".into()),
            supported_architectures: vec![],
            supported_representations: vec![],
            source_path: Some("src/ecs/cimage/kernels/f32_to_half.metal".into()),
            source_entry_point: Some("f32_to_half".into()),
            abi: KernelAbi {
                version: 1,
                buffers: vec![],
                constants: vec![],
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
        cat.register_palettized_gemv();
        cat.register_palettized_swiglu();
        cat.register_palettized_gemm();
        cat.register_nf4_tile640_gemv();
        cat.register_ternary_gemv_standard();
        cat.register_ternary_gemm();
        cat.register_q4_block_sym_gemv();
        cat.register_kv_mixed();
        cat.register_fused_gate_up();
        cat.register_tile640_pack();
        cat.register_megakernel_int4();
        cat.register_persistent_gemv();
        cat.register_vision_projection();
        cat.register_tts_codec();
        cat.register_nf4tile640();
        cat.register_linear_int8();
        cat.register_mul();
        cat.register_kv_append();
        cat.register_ternary_cimage_gemv();
        cat.register_f32_to_half();
        cat
    }
}

/// Static default catalogue for production source resolution.
static DEFAULT_CATALOGUE: LazyLock<MetalImplementationCatalogue> = LazyLock::new(MetalImplementationCatalogue::default);

/// Retrieve source text for a kernel by semantic ID.
/// Returns None when the ID is not registered or the file cannot be read.
pub fn catalogue_source_for(semantic_id: &KernelSemanticId) -> Option<String> {
    let cat = &*DEFAULT_CATALOGUE;
    cat.iter()
        .find(|r| &r.semantic_id == semantic_id)
        .and_then(|r| r.source_path.as_ref())
        .and_then(|path| std::fs::read_to_string(path).ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_catalogue_default_has_all_registrations() {
        let catalogue = MetalImplementationCatalogue::default();
        assert!(
            catalogue.len() >= 32,
            "expected >=32 registrations, got {}",
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
        // Verify exact ABI matches the cimage_linear_nf4.metal kernel signature:
        //   buffer(0): input   (device const float*)
        //   buffer(1): codes   (device const uchar*)
        //   buffer(2): scales  (device const float*)
        //   buffer(3): biases  (device const float*)
        //   buffer(4): output  (device float*)
        //   buffer(5): constants (constant MlpConstants&)
        assert_eq!(
            nf4_reg.abi.buffers.len(),
            6,
            "linear_nf4 abi should have 6 buffer bindings (input, codes, scales, biases, output, constants)"
        );
        assert_eq!(
            nf4_reg.abi.buffers[0].name, "input",
            "buffer(0) should be input"
        );
        assert_eq!(nf4_reg.abi.buffers[0].slot, 0, "buffer(0) slot should be 0");
        assert_eq!(
            nf4_reg.abi.buffers[1].name, "codes",
            "buffer(1) should be codes"
        );
        assert_eq!(nf4_reg.abi.buffers[1].slot, 1, "buffer(1) slot should be 1");
        assert_eq!(
            nf4_reg.abi.buffers[2].name, "scales",
            "buffer(2) should be scales"
        );
        assert_eq!(nf4_reg.abi.buffers[2].slot, 2, "buffer(2) slot should be 2");
        assert_eq!(
            nf4_reg.abi.buffers[3].name, "biases",
            "buffer(3) should be biases"
        );
        assert_eq!(nf4_reg.abi.buffers[3].slot, 3, "buffer(3) slot should be 3");
        assert_eq!(
            nf4_reg.abi.buffers[4].name, "output",
            "buffer(4) should be output"
        );
        assert_eq!(nf4_reg.abi.buffers[4].slot, 4, "buffer(4) slot should be 4");
        assert_eq!(
            nf4_reg.abi.buffers[5].name, "constants",
            "buffer(5) should be constants (MlpConstants struct)"
        );
        assert_eq!(nf4_reg.abi.buffers[5].slot, 5, "buffer(5) slot should be 5");
        // Kernel uses no Metal function constants — all parameters are passed via buffers
        assert!(
            nf4_reg.abi.constants.is_empty(),
            "linear_nf4 should have no function constants (MlpConstants is a buffer at slot 5)"
        );
        assert_eq!(
            nf4_reg.abi.dispatch_geometry,
            DispatchGeometryPolicy::FromOutputBuffer,
            "linear_nf4 dispatch should be derived from output buffer size"
        );
    }

    #[test]
    fn test_catalogue_resolves_nf4_linear_with_authoritative_source() {
        // Prove the catalogue can resolve the NF4 linear kernel with
        // real source path, entry point, and ABI — the foundation for
        // migrating production dispatch from include_str! to catalogue lookups.
        let catalogue = MetalImplementationCatalogue::default();

        // 1. Semantic lookup returns exactly one registration
        let registrations = catalogue.for_semantic(&KernelSemanticId("prism.linear.nf4.v1".into()));
        assert_eq!(
            registrations.len(),
            1,
            "expected exactly one NF4 linear registration"
        );

        // 2. Implementation ID matches
        let nf4_reg = registrations[0];
        assert_eq!(nf4_reg.implementation_id.0, "metal.primitive.linear_nf4.v1");

        // 3. Source path is populated and points to the correct template
        let source_path = nf4_reg
            .source_path
            .as_ref()
            .expect("linear_nf4 should have source_path");
        assert!(
            source_path.contains("cimage_linear_nf4.metal"),
            "source_path should reference cimage_linear_nf4.metal, got: {source_path}"
        );

        // 4. Entry point matches the kernel function in the .metal file
        let entry_point = nf4_reg
            .source_entry_point
            .as_ref()
            .expect("linear_nf4 should have source_entry_point");
        assert_eq!(
            *entry_point, "cimage_linear_nf4",
            "entry point should be cimage_linear_nf4"
        );

        // 5. ABI has the exact 6 buffer slots matching the kernel signature
        assert_eq!(nf4_reg.abi.buffers.len(), 6);
        for (i, buf) in nf4_reg.abi.buffers.iter().enumerate() {
            assert_eq!(
                buf.slot as usize, i,
                "buffer slot index should match position for buffer {i}"
            );
            assert!(!buf.optional, "all NF4 linear buffers are required");
        }

        // 6. No function constants — MlpConstants is a constant address-space buffer
        assert!(nf4_reg.abi.constants.is_empty());

        // 7. Dispatch geometry policy is from output buffer (one thread per output element)
        assert_eq!(
            nf4_reg.abi.dispatch_geometry,
            DispatchGeometryPolicy::FromOutputBuffer
        );

        // 8. Supported representations include Nf4Tile640
        assert_eq!(
            nf4_reg.supported_representations,
            vec![TensorRepresentation::Nf4Tile640(128)]
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
