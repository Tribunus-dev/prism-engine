//! Generation management API — inspect, promote, rollback, list.
//!
//! Plan Section 15 Phase 13: "Server APIs, progress, generation management,
//! evidence dashboard."
//!
//! Provides an HTTP-accessible API surface for managing compiled generations,
//! including listing generations, promoting new generations atomically,
//! rolling back to a parent generation, and storing content-addressed payloads.

use crate::ecs::canonical::generation::CimageGeneration;
use crate::ecs::canonical::identity::*;
use crate::ecs::cimage::generation_store::{ContentStore, GenerationStore, PromotionTransaction};

/// Generation management API — inspect, promote, rollback, list.
///
/// Wraps the `ContentStore` and `GenerationStore` with an ergonomic,
/// HTTP-accessible interface for lifecycle operations.
pub struct GenerationApi {
    pub content_store: ContentStore,
    pub generation_store: GenerationStore,
}

impl GenerationApi {
    /// Create a new empty API instance.
    pub fn new() -> Self {
        Self {
            content_store: ContentStore::new(),
            generation_store: GenerationStore::new(),
        }
    }

    /// List all generation IDs currently known to the store.
    ///
    pub fn list_generations(&self) -> Vec<GenerationId> {
        self.generation_store.ids()
    }

    /// Get the current (promoted) generation, if any.
    pub fn current_generation(&self) -> Option<&CimageGeneration> {
        self.generation_store.current()
    }

    /// Atomically promote a new generation.
    ///
    /// Validates that all referenced payloads exist and the parent
    /// generation is known, then commits the generation as current.
    pub fn promote(&mut self, generation: CimageGeneration) -> Result<GenerationId, String> {
        let mut tx = PromotionTransaction::new(
            &mut self.content_store,
            &mut self.generation_store,
            generation,
        );
        tx.commit()
    }

    /// Rollback to the parent generation.
    ///
    /// Fails if there is no current generation or no parent.
    pub fn rollback(&mut self) -> Result<GenerationId, String> {
        let current = self
            .generation_store
            .current()
            .ok_or_else(|| "no current generation".to_string())?;
        let parent = current
            .parent_generation
            .clone()
            .ok_or_else(|| "no parent generation".to_string())?;
        self.generation_store.set_current(parent.clone())?;
        Ok(parent)
    }

    /// Store a content-addressed payload by its canonical digest.
    pub fn store_payload(&mut self, id: PhysicalSegmentId, data: Vec<u8>) {
        self.content_store.store(id, data);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::canonical::generation::RepresentationBinding;
    use crate::ecs::canonical::{ExecutionGraph, MemoryPlan, RuntimeStatePlan};
    use crate::ecs::execution_profile::PhysicalTileLayout;
    use crate::ecs::plan::CodecFamily;
    use std::collections::BTreeMap;

    #[test]
    fn test_generation_api_promote() {
        let mut api = GenerationApi::new();

        // Store a payload
        let seg_id = PhysicalSegmentId("seg1".into());
        api.store_payload(seg_id.clone(), vec![1, 2, 3, 4]);

        // Create a minimal generation
        let mut bindings = BTreeMap::new();
        bindings.insert(
            LogicalTensorId("t1".into()),
            RepresentationBinding {
                representation_id: RepresentationId("r1".into()),
                codec: CodecFamily::Nf4,
                layout: PhysicalTileLayout::default(),
                primary_segment: seg_id,
                scale_segments: vec![],
                residual_segments: vec![],
                source_representation: None,
                acceptance_receipt: ReceiptId("r".into()),
            },
        );

        let gen = CimageGeneration {
            generation_id: GenerationId("gen1".into()),
            parent_generation: None,
            base_model: ModelSourceId("m".into()),
            compiler_identity: CompilerIdentity {
                name: "tc".into(),
                version: "1".into(),
                build_hash: None,
                build_timestamp: None,
            },
            hardware_profile: HardwareProfileId("h".into()),
            tensor_bindings: bindings,
            kernel_bindings: BTreeMap::new(),
            engram_bindings: BTreeMap::new(),
            execution_graph: ExecutionGraph {
                regions: vec![],
                edges: vec![],
                state: RuntimeStatePlan {
                    max_context_tokens: 4096,
                    kv_cache_bytes_per_token: 256,
                    total_kv_cache_bytes: 1048576,
                },
                memory: MemoryPlan {
                    total_activation_bytes: 0,
                    total_weight_bytes: 0,
                    arena_region_count: 0,
                },
            },
            receipt_root: ReceiptId("root".into()),
            created_at: Timestamp("t".into()),
        };

        let result = api.promote(gen);
        assert!(result.is_ok());
    }

    #[test]
    fn test_generation_api_rollback() {
        let mut api = GenerationApi::new();

        // Store payloads and create two generations
        let seg1 = PhysicalSegmentId("seg1".into());
        let seg2 = PhysicalSegmentId("seg2".into());
        api.store_payload(seg1.clone(), vec![1, 2, 3, 4]);
        api.store_payload(seg2.clone(), vec![5, 6, 7, 8]);

        fn make_binding(seg: PhysicalSegmentId) -> RepresentationBinding {
            RepresentationBinding {
                representation_id: RepresentationId("r".into()),
                codec: CodecFamily::Nf4,
                layout: PhysicalTileLayout::default(),
                primary_segment: seg,
                scale_segments: vec![],
                residual_segments: vec![],
                source_representation: None,
                acceptance_receipt: ReceiptId("r".into()),
            }
        }

        fn make_generation(
            id: &str,
            parent: Option<GenerationId>,
            bindings: BTreeMap<LogicalTensorId, RepresentationBinding>,
        ) -> CimageGeneration {
            CimageGeneration {
                generation_id: GenerationId(id.into()),
                parent_generation: parent,
                base_model: ModelSourceId("m".into()),
                compiler_identity: CompilerIdentity {
                    name: "tc".into(),
                    version: "1".into(),
                    build_hash: None,
                    build_timestamp: None,
                },
                hardware_profile: HardwareProfileId("h".into()),
                tensor_bindings: bindings,
                kernel_bindings: BTreeMap::new(),
                engram_bindings: BTreeMap::new(),
                execution_graph: ExecutionGraph {
                    regions: vec![],
                    edges: vec![],
                    state: RuntimeStatePlan {
                        max_context_tokens: 4096,
                        kv_cache_bytes_per_token: 256,
                        total_kv_cache_bytes: 1048576,
                    },
                    memory: MemoryPlan {
                        total_activation_bytes: 0,
                        total_weight_bytes: 0,
                        arena_region_count: 0,
                    },
                },
                receipt_root: ReceiptId("root".into()),
                created_at: Timestamp("t".into()),
            }
        }

        let mut b1 = BTreeMap::new();
        b1.insert(LogicalTensorId("t1".into()), make_binding(seg1));
        let gen1 = make_generation("gen1", None, b1);

        let mut b2 = BTreeMap::new();
        b2.insert(LogicalTensorId("t1".into()), make_binding(seg2));
        let gen2 = make_generation("gen2", Some(GenerationId("gen1".into())), b2);

        // Promote gen1, then gen2
        api.promote(gen1).unwrap();
        api.promote(gen2).unwrap();
        assert_eq!(
            api.generation_store.current_id(),
            Some(&GenerationId("gen2".into()))
        );

        // Rollback to gen1
        let rolled = api.rollback().unwrap();
        assert_eq!(rolled, GenerationId("gen1".into()));
        assert_eq!(
            api.generation_store.current_id(),
            Some(&GenerationId("gen1".into()))
        );
    }

    #[test]
    fn test_generation_api_rollback_fails_no_parent() {
        let mut api = GenerationApi::new();
        let seg = PhysicalSegmentId("seg".into());
        api.store_payload(seg.clone(), vec![1, 2, 3]);

        let mut bindings = BTreeMap::new();
        bindings.insert(
            LogicalTensorId("t".into()),
            RepresentationBinding {
                representation_id: RepresentationId("r".into()),
                codec: CodecFamily::Nf4,
                layout: PhysicalTileLayout::default(),
                primary_segment: seg,
                scale_segments: vec![],
                residual_segments: vec![],
                source_representation: None,
                acceptance_receipt: ReceiptId("r".into()),
            },
        );
        let gen = CimageGeneration {
            generation_id: GenerationId("g".into()),
            parent_generation: None,
            base_model: ModelSourceId("m".into()),
            compiler_identity: CompilerIdentity {
                name: "tc".into(),
                version: "1".into(),
                build_hash: None,
                build_timestamp: None,
            },
            hardware_profile: HardwareProfileId("h".into()),
            tensor_bindings: bindings,
            kernel_bindings: BTreeMap::new(),
            engram_bindings: BTreeMap::new(),
            execution_graph: ExecutionGraph {
                regions: vec![],
                edges: vec![],
                state: RuntimeStatePlan {
                    max_context_tokens: 4096,
                    kv_cache_bytes_per_token: 256,
                    total_kv_cache_bytes: 1048576,
                },
                memory: MemoryPlan {
                    total_activation_bytes: 0,
                    total_weight_bytes: 0,
                    arena_region_count: 0,
                },
            },
            receipt_root: ReceiptId("root".into()),
            created_at: Timestamp("t".into()),
        };
        api.promote(gen).unwrap();
        // Only one generation with no parent — should fail
        assert!(api.rollback().is_err());
    }

    #[test]
    fn test_generation_api_list_empty() {
        let api = GenerationApi::new();
        let list = api.list_generations();
        assert!(list.is_empty());
    }

    #[test]
    fn test_generation_api_current_none() {
        let api = GenerationApi::new();
        assert!(api.current_generation().is_none());
    }
}
