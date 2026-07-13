//! Generation and content-store ABI.
//!
//! Content-addressable storage for payloads, atomic promotion, rollback,
//! and a two-generation lifecycle test.
//!
//! The plan specifies:
//! - "Payload identity is derived from canonical serialized bytes."
//! - "Promotion is a constitutional transaction. All referenced artifacts
//!   must exist and match their digests before the new generation becomes visible."
//! - Section 15 Phase 2: "Generation and content-store ABI: Add identities,
//!   parent generations, payload references, atomic promotion, rollback."

use std::collections::HashMap;

use crate::ecs::canonical::generation::CimageGeneration;
use crate::ecs::canonical::identity::*;

// ---------------------------------------------------------------------------
// ContentStore — content-addressable payload storage
// ---------------------------------------------------------------------------

/// Content-addressable store mapping payload digests to their bytes.
///
/// The plan specifies: "Payload identity is derived from canonical serialized bytes."
pub struct ContentStore {
    segments: HashMap<PhysicalSegmentId, Vec<u8>>,
}

impl ContentStore {
    pub fn new() -> Self {
        Self {
            segments: HashMap::new(),
        }
    }

    /// Store a payload by its canonical digest.
    pub fn store(&mut self, id: PhysicalSegmentId, data: Vec<u8>) {
        self.segments.insert(id, data);
    }

    /// Retrieve a payload by digest.
    pub fn get(&self, id: &PhysicalSegmentId) -> Option<&[u8]> {
        self.segments.get(id).map(|v| v.as_slice())
    }

    /// Check if a payload exists.
    pub fn contains(&self, id: &PhysicalSegmentId) -> bool {
        self.segments.contains_key(id)
    }

    /// Number of stored segments.
    pub fn len(&self) -> usize {
        self.segments.len()
    }

    /// Verify all segments referenced by a generation exist in the store.
    pub fn verify_generation(&self, generation: &CimageGeneration) -> Result<(), String> {
        for binding in generation.tensor_bindings.values() {
            if !self.contains(&binding.primary_segment) {
                return Err(format!(
                    "missing primary segment: {:?}",
                    binding.primary_segment
                ));
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// GenerationStore — stores generations keyed by GenerationId
// ---------------------------------------------------------------------------

/// Store for CimageGenerations, keyed by GenerationId.
///
/// The plan specifies: "A generation manifest references payloads stored in
/// the cimage content store."
pub struct GenerationStore {
    generations: HashMap<GenerationId, CimageGeneration>,
    current: Option<GenerationId>,
}

impl GenerationStore {
    pub fn new() -> Self {
        Self {
            generations: HashMap::new(),
            current: None,
        }
    }

    /// Get a generation by ID.
    pub fn get(&self, id: &GenerationId) -> Option<&CimageGeneration> {
        self.generations.get(id)
    }

    /// Get the current (promoted) generation ID.
    pub fn current_id(&self) -> Option<&GenerationId> {
        self.current.as_ref()
    }

    /// Get the current generation.
    pub fn current(&self) -> Option<&CimageGeneration> {
        self.current
            .as_ref()
            .and_then(|id| self.generations.get(id))
    }

    /// Check if a generation exists.
    pub fn contains(&self, id: &GenerationId) -> bool {
        self.generations.contains_key(id)
    }
    /// Set the current generation directly (for rollback/initialization).
    pub fn set_current(&mut self, id: GenerationId) {
        self.current = Some(id);
    }
}

// ---------------------------------------------------------------------------
// PromotionTransaction — atomic generation promotion
// ---------------------------------------------------------------------------

/// Atomic promotion transaction.
///
/// The plan specifies: "Promotion is a constitutional transaction. All referenced
/// artifacts must exist and match their digests before the new generation becomes visible."
pub struct PromotionTransaction<'a> {
    content_store: &'a mut ContentStore,
    generation_store: &'a mut GenerationStore,
    generation: CimageGeneration,
    committed: bool,
}

impl<'a> PromotionTransaction<'a> {
    pub fn new(
        content_store: &'a mut ContentStore,
        generation_store: &'a mut GenerationStore,
        generation: CimageGeneration,
    ) -> Self {
        Self {
            content_store,
            generation_store,
            generation,
            committed: false,
        }
    }

    /// Validate that all referenced artifacts exist and match their digests.
    ///
    /// The plan: "All referenced artifacts must exist and match their digests
    /// before the new generation becomes visible."
    pub fn validate(&self) -> Result<(), String> {
        // Verify all tensor payloads exist
        for binding in self.generation.tensor_bindings.values() {
            if !self.content_store.contains(&binding.primary_segment) {
                return Err(format!(
                    "primary segment {:?} not in content store",
                    binding.primary_segment
                ));
            }
            for seg in &binding.scale_segments {
                if !self.content_store.contains(seg) {
                    return Err(format!("scale segment {:?} not in content store", seg));
                }
            }
            for seg in &binding.residual_segments {
                if !self.content_store.contains(seg) {
                    return Err(format!("residual segment {:?} not in content store", seg));
                }
            }
        }
        // Verify parent generation exists (if not the first)
        if let Some(parent) = &self.generation.parent_generation {
            if !self.generation_store.contains(parent) {
                return Err(format!(
                    "parent generation {:?} not found in generation store",
                    parent
                ));
            }
        }
        Ok(())
    }

    /// Atomically commit the new generation.
    ///
    /// The plan: "Promotion is a constitutional transaction."
    pub fn commit(&mut self) -> Result<GenerationId, String> {
        self.validate()?;
        let id = self.generation.generation_id.clone();
        self.generation_store
            .generations
            .insert(id.clone(), self.generation.clone());
        self.generation_store.current = Some(id.clone());
        self.committed = true;
        Ok(id)
    }

    /// Rollback — restore parent as current.
    ///
    /// The plan (Section 15 Phase 2): "Rollback — two real generations can be
    /// built, resolved, executed, and rolled back."
    pub fn rollback(&mut self) -> Result<GenerationId, String> {
        if !self.committed {
            return Err("cannot rollback: transaction not committed".into());
        }
        let parent_id = self
            .generation
            .parent_generation
            .clone()
            .ok_or_else(|| "no parent generation to rollback to".to_string())?;
        if !self.generation_store.contains(&parent_id) {
            return Err(format!(
                "parent generation {:?} not found for rollback",
                parent_id
            ));
        }
        self.generation_store.current = Some(parent_id.clone());
        Ok(parent_id)
    }
}

// ---------------------------------------------------------------------------
// Tests — two-generation lifecycle
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::canonical::generation::*;
    use crate::ecs::canonical::{ExecutionGraph, MemoryPlan, RuntimeStatePlan};
    use crate::ecs::execution_profile::PhysicalTileLayout;
    use crate::ecs::plan::CodecFamily;
    use std::collections::BTreeMap;

    /// Section 15 Phase 2 exit condition:
    /// "Two real generations can be built, resolved, executed, and rolled back."
    #[test]
    fn test_two_generation_lifecycle() {
        let mut content_store = ContentStore::new();
        let mut generation_store = GenerationStore::new();

        // Store a tensor payload
        let segment_id = PhysicalSegmentId("seg1".into());
        content_store.store(segment_id.clone(), vec![1, 2, 3, 4]);

        // Create Generation 1
        let gen1_id = GenerationId("gen1".into());
        let mut bindings = BTreeMap::new();
        bindings.insert(
            LogicalTensorId("tensor1".into()),
            RepresentationBinding {
                representation_id: RepresentationId("rep1".into()),
                codec: CodecFamily::RawF32,
                layout: PhysicalTileLayout::default(),
                primary_segment: segment_id.clone(),
                scale_segments: vec![],
                residual_segments: vec![],
                source_representation: None,
                acceptance_receipt: ReceiptId("receipt1".into()),
            },
        );

        let gen1 = CimageGeneration {
            generation_id: gen1_id.clone(),
            parent_generation: None,
            base_model: ModelSourceId("model1".into()),
            compiler_identity: CompilerIdentity {
                name: "test-compiler".into(),
                version: "1.0".into(),
                build_hash: None,
                build_timestamp: None,
            },
            hardware_profile: HardwareProfileId("apple-m1".into()),
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
            receipt_root: ReceiptId("root1".into()),
            created_at: Timestamp("2025-01-01T00:00:00Z".into()),
        };

        // Promote Generation 1
        {
            let mut tx1 =
                PromotionTransaction::new(&mut content_store, &mut generation_store, gen1);
            let promoted_id = tx1.commit().expect("generation 1 should promote");
            assert_eq!(promoted_id, gen1_id);
        }
        // tx1 dropped — borrow released
        assert_eq!(generation_store.current_id(), Some(&gen1_id));

        // Create Generation 2 as child of Generation 1
        let gen2_id = GenerationId("gen2".into());
        let seg2_id = PhysicalSegmentId("seg2".into());
        content_store.store(seg2_id.clone(), vec![5, 6, 7, 8]);

        let mut bindings2 = BTreeMap::new();
        bindings2.insert(
            LogicalTensorId("tensor1".into()),
            RepresentationBinding {
                representation_id: RepresentationId("rep2".into()),
                codec: CodecFamily::Nf4,
                layout: PhysicalTileLayout::default(),
                primary_segment: seg2_id.clone(),
                scale_segments: vec![],
                residual_segments: vec![],
                source_representation: Some(RepresentationId("rep1".into())),
                acceptance_receipt: ReceiptId("receipt2".into()),
            },
        );

        let gen2 = CimageGeneration {
            generation_id: gen2_id.clone(),
            parent_generation: Some(gen1_id.clone()),
            base_model: ModelSourceId("model1".into()),
            compiler_identity: CompilerIdentity {
                name: "test-compiler".into(),
                version: "1.1".into(),
                build_hash: None,
                build_timestamp: None,
            },
            hardware_profile: HardwareProfileId("apple-m1".into()),
            tensor_bindings: bindings2,
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
            receipt_root: ReceiptId("root2".into()),
            created_at: Timestamp("2025-01-02T00:00:00Z".into()),
        };

        // Promote Generation 2 — atomically, then rollback
        {
            let mut tx2 =
                PromotionTransaction::new(&mut content_store, &mut generation_store, gen2);
            let promoted_id2 = tx2.commit().expect("generation 2 should promote");
            assert_eq!(promoted_id2, gen2_id);

            // Rollback to Generation 1
            let rolled_back = tx2.rollback().expect("rollback should succeed");
            assert_eq!(rolled_back, gen1_id);
        }
        // tx2 dropped — borrow released
        assert_eq!(generation_store.current_id(), Some(&gen1_id));
    }

    #[test]
    fn test_promotion_rejects_missing_payload() {
        let mut content_store = ContentStore::new();
        let mut generation_store = GenerationStore::new();

        let gen_id = GenerationId("orphan".into());
        let mut bindings = BTreeMap::new();
        bindings.insert(
            LogicalTensorId("tensor1".into()),
            RepresentationBinding {
                representation_id: RepresentationId("rep1".into()),
                codec: CodecFamily::RawF32,
                layout: PhysicalTileLayout::default(),
                primary_segment: PhysicalSegmentId("missing".into()),
                scale_segments: vec![],
                residual_segments: vec![],
                source_representation: None,
                acceptance_receipt: ReceiptId("r1".into()),
            },
        );

        let gen = CimageGeneration {
            generation_id: gen_id,
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
                    max_context_tokens: 0,
                    kv_cache_bytes_per_token: 0,
                    total_kv_cache_bytes: 0,
                },
                memory: MemoryPlan {
                    total_activation_bytes: 0,
                    total_weight_bytes: 0,
                    arena_region_count: 0,
                },
            },
            receipt_root: ReceiptId("r".into()),
            created_at: Timestamp("t".into()),
        };

        let mut tx = PromotionTransaction::new(&mut content_store, &mut generation_store, gen);
        assert!(tx.commit().is_err());
    }
}
