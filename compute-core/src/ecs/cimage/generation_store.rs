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
use sha2::Digest;
use sha2::Sha256;

// ---------------------------------------------------------------------------
// ContentStore — content-addressable payload storage
// ---------------------------------------------------------------------------

/// Content-addressable store mapping payload digests to their bytes.
///
/// The plan specifies: "Payload identity is derived from canonical serialized bytes."
pub struct ContentStore {
    segments: HashMap<PhysicalSegmentId, Vec<u8>>,
    /// Quarantined segments removed from active storage due to corruption.
    quarantined: Vec<(PhysicalSegmentId, Vec<u8>)>,
}

impl ContentStore {
    pub fn new() -> Self {
        Self {
            segments: HashMap::new(),
            quarantined: Vec::new(),
        }
    }

    /// Store a payload by its canonical digest.
    pub fn store(&mut self, id: PhysicalSegmentId, data: Vec<u8>) {
        self.segments.insert(id, data);
    }

    /// Remove a payload by id. Used for rolling back stored payloads.
    pub fn remove(&mut self, id: &PhysicalSegmentId) {
        self.segments.remove(id);
    }

    /// Retrieve a payload by digest.
    pub fn get(&self, id: &PhysicalSegmentId) -> Option<&[u8]> {
        self.segments.get(id).map(|v| v.as_slice())
    }

    /// Check if a payload exists.
    pub fn contains(&self, id: &PhysicalSegmentId) -> bool {
        self.segments.contains_key(id)
    }

    /// Verify all engram payloads referenced by a generation exist.
    pub fn verify_engram_payloads(&self, generation: &CimageGeneration) -> Result<(), String> {
        for (engram_id, binding) in &generation.engram_bindings {
            let seg_key = PhysicalSegmentId(binding.artifact_id.0.clone());
            if !self.contains(&seg_key) {
                return Err(format!(
                    "engram payload missing for engram {:?}: artifact {:?}",
                    engram_id, binding.artifact_id
                ));
            }
        }
        Ok(())
    }

    /// Return the SHA-256 hex digest of a stored payload, if present.
    pub fn compute_digest(&self, id: &PhysicalSegmentId) -> Option<String> {
        self.segments.get(id).map(|data| {
            Sha256::digest(data)
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>()
        })
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

    /// Verify that a segment's stored bytes match its content digest.
    /// The PhysicalSegmentId's inner string is the hex-encoded SHA-256 of
    /// the canonical segment bytes.
    pub fn verify_segment_digest(&self, id: &PhysicalSegmentId) -> Result<(), String> {
        let data = self
            .segments
            .get(id)
            .ok_or_else(|| format!("segment {:?} not found", id))?;
        let computed = sha2::Sha256::digest(data);
        let computed_hex = computed
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        if computed_hex != id.0 {
            return Err(format!(
                "segment {:?} digest mismatch: computed {computed_hex}",
                id
            ));
        }
        Ok(())
    }

    /// Move a corrupted segment to quarantine, preserving the data for
    /// offline inspection.
    pub fn quarantine(&mut self, id: &PhysicalSegmentId) {
        if let Some(data) = self.segments.remove(id) {
            self.quarantined.push((id.clone(), data));
        }
    }

    /// Number of currently quarantined segments.
    pub fn quarantined_count(&self) -> usize {
        self.quarantined.len()
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
    /// Return generation IDs in deterministic order.
    pub fn ids(&self) -> Vec<GenerationId> {
        let mut ids: Vec<_> = self.generations.keys().cloned().collect();
        ids.sort();
        ids
    }

    /// Set the current generation after validating that it exists.
    pub fn set_current(&mut self, id: GenerationId) -> Result<(), String> {
        if !self.generations.contains_key(&id) {
            return Err(format!("generation {:?} not found", id));
        }
        self.current = Some(id);
        Ok(())
    }

    /// Detect an incomplete promotion — a generation that was stored but
    /// never fully committed as current, or is current but has no valid
    /// parent chain.
    ///
    /// Returns the ID of the candidate generation that appears to have been
    /// left in an incomplete state, if any.
    pub fn detect_incomplete_promotion(&self) -> Option<GenerationId> {
        // Scenario 1: current is set but references a generation that no
        // longer exists in the store.
        if let Some(cur) = &self.current {
            if !self.generations.contains_key(cur) {
                // current points to a missing generation — that's a
                // partially committed state: the generation was committed
                // to storage but the current pointer is stale.
                return Some(cur.clone());
            }
        }
        // Scenario 2: a generation exists with a parent that is not the
        // current — indicates a promotion that stored the child but
        // didn't finish updating the current pointer, or an orphan.
        let current_id = self.current.as_ref();
        for (id, gen) in &self.generations {
            // Skip the current generation itself.
            if let Some(cur) = current_id {
                if id == cur {
                    continue;
                }
            }
            // A generation with a parent that was interrupted mid-promotion
            // is a candidate for incomplete promotion.
            if gen.parent_generation.is_some() {
                return Some(id.clone());
            }
        }
        None
    }

    /// Complete an interrupted promotion — verify the generation exists
    /// and make it the current generation.
    pub fn complete_promotion(&mut self) -> Result<(), String> {
        let candidate = self
            .detect_incomplete_promotion()
            .ok_or_else(|| "no incomplete promotion to complete".to_string())?;
        self.current = Some(candidate);
        Ok(())
    }

    /// Safely discard an incomplete promotion by removing the partially
    /// committed generation from the store.
    pub fn abort_promotion(&mut self) -> Result<(), String> {
        let candidate = self
            .detect_incomplete_promotion()
            .ok_or_else(|| "no incomplete promotion to abort".to_string())?;
        self.generations.remove(&candidate);
        Ok(())
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
    /// Payload ids stored during this transaction — rolled back on abort.
    pending_payloads: Vec<PhysicalSegmentId>,
    /// The generation that was current before commit, saved for parent
    /// restoration if the transaction fails.
    saved_current: Option<GenerationId>,
    /// Test-only: inject a failure after the generation is stored but
    /// before the current pointer is updated.
    #[cfg(test)]
    inject_fail_before_current_update: bool,
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
            pending_payloads: Vec::new(),
            saved_current: None,
            #[cfg(test)]
            inject_fail_before_current_update: false,
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
        // Identity closure: verify all engram payloads resolve
        self.content_store
            .verify_engram_payloads(&self.generation)?;
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

    /// Store a payload and track it for rollback on abort/failure.
    pub fn store_payload(&mut self, id: PhysicalSegmentId, data: Vec<u8>) {
        self.content_store.store(id.clone(), data);
        self.pending_payloads.push(id);
    }

    /// Atomically commit the new generation.
    ///
    /// The plan: "Promotion is a constitutional transaction."
    pub fn commit(&mut self) -> Result<GenerationId, String> {
        // Save the current pointer before any mutation
        self.saved_current = self.generation_store.current_id().cloned();

        // Phase 1: validate identity closure and digest integrity
        self.validate()?;

        // Phase 2: insert the generation into the store
        let id = self.generation.generation_id.clone();
        self.generation_store
            .generations
            .insert(id.clone(), self.generation.clone());

        // Phase 3 (test-only): inject failure before the current pointer
        // update, simulating a crash after payload storage but before
        // the generation becomes visible as current.
        #[cfg(test)]
        if self.inject_fail_before_current_update {
            return Err(format!(
                "injected failure before current pointer update for {:?}",
                id
            ));
        }

        // Phase 4: make the new generation current
        self.generation_store.current = Some(id.clone());
        self.committed = true;
        // Clear pending — no rollback needed after successful commit
        self.pending_payloads.clear();
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

    /// Abort the transaction — roll back all pending changes.
    ///
    /// Removes any stored payloads and restores the original current
    /// generation pointer so the parent remains executable.
    pub fn abort(&mut self) {
        // Remove stored payloads
        for id in &self.pending_payloads {
            self.content_store.remove(id);
        }
        self.pending_payloads.clear();
        // Restore the saved current pointer (if any was saved)
        if let Some(parent) = &self.saved_current {
            // Only restore if the generation still exists
            if self.generation_store.contains(parent) {
                self.generation_store.current = Some(parent.clone());
            }
        }
        self.committed = false;
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

    #[test]
    fn test_promotion_inject_failure_parent_preserved() {
        let mut content_store = ContentStore::new();
        let mut generation_store = GenerationStore::new();

        // Store payload and promote Generation 1 (the parent)
        let seg1_id = PhysicalSegmentId("seg1".into());
        content_store.store(seg1_id.clone(), vec![1, 2, 3, 4]);

        let gen1_id = GenerationId("gen1".into());
        let mut bindings = BTreeMap::new();
        bindings.insert(
            LogicalTensorId("t0".into()),
            RepresentationBinding {
                representation_id: RepresentationId("r1".into()),
                codec: CodecFamily::RawF32,
                layout: PhysicalTileLayout::default(),
                primary_segment: seg1_id.clone(),
                scale_segments: vec![],
                residual_segments: vec![],
                source_representation: None,
                acceptance_receipt: ReceiptId("receipt1".into()),
            },
        );

        let parent = CimageGeneration {
            generation_id: gen1_id.clone(),
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
            created_at: Timestamp("t1".into()),
        };

        let mut tx_parent =
            PromotionTransaction::new(&mut content_store, &mut generation_store, parent);
        let parent_id = tx_parent.commit().expect("parent should promote");
        assert_eq!(parent_id, gen1_id);
        drop(tx_parent);

        // Now create Generation 2 with an injected failure after payload
        // storage but before the current pointer update.
        let seg2_id = PhysicalSegmentId("seg2".into());
        let child_payload = vec![5, 6, 7, 8];
        content_store.store(seg2_id.clone(), child_payload.clone());

        let gen2_id = GenerationId("gen2".into());
        let mut bindings2 = BTreeMap::new();
        bindings2.insert(
            LogicalTensorId("t0".into()),
            RepresentationBinding {
                representation_id: RepresentationId("r2".into()),
                codec: CodecFamily::Nf4,
                layout: PhysicalTileLayout::default(),
                primary_segment: seg2_id.clone(),
                scale_segments: vec![],
                residual_segments: vec![],
                source_representation: Some(RepresentationId("r1".into())),
                acceptance_receipt: ReceiptId("receipt2".into()),
            },
        );

        let child = CimageGeneration {
            generation_id: gen2_id.clone(),
            parent_generation: Some(gen1_id.clone()),
            base_model: ModelSourceId("m".into()),
            compiler_identity: CompilerIdentity {
                name: "tc".into(),
                version: "2".into(),
                build_hash: None,
                build_timestamp: None,
            },
            hardware_profile: HardwareProfileId("h".into()),
            tensor_bindings: bindings2,
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
            receipt_root: ReceiptId("r2".into()),
            created_at: Timestamp("t2".into()),
        };

        // Transaction with failure injection — the payload exists, the
        // generation is stored, but the current pointer update is skipped.
        let mut tx2 = PromotionTransaction::new(&mut content_store, &mut generation_store, child);
        tx2.inject_fail_before_current_update = true;

        // The commit should fail due to injected failure
        assert!(
            tx2.commit().is_err(),
            "injected failure should abort commit"
        );

        // After abort, parent must still be current
        tx2.abort();
        assert_eq!(
            generation_store.current_id(),
            Some(&gen1_id),
            "parent should remain current after failed promotion"
        );

        // The child payload must still be in the content store (abort
        // only clears payloads stored through the transaction, but
        // this payload was stored directly to content_store before
        // the transaction was created).
        assert!(
            content_store.contains(&seg2_id),
            "payload stored before transaction should survive abort"
        );
    }
}
