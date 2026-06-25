// ── Prism LLM Inference — KV Cache Manager ─────────────────────────────
//
// Manages KV-cache epochs and pages. Epochs follow a strict lifecycle:
// Building → Active → Superseded → Draining → Reclaimable.
// Sparse retention creates a new epoch from selected pages of a source epoch,
// marking the source as Superseded. Context refresh creates a fresh epoch.

use std::collections::HashMap;

use parking_lot::Mutex;

use crate::image::types::ArtifactDigest;

use super::super::server::{
    ContextRefreshPlan, KvDispatchView, KvEpoch, KvEpochId, KvEpochReceipt, KvEpochState, KvPage,
    KvPageId, KvPageState, RopePositionContract, SparseRetentionPlan,
};

// ── Internal epoch tracking ──────────────────────────────────────────

/// Internal bookkeeping for a single KV-cache epoch.
struct EpochRecord {
    /// The epoch descriptor.
    epoch: KvEpoch,
    /// Pages belonging to this epoch, keyed by page id.
    pages: HashMap<KvPageId, KvPage>,
    /// Monotonically increasing counter for allocating page ids within this epoch.
    next_page_index: u64,
}

// ── KvManager ────────────────────────────────────────────────────────

/// Manages KV-cache epochs and pages for LLM inference.
///
/// Epoch lifecycle:
///   `Building → Active → Superseded → Draining → Reclaimable`
///
/// - Pages can only be added while the epoch is `Building`.
/// - Sealing transitions an epoch to `Active` (ready for dispatch).
/// - Sparse retention creates a child epoch from selected pages and marks the
///   source as `Superseded`.
/// - Draining transitions an `Active` or `Superseded` epoch to `Reclaimable`.
pub struct KvManager {
    inner: Mutex<KvManagerInner>,
}

#[allow(dead_code)]
struct KvManagerInner {
    /// Number of tokens per page.
    page_token_capacity: u32,
    /// Maximum number of context tokens the model supports.
    max_context: u32,
    /// All epochs, keyed by epoch id.
    epochs: HashMap<KvEpochId, EpochRecord>,
    /// Monotonically increasing counter for allocating epoch ids.
    next_epoch_index: u64,
}

impl KvManager {
    /// Creates a new `KvManager`.
    pub fn new(page_token_capacity: u32, max_context: u32) -> Self {
        Self {
            inner: Mutex::new(KvManagerInner {
                page_token_capacity,
                max_context,
                epochs: HashMap::new(),
                next_epoch_index: 0,
            }),
        }
    }

    /// Creates a new epoch in the `Building` state.
    ///
    /// `parent` optionally links this epoch to a prior epoch
    /// (e.g. the source of a sparse-retention or context-refresh pipeline).
    pub fn create_epoch(&self, parent: Option<KvEpochId>) -> Result<KvEpochId, String> {
        let mut inner = self.inner.lock();
        let epoch_id = KvEpochId(inner.next_epoch_index);
        inner.next_epoch_index += 1;

        let epoch = KvEpoch {
            epoch_id,
            parent_epoch: parent,
            generation_token_index: 0,
            logical_context_length: 0,
            retained_token_count: 0,
            state: KvEpochState::Building,
        };

        inner.epochs.insert(
            epoch_id,
            EpochRecord {
                epoch,
                pages: HashMap::new(),
                next_page_index: 0,
            },
        );

        Ok(epoch_id)
    }

    /// Adds a page to a `Building` epoch.
    ///
    /// Returns the assigned `KvPageId`. The page's `page_id` field is
    /// overwritten with the manager-assigned id.
    pub fn add_page(&self, epoch_id: &KvEpochId, page: KvPage) -> Result<KvPageId, String> {
        let mut inner = self.inner.lock();
        let record = inner
            .epochs
            .get_mut(epoch_id)
            .ok_or_else(|| format!("epoch {:?} not found", epoch_id))?;

        if record.epoch.state != KvEpochState::Building {
            return Err(format!(
                "epoch {:?} is in state {:?}, can only add pages to Building epochs",
                epoch_id, record.epoch.state
            ));
        }

        let page_id = KvPageId(record.next_page_index);
        record.next_page_index += 1;

        let stored_page = KvPage { page_id, ..page };

        record.pages.insert(page_id, stored_page);

        Ok(page_id)
    }

    /// Seals a `Building` epoch, transitioning it to `Active`.
    ///
    /// Computes the logical context length from page token ranges and marks
    /// all pages as `Sealed`. Returns a `KvEpochReceipt`.
    pub fn seal_epoch(&self, epoch_id: &KvEpochId) -> Result<KvEpochReceipt, String> {
        let mut inner = self.inner.lock();
        let record = inner
            .epochs
            .get_mut(epoch_id)
            .ok_or_else(|| format!("epoch {:?} not found", epoch_id))?;

        if record.epoch.state != KvEpochState::Building {
            return Err(format!(
                "epoch {:?} is in state {:?}, expected Building",
                epoch_id, record.epoch.state
            ));
        }

        // Compute logical context length from page token ranges.
        let logical_context: u32 = record
            .pages
            .values()
            .map(|p| p.token_range.1 - p.token_range.0)
            .sum();

        record.epoch.state = KvEpochState::Active;
        record.epoch.logical_context_length = logical_context;
        record.epoch.retained_token_count = logical_context;

        // Transition all pages to Sealed.
        for page in record.pages.values_mut() {
            page.state = KvPageState::Sealed;
        }

        Ok(KvEpochReceipt {
            epoch_id: record.epoch.epoch_id,
            parent_epoch: record.epoch.parent_epoch,
            logical_context_length: record.epoch.logical_context_length,
            state: record.epoch.state,
        })
    }

    /// Creates a dispatch view into an `Active` epoch at the given decode position.
    ///
    /// The returned `KvDispatchView` carries an absolute rope-position contract
    /// starting at `position`.
    pub fn create_dispatch_view(
        &self,
        epoch_id: &KvEpochId,
        position: u32,
    ) -> Result<KvDispatchView, String> {
        let inner = self.inner.lock();
        let record = inner
            .epochs
            .get(epoch_id)
            .ok_or_else(|| format!("epoch {:?} not found", epoch_id))?;

        if record.epoch.state != KvEpochState::Active {
            return Err(format!(
                "epoch {:?} is in state {:?}, must be Active for dispatch",
                epoch_id, record.epoch.state
            ));
        }

        Ok(KvDispatchView {
            epoch_id: *epoch_id,
            absolute_decode_position: position,
            rope_position_contract: RopePositionContract::Absolute { start: position },
        })
    }

    /// Builds a `SparseRetentionPlan` that retains selected pages from a
    /// source `Active` epoch.
    ///
    /// The plan reserves a target epoch id. Pages not in `retained` are
    /// listed as `removed_pages`. The plan sets
    /// `preserves_absolute_positions` to `true`.
    pub fn build_sparse_retention_plan(
        &self,
        source: &KvEpochId,
        retained: Vec<KvPageId>,
    ) -> Result<SparseRetentionPlan, String> {
        let mut inner = self.inner.lock();
        let source_record = inner
            .epochs
            .get(source)
            .ok_or_else(|| format!("source epoch {:?} not found", source))?;

        if source_record.epoch.state != KvEpochState::Active {
            return Err(format!(
                "source epoch {:?} is in state {:?}, must be Active",
                source, source_record.epoch.state
            ));
        }

        // Compute removed pages (present in source but not in retained).
        let retained_set: std::collections::HashSet<KvPageId> = retained.iter().copied().collect();

        let removed_pages: Vec<KvPageId> = source_record
            .pages
            .keys()
            .copied()
            .filter(|pid| !retained_set.contains(pid))
            .collect();

        // Reserve a target epoch id.
        let target_epoch_id = KvEpochId(inner.next_epoch_index);
        inner.next_epoch_index += 1;

        Ok(SparseRetentionPlan {
            source_epoch: *source,
            retained_pages: retained,
            removed_pages,
            preserves_absolute_positions: true,
            target_epoch: target_epoch_id,
        })
    }

    /// Executes a `SparseRetentionPlan`.
    ///
    /// Marks the source epoch as `Superseded`, transitions retained pages to
    /// `RetainedSparse` and removed pages to `PendingReclaim`, then creates
    /// a new `Active` epoch containing copies of the retained pages. Returns
    /// a `KvEpochReceipt` for the new epoch.
    pub fn execute_sparse_retention(
        &self,
        plan: &SparseRetentionPlan,
    ) -> Result<KvEpochReceipt, String> {
        let mut inner = self.inner.lock();

        // Validate and mark source epoch as Superseded.
        let source_record = inner
            .epochs
            .get_mut(&plan.source_epoch)
            .ok_or_else(|| format!("source epoch {:?} not found", plan.source_epoch))?;

        if source_record.epoch.state != KvEpochState::Active {
            return Err(format!(
                "source epoch {:?} is in state {:?}, must be Active for sparse retention",
                plan.source_epoch, source_record.epoch.state
            ));
        }

        source_record.epoch.state = KvEpochState::Superseded;

        // Compute total retained token count and update page states.
        let retained_count: u32 = plan
            .retained_pages
            .iter()
            .filter_map(|pid| {
                source_record
                    .pages
                    .get(pid)
                    .map(|p| p.token_range.1 - p.token_range.0)
            })
            .sum();

        for page in source_record.pages.values_mut() {
            if plan.retained_pages.contains(&page.page_id) {
                page.state = KvPageState::RetainedSparse;
            } else {
                page.state = KvPageState::PendingReclaim;
            }
        }

        // Create the new epoch from retained pages.
        let new_epoch = KvEpoch {
            epoch_id: plan.target_epoch,
            parent_epoch: Some(plan.source_epoch),
            generation_token_index: source_record.epoch.generation_token_index,
            logical_context_length: retained_count,
            retained_token_count: retained_count,
            state: KvEpochState::Active,
        };

        let mut new_pages = HashMap::new();
        let mut next_page_index = 0u64;
        for pid in &plan.retained_pages {
            if let Some(src_page) = source_record.pages.get(pid) {
                let retained_page = KvPage {
                    page_id: KvPageId(next_page_index),
                    ..src_page.clone()
                };
                new_pages.insert(retained_page.page_id, retained_page);
                next_page_index += 1;
            }
        }

        let new_record = EpochRecord {
            epoch: new_epoch,
            pages: new_pages,
            next_page_index,
        };

        inner.epochs.insert(plan.target_epoch, new_record);

        Ok(KvEpochReceipt {
            epoch_id: plan.target_epoch,
            parent_epoch: Some(plan.source_epoch),
            logical_context_length: retained_count,
            state: KvEpochState::Active,
        })
    }

    /// Builds a `ContextRefreshPlan` from a source `Active` epoch.
    ///
    /// Retains the token ranges of all existing pages and reserves a target
    /// epoch id. The `new_prompt_digest` is a placeholder; the caller fills
    /// it with the actual prompt digest before execution.
    pub fn build_context_refresh_plan(
        &self,
        source: &KvEpochId,
    ) -> Result<ContextRefreshPlan, String> {
        let mut inner = self.inner.lock();
        let record = inner
            .epochs
            .get(source)
            .ok_or_else(|| format!("source epoch {:?} not found", source))?;

        if record.epoch.state != KvEpochState::Active {
            return Err(format!(
                "source epoch {:?} is in state {:?}, must be Active",
                source, record.epoch.state
            ));
        }

        // Derive retained source ranges from existing page token ranges.
        let retained_ranges: Vec<(u32, u32)> =
            record.pages.values().map(|p| p.token_range).collect();

        // Reserve a target epoch id.
        let target_epoch_id = KvEpochId(inner.next_epoch_index);
        inner.next_epoch_index += 1;

        Ok(ContextRefreshPlan {
            source_epoch: *source,
            retained_source_ranges: retained_ranges,
            new_prompt_digest: ArtifactDigest(String::new()),
            target_epoch: target_epoch_id,
        })
    }

    /// Drains an epoch, transitioning it from `Active` or `Superseded` to
    /// `Reclaimable`.
    ///
    /// All pages are marked `PendingReclaim` as part of the drain.
    pub fn drain_epoch(&self, epoch_id: &KvEpochId) -> Result<(), String> {
        let mut inner = self.inner.lock();
        let record = inner
            .epochs
            .get_mut(epoch_id)
            .ok_or_else(|| format!("epoch {:?} not found", epoch_id))?;

        match record.epoch.state {
            KvEpochState::Active | KvEpochState::Superseded => {
                record.epoch.state = KvEpochState::Draining;
            }
            _ => {
                return Err(format!(
                    "epoch {:?} is in state {:?}, can only drain Active or Superseded epochs",
                    epoch_id, record.epoch.state
                ));
            }
        }

        // Mark all pages as pending reclaim.
        for page in record.pages.values_mut() {
            page.state = KvPageState::PendingReclaim;
        }

        // Complete the drain — transition to Reclaimable.
        record.epoch.state = KvEpochState::Reclaimable;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_page(page_id: KvPageId, token_start: u32, token_end: u32) -> KvPage {
        use crate::llm::manifest::IslandAllocationId;
        KvPage {
            page_id,
            layer_range: (0, 1),
            token_range: (token_start, token_end),
            original_position_range: (token_start, token_end),
            allocation_id: IslandAllocationId(0),
            residency: String::new(),
            state: KvPageState::Allocated,
        }
    }

    #[test]
    fn test_create_epoch() {
        let mgr = KvManager::new(64, 4096);
        let id = mgr.create_epoch(None).unwrap();
        let inner = mgr.inner.lock();
        let record = inner.epochs.get(&id).unwrap();
        assert_eq!(record.epoch.state, KvEpochState::Building);
        assert!(record.epoch.parent_epoch.is_none());
    }

    #[test]
    fn test_create_epoch_with_parent() {
        let mgr = KvManager::new(64, 4096);
        let parent = mgr.create_epoch(None).unwrap();
        let child = mgr.create_epoch(Some(parent)).unwrap();
        let inner = mgr.inner.lock();
        let record = inner.epochs.get(&child).unwrap();
        assert_eq!(record.epoch.parent_epoch, Some(parent));
    }

    #[test]
    fn test_add_page_and_seal() {
        let mgr = KvManager::new(64, 4096);
        let epoch_id = mgr.create_epoch(None).unwrap();
        let page = make_page(KvPageId(999), 0, 64);
        let page_id = mgr.add_page(&epoch_id, page).unwrap();

        // Verify page id was assigned (not the one from input).
        assert_ne!(page_id, KvPageId(999));

        let receipt = mgr.seal_epoch(&epoch_id).unwrap();
        assert_eq!(receipt.epoch_id, epoch_id);
        assert_eq!(receipt.state, KvEpochState::Active);
        assert_eq!(receipt.logical_context_length, 64);

        let inner = mgr.inner.lock();
        let record = inner.epochs.get(&epoch_id).unwrap();
        assert_eq!(record.epoch.state, KvEpochState::Active);
        assert_eq!(record.pages.len(), 1);
        let stored = record.pages.get(&page_id).unwrap();
        assert_eq!(stored.state, KvPageState::Sealed);
    }

    #[test]
    fn test_add_page_to_non_building_epoch_fails() {
        let mgr = KvManager::new(64, 4096);
        let epoch_id = mgr.create_epoch(None).unwrap();
        mgr.seal_epoch(&epoch_id).unwrap();
        let page = make_page(KvPageId(0), 0, 64);
        assert!(mgr.add_page(&epoch_id, page).is_err());
    }

    #[test]
    fn test_seal_non_building_epoch_fails() {
        let mgr = KvManager::new(64, 4096);
        let epoch_id = mgr.create_epoch(None).unwrap();
        mgr.seal_epoch(&epoch_id).unwrap();
        assert!(mgr.seal_epoch(&epoch_id).is_err());
    }

    #[test]
    fn test_create_dispatch_view() {
        let mgr = KvManager::new(64, 4096);
        let epoch_id = mgr.create_epoch(None).unwrap();
        mgr.seal_epoch(&epoch_id).unwrap();
        let view = mgr.create_dispatch_view(&epoch_id, 42).unwrap();
        assert_eq!(view.epoch_id, epoch_id);
        assert_eq!(view.absolute_decode_position, 42);
    }

    #[test]
    fn test_dispatch_view_on_non_active_fails() {
        let mgr = KvManager::new(64, 4096);
        let epoch_id = mgr.create_epoch(None).unwrap();
        assert!(mgr.create_dispatch_view(&epoch_id, 0).is_err());
    }

    #[test]
    fn test_sparse_retention_roundtrip() {
        let mgr = KvManager::new(64, 4096);
        let epoch_id = mgr.create_epoch(None).unwrap();

        let p0 = mgr
            .add_page(&epoch_id, make_page(KvPageId(0), 0, 64))
            .unwrap();
        let p1 = mgr
            .add_page(&epoch_id, make_page(KvPageId(1), 64, 128))
            .unwrap();
        let _p2 = mgr
            .add_page(&epoch_id, make_page(KvPageId(2), 128, 192))
            .unwrap();
        mgr.seal_epoch(&epoch_id).unwrap();

        // Retain only the first two pages.
        let plan = mgr
            .build_sparse_retention_plan(&epoch_id, vec![p0, p1])
            .unwrap();
        assert_eq!(plan.source_epoch, epoch_id);
        assert_eq!(plan.retained_pages, vec![p0, p1]);
        assert_eq!(plan.removed_pages.len(), 1);

        let receipt = mgr.execute_sparse_retention(&plan).unwrap();
        assert_eq!(receipt.parent_epoch, Some(epoch_id));
        assert_eq!(receipt.state, KvEpochState::Active);
        assert_eq!(receipt.logical_context_length, 128);

        // Source should now be Superseded.
        let inner = mgr.inner.lock();
        let src = inner.epochs.get(&epoch_id).unwrap();
        assert_eq!(src.epoch.state, KvEpochState::Superseded);

        // Retained pages in source should be RetainedSparse.
        for pid in &[p0, p1] {
            assert_eq!(
                src.pages.get(pid).unwrap().state,
                KvPageState::RetainedSparse
            );
        }

        // New epoch exists and is Active.
        let new_rec = inner.epochs.get(&receipt.epoch_id).unwrap();
        assert_eq!(new_rec.epoch.state, KvEpochState::Active);
        assert_eq!(new_rec.pages.len(), 2);
    }

    #[test]
    fn test_build_context_refresh_plan() {
        let mgr = KvManager::new(64, 4096);
        let epoch_id = mgr.create_epoch(None).unwrap();
        mgr.add_page(&epoch_id, make_page(KvPageId(0), 0, 64))
            .unwrap();
        mgr.seal_epoch(&epoch_id).unwrap();

        let plan = mgr.build_context_refresh_plan(&epoch_id).unwrap();
        assert_eq!(plan.source_epoch, epoch_id);
        // Should have retained the one page's token range.
        assert_eq!(plan.retained_source_ranges, vec![(0, 64)]);
    }

    #[test]
    fn test_drain_epoch() {
        let mgr = KvManager::new(64, 4096);
        let epoch_id = mgr.create_epoch(None).unwrap();
        mgr.add_page(&epoch_id, make_page(KvPageId(0), 0, 64))
            .unwrap();
        mgr.seal_epoch(&epoch_id).unwrap();

        mgr.drain_epoch(&epoch_id).unwrap();

        let inner = mgr.inner.lock();
        let record = inner.epochs.get(&epoch_id).unwrap();
        assert_eq!(record.epoch.state, KvEpochState::Reclaimable);

        // All pages should be PendingReclaim.
        for page in record.pages.values() {
            assert_eq!(page.state, KvPageState::PendingReclaim);
        }
    }

    #[test]
    fn test_drain_building_epoch_fails() {
        let mgr = KvManager::new(64, 4096);
        let epoch_id = mgr.create_epoch(None).unwrap();
        assert!(mgr.drain_epoch(&epoch_id).is_err());
    }

    #[test]
    fn test_unknown_epoch_fails() {
        let mgr = KvManager::new(64, 4096);
        let unknown = KvEpochId(999);
        assert!(mgr
            .add_page(&unknown, make_page(KvPageId(0), 0, 64))
            .is_err());
        assert!(mgr.seal_epoch(&unknown).is_err());
        assert!(mgr.create_dispatch_view(&unknown, 0).is_err());
        assert!(mgr.build_sparse_retention_plan(&unknown, vec![]).is_err());
        assert!(mgr.build_context_refresh_plan(&unknown).is_err());
        assert!(mgr.drain_epoch(&unknown).is_err());
    }
}
// ── compute-core KV arena integration ────────────────────────────────
// Maps Prism KvEpochId → kv_arena::SequenceId and KvPageId →
// PhysicalBlockId. The Prism epoch state machine (Building→Active→
// Superseded→Draining→Reclaimable) is maintained in Prism; kv_arena
// handles physical page allocation with COW refcounting and prefix caching.
//
// Full per-session KV initialization with MLX arrays from
// ProfiledInferenceSession is deferred to PRISM-KV-ARENA-INTEGRATION-0001.
// This block establishes the type mappings and integration points.
#[cfg(feature = "prism-backend")]
mod prism_backend {
    use super::*;
    use tribunus_compute_core::kv_arena::SequenceId;
    use tribunus_compute_core::kv_arena::block::PhysicalBlockId;

    /// Maps the KvEpochId to a compute-core SequenceId for physical page allocation.
    #[allow(dead_code)]
    pub fn epoch_to_sequence(epoch_id: &KvEpochId) -> SequenceId {
        SequenceId(epoch_id.0)
    }

    /// Maps a KvPageId to a PhysicalBlockId reference.
    #[allow(dead_code)]
    pub fn page_to_block(page_id: &KvPageId) -> PhysicalBlockId {
        PhysicalBlockId(page_id.0 as u32)
    }

    impl KvManager {
        /// Attempt to allocate KV pages using compute-core's kv_arena.
        /// Returns `Unsupported` until a live inference session provides the
        /// MLX arrays needed for per-layer KV cache initialization.
        pub fn try_arena_admit(&self, _epoch: &KvEpochId, _token_count: u32)
            -> Result<tribunus_compute_core::kv_arena::AdmissionReceipt, String>
        {
            Err("kv_arena integration requires active inference session with MLX arrays".into())
        }

        /// Number of active epochs registered in the arena.
        pub fn arena_epoch_count(&self) -> usize {
            // Stub: returns 0 until KvBlockArena is initialized with a
            // live inference session handle.
            0
        }
    }

    #[cfg(test)]
    mod arena_tests {
        use super::*;

        #[test]
        fn epoch_to_sequence_roundtrip() {
            let eid = KvEpochId(42);
            let seq = epoch_to_sequence(&eid);
            assert_eq!(seq.0, 42);
        }

        #[test]
        fn page_to_block_id_roundtrip() {
            let pid = KvPageId(99);
            let block = page_to_block(&pid);
            assert_eq!(block.0, 99);
        }

        #[test]
        fn try_arena_admit_returns_error_without_session() {
            let mgr = KvManager::new(128, 4096);
            let epoch = mgr.create_epoch(None).unwrap();
            let result = mgr.try_arena_admit(&epoch, 64);
            assert!(result.is_err());
            assert!(result.unwrap_err().contains("requires active inference"));
        }

        #[test]
        fn arena_epoch_count_starts_zero() {
            let mgr = KvManager::new(128, 4096);
            assert_eq!(mgr.arena_epoch_count(), 0);
        }
    }
}
