use prism_ecs_core::Entity;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

const SENTINEL: u32 = u32::MAX;

/// A sparse set data structure for component storage.
///
/// - `dense`: contiguous component values (cache-friendly iteration)
/// - `entities`: parallel entity IDs (same index as dense)
/// - `sparse`: entity ID -> dense index (or SENTINEL for absent)
///
/// Insertion and removal are O(1). Iteration is contiguous over dense.
/// No slot is allocated for entities that lack the component.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseSet<T> {
    dense: Vec<T>,
    entities: Vec<u64>,
    sparse: Vec<u32>,
    #[serde(skip)]
    _marker: PhantomData<T>,
}

impl<T: PartialEq> PartialEq for SparseSet<T> {
    fn eq(&self, other: &Self) -> bool {
        self.dense == other.dense && self.entities == other.entities
    }
}

impl<T> SparseSet<T> {
    pub fn new() -> Self {
        Self {
            dense: Vec::new(),
            entities: Vec::new(),
            sparse: Vec::new(),
            _marker: PhantomData,
        }
    }

    /// Insert a component for an entity. Replaces existing value if present.
    /// When the sparse slot points to a dense entry whose entity handle differs
    /// (generation changed via slot reuse), both the value and handle are replaced.
    pub fn insert(&mut self, entity: u64, value: T) {
        let idx = self.sparse_idx(entity);
        if idx < self.sparse.len() as u32 && self.sparse[idx as usize] != SENTINEL {
            let dense_idx = self.sparse[idx as usize] as usize;
            if self.entities[dense_idx] == entity {
                // Same generation — update existing value
                self.dense[dense_idx] = value;
            } else {
                // Different generation — stale handle or reused slot
                // Replace both value and entity handle
                self.dense[dense_idx] = value;
                self.entities[dense_idx] = entity;
            }
        } else {
            // Insert new
            self.ensure_sparse(entity);
            let dense_idx = self.dense.len();
            self.dense.push(value);
            self.entities.push(entity);
            self.sparse[idx as usize] = dense_idx as u32;
        }
    }

    /// Get a reference to the component for an entity, if present.
    pub fn get(&self, entity: u64) -> Option<&T> {
        let idx = self.sparse_idx(entity);
        if idx < self.sparse.len() as u32 {
            let dense_idx = self.sparse[idx as usize];
            if dense_idx != SENTINEL {
                // Verify full handle matches (generation-aware)
                if self.entities[dense_idx as usize] == entity {
                    return Some(&self.dense[dense_idx as usize]);
                }
            }
        }
        None
    }

    /// Remove the component for an entity, returning it if present.
    pub fn remove(&mut self, entity: u64) -> Option<T> {
        let idx = self.sparse_idx(entity);
        if idx >= self.sparse.len() as u32 {
            return None;
        }
        let dense_idx = self.sparse[idx as usize];
        if dense_idx == SENTINEL {
            return None;
        }
        // Verify full handle matches (generation-aware)
        if self.entities[dense_idx as usize] != entity {
            return None;
        }
        // Swap with last element
        let last = self.dense.len() - 1;
        if dense_idx as usize != last {
            self.dense.swap(dense_idx as usize, last);
            self.entities.swap(dense_idx as usize, last);
            let moved_entity = self.entities[dense_idx as usize];
            let moved_sparse_idx = self.sparse_idx(moved_entity);
            self.sparse[moved_sparse_idx as usize] = dense_idx;
        }
        let value = self.dense.pop();
        self.entities.pop();
        self.sparse[idx as usize] = SENTINEL;
        value
    }

    /// Returns true if the entity has this component.
    pub fn contains(&self, entity: u64) -> bool {
        self.get(entity).is_some()
    }

    /// Number of entities with this component.
    pub fn len(&self) -> usize {
        self.dense.len()
    }

    pub fn is_empty(&self) -> bool {
        self.dense.is_empty()
    }

    /// Iterate over (entity_id, &component) pairs in contiguous order.
    pub fn iter(&self) -> impl Iterator<Item = (u64, &T)> + '_ {
        self.entities.iter().copied().zip(self.dense.iter())
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.dense.clear();
        self.entities.clear();
        self.sparse.clear();
    }

    fn sparse_idx(&self, entity: u64) -> u32 {
        entity as u32 // entity IDs fit in u32 for this impl
    }

    fn ensure_sparse(&mut self, entity: u64) {
        let idx = self.sparse_idx(entity) as usize;
        if idx >= self.sparse.len() {
            self.sparse.resize(idx + 1, SENTINEL);
        }
    }
}

impl<T> Default for SparseSet<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Behavioral equivalence test helper.
pub fn assert_sparse_equivalence<T: Clone + PartialEq + std::fmt::Debug>(
    set: &SparseSet<T>,
    map: &std::collections::BTreeMap<Entity, T>,
) {
    for (&entity, value) in map {
        assert_eq!(
            set.get(entity.id()),
            Some(value),
            "sparse set missing entity {}",
            entity.id()
        );
    }
    for (entity, value) in set.iter() {
        assert_eq!(
            map.get(&Entity::new(entity, 0)),
            Some(value),
            "BTreeMap missing entity {}",
            entity
        );
    }
    assert_eq!(set.len(), map.len(), "length mismatch");
}
