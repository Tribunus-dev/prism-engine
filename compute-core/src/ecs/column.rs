//! Column<T> — SparseSet component storage using u64 entity IDs.
//!
//! This is the concrete storage backend for the existing `ComponentStore`,
//! replacing `HashMap<EntityId, T>` with a dense+sparse column. It provides
//! O(1) lookup and cache-friendly dense iteration, matching the runtime
//! world's `ComponentVec` but with u64 entity IDs (no generation wrapper).

use serde::{Deserialize, Serialize};
use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt::Debug;

/// Dense SparseSet column keyed by u64 entity IDs.
///
/// Each column stores one component type. Entity IDs index into a sparse
/// array that points to a dense storage slot. Swap-remove maintains O(1)
/// amortized removal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Column<T> {
    /// Dense array — contiguous component values.
    dense: Vec<T>,
    /// Sparse index: entity_id -> Some(dense_idx) or None.
    sparse: Vec<Option<u32>>,
    /// Entity id for each dense entry (for removal fixup and iteration).
    entity_ids: Vec<u64>,
}

impl<T> Column<T> {
    pub fn new() -> Self {
        Self {
            dense: Vec::new(),
            sparse: Vec::new(),
            entity_ids: Vec::new(),
        }
    }

    /// Insert or replace a component for an entity.
    pub fn insert(&mut self, entity: u64, value: T) {
        let id = entity as usize;
        if id >= self.sparse.len() {
            self.sparse.resize(id + 1, None);
        }
        if let Some(dense_idx) = self.sparse[id] {
            // Replace in-place
            self.dense[dense_idx as usize] = value;
            return;
        }
        let dense_idx = self.dense.len() as u32;
        self.sparse[id] = Some(dense_idx);
        self.dense.push(value);
        self.entity_ids.push(entity);
    }

    /// Get a shared reference to an entity's component.
    pub fn get(&self, entity: u64) -> Option<&T> {
        let id = entity as usize;
        let dense_idx = self.sparse.get(id).and_then(|&opt| opt)?;
        self.dense.get(dense_idx as usize)
    }

    /// Get a mutable reference to an entity's component.
    pub fn get_mut(&mut self, entity: u64) -> Option<&mut T> {
        let id = entity as usize;
        let dense_idx = self.sparse.get_mut(id).and_then(|opt| *opt)?;
        self.dense.get_mut(dense_idx as usize)
    }

    /// Remove an entity's component, returning the old value.
    /// Uses swap-remove for O(1) amortized.
    pub fn remove(&mut self, entity: u64) -> Option<T> {
        let id = entity as usize;
        let dense_idx = self.sparse.get_mut(id)?.take()? as usize;
        let last = self.dense.len() - 1;
        if dense_idx != last {
            self.dense.swap(dense_idx, last);
            let swapped = self.entity_ids[last];
            self.sparse[swapped as usize] = Some(dense_idx as u32);
            self.entity_ids.swap(dense_idx, last);
        }
        let value = self.dense.pop();
        self.entity_ids.pop();
        value
    }

    /// Check if an entity has a component of this type.
    pub fn has(&self, entity: u64) -> bool {
        let id = entity as usize;
        self.sparse.get(id).and_then(|&opt| opt).is_some()
    }

    /// Number of components stored.
    pub fn len(&self) -> usize {
        self.dense.len()
    }
    pub fn is_empty(&self) -> bool {
        self.dense.is_empty()
    }

    /// Clear all components (does not reset sparse indices).
    pub fn clear(&mut self) {
        self.dense.clear();
        self.entity_ids.clear();
        // Reset sparse indices to prevent stale-index OOB panic on reinsert.
        for entry in &mut self.sparse {
            *entry = None;
        }
    }

    /// Access the dense array directly.
    pub fn dense(&self) -> &[T] {
        &self.dense
    }

    /// Access entity IDs in dense order.
    pub fn entity_ids(&self) -> &[u64] {
        &self.entity_ids
    }

    /// Iterate over (entity_id, &T) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (u64, &T)> + '_ {
        self.dense
            .iter()
            .enumerate()
            .map(|(i, v)| (self.entity_ids[i], v))
    }
}

// ---------------------------------------------------------------------------
// ErasedColumn trait — type-erased interface for uniform column operations
// ---------------------------------------------------------------------------

/// Type-erased interface for component columns.
///
/// Enables operations like "remove this entity from every column" without
/// knowing the concrete component type at the call site.
pub trait ErasedColumn: Debug + Send + Sync {
    /// Remove the entity from this column (swap-remove semantics).
    fn remove_entity(&mut self, entity: u64) -> bool;

    /// Check if this column contains the entity.
    fn has_entity(&self, entity: u64) -> bool;

    /// Remove all entries from this column.
    fn clear(&mut self);

    /// Number of entries in this column.
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Downcast to `Any` for typed access.
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;

    /// Human-readable type name for diagnostics.
    fn type_name(&self) -> &'static str;
}

impl<T: 'static + Send + Sync + Debug> ErasedColumn for Column<T> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn remove_entity(&mut self, entity: u64) -> bool {
        self.remove(entity).is_some()
    }

    fn has_entity(&self, entity: u64) -> bool {
        self.has(entity)
    }

    fn clear(&mut self) {
        Column::clear(self);
    }

    fn len(&self) -> usize {
        Column::len(self)
    }

    fn type_name(&self) -> &'static str {
        std::any::type_name::<T>()
    }
}

// ---------------------------------------------------------------------------
// ColumnStore — holds all columns by TypeId
// ---------------------------------------------------------------------------

/// Type-erased column storage, indexed by (TypeId, EntityId).
#[derive(Debug, Default)]
pub struct ColumnStore {
    pub(crate) data: HashMap<TypeId, Box<dyn ErasedColumn>>,
}

impl ColumnStore {
    /// Remove an entity from every registered column.
    /// Returns true if the entity was found in at least one column.
    pub fn remove_entity_from_all(&mut self, entity: u64) -> bool {
        let mut found = false;
        for (_, col) in self.data.iter_mut() {
            if col.remove_entity(entity) {
                found = true;
            }
        }
        found
    }

    /// Return (count, type_name) for every non-empty column.
    pub fn diagnose(&self) -> Vec<(usize, &'static str)> {
        self.data
            .iter()
            .map(|(_, col)| (col.len(), col.type_name()))
            .filter(|(len, _)| *len > 0)
            .collect()
    }

    /// Get or create a column for component type T, returning a mutable ref.
    pub fn column_mut<T: 'static + Send + Sync + std::fmt::Debug>(&mut self) -> &mut Column<T> {
        let key = TypeId::of::<T>();
        self.data
            .entry(key)
            .or_insert_with(|| Box::new(Column::<T>::new()))
            .as_any_mut()
            .downcast_mut::<Column<T>>()
            .expect("Column<T> type mismatch")
    }

    /// Get a shared reference to a column for component type T.
    pub fn column<T: 'static + Send + Sync + std::fmt::Debug>(&self) -> Option<&Column<T>> {
        let key = TypeId::of::<T>();
        Some(
            self.data
                .get(&key)?
                .as_any()
                .downcast_ref::<Column<T>>()
                .expect("Column<T> type mismatch"),
        )
    }

    /// Get a mutable reference to a column for component type T.
    pub fn column_mut_ref<T: 'static + Send + Sync + std::fmt::Debug>(
        &mut self,
    ) -> Option<&mut Column<T>> {
        let key = TypeId::of::<T>();
        Some(
            self.data
                .get_mut(&key)?
                .as_any_mut()
                .downcast_mut::<Column<T>>()
                .expect("Column<T> type mismatch"),
        )
    }

    /// Insert a component for an entity.
    pub fn insert<T: 'static + Send + Sync + std::fmt::Debug>(&mut self, entity: u64, value: T) {
        self.column_mut::<T>().insert(entity, value);
    }

    /// Get a shared reference to an entity's component.
    pub fn get<T: 'static + Send + Sync + std::fmt::Debug>(&self, entity: u64) -> Option<&T> {
        self.column::<T>()?.get(entity)
    }

    /// Remove an entity's component.
    pub fn remove<T: 'static + Send + Sync + std::fmt::Debug>(&mut self, entity: u64) -> Option<T> {
        self.column_mut_ref::<T>()?.remove(entity)
    }

    /// Check if an entity has a component of this type.
    pub fn has<T: 'static + Send + Sync + std::fmt::Debug>(&self, entity: u64) -> bool {
        self.column::<T>().map(|c| c.has(entity)).unwrap_or(false)
    }

    /// Check if a column exists and is non-empty.
    pub fn contains_column<T: 'static + Send + Sync + std::fmt::Debug>(&self) -> bool {
        self.column::<T>().map(|c| c.len() > 0).unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clear_does_not_cause_oob_on_reinsert() {
        let mut col: Column<i32> = Column::new();
        col.insert(42, 100);
        col.insert(84, 200);
        assert_eq!(col.len(), 2);

        col.clear();
        assert_eq!(col.len(), 0);

        // Reinsert on a previously-stored entity — stale sparse index must
        // be reset or this would try to write into the empty dense vector.
        col.insert(42, 300);
        assert_eq!(col.len(), 1);
        assert_eq!(col.get(42), Some(&300));
    }
}
