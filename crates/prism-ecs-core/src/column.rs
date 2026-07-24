//! Column<T> — Generation-aware SparseSet component storage keyed by Entity(id, gen).
//!
//! This is the concrete storage backend for the existing `ComponentStore`,
//! replacing `HashMap<EntityId, T>` with a dense+sparse column. Every access
//! checks the entity generation, preventing stale handles from observing or
//! mutating components belonging to a newer entity occupying the same slot.

use serde::{Deserialize, Serialize};
use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt::Debug;

use crate::Entity;

/// Generation-aware SparseSet column keyed by Entity(id, generation).
///
/// Each column stores one component type. Entity id() values index into a
/// sparse array that points to a dense storage slot. Every access validates
/// that the stored entity generation matches the requested generation —
/// stale handles (from despawned/respawned slots) are rejected.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Column<T> {
    /// Dense array — contiguous component values.
    dense: Vec<T>,
    /// Sparse index: entity_id -> Some(dense_idx) or None.
    sparse: Vec<Option<u32>>,
    /// Entity (id, generation) for each dense entry (for removal fixup and iteration).
    entities: Vec<Entity>,
}

impl<T> Default for Column<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Column<T> {
    /// Debug-only invariant check: sparse ↔ dense ↔ entities consistency.
    fn debug_check_invariants(&self) {
        debug_assert_eq!(
            self.dense.len(),
            self.entities.len(),
            "dense/entities length mismatch"
        );
        for (i, slot) in self.sparse.iter().enumerate() {
            if let Some(dense_idx) = slot {
                let idx = *dense_idx as usize;
                debug_assert!(
                    idx < self.entities.len(),
                    "sparse[{}] points past entities end ({})",
                    i,
                    self.entities.len()
                );
                debug_assert_eq!(
                    self.entities[idx].id() as usize,
                    i,
                    "sparse[{}] points to entity[{}] with wrong id {}",
                    i,
                    idx,
                    self.entities[idx].id()
                );
            }
        }
    }

    pub fn new() -> Self {
        Self {
            dense: Vec::new(),
            sparse: Vec::new(),
            entities: Vec::new(),
        }
    }

    /// Insert or replace a component for an entity (generation-aware).
    ///
    /// If the slot is occupied by a stale generation (entity reused after
    /// despawn), the stale entry is evicted first by swap-remove, then the
    /// new value is inserted. Same-generation replaces in place.
    pub fn insert(&mut self, entity: Entity, value: T) {
        let id = entity.id() as usize;
        if id >= self.sparse.len() {
            self.sparse.resize(id + 1, None);
        }
        if let Some(dense_idx) = self.sparse[id] {
            let stored = &self.entities[dense_idx as usize];
            if stored.generation() == entity.generation() {
                // Same generation — replace in-place
                self.dense[dense_idx as usize] = value;
                return;
            }
            // Stale generation — evict old entry first
            self.remove_dense_entry(dense_idx as usize);
        }
        let dense_idx = self.dense.len() as u32;
        self.sparse[id] = Some(dense_idx);
        self.dense.push(value);
        self.entities.push(entity);
        self.debug_check_invariants();
    }

    /// Get a shared reference to an entity's component (generation-checked).
    /// Returns None if the entity has no component or the handle is stale.
    pub fn get(&self, entity: Entity) -> Option<&T> {
        let id = entity.id() as usize;
        let dense_idx = self.sparse.get(id).and_then(|&opt| opt)?;
        let stored = self.entities.get(dense_idx as usize)?;
        if stored.generation() != entity.generation() {
            return None;
        }
        self.dense.get(dense_idx as usize)
    }

    /// Get a mutable reference to an entity's component (generation-checked).
    /// Returns None if the entity has no component or the handle is stale.
    pub fn get_mut(&mut self, entity: Entity) -> Option<&mut T> {
        let id = entity.id() as usize;
        let dense_idx = self.sparse.get_mut(id).and_then(|opt| *opt)?;
        let stored = self.entities.get(dense_idx as usize)?;
        if stored.generation() != entity.generation() {
            return None;
        }
        self.dense.get_mut(dense_idx as usize)
    }

    /// Remove an entity's component (generation-checked), returning the old value.
    /// Uses swap-remove for O(1) amortized.
    /// Returns None if the entity has no component or the handle is stale.
    pub fn remove(&mut self, entity: Entity) -> Option<T> {
        let id = entity.id() as usize;
        let dense_idx = *self.sparse.get(id)?;
        let dense_idx = dense_idx? as usize;
        let stored = self.entities.get(dense_idx)?;
        if stored.generation() != entity.generation() {
            return None;
        }
        self.sparse[id] = None;
        let val = self.remove_dense_entry(dense_idx);
        self.debug_check_invariants();
        Some(val)
    }

    /// Check if an entity has a component of this type (generation-checked).
    /// Returns false for stale handles.
    pub fn has(&self, entity: Entity) -> bool {
        let id = entity.id() as usize;
        match self.sparse.get(id).and_then(|&opt| opt) {
            Some(dense_idx) => self
                .entities
                .get(dense_idx as usize)
                .map(|e| e.generation() == entity.generation())
                .unwrap_or(false),
            None => false,
        }
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
        self.entities.clear();
        // Reset sparse indices to prevent stale-index OOB panic on reinsert.
        for entry in &mut self.sparse {
            *entry = None;
        }
        self.debug_check_invariants();
    }

    /// Access the dense array directly.
    pub fn dense(&self) -> &[T] {
        &self.dense
    }

    /// Access the dense array mutably.
    pub fn dense_mut(&mut self) -> &mut [T] {
        &mut self.dense
    }

    /// Access the stored entities in dense order (with generation).
    pub fn entities(&self) -> &[Entity] {
        &self.entities
    }

    /// Iterate over (Entity, &T) pairs (generation included).
    pub fn iter(&self) -> impl Iterator<Item = (Entity, &T)> + '_ {
        self.dense
            .iter()
            .enumerate()
            .map(|(i, v)| (self.entities[i], v))
    }

    /// Internal: swap-remove the entry at the given dense index.
    /// The caller is responsible for clearing the sparse entry before calling
    /// this when performing a generation-checked remove.
    fn remove_dense_entry(&mut self, dense_idx: usize) -> T {
        let last = self.dense.len() - 1;
        if dense_idx != last {
            self.dense.swap(dense_idx, last);
            let swapped = self.entities[last];
            self.sparse[swapped.id() as usize] = Some(dense_idx as u32);
            self.entities.swap(dense_idx, last);
        }
        self.entities.pop();
        let val = self
            .dense
            .pop()
            .expect("remove_dense_entry on empty column");
        self.debug_check_invariants();
        val
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
    /// Remove the entity from this column (generation-checked).
    fn remove_entity(&mut self, entity: Entity) -> bool;

    /// Check if this column contains the entity (generation-checked).
    fn has_entity(&self, entity: Entity) -> bool;

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

    fn remove_entity(&mut self, entity: Entity) -> bool {
        self.remove(entity).is_some()
    }

    fn has_entity(&self, entity: Entity) -> bool {
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
    /// Remove an entity from every registered column (generation-checked).
    pub fn remove_entity_from_all(&mut self, entity: Entity) -> bool {
        let mut found = false;
        for col in self.data.values_mut() {
            if col.remove_entity(entity) {
                found = true;
            }
        }
        found
    }

    /// Return (count, type_name) for every non-empty column.
    pub fn diagnose(&self) -> Vec<(usize, &'static str)> {
        self.data
            .values()
            .map(|col| (col.len(), col.type_name()))
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

    /// Insert a component for an entity (generation-aware).
    pub fn insert<T: 'static + Send + Sync + std::fmt::Debug>(&mut self, entity: Entity, value: T) {
        self.column_mut::<T>().insert(entity, value);
    }

    /// Get a shared reference to an entity's component (generation-checked).
    pub fn get<T: 'static + Send + Sync + std::fmt::Debug>(&self, entity: Entity) -> Option<&T> {
        self.column::<T>()?.get(entity)
    }

    /// Remove an entity's component (generation-checked).
    pub fn remove<T: 'static + Send + Sync + std::fmt::Debug>(
        &mut self,
        entity: Entity,
    ) -> Option<T> {
        self.column_mut_ref::<T>()?.remove(entity)
    }

    /// Check if an entity has a component of this type (generation-checked).
    pub fn has<T: 'static + Send + Sync + std::fmt::Debug>(&self, entity: Entity) -> bool {
        self.column::<T>().map(|c| c.has(entity)).unwrap_or(false)
    }

    /// Check if a column exists and is non-empty.
    pub fn contains_column<T: 'static + Send + Sync + std::fmt::Debug>(&self) -> bool {
        self.column::<T>().map(|c| !c.is_empty()).unwrap_or(false)
    }
}
