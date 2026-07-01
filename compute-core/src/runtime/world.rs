//! Minimal ECS world for Prism Engine agent management.
//!
//! An entity is a u32 index into component storage arrays.  A generation
//! counter guards against stale references after entity reclamation.  Each
//! component type is stored as a SparseSet (dense array + sparse index)
//! for cache-friendly iteration.
//!
//! Resources are singletons (TypeId → Box<dyn Any>) shared across systems.

use std::any::TypeId;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Entity
// ---------------------------------------------------------------------------

/// Unique entity identifier.  The u32 pairs with a generation counter stored
/// in the World to detect use-after-despawn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Entity(pub u32);

// ---------------------------------------------------------------------------
// Component trait
// ---------------------------------------------------------------------------

/// Marker trait for types that can be stored as components.
pub trait Component: 'static {}
impl<T: 'static> Component for T {}

// ---------------------------------------------------------------------------
// Resource trait
// ---------------------------------------------------------------------------

/// Marker trait for singleton resources accessible by systems.
pub trait Resource: 'static + Send + Sync {}
impl<T: 'static + Send + Sync> Resource for T {}

// ---------------------------------------------------------------------------
// Component storage — SparseSet
// ---------------------------------------------------------------------------

/// Dense storage backed by a sparse index for O(1) lookup and cache-friendly
/// dense iteration.
pub struct ComponentVec<T: Component> {
    /// Dense array — contiguous component data.  Indexed by `sparse[i]`.
    dense: Vec<T>,
    /// Sparse index: entity_id → Some(dense_idx) or None.
    sparse: Vec<Option<u32>>,
    /// Entity id for each dense entry (needed for iteration + removal).
    entity_ids: Vec<u32>,
}

impl<T: Component> ComponentVec<T> {
    pub fn new() -> Self {
        Self {
            dense: Vec::new(),
            sparse: Vec::new(),
            entity_ids: Vec::new(),
        }
    }

    pub fn insert(&mut self, entity: Entity, value: T) {
        let id = entity.0 as usize;
        if id >= self.sparse.len() {
            self.sparse.resize(id + 1, None);
        }
        if let Some(dense_idx) = self.sparse[id] {
            // Entity already has this component — swap in place
            self.dense[dense_idx as usize] = value;
            return;
        }
        let dense_idx = self.dense.len() as u32;
        self.sparse[id] = Some(dense_idx);
        self.dense.push(value);
        self.entity_ids.push(entity.0);
    }

    pub fn get(&self, entity: Entity) -> Option<&T> {
        let id = entity.0 as usize;
        self.sparse
            .get(id)?
            .and_then(|dense_idx| {
                let i = dense_idx as usize;
                if i < self.dense.len() { Some(&self.dense[i]) } else { None }
            })
    }

    pub fn get_mut(&mut self, entity: Entity) -> Option<&mut T> {
        let id = entity.0 as usize;
        self.sparse
            .get_mut(id)?
            .and_then(|dense_idx| {
                let i = dense_idx as usize;
                if i < self.dense.len() { Some(&mut self.dense[i]) } else { None }
            })
    }

    pub fn remove(&mut self, entity: Entity) -> Option<T> {
        let id = entity.0 as usize;
        let dense_idx = self.sparse.get_mut(id)?.take()? as usize;
        let last = self.dense.len() - 1;
        if dense_idx != last {
            self.dense.swap(dense_idx, last);
            let swapped_entity = self.entity_ids[last];
            self.sparse[swapped_entity as usize] = Some(dense_idx as u32);
            self.entity_ids.swap(dense_idx, last);
        }
        let value = self.dense.pop();
        self.entity_ids.pop();
        value
    }

    pub fn has(&self, entity: Entity) -> bool {
        let id = entity.0 as usize;
        self.sparse.get(id).and_then(|&d| d).is_some()
    }

    pub fn iter(&self) -> impl Iterator<Item = (Entity, &T)> + '_ {
        self.dense
            .iter()
            .enumerate()
            .map(|(i, v)| (Entity(self.entity_ids[i]), v))
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (Entity, &mut T)> + '_ {
        self.dense
            .iter_mut()
            .enumerate()
            .map(|(i, v)| (Entity(self.entity_ids[i]), v))
    }

    pub fn len(&self) -> usize {
        self.dense.len()
    }

    pub fn is_empty(&self) -> bool {
        self.dense.is_empty()
    }

    pub fn clear(&mut self) {
        self.dense.clear();
        self.entity_ids.clear();
        // Don't clear sparse — generation tracking handles reuse.
        // Resetting sparse would break the invariant.  Entities that were
        // removed still have None in sparse.
    }
}

// ---------------------------------------------------------------------------
// World
// ---------------------------------------------------------------------------

/// The ECS world.  Owns all entities, components, and resources.
pub struct World {
    /// Generations for each entity slot.  Incremented on despawn.
    generations: Vec<u32>,
    /// Free entity slots available for reuse.
    free_list: Vec<u32>,
    /// Next entity ID to try if free_list is empty.
    next_entity: u32,
    /// Maximum entity capacity (soft cap to bound memory).
    max_entities: u32,

    /// Component storage keyed by TypeId.
    components: HashMap<TypeId, Box<dyn std::any::Any>>,
    /// Resource storage keyed by TypeId.
    resources: HashMap<TypeId, Box<dyn std::any::Any + Send + Sync>>,
    /// Deserializer functions keyed by TypeId for type-erased insertion.
    deserializers: HashMap<TypeId, fn(&mut World, Entity, &[u8])>,
}

unsafe impl Send for World {}
unsafe impl Sync for World {}

impl World {
    /// Create a new empty world with the given entity capacity.
    pub fn with_capacity(max_entities: u32) -> Self {
        Self {
            generations: Vec::new(),
            free_list: Vec::new(),
            next_entity: 0,
            max_entities,
            components: HashMap::new(),
            resources: HashMap::new(),
            deserializers: HashMap::new(),
        }
    }

    /// Create a new world with default capacity (32).
    pub fn new() -> Self {
        Self::with_capacity(32)
    }

    /// Spawn a new entity, returning its handle.
    /// Returns None if at capacity.
    pub fn spawn(&mut self) -> Option<Entity> {
        let id = if let Some(free) = self.free_list.pop() {
            free
        } else {
            let id = self.next_entity;
            if id >= self.max_entities {
                return None;
            }
            self.next_entity += 1;
            // Ensure generation storage is sized
            if id as usize >= self.generations.len() {
                self.generations.resize((id + 1) as usize, 0);
            }
            id
        };
        Some(Entity(id))
    }

    /// Check if an entity is still alive.
    pub fn is_alive(&self, entity: Entity) -> bool {
        let id = entity.0 as usize;
        id < self.generations.len() && !self.free_list.contains(&entity.0)
    }

    /// Despawn an entity, removing all its components and freeing the slot.
    pub fn despawn(&mut self, entity: Entity) {
        let id = entity.0 as usize;
        if id >= self.generations.len() {
            return;
        }
        // Remove this entity's components from all storages
        for (_, storage) in self.components.iter_mut() {
            // Each storage type needs to support removal by entity ID.
            // We use a helper that downcasts and calls remove.
            remove_component_from_any(storage.as_mut(), entity);
        }
        self.generations[id] += 1;
        self.free_list.push(entity.0);
    }

    // -----------------------------------------------------------------------
    // Component operations
    // -----------------------------------------------------------------------

    /// Register a component type for use.  Returns true if newly registered.
    pub fn register_component<T: Component>(&mut self) -> bool {
        let tid = TypeId::of::<ComponentVec<T>>();
        if self.components.contains_key(&tid) {
            return false;
        }
        self.components
            .insert(tid, Box::new(ComponentVec::<T>::new()));
        true
    }

    /// Ensure a component type's storage exists.
    fn ensure_storage<T: Component>(&mut self) -> &mut ComponentVec<T> {
        let tid = TypeId::of::<ComponentVec<T>>();
        self.components
            .entry(tid)
            .or_insert_with(|| Box::new(ComponentVec::<T>::new()))
            .downcast_mut::<ComponentVec<T>>()
            .expect("ComponentVec<T> storage type mismatch")
    }

    fn storage<T: Component>(&self) -> Option<&ComponentVec<T>> {
        let tid = TypeId::of::<ComponentVec<T>>();
        self.components
            .get(&tid)?
            .downcast_ref::<ComponentVec<T>>()
    }

    fn storage_mut<T: Component>(&mut self) -> Option<&mut ComponentVec<T>> {
        let tid = TypeId::of::<ComponentVec<T>>();
        self.components
            .get_mut(&tid)?
            .downcast_mut::<ComponentVec<T>>()
    }

    /// Insert a component onto an entity.  Panics if the entity is out of
    /// range (spawn must be called first).
    pub fn insert<T: Component>(&mut self, entity: Entity, component: T) {
        let id = entity.0 as usize;
        if id >= self.generations.len() || self.free_list.contains(&entity.0) {
            panic!(
                "Cannot insert component on dead or out-of-range entity {}",
                entity.0
            );
        }
        self.ensure_storage::<T>().insert(entity, component);
    }

    /// Get a shared reference to an entity's component.
    pub fn get<T: Component>(&self, entity: Entity) -> Option<&T> {
        self.storage::<T>()?.get(entity)
    }

    /// Get an exclusive reference to an entity's component.
    pub fn get_mut<T: Component>(&mut self, entity: Entity) -> Option<&mut T> {
        self.storage_mut::<T>()?.get_mut(entity)
    }

    /// Remove an entity's component, returning the old value.
    pub fn remove<T: Component>(&mut self, entity: Entity) -> Option<T> {
        self.storage_mut::<T>()?.remove(entity)
    }

    /// Check whether an entity has a specific component.
    pub fn has<T: Component>(&self, entity: Entity) -> bool {
        self.storage::<T>()
            .map(|s| s.has(entity))
            .unwrap_or(false)
    }

    // -----------------------------------------------------------------------
    // Resources
    // -----------------------------------------------------------------------

    /// Insert a resource.
    pub fn insert_resource<T: Resource>(&mut self, resource: T) {
        let tid = TypeId::of::<T>();
        self.resources.insert(tid, Box::new(resource));
    }

    /// Get a shared reference to a resource.
    pub fn get_resource<T: Resource>(&self) -> Option<&T> {
        let tid = TypeId::of::<T>();
        self.resources
            .get(&tid)?
            .downcast_ref::<T>()
    }

    /// Get an exclusive reference to a resource.
    pub fn get_resource_mut<T: Resource>(&mut self) -> Option<&mut T> {
        let tid = TypeId::of::<T>();
        self.resources
            .get_mut(&tid)?
            .downcast_mut::<T>()
    }

    // -----------------------------------------------------------------------
    // Query — iterate entities matching a set of component types
    // -----------------------------------------------------------------------

    /// Iterate over all entity IDs that have component type A.
    /// Returns an iterator of Entity handles.  The caller then calls
    /// `world.get::<T>(entity)` to access component data.  This avoids
    /// complex borrow-checker gymnastics with lifetime-tied references.
    pub fn iter_entities_with<A: Component>(&self) -> EntityIter<'_, A> {
        EntityIter {
            storage: self.storage::<A>(),
            cursor: 0,
        }
    }

    /// Check whether any entity has component type A.
    pub fn any_with<A: Component>(&self) -> bool {
        self.storage::<A>()
            .map(|s| s.len() > 0)
            .unwrap_or(false)
    }

    /// Dump component storage for debugging.
    pub fn entity_count(&self) -> usize {
        self.components
            .values()
            .filter_map(|s| {
                crate::runtime::world::downcast_component_vec_len(
                    s as &dyn std::any::Any,
                )
            })
            .next()
            .unwrap_or(0)
    }

    /// Count of all registered components across all entities (for metrics).
    pub fn total_component_count(&self) -> usize {
        self.components
            .values()
            .map(|s| crate::runtime::world::component_vec_len(s as &dyn std::any::Any))
            .sum()
    }

    // -----------------------------------------------------------------------
    // Type-erased deserialization
    // -----------------------------------------------------------------------

    /// Register a deserializer for a component type `T`, enabling
    /// type-erased insertion via [`insert_raw`].
    ///
    /// Must be called before any `insert_raw` with this TypeId.
    pub fn register_deserializer<T: Component + serde::de::DeserializeOwned>(&mut self) {
        fn deserialize_and_insert<T: Component + serde::de::DeserializeOwned>(
            world: &mut World,
            entity: Entity,
            bytes: &[u8],
        ) {
            let value: T = serde_json::from_slice(bytes)
                .expect("Failed to deserialize component in insert_raw");
            world.insert::<T>(entity, value);
        }
        self.deserializers
            .insert(TypeId::of::<T>(), deserialize_and_insert::<T>);
    }

    /// Type-erased component insertion.
    ///
    /// Deserializes `bytes` as component type `type_id` and inserts it onto
    /// `entity`.  Returns an error if no deserializer has been registered for
    /// `type_id`, or if deserialization fails.
    pub fn insert_raw(
        &mut self,
        entity: Entity,
        type_id: TypeId,
        bytes: &[u8],
    ) -> Result<(), String> {
        let deserializer = self
            .deserializers
            .get(&type_id)
            .ok_or_else(|| format!("No deserializer registered for TypeId {:?}", type_id))?;
        deserializer(self, entity, bytes);
        Ok(())
    }
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// EntityIter — iterate entity IDs with a given component
// ---------------------------------------------------------------------------

/// Iterates over entity IDs that have a given component type.
/// The caller accesses component data via `World::get<T>(entity)`.
pub struct EntityIter<'w, A: Component> {
    storage: Option<&'w ComponentVec<A>>,
    cursor: usize,
}

impl<'w, A: Component> Iterator for EntityIter<'w, A> {
    type Item = Entity;

    fn next(&mut self) -> Option<Self::Item> {
        let storage = self.storage.as_ref()?;
        if self.cursor >= storage.entity_ids.len() {
            return None;
        }
        let id = storage.entity_ids[self.cursor];
        self.cursor += 1;
        Some(Entity(id))
    }
}

// ---------------------------------------------------------------------------
// Internal helpers for downcasting component Vecs
// ---------------------------------------------------------------------------

fn downcast_component_vec_len(storage: &dyn std::any::Any) -> Option<usize> {
    storage
        .downcast_ref::<ComponentVec<crate::runtime::agent_slot::AgentSlot>>()
        .map(|v| v.len())
}

fn component_vec_len(storage: &dyn std::any::Any) -> usize {
    if let Some(v) = storage
        .downcast_ref::<ComponentVec<crate::runtime::agent_slot::AgentSlot>>()
    {
        v.len()
    } else if let Some(v) = storage
        .downcast_ref::<ComponentVec<crate::runtime::components::KVCacheRef>>()
    {
        v.len()
    } else if let Some(v) = storage
        .downcast_ref::<ComponentVec<crate::runtime::components::AgentPayload>>()
    {
        v.len()
    } else if let Some(v) = storage
        .downcast_ref::<ComponentVec<crate::runtime::components::ToolRegistry>>()
    {
        v.len()
    } else {
        0
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Remove a component from any storage by downcasting.
fn remove_component_from_any(storage: &mut dyn std::any::Any, entity: Entity) {
    // Try each known component type
    if let Some(s) = storage.downcast_mut::<ComponentVec<crate::runtime::agent_slot::AgentSlot>>() {
        s.remove(entity);
    } else if let Some(s) = storage.downcast_mut::<ComponentVec<crate::runtime::components::KVCacheRef>>() {
        s.remove(entity);
    } else if let Some(s) = storage.downcast_mut::<ComponentVec<crate::runtime::components::AgentPayload>>() {
        s.remove(entity);
    } else if let Some(s) = storage.downcast_mut::<ComponentVec<crate::runtime::components::ToolRegistry>>() {
        s.remove(entity);
    }
}
