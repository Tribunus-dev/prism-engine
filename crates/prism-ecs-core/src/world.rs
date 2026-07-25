use std::any::{Any, TypeId};
use std::collections::HashMap;

use crate::column::Column;
use crate::component::Component;
use crate::entity::{Entity, EntityAllocation, EntityKind, SpawnedEntity};
use crate::epoch::WorldEpoch;
use crate::error::WorldError;
use crate::mutation::MutationPolicy;
use crate::query::{Query, Query2, Query3, QueryMut};
use crate::resource::{ResourceMut, ResourceRef};
use crate::store::{ComponentStore, ResourceStore};
use crate::WorldCapacity;

type StagingAction = Box<dyn FnOnce(&mut ComponentStore) + Send + 'static>;

// ---------------------------------------------------------------------------
// Per-entity slot — generation persists across despawn/reuse cycles.
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct EntitySlot {
    generation: u32,
    occupant: Option<Occupant>,
}

#[derive(Debug)]
pub struct Occupant {
    kind: EntityKind,
    name: Option<String>,
}

impl Default for EntitySlot {
    fn default() -> Self {
        Self {
            generation: 0,
            occupant: Some(Occupant {
                kind: EntityKind::Model,
                name: None,
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// Generation-safe entity reference.
// ---------------------------------------------------------------------------

/// Generation-safe entity reference for accessing entity components.
/// Created by [`World::entity_ref()`], which performs a generation check
/// to ensure the handle is still valid.
#[allow(dead_code)]
#[derive(Debug)]
pub struct EntityRef<'w> {
    pub(crate) entity: Entity,
    pub(crate) generation: u32,
    pub world: &'w World,
}

// ---------------------------------------------------------------------------
// World — core ECS container.
// ---------------------------------------------------------------------------

/// The ECS world — all entities, components, and resources.
///
/// Core fields (component store, resource store, entity metadata) are defined
/// directly. Compute-core-specific state (systems, epoch, journal, events) is
/// stored via type-erased [`extensions`] so that this crate has no dependency
/// on [`tribunus_compute_core`](crate).
pub struct World {
    pub(crate) component_store: ComponentStore,
    pub(crate) resource_store: ResourceStore,
    pub(crate) entity_meta: Vec<Option<EntitySlot>>,
    pub(crate) next_id: u64,
    pub(crate) free_list: Vec<u64>,
    pub(crate) staging: Vec<StagingAction>,
    pub(crate) component_versions: HashMap<u64, u64>,
    /// Mutation access policy. Controls whether direct mutations are allowed
    /// or must go through WorldTxn. Defaults to Bootstrap for backward
    /// compatibility during migration.
    pub(crate) mutation_policy: MutationPolicy,
    /// Type-erased extension map — used by [`tribunus_compute_core`] to store
    /// `SystemStage`, `WorldEpoch`, `Vec<ComponentChange>`, `Vec<DomainEvent>`,
    /// and any other compute-core-specific state without coupling this crate.
    pub(crate) extensions: HashMap<TypeId, Box<dyn Any + Send + 'static>>,
}

impl std::fmt::Debug for World {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("World")
            .field(
                "entity_count",
                &self
                    .entity_meta
                    .iter()
                    .filter(|s| s.as_ref().and_then(|s| s.occupant.as_ref()).is_some())
                    .count(),
            )
            .field("staged_changes", &self.staging.len())
            .field("extension_count", &self.extensions.len())
            .finish()
    }
}

impl World {
    // ── Constructors ──────────────────────────────────────────────────────────

    /// Create a new empty world.
    ///
    /// Core ECS state is initialized. Compute-core extensions
    /// (epoch, journal, events, systems) must be registered separately by
    /// the caller — see [`set_extension`](Self::set_extension).
    pub fn new() -> Self {
        Self {
            component_store: ComponentStore::default(),
            resource_store: ResourceStore::default(),
            entity_meta: Vec::new(),
            next_id: 1,
            free_list: Vec::new(),
            staging: Vec::new(),
            component_versions: HashMap::new(),
            mutation_policy: MutationPolicy::Bootstrap,
            extensions: HashMap::new(),
        }
    }

    /// Create a new world with the given capacity hints.
    pub fn with_capacity(capacity: &WorldCapacity) -> Self {
        Self {
            component_store: ComponentStore::default(),
            resource_store: ResourceStore::default(),
            entity_meta: Vec::with_capacity(capacity.entity_capacity as usize),
            next_id: 1,
            free_list: Vec::new(),
            staging: Vec::new(),
            component_versions: HashMap::new(),
            mutation_policy: MutationPolicy::Bootstrap,
            extensions: HashMap::new(),
        }
    }

    // ── Component store accessors ───────────────────────────────────────────

    /// Access the component store (shared reference).
    pub fn component_store(&self) -> &ComponentStore {
        &self.component_store
    }

    /// Access the component store (mutable reference).
    pub fn component_store_mut(&mut self) -> &mut ComponentStore {
        &mut self.component_store
    }

    /// Access the component versions map (mutable reference).
    pub fn component_versions_mut(&mut self) -> &mut HashMap<u64, u64> {
        &mut self.component_versions
    }

    // ── Extension accessors ───────────────────────────────────────────────────

    /// Store a type-erased extension value.
    pub fn set_extension<T: 'static + Send>(&mut self, ext: T) {
        self.extensions.insert(TypeId::of::<T>(), Box::new(ext));
    }

    /// Borrow a type-erased extension.
    pub fn get_extension<T: 'static>(&self) -> Option<&T> {
        self.extensions.get(&TypeId::of::<T>())?.downcast_ref::<T>()
    }

    /// Mutably borrow a type-erased extension.
    pub fn get_extension_mut<T: 'static>(&mut self) -> Option<&mut T> {
        self.extensions
            .get_mut(&TypeId::of::<T>())?
            .downcast_mut::<T>()
    }

    // ─── Entity lifetime ──────────────────────────────────────────────────────

    /// Validate that an entity handle is still valid (the slot is occupied and
    /// the generation matches). Returns None if the handle is stale or the
    /// entity has been despawned.
    fn validate_generation(&self, entity: Entity) -> Option<()> {
        if entity.0 == 0 {
            return None;
        }
        let idx = (entity.0 - 1) as usize;
        self.entity_meta.get(idx).and_then(|slot| {
            let slot = slot.as_ref()?;
            if slot.occupant.is_some() && slot.generation == entity.1 {
                Some(())
            } else {
                None
            }
        })
    }

    /// Spawn entity with kind and optional name.
    pub fn spawn(
        &mut self,
        kind: EntityKind,
        name: Option<String>,
    ) -> Result<SpawnedEntity, WorldError> {
        if !self.mutation_policy.direct_mutations_allowed() {
            return Err(WorldError::DirectMutationDisallowed { operation: "spawn" });
        }
        // Construct the Occupant with the final name upfront. The previous shape
        // set `name: None` and then re-borrowed the slot through a chain of
        // `Option::unwrap` calls to set the name; that produced 9 unwraps the
        // constitutional no-panic rule rejects. Building once is also clearer.
        let occupant = Occupant { kind, name };
        let (entity, allocation) = if let Some(free) = self.free_list.pop() {
            let idx = (free - 1) as usize;
            if idx < self.entity_meta.len() {
                if let Some(Some(slot)) = self.entity_meta.get_mut(idx) {
                    let prev_gen = slot.generation;
                    slot.generation += 1;
                    slot.occupant = Some(occupant);
                    (
                        Entity::new(free, prev_gen + 1),
                        EntityAllocation::ReusedSlot {
                            previous_generation: prev_gen,
                        },
                    )
                } else {
                    self.entity_meta[idx] = Some(EntitySlot {
                        generation: 0,
                        occupant: Some(occupant),
                    });
                    (Entity::new(free, 0), EntityAllocation::NewSlot)
                }
            } else {
                self.entity_meta.push(Some(EntitySlot {
                    generation: 0,
                    occupant: Some(occupant),
                }));
                (Entity::new(free, 0), EntityAllocation::NewSlot)
            }
        } else {
            let id = self.next_id;
            self.next_id += 1;
            self.entity_meta.push(Some(EntitySlot {
                generation: 0,
                occupant: Some(occupant),
            }));
            (Entity::new(id, 0), EntityAllocation::NewSlot)
        };
        Ok(SpawnedEntity { entity, allocation })
    }

    /// Get the name of an entity.
    pub fn name(&self, entity: impl Into<Entity>) -> Option<&str> {
        let entity: Entity = entity.into();
        self.validate_generation(entity)?;
        let idx = (entity.0 - 1) as usize;
        self.entity_meta
            .get(idx)
            .and_then(|m| m.as_ref()?.occupant.as_ref())
            .and_then(|o| o.name.as_deref())
    }

    /// Find all entities of a given kind.
    pub fn entities_of_kind(&self, kind: EntityKind) -> Vec<Entity> {
        self.entity_meta
            .iter()
            .enumerate()
            .filter(|(_, slot)| {
                slot.as_ref()
                    .and_then(|s| s.occupant.as_ref())
                    .is_some_and(|o| o.kind == kind)
            })
            .map(|(i, slot)| {
                let gen = slot.as_ref().map(|s| s.generation).unwrap_or(0);
                Entity::new((i + 1) as u64, gen)
            })
            .collect()
    }

    /// Get the kind of an entity (alias for entity_kind).
    pub fn kind(&self, entity: impl Into<Entity>) -> Option<EntityKind> {
        let entity: Entity = entity.into();
        self.validate_generation(entity)?;
        self.entity_kind(entity)
    }

    // ── Component access ──────────────────────────────────────────────────────

    pub fn remove_component<T: Component>(
        &mut self,
        entity: impl Into<Entity>,
    ) -> Result<Option<T>, WorldError> {
        if !self.mutation_policy.direct_mutations_allowed() {
            return Err(WorldError::DirectMutationDisallowed {
                operation: "remove_component",
            });
        }
        let entity: Entity = entity.into();
        if self.validate_generation(entity).is_none() {
            return Ok(None);
        }
        Ok(self.component_store.remove::<T>(entity))
    }

    pub fn add_component<T: Component>(
        &mut self,
        entity: impl Into<Entity>,
        component: T,
    ) -> Result<(), WorldError> {
        if !self.mutation_policy.direct_mutations_allowed() {
            return Err(WorldError::DirectMutationDisallowed {
                operation: "add_component",
            });
        }
        let entity: Entity = entity.into();
        if self.validate_generation(entity).is_none() {
            return Err(WorldError::StaleHandle { entity });
        }
        self.component_store.insert::<T>(entity, component);
        Ok(())
    }

    /// Stage a component insert to be applied at [`commit_stage`]. Returns
    /// `WorldError::StaleHandle` if the entity handle is not valid. Constitutional
    /// code does not panic on a stale handle; the caller decides whether to retry,
    /// surface the error, or drop the staged operation.
    pub fn stage_component<T: Component>(
        &mut self,
        entity: impl Into<Entity>,
        component: T,
    ) -> Result<(), WorldError> {
        let entity: Entity = entity.into();
        self.validate_generation(entity)
            .ok_or(WorldError::StaleHandle { entity })?;
        self.staging
            .push(Box::new(move |store: &mut ComponentStore| {
                store.insert::<T>(entity, component);
            }));
        Ok(())
    }

    pub fn commit_stage(&mut self) {
        let staging = std::mem::take(&mut self.staging);
        for op in staging {
            op(&mut self.component_store);
        }
    }

    /// Discard deferred component insert operations added via [`stage_component`].
    pub fn rollback_stage(&mut self) {
        self.staging.clear();
    }

    pub fn get_component<T: Component>(&self, entity: impl Into<Entity>) -> Option<&T> {
        let entity: Entity = entity.into();
        self.validate_generation(entity)?;
        self.component_store.get::<T>(entity)
    }

    pub fn get_component_mut<T: Component>(&mut self, entity: impl Into<Entity>) -> Option<&mut T> {
        let e: Entity = entity.into();
        self.validate_generation(e)?;
        self.component_store.column_mut::<T>().get_mut(e)
    }

    /// Canonical: insert or replace a component on an entity.
    /// Returns error if the entity is stale or dead.
    pub fn insert_component<T: Component>(
        &mut self,
        entity: impl Into<Entity>,
        component: T,
    ) -> Result<(), WorldError> {
        let entity: Entity = entity.into();
        if !self.is_alive(entity) {
            return Err(WorldError::StaleHandle { entity });
        }
        if !self.mutation_policy.direct_mutations_allowed() {
            return Err(WorldError::DirectMutationDisallowed {
                operation: "insert_component",
            });
        }
        self.component_store.insert::<T>(entity, component);
        Ok(())
    }

    /// Canonical: read a component from an entity.
    pub fn component<T: Component>(&self, entity: impl Into<Entity>) -> Result<&T, WorldError> {
        let entity: Entity = entity.into();
        if !self.is_alive(entity) {
            return Err(WorldError::StaleHandle { entity });
        }
        self.component_store
            .get::<T>(entity)
            .ok_or(WorldError::MissingComponent {
                entity,
                type_name: std::any::type_name::<T>(),
            })
    }

    /// Canonical: mutable read of a component.
    pub fn component_mut<T: Component>(
        &mut self,
        entity: impl Into<Entity>,
    ) -> Result<&mut T, WorldError> {
        let e: Entity = entity.into();
        if !self.is_alive(e) {
            return Err(WorldError::StaleHandle { entity: e });
        }
        self.component_store
            .column_mut::<T>()
            .get_mut(e)
            .ok_or(WorldError::MissingComponent {
                entity: e,
                type_name: std::any::type_name::<T>(),
            })
    }

    /// Canonical: check if an entity has a component.
    pub fn has_component<T: Component>(&self, entity: impl Into<Entity>) -> bool {
        let entity: Entity = entity.into();
        self.is_alive(entity) && self.component_store.contains::<T>(entity)
    }

    // ── Resources ─────────────────────────────────────────────────────────────

    pub fn add_resource<T: 'static + Send + Sync>(&mut self, resource: T) {
        self.resource_store.insert::<T>(resource);
    }

    pub fn get_resource<T: 'static + Send + Sync>(&self) -> Option<&T> {
        self.resource_store.get::<T>()
    }

    pub fn get_resource_mut<T: 'static + Send + Sync>(&mut self) -> Option<&mut T> {
        self.resource_store.get_mut::<T>()
    }

    /// Typed resource: insert a resource, returning error if already exists.
    pub fn insert_resource<T: 'static + Send + Sync>(
        &mut self,
        resource: T,
    ) -> Result<(), WorldError> {
        if self.resource_store.contains::<T>() {
            return Err(WorldError::DuplicateResource {
                type_name: std::any::type_name::<T>(),
            });
        }
        self.resource_store.insert::<T>(resource);
        Ok(())
    }

    /// Typed resource: get a guarded shared reference.
    pub fn resource<T: 'static + Send + Sync>(&self) -> Result<ResourceRef<'_, T>, WorldError> {
        self.resource_store
            .get::<T>()
            .map(ResourceRef::new)
            .ok_or(WorldError::MissingResource {
                type_name: std::any::type_name::<T>(),
            })
    }

    /// Typed resource: get a guarded mutable reference.
    pub fn resource_mut<T: 'static + Send + Sync>(
        &mut self,
    ) -> Result<ResourceMut<'_, T>, WorldError> {
        self.resource_store
            .get_mut::<T>()
            .map(ResourceMut::new)
            .ok_or(WorldError::MissingResource {
                type_name: std::any::type_name::<T>(),
            })
    }

    /// Check if a resource type exists.
    pub fn has_resource<T: 'static + Send + Sync>(&self) -> bool {
        self.resource_store.contains::<T>()
    }

    /// Remove a resource, returning it.
    pub fn remove_resource<T: 'static + Send + Sync>(&mut self) -> Result<T, WorldError> {
        self.resource_store
            .remove::<T>()
            .ok_or(WorldError::MissingResource {
                type_name: std::any::type_name::<T>(),
            })
    }

    // ── Entity management ─────────────────────────────────────────────────────

    /// Returns the next entity ID that will be assigned, without consuming it.
    pub fn next_entity_id(&self) -> u64 {
        self.next_id
    }

    /// Set the mutation access policy.
    /// Use `MutationPolicy::Bootstrap` for initial world construction,
    /// then transition to `MutationPolicy::TransactionalOnly` before
    /// exposing the world to concurrent consumers.
    pub fn set_direct_mutation_allowed(&mut self, allowed: bool) {
        self.mutation_policy = if allowed {
            MutationPolicy::Bootstrap
        } else {
            MutationPolicy::TransactionalOnly
        };
    }

    /// Set the mutation policy directly.
    pub fn set_mutation_policy(&mut self, policy: MutationPolicy) {
        self.mutation_policy = policy;
    }

    /// Get the current mutation policy.
    pub fn mutation_policy(&self) -> MutationPolicy {
        self.mutation_policy
    }

    /// Check if direct mutation is currently allowed (convenience).
    pub fn is_direct_mutation_allowed(&self) -> bool {
        self.mutation_policy.direct_mutations_allowed()
    }

    /// Spawn an entity at a specific reserved ID (used by WorldTxn during commit).
    ///
    /// Idempotent: if the entity slot already exists at this ID, the call is
    /// a no-op. This allows phase 1aa (reservation) and phase 3c (apply) to
    /// both call it without double-pushing entity metadata.
    pub fn spawn_entity_with_id(&mut self, id: u64, kind: EntityKind) -> Entity {
        let idx = (id - 1) as usize;
        if idx < self.entity_meta.len()
            && self.entity_meta[idx]
                .as_ref()
                .and_then(|s| s.occupant.as_ref())
                .is_some()
        {
            // Slot already occupied — no-op (idempotent).
            let gen = self.entity_meta[idx]
                .as_ref()
                .map(|s| s.generation)
                .unwrap_or(0);
            return Entity::new(id, gen);
        }
        // Grow the metadata vector to fit the requested id.
        while self.entity_meta.len() < idx + 1 {
            self.entity_meta.push(None);
        }
        let gen = 0;
        self.entity_meta[idx] = Some(EntitySlot {
            generation: gen,
            occupant: Some(Occupant { kind, name: None }),
        });
        // Advance the auto-allocator so next_entity_id() returns valid values.
        if id >= self.next_id {
            self.next_id = id + 1;
        }
        Entity::new(id, gen)
    }

    pub fn spawn_entity(&mut self, kind: EntityKind) -> Entity {
        assert!(
            self.mutation_policy.direct_mutations_allowed(),
            "direct spawn_entity() called outside WorldTxn — use WorldTxn::stage_spawn()"
        );
        let (id, generation) = if let Some(free) = self.free_list.pop() {
            let idx = (free - 1) as usize;
            if idx < self.entity_meta.len() {
                if let Some(Some(slot)) = self.entity_meta.get_mut(idx) {
                    let gen = slot.generation + 1;
                    slot.occupant = Some(Occupant { kind, name: None });
                    slot.generation = gen;
                    (free, gen)
                } else {
                    self.entity_meta[idx] = Some(EntitySlot {
                        generation: 0,
                        occupant: Some(Occupant { kind, name: None }),
                    });
                    (free, 0)
                }
            } else {
                self.entity_meta.push(Some(EntitySlot {
                    generation: 0,
                    occupant: Some(Occupant { kind, name: None }),
                }));
                (free, 0)
            }
        } else {
            let id = self.next_id;
            self.next_id += 1;
            self.entity_meta.push(Some(EntitySlot {
                generation: 0,
                occupant: Some(Occupant { kind, name: None }),
            }));
            (id, 0)
        };
        Entity::new(id, generation)
    }

    pub fn entity_kind(&self, entity: impl Into<Entity>) -> Option<EntityKind> {
        let entity: Entity = entity.into();
        self.validate_generation(entity)?;
        let idx = (entity.0 - 1) as usize;
        self.entity_meta
            .get(idx)
            .and_then(|slot| slot.as_ref()?.occupant.as_ref().map(|o| o.kind))
    }

    /// Check whether an entity handle is still valid (slot occupied and generation matches).
    pub fn is_alive(&self, entity: impl Into<Entity>) -> bool {
        let entity: Entity = entity.into();
        self.validate_generation(entity).is_some()
    }

    /// Check an entity's generation, returning (alive, generation).
    fn check_generation(&self, entity: Entity) -> (bool, u32) {
        if entity.0 == 0 {
            return (false, 0);
        }
        let idx = (entity.0 - 1) as usize;
        match self.entity_meta.get(idx).and_then(|s| s.as_ref()) {
            Some(slot) if slot.occupant.is_some() && slot.generation == entity.1 => {
                (true, slot.generation)
            }
            Some(slot) => (false, slot.generation),
            None => (false, 0),
        }
    }

    /// Create a generation-safe entity reference for component access.
    pub fn entity_ref(&self, entity: Entity) -> Result<EntityRef<'_>, WorldError> {
        let (alive, gen) = self.check_generation(entity);
        if !alive {
            return Err(WorldError::StaleHandle { entity });
        }
        Ok(EntityRef {
            entity,
            generation: gen,
            world: self,
        })
    }

    /// Despawn an entity: advance generation and release the slot for reuse.
    ///
    /// Returns `Ok(true)` if the entity was despawned by this call, `Ok(false)` if
    /// the entity was already dead (idempotent), and `Err(WorldError::StaleHandle)`
    /// if the handle refers to an entity that has never existed or whose generation
    /// does not match. Constitutional code does not panic on a stale handle.
    pub fn despawn(&mut self, entity: impl Into<Entity>) -> Result<bool, WorldError> {
        let entity: Entity = entity.into();
        if !self.is_alive(entity) {
            return Ok(false);
        }
        // Live but generation-mismatched = caller has a stale handle. The handle
        // resolves to a slot but the slot's generation is past the handle's
        // generation; the entity has been despawned and possibly re-spawned.
        self.validate_generation(entity)
            .ok_or(WorldError::StaleHandle { entity })?;
        let idx = (entity.0 - 1) as usize;
        if let Some(Some(slot)) = self.entity_meta.get_mut(idx) {
            slot.generation += 1;
            slot.occupant = None;
        }
        self.free_list.push(entity.0);
        Ok(true)
    }

    // ── Queries ───────────────────────────────────────────────────────────────

    /// Iterate over all entities that have component type A.
    pub fn query<'w, A: Component>(&'w self) -> Query<'w, A> {
        let col: Option<&Column<A>> = self.component_store.column::<A>();
        Query { col, cursor: 0 }
    }

    /// Iterate mutably over all entities that have component type A.
    pub fn query_mut<'w, A: Component>(&'w mut self) -> QueryMut<'w, A> {
        let col: Option<&mut Column<A>> = Some(self.component_store.column_mut::<A>());
        QueryMut { col, cursor: 0 }
    }

    /// Iterate over all entities that have BOTH component A and component B.
    pub fn query2<'w, A: Component, B: Component>(&'w self) -> Query2<'w, A, B> {
        let col_a: Option<&Column<A>> = self.component_store.column::<A>();
        let col_b: Option<&Column<B>> = self.component_store.column::<B>();
        Query2 {
            col_a,
            col_b,
            cursor: 0,
        }
    }

    /// Iterate over all entities that have ALL of components A, B, and C.
    pub fn query3<'w, A: Component, B: Component, C: Component>(&'w self) -> Query3<'w, A, B, C> {
        let col_a: Option<&Column<A>> = self.component_store.column::<A>();
        let col_b: Option<&Column<B>> = self.component_store.column::<B>();
        let col_c: Option<&Column<C>> = self.component_store.column::<C>();
        Query3 {
            col_a,
            col_b,
            col_c,
            cursor: 0,
        }
    }

    // ── Introspection ─────────────────────────────────────────────────────────

    pub fn entity_count(&self) -> usize {
        self.entity_meta
            .iter()
            .filter(|s| s.as_ref().and_then(|s| s.occupant.as_ref()).is_some())
            .count()
    }

    /// Return all alive entities in the world.
    pub fn all_entities(&self) -> Vec<Entity> {
        self.entity_meta
            .iter()
            .enumerate()
            .filter_map(|(i, slot)| {
                let slot = slot.as_ref()?;
                if slot.occupant.is_some() {
                    Some(Entity::new((i + 1) as u64, slot.generation))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Check whether an entity exists in the world.
    pub fn has_entity(&self, entity: impl Into<Entity>) -> bool {
        let entity: Entity = entity.into();
        self.is_alive(entity)
    }

    /// Read the component version (write count) for a given entity.
    /// Returns 0 if no writes have occurred for this entity.
    pub fn component_version(&self, entity: Entity) -> u64 {
        self.component_versions
            .get(&entity.id())
            .copied()
            .unwrap_or(0)
    }

    /// Expose component_store for preflight closure evaluation during prepare().
    pub fn component_store_ref(&self) -> &ComponentStore {
        &self.component_store
    }

    /// Get the current world epoch (from extensions, defaults to epoch 0).
    pub fn current_epoch(&self) -> WorldEpoch {
        self.get_extension::<WorldEpoch>()
            .copied()
            .unwrap_or(WorldEpoch(0))
    }

    /// Set the current world epoch (stored as an extension).
    pub fn set_epoch(&mut self, epoch: WorldEpoch) {
        self.set_extension(epoch);
    }
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}
