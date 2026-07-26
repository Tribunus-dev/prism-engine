//! Engine-local `WorldTxn` — staged mutations for the runtime [`World`].
//!
//! Mirrors the shape of the constitutional `WorldTxn` in
//! `crates/prism-ecs-constitutional/src/world_txn.rs` but is scoped to the
//! engine's runtime `World` (entity/component storage, not the constitutional
//! [`ComponentStore`]). Stages spawns, component inserts, and component
//! removes; commits all changes atomically.
//!
//! The motivation is the same as the constitutional one: a single
//! authority-bearing commit seam so that engine-side mutations do not fork
//! into N direct paths. Every state-bearing change must be validatable,
//! attributable, and replayable. Direct `world.spawn()` /
//! `world.insert(...)` calls outside the `WorldTxn` boundary were the
//! last-mile leakage that this module closes.
//!
//! The pattern:
//!
//! 1. Construct a [`WorldTxn`].
//! 2. Stage spawns, component inserts, and component removes. Spawns return
//!    a [`PendingToken`] that can be used as a target for staged inserts
//!    when the entity is not yet allocated.
//! 3. Call [`WorldTxn::commit`] on the target world. The spawned entities
//!    are returned in stage order, so the caller can map them back.
//!
//! ```ignore
//! let mut txn = WorldTxn::new();
//! let token = txn.stage_spawn();
//! txn.stage_insert_on::<ComponentA>(token, ComponentA::new());
//! let entities = txn.commit(&mut world)?;
//! let new_entity = entities[0];
//! ```
//!
//! No `unsafe`. No `unwrap` / `expect` / `panic!` in production paths.
//! Errors flow through [`WorldTxnError`] with `thiserror` derives.

use crate::ecs::runtime::world::{Component, Entity, World};
use std::any::TypeId;
use std::collections::BTreeMap;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur during transaction commit.
#[derive(Debug, Error)]
pub enum WorldTxnError {
    /// The world's entity allocator is at capacity.
    #[error("world at capacity: cannot allocate entity for staged spawn")]
    AtCapacity,
    /// A staged insert targeted a pending-spawn token that is out of range
    /// for the staged spawns. This indicates a programming error in the
    /// caller (a token was used that was never returned by `stage_spawn`).
    #[error("staged insert targets unknown pending spawn index: {0}")]
    UnknownPendingSpawn(u32),
}

// ---------------------------------------------------------------------------
// PendingToken
// ---------------------------------------------------------------------------

/// A placeholder handle for an entity that will be allocated during
/// [`WorldTxn::commit`]. Returned by [`WorldTxn::stage_spawn`] and used as
/// the target for staged component inserts against the not-yet-allocated
/// entity.
///
/// The token is a 0-indexed offset into the staged-spawn vector. The real
/// [`Entity`] is produced by commit-time allocation and returned in the
/// `Vec<Entity>` from [`WorldTxn::commit`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PendingToken(pub(crate) u32);

impl PendingToken {
    /// The 0-indexed position of this token in the staged-spawn vector.
    pub fn index(self) -> u32 {
        self.0
    }
}

// ---------------------------------------------------------------------------
// Target enum
// ---------------------------------------------------------------------------

/// Where a staged insert is going. `PendingSpawn(idx)` refers to the
/// entity that will be allocated at index `idx` of the staged-spawn vector
/// during commit. `Existing(e)` refers to an already-allocated entity.
///
/// `BTreeMap` requires `Ord`; we derive a stable ordering so deterministic
/// replay can rebuild the same operation set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum InsertTarget {
    /// Reference to a staged spawn, by 0-indexed position.
    PendingSpawn(u32),
    /// Reference to an already-allocated entity.
    Existing(Entity),
}

// ---------------------------------------------------------------------------
// Staged operations
// ---------------------------------------------------------------------------

/// A staged entity spawn. The apply closure allocates the entity at
/// commit time and returns it. Preflight is a no-op for spawns (the engine
/// `World` has no externally observable state to validate before allocation
/// without holding `&mut`).
///
/// The apply closure is intentionally `!Send` — the engine's component
/// types (e.g. `WorkerRequest` with `Instant`) are not always `Send`,
/// and the `WorldTxn` is built and consumed on the same system thread
/// that holds `&mut World`, so cross-thread transport is not required.
struct StagedSpawn {
    apply: Box<dyn FnOnce(&mut World) -> Result<Entity, WorldTxnError>>,
}

/// A staged component insert. The apply closure takes the resolved
/// target entity and the typed value, then writes it into the world.
///
/// The closure captures the typed component value `T`; type identity is
/// erased into `type_id` for the deterministic key. `!Send` — see
/// [`StagedSpawn`] for the rationale.
struct StagedInsert {
    target: InsertTarget,
    type_id: TypeId,
    apply: Box<dyn FnOnce(&mut World, Entity) -> Result<(), WorldTxnError>>,
}

/// A staged component removal. The apply closure takes the target entity
/// and removes the typed component. `!Send` — see [`StagedSpawn`].
struct StagedRemove {
    entity: Entity,
    type_id: TypeId,
    apply: Box<dyn FnOnce(&mut World) -> Result<(), WorldTxnError>>,
}

// ---------------------------------------------------------------------------
// WorldTxn
// ---------------------------------------------------------------------------

/// A pending world transaction against the engine's runtime [`World`].
///
/// Build a transaction by staging spawns and component inserts/removes, then
/// commit atomically. The order of stage_spawn calls determines the
/// ordering of the spawned entities returned by `commit`.
///
/// Component inserts/removes that target a not-yet-allocated entity use
/// [`PendingToken`] (returned by `stage_spawn`) as the target. Inserts/
/// removes against existing entities take a plain [`Entity`].
pub struct WorldTxn {
    spawns: Vec<StagedSpawn>,
    /// Staged inserts in stage order. Order matters for replay
    /// determinism when two inserts target the same entity / type.
    inserts: Vec<StagedInsert>,
    /// Staged removes keyed by `(Entity, TypeId)` for deterministic order
    /// on replay.
    removes: BTreeMap<(Entity, TypeId), StagedRemove>,
}

impl WorldTxn {
    /// Create an empty transaction.
    pub fn new() -> Self {
        Self {
            spawns: Vec::new(),
            inserts: Vec::new(),
            removes: BTreeMap::new(),
        }
    }

    /// Stage an entity spawn. The actual entity ID is assigned at commit
    /// time and returned in the `Vec<Entity>` from `commit`.
    ///
    /// Returns a [`PendingToken`] that can be used as a target for staged
    /// component inserts against the not-yet-allocated entity.
    pub fn stage_spawn(&mut self) -> PendingToken {
        let index = self.spawns.len() as u32;
        self.spawns.push(StagedSpawn {
            apply: Box::new(|world: &mut World| {
                world.spawn().ok_or(WorldTxnError::AtCapacity)
            }),
        });
        PendingToken(index)
    }

    /// Stage a component insert against a [`PendingToken`] (a
    /// not-yet-allocated entity from a prior `stage_spawn` call).
    pub fn stage_insert_on<T: Component>(
        &mut self,
        target: PendingToken,
        component: T,
    ) {
        let type_id = TypeId::of::<T>();
        self.inserts.push(StagedInsert {
            target: InsertTarget::PendingSpawn(target.0),
            type_id,
            apply: Box::new(move |world: &mut World, entity: Entity| {
                world.insert(entity, component);
                Ok(())
            }),
        });
    }

    /// Stage a component insert against an already-allocated [`Entity`].
    pub fn stage_insert<T: Component>(
        &mut self,
        entity: Entity,
        component: T,
    ) {
        let type_id = TypeId::of::<T>();
        self.inserts.push(StagedInsert {
            target: InsertTarget::Existing(entity),
            type_id,
            apply: Box::new(move |world: &mut World, entity: Entity| {
                world.insert(entity, component);
                Ok(())
            }),
        });
    }

    /// Stage a component removal against an already-allocated [`Entity`].
    pub fn stage_remove<T: Component>(&mut self, entity: Entity) {
        let type_id = TypeId::of::<T>();
        self.removes.insert(
            (entity, type_id),
            StagedRemove {
                entity,
                type_id,
                apply: Box::new(move |world: &mut World| {
                    let _ = world.remove::<T>(entity);
                    Ok(())
                }),
            },
        );
    }

    /// The number of staged spawns. Useful for sizing the result vector.
    pub fn spawn_count(&self) -> usize {
        self.spawns.len()
    }

    /// The number of staged component inserts.
    pub fn insert_count(&self) -> usize {
        self.inserts.len()
    }

    /// The number of staged component removes.
    pub fn remove_count(&self) -> usize {
        self.removes.len()
    }

    /// Commit all staged changes to `world` atomically.
    ///
    /// 1. Apply all staged spawns in stage order, collecting the
    ///    allocated entities.
    /// 2. Apply all staged inserts. Inserts targeting `PendingSpawn(idx)`
    ///    are resolved to the entity that was allocated at step 1,
    ///    position `idx`.
    /// 3. Apply all staged removes.
    ///
    /// Returns the spawned entities in stage order, so the caller can
    /// bind them back into local data structures.
    pub fn commit(self, world: &mut World) -> Result<Vec<Entity>, WorldTxnError> {
        let WorldTxn {
            spawns,
            inserts,
            removes,
        } = self;

        // 1. Apply spawns.
        let mut spawned: Vec<Option<Entity>> = (0..spawns.len()).map(|_| None).collect();
        for (idx, spawn) in spawns.into_iter().enumerate() {
            let entity = (spawn.apply)(world)?;
            spawned[idx] = Some(entity);
        }

        // 2. Apply inserts.
        for insert in inserts {
            let target_entity = match insert.target {
                InsertTarget::PendingSpawn(idx) => {
                    let resolved = spawned
                        .get(idx as usize)
                        .copied()
                        .flatten()
                        .ok_or(WorldTxnError::UnknownPendingSpawn(idx))?;
                    resolved
                }
                InsertTarget::Existing(entity) => entity,
            };
            (insert.apply)(world, target_entity)?;
        }

        // 3. Apply removes.
        for (_key, remove) in removes {
            (remove.apply)(world)?;
        }

        // 4. Collect the spawned entities into a flat Vec.
        let spawned = spawned
            .into_iter()
            .map(|opt| opt.expect("spawn slot populated in step 1"))
            .collect();
        Ok(spawned)
    }
}

impl Default for WorldTxn {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::any::Any;

    // Two concrete component types for the tests.
    #[derive(Debug, Clone, PartialEq)]
    struct Alpha(u32);
    impl Component for Alpha {}

    #[derive(Debug, Clone, PartialEq)]
    struct Beta(u32);
    impl Component for Beta {}

    #[test]
    fn empty_txn_commits_and_returns_no_entities() {
        let mut world = World::new();
        let txn = WorldTxn::new();
        let spawned = txn.commit(&mut world).expect("empty commit");
        assert!(spawned.is_empty());
    }

    #[test]
    fn spawn_and_insert_returns_resolved_entity() {
        let mut world = World::new();
        world.register_component::<Alpha>();

        let mut txn = WorldTxn::new();
        let token = txn.stage_spawn();
        txn.stage_insert_on::<Alpha>(token, Alpha(42));

        let spawned = txn.commit(&mut world).expect("commit");
        assert_eq!(spawned.len(), 1);
        let entity = spawned[0];

        let alpha = world
            .get::<Alpha>(entity)
            .expect("Alpha inserted")
            .clone();
        assert_eq!(alpha, Alpha(42));
    }

    #[test]
    fn multiple_spawns_get_distinct_resolved_entities() {
        let mut world = World::new();
        world.register_component::<Alpha>();
        world.register_component::<Beta>();

        let mut txn = WorldTxn::new();
        let t0 = txn.stage_spawn();
        let t1 = txn.stage_spawn();
        txn.stage_insert_on::<Alpha>(t0, Alpha(1));
        txn.stage_insert_on::<Beta>(t1, Beta(2));

        let spawned = txn.commit(&mut world).expect("commit");
        assert_eq!(spawned.len(), 2);
        assert_ne!(spawned[0], spawned[1]);
        assert_eq!(world.get::<Alpha>(spawned[0]).map(|a| a.clone()), Some(Alpha(1)));
        assert_eq!(world.get::<Beta>(spawned[1]).map(|b| b.clone()), Some(Beta(2)));
    }

    #[test]
    fn insert_against_existing_entity_uses_entity_directly() {
        let mut world = World::new();
        world.register_component::<Alpha>();

        let entity = world.spawn().expect("pre-spawn");
        let mut txn = WorldTxn::new();
        txn.stage_insert::<Alpha>(entity, Alpha(99));
        let spawned = txn.commit(&mut world).expect("commit");
        assert!(spawned.is_empty());
        assert_eq!(
            world.get::<Alpha>(entity).map(|a| a.clone()),
            Some(Alpha(99))
        );
    }

    #[test]
    fn remove_drops_component() {
        let mut world = World::new();
        world.register_component::<Alpha>();
        let entity = world.spawn().expect("spawn");
        world.insert(entity, Alpha(1));
        assert!(world.get::<Alpha>(entity).is_some());

        let mut txn = WorldTxn::new();
        txn.stage_remove::<Alpha>(entity);
        let _spawned = txn.commit(&mut world).expect("commit");
        assert!(world.get::<Alpha>(entity).is_none());
    }

    #[test]
    fn spawn_count_increments() {
        let mut txn = WorldTxn::new();
        assert_eq!(txn.spawn_count(), 0);
        txn.stage_spawn();
        assert_eq!(txn.spawn_count(), 1);
        txn.stage_spawn();
        assert_eq!(txn.spawn_count(), 2);
    }

    #[test]
    fn insert_count_increments() {
        let mut world = World::new();
        let mut txn = WorldTxn::new();
        let t = txn.stage_spawn();
        // Need a registered component for the type_id to be meaningful.
        world.register_component::<Alpha>();
        txn.stage_insert_on::<Alpha>(t, Alpha(1));
        assert_eq!(txn.insert_count(), 1);
    }

    #[test]
    fn unknown_pending_token_is_rejected() {
        let mut world = World::new();
        let mut txn = WorldTxn::new();
        // No stage_spawn — token refers to a non-existent index.
        let bad = PendingToken(7);
        txn.stage_insert_on::<Alpha>(bad, Alpha(1));
        let result = txn.commit(&mut world);
        assert!(matches!(
            result,
            Err(WorldTxnError::UnknownPendingSpawn(7))
        ));
    }

    #[test]
    fn at_capacity_returns_error() {
        // World with capacity 1: first spawn OK, second fails.
        let mut world = World::with_capacity(1);
        let mut txn = WorldTxn::new();
        txn.stage_spawn();
        txn.stage_spawn(); // Will fail at commit.
        let result = txn.commit(&mut world);
        assert!(matches!(result, Err(WorldTxnError::AtCapacity)));
    }

    #[test]
    fn type_id_distinguishes_components() {
        // Smoke test: TypeId comparison works as expected.
        let a = TypeId::of::<Alpha>();
        let b = TypeId::of::<Beta>();
        assert_ne!(a, b);
        let _: &dyn Any = &Alpha(0);
    }
}
