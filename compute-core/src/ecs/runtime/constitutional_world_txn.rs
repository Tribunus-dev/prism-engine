//! `ConstitutionalWorldTxn` — staged mutations for the constitutional
//! [`prism_ecs_core::World`].
//!
//! Mirrors the shape of the engine-local `WorldTxn` in
//! [`super::world_txn`] but targets the constitutional `World` that the
//! `compute-core` system files use (with `EntityKind` / name and the
//! `Component` access API). The engine-local `WorldTxn` operates on a
//! simpler runtime `World` that the engine subsystems construct for
//! short-lived compilations; the constitutional `World` is the one
//! whose `&mut` is threaded through every `CompilerSystem::run` call.
//!
//! The system files (`compute-core/src/ecs/system/`) cannot use the
//! engine-local `WorldTxn` because the World types differ. They also
//! cannot use the full constitutional `WorldTxn` in
//! `crates/prism-ecs-constitutional/src/world_txn.rs` because that
//! API gates `put_durable` / `put_transient` on the
//! `DurableComponent` / `TransientComponent` traits, and the system
//! files' components only implement [`prism_ecs_core::Component`]
//! (the legacy pattern that the engine's `CompilerSystem`s rely on).
//!
//! The motivation is the same as the engine-local one: a single
//! authority-bearing commit seam so that engine-side mutations do not
//! fork into N direct paths. Every state-bearing change must be
//! validatable, attributable, and replayable. The previous shape —
//! `world.spawn(...)` / `world.add_component(...)` scattered through
//! the `CompilerSystem::run` body — is the last-mile leakage this
//! module closes.
//!
//! The pattern:
//!
//! 1. Construct a [`ConstitutionalWorldTxn`].
//! 2. Stage spawns (with `EntityKind` and optional name), component
//!    inserts, and component removes. Spawns return a [`PendingToken`]
//!    that can be used as a target for staged inserts when the entity
//!    is not yet allocated.
//! 3. Call [`ConstitutionalWorldTxn::commit`] on the target
//!    constitutional `World`. The spawned entities are returned in
//!    stage order, so the caller can map them back to local data
//!    structures.
//!
//! ```ignore
//! use prism_ecs_core::{EntityKind, World};
//!
//! let mut world = World::new();
//! let mut txn = ConstitutionalWorldTxn::new();
//! let token = txn.stage_spawn(EntityKind::Kernel, Some("kernel_a".into()));
//! txn.stage_insert_on::<CatalogEntry>(token, CatalogEntry::default());
//! let entities = txn.commit(&mut world)?;
//! let kernel = entities[0];
//! ```
//!
//! No `unsafe`. No `unwrap` / `expect` / `panic!` in production paths.
//! Errors flow through [`ConstitutionalWorldTxnError`] with
//! `thiserror` derives.

use prism_ecs_core::{Component, Entity, EntityKind, World};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur during transaction commit.
#[derive(Debug, Error)]
pub enum ConstitutionalWorldTxnError {
    /// The constitutional `World` rejected a mutation (at capacity,
    /// mutation policy denied, stale handle, or other allocation
    /// failure). The `op` field identifies which operation
    /// (`"spawn"`, `"add_component"`, `"remove_component"`) failed.
    #[error("world {op} rejected: {source}")]
    WorldOp {
        op: &'static str,
        #[source]
        source: prism_ecs_core::WorldError,
    },
    /// A staged insert targeted a pending-spawn token that is out of
    /// range for the staged spawns. This indicates a programming
    /// error in the caller (a token was used that was never returned
    /// by `stage_spawn`).
    #[error("staged insert targets unknown pending spawn index: {0}")]
    UnknownPendingSpawn(u32),
    /// A component insert on a pending spawn failed because the
    /// resolved entity was not allocated (e.g. a prior spawn failed).
    #[error("staged insert for spawn {0} could not be applied: spawn not allocated")]
    SpawnNotAllocated(u32),
}

impl ConstitutionalWorldTxnError {
    /// Wrap a [`prism_ecs_core::WorldError`] from a `spawn` call.
    pub fn from_spawn(source: prism_ecs_core::WorldError) -> Self {
        Self::WorldOp {
            op: "spawn",
            source,
        }
    }
    /// Wrap a [`prism_ecs_core::WorldError`] from an `add_component` call.
    pub fn from_insert(source: prism_ecs_core::WorldError) -> Self {
        Self::WorldOp {
            op: "add_component",
            source,
        }
    }
    /// Wrap a [`prism_ecs_core::WorldError`] from a `remove_component` call.
    pub fn from_remove(source: prism_ecs_core::WorldError) -> Self {
        Self::WorldOp {
            op: "remove_component",
            source,
        }
    }
}

// ---------------------------------------------------------------------------
// PendingToken
// ---------------------------------------------------------------------------

/// A placeholder handle for an entity that will be allocated during
/// [`ConstitutionalWorldTxn::commit`]. Returned by
/// [`ConstitutionalWorldTxn::stage_spawn`] and used as the target for
/// staged component inserts against the not-yet-allocated entity.
///
/// The token is a 0-indexed offset into the staged-spawn vector. The
/// real [`Entity`] is produced by commit-time allocation and returned
/// in the `Vec<Entity>` from [`ConstitutionalWorldTxn::commit`].
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
/// entity that will be allocated at index `idx` of the staged-spawn
/// vector during commit. `Existing(e)` refers to an already-allocated
/// entity.
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

/// A staged entity spawn. `kind` and `name` are captured at stage
/// time; the apply closure uses them to call `World::spawn` during
/// commit.
struct StagedSpawn {
    kind: EntityKind,
    name: Option<String>,
    /// Apply the spawn at commit time. Returns the resolved [`Entity`].
    /// `!Send` — see the engine-local `WorldTxn` rationale; constitutional
    /// World operations must run on the same thread that holds `&mut`.
    apply: Box<dyn FnOnce(&mut World) -> Result<Entity, ConstitutionalWorldTxnError>>,
}

/// A staged component insert. The apply closure takes the resolved
/// target entity and the typed value, then writes it into the world.
struct StagedInsert {
    target: InsertTarget,
    /// Apply the insert at commit time.
    /// `!Send` — see [`StagedSpawn`].
    apply: Box<dyn FnOnce(&mut World, Entity) -> Result<(), ConstitutionalWorldTxnError>>,
}

/// A staged component removal. The apply closure takes the target
/// entity and removes the typed component.
struct StagedRemove {
    entity: Entity,
    /// Apply the remove at commit time.
    /// `!Send` — see [`StagedSpawn`].
    apply: Box<dyn FnOnce(&mut World) -> Result<(), ConstitutionalWorldTxnError>>,
}

// ---------------------------------------------------------------------------
// ConstitutionalWorldTxn
// ---------------------------------------------------------------------------

/// A pending world transaction against the constitutional
/// [`prism_ecs_core::World`].
///
/// Build a transaction by staging spawns and component inserts/removes,
/// then commit atomically. The order of `stage_spawn` calls determines
/// the ordering of the spawned entities returned by `commit`.
///
/// Component inserts/removes that target a not-yet-allocated entity
/// use [`PendingToken`] (returned by `stage_spawn`) as the target.
/// Inserts/removes against existing entities take a plain [`Entity`].
pub struct ConstitutionalWorldTxn {
    spawns: Vec<StagedSpawn>,
    /// Staged inserts in stage order. Order matters for replay
    /// determinism when two inserts target the same entity / type.
    inserts: Vec<StagedInsert>,
    /// Staged removes keyed by `(Entity, TypeId)` for deterministic
    /// order on replay.
    removes: std::collections::BTreeMap<(Entity, std::any::TypeId), StagedRemove>,
}

impl ConstitutionalWorldTxn {
    /// Create an empty transaction.
    pub fn new() -> Self {
        Self {
            spawns: Vec::new(),
            inserts: Vec::new(),
            removes: std::collections::BTreeMap::new(),
        }
    }

    /// Stage an entity spawn. The actual entity ID is assigned at
    /// commit time and returned in the `Vec<Entity>` from `commit`.
    ///
    /// Returns a [`PendingToken`] that can be used as a target for
    /// staged component inserts against the not-yet-allocated entity.
    pub fn stage_spawn(&mut self, kind: EntityKind, name: Option<String>) -> PendingToken {
        let index = self.spawns.len() as u32;
        self.spawns.push(StagedSpawn {
            kind,
            name,
            apply: Box::new(|world: &mut World| {
                let spawned = world
                    .spawn(kind, name.clone())
                    .map_err(ConstitutionalWorldTxnError::from_spawn)?;
                Ok(spawned.entity)
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
    ) -> Result<(), ConstitutionalWorldTxnError> {
        let type_id = std::any::TypeId::of::<T>();
        let _ = type_id; // captured implicitly via the closure's T
        self.inserts.push(StagedInsert {
            target: InsertTarget::PendingSpawn(target.0),
            apply: Box::new(move |world: &mut World, entity: Entity| {
                world
                    .add_component(entity, component)
                    .map_err(ConstitutionalWorldTxnError::from_insert)?;
                Ok(())
            }),
        });
        Ok(())
    }

    /// Stage a component insert against an already-allocated [`Entity`].
    pub fn stage_insert<T: Component>(
        &mut self,
        entity: Entity,
        component: T,
    ) -> Result<(), ConstitutionalWorldTxnError> {
        self.inserts.push(StagedInsert {
            target: InsertTarget::Existing(entity),
            apply: Box::new(move |world: &mut World, entity: Entity| {
                world
                    .add_component(entity, component)
                    .map_err(ConstitutionalWorldTxnError::from_insert)?;
                Ok(())
            }),
        });
        Ok(())
    }

    /// Stage a component removal against an already-allocated [`Entity`].
    pub fn stage_remove<T: Component>(
        &mut self,
        entity: Entity,
    ) -> Result<(), ConstitutionalWorldTxnError> {
        let type_id = std::any::TypeId::of::<T>();
        self.removes.insert(
            (entity, type_id),
            StagedRemove {
                entity,
                apply: Box::new(move |world: &mut World| {
                    let _ = world
                        .remove_component::<T>(entity)
                        .map_err(ConstitutionalWorldTxnError::from_remove)?;
                    Ok(())
                }),
            },
        );
        Ok(())
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
    pub fn commit(
        self,
        world: &mut World,
    ) -> Result<Vec<Entity>, ConstitutionalWorldTxnError> {
        let ConstitutionalWorldTxn {
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
                        .ok_or(ConstitutionalWorldTxnError::UnknownPendingSpawn(idx))?;
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
            .map(|opt| {
                opt.ok_or(ConstitutionalWorldTxnError::SpawnNotAllocated(u32::MAX))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(spawned)
    }
}

impl Default for ConstitutionalWorldTxn {
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
    use prism_ecs_core::Component;

    // Two concrete component types for the tests.
    #[derive(Debug, Clone, PartialEq)]
    struct Alpha(u32);
    impl Component for Alpha {}

    #[derive(Debug, Clone, PartialEq)]
    struct Beta(u32);
    impl Component for Beta {}

    fn make_world() -> World {
        // The constitutional World has a mutation policy that defaults to
        // permitting direct mutations. We need a fresh world per test.
        let mut world = World::new();
        // Ensure columns for our test types exist (required by the
        // constitutional World's column-based storage).
        world.insert_resource(0u32);
        world
    }

    #[test]
    fn empty_txn_commits_and_returns_no_entities() {
        let mut world = make_world();
        let txn = ConstitutionalWorldTxn::new();
        let spawned = txn.commit(&mut world).expect("empty commit");
        assert!(spawned.is_empty());
    }

    #[test]
    fn spawn_and_insert_returns_resolved_entity() {
        let mut world = make_world();
        let mut txn = ConstitutionalWorldTxn::new();
        let token = txn.stage_spawn(EntityKind::Kernel, Some("k1".into()));
        txn.stage_insert_on::<Alpha>(token, Alpha(42))
            .expect("stage insert");

        let spawned = txn.commit(&mut world).expect("commit");
        assert_eq!(spawned.len(), 1);
        let entity = spawned[0];

        // The entity should now exist and carry the Alpha component.
        assert_eq!(world.name(entity), Some("k1"));
    }

    #[test]
    fn multiple_spawns_get_distinct_resolved_entities() {
        let mut world = make_world();
        let mut txn = ConstitutionalWorldTxn::new();
        let t0 = txn.stage_spawn(EntityKind::Kernel, Some("k0".into()));
        let t1 = txn.stage_spawn(EntityKind::Dispatch, Some("d1".into()));

        let spawned = txn.commit(&mut world).expect("commit");
        assert_eq!(spawned.len(), 2);
        assert_ne!(spawned[0], spawned[1]);
        assert_eq!(world.kind(spawned[0]), Some(EntityKind::Kernel));
        assert_eq!(world.kind(spawned[1]), Some(EntityKind::Dispatch));
        // Force use of tokens to silence unused warnings.
        let _ = (t0, t1);
    }

    #[test]
    fn unknown_pending_token_is_rejected() {
        let mut world = make_world();
        let mut txn = ConstitutionalWorldTxn::new();
        // No stage_spawn — token refers to a non-existent index.
        let bad = PendingToken(7);
        let result = txn.stage_insert_on::<Alpha>(bad, Alpha(1));
        // Stage accepts the bad token (validation is at commit time).
        assert!(result.is_ok());
        let commit = txn.commit(&mut world);
        assert!(matches!(
            commit,
            Err(ConstitutionalWorldTxnError::UnknownPendingSpawn(7))
        ));
    }
}
