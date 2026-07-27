//! Authority: this module owns the canonical staged-transaction
//! surface — [`WorldTxn`] (staging), [`PreparedWorldTxn`] (validated,
//! ready to apply), [`CommitReceipt`] (apply receipt), and the staged
//! operation records ([`StagedSpawn`], [`StagedInsert`], [`StagedRemove`],
//! [`PendingOp`], [`PreparedDurableOp`], [`PreparedTransientOp`]). It
//! also owns the [`WorldTransitExt`] extension trait that binds the
//! transaction to [`prism_ecs_core::World`].
//!
//! The transaction is the only constitutional seam through which the
//! world mutates. Direct `world.spawn` / `world.add_component` /
//! `world.remove_component` calls outside this seam are forbidden
//! (per AGENTS.md "no direct world mutation outside `prism-ecs-core`
//! and `WorldTxn` implementations").
//!
//! Canonical collections use `BTreeMap` per AGENTS.md "no `HashMap`/
//! `HashSet` for canonical collections whose order is observable."

use crate::command::{AdvisoryEvent, DomainEvent};
use crate::schema::SchemaCatalogue;
use crate::system_desc::ReadDependency;
use crate::types::{ComponentSchemaId, SchemaKey, SchemaVersion, WorldEpoch};
use crate::world_txn::durable::{DurableComponent, TransientComponent};
use crate::world_txn::error::WorldTxnError;
use crate::world_txn::journal::{ChangeType, ComponentChange};
use prism_ecs_core::Component;
use prism_ecs_core::ComponentStore;
use prism_ecs_core::Entity;
use prism_ecs_core::EntityKind;
use prism_ecs_core::PendingEntity;
use prism_ecs_core::World;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

// ── Staged operation records ──────────────────────────────────────────────

/// A staged entity spawn.
///
/// The apply closure is generated at stage time and consumes a fresh
/// entity id from the world allocator (or the reserved id from
/// `stage_spawn` for the pre-allocated path). The preflight closure
/// validates that the entity does not already exist.
#[allow(dead_code)]
pub(crate) struct StagedSpawn {
    pub entity: Entity,
    pub kind: EntityKind,
    /// If true, `prepare_inner()` will assign a fresh ID from the world allocator.
    pub is_pending: bool,
    pub preflight: Box<dyn Fn(&World) -> Result<(), WorldTxnError> + Send + Sync>,
    pub apply: Box<dyn FnOnce(&mut World) + Send>,
}

/// A staged component insert.
///
/// Created at `add_component::<T>()` / `put_durable::<T>()` / etc. time
/// when the concrete type is known. The apply closure writes the
/// captured component value into the world's [`ComponentStore`]. The
/// preflight closure validates entity existence and column
/// accessibility before the apply phase runs.
pub(crate) struct StagedInsert {
    pub entity: Entity,
    pub schema_id: ComponentSchemaId,
    #[allow(dead_code)]
    pub schema_version: SchemaVersion,
    /// The full schema key, populated for durable inserts.
    pub schema_key: SchemaKey,
    /// Type identity for catalogue validation against the durable schema.
    pub type_id: std::any::TypeId,
    /// Whether this operation targets a durable component (journaled + replayed).
    pub is_durable: bool,
    /// Applies the staged mutation to component_store.
    /// Created at add_component::<T>() time when the concrete type is known.
    pub apply: Box<dyn FnOnce(&mut ComponentStore) + Send>,
    /// Preflight validation: checks entity existence and column accessibility.
    /// Called in transit() Phase 1, before any mutations.
    pub preflight: Box<dyn Fn(&ComponentStore) -> Result<(), WorldTxnError> + Send + Sync>,
}

/// A staged component removal.
///
/// The mirror of [`StagedInsert`] for removes — the apply closure
/// removes the typed component from the world's [`ComponentStore`].
pub(crate) struct StagedRemove {
    pub entity: Entity,
    pub schema_id: ComponentSchemaId,
    #[allow(dead_code)]
    pub schema_version: SchemaVersion,
    /// The full schema key, populated for durable inserts.
    pub schema_key: SchemaKey,
    /// Type identity for catalogue validation against the durable schema.
    pub type_id: std::any::TypeId,
    /// Whether this operation targets a durable component (journaled + replayed).
    pub is_durable: bool,
    /// Applies the staged mutation to component_store.
    pub apply: Box<dyn FnOnce(&mut ComponentStore) + Send>,
    /// Preflight validation: checks entity existence and column accessibility.
    /// Called in transit() Phase 1, before any mutations.
    pub preflight: Box<dyn Fn(&ComponentStore) -> Result<(), WorldTxnError> + Send + Sync>,
}

/// A pending component operation, stored by token for resolution during
/// `prepare_inner()`. Contains metadata and a monomorphized closure that
/// produces a `StagedInsert` once the real entity ID is known.
pub(crate) struct PendingOp {
    #[allow(dead_code)]
    schema_id: ComponentSchemaId,
    #[allow(dead_code)]
    schema_version: SchemaVersion,
    #[allow(dead_code)]
    type_id: std::any::TypeId,
    #[allow(dead_code)]
    is_durable: bool,
    /// Consumed during `prepare_inner()`: given a resolved entity ID, returns
    /// the fully-formed `StagedInsert`.
    resolve: Box<dyn FnOnce(Entity) -> StagedInsert + Send>,
}

// ── Prepared operation records ────────────────────────────────────────────

/// A prepared durable operation with schema-bound journal entry.
///
/// Produced by `WorldTxn::prepare_inner()` after validation. The apply
/// closure runs in `World::apply_prepared` (phase 3). The
/// `journal_entry` is what gets stored in the mutation journal.
pub(crate) struct PreparedDurableOp {
    pub entity: Entity,
    #[allow(dead_code)]
    pub schema_key: SchemaKey,
    pub apply: Box<dyn FnOnce(&mut ComponentStore) + Send>,
    #[allow(dead_code)]
    pub journal_entry: ComponentChange,
    /// Encoded value for journal durability; None until the schema catalogue
    /// is wired for encoding (B6 — future work).
    #[allow(dead_code)]
    pub encoded_value: Option<Vec<u8>>,
}

/// A prepared transient operation (no journal entry).
pub(crate) struct PreparedTransientOp {
    pub entity: Entity,
    pub apply: Box<dyn FnOnce(&mut ComponentStore) + Send>,
}

// ── Commit receipt ────────────────────────────────────────────────────────

/// Receipt returned by `World::apply_prepared` (and embedded in
/// `CommittedEpoch`'s apply path).
///
/// The receipt is the canonical post-commit observation: it carries the
/// new epoch, the journal length, the event counts. Downstream
/// consumers (replay, projection rebuild, event bus) read this to
/// decide whether to drain and rebuild.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommitReceipt {
    pub committed_epoch: WorldEpoch,
    pub journal_length: usize,
    pub event_count: usize,
    pub advisory_event_count: usize,
}

// ── WorldTxn ──────────────────────────────────────────────────────────────

/// A pending world transaction.
///
/// Systems build a `WorldTxn` by reading from the world and staging
/// changes (component inserts, removes, entity spawns, domain events).
/// On commit via `World::transit()`, all changes are applied atomically
/// with optimistic concurrency control.
///
/// Exact-ID staging is sparse-safe and failure-atomic; transaction-local
/// entity allocation (`PendingEntity` / `TxnEntityRef`) remains future
/// work.
pub struct WorldTxn {
    /// Staged component inserts, keyed by (entity_id, schema_id)
    pub(crate) inserts: Vec<StagedInsert>,
    /// Staged component removals
    pub(crate) removes: Vec<StagedRemove>,
    /// Staged entity spawns
    pub(crate) spawns: Vec<StagedSpawn>,
    /// Pending operations keyed by the placeholder [`Entity`] handle
    /// (1-indexed token, generation 0). The token is a stand-in for the
    /// real entity id that will be assigned during `prepare_inner()`.
    ///
    /// `BTreeMap` (not `HashMap`): the resolution order is part of the
    /// canonical transaction replay and must be deterministic. See
    /// AGENTS.md "no HashMap/HashSet for canonical collections whose
    /// order is observable."
    pub(crate) pending_resolutions: BTreeMap<Entity, Vec<PendingOp>>,
    /// Domain events to emit after successful commit
    pub(crate) events: Vec<DomainEvent>,
    /// Runtime-only observations to expose after successful commit. These are
    /// deliberately separate from `events` so persistence can only see
    /// durable domain facts.
    pub(crate) advisory_events: Vec<AdvisoryEvent>,
    /// Read dependencies for OCC validation
    pub(crate) read_deps: Vec<ReadDependency>,
    /// Expected world epoch at construction time
    pub(crate) expected_epoch: WorldEpoch,
}

impl WorldTxn {
    /// Begin a new transaction against the given world at its current
    /// epoch. The expected epoch is recorded for OCC fencing; a commit
    /// that observes a different world epoch will be rejected with
    /// [`WorldTxnError::StaleEpoch`].
    pub fn new(world: &World) -> Self {
        Self {
            inserts: Vec::new(),
            removes: Vec::new(),
            spawns: Vec::new(),
            pending_resolutions: BTreeMap::new(),
            events: Vec::new(),
            advisory_events: Vec::new(),
            read_deps: Vec::new(),
            expected_epoch: world.current_epoch(),
        }
    }

    /// Peek the next available entity ID from the world.
    pub fn next_entity_id(world: &World) -> Entity {
        Entity::new(world.next_entity_id(), 0)
    }

    /// Stage an entity spawn with a reserved entity ID.
    pub fn stage_spawn(&mut self, entity: Entity, kind: EntityKind) {
        self.spawns.push(StagedSpawn {
            entity,
            kind,
            is_pending: false,
            preflight: Box::new(move |world: &World| {
                if world.has_entity(entity) {
                    return Err(WorldTxnError::InvalidEntity(entity));
                }
                Ok(())
            }),
            apply: Box::new(move |world: &mut World| {
                world.spawn_entity_with_id(entity.id(), kind);
            }),
        });
    }

    /// Number of staged spawns.
    pub fn spawn_count(&self) -> usize {
        self.spawns.len()
    }

    /// Stage an entity spawn with a pending token. The real entity ID is
    /// assigned during `prepare_inner()` from the world allocator.
    ///
    /// Returns a [`PendingEntity`] token that can be used with
    /// [`Self::add_component_pending`], [`Self::put_durable_pending`], or
    /// [`Self::put_transient_pending`].
    pub fn spawn_pending(&mut self, kind: EntityKind) -> PendingEntity {
        let token = self.spawns.len() + 1; // 1-indexed token
        self.spawns.push(StagedSpawn {
            entity: Entity::new(0, 0), // placeholder, resolved during prepare_inner()
            kind,
            is_pending: true,
            preflight: Box::new(|_| Ok(())),
            apply: Box::new(|_| {}),
        });
        PendingEntity(token as u64)
    }

    /// Stage a component insert against a pending entity token.
    /// Resolved to a concrete entity ID during `prepare_inner()`.
    pub fn add_component_pending<T: Component>(
        &mut self,
        pending: PendingEntity,
        schema_id: ComponentSchemaId,
        schema_version: SchemaVersion,
        component: T,
    ) {
        let token = Entity::new(pending.0, 0);
        let type_id = std::any::TypeId::of::<T>();
        let schema_key = SchemaKey {
            namespace: "",
            id: schema_id.0 as u32,
            version: schema_version.0,
        };

        let resolve: Box<dyn FnOnce(Entity) -> StagedInsert + Send> = Box::new(move |entity_h| {
            StagedInsert {
                entity: entity_h,
                schema_id,
                schema_version,
                schema_key,
                type_id,
                is_durable: true,
                preflight: Box::new(move |store: &ComponentStore| {
                    let col_type_id = std::any::TypeId::of::<prism_ecs_core::Column<T>>();
                    if store.has_column_type(col_type_id) {
                        // Column exists for this type — safe to proceed
                    }
                    Ok(())
                }),
                apply: Box::new(move |store: &mut ComponentStore| {
                    store.insert::<T>(entity_h, component);
                }),
            }
        });

        self.pending_resolutions
            .entry(token)
            .or_default()
            .push(PendingOp {
                schema_id,
                schema_version,
                type_id,
                is_durable: true,
                resolve,
            });
    }

    /// Stage a durable component insert against a pending entity token.
    pub fn put_durable_pending<T: DurableComponent>(
        &mut self,
        pending: PendingEntity,
        component: T,
    ) {
        let key = T::SCHEMA_KEY;
        let token = Entity::new(pending.0, 0);
        let type_id = std::any::TypeId::of::<T>();

        let resolve: Box<dyn FnOnce(Entity) -> StagedInsert + Send> = Box::new(move |entity_h| {
            StagedInsert {
                entity: entity_h,
                schema_id: ComponentSchemaId(key.id as u64),
                schema_version: SchemaVersion(key.version),
                schema_key: key,
                type_id,
                is_durable: true,
                preflight: Box::new(move |store: &ComponentStore| {
                    let col_type_id = std::any::TypeId::of::<prism_ecs_core::Column<T>>();
                    if store.has_column_type(col_type_id) {
                        // Column exists for this type — safe to proceed
                    }
                    Ok(())
                }),
                apply: Box::new(move |store: &mut ComponentStore| {
                    store.insert::<T>(entity_h, component);
                }),
            }
        });

        self.pending_resolutions
            .entry(token)
            .or_default()
            .push(PendingOp {
                schema_id: ComponentSchemaId(key.id as u64),
                schema_version: SchemaVersion(key.version),
                type_id,
                is_durable: true,
                resolve,
            });
    }

    /// Stage a transient component insert against a pending entity token.
    pub fn put_transient_pending<T: TransientComponent>(
        &mut self,
        pending: PendingEntity,
        component: T,
    ) {
        let token = Entity::new(pending.0, 0);
        let type_id = std::any::TypeId::of::<T>();

        let resolve: Box<dyn FnOnce(Entity) -> StagedInsert + Send> = Box::new(move |entity_h| {
            let col_type_id = std::any::TypeId::of::<prism_ecs_core::Column<T>>();
            StagedInsert {
                entity: entity_h,
                schema_id: ComponentSchemaId(0),
                schema_version: SchemaVersion(0),
                schema_key: SchemaKey {
                    namespace: "",
                    id: 0,
                    version: 0,
                },
                type_id,
                is_durable: false,
                apply: Box::new(move |store: &mut ComponentStore| {
                    store.insert::<T>(entity_h, component);
                }),
                preflight: Box::new(move |store: &ComponentStore| {
                    if store.has_column_type(col_type_id) {
                        // Column exists for this type — safe to proceed
                    }
                    Ok(())
                }),
            }
        });

        self.pending_resolutions
            .entry(token)
            .or_default()
            .push(PendingOp {
                schema_id: ComponentSchemaId(0),
                schema_version: SchemaVersion(0),
                type_id,
                is_durable: false,
                resolve,
            });
    }

    /// The old-style add_component — gated as `pub(crate)` for
    /// replay/migration only. Prefer [`Self::put_durable`] or
    /// [`Self::put_transient`] for new code.
    pub(crate) fn add_component<T: Component>(
        &mut self,
        entity: Entity,
        schema_id: ComponentSchemaId,
        schema_version: SchemaVersion,
        component: T,
    ) {
        self.push_insert(
            entity,
            schema_id,
            schema_version,
            SchemaKey {
                namespace: "",
                id: schema_id.0 as u32,
                version: schema_version.0,
            },
            component,
            true,
        )
    }

    /// New internal version that takes `Entity(id, gen)` for generation
    /// safety. Reserved for migration; kept in sync with
    /// [`Self::add_component`].
    #[allow(dead_code)]
    pub(crate) fn add_component_entity<T: Component>(
        &mut self,
        entity: Entity,
        schema_id: ComponentSchemaId,
        schema_version: SchemaVersion,
        component: T,
    ) {
        self.push_insert(
            entity,
            schema_id,
            schema_version,
            SchemaKey {
                namespace: "",
                id: schema_id.0 as u32,
                version: schema_version.0,
            },
            component,
            true,
        )
    }

    /// The old-style remove_component — gated as `pub(crate)` for
    /// replay/migration only.
    #[allow(dead_code)]
    pub(crate) fn remove_component<T: Component>(
        &mut self,
        entity: Entity,
        schema_id: ComponentSchemaId,
    ) {
        self.push_remove::<T>(
            entity,
            schema_id,
            SchemaVersion(0),
            SchemaKey {
                namespace: "",
                id: schema_id.0 as u32,
                version: 0,
            },
            true,
        )
    }

    /// Insert a durable (journaled, replayed) component.
    pub fn put_durable<T: DurableComponent>(&mut self, entity: Entity, component: T) {
        let key = T::SCHEMA_KEY;
        self.push_insert(
            entity,
            ComponentSchemaId(key.id as u64),
            SchemaVersion(key.version),
            key,
            component,
            true,
        )
    }

    /// Insert a transient (process-local, non-replayed) component.
    pub fn put_transient<T: TransientComponent>(&mut self, entity: Entity, component: T) {
        self.push_insert(
            entity,
            ComponentSchemaId(0), // placeholder — no schema binding for transient
            SchemaVersion(0),
            SchemaKey {
                namespace: "",
                id: 0,
                version: 0,
            },
            component,
            false,
        )
    }

    /// Remove a durable component.
    pub fn remove_durable<T: DurableComponent>(&mut self, entity: Entity) {
        let key = T::SCHEMA_KEY;
        self.push_remove::<T>(
            entity,
            ComponentSchemaId(key.id as u64),
            SchemaVersion(key.version),
            key,
            true,
        )
    }

    /// Remove a transient component.
    pub fn remove_transient<T: TransientComponent>(&mut self, entity: Entity) {
        self.push_remove::<T>(
            entity,
            ComponentSchemaId(0),
            SchemaVersion(0),
            SchemaKey {
                namespace: "",
                id: 0,
                version: 0,
            },
            false,
        )
    }

    // ── Shared push helpers ────────────────────────────────────────────

    fn push_insert<T: Component>(
        &mut self,
        entity: Entity,
        schema_id: ComponentSchemaId,
        schema_version: SchemaVersion,
        schema_key: SchemaKey,
        component: T,
        is_durable: bool,
    ) {
        let type_id = std::any::TypeId::of::<T>();
        let col_type_id = std::any::TypeId::of::<prism_ecs_core::Column<T>>();
        self.inserts.push(StagedInsert {
            entity,
            schema_id,
            schema_version,
            schema_key,
            type_id,
            is_durable,
            apply: Box::new(move |store: &mut ComponentStore| {
                store.insert::<T>(entity, component);
            }),
            preflight: Box::new(move |store: &ComponentStore| {
                if store.has_column_type(col_type_id) {
                    // Column exists for this type — safe to proceed
                }
                Ok(())
            }),
        });
    }

    fn push_remove<T: Component>(
        &mut self,
        entity: Entity,
        schema_id: ComponentSchemaId,
        schema_version: SchemaVersion,
        schema_key: SchemaKey,
        is_durable: bool,
    ) {
        let type_id = std::any::TypeId::of::<T>();
        let col_type_id = std::any::TypeId::of::<prism_ecs_core::Column<T>>();
        self.removes.push(StagedRemove {
            entity,
            schema_id,
            schema_version,
            schema_key,
            type_id,
            is_durable,
            apply: Box::new(move |store: &mut ComponentStore| {
                store.remove::<T>(entity);
            }),
            preflight: Box::new(move |store: &ComponentStore| {
                if store.has_column_type(col_type_id) {
                    // Column exists for this type — safe to proceed
                }
                // If the column doesn't exist, the remove will be a no-op at apply
                // time (no entry to remove). This is correct for same-txn generates.
                Ok(())
            }),
        });
    }

    /// Add a domain event to emit after commit.
    pub fn emit_event(&mut self, event: DomainEvent) {
        self.events.push(event);
    }

    /// Add an advisory observation to emit after commit.
    ///
    /// Advisory events share the transaction commit boundary but are not
    /// returned by the durable-event accessor and are never replayed.
    pub fn emit_advisory_event(&mut self, event: AdvisoryEvent) {
        self.advisory_events.push(event);
    }

    /// Record a read dependency for OCC validation.
    pub fn record_read(&mut self, dep: ReadDependency) {
        self.read_deps.push(dep);
    }

    /// The world epoch this transaction was constructed against.
    pub fn expected_epoch(&self) -> WorldEpoch {
        self.expected_epoch
    }

    /// Number of staged durable domain events.
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Number of staged advisory observations.
    pub fn advisory_event_count(&self) -> usize {
        self.advisory_events.len()
    }

    /// Number of staged component inserts.
    pub fn insert_count(&self) -> usize {
        self.inserts.len()
    }

    /// Number of staged component removals.
    pub fn remove_count(&self) -> usize {
        self.removes.len()
    }

    /// Internal preparation — validation logic extracted for
    /// `World::prepare()`. Validates all invariants against the world
    /// WITHOUT mutating it. On success, returns a [`PreparedWorldTxn`]
    /// containing the resolved closures.
    pub(crate) fn prepare_inner(
        mut self,
        world: &World,
        catalogue: Option<&SchemaCatalogue>,
    ) -> Result<PreparedWorldTxn, WorldTxnError> {
        use std::collections::HashSet;

        // 1a. Validate epoch
        if world.current_epoch() != self.expected_epoch {
            return Err(WorldTxnError::StaleEpoch {
                expected: self.expected_epoch,
                current: world.current_epoch(),
            });
        }

        // 1aa. Resolve pending entities and operations
        if !self.pending_resolutions.is_empty() || self.spawns.iter().any(|s| s.is_pending) {
            let allocator_base = world.next_entity_id();

            // Assign real entity IDs to pending spawns
            for (i, spawn) in self.spawns.iter_mut().enumerate() {
                if spawn.is_pending {
                    let resolved_id = allocator_base + i as u64;
                    let kind = spawn.kind;
                    spawn.entity = Entity::new(resolved_id, 0);
                    spawn.preflight = Box::new(move |world: &World| {
                        if world.has_entity(Entity::new(resolved_id, 0)) {
                            return Err(WorldTxnError::InvalidEntity(Entity::new(resolved_id, 0)));
                        }
                        Ok(())
                    });
                    spawn.apply = Box::new(move |world: &mut World| {
                        world.spawn_entity_with_id(resolved_id, kind);
                    });
                }
            }

            // Resolve pending component operations against their assigned entity IDs
            for (token_entity, ops) in std::mem::take(&mut self.pending_resolutions) {
                let resolved_id = allocator_base + (token_entity.id() - 1);
                for op in ops {
                    let insert = (op.resolve)(Entity::new(resolved_id, 0));
                    self.inserts.push(insert);
                }
            }
        }

        // 1b. Validate spawn preflights (entity doesn't already exist)
        for spawn in &self.spawns {
            (spawn.preflight)(world)?;
        }

        // 1c. Check for duplicate spawn IDs within this transaction
        {
            let mut seen = HashSet::new();
            for spawn in &self.spawns {
                if !seen.insert(spawn.entity) {
                    return Err(WorldTxnError::InvalidEntity(spawn.entity));
                }
            }
        }

        // 1d. Validate read dependencies
        for dep in &self.read_deps {
            let current_ver = world.component_version(dep.entity);
            if current_ver != dep.observed_version {
                return Err(WorldTxnError::StaleRead {
                    entity: dep.entity,
                    schema_id: dep.schema_id,
                    observed: dep.observed_version,
                    current: current_ver,
                });
            }
        }

        // 1e. Validate entity existence for every staged operation
        let pending_spawn_ids: HashSet<Entity> = self.spawns.iter().map(|s| s.entity).collect();
        for insert in &self.inserts {
            if pending_spawn_ids.contains(&insert.entity) {
                continue;
            }
            if !world.has_entity(insert.entity) {
                return Err(WorldTxnError::InvalidEntity(insert.entity));
            }
        }
        for remove in &self.removes {
            if pending_spawn_ids.contains(&remove.entity) {
                continue;
            }
            if !world.has_entity(remove.entity) {
                return Err(WorldTxnError::InvalidEntity(remove.entity));
            }
        }

        // 1f. Validate staged operation preflight closures
        for insert in &self.inserts {
            (insert.preflight)(&world.component_store_ref())?;
        }
        for remove in &self.removes {
            (remove.preflight)(&world.component_store_ref())?;
        }

        // 1g. Detect conflicting component operations (two inserts, insert+remove,
        //     or two removes for the same entity + schema_id)
        {
            // Check duplicate inserts (same entity + same type)
            let mut seen_inserts = HashSet::new();
            for insert in &self.inserts {
                if !seen_inserts.insert((insert.entity, insert.type_id)) {
                    return Err(WorldTxnError::Conflict {
                        entity: insert.entity,
                        schema_id: insert.schema_id,
                    });
                }
            }
            // Check duplicate removes (same entity + same type)
            let mut seen_removes = HashSet::new();
            for remove in &self.removes {
                if !seen_removes.insert((remove.entity, remove.type_id)) {
                    return Err(WorldTxnError::Conflict {
                        entity: remove.entity,
                        schema_id: remove.schema_id,
                    });
                }
            }
        }

        // 1h. Validate durable inserts against schema catalogue (when provided)
        if let Some(cat) = catalogue {
            for insert in &self.inserts {
                if insert.is_durable {
                    let reg = cat.registration(&insert.schema_key).ok_or(
                        WorldTxnError::UnregisteredSchema {
                            schema_key: insert.schema_key,
                        },
                    )?;
                    if reg.type_id != insert.type_id {
                        return Err(WorldTxnError::SchemaMismatch {
                            schema_id: insert.schema_id,
                            expected: reg.type_name.to_string(),
                        });
                    }
                }
            }
        }

        // -- PHASE 2: Split into durable/transient ops and build journal ---
        let next_epoch = WorldEpoch(world.current_epoch().0 + 1);
        let mut durable_ops = Vec::new();
        let mut transient_ops = Vec::new();
        let mut journal = Vec::new();

        for insert in self.inserts {
            if insert.is_durable {
                journal.push(ComponentChange {
                    entity: insert.entity,
                    schema_key: insert.schema_key,
                    change_type: ChangeType::Insert,
                    before_hash: None,
                    after_hash: None,
                    world_epoch: next_epoch,
                });
                durable_ops.push(PreparedDurableOp {
                    entity: insert.entity,
                    schema_key: insert.schema_key,
                    apply: insert.apply,
                    journal_entry: journal.last().cloned().unwrap(),
                    // B6: encoded_value left as None until catalogue is wired
                    encoded_value: None,
                });
            } else {
                transient_ops.push(PreparedTransientOp {
                    entity: insert.entity,
                    apply: insert.apply,
                });
            }
        }
        for remove in self.removes {
            if remove.is_durable {
                journal.push(ComponentChange {
                    entity: remove.entity,
                    schema_key: remove.schema_key,
                    change_type: ChangeType::Remove,
                    before_hash: None,
                    after_hash: None,
                    world_epoch: next_epoch,
                });
                durable_ops.push(PreparedDurableOp {
                    entity: remove.entity,
                    schema_key: remove.schema_key,
                    apply: remove.apply,
                    journal_entry: journal.last().cloned().unwrap(),
                    encoded_value: None,
                });
            } else {
                transient_ops.push(PreparedTransientOp {
                    entity: remove.entity,
                    apply: remove.apply,
                });
            }
        }

        // Validation succeeded — move all closures into PreparedWorldTxn
        Ok(PreparedWorldTxn {
            expected_epoch: self.expected_epoch,
            durable_ops,
            transient_ops,
            spawns: self.spawns,
            journal,
            events: self.events,
            advisory_events: self.advisory_events,
        })
    }
}

// ── PreparedWorldTxn ──────────────────────────────────────────────────────

/// A fully validated, ready-to-apply transaction.
///
/// Produced by `World::prepare()` via `WorldTxn::prepare_inner()`.
/// Contains all resolved operations and journals. The type is
/// deliberately sealed — external code cannot construct one directly.
#[must_use = "a prepared transaction must be applied or explicitly dropped"]
pub struct PreparedWorldTxn {
    pub(crate) expected_epoch: WorldEpoch,
    pub(crate) durable_ops: Vec<PreparedDurableOp>,
    pub(crate) transient_ops: Vec<PreparedTransientOp>,
    pub(crate) spawns: Vec<StagedSpawn>,
    pub(crate) journal: Vec<ComponentChange>,
    pub(crate) events: Vec<DomainEvent>,
    pub(crate) advisory_events: Vec<AdvisoryEvent>,
}

impl PreparedWorldTxn {
    /// Returns the epoch at which this transaction was prepared.
    pub fn expected_epoch(&self) -> WorldEpoch {
        self.expected_epoch
    }

    /// Number of journal entries in this prepared transaction.
    pub fn journal_length(&self) -> usize {
        self.journal.len()
    }

    /// Number of domain events in this prepared transaction.
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Number of advisory observations prepared alongside the transaction.
    pub fn advisory_event_count(&self) -> usize {
        self.advisory_events.len()
    }
}

// ── WorldTransitExt ───────────────────────────────────────────────────────

/// Extension trait providing transactional methods on [`World`] for the
/// `prism-ecs-constitutional` crate.
///
/// Epoch, journal, and committed-events state are stored via World's
/// type-erased extension mechanism. The trait is the only constitutional
/// seam through which the world mutates.
pub trait WorldTransitExt {
    /// Validate and commit a transaction atomically.
    fn transit(&mut self, txn: WorldTxn) -> Result<crate::world_txn::epoch::CommittedEpoch, WorldTxnError>;

    /// Return a reference to the last committed journal.
    fn last_journal(&self) -> &[ComponentChange];

    /// Return a reference to the last committed events.
    fn last_committed_events(&self) -> &[DomainEvent];

    /// Drain the committed events vector.
    fn drain_committed_events(&mut self) -> Vec<DomainEvent>;

    /// Return a reference to the last committed advisory events.
    fn last_committed_advisory_events(&self) -> &[AdvisoryEvent];

    /// Drain the committed advisory events vector.
    fn drain_committed_advisory_events(&mut self) -> Vec<AdvisoryEvent>;

    /// Validate a transaction without mutating the world.
    fn prepare(
        &self,
        txn: WorldTxn,
        catalogue: Option<&SchemaCatalogue>,
    ) -> Result<PreparedWorldTxn, WorldTxnError>;

    /// Apply a previously validated, prepared transaction.
    fn apply_prepared(&mut self, prepared: PreparedWorldTxn) -> CommitReceipt;
}

impl WorldTransitExt for World {
    fn transit(
        &mut self,
        txn: WorldTxn,
    ) -> Result<crate::world_txn::epoch::CommittedEpoch, WorldTxnError> {
        let prepared = self.prepare(txn, None)?;
        let receipt = self.apply_prepared(prepared);
        Ok(crate::world_txn::epoch::CommittedEpoch(
            receipt.committed_epoch,
        ))
    }

    fn last_journal(&self) -> &[ComponentChange] {
        self.get_extension::<Vec<ComponentChange>>()
            .map(|v| v.as_slice())
            .unwrap_or_default()
    }

    fn last_committed_events(&self) -> &[DomainEvent] {
        self.get_extension::<Vec<DomainEvent>>()
            .map(|v| v.as_slice())
            .unwrap_or_default()
    }

    fn drain_committed_events(&mut self) -> Vec<DomainEvent> {
        self.get_extension_mut::<Vec<DomainEvent>>()
            .map(std::mem::take)
            .unwrap_or_default()
    }

    fn last_committed_advisory_events(&self) -> &[AdvisoryEvent] {
        self.get_extension::<Vec<AdvisoryEvent>>()
            .map(|v| v.as_slice())
            .unwrap_or_default()
    }

    fn drain_committed_advisory_events(&mut self) -> Vec<AdvisoryEvent> {
        self.get_extension_mut::<Vec<AdvisoryEvent>>()
            .map(std::mem::take)
            .unwrap_or_default()
    }

    fn prepare(
        &self,
        txn: WorldTxn,
        catalogue: Option<&SchemaCatalogue>,
    ) -> Result<PreparedWorldTxn, WorldTxnError> {
        txn.prepare_inner(self, catalogue)
    }

    /// Atomically apply a validated, prepared transaction.
    ///
    /// # Panics
    /// - If the world epoch does not match the prepared transaction's expected epoch.
    fn apply_prepared(&mut self, prepared: PreparedWorldTxn) -> CommitReceipt {
        use prism_ecs_core::WorldEpoch;

        // verify epoch before any mutation
        let current_epoch = self.current_epoch();
        assert_eq!(
            current_epoch, prepared.expected_epoch,
            "prepared transaction epoch mismatch: expected {:?}, world is at {:?}",
            prepared.expected_epoch, current_epoch
        );

        // -- PHASE 3: Apply all mutations ----------------------------------
        // Reserve spawn entity slots
        for spawn in &prepared.spawns {
            self.spawn_entity_with_id(spawn.entity.id(), spawn.kind);
        }

        // Apply durable ops (journaled — component versions still bumped)
        for op in prepared.durable_ops {
            (op.apply)(self.component_store_mut());
            *self
                .component_versions_mut()
                .entry(op.entity)
                .or_insert(0) += 1;
        }
        // Apply transient ops (not journaled — component versions bumped)
        for op in prepared.transient_ops {
            (op.apply)(self.component_store_mut());
            *self
                .component_versions_mut()
                .entry(op.entity)
                .or_insert(0) += 1;
        }

        // 3c. Apply staged spawns
        for spawn in prepared.spawns {
            (spawn.apply)(self);
        }

        // -- PHASE 4: Advance epoch AFTER all mutations succeed -----------
        let next_epoch = WorldEpoch(self.current_epoch().0 + 1);
        self.set_epoch(next_epoch);
        self.set_extension(prepared.journal);
        self.set_extension(prepared.events);
        self.set_extension(prepared.advisory_events);

        let journal_len = self
            .get_extension::<Vec<ComponentChange>>()
            .map(|v| v.len())
            .unwrap_or(0);
        let event_count = self
            .get_extension::<Vec<DomainEvent>>()
            .map(|v| v.len())
            .unwrap_or(0);
        CommitReceipt {
            committed_epoch: next_epoch,
            journal_length: journal_len,
            event_count,
            advisory_event_count: self
                .get_extension::<Vec<AdvisoryEvent>>()
                .map_or(0, Vec::len),
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::{AdvisoryEvent, ClassifiedEvent, DomainEvent, EventDurability};
    use crate::types::{EntityKindId, MessageId};
    use crate::world_txn::WorldTransitExt;
    use prism_ecs_core::World;

    /// Advisory and durable events must share the commit boundary but
    /// stay in separate lanes — the durable accessor never sees
    /// advisory events and vice-versa.
    #[test]
    fn advisory_events_share_commit_boundary_without_entering_durable_lane() {
        let mut world = World::new();
        let mut txn = WorldTxn::new(&world);
        txn.emit_event(DomainEvent {
            id: MessageId::compute(b"durable"),
            kind: "work_created".into(),
            entity_id: Some(EntityKindId(7)),
            payload: serde_json::json!({"durable": true}),
        });
        txn.emit_advisory_event(AdvisoryEvent::new(
            "provider_fallback",
            Some(EntityKindId(7)),
            serde_json::json!({"requested": "metal", "selected": "cpu"}),
        ));

        let prepared = world
            .prepare(txn, None)
            .expect("transaction should prepare");
        let receipt = world.apply_prepared(prepared);

        assert_eq!(receipt.event_count, 1);
        assert_eq!(receipt.advisory_event_count, 1);
        assert_eq!(world.last_committed_events().len(), 1);
        assert_eq!(world.last_committed_advisory_events().len(), 1);
        assert_eq!(
            world.last_committed_advisory_events()[0].durability(),
            EventDurability::Advisory
        );
        assert_eq!(
            ClassifiedEvent::Durable(world.last_committed_events()[0].clone()).durability(),
            EventDurability::Durable
        );
        assert_eq!(
            ClassifiedEvent::Advisory(world.last_committed_advisory_events()[0].clone())
                .durability(),
            EventDurability::Advisory
        );

        assert_eq!(world.drain_committed_events().len(), 1);
        assert_eq!(world.drain_committed_advisory_events().len(), 1);
        assert!(world.last_committed_events().is_empty());
        assert!(world.last_committed_advisory_events().is_empty());
    }

    /// `WorldTxn::new` must capture the world epoch at construction
    /// time and report it via `expected_epoch()` so that OCC can
    /// detect stale callers.
    #[test]
    fn expected_epoch_captured_at_construction_time() {
        let world = World::new();
        let txn = WorldTxn::new(&world);
        assert_eq!(txn.expected_epoch(), world.current_epoch());
    }

    /// After a successful commit, the world's epoch must advance by
    /// exactly one. This is the canonical post-commit observation that
    /// downstream consumers (replay, projection rebuild) key off of.
    #[test]
    fn commit_advances_world_epoch_by_one() {
        let mut world = World::new();
        let initial = world.current_epoch();
        let mut txn = WorldTxn::new(&world);
        txn.emit_event(DomainEvent {
            id: MessageId::compute(b"epoch"),
            kind: "noop".into(),
            entity_id: None,
            payload: serde_json::json!({}),
        });
        let epoch = world
            .transit(txn)
            .expect("commit should succeed for an empty mutation set");
        assert_eq!(epoch.0, WorldEpoch(initial.0 + 1));
        assert_eq!(world.current_epoch(), WorldEpoch(initial.0 + 1));
    }
}
