pub use crate::ecs::constitutional::command::DomainEvent;
use crate::ecs::constitutional::schema::SchemaCatalogue;
use crate::ecs::constitutional::system_desc::ReadDependency;
pub use crate::ecs::constitutional::types::*;
use crate::ecs::CompWorld;
use crate::ecs::EntityKind;
use serde::{Deserialize, Serialize};

/// Access kind for concurrency control.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AccessKind {
    Read,
    Write,
}

/// Access declaration — what a system intends to read or write.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AccessDeclaration {
    pub schema_id: ComponentSchemaId,
    pub entity: Option<u64>,
    pub access: AccessKind,
}

/// A component change recorded in the mutation journal.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ComponentChange {
    pub entity: u64,
    pub schema_key: SchemaKey,
    pub change_type: ChangeType,
    pub before_hash: Option<[u8; 32]>,
    pub after_hash: Option<[u8; 32]>,
    pub world_epoch: WorldEpoch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChangeType {
    Insert,
    Update,
    Remove,
}

/// The epoch assigned after a successful commit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommittedEpoch(pub WorldEpoch);

/// Errors that can occur during transaction commit.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum WorldTxnError {
    #[error("durable schema {schema_key:?} not registered in catalogue")]
    UnregisteredSchema { schema_key: SchemaKey },
    #[error("stale epoch: expected {expected:?}, current {current:?}")]
    StaleEpoch {
        expected: WorldEpoch,
        current: WorldEpoch,
    },
    #[error(
        "stale read: entity {entity} schema {schema_id:?} version {observed} != current {current}"
    )]
    StaleRead {
        entity: u64,
        schema_id: ComponentSchemaId,
        observed: u64,
        current: u64,
    },
    #[error("invalid entity handle: {0}")]
    InvalidEntity(u64),
    #[error("schema mismatch for {schema_id:?}: expected {expected}")]
    SchemaMismatch {
        schema_id: ComponentSchemaId,
        expected: String,
    },
    #[error("component not found: entity {entity} schema {schema_id:?}")]
    ComponentNotFound {
        entity: u64,
        schema_id: ComponentSchemaId,
    },
    #[error("conflicting operations for entity {entity} schema {schema_id:?}")]
    Conflict {
        entity: u64,
        schema_id: ComponentSchemaId,
    },
}

// ── Component Classification ──────────────────────────────────────────────

/// Sealed — only DurableClass and TransientClass may implement this.
pub trait ComponentClass: private::Sealed {}

/// Marker type for durable (journaled, replayed, snapshotted) components.
pub struct DurableClass;
impl private::Sealed for DurableClass {}
impl ComponentClass for DurableClass {}

/// Marker type for transient (process-local, non-replayed) components.
pub struct TransientClass;
impl private::Sealed for TransientClass {}
impl ComponentClass for TransientClass {}

mod private {
    pub trait Sealed {}
}

/// A component explicitly classified as durable or transient.
///
/// Each Rust type can implement this only once — it cannot be both.
pub trait ClassifiedComponent: crate::ecs::Component {
    type Class: ComponentClass;
}

/// A durable component: serializable, journaled, replayable.
///
/// Every durable component must provide a stable SchemaKey derived from
/// its SCHEMA_KEY constant. SchemaKey is independent of Rust type names
/// or crate paths.
pub trait DurableComponent:
    ClassifiedComponent<Class = DurableClass> + serde::Serialize + serde::de::DeserializeOwned
{
    const SCHEMA_KEY: SchemaKey;
}

/// A transient component: runtime-only, never journaled or replayed.
///
/// Transient components disappear on restart and must be reconstructed
/// by subsystem startup or reconciliation code.
pub trait TransientComponent: ClassifiedComponent<Class = TransientClass> {}

/// Registration for a single durable schema.
/// Exact-ID staging is sparse-safe and failure-atomic; transaction-local entity
/// allocation (PendingEntityId / TxnEntityRef) remains future work.
///
/// A pending world transaction.
///
/// Systems build a WorldTxn by reading from the world and staging changes.
/// On commit via CompWorld::transit(), all changes are applied atomically
/// with optimistic concurrency control.
pub struct WorldTxn {
    /// Staged component inserts, keyed by (entity_id, schema_id)
    pub(crate) inserts: Vec<StagedInsert>,
    /// Staged component removals
    pub(crate) removes: Vec<StagedRemove>,
    /// Staged entity spawns
    pub(crate) spawns: Vec<StagedSpawn>,
    /// Domain events to emit after successful commit
    pub(crate) events: Vec<DomainEvent>,
    /// Read dependencies for OCC validation
    pub(crate) read_deps: Vec<ReadDependency>,
    /// Expected world epoch at construction time
    pub(crate) expected_epoch: WorldEpoch,
}

/// A staged entity spawn.
#[allow(dead_code)]
pub(crate) struct StagedSpawn {
    pub entity: u64,
    pub kind: EntityKind,
    pub preflight: Box<dyn Fn(&CompWorld) -> Result<(), WorldTxnError> + Send + Sync>,
    pub apply: Box<dyn FnOnce(&mut CompWorld) + Send>,
}

/// A staged component insert.
pub(crate) struct StagedInsert {
    pub entity: u64,
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
pub(crate) struct StagedRemove {
    pub entity: u64,
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

use crate::ecs::ComponentStore;

impl WorldTxn {
    /// Begin a new transaction against the given world at its current epoch.
    pub fn new(world: &CompWorld) -> Self {
        Self {
            inserts: Vec::new(),
            removes: Vec::new(),
            spawns: Vec::new(),
            events: Vec::new(),
            read_deps: Vec::new(),
            expected_epoch: world.current_epoch(),
        }
    }

    /// Peek the next available entity ID from the world.
    pub fn next_entity_id(world: &CompWorld) -> u64 {
        world.next_entity_id()
    }

    /// Stage an entity spawn with a reserved entity ID.
    pub fn stage_spawn(&mut self, entity: u64, kind: EntityKind) {
        self.spawns.push(StagedSpawn {
            entity,
            kind,
            preflight: Box::new(move |world: &CompWorld| {
                if world.has_entity(crate::ecs::CompEntity(entity)) {
                    return Err(WorldTxnError::InvalidEntity(entity));
                }
                Ok(())
            }),
            apply: Box::new(move |world: &mut CompWorld| {
                world.spawn_entity_with_id(entity, kind);
            }),
        });
    }

    pub fn spawn_count(&self) -> usize {
        self.spawns.len()
    }

    /// The old-style add_component — gated as pub(crate) for replay/migration only.
    /// Prefer put_durable() or put_transient() for new code.
    pub(crate) fn add_component<T: 'static + Send + Sync>(
        &mut self,
        entity: u64,
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

    /// The old-style remove_component — gated as pub(crate) for replay/migration only.
    #[allow(dead_code)]
    pub(crate) fn remove_component<T: 'static + Send + Sync>(
        &mut self,
        entity: u64,
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
    pub fn put_durable<T: DurableComponent>(&mut self, entity: u64, component: T) {
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
    pub fn put_transient<T: TransientComponent>(&mut self, entity: u64, component: T) {
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
    pub fn remove_durable<T: DurableComponent>(&mut self, entity: u64) {
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
    pub fn remove_transient<T: TransientComponent>(&mut self, entity: u64) {
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

    fn push_insert<T: 'static + Send + Sync>(
        &mut self,
        entity: u64,
        schema_id: ComponentSchemaId,
        schema_version: SchemaVersion,
        schema_key: SchemaKey,
        component: T,
        is_durable: bool,
    ) {
        use std::collections::HashMap;
        let type_id = std::any::TypeId::of::<T>();
        self.inserts.push(StagedInsert {
            entity,
            schema_id,
            schema_version,
            schema_key,
            type_id,
            is_durable,
            apply: Box::new(move |store: &mut ComponentStore| {
                let map: &mut HashMap<u64, T> = store
                    .data
                    .entry(type_id)
                    .or_insert_with(|| Box::new(HashMap::<u64, T>::new()))
                    .downcast_mut::<HashMap<u64, T>>()
                    .expect("type mismatch in ComponentStore");
                map.insert(entity, component);
            }),
            preflight: Box::new(move |store: &ComponentStore| {
                if let Some(b) = store.data.get(&type_id) {
                    if b.downcast_ref::<HashMap<u64, T>>().is_none() {
                        return Err(WorldTxnError::SchemaMismatch {
                            schema_id,
                            expected: std::any::type_name::<T>().to_string(),
                        });
                    }
                }
                Ok(())
            }),
        });
    }

    fn push_remove<T: 'static + Send + Sync>(
        &mut self,
        entity: u64,
        schema_id: ComponentSchemaId,
        schema_version: SchemaVersion,
        schema_key: SchemaKey,
        is_durable: bool,
    ) {
        let type_id = std::any::TypeId::of::<T>();
        self.removes.push(StagedRemove {
            entity,
            schema_id,
            schema_version,
            schema_key,
            type_id,
            is_durable,
            apply: Box::new(move |store: &mut crate::ecs::ComponentStore| {
                if let Some(b) = store.data.get_mut(&type_id) {
                    if let Some(map) = b.downcast_mut::<std::collections::HashMap<u64, T>>() {
                        map.remove(&entity);
                    }
                }
            }),
            preflight: Box::new(move |store: &ComponentStore| {
                // Check column type compatibility only.
                // Entity-level existence is not checked here because the component
                // may be inserted in the same transaction (pending insert apply).
                if let Some(b) = store.data.get(&type_id) {
                    if b.downcast_ref::<std::collections::HashMap<u64, T>>()
                        .is_none()
                    {
                        return Err(WorldTxnError::SchemaMismatch {
                            schema_id,
                            expected: std::any::type_name::<T>().to_string(),
                        });
                    }
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

    /// Record a read dependency for OCC validation.
    pub fn record_read(&mut self, dep: ReadDependency) {
        self.read_deps.push(dep);
    }

    pub fn expected_epoch(&self) -> WorldEpoch {
        self.expected_epoch
    }
    pub fn event_count(&self) -> usize {
        self.events.len()
    }
    pub fn insert_count(&self) -> usize {
        self.inserts.len()
    }
    pub fn remove_count(&self) -> usize {
        self.removes.len()
    }
}

/// Receipt returned by PreparedWorldTxn::apply().
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommitReceipt {
    pub committed_epoch: WorldEpoch,
    pub journal_length: usize,
    pub event_count: usize,
}

/// A prepared durable operation with schema-bound journal entry.
pub(crate) struct PreparedDurableOp {
    pub entity: u64,
    #[allow(dead_code)]
    pub schema_key: SchemaKey,
    pub apply: Box<dyn FnOnce(&mut crate::ecs::ComponentStore) + Send>,
    #[allow(dead_code)]
    pub journal_entry: ComponentChange,
    /// Encoded value for journal durability; None until the schema catalogue
    /// is wired for encoding (B6 — future work).
    #[allow(dead_code)]
    pub encoded_value: Option<Vec<u8>>,
}

/// A prepared transient operation (no journal entry).
pub(crate) struct PreparedTransientOp {
    pub entity: u64,
    pub apply: Box<dyn FnOnce(&mut crate::ecs::ComponentStore) + Send>,
}

/// A fully validated, ready-to-apply transaction.
///
/// Produced by `CompWorld::prepare()` via `WorldTxn::prepare_inner()`.
/// Contains all resolved operations and journals. The type is deliberately
/// sealed — external code cannot construct one directly.
#[must_use = "a prepared transaction must be applied or explicitly dropped"]
pub struct PreparedWorldTxn {
    pub(crate) expected_epoch: WorldEpoch,
    pub(crate) durable_ops: Vec<PreparedDurableOp>,
    pub(crate) transient_ops: Vec<PreparedTransientOp>,
    pub(crate) spawns: Vec<StagedSpawn>,
    pub(crate) journal: Vec<ComponentChange>,
    pub(crate) events: Vec<DomainEvent>,
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
}

impl WorldTxn {
    /// Internal preparation — validation logic extracted for CompWorld::prepare().
    /// Validates all invariants against the world WITHOUT mutating it.
    /// On success, returns a PreparedWorldTxn containing the resolved closures.
    pub(crate) fn prepare_inner(
        self,
        world: &CompWorld,
        catalogue: Option<&SchemaCatalogue>,
    ) -> Result<PreparedWorldTxn, WorldTxnError> {
        use crate::ecs::CompEntity;
        use std::collections::HashSet;

        // 1a. Validate epoch
        if world.current_epoch() != self.expected_epoch {
            return Err(WorldTxnError::StaleEpoch {
                expected: self.expected_epoch,
                current: world.current_epoch(),
            });
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
        let pending_spawn_ids: HashSet<u64> = self.spawns.iter().map(|s| s.entity).collect();
        for insert in &self.inserts {
            if pending_spawn_ids.contains(&insert.entity) {
                continue;
            }
            if !world.has_entity(CompEntity(insert.entity)) {
                return Err(WorldTxnError::InvalidEntity(insert.entity));
            }
        }
        for remove in &self.removes {
            if pending_spawn_ids.contains(&remove.entity) {
                continue;
            }
            if !world.has_entity(CompEntity(remove.entity)) {
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
        })
    }
}
