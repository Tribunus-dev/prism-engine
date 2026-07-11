pub use crate::ecs::constitutional::command::DomainEvent;
use crate::ecs::constitutional::system_desc::ReadDependency;
pub use crate::ecs::constitutional::types::*;
use crate::ecs::CompWorld;
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComponentChange {
    pub entity: u64,
    pub schema_id: ComponentSchemaId,
    pub schema_version: SchemaVersion,
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
}

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
    /// Domain events to emit after successful commit
    pub(crate) events: Vec<DomainEvent>,
    /// Read dependencies for OCC validation
    pub(crate) read_deps: Vec<ReadDependency>,
    /// Expected world epoch at construction time
    pub(crate) expected_epoch: WorldEpoch,
}

/// A staged component insert.
pub(crate) struct StagedInsert {
    pub entity: u64,
    pub schema_id: ComponentSchemaId,
    pub schema_version: SchemaVersion,
    /// Applies the staged mutation to component_store.
    /// Created at add_component::<T>() time when the concrete type is known.
    pub apply: Box<dyn FnOnce(&mut ComponentStore) + Send>,
}

/// A staged component removal.
pub(crate) struct StagedRemove {
    pub entity: u64,
    pub schema_id: ComponentSchemaId,
}

use crate::ecs::ComponentStore;

impl WorldTxn {
    /// Begin a new transaction against the given world at its current epoch.
    pub fn new(world: &CompWorld) -> Self {
        Self {
            inserts: Vec::new(),
            removes: Vec::new(),
            events: Vec::new(),
            read_deps: Vec::new(),
            expected_epoch: world.current_epoch(),
        }
    }

    /// Stage a component insert or update.
    pub fn add_component<T: 'static + Send + Sync>(
        &mut self,
        entity: u64,
        schema_id: ComponentSchemaId,
        schema_version: SchemaVersion,
        component: T,
    ) {
        use std::collections::HashMap;
        let type_id = std::any::TypeId::of::<T>();
        self.inserts.push(StagedInsert {
            entity,
            schema_id,
            schema_version,
            apply: Box::new(move |store: &mut ComponentStore| {
                let map: &mut HashMap<u64, T> = store
                    .data
                    .entry(type_id)
                    .or_insert_with(|| Box::new(HashMap::<u64, T>::new()))
                    .downcast_mut::<HashMap<u64, T>>()
                    .expect("type mismatch in ComponentStore");
                map.insert(entity, component);
            }),
        });
    }

    /// Stage a component removal.
    pub fn remove_component(&mut self, entity: u64, schema_id: ComponentSchemaId) {
        self.removes.push(StagedRemove { entity, schema_id });
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
