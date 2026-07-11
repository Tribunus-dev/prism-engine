use crate::ecs::constitutional::types::*;
use serde::{Deserialize, Serialize};

/// Component version for optimistic concurrency control.
pub type ComponentVersion = u64;

/// A read dependency recorded during a transaction — the entity, component schema,
/// and the version that was observed.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ReadDependency {
    pub entity: u64, // process-local entity id
    pub schema_id: ComponentSchemaId,
    pub observed_version: ComponentVersion,
}

/// Declares what a system reads and writes — used by the scheduler for concurrency
/// detection and by WorldTxn for permission enforcement.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SystemDescriptor {
    pub name: String,
    pub read_schemas: Vec<ComponentSchemaId>,
    pub write_schemas: Vec<ComponentSchemaId>,
}
