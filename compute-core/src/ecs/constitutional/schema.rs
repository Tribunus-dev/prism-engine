use crate::ecs::constitutional::types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Schema registry skeleton. Not yet wired into component operations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SchemaRegistry {
    schemas: HashMap<ComponentSchemaId, SchemaEntry>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SchemaEntry {
    pub schema_id: ComponentSchemaId,
    pub version: SchemaVersion,
    pub type_name: String,
    pub description: String,
}

impl SchemaRegistry {
    pub fn new() -> Self {
        Self {
            schemas: HashMap::new(),
        }
    }

    pub fn register(&mut self, entry: SchemaEntry) {
        self.schemas.insert(entry.schema_id, entry);
    }

    pub fn get(&self, id: &ComponentSchemaId) -> Option<&SchemaEntry> {
        self.schemas.get(id)
    }
}
