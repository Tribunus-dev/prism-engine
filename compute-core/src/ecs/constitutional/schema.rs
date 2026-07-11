use crate::ecs::constitutional::types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Whether a component type survives across sessions or is session-scoped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComponentDurability {
    Durable,
    Ephemeral,
}

impl Default for ComponentDurability {
    fn default() -> Self {
        Self::Durable
    }
}

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
    #[serde(default)]
    pub durability: ComponentDurability,
    #[serde(skip)]
    pub type_id: Option<std::any::TypeId>,
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

    /// Register a schema with concrete type binding.
    pub fn register_for_type<T: 'static + Send + Sync>(
        &mut self,
        schema_id: ComponentSchemaId,
        version: SchemaVersion,
        name: &str,
        description: &str,
        durability: ComponentDurability,
    ) {
        self.schemas.insert(
            schema_id,
            SchemaEntry {
                schema_id,
                version,
                type_name: name.to_string(),
                type_id: Some(std::any::TypeId::of::<T>()),
                description: description.to_string(),
                durability,
            },
        );
    }

    /// Verify that a schema_id is registered for the given type.
    pub fn verify_type<T: 'static + 'static>(
        &self,
        schema_id: ComponentSchemaId,
    ) -> Result<(), String> {
        match self.get(&schema_id) {
            Some(entry) => match entry.type_id {
                Some(tid) if tid == std::any::TypeId::of::<T>() => Ok(()),
                Some(_) => Err(format!(
                    "schema {:?} registered for different type, not {}",
                    schema_id,
                    std::any::type_name::<T>()
                )),
                None => Ok(()), // legacy entry, no type info
            },
            None => Err(format!("schema {:?} not registered", schema_id)),
        }
    }

    pub fn get(&self, id: &ComponentSchemaId) -> Option<&SchemaEntry> {
        self.schemas.get(id)
    }
}
