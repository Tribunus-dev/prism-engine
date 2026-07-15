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

// ═══════════════════════════════════════════════════════════════════════
//  SchemaCatalogue — deterministic durable schema catalogue
// ═══════════════════════════════════════════════════════════════════════

/// Registration for a single durable schema, provided at startup.
///
/// The `replay_apply` callback uses `CompEntity` for entity identity internally.
/// The canonical entity type [`Entity`](crate::ecs::Entity) `(u64, u32)` is
/// preferred for new code outside the constitutional domain.
#[derive(Clone)]
pub struct DurableSchemaRegistration {
    pub key: SchemaKey,
    pub type_id: std::any::TypeId,
    pub type_name: &'static str,
    pub encode: fn(&dyn std::any::Any) -> Vec<u8>,
    pub decode: fn(&[u8]) -> Box<dyn std::any::Any>,
    pub replay_apply: fn(&mut crate::ecs::World, crate::ecs::CompEntity, &[u8]),
}

impl std::fmt::Debug for DurableSchemaRegistration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DurableSchemaRegistration")
            .field("key", &self.key)
            .field("type_name", &self.type_name)
            .finish()
    }
}

/// Validated, deterministic dataset of all durable schema registrations.
#[derive(Debug, Clone)]
pub struct SchemaCatalogue {
    schemas: std::collections::BTreeMap<SchemaKey, DurableSchemaRegistration>,
    digest: [u8; 32],
}

impl SchemaCatalogue {
    /// Build a catalogue from registrations collected in arbitrary order.
    ///
    /// Validation:
    /// - Duplicate schema keys => error
    /// - One Rust type registered under incompatible keys => error
    /// - Version zero => error
    /// - Reserved namespaces => error
    pub fn build(
        registrations: impl IntoIterator<Item = DurableSchemaRegistration>,
    ) -> Result<Self, String> {
        let mut by_key = std::collections::BTreeMap::new();
        let mut by_type: std::collections::HashMap<std::any::TypeId, SchemaKey> =
            std::collections::HashMap::new();

        for reg in registrations {
            if reg.key.version == 0 {
                return Err(format!(
                    "schema {}:{} version 0 is reserved (use >= 1)",
                    reg.key.namespace, reg.key.id
                ));
            }
            if by_key.contains_key(&reg.key) {
                return Err(format!(
                    "duplicate schema key {}:{}:{}",
                    reg.key.namespace, reg.key.id, reg.key.version
                ));
            }
            if let Some(existing) = by_type.get(&reg.type_id) {
                if existing.namespace != reg.key.namespace || existing.id != reg.key.id {
                    return Err(format!(
                        "type {} registered under both {}:{} and {}:{}",
                        reg.type_name,
                        existing.namespace,
                        existing.id,
                        reg.key.namespace,
                        reg.key.id
                    ));
                }
            }
            if reg.key.namespace.starts_with('_') {
                return Err(format!(
                    "reserved namespace prefix '_' used by {}:{}",
                    reg.key.namespace, reg.key.id
                ));
            }
            by_type.insert(reg.type_id, reg.key);
            by_key.insert(reg.key, reg);
        }

        // Deterministic digest over sorted registrations
        use blake3::Hasher;
        let mut hasher = Hasher::new();
        hasher.update(b"prism.schema.catalogue.v1");
        for (key, _reg) in &by_key {
            hasher.update(key.namespace.as_bytes());
            hasher.update(&key.id.to_le_bytes());
            hasher.update(&key.version.to_le_bytes());
        }
        let digest = hasher.finalize().into();

        Ok(Self {
            schemas: by_key,
            digest,
        })
    }

    /// Get the deterministic catalogue digest.
    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }

    /// Look up a registration by schema key.
    pub fn registration(&self, key: &SchemaKey) -> Option<&DurableSchemaRegistration> {
        self.schemas.get(key)
    }

    /// Look up a replay applier for the given schema key.
    ///
    /// Returns a function that applies a persisted component value to an
    /// entity in the world. Internally uses `CompEntity`; the canonical
    /// [`Entity`](crate::ecs::Entity) `(u64, u32)` type is preferred for
    /// new code outside the constitutional domain.
    pub fn replay_applier(
        &self,
        key: &SchemaKey,
    ) -> Option<fn(&mut crate::ecs::World, crate::ecs::CompEntity, &[u8])> {
        self.schemas.get(key).map(|reg| reg.replay_apply)
    }

    /// Number of registered schemas.
    pub fn len(&self) -> usize {
        self.schemas.len()
    }

    /// Iterate over all registrations in sorted key order.
    pub fn iter(&self) -> impl Iterator<Item = &DurableSchemaRegistration> {
        self.schemas.values()
    }

    /// Check whether a schema key is registered in this catalogue.
    pub fn contains(&self, key: &SchemaKey) -> bool {
        self.schemas.contains_key(key)
    }

    /// Build an empty catalogue with a deterministic digest.
    pub fn empty() -> Self {
        use blake3::Hasher;
        let mut hasher = Hasher::new();
        hasher.update(b"prism.schema.catalogue.v1");
        let digest = hasher.finalize().into();
        Self {
            schemas: std::collections::BTreeMap::new(),
            digest,
        }
    }
}
