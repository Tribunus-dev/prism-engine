use std::any::{Any, TypeId};
use std::collections::HashMap;
#[cfg(feature = "lmdb")]
use std::path::Path;
#[cfg(feature = "lmdb")]
use std::sync::Arc;
#[cfg(feature = "lmdb")]
use lmdb::{Cursor, Transaction};

use crate::column::Column;
use crate::component::Component;
use crate::entity::Entity;
use crate::error::WorldError;

/// Type-erased storage for components, indexed by (TypeId, EntityId).
///
/// This is the original HashMap-based store. The newer `ColumnStore` provides
/// generation-aware SparseSet storage; this wrapper provides backward-compatible
/// access on top of columnar storage.
#[derive(Debug)]
pub struct ComponentStore {
    pub(crate) data: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
}

impl Default for ComponentStore {
    fn default() -> Self {
        Self {
            data: HashMap::new(),
        }
    }
}

impl ComponentStore {
    /// Get or create a column for component type T.
    pub fn column_mut<T: Component>(&mut self) -> &mut Column<T> {
        let key = TypeId::of::<Column<T>>();
        self.data
            .entry(key)
            .or_insert_with(|| Box::new(Column::<T>::new()))
            .downcast_mut::<Column<T>>()
            .expect("Column<T> type mismatch in ComponentStore")
    }

    /// Get a shared reference to a column.
    pub fn column<T: Component>(&self) -> Option<&Column<T>> {
        let key = TypeId::of::<Column<T>>();
        self.data.get(&key)?.downcast_ref::<Column<T>>()
    }

    /// Canonical: insert or replace a component.
    pub fn insert_component<T: Component>(
        &mut self,
        entity: Entity,
        value: T,
    ) -> Result<(), WorldError> {
        self.insert::<T>(entity, value);
        Ok(())
    }

    /// Canonical: read a component.
    pub fn component<T: Component>(&self, entity: Entity) -> Result<&T, WorldError> {
        self.get::<T>(entity).ok_or(WorldError::MissingComponent {
            entity,
            type_name: std::any::type_name::<T>(),
        })
    }

    /// Canonical: mutable read of a component.
    pub fn component_mut<T: Component>(&mut self, entity: Entity) -> Result<&mut T, WorldError> {
        self.column_mut::<T>()
            .get_mut(entity)
            .ok_or(WorldError::MissingComponent {
                entity,
                type_name: std::any::type_name::<T>(),
            })
    }

    /// Canonical: check if entity has a component.
    pub fn has_component<T: Component>(&self, entity: Entity) -> bool {
        self.contains::<T>(entity)
    }

    /// Check whether a column exists for the given TypeId.
    pub fn has_column_type(&self, type_id: TypeId) -> bool {
        self.data.contains_key(&type_id)
    }

    pub fn insert<T: Component>(&mut self, entity: Entity, value: T) {
        self.column_mut::<T>().insert(entity, value);
    }

    pub fn get<T: Component>(&self, entity: Entity) -> Option<&T> {
        self.column::<T>()?.get(entity)
    }

    pub fn remove<T: Component>(&mut self, entity: Entity) -> Option<T> {
        self.column_mut::<T>().remove(entity)
    }

    pub fn contains<T: Component>(&self, entity: Entity) -> bool {
        self.column::<T>().map(|c| c.has(entity)).unwrap_or(false)
    }
}

/// Type-erased storage for global resources (not per-entity).
#[derive(Debug)]
pub struct ResourceStore {
    pub(crate) data: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
}

impl Default for ResourceStore {
    fn default() -> Self {
        Self {
            data: HashMap::new(),
        }
    }
}

impl ResourceStore {
    pub fn insert<T: 'static + Send + Sync>(&mut self, resource: T) {
        self.data.insert(TypeId::of::<T>(), Box::new(resource));
    }

    pub fn get<T: 'static + Send + Sync>(&self) -> Option<&T> {
        self.data
            .get(&TypeId::of::<T>())
            .and_then(|b| b.downcast_ref::<T>())
    }

    pub fn contains<T: 'static + Send + Sync>(&self) -> bool {
        self.data.contains_key(&TypeId::of::<T>())
    }

    pub fn remove<T: 'static + Send + Sync>(&mut self) -> Option<T> {
        self.data
            .remove(&TypeId::of::<T>())
            .and_then(|b| b.downcast::<T>().ok().map(|b| *b))
    }

    pub fn get_mut<T: 'static + Send + Sync>(&mut self) -> Option<&mut T> {
        self.data
            .get_mut(&TypeId::of::<T>())
            .and_then(|b| b.downcast_mut::<T>())
    }
}

// ---------------------------------------------------------------------------
// MemoryStore: an ECS Resource for key-value storage backed by in-memory
// HashMap or optionally LMDB (feature = "lmdb").
// ---------------------------------------------------------------------------

/// Errors from memory store operations.
#[derive(Debug)]
pub enum MemoryStoreError {
    /// Key not found in store.
    KeyNotFound(Vec<u8>),
    /// Backend error from the storage layer (LMDB or in-memory).
    Backend(String),
}

impl std::fmt::Display for MemoryStoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryStoreError::KeyNotFound(key) => {
                write!(f, "key not found: {}", String::from_utf8_lossy(key))
            }
            MemoryStoreError::Backend(msg) => write!(f, "backend error: {msg}"),
        }
    }
}

impl std::error::Error for MemoryStoreError {}

/// LMDB backend handle (only compiled when the `lmdb` feature is active).
#[cfg(feature = "lmdb")]
#[derive(Debug)]
struct LmdbBackend {
    env: Arc<lmdb::Environment>,
    db: lmdb::Database,
}

/// In-memory key-value store (ECS Resource).
///
/// Default storage is a `HashMap<Vec<u8>, Vec<u8>>`. When the `lmdb` feature
/// is enabled, `open_lmdb` opens a persistent LMDB environment instead. All
/// operations are available regardless of the backend.
#[derive(Debug)]
pub struct MemoryStore {
    inner: HashMap<Vec<u8>, Vec<u8>>,
    #[cfg(feature = "lmdb")]
    lmdb: Option<LmdbBackend>,
}

impl MemoryStore {
    /// Create a new in-memory store (no persistence).
    pub fn new() -> Self {
        MemoryStore {
            inner: HashMap::new(),
            #[cfg(feature = "lmdb")]
            lmdb: None,
        }
    }

    /// Open or create an LMDB-backed store at the given directory path.
    ///
    /// `map_size` is the maximum size of the database file in bytes.
    #[cfg(feature = "lmdb")]
    pub fn open_lmdb(path: impl AsRef<Path>, map_size: usize) -> Result<Self, MemoryStoreError> {
        use lmdb::DatabaseFlags;

        std::fs::create_dir_all(path.as_ref())
            .map_err(|e| MemoryStoreError::Backend(format!("create dir: {e}")))?;

        let env = lmdb::Environment::new()
            .set_max_dbs(16)
            .set_map_size(map_size)
            .open(path.as_ref())
            .map_err(|e| MemoryStoreError::Backend(e.to_string()))?;

        let db = env
            .create_db(None, DatabaseFlags::default())
            .map_err(|e| MemoryStoreError::Backend(e.to_string()))?;

        Ok(MemoryStore {
            inner: HashMap::new(),
            lmdb: Some(LmdbBackend {
                env: Arc::new(env),
                db,
            }),
        })
    }

    /// Write a key-value pair to the store.
    pub fn put(&mut self, key: &[u8], value: &[u8]) -> Result<(), MemoryStoreError> {
        #[cfg(feature = "lmdb")]
        if let Some(backend) = &self.lmdb {
            let mut txn = backend
                .env
                .begin_rw_txn()
                .map_err(|e| MemoryStoreError::Backend(e.to_string()))?;
            txn.put(backend.db, &key, &value, lmdb::WriteFlags::default())
                .map_err(|e| MemoryStoreError::Backend(e.to_string()))?;
            txn.commit()
                .map_err(|e| MemoryStoreError::Backend(e.to_string()))?;
            return Ok(());
        }

        self.inner.insert(key.to_vec(), value.to_vec());
        Ok(())
    }

    /// Read a value by key.
    pub fn get(&self, key: &[u8]) -> Result<Vec<u8>, MemoryStoreError> {
        #[cfg(feature = "lmdb")]
        if let Some(backend) = &self.lmdb {
            let txn = backend
                .env
                .begin_ro_txn()
                .map_err(|e| MemoryStoreError::Backend(e.to_string()))?;
            match txn.get(backend.db, &key) {
                Ok(val) => return Ok(val.to_vec()),
                Err(lmdb::Error::NotFound) => {
                    return Err(MemoryStoreError::KeyNotFound(key.to_vec()))
                }
                Err(e) => return Err(MemoryStoreError::Backend(e.to_string())),
            }
        }

        self.inner
            .get(key)
            .cloned()
            .ok_or_else(|| MemoryStoreError::KeyNotFound(key.to_vec()))
    }

    /// Delete a key-value pair.
    pub fn delete(&mut self, key: &[u8]) -> Result<(), MemoryStoreError> {
        #[cfg(feature = "lmdb")]
        if let Some(backend) = &self.lmdb {
            let mut txn = backend
                .env
                .begin_rw_txn()
                .map_err(|e| MemoryStoreError::Backend(e.to_string()))?;
            txn.del(backend.db, &key, None)
                .map_err(|e| MemoryStoreError::Backend(e.to_string()))?;
            txn.commit()
                .map_err(|e| MemoryStoreError::Backend(e.to_string()))?;
            return Ok(());
        }

        self.inner.remove(key);
        Ok(())
    }

    /// Query entries with a key prefix. Returns matching keys and their values.
    pub fn query_prefix(&self, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>, MemoryStoreError> {
        #[cfg(feature = "lmdb")]
        if let Some(backend) = &self.lmdb {
            let txn = backend
                .env
                .begin_ro_txn()
                .map_err(|e| MemoryStoreError::Backend(e.to_string()))?;
            let mut cursor = txn
                .open_ro_cursor(backend.db)
                .map_err(|e| MemoryStoreError::Backend(e.to_string()))?;
            let mut results = Vec::new();
            let iter = cursor.iter_from(prefix);
            for result in iter {
                let (key_bytes, val) =
                    result.map_err(|e| MemoryStoreError::Backend(e.to_string()))?;
                if key_bytes.starts_with(prefix) {
                    results.push((key_bytes.to_vec(), val.to_vec()));
                } else {
                    break;
                }
            }
            return Ok(results);
        }

        Ok(self
            .inner
            .iter()
            .filter(|(k, _)| k.starts_with(prefix))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect())
    }
}

impl Default for MemoryStore {
    fn default() -> Self {
        Self::new()
    }
}
