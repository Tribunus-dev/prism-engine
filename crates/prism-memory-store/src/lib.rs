//! LMDB-backed memory stores for agent state, facts, observations, and documents.
//!
//! Ported from PrismCore's LMDB stores. Uses lmdb-rkv for the zero-copy KV store.

use lmdb::{Cursor, Database, DatabaseFlags, Environment, Transaction, WriteFlags};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;

/// Errors from memory store operations.
#[derive(Debug, thiserror::Error)]
pub enum MemoryStoreError {
    #[error("LMDB error: {0}")]
    Lmdb(#[from] lmdb::Error),
    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("Key not found: {0}")]
    KeyNotFound(String),
}

/// Value types stored in LMDB databases.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub key: String,
    pub value: Vec<u8>,
    pub content_type: String,
    pub created_at: String,
}

/// LMDB-backed memory store — durable, zero-copy, thread-safe.
pub struct LmdbMemoryStore {
    env: Arc<Environment>,
    db: Database,
}

impl LmdbMemoryStore {
    /// Open or create an LMDB environment at the given directory path.
    ///
    /// `map_size` is the maximum size of the database in bytes.
    pub fn open<P: AsRef<Path>>(path: P, map_size: usize) -> Result<Self, MemoryStoreError> {
        std::fs::create_dir_all(path.as_ref())
            .map_err(|e| MemoryStoreError::KeyNotFound(format!("create dir: {e}")))?;

        let env = Environment::new()
            .set_max_dbs(16)
            .set_map_size(map_size)
            .open(path.as_ref())?;

        let db = env.create_db(None, DatabaseFlags::default())?;

        Ok(Self {
            env: Arc::new(env),
            db,
        })
    }

    /// Write a key-value pair to the store. Key and value are stored as raw bytes.
    ///
    /// Accepts any key/value types that implement `AsRef<[u8]>`, e.g. `&[u8]`, `&str`, `Vec<u8>`.
    pub fn put<K: AsRef<[u8]>, D: AsRef<[u8]>>(
        &self,
        key: &K,
        value: &D,
    ) -> Result<(), MemoryStoreError> {
        let mut txn = self.env.begin_rw_txn()?;
        txn.put(self.db, key, value, WriteFlags::default())?;
        txn.commit()?;
        Ok(())
    }

    /// Read a value by key. Requires a sized key type.
    pub fn get<K: AsRef<[u8]>>(&self, key: &K) -> Result<Vec<u8>, MemoryStoreError> {
        let txn = self.env.begin_ro_txn()?;
        match txn.get(self.db, key) {
            Ok(val) => Ok(val.to_vec()),
            Err(lmdb::Error::NotFound) => Err(MemoryStoreError::KeyNotFound(
                String::from_utf8_lossy(key.as_ref()).into_owned(),
            )),
            Err(e) => Err(e.into()),
        }
    }

    /// Delete a key-value pair. Requires a sized key type.
    pub fn delete<K: AsRef<[u8]>>(&self, key: &K) -> Result<(), MemoryStoreError> {
        let mut txn = self.env.begin_rw_txn()?;
        txn.del(self.db, key, None)?;
        txn.commit()?;
        Ok(())
    }

    /// Query entries with a key prefix. Returns matching keys and their values.
    pub fn query_prefix(&self, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>, MemoryStoreError> {
        let txn = self.env.begin_ro_txn()?;
        let mut cursor = txn.open_ro_cursor(self.db)?;
        let mut results = Vec::new();

        let iter = cursor.iter_from(prefix);
        for result in iter {
            let (key_bytes, val) = result?;
            if key_bytes.starts_with(prefix) {
                results.push((key_bytes.to_vec(), val.to_vec()));
            } else {
                break;
            }
        }

        Ok(results)
    }

    /// Return the LMDB environment for advanced usage.
    pub fn env(&self) -> &Arc<Environment> {
        &self.env
    }

    /// Return the default database handle.
    pub fn db(&self) -> Database {
        self.db
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_put_get_delete() {
        let dir = tempdir().unwrap();
        let store = LmdbMemoryStore::open(dir.path(), 1_048_576).unwrap();

        store.put(b"test:key1", b"value1").unwrap();
        assert_eq!(store.get(b"test:key1").unwrap(), b"value1");

        store.delete(b"test:key1").unwrap();
        assert!(matches!(
            store.get(b"test:key1"),
            Err(MemoryStoreError::KeyNotFound(_))
        ));
    }

    #[test]
    fn test_query_prefix() {
        let dir = tempdir().unwrap();
        let store = LmdbMemoryStore::open(dir.path(), 1_048_576).unwrap();

        store.put(b"facts:apple", b"fruit").unwrap();
        store.put(b"facts:banana", b"fruit").unwrap();
        store.put(b"facts:carrot", b"vegetable").unwrap();
        store.put(b"observations:sky", b"blue").unwrap();

        // Facts prefix
        let facts = store.query_prefix(b"facts:").unwrap();
        assert_eq!(facts.len(), 3);

        // Non-existent prefix
        let empty = store.query_prefix(b"nonexistent:").unwrap();
        assert!(empty.is_empty());
    }
}
