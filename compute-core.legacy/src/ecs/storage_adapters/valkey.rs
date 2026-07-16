//! Valkey storage adapter for coordination records.
#![cfg(feature = "storage-adapters")]

use crate::storage_kernel::{CoordinationPort, CoordinationWorkRecord};
use crate::Result;
use async_trait::async_trait;
use redis::AsyncCommands;

pub struct ValkeyAdapter {
    client: redis::Client,
}

impl ValkeyAdapter {
    pub fn new(url: &str) -> Result<Self> {
        let client = redis::Client::open(url)
            .map_err(|e| crate::Error::new(crate::Status::InternalError, e.to_string()))?;
        Ok(Self { client })
    }

    /// Acquire a distributed work slot.
    pub async fn acquire_work(&self, record: &CoordinationWorkRecord) -> Result<bool> {
        let mut conn = self
            .client
            .get_multiplexed_async_connection()
            .await
            .map_err(|e| crate::Error::new(crate::Status::InternalError, e.to_string()))?;
        let acquired: bool = redis::cmd("SET")
            .arg(&record.key)
            .arg(&record.payload)
            .arg("NX")
            .arg("EX")
            .arg(record.ttl_secs)
            .query_async(&mut conn)
            .await
            .map_err(|e| crate::Error::new(crate::Status::InternalError, e.to_string()))?;
        Ok(acquired)
    }

    /// Release a work slot by key.
    pub async fn release_work(&self, key: &str) -> Result<()> {
        let mut conn = self
            .client
            .get_multiplexed_async_connection()
            .await
            .map_err(|e| crate::Error::new(crate::Status::InternalError, e.to_string()))?;
        redis::cmd("DEL")
            .arg(key)
            .query_async(&mut conn)
            .await
            .map_err(|e| crate::Error::new(crate::Status::InternalError, e.to_string()))?;
        Ok(())
    }
}
