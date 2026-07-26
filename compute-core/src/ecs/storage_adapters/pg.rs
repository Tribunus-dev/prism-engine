//! Postgres storage adapter for durable authority records.
#![cfg(feature = "storage-adapters")]

use crate::storage_kernel::{DurableAuthorityPort, DurableReceiptRecord};
use crate::Result;
use async_trait::async_trait;
use tokio_postgres::{Client, NoTls};

pub struct PgAdapter {
    client: Client,
}

impl PgAdapter {
    pub async fn connect(config: &str) -> Result<Self> {
        let (client, connection) = tokio_postgres::connect(config, NoTls)
            .await
            .map_err(|e| crate::Error::new(crate::Status::InternalError, e.to_string()))?;
        tokio::spawn(async move {
            if let Err(e) = connection.await {
                eprintln!("Postgres connection error: {}", e);
            }
        });
        Ok(Self { client })
    }

    /// Store a receipt record in the authority ledger.
    pub async fn store_receipt(&self, record: &DurableReceiptRecord) -> Result<()> {
        self.client
            .execute(
                "INSERT INTO authority_receipts (id, kind, authority, payload, issued_at, expires_at)
                 VALUES ($1, $2, $3, $4, $5, $6)
                 ON CONFLICT (id) DO NOTHING",
                &[
                    &record.id,
                    &record.kind,
                    &record.authority,
                    &record.payload,
                    &record.issued_at,
                    &record.expires_at,
                ],
            )
            .await
            .map_err(|e| crate::Error::new(crate::Status::InternalError, e.to_string()))?;
        Ok(())
    }

    /// Look up a receipt by unique identifier.
    pub async fn lookup_receipt(&self, id: &str) -> Result<Option<DurableReceiptRecord>> {
        let row = self
            .client
            .query_opt(
                "SELECT id, kind, authority, payload, issued_at, expires_at
                 FROM authority_receipts WHERE id = $1",
                &[&id],
            )
            .await
            .map_err(|e| crate::Error::new(crate::Status::InternalError, e.to_string()))?;
        Ok(row.map(|r| DurableReceiptRecord {
            id: r.get(0),
            kind: r.get(1),
            authority: r.get(2),
            payload: r.get(3),
            issued_at: r.get(4),
            expires_at: r.get(5),
        }))
    }
}
