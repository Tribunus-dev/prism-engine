use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use crate::artifact::ArtifactId;
use crate::db::DbManager;

/// Unique identifier for a single tool invocation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInvocationId(pub uuid::Uuid);

impl ToolInvocationId {
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4())
    }
}

impl std::fmt::Display for ToolInvocationId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Status of an operation recorded in the ledger.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvidenceStatus {
    Success,
    Failure(String),
    Partial(Vec<String>),
}

/// A set of named metrics associated with a receipt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricSet {
    pub values: HashMap<String, f64>,
}

impl MetricSet {
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
        }
    }

    pub fn with(mut self, key: &str, value: f64) -> Self {
        self.values.insert(key.to_string(), value);
        self
    }
}

/// A diagnostic message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diagnostic {
    pub severity: String,
    pub message: String,
    pub location: Option<String>,
}

/// A structured receipt recording a tool operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceReceipt {
    pub invocation_id: ToolInvocationId,
    pub tool: String,
    pub operation: String,
    pub inputs: Vec<ArtifactId>,
    pub outputs: Vec<ArtifactId>,
    pub environment: String,
    pub target: Option<String>,
    pub source_revision: Option<String>,
    pub status: EvidenceStatus,
    pub metrics: MetricSet,
    pub diagnostics: Vec<Diagnostic>,
    pub started_at: DateTime<Utc>,
    pub duration_ms: i64,
}

/// Stores and queries evidence receipts. Backed by the shared DbManager's
/// SQLite database.
#[derive(Clone)]
pub struct EvidenceLedger {
    db: Arc<DbManager>,
}

impl EvidenceLedger {
    /// Open the ledger using the shared database connection pool.
    pub fn open(db: Arc<DbManager>) -> Result<Self> {
        db.with_writer(|conn| {
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS evidence_receipts (
                    invocation_id   TEXT PRIMARY KEY,
                    tool            TEXT NOT NULL,
                    operation       TEXT NOT NULL,
                    environment     TEXT NOT NULL DEFAULT '',
                    target          TEXT,
                    source_revision TEXT,
                    status          TEXT NOT NULL DEFAULT 'Success',
                    metrics_json    TEXT NOT NULL DEFAULT '{}',
                    diagnostics_json TEXT NOT NULL DEFAULT '[]',
                    started_at      TEXT NOT NULL,
                    duration_ms     INTEGER NOT NULL DEFAULT 0,
                    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
                );
                CREATE INDEX IF NOT EXISTS idx_evidence_tool ON evidence_receipts(tool);
                CREATE INDEX IF NOT EXISTS idx_evidence_operation ON evidence_receipts(operation);
                CREATE TABLE IF NOT EXISTS evidence_inputs (
                    invocation_id TEXT NOT NULL REFERENCES evidence_receipts(invocation_id) ON DELETE CASCADE,
                    artifact_hash BLOB NOT NULL,
                    role TEXT NOT NULL DEFAULT 'input',
                    seq INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS evidence_outputs (
                    invocation_id TEXT NOT NULL REFERENCES evidence_receipts(invocation_id) ON DELETE CASCADE,
                    artifact_hash BLOB NOT NULL,
                    role TEXT NOT NULL DEFAULT 'output',
                    seq INTEGER NOT NULL
                );\n"
            )?;
            Ok(())
        })?;

        Ok(Self { db })
    }

    /// Record a receipt in the ledger.
    pub fn record(&self, receipt: &EvidenceReceipt) -> Result<()> {
        let id = receipt.invocation_id.0.to_string();
        let metrics = serde_json::to_string(&receipt.metrics.values)?;
        let diagnostics = serde_json::to_string(&receipt.diagnostics)?;
        let status_detail = match &receipt.status {
            EvidenceStatus::Failure(msg) => msg.clone(),
            EvidenceStatus::Partial(msgs) => msgs.join("; "),
            _ => String::new(),
        };

        self.db.with_writer(|conn| {
            conn.execute(
                "INSERT INTO evidence_receipts (invocation_id, tool, operation, environment, target, source_revision, status, metrics_json, diagnostics_json, started_at, duration_ms)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
                rusqlite::params![
                    id, receipt.tool, receipt.operation, receipt.environment,
                    receipt.target, receipt.source_revision, status_detail,
                    metrics, diagnostics, receipt.started_at.to_rfc3339(),
                    receipt.duration_ms,
                ],
            )?;
            Ok(())
        })
    }

    /// Query receipts for a specific tool, optionally filtered by operation.
    pub fn query(
        &self,
        tool: &str,
        operation: Option<&str>,
        limit: usize,
    ) -> Result<Vec<EvidenceReceipt>> {
        self.db.with_reader(|conn| {
            let (sql, params): (String, Vec<Box<dyn rusqlite::types::ToSql>>) = if let Some(op) = operation {
                (
                    "SELECT invocation_id, tool, operation, environment, target, source_revision, status, metrics_json, diagnostics_json, started_at, duration_ms
                     FROM evidence_receipts WHERE tool = ?1 AND operation = ?2 ORDER BY started_at DESC LIMIT ?3".into(),
                    vec![
                        Box::new(tool.to_string()),
                        Box::new(op.to_string()),
                        Box::new(limit as i64),
                    ],
                )
            } else {
                (
                    "SELECT invocation_id, tool, operation, environment, target, source_revision, status, metrics_json, diagnostics_json, started_at, duration_ms
                     FROM evidence_receipts WHERE tool = ?1 ORDER BY started_at DESC LIMIT ?2".into(),
                    vec![
                        Box::new(tool.to_string()),
                        Box::new(limit as i64),
                    ],
                )
            };

            let mut stmt = conn.prepare(&sql)?;
            let rows = stmt.query_map(rusqlite::params_from_iter(params.iter()), |row| {
                let metrics_map: HashMap<String, f64> =
                    serde_json::from_str(&row.get::<_, String>(7)?).unwrap_or_default();
                let diags: Vec<Diagnostic> =
                    serde_json::from_str(&row.get::<_, String>(8)?).unwrap_or_default();
                let status_detail: String = row.get(6)?;
                let status = if status_detail.is_empty() {
                    EvidenceStatus::Success
                } else {
                    EvidenceStatus::Failure(status_detail)
                };

                Ok(EvidenceReceipt {
                    invocation_id: ToolInvocationId(
                        row.get::<_, String>(0)?.parse().unwrap_or_default(),
                    ),
                    tool: row.get(1)?,
                    operation: row.get(2)?,
                    inputs: vec![],
                    outputs: vec![],
                    environment: row.get(3)?,
                    target: row.get(4)?,
                    source_revision: row.get(5)?,
                    status,
                    metrics: MetricSet {
                        values: metrics_map,
                    },
                    diagnostics: diags,
                    started_at: row.get::<_, String>(9)?.parse().unwrap_or_else(|_| Utc::now()),
                    duration_ms: row.get::<_, i64>(10)?,
                })
            })?;

            rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
        })
    }
}

impl crate::storage::EvidenceStore for EvidenceLedger {
    fn record(&self, receipt: &EvidenceReceipt) -> Result<()> {
        self.record(receipt)
    }
    fn query(
        &self,
        tool: &str,
        operation: Option<&str>,
        limit: usize,
    ) -> Result<Vec<EvidenceReceipt>> {
        self.query(tool, operation, limit)
    }
}
