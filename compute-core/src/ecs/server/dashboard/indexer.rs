#![cfg(feature = "server-dashboard")]

use std::fmt;
use std::path::Path;

use sha2::{Digest, Sha256};
use uuid::Uuid;
use walkdir::WalkDir;

use crate::ecs::cimage::{
    CImageArtifactKind, CImageLoader, CImagePayloadEntry, CImagePayloadKind, CImagePayloadRef,
    CImageTensorEntry, LoadedCImageV0,
};
use crate::ecs::server::dashboard::models::DashboardCImageSummary;

// ---------------------------------------------------------------------------
// PostgreSQL DDL — executed at connection time to ensure tables exist.
// These match the schema_design.md PostgreSQL types (not the SQLite version in
// schema.rs) since EvidenceIndexer uses sqlx_postgres::PgPool.
// ---------------------------------------------------------------------------

/// `cimage_artifacts` — one row per scanned `.cimage` file.
pub const CREATE_ARTIFACTS: &str = "CREATE TABLE IF NOT EXISTS cimage_artifacts (
    digest TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    artifact_kind TEXT NOT NULL,
    model_family TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    tensor_count INTEGER NOT NULL DEFAULT 0,
    receipt_count INTEGER NOT NULL DEFAULT 0,
    validation_status TEXT NOT NULL DEFAULT 'Unknown',
    compiler_policy_digest TEXT,
    hardware_profile TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    manifest_json JSONB
);";

/// `cimage_tensors` — one row per tensor entry in the manifest.
pub const CREATE_TENSORS: &str = "CREATE TABLE IF NOT EXISTS cimage_tensors (
    id BIGSERIAL PRIMARY KEY,
    artifact_digest TEXT NOT NULL REFERENCES cimage_artifacts(digest),
    tensor_key TEXT NOT NULL,
    tensor_class TEXT NOT NULL,
    codec TEXT NOT NULL,
    group_size INTEGER,
    effective_bpw NUMERIC(10,4),
    logical_shape INTEGER[],
    payload_size BIGINT,
    promotion_status TEXT NOT NULL DEFAULT 'ResearchOnly',
    UNIQUE(artifact_digest, tensor_key)
);
CREATE INDEX IF NOT EXISTS idx_tensors_artifact ON cimage_tensors(artifact_digest);
CREATE INDEX IF NOT EXISTS idx_tensors_codec ON cimage_tensors(codec);
CREATE INDEX IF NOT EXISTS idx_tensors_class ON cimage_tensors(tensor_class);";

/// `admission_receipts` — quantization trial results for a tensor.
pub const CREATE_ADMISSIONS: &str = "CREATE TABLE IF NOT EXISTS admission_receipts (
    receipt_id UUID PRIMARY KEY,
    artifact_digest TEXT NOT NULL REFERENCES cimage_artifacts(digest),
    tensor_key TEXT NOT NULL,
    codec TEXT NOT NULL,
    group_size INTEGER NOT NULL,
    effective_bpw NUMERIC(10,4),
    zero_fraction NUMERIC(8,6),
    neg_fraction NUMERIC(8,6),
    pos_fraction NUMERIC(8,6),
    scale_mean NUMERIC(12,6),
    scale_std NUMERIC(12,6),
    operator_nrmse NUMERIC(12,8),
    output_cosine NUMERIC(10,8),
    activation_shift_l2 NUMERIC(12,8),
    deadzone_collapse BOOLEAN DEFAULT FALSE,
    rescue_required BOOLEAN DEFAULT FALSE,
    rescue_codec TEXT,
    promotion_status TEXT NOT NULL,
    raw_json JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_admissions_artifact ON admission_receipts(artifact_digest);
CREATE INDEX IF NOT EXISTS idx_admissions_codec ON admission_receipts(codec);
CREATE INDEX IF NOT EXISTS idx_admissions_status ON admission_receipts(promotion_status);";

/// `execution_receipts` — Metal dispatch timing and validation results.
pub const CREATE_EXECUTION: &str = "CREATE TABLE IF NOT EXISTS execution_receipts (
    receipt_id UUID PRIMARY KEY,
    artifact_digest TEXT REFERENCES cimage_artifacts(digest),
    tensor_key TEXT NOT NULL,
    kernel_name TEXT NOT NULL,
    backend TEXT NOT NULL,
    command_buffer_ms NUMERIC(12,4),
    effective_bandwidth_gbps NUMERIC(10,4),
    metal_vs_cpu_nrmse NUMERIC(12,8),
    validation_passed BOOLEAN DEFAULT FALSE,
    raw_json JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);";

/// Run all CREATE TABLE statements in dependency order.
pub async fn ensure_tables(pool: &sqlx_postgres::PgPool) -> Result<(), sqlx_postgres::Error> {
    let statements = [
        CREATE_ARTIFACTS,
        CREATE_TENSORS,
        CREATE_ADMISSIONS,
        CREATE_EXECUTION,
    ];
    for stmt in &statements {
        sqlx_postgres::raw_sql(stmt).execute(pool).await?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during evidence indexing.
#[derive(Debug, Clone)]
pub enum DashboardError {
    Db(String),
    Io(String),
    Parse(String),
    Cimage(String),
}

impl fmt::Display for DashboardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DashboardError::Db(msg) => write!(f, "database error: {msg}"),
            DashboardError::Io(msg) => write!(f, "I/O error: {msg}"),
            DashboardError::Parse(msg) => write!(f, "parse error: {msg}"),
            DashboardError::Cimage(msg) => write!(f, "cimage error: {msg}"),
        }
    }
}

impl std::error::Error for DashboardError {}

impl From<sqlx_postgres::Error> for DashboardError {
    fn from(e: sqlx_postgres::Error) -> Self {
        DashboardError::Db(e.to_string())
    }
}

impl From<std::io::Error> for DashboardError {
    fn from(e: std::io::Error) -> Self {
        DashboardError::Io(e.to_string())
    }
}

impl From<crate::ecs::cimage::CImageError> for DashboardError {
    fn from(e: crate::ecs::cimage::CImageError) -> Self {
        DashboardError::Cimage(e.to_string())
    }
}

impl From<serde_json::Error> for DashboardError {
    fn from(e: serde_json::Error) -> Self {
        DashboardError::Parse(e.to_string())
    }
}

// ---------------------------------------------------------------------------
// EvidenceIndexer
// ---------------------------------------------------------------------------

/// The evidence indexer ties together PostgreSQL (persistence), DuckDB
/// (analytical queries), and Valkey (caching) for the dashboard backend.
pub struct EvidenceIndexer {
    pub pool: sqlx_postgres::PgPool,
    pub duckdb: duckdb::Connection,
    pub valkey: Option<fred::clients::RedisClient>,
}

impl EvidenceIndexer {
    /// Create a new `EvidenceIndexer`, connecting to PostgreSQL and (optionally)
    /// Valkey, opening an in-memory DuckDB connection, and ensuring tables exist.
    pub async fn create_indexer(
        pg_conn_str: &str,
        valkey_url: Option<&str>,
    ) -> Result<Self, DashboardError> {
        let pool = sqlx_postgres::PgPool::connect(pg_conn_str).await?;

        // Ensure all dashboard tables exist.
        ensure_tables(&pool).await.map_err(DashboardError::Db)?;

        // In-memory DuckDB for analytical views.
        let duckdb = duckdb::Connection::open_in_memory()
            .map_err(|e| DashboardError::Db(format!("duckdb connect: {e}")))?;

        // Optional Valkey connection.
        let valkey = if let Some(url) = valkey_url {
            let config = fred::types::RedisConfig::from_url(url)
                .map_err(|e| DashboardError::Db(format!("valkey config: {e}")))?;
            let client = fred::clients::RedisClient::new(config, None, None, None);
            client
                .connect()
                .await
                .map_err(|e| DashboardError::Db(format!("valkey connect: {e}")))?;
            client
                .wait_for_connect()
                .await
                .map_err(|e| DashboardError::Db(format!("valkey wait: {e}")))?;
            Some(client)
        } else {
            None
        };

        Ok(EvidenceIndexer {
            pool,
            duckdb,
            valkey,
        })
    }

    /// Recursively scan `dir` for `*.cimage` files, load each one, extract its
    /// manifest and receipt data, and insert rows into PostgreSQL.
    ///
    /// Returns a summary for each successfully indexed artifact.
    pub async fn scan_and_index(
        &self,
        dir: &Path,
    ) -> Result<Vec<DashboardCImageSummary>, DashboardError> {
        let mut summaries = Vec::new();

        let entries: Vec<_> = WalkDir::new(dir)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
            .filter(|e| {
                e.path()
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext| ext.eq_ignore_ascii_case("cimage"))
                    .unwrap_or(false)
            })
            .map(|e| e.path().to_path_buf())
            .collect();

        for path in &entries {
            match self.index_single_cimage(path).await {
                Ok(summary) => summaries.push(summary),
                Err(e) => {
                    // Log but continue with remaining files.
                    eprintln!("[dashboard/indexer] skipping {}: {e}", path.display());
                }
            }
        }

        Ok(summaries)
    }

    // ── internal helpers ─────────────────────────────────────────────────

    /// Index a single cimage file: load, extract, and persist.
    async fn index_single_cimage(
        &self,
        path: &Path,
    ) -> Result<DashboardCImageSummary, DashboardError> {
        let loaded = CImageLoader::load_v0(path)?;

        // Compute digest over header bytes.
        let header_bytes = &loaded.raw_file_bytes[..loaded.header.header_len as usize];
        let digest = format!("{:x}", Sha256::digest(header_bytes));

        // ── Insert into cimage_artifacts ──────────────────────────────
        let manifest_json =
            serde_json::to_value(&loaded.manifest).map_err(DashboardError::Parse)?;
        let artifact_kind_str = artifact_kind_to_str(loaded.manifest.artifact_kind);

        sqlx_postgres::query(
            r#"INSERT INTO cimage_artifacts
               (digest, path, artifact_kind, model_family, schema_version,
                tensor_count, receipt_count, compiler_policy_digest,
                hardware_profile, manifest_json)
               VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
               ON CONFLICT (digest) DO UPDATE SET
                 path = EXCLUDED.path,
                 tensor_count = EXCLUDED.tensor_count,
                 receipt_count = EXCLUDED.receipt_count,
                 manifest_json = EXCLUDED.manifest_json"#,
        )
        .bind(&digest)
        .bind(path.to_string_lossy().as_ref())
        .bind(&artifact_kind_str)
        .bind(&loaded.manifest.model_family)
        .bind(loaded.manifest.schema_version as i32)
        .bind(loaded.manifest.tensors.len() as i32)
        .bind(loaded.manifest.receipts.len() as i32)
        .bind(&loaded.manifest.compiler_policy_digest)
        .bind(format!("{:?}", loaded.manifest.layout_profile))
        .bind(&manifest_json)
        .execute(&self.pool)
        .await?;

        // ── Insert each tensor ───────────────────────────────────────
        for tensor in &loaded.manifest.tensors {
            let group_size = tensor.physical_layout.group_size;
            let codec_str = format!("{:?}", tensor.codec);
            let logical_shape: Vec<i32> = tensor.logical_shape.iter().map(|&v| v as i32).collect();

            // Compute payload_size by looking up the payload entry.
            let payload_size = resolve_payload_size(&loaded, tensor);

            sqlx_postgres::query(
                r#"INSERT INTO cimage_tensors
                   (artifact_digest, tensor_key, tensor_class, codec, group_size,
                    logical_shape, payload_size)
                   VALUES ($1,$2,$3,$4,$5,$6,$7)
                   ON CONFLICT (artifact_digest, tensor_key) DO UPDATE SET
                     tensor_class = EXCLUDED.tensor_class,
                     codec = EXCLUDED.codec,
                     group_size = EXCLUDED.group_size,
                     payload_size = EXCLUDED.payload_size"#,
            )
            .bind(&digest)
            .bind(&tensor.tensor_key)
            .bind(&tensor.tensor_class)
            .bind(&codec_str)
            .bind(group_size as i32)
            .bind(&logical_shape)
            .bind(payload_size)
            .execute(&self.pool)
            .await?;
        }

        // ── Extract and insert receipt payloads ──────────────────────
        let receipt_entries: Vec<&CImagePayloadEntry> = loaded
            .payload_directory
            .payloads
            .iter()
            .filter(|e| {
                matches!(
                    e.payload_kind,
                    CImagePayloadKind::ReceiptJson | CImagePayloadKind::TernaryAdmissionReceiptJson
                )
            })
            .collect();

        for entry in &receipt_entries {
            let start = entry.offset as usize;
            let end = start + entry.len as usize;
            let receipt_bytes = &loaded.payload_blob[start..end];
            let receipt_json: serde_json::Value =
                serde_json::from_slice(receipt_bytes).map_err(DashboardError::Parse)?;

            if is_admission_receipt(&receipt_json) {
                insert_admission_receipt(&self.pool, &digest, &receipt_json).await?;
            } else {
                insert_execution_receipt(&self.pool, &digest, &receipt_json).await?;
            }
        }

        Ok(DashboardCImageSummary {
            digest,
            path: path.to_string_lossy().to_string(),
            artifact_kind: artifact_kind_str,
            model_family: loaded.manifest.model_family.clone(),
            schema_version: loaded.manifest.schema_version as i32,
            tensor_count: loaded.manifest.tensors.len() as i32,
            receipt_count: loaded.manifest.receipts.len() as i32,
            validation_status: "Unknown".to_string(),
            compiler_policy_digest: Some(loaded.manifest.compiler_policy_digest),
            hardware_profile: Some(format!("{:?}", loaded.manifest.layout_profile)),
            created_at: String::new(),
        })
    }
}

// ---------------------------------------------------------------------------
// Free-standing insert helpers (no &self needed — only &PgPool)
// ---------------------------------------------------------------------------

/// Insert an admission receipt row from its JSON value.
async fn insert_admission_receipt(
    pool: &sqlx_postgres::PgPool,
    artifact_digest: &str,
    json: &serde_json::Value,
) -> Result<(), DashboardError> {
    let receipt_id: Uuid = json
        .get("receipt_id")
        .and_then(|v| v.as_str())
        .and_then(|s| Uuid::parse_str(s).ok())
        .unwrap_or_else(Uuid::new_v4);

    let tensor_key = json
        .get("tensor_key")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    let codec = json
        .get("codec")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    let group_size = json.get("group_size").and_then(|v| v.as_i64()).unwrap_or(0) as i32;

    let effective_bpw = json.get("effective_bpw").and_then(|v| v.as_f64());
    let zero_fraction = json.get("zero_fraction").and_then(|v| v.as_f64());
    let neg_fraction = json.get("neg_fraction").and_then(|v| v.as_f64());
    let pos_fraction = json.get("pos_fraction").and_then(|v| v.as_f64());
    let scale_mean = json.get("scale_mean").and_then(|v| v.as_f64());
    let scale_std = json.get("scale_std").and_then(|v| v.as_f64());
    let operator_nrmse = json.get("operator_nrmse").and_then(|v| v.as_f64());
    let output_cosine = json.get("output_cosine").and_then(|v| v.as_f64());
    let activation_shift_l2 = json.get("activation_shift_l2").and_then(|v| v.as_f64());
    let deadzone_collapse = json
        .get("deadzone_collapse")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let rescue_required = json
        .get("rescue_required")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let rescue_codec = json.get("rescue_codec").and_then(|v| v.as_str());
    let promotion_status = json
        .get("promotion_status")
        .and_then(|v| v.as_str())
        .unwrap_or("ResearchOnly");

    sqlx_postgres::query(
        r#"INSERT INTO admission_receipts
           (receipt_id, artifact_digest, tensor_key, codec, group_size,
            effective_bpw, zero_fraction, neg_fraction, pos_fraction,
            scale_mean, scale_std, operator_nrmse, output_cosine,
            activation_shift_l2, deadzone_collapse, rescue_required,
            rescue_codec, promotion_status, raw_json)
           VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19)
           ON CONFLICT (receipt_id) DO UPDATE SET
             operator_nrmse = EXCLUDED.operator_nrmse,
             output_cosine = EXCLUDED.output_cosine,
             promotion_status = EXCLUDED.promotion_status"#,
    )
    .bind(receipt_id)
    .bind(artifact_digest)
    .bind(tensor_key)
    .bind(codec)
    .bind(group_size)
    .bind(effective_bpw)
    .bind(zero_fraction)
    .bind(neg_fraction)
    .bind(pos_fraction)
    .bind(scale_mean)
    .bind(scale_std)
    .bind(operator_nrmse)
    .bind(output_cosine)
    .bind(activation_shift_l2)
    .bind(deadzone_collapse)
    .bind(rescue_required)
    .bind(rescue_codec)
    .bind(promotion_status)
    .bind(json)
    .execute(pool)
    .await?;

    Ok(())
}

/// Insert an execution receipt row from its JSON value.
async fn insert_execution_receipt(
    pool: &sqlx_postgres::PgPool,
    artifact_digest: &str,
    json: &serde_json::Value,
) -> Result<(), DashboardError> {
    let receipt_id: Uuid = json
        .get("receipt_id")
        .and_then(|v| v.as_str())
        .and_then(|s| Uuid::parse_str(s).ok())
        .unwrap_or_else(Uuid::new_v4);

    let tensor_key = json
        .get("tensor_key")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    let kernel_name = json
        .get("kernel_name")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    let backend = json
        .get("backend")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    let command_buffer_ms = json.get("command_buffer_ms").and_then(|v| v.as_f64());
    let effective_bandwidth_gbps = json
        .get("effective_bandwidth_gbps")
        .and_then(|v| v.as_f64());
    let metal_vs_cpu_nrmse = json.get("metal_vs_cpu_nrmse").and_then(|v| v.as_f64());
    let validation_passed = json
        .get("validation_passed")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    sqlx_postgres::query(
        r#"INSERT INTO execution_receipts
           (receipt_id, artifact_digest, tensor_key, kernel_name, backend,
            command_buffer_ms, effective_bandwidth_gbps, metal_vs_cpu_nrmse,
            validation_passed, raw_json)
           VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
           ON CONFLICT (receipt_id) DO UPDATE SET
             command_buffer_ms = EXCLUDED.command_buffer_ms,
             validation_passed = EXCLUDED.validation_passed"#,
    )
    .bind(receipt_id)
    .bind(artifact_digest)
    .bind(tensor_key)
    .bind(kernel_name)
    .bind(backend)
    .bind(command_buffer_ms)
    .bind(effective_bandwidth_gbps)
    .bind(metal_vs_cpu_nrmse)
    .bind(validation_passed)
    .bind(json)
    .execute(pool)
    .await?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Convert CImageArtifactKind to a human-readable string.
fn artifact_kind_to_str(kind: CImageArtifactKind) -> String {
    match kind {
        CImageArtifactKind::SyntheticShard => "SyntheticShard".to_string(),
        CImageArtifactKind::ModelShard => "ModelShard".to_string(),
        CImageArtifactKind::FullModel => "FullModel".to_string(),
        CImageArtifactKind::AssistantGraphProof => "AssistantGraphProof".to_string(),
    }
}

/// Resolve the payload size (in bytes) for a tensor entry's payload ref.
fn resolve_payload_size(loaded: &LoadedCImageV0, tensor: &CImageTensorEntry) -> Option<i64> {
    match &tensor.payload_ref {
        CImagePayloadRef::Single { payload_id } => loaded
            .payload_directory
            .payloads
            .iter()
            .find(|e| e.payload_id == *payload_id)
            .map(|e| e.len as i64),
        CImagePayloadRef::MixedPrecision {
            base_payload_id,
            override_table_payload_id,
            sidecar_payload_ids,
        } => {
            let mut total: i64 = 0;
            let all_ids = std::iter::once(base_payload_id.as_str())
                .chain(std::iter::once(override_table_payload_id.as_str()))
                .chain(sidecar_payload_ids.iter().map(|s| s.as_str()));
            for pid in all_ids {
                if let Some(entry) = loaded
                    .payload_directory
                    .payloads
                    .iter()
                    .find(|e| e.payload_id == pid)
                {
                    total += entry.len as i64;
                }
            }
            if total > 0 {
                Some(total)
            } else {
                None
            }
        }
    }
}

/// Determine whether a receipt JSON payload represents an admission receipt
/// (presence of `operator_nrmse`) or an execution receipt (execution path).
fn is_admission_receipt(json: &serde_json::Value) -> bool {
    json.get("operator_nrmse").is_some()
}
