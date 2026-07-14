//! Optional distributed storage profile for the MCP daemon.
//!
//! SQLite remains the default local profile. The trifecta profile makes the
//! ownership boundary explicit: PostgreSQL is durable authority, Valkey is
//! coordination, and DuckDB is analytical projection.

#[derive(Debug, Clone)]
pub struct BackendConfig {
    pub profile: String,
    pub postgres_url: Option<String>,
    pub valkey_url: Option<String>,
    pub duckdb_path: Option<String>,
}

impl BackendConfig {
    pub fn from_env() -> Self {
        Self {
            profile: std::env::var("PRISM_MCPD_STORAGE").unwrap_or_else(|_| "trifecta".into()),
            postgres_url: std::env::var("PRISM_MCPD_POSTGRES_URL").ok(),
            valkey_url: std::env::var("PRISM_MCPD_VALKEY_URL").ok(),
            duckdb_path: std::env::var("PRISM_MCPD_DUCKDB_PATH").ok(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BackendHealth {
    pub profile: &'static str,
    pub postgres: BackendStatus,
    pub valkey: BackendStatus,
    pub duckdb: BackendStatus,
}

#[derive(Debug, Clone)]
pub enum BackendStatus {
    Disabled,
    Healthy,
    Failed(String),
}

impl BackendHealth {
    pub fn local() -> Self {
        Self {
            profile: "sqlite-local",
            postgres: BackendStatus::Disabled,
            valkey: BackendStatus::Disabled,
            duckdb: BackendStatus::Disabled,
        }
    }

    pub fn as_json(&self) -> serde_json::Value {
        serde_json::json!({
            "profile": self.profile,
            "postgres": status_json(&self.postgres),
            "valkey": status_json(&self.valkey),
            "duckdb": status_json(&self.duckdb),
        })
    }
}

fn status_json(status: &BackendStatus) -> serde_json::Value {
    match status {
        BackendStatus::Disabled => serde_json::json!({"configured": false, "healthy": false}),
        BackendStatus::Healthy => serde_json::json!({"configured": true, "healthy": true}),
        BackendStatus::Failed(error) => {
            serde_json::json!({"configured": true, "healthy": false, "error": error})
        }
    }
}

#[cfg(feature = "trifecta")]
pub fn validate(config: &BackendConfig) -> anyhow::Result<BackendHealth> {
    if config.profile == "sqlite" {
        return Ok(BackendHealth::local());
    }
    if config.postgres_url.is_none() || config.valkey_url.is_none() || config.duckdb_path.is_none()
    {
        anyhow::bail!(
            "trifecta is the production storage default; set PRISM_MCPD_POSTGRES_URL, PRISM_MCPD_VALKEY_URL, and PRISM_MCPD_DUCKDB_PATH, or explicitly set PRISM_MCPD_STORAGE=sqlite for local/test mode"
        );
    }
    let mut health = BackendHealth {
        profile: "postgres-valkey-duckdb",
        postgres: BackendStatus::Disabled,
        valkey: BackendStatus::Disabled,
        duckdb: BackendStatus::Disabled,
    };

    if let Some(url) = &config.postgres_url {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        let result = runtime.block_on(async {
            let (client, connection) = tokio_postgres::connect(url, tokio_postgres::NoTls).await?;
            tokio::spawn(async move {
                let _ = connection.await;
            });
            client.simple_query("SELECT 1").await
        });
        health.postgres = match result {
            Ok(_) => BackendStatus::Healthy,
            Err(error) => BackendStatus::Failed(error.to_string()),
        };
    }

    if let Some(url) = &config.valkey_url {
        health.valkey = match redis::Client::open(url.as_str())
            .and_then(|client| client.get_connection())
            .and_then(|mut connection| redis::cmd("PING").query::<String>(&mut connection))
        {
            Ok(_) => BackendStatus::Healthy,
            Err(error) => BackendStatus::Failed(error.to_string()),
        };
    }

    if let Some(path) = &config.duckdb_path {
        health.duckdb = match duckdb::Connection::open(path)
            .and_then(|connection| connection.execute_batch("SELECT 1"))
        {
            Ok(_) => BackendStatus::Healthy,
            Err(error) => BackendStatus::Failed(error.to_string()),
        };
    }

    Ok(health)
}

#[cfg(feature = "trifecta")]
pub fn initialize(config: &BackendConfig) -> anyhow::Result<()> {
    if config.profile == "sqlite" {
        return Ok(());
    }

    let postgres_url = config.postgres_url.as_deref().unwrap();
    let valkey_url = config.valkey_url.as_deref().unwrap();
    let duckdb_path = config.duckdb_path.as_deref().unwrap();

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async {
        let (client, connection) =
            tokio_postgres::connect(postgres_url, tokio_postgres::NoTls).await?;
        tokio::spawn(async move {
            let _ = connection.await;
        });
        client.batch_execute("SET lock_timeout = '5s'; SELECT pg_advisory_lock(hashtextextended('prism-mcpd-schema-v1', 0));").await?;
        let result = client.batch_execute(POSTGRES_SCHEMA).await;
        let _ = client.batch_execute("SELECT pg_advisory_unlock(hashtextextended('prism-mcpd-schema-v1', 0));").await;
        result
    })?;

    let mut valkey = redis::Client::open(valkey_url)?.get_connection()?;
    redis::cmd("SET")
        .arg("prism:mcpd:storage:ready")
        .arg("1")
        .arg("EX")
        .arg(60)
        .query::<()>(&mut valkey)?;

    let duckdb = duckdb::Connection::open(duckdb_path)?;
    duckdb.execute_batch(DUCKDB_SCHEMA)?;
    Ok(())
}

#[cfg(not(feature = "trifecta"))]
pub fn initialize(_config: &BackendConfig) -> anyhow::Result<()> {
    Ok(())
}

#[cfg(feature = "trifecta")]
const POSTGRES_SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS prism_jobs (
    id TEXT PRIMARY KEY,
    tool TEXT NOT NULL,
    operation TEXT NOT NULL,
    state TEXT NOT NULL,
    progress_message TEXT,
    progress_percent DOUBLE PRECISION,
    receipt_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS prism_jobs_updated_idx ON prism_jobs (updated_at DESC);
CREATE TABLE IF NOT EXISTS prism_job_events (
    id BIGSERIAL PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES prism_jobs(id) ON DELETE CASCADE,
    event_type TEXT NOT NULL,
    message TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS prism_evidence_receipts (
    id TEXT PRIMARY KEY,
    operation TEXT NOT NULL,
    status TEXT NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS prism_experiments (id TEXT PRIMARY KEY, document JSONB NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now(), updated_at TIMESTAMPTZ NOT NULL DEFAULT now());
CREATE TABLE IF NOT EXISTS prism_benchmark_plans (id TEXT PRIMARY KEY, name TEXT NOT NULL, spec JSONB NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now(), updated_at TIMESTAMPTZ NOT NULL DEFAULT now());
CREATE TABLE IF NOT EXISTS prism_benchmark_reports (id TEXT PRIMARY KEY, plan_id TEXT NOT NULL, elapsed_ms DOUBLE PRECISION NOT NULL, exit_code INTEGER NOT NULL, output TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now());
CREATE TABLE IF NOT EXISTS prism_benchmark_baselines (name TEXT PRIMARY KEY, report_id TEXT NOT NULL, updated_at TIMESTAMPTZ NOT NULL DEFAULT now());
CREATE TABLE IF NOT EXISTS prism_artifacts (id_hash TEXT PRIMARY KEY, kind TEXT NOT NULL, byte_len BIGINT NOT NULL, media_type TEXT NOT NULL, producer TEXT NOT NULL, target TEXT, created_at TIMESTAMPTZ NOT NULL DEFAULT now());
CREATE INDEX IF NOT EXISTS prism_artifacts_kind_idx ON prism_artifacts(kind);
CREATE TABLE IF NOT EXISTS prism_leases (
    lease_key TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS prism_coord_sessions (session_id TEXT PRIMARY KEY, agent_id TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'active', purpose TEXT, last_heartbeat_at TIMESTAMPTZ NOT NULL DEFAULT now(), closed_at TIMESTAMPTZ);
CREATE TABLE IF NOT EXISTS prism_coord_work (work_id TEXT PRIMARY KEY, title TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'queued', priority INTEGER NOT NULL DEFAULT 0, created_by TEXT, created_at TIMESTAMPTZ NOT NULL DEFAULT now());
CREATE TABLE IF NOT EXISTS prism_coord_claims (claim_id TEXT PRIMARY KEY, work_id TEXT NOT NULL REFERENCES prism_coord_work(work_id), session_id TEXT NOT NULL REFERENCES prism_coord_sessions(session_id), status TEXT NOT NULL DEFAULT 'active', claimed_at TIMESTAMPTZ NOT NULL DEFAULT now(), expires_at TIMESTAMPTZ NOT NULL, released_at TIMESTAMPTZ);
CREATE UNIQUE INDEX IF NOT EXISTS prism_coord_active_claim ON prism_coord_claims(work_id) WHERE status='active';
CREATE TABLE IF NOT EXISTS prism_coord_locks (lock_id TEXT PRIMARY KEY, path TEXT NOT NULL, lock_kind TEXT NOT NULL, session_id TEXT NOT NULL REFERENCES prism_coord_sessions(session_id), status TEXT NOT NULL DEFAULT 'active', expires_at TIMESTAMPTZ NOT NULL, acquired_at TIMESTAMPTZ NOT NULL DEFAULT now(), released_at TIMESTAMPTZ);
CREATE INDEX IF NOT EXISTS prism_coord_lock_path ON prism_coord_locks(path) WHERE status='active';
CREATE TABLE IF NOT EXISTS prism_coord_handoffs (handoff_id TEXT PRIMARY KEY, work_id TEXT NOT NULL REFERENCES prism_coord_work(work_id), from_session TEXT NOT NULL, to_session TEXT NOT NULL, context JSONB NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now());
CREATE TABLE IF NOT EXISTS prism_coord_events (sequence BIGSERIAL PRIMARY KEY, event_type TEXT NOT NULL, session_id TEXT NOT NULL, payload JSONB NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now());
CREATE TABLE IF NOT EXISTS prism_projection_events (
    id BIGSERIAL PRIMARY KEY,
    record_id TEXT NOT NULL,
    receipt_id TEXT,
    data_hash TEXT NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS prism_documents (
    id TEXT PRIMARY KEY, title TEXT NOT NULL, doc_type TEXT NOT NULL, content_md TEXT NOT NULL,
    version BIGINT NOT NULL DEFAULT 1, status TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now(), updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS prism_document_sections (
    id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES prism_documents(id) ON DELETE CASCADE,
    heading TEXT NOT NULL, word_count BIGINT NOT NULL, content TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS prism_document_sections_content_idx ON prism_document_sections USING gin (to_tsvector('english', content));
"#;

#[cfg(feature = "trifecta")]
const DUCKDB_SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS benchmark_projection (
    report_id VARCHAR PRIMARY KEY,
    plan_id VARCHAR,
    elapsed_ms DOUBLE,
    exit_code INTEGER,
    output VARCHAR,
    observed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS trace_projection (
    trace_id VARCHAR,
    event_index BIGINT,
    operation VARCHAR,
    duration_ms DOUBLE,
    payload JSON,
    observed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS tensor_projection (
    tensor_id VARCHAR,
    candidate_id VARCHAR,
    codec_family VARCHAR,
    error_metric DOUBLE,
    latency_us DOUBLE,
    memory_bytes BIGINT,
    observed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS kernel_projection (
    name VARCHAR PRIMARY KEY,
    backend VARCHAR NOT NULL,
    artifact_hash VARCHAR NOT NULL,
    byte_len BIGINT NOT NULL,
    target VARCHAR,
    registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS replay_projection (
    replay_id VARCHAR,
    status VARCHAR NOT NULL,
    payload JSON,
    observed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS experiment_projection (
    experiment_id VARCHAR PRIMARY KEY,
    document JSON,
    observed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS benchmark_plan_projection (
    plan_id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    spec JSON,
    observed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS benchmark_baseline_projection (
    baseline_name VARCHAR PRIMARY KEY,
    report_id VARCHAR NOT NULL,
    observed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS artifact_projection (
    id_hash VARCHAR PRIMARY KEY,
    kind VARCHAR NOT NULL,
    byte_len BIGINT NOT NULL,
    media_type VARCHAR NOT NULL,
    producer VARCHAR NOT NULL,
    target VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"#;

#[cfg(not(feature = "trifecta"))]
pub fn validate(config: &BackendConfig) -> anyhow::Result<BackendHealth> {
    if config.profile == "sqlite" {
        return Ok(BackendHealth::local());
    }
    if config.configured() {
        anyhow::bail!(
            "external storage URLs are configured, but prism-mcpd was built without the `trifecta` feature"
        );
    }
    Ok(BackendHealth::local())
}
