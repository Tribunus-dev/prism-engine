//! Optional distributed storage profile for the MCP daemon.
//!
//! SQLite remains the default local profile. The trifecta profile makes the
//! ownership boundary explicit: PostgreSQL is durable authority, Valkey is
//! coordination.

#[derive(Debug, Clone)]
pub struct BackendConfig {
    pub profile: String,
    pub postgres_url: Option<String>,
    pub valkey_url: Option<String>,
}

impl BackendConfig {
    pub fn from_env() -> Self {
        Self {
            profile: std::env::var("PRISM_MCPD_STORAGE").unwrap_or_else(|_| "trifecta".into()),
            postgres_url: std::env::var("PRISM_MCPD_POSTGRES_URL").ok(),
            valkey_url: std::env::var("PRISM_MCPD_VALKEY_URL").ok(),
        }
    }

    #[cfg(not(feature = "trifecta"))]
    fn configured(&self) -> bool {
        self.postgres_url.is_some() || self.valkey_url.is_some()
    }
}

#[derive(Debug, Clone)]
pub struct BackendHealth {
    pub profile: &'static str,
    pub postgres: BackendStatus,
    pub valkey: BackendStatus,
}

#[derive(Debug, Clone)]
#[cfg_attr(not(feature = "trifecta"), allow(dead_code))]
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
        }
    }

    pub fn as_json(&self) -> serde_json::Value {
        serde_json::json!({
            "profile": self.profile,
            "postgres": status_json(&self.postgres),
            "valkey": status_json(&self.valkey),
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
    if config.postgres_url.is_none() || config.valkey_url.is_none() {
        anyhow::bail!(
            "trifecta is the production storage default; set PRISM_MCPD_POSTGRES_URL and PRISM_MCPD_VALKEY_URL, or explicitly set PRISM_MCPD_STORAGE=sqlite for local/test mode"
        );
    }
    let mut health = BackendHealth {
        profile: "postgres-valkey",
        postgres: BackendStatus::Disabled,
        valkey: BackendStatus::Disabled,
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

    Ok(health)
}

#[cfg(feature = "trifecta")]
pub fn initialize(config: &BackendConfig) -> anyhow::Result<()> {
    if config.profile == "sqlite" {
        return Ok(());
    }

    let postgres_url = config.postgres_url.as_deref().unwrap();
    let valkey_url = config.valkey_url.as_deref().unwrap();

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

    Ok(())
}

#[cfg(not(feature = "trifecta"))]
pub fn initialize(_config: &BackendConfig) -> anyhow::Result<()> {
    Ok(())
}

#[cfg(feature = "trifecta")]
const POSTGRES_SCHEMA: &str = r#"
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
