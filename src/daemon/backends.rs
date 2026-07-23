//! Optional distributed storage profile for the MCP daemon.
//! Storage backend configuration for the MCP daemon.
//!
//! The trifecta profile is the default: PostgreSQL is durable authority,
//! Valkey is coordination. On startup, the daemon auto-creates the database
//! if it does not exist, using the PostgreSQL maintenance database.

use std::net::ToSocketAddrs;
use std::path::PathBuf;
use std::time::Duration;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValkeyConfig {
    pub url: String,
    pub managed: bool,
    pub data_dir: PathBuf,
    pub binary: String,
    pub readiness_timeout: Duration,
}

impl ValkeyConfig {
    pub fn from_env() -> Self {
        let explicit_url = std::env::var("PRISM_MCPD_VALKEY_URL").ok();
        let managed = std::env::var("PRISM_MCPD_VALKEY_MANAGED")
            .map(|value| matches!(value.as_str(), "1" | "true" | "yes"))
            .unwrap_or(explicit_url.is_none());
        let state_dir = std::env::var("PRISM_MCPD_STATE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(".prism-mcpd"));
        Self {
            url: explicit_url.unwrap_or_else(|| "redis://127.0.0.1:6379".into()),
            managed,
            data_dir: std::env::var("PRISM_MCPD_VALKEY_DATA_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|_| state_dir.join("valkey")),
            binary: std::env::var("PRISM_MCPD_VALKEY_BIN")
                .unwrap_or_else(|_| "valkey-server".into()),
            readiness_timeout: Duration::from_millis(
                std::env::var("PRISM_MCPD_VALKEY_READY_TIMEOUT_MS")
                    .ok()
                    .and_then(|value| value.parse().ok())
                    .unwrap_or(10_000),
            ),
        }
    }

    pub fn command_args(&self) -> Vec<String> {
        let port = self
            .url
            .rsplit_once(':')
            .and_then(|(_, value)| value.parse::<u16>().ok())
            .unwrap_or(6379);
        vec![
            "--dir".into(),
            self.data_dir.to_string_lossy().into_owned(),
            "--port".into(),
            port.to_string(),
            "--save".into(),
            "".into(),
            "--appendonly".into(),
            "yes".into(),
        ]
    }
}

#[derive(Debug, Clone)]
pub struct BackendConfig {
    pub profile: String,
    pub postgres_url: Option<String>,
    pub valkey_url: Option<String>,
    pub valkey: ValkeyConfig,
}

impl BackendConfig {
    pub fn from_env() -> Self {
        let valkey = ValkeyConfig::from_env();
        Self {
            profile: std::env::var("PRISM_MCPD_STORAGE").unwrap_or_else(|_| "trifecta".into()),
            postgres_url: std::env::var("PRISM_MCPD_POSTGRES_URL")
                .ok()
                .or_else(|| Some("postgresql://localhost:5432/prism_mcpd".into())),
            valkey_url: Some(valkey.url.clone()),
            valkey,
        }
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
    Healthy,
    Failed(String),
}

impl BackendHealth {
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
        BackendStatus::Healthy => serde_json::json!({"configured": true, "healthy": true}),
        BackendStatus::Failed(error) => {
            serde_json::json!({"configured": true, "healthy": false, "error": error})
        }
    }
}

#[cfg(feature = "trifecta")]
pub fn validate(config: &BackendConfig) -> anyhow::Result<BackendHealth> {
    if config.profile != "trifecta" {
        anyhow::bail!(
            "unsupported storage profile '{}'; prism-mcpd requires the trifecta PostgreSQL/Valkey profile",
            config.profile
        );
    }
    if config.postgres_url.is_none() || config.valkey_url.is_none() {
        anyhow::bail!(
            "set PRISM_MCPD_POSTGRES_URL and PRISM_MCPD_VALKEY_URL for the required trifecta storage profile"
        );
    }
    let mut health = BackendHealth {
        profile: "postgres-valkey",
        postgres: BackendStatus::Failed("not checked".into()),
        valkey: BackendStatus::Failed("not checked".into()),
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
    let postgres_url = config.postgres_url.as_deref().unwrap();
    let valkey_url = config.valkey_url.as_deref().unwrap();

    // Readiness, database creation, advisory-lock serialization, and schema
    // migrations are owned by the storage layer rather than the supervisor.
    crate::daemon::trifecta_store::migrations::bootstrap(postgres_url)?;

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
    anyhow::bail!("prism-mcpd requires the `trifecta` feature")
}

#[cfg(not(feature = "trifecta"))]
pub fn validate(config: &BackendConfig) -> anyhow::Result<BackendHealth> {
    let _ = config;
    anyhow::bail!("prism-mcpd requires the `trifecta` feature")
}
#[cfg(all(test, feature = "trifecta"))]
mod tests {
    use super::*;

    #[test]
    fn sqlite_profile_is_rejected_before_connecting() {
        let config = BackendConfig {
            profile: "sqlite".into(),
            postgres_url: None,
            valkey_url: None,
            valkey: ValkeyConfig {
                url: "redis://127.0.0.1:6379".into(),
                managed: false,
                data_dir: std::path::PathBuf::from("target/test-valkey"),
                binary: "valkey-server".into(),
                readiness_timeout: std::time::Duration::from_millis(1),
            },
        };
        let error = validate(&config).expect_err("SQLite must not remain a daemon profile");
        assert!(error.to_string().contains("unsupported storage profile"));
    }
}

/// Owns locally launched Trifecta services without taking ownership of
/// services that were already running. The supervisor is intentionally
/// synchronous because daemon startup is synchronous and must fail closed.
#[derive(Debug)]
pub struct TrifectaSupervisor {
    children: Vec<std::process::Child>,
    data_dir: std::path::PathBuf,
}

#[derive(Debug)]
pub enum TrifectaSupervisorError {
    Io(std::io::Error),
    InvalidEndpoint(String),
    BinaryUnavailable {
        service: &'static str,
        path: String,
    },
    StartupTimeout {
        service: &'static str,
        endpoint: String,
    },
    ProcessExited {
        service: &'static str,
        status: std::process::ExitStatus,
    },
}

impl std::fmt::Display for TrifectaSupervisorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "trifecta supervisor I/O error: {e}"),
            Self::InvalidEndpoint(e) => write!(f, "invalid trifecta endpoint: {e}"),
            Self::BinaryUnavailable { service, path } => {
                write!(f, "{service} binary unavailable: {path}")
            }
            Self::StartupTimeout { service, endpoint } => {
                write!(f, "{service} did not become ready at {endpoint}")
            }
            Self::ProcessExited { service, status } => {
                write!(f, "{service} exited during startup: {status}")
            }
        }
    }
}

impl std::error::Error for TrifectaSupervisorError {}
impl From<std::io::Error> for TrifectaSupervisorError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl TrifectaSupervisor {
    pub fn start(
        config: &BackendConfig,
        state_dir: impl AsRef<std::path::Path>,
    ) -> anyhow::Result<Self> {
        let data_dir = std::env::var_os("PRISM_MCPD_TRIFECTA_DATA_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| state_dir.as_ref().join("trifecta"));
        std::fs::create_dir_all(&data_dir)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&data_dir, std::fs::Permissions::from_mode(0o700))?;
        }
        let mut supervisor = Self {
            children: Vec::new(),
            data_dir,
        };
        if let Some(url) = config.postgres_url.as_deref() {
            supervisor.ensure_postgres(url)?;
        }
        if let Some(url) = config.valkey_url.as_deref() {
            supervisor.ensure_valkey(url, &config.valkey)?;
        }
        Ok(supervisor)
    }

    fn ensure_postgres(&mut self, url: &str) -> anyhow::Result<()> {
        let endpoint = endpoint(url, 5432)?;
        if tcp_ready(endpoint) {
            return Ok(());
        }
        let binary = bundled_binary("PRISM_MCPD_POSTGRES_BIN", "postgres");
        if !binary_available(&binary) {
            return Err(TrifectaSupervisorError::BinaryUnavailable {
                service: "postgres",
                path: binary,
            }
            .into());
        }
        let dir = self.data_dir.join("postgres");
        std::fs::create_dir_all(&dir)?;
        if !dir.join("PG_VERSION").exists() {
            let initdb = bundled_binary("PRISM_MCPD_INITDB_BIN", "initdb");
            if !binary_available(&initdb) {
                return Err(TrifectaSupervisorError::BinaryUnavailable {
                    service: "initdb",
                    path: initdb,
                }
                .into());
            }
            let status = std::process::Command::new(&initdb)
                .arg("-D")
                .arg(&dir)
                .arg("-L")
                .arg(
                    std::path::Path::new(&initdb)
                        .parent()
                        .and_then(|bin| bin.parent())
                        .map(|root| root.join("share/postgresql"))
                        .unwrap_or_else(|| self.data_dir.join("postgres/share/postgresql")),
                )
                .arg("--no-locale")
                .arg("--auth=trust")
                .status()?;
            if !status.success() {
                return Err(TrifectaSupervisorError::ProcessExited {
                    service: "initdb",
                    status,
                }
                .into());
            }
        }
        let port = endpoint.port().to_string();
        let child = std::process::Command::new(&binary)
            .arg("-D")
            .arg(&dir)
            .arg("-p")
            .arg(port)
            .spawn()?;
        self.children.push(child);
        wait_ready(
            "postgres",
            endpoint,
            std::time::Duration::from_secs(15),
            &mut self.children.last_mut().unwrap(),
        )
    }

    fn ensure_valkey(&mut self, url: &str, config: &ValkeyConfig) -> anyhow::Result<()> {
        let endpoint = endpoint(url, 6379)?;
        if tcp_ready(endpoint) {
            return Ok(());
        }
        if !config.managed {
            return Err(TrifectaSupervisorError::StartupTimeout {
                service: "valkey",
                endpoint: endpoint.to_string(),
            }
            .into());
        }
        let binary = if std::env::var_os("PRISM_MCPD_VALKEY_BIN").is_some() {
            config.binary.clone()
        } else {
            bundled_binary("PRISM_MCPD_VALKEY_BIN", "valkey-server")
        };
        if !binary_available(&binary) {
            return Err(TrifectaSupervisorError::BinaryUnavailable {
                service: "valkey",
                path: binary,
            }
            .into());
        }
        let dir = config.data_dir.clone();
        std::fs::create_dir_all(&dir)?;
        let child = std::process::Command::new(&binary)
            .args(config.command_args())
            .spawn()?;
        self.children.push(child);
        wait_ready(
            "valkey",
            endpoint,
            config.readiness_timeout,
            &mut self.children.last_mut().unwrap(),
        )
    }
}

impl Drop for TrifectaSupervisor {
    fn drop(&mut self) {
        for child in &mut self.children {
            #[cfg(unix)]
            {
                unsafe {
                    libc::kill(child.id() as libc::pid_t, libc::SIGTERM);
                }
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
                while std::time::Instant::now() < deadline {
                    match child.try_wait() {
                        Ok(Some(_)) => break,
                        Ok(None) => std::thread::sleep(std::time::Duration::from_millis(50)),
                        Err(_) => break,
                    }
                }
            }
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn endpoint(url: &str, default_port: u16) -> Result<std::net::SocketAddr, TrifectaSupervisorError> {
    let parsed = url::Url::parse(url)
        .map_err(|e| TrifectaSupervisorError::InvalidEndpoint(e.to_string()))?;
    let host = parsed.host_str().unwrap_or("127.0.0.1");
    let port = parsed.port().unwrap_or(default_port);
    format!("{}:{}", host, port)
        .to_socket_addrs()
        .map_err(|e| TrifectaSupervisorError::InvalidEndpoint(e.to_string()))?
        .next()
        .ok_or_else(|| TrifectaSupervisorError::InvalidEndpoint("endpoint did not resolve".into()))
}

fn binary_available(binary: &str) -> bool {
    std::path::Path::new(binary).is_file()
        || std::env::var_os("PATH")
            .unwrap_or_default()
            .to_string_lossy()
            .split(':')
            .any(|p| std::path::Path::new(p).join(binary).is_file())
}

fn bundled_binary(env_key: &str, name: &str) -> String {
    if let Ok(value) = std::env::var(env_key) {
        return value;
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(prefix) = exe.parent().and_then(|bin| bin.parent()) {
            let candidate = prefix
                .join("libexec/prism/trifecta")
                .join(if name == "postgres" || name == "initdb" {
                    "postgresql/bin"
                } else {
                    "valkey"
                })
                .join(name);
            if candidate.is_file() {
                return candidate.to_string_lossy().into_owned();
            }
        }
    }
    name.to_string()
}

fn tcp_ready(endpoint: std::net::SocketAddr) -> bool {
    std::net::TcpStream::connect_timeout(&endpoint, std::time::Duration::from_millis(150)).is_ok()
}

fn wait_ready(
    service: &'static str,
    endpoint: std::net::SocketAddr,
    timeout: std::time::Duration,
    child: &mut std::process::Child,
) -> anyhow::Result<()> {
    let deadline = std::time::Instant::now() + timeout;
    while std::time::Instant::now() < deadline {
        if tcp_ready(endpoint) {
            return Ok(());
        }
        if let Some(status) = child.try_wait()? {
            return Err(TrifectaSupervisorError::ProcessExited { service, status }.into());
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    Err(TrifectaSupervisorError::StartupTimeout {
        service,
        endpoint: endpoint.to_string(),
    }
    .into())
}

#[cfg(test)]
mod supervisor_tests {
    use super::*;

    #[test]
    fn parses_backend_urls_with_default_ports() {
        assert_eq!(endpoint("redis://127.0.0.1", 6379).unwrap().port(), 6379);
        assert_eq!(
            endpoint("postgresql://localhost:55432/prism", 5432)
                .unwrap()
                .port(),
            55432
        );
    }

    #[test]
    fn rejects_malformed_backend_urls() {
        assert!(matches!(
            endpoint("not a url", 5432),
            Err(TrifectaSupervisorError::InvalidEndpoint(_))
        ));
    }
}
