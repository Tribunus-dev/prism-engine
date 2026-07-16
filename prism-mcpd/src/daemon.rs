use std::io::{BufRead, BufReader, BufWriter, Write};
use std::os::unix::io::AsRawFd;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};

use crossbeam_channel::{bounded, Receiver, Sender};
use parking_lot::Mutex;
use prism_ecs_server::inference::ModelRegistry;
use prism_mcp_core::{
    ArtifactStore, ConnectionId, DaemonState, DbManager, EvidenceLedger, FileLock, JobManager,
    ProcessCache, RequestEnvelope, ResourceLeaseManager, ResponseFrame, Scheduler, SchedulerHandle,
    WorkJournal,
};
use tokio::sync::broadcast;

use crate::backends::{self, BackendHealth};
use crate::tools;

struct StatePaths {
    socket_path: String,
    lock_path: String,
    pid_path: String,
    file_lock_path: String,
    db_path: String,
    staging_dir: String,
}

struct RuntimeFiles {
    socket_path: String,
    pid_path: String,
}

#[derive(Clone)]
struct HealthState {
    db_path: String,
    artifact_dir: String,
    artifact_db_path: String,
    scheduler_heartbeat_ms: Arc<AtomicU64>,
    work_tx: Sender<RequestEnvelope>,
    connection_count: Arc<AtomicU64>,
    backend_health: BackendHealth,
}

impl HealthState {
    fn snapshot(&self) -> serde_json::Value {
        let heartbeat_age_ms =
            now_ms().saturating_sub(self.scheduler_heartbeat_ms.load(Ordering::Relaxed));
        let database_ok = self.backend_health.profile != "sqlite-local"
            || sqlite_quick_check(Path::new(&self.db_path));
        let artifact_database_ok = self.backend_health.profile != "sqlite-local"
            || sqlite_quick_check(Path::new(&self.artifact_db_path));
        let artifacts_ok = std::fs::metadata(&self.artifact_dir)
            .map(|metadata| metadata.is_dir() && !metadata.permissions().readonly())
            .unwrap_or(false);
        let scheduler_ok = heartbeat_age_ms < 3_000;
        let status = if database_ok && artifact_database_ok && artifacts_ok && scheduler_ok {
            "healthy"
        } else {
            "unhealthy"
        };
        serde_json::json!({
            "status": status,
            "protocol": 1,
            "build_id": env!("PRISM_MCPD_BUILD_ID"),
            "pid": std::process::id(),
            "database_ok": database_ok,
            "artifact_database_ok": artifact_database_ok,
            "artifacts_ok": artifacts_ok,
            "scheduler_ok": scheduler_ok,
            "scheduler_heartbeat_age_ms": heartbeat_age_ms,
            "queue_depth": self.work_tx.len(),
            "queue_capacity": self.work_tx.capacity(),
            "connections": self.connection_count.load(Ordering::Relaxed)
            ,"storage": self.backend_health.as_json()
        })
    }
}

impl Drop for RuntimeFiles {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.socket_path);
        let _ = std::fs::remove_file(&self.pid_path);
    }
}

impl StatePaths {
    fn new(base: &str) -> Self {
        Self {
            socket_path: format!("{}/mcpd.sock", base),
            lock_path: format!("{}/mcpd.lock", base),
            pid_path: format!("{}/mcpd.pid", base),
            file_lock_path: format!("{}/file.lock", base),
            db_path: format!("{}/knowledge.db", base),
            staging_dir: format!("{}/staging", base),
        }
    }
}

pub fn run_daemon(state_dir: &str, artifact_dir: &str) -> anyhow::Result<()> {
    let paths = StatePaths::new(state_dir);
    std::fs::create_dir_all(state_dir)?;
    std::fs::create_dir_all(artifact_dir)?;
    let backend_config = backends::BackendConfig::from_env();
    let backend_health = backends::validate(&backend_config)?;
    backends::initialize(&backend_config)?;

    // Singleton lock
    let singleton_lock = FileLock::new(Path::new(&paths.lock_path));
    let _singleton_guard = singleton_lock.try_lock()?.ok_or_else(|| {
        anyhow::anyhow!(
            "another daemon is already running (lock held at {})",
            paths.lock_path
        )
    })?;
    let _ = std::fs::remove_file(&paths.socket_path);
    let _runtime_files = RuntimeFiles {
        socket_path: paths.socket_path.clone(),
        pid_path: paths.pid_path.clone(),
    };
    let terminate = Arc::new(AtomicBool::new(false));
    signal_hook::flag::register(signal_hook::consts::SIGTERM, terminate.clone())?;
    signal_hook::flag::register(signal_hook::consts::SIGINT, terminate.clone())?;

    // Open database
    let db_path = Path::new(&paths.db_path);
    let schema = include_str!("../migrations/001_schema.sql");
    let db = if backend_config.profile == "sqlite" {
        let database = Arc::new(DbManager::open(db_path, schema, 8)?);
        if !sqlite_quick_check(db_path) {
            anyhow::bail!(
                "knowledge database failed SQLite integrity check: {}",
                db_path.display()
            );
        }
        Some(database)
    } else {
        None
    };

    let artifact_store: Arc<dyn prism_mcp_core::ArtifactRepository> =
        if backend_config.profile == "trifecta" {
            #[cfg(feature = "trifecta")]
            {
                crate::trifecta_store::PostgresArtifactRepository::connect(
                    Path::new(artifact_dir),
                    backend_config
                        .postgres_url
                        .as_deref()
                        .expect("validated PostgreSQL URL"),
                )?
            }
            #[cfg(not(feature = "trifecta"))]
            {
                anyhow::bail!("trifecta artifact storage requires the `trifecta` feature")
            }
        } else {
            Arc::new(ArtifactStore::open(Path::new(artifact_dir))?)
        };
    let artifact_db_path = if backend_config.profile == "trifecta" {
        Path::new(
            backend_config
                .duckdb_path
                .as_deref()
                .expect("validated DuckDB path"),
        )
        .to_path_buf()
    } else {
        Path::new(artifact_dir).join("metadata.db")
    };
    let evidence_ledger: Arc<dyn prism_mcp_core::EvidenceStore> =
        if backend_config.profile == "trifecta" {
            #[cfg(feature = "trifecta")]
            {
                crate::trifecta_store::PostgresEvidenceStore::connect(
                    backend_config
                        .postgres_url
                        .as_deref()
                        .expect("validated PostgreSQL URL"),
                )?
            }
            #[cfg(not(feature = "trifecta"))]
            {
                anyhow::bail!("trifecta evidence storage requires the `trifecta` feature")
            }
        } else {
            Arc::new(EvidenceLedger::open(
                db.as_ref().expect("local database").clone(),
            )?)
        };
    let file_lock = FileLock::new(Path::new(&paths.file_lock_path));
    let work_journal = WorkJournal::new(Path::new(&paths.staging_dir));
    let process_cache = ProcessCache::new(32);

    // ── MeasuredEvaluator (hardware-backed evolution evaluator) ─────
    // Opens a separate DuckDB connection for benchmark projections.
    // Gated behind the `measure` feature (requires `trifecta`).
    #[cfg(feature = "measure")]
    let measured_evaluator = {
        if let Some(duckdb_path) = &backend_config.duckdb_path {
            match duckdb::Connection::open(duckdb_path) {
                Ok(conn) => {
                    let eval =
                        prism_runtime::measured::MeasuredEvaluator::new(0.1).with_duckdb(conn);
                    Some(eval)
                }
                Err(e) => {
                    eprintln!("prism-mcpd: failed to open DuckDB for MeasuredEvaluator: {e}");
                    None
                }
            }
        } else {
            None
        }
    };
    #[cfg(not(feature = "measure"))]
    let measured_evaluator: Option<()> = None;
    let _measured_evaluator = measured_evaluator;

    // Global work queue: reader threads push, scheduler pulls
    let (work_tx, work_rx) = bounded::<RequestEnvelope>(256);
    let work_tx_for_handler = work_tx.clone();

    // Create shared model registry
    let model_registry = Arc::new(Mutex::new(ModelRegistry::new()));

    // Register tools
    let mut tools_map = tools::register_basic(model_registry)?;

    // Shared state
    let conn_count = Arc::new(AtomicU64::new(0));
    let idle_gen = Arc::new(AtomicU64::new(0));

    // Create job manager and resource leases
    let job_manager: Arc<dyn prism_mcp_core::JobStore> = if backend_config.profile == "trifecta" {
        #[cfg(feature = "trifecta")]
        {
            crate::trifecta_store::PostgresJobStore::connect(
                backend_config
                    .postgres_url
                    .as_deref()
                    .expect("validated PostgreSQL URL"),
            )?
        }
        #[cfg(not(feature = "trifecta"))]
        {
            anyhow::bail!("trifecta job storage requires the `trifecta` feature")
        }
    } else {
        Arc::new(JobManager::new(
            db.as_ref().expect("local database").clone(),
        )?)
    };
    let resource_leases: Arc<dyn prism_mcp_core::LeaseStore> =
        if backend_config.profile == "trifecta" {
            #[cfg(feature = "trifecta")]
            {
                crate::trifecta_store::ValkeyLeaseStore::connect(
                    backend_config
                        .valkey_url
                        .as_deref()
                        .expect("validated Valkey URL"),
                )?
            }
            #[cfg(not(feature = "trifecta"))]
            {
                anyhow::bail!("trifecta lease storage requires the `trifecta` feature")
            }
        } else {
            Arc::new(ResourceLeaseManager::new())
        };
    let coordination_store: Option<Arc<dyn prism_mcp_core::CoordinationStore>> =
        if backend_config.profile == "trifecta" {
            #[cfg(feature = "trifecta")]
            {
                Some(crate::trifecta_store::PostgresCoordinationStore::connect(
                    backend_config
                        .postgres_url
                        .as_deref()
                        .expect("validated PostgreSQL URL"),
                )?)
            }
            #[cfg(not(feature = "trifecta"))]
            {
                anyhow::bail!("trifecta coordination storage requires the `trifecta` feature")
            }
        } else {
            None
        };
    let projection_store: Arc<dyn prism_mcp_core::ProjectionStore> =
        if backend_config.profile == "trifecta" {
            #[cfg(feature = "trifecta")]
            {
                crate::trifecta_store::DuckDbProjectionStore::open(
                    backend_config
                        .duckdb_path
                        .as_deref()
                        .expect("validated DuckDB path"),
                )?
            }
            #[cfg(not(feature = "trifecta"))]
            {
                anyhow::bail!("trifecta projections require the `trifecta` feature")
            }
        } else {
            db.as_ref().expect("local database").clone()
        };
    let experiment_store: Arc<dyn prism_mcp_core::ExperimentStore> =
        if backend_config.profile == "trifecta" {
            #[cfg(feature = "trifecta")]
            {
                crate::trifecta_store::PostgresExperimentStore::connect(
                    backend_config
                        .postgres_url
                        .as_deref()
                        .expect("validated PostgreSQL URL"),
                )?
            }
            #[cfg(not(feature = "trifecta"))]
            {
                anyhow::bail!("trifecta experiment storage requires the `trifecta` feature")
            }
        } else {
            db.as_ref().expect("local database").clone()
        };
    let benchmark_store: Arc<dyn prism_mcp_core::BenchmarkStore> =
        if backend_config.profile == "trifecta" {
            #[cfg(feature = "trifecta")]
            {
                crate::trifecta_store::PostgresBenchmarkStore::connect(
                    backend_config
                        .postgres_url
                        .as_deref()
                        .expect("validated PostgreSQL URL"),
                )?
            }
            #[cfg(not(feature = "trifecta"))]
            {
                anyhow::bail!("trifecta benchmark storage requires the `trifecta` feature")
            }
        } else {
            db.as_ref().expect("local database").clone()
        };
    let knowledge_store: Arc<dyn prism_mcp_core::KnowledgeStore> =
        if backend_config.profile == "trifecta" {
            #[cfg(feature = "trifecta")]
            {
                crate::trifecta_store::PostgresKnowledgeStore::connect(
                    backend_config
                        .postgres_url
                        .as_deref()
                        .expect("validated PostgreSQL URL"),
                )?
            }
            #[cfg(not(feature = "trifecta"))]
            {
                anyhow::bail!("trifecta knowledge storage requires the `trifecta` feature")
            }
        } else {
            db.as_ref().expect("local database").clone()
        };

    // Build DaemonState with Phase 1 tools, then register stateful handlers
    let partial_tools = Arc::new(tools_map.clone());
    let mut state = DaemonState {
        coordination_store,
        tools: partial_tools.clone(),
        artifact_store,
        evidence_ledger,
        file_lock,
        work_journal,
        process_cache,
        scheduler_handle: SchedulerHandle {
            release_sender: work_tx_for_handler,
        },
        job_manager,
        resource_leases,
        projection_store,
        experiment_store,
        benchmark_store,
        knowledge_store,
        connection_count: conn_count.clone(),
        idle_generation: idle_gen.clone(),
    };

    // Phase 2: register stateful crate handlers that need &DaemonState
    let stateful_map = tools::register_stateful(&state)?;
    tools_map.extend(stateful_map);
    let tools = Arc::new(tools_map);
    state.tools = tools.clone();
    let state = Arc::new(state);

    // Start scheduler in its own thread
    let sched = Scheduler::new(work_rx, tools, state.clone());
    let health = HealthState {
        db_path: paths.db_path.clone(),
        artifact_dir: artifact_dir.to_string(),
        artifact_db_path: artifact_db_path.to_string_lossy().into_owned(),
        scheduler_heartbeat_ms: sched.heartbeat(),
        work_tx: work_tx.clone(),
        connection_count: conn_count.clone(),
        backend_health,
    };
    std::thread::spawn(move || {
        sched.run();
    });

    // ── Model registry (shared between MCP handlers and dashboard) ──────
    let model_registry: Arc<parking_lot::Mutex<prism_ecs_server::inference::ModelRegistry>> =
        Arc::new(parking_lot::Mutex::new(
            prism_ecs_server::inference::ModelRegistry::new(),
        ));

    // ── Dashboard HTTP server ───────────────────────────────────────────
    {
        let registry = model_registry.clone();
        std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().expect("dashboard tokio runtime");
            rt.block_on(async move {
                let (model_tx, _) = broadcast::channel::<Vec<String>>(16);
                let world = Arc::new(Mutex::new(prism_ecs_core::World::new()));
                let state = crate::dashboard::DashboardState {
                    registry,
                    model_tx,
                    world,
                };
                let app = crate::dashboard::router(state);
                let listener = tokio::net::TcpListener::bind("127.0.0.1:8080")
                    .await
                    .expect("dashboard bind 127.0.0.1:8080");
                tracing::info!("Dashboard listening on http://127.0.0.1:8080");
                eprintln!("prism-mcpd: dashboard at http://127.0.0.1:8080");
                axum::serve(listener, app).await.expect("dashboard serve");
            });
        });
    }

    // Write PID
    let _ = std::fs::write(&paths.pid_path, format!("{}", std::process::id()));

    // Bind and listen
    let listener = UnixListener::bind(&paths.socket_path)?;
    eprintln!("prism-mcpd: listening on {}", paths.socket_path);

    // Accept loop (non-blocking)
    listener.set_nonblocking(true)?;
    // Track connections to restore blocking mode after accept
    while !terminate.load(Ordering::Relaxed) {
        if conn_count.load(Ordering::Relaxed) == 0 {
            idle_gen.fetch_add(1, Ordering::Relaxed);
        }

        match listener.accept() {
            Ok((stream, _addr)) => {
                // Restore blocking mode — the accepted stream inherits
                // O_NONBLOCK from the listener on macOS, which causes
                // BufWriter to silently discard data on flush errors.
                if let Err(e) = set_blocking(&stream) {
                    eprintln!("prism-mcpd: set_blocking: {e}");
                    return Err(anyhow::anyhow!("set_blocking: {e}"));
                }
                conn_count.fetch_add(1, Ordering::Relaxed);
                let conn_count_clone = conn_count.clone();
                let work_tx_clone = work_tx.clone();
                let health_clone = health.clone();
                let conn_id = ConnectionId::new();
                let (response_tx, response_rx) = bounded::<ResponseFrame>(64);

                std::thread::spawn(move || {
                    handle_connection(
                        stream,
                        conn_id,
                        work_tx_clone,
                        response_tx,
                        response_rx,
                        health_clone,
                    );
                    conn_count_clone.fetch_sub(1, Ordering::Relaxed);
                });
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(100));
            }
            Err(e) => {
                eprintln!("prism-mcpd: accept error: {}", e);
                std::thread::sleep(Duration::from_secs(1));
            }
        }
    }
    Ok(())
}

fn set_blocking(stream: &UnixStream) -> std::io::Result<()> {
    let fd = stream.as_raw_fd();
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
    if flags < 0 {
        return Err(std::io::Error::last_os_error());
    }
    if unsafe { libc::fcntl(fd, libc::F_SETFL, flags & !libc::O_NONBLOCK) } < 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

fn handle_connection(
    stream: UnixStream,
    conn_id: ConnectionId,
    work_tx: Sender<RequestEnvelope>,
    response_tx: Sender<ResponseFrame>,
    response_rx: Receiver<ResponseFrame>,
    health: HealthState,
) {
    let reader = match stream.try_clone() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("prism-mcpd: failed to clone stream: {e}");
            return;
        }
    };
    let writer = stream;

    // Reader thread: socket → work queue
    let reader_handle = {
        let work_tx = work_tx.clone();
        std::thread::spawn(move || {
            let mut buf = BufReader::new(reader);
            let mut line = String::new();
            while matches!(buf.read_line(&mut line), Ok(n) if n > 0) {
                if line.trim().is_empty() {
                    line.clear();
                    continue;
                }
                if let Ok(frame) = serde_json::from_str::<serde_json::Value>(line.trim()) {
                    if frame["method"] == "prism/health" {
                        let json = serde_json::json!({
                            "jsonrpc": "2.0",
                            "id": frame["id"],
                            "result": health.snapshot()
                        });
                        if response_tx
                            .send(ResponseFrame {
                                connection_id: conn_id,
                                json: json.to_string(),
                            })
                            .is_err()
                        {
                            break;
                        }
                        line.clear();
                        continue;
                    }
                }
                let env = RequestEnvelope {
                    connection_id: conn_id,
                    response_tx: response_tx.clone(),
                    frame: line.trim().to_string(),
                };
                if work_tx.send(env).is_err() {
                    break;
                }
                line.clear();
            }
        })
    };

    // Writer thread: response channel → socket (independent)
    let writer_handle = std::thread::spawn(move || {
        let mut buf = BufWriter::new(writer);
        while let Ok(msg) = response_rx.recv() {
            if writeln!(buf, "{}", msg.json).is_err() {
                break;
            }
            buf.flush().ok();
        }
    });

    let _ = reader_handle.join();
    let _ = writer_handle.join();
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn sqlite_quick_check(path: &Path) -> bool {
    let Ok(db) = rusqlite::Connection::open(path) else {
        return false;
    };
    if db.busy_timeout(Duration::from_millis(100)).is_err() {
        return false;
    }
    db.query_row("PRAGMA quick_check", [], |row| row.get::<_, String>(0))
        .map(|result| result == "ok")
        .unwrap_or(false)
}
