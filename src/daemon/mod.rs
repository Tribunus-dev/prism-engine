use std::os::unix::net::UnixListener;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crossbeam_channel::bounded;
use parking_lot::Mutex;
use prism_ecs_ir::evolution::evaluate::EvaluationStrategy;
use prism_ecs_runtime::schedule::{
    AdmitSystem, CleanupSystem, CollectSystem, DispatchSystem, LeaseSystem, ObserveSystem,
    PlanSystem, PublishSystem,
};
use prism_ecs_runtime::RuntimeKernel;
use prism_ecs_runtime::RuntimeSchedule;
use prism_ecs_protocol_adapter::{NoopWorkflowCancellation, ProjectionWorkflowStore, WorkflowClient};
use prism_ecs_server::inference::ModelRegistry;
use prism_ecs_server::runtime::{PrismInferenceServer, ServerConfig};
use prism_ecs_server::runtime::server_types::InferenceExecutionPolicy;
use prism_mcp_core::{
    ConnectionId, DaemonState, FileLock, ProcessCache, RequestEnvelope, ResponseFrame, Scheduler,
    SchedulerHandle, WorkJournal,
};
use tokio::sync::broadcast;

pub mod backends;
pub mod connection;
pub mod dashboard;
pub mod dispatcher;
pub mod health;
pub mod provenance;
pub mod proxy;
pub mod tools;
pub mod trifecta_store;
use connection::handle_connection;
use connection::set_blocking;
use health::HealthState;

struct StatePaths {
    socket_path: String,
    lock_path: String,
    pid_path: String,
    file_lock_path: String,
    staging_dir: String,
}

struct RuntimeFiles {
    socket_path: String,
    pid_path: String,
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
            staging_dir: format!("{}/staging", base),
        }
    }
}

pub fn run_daemon(state_dir: &str, artifact_dir: &str) -> anyhow::Result<()> {
    let paths = StatePaths::new(state_dir);
    std::fs::create_dir_all(state_dir)?;
    std::fs::create_dir_all(artifact_dir)?;

    // Claim singleton authority before initializing external backends. This
    // makes concurrent starts cheap and ensures only the lock owner writes
    // runtime identity files.
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
    let _ = std::fs::write(&paths.pid_path, format!("{}", std::process::id()));

    let backend_config = backends::BackendConfig::from_env();
    let _trifecta_supervisor = backends::TrifectaSupervisor::start(&backend_config, state_dir)?;
    let backend_health = backends::validate(&backend_config)?;
    backends::initialize(&backend_config)?;
    let terminate = Arc::new(AtomicBool::new(false));
    signal_hook::flag::register(signal_hook::consts::SIGTERM, terminate.clone())?;
    signal_hook::flag::register(signal_hook::consts::SIGINT, terminate.clone())?;

    let postgres_url = backend_config
        .postgres_url
        .as_deref()
        .expect("validated PostgreSQL URL");
    let valkey_url = backend_config
        .valkey_url
        .as_deref()
        .expect("validated Valkey URL");
    let artifact_store: Arc<dyn prism_mcp_core::ArtifactRepository> =
        crate::daemon::trifecta_store::PostgresArtifactRepository::connect(
            Path::new(artifact_dir),
            postgres_url,
        )?;
    let evidence_ledger: Arc<dyn prism_mcp_core::EvidenceStore> =
        crate::daemon::trifecta_store::PostgresEvidenceStore::connect(postgres_url)?;
    let conversation_store: Arc<dyn prism_mcp_core::ConversationStore> =
        crate::daemon::trifecta_store::PostgresConversationStore::connect(postgres_url)?;
    let file_lock = FileLock::new(Path::new(&paths.file_lock_path));
    let work_journal = WorkJournal::new(Path::new(&paths.staging_dir));
    let process_cache = ProcessCache::new(32);

    // ── MeasuredEvaluator (hardware-backed evolution evaluator) ─────
    // Uses prism-ecs-duckdb columnar table (pure Rust, zero C++ deps).
    #[cfg(feature = "measure")]
    let measured_evaluator = {
        let eval = prism_ecs_server::engine::MeasuredEvaluator::new(0.1);
        Some(eval)
    };
    #[cfg(not(feature = "measure"))]
    let measured_evaluator: Option<prism_ecs_server::engine::MeasuredEvaluator> = None;
    let measured_evaluator: Option<Arc<dyn EvaluationStrategy + Send + Sync>> =
        measured_evaluator.map(|e| Arc::new(e) as Arc<dyn EvaluationStrategy + Send + Sync>);

    // Global work queue: reader threads push, scheduler pulls
    let (work_tx, work_rx) = bounded::<RequestEnvelope>(256);
    let work_tx_for_handler = work_tx.clone();

    // Create shared model registry
    let ecs_inference = Arc::new(PrismInferenceServer::new(ServerConfig {
        cimage_path: artifact_dir.to_string(),
        context_profiles: Vec::new(),
        execution_policy: InferenceExecutionPolicy::HybridMetalAccelerate,
        max_concurrent_sessions: 64,
        http_listen: None,
        receipt_store_path: format!("{state_dir}/ecs-receipts.jsonl"),
        memory_elevated_threshold_bytes: 8 * 1024 * 1024 * 1024,
        memory_critical_threshold_bytes: 12 * 1024 * 1024 * 1024,
    }));
    let registry = ModelRegistry::new();
    registry.attach_ecs_server(ecs_inference.clone())?;
    let model_registry = Arc::new(Mutex::new(registry));

    // ── Production kernel with durable adapters ──────────────────────

    struct WallClock;
    impl prism_ecs_runtime::KernelClock for WallClock {
        fn now_ms(&self) -> u64 {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64
        }
    }

    struct DaemonLeaseCoordinator {
        store: Arc<dyn prism_mcp_core::LeaseStore>,
    }
    impl prism_ecs_runtime::LeaseCoordinator for DaemonLeaseCoordinator {
        fn acquire(&self, key: &str, ttl_ms: u64) -> Result<bool, prism_ecs_runtime::RuntimeError> {
            let ttl_secs = ttl_ms.max(1).div_ceil(1000);
            self.store
                .acquire(key, "prism-daemon", ttl_secs)
                .map_err(|e| prism_ecs_runtime::RuntimeError::Lease(e.to_string()))
        }
        fn renew(&self, key: &str, ttl_ms: u64) -> Result<bool, prism_ecs_runtime::RuntimeError> {
            let ttl_secs = ttl_ms.max(1).div_ceil(1000);
            self.store
                .acquire(key, "prism-daemon", ttl_secs)
                .map_err(|e| prism_ecs_runtime::RuntimeError::Lease(e.to_string()))
        }
        fn release(&self, key: &str) -> Result<(), prism_ecs_runtime::RuntimeError> {
            self.store
                .release(key, "prism-daemon")
                .map_err(|e| prism_ecs_runtime::RuntimeError::Lease(e.to_string()))
        }
    }

    let resource_leases: Arc<dyn prism_mcp_core::LeaseStore> =
        crate::daemon::trifecta_store::ValkeyLeaseStore::connect(valkey_url)?;

    let command_store: Box<dyn prism_ecs_runtime::CommandStore> = {
        let url = backend_config
            .postgres_url
            .as_deref()
            .expect("validated PostgreSQL URL");
        Box::new(
            crate::daemon::trifecta_store::PostgresCommandStore::connect(url)
                .expect("PostgresCommandStore connect"),
        )
    };

    let lease_coordinator: Box<dyn prism_ecs_runtime::LeaseCoordinator> =
        Box::new(DaemonLeaseCoordinator {
            store: resource_leases.clone(),
        });

    // ── Production kernel with durable adapters ──────────────────────
    let daemon_instance_id = format!("{}-{}", std::process::id(), uuid::Uuid::new_v4());

    let snapshot_store: Box<dyn prism_ecs_runtime::SnapshotStore> = {
        let url = backend_config
            .postgres_url
            .as_deref()
            .expect("validated PostgreSQL URL");
        Box::new(
            crate::daemon::trifecta_store::PostgresSnapshotStore::connect(url)
                .expect("PostgresSnapshotStore connect"),
        )
    };

    let tick_receipt_store: Box<dyn prism_ecs_runtime::TickReceiptStore> = {
        let url = backend_config
            .postgres_url
            .as_deref()
            .expect("validated PostgreSQL URL");
        Box::new(
            crate::daemon::trifecta_store::PostgresTickReceiptStore::connect(url)
                .expect("PostgresTickReceiptStore connect"),
        )
    };

    let kernel = RuntimeKernel::with_ports(
        command_store,
        snapshot_store,
        tick_receipt_store,
        lease_coordinator,
        Box::new(WallClock),
    );
    let kernel = std::sync::Arc::new(kernel);
    let kernel_handle = kernel.handle();
    let projection_store: Arc<dyn prism_mcp_core::ProjectionStore> =
        crate::daemon::trifecta_store::DuckDbProjectionStore::open_postgres(postgres_url)?;
    let provenance_store: Arc<dyn prism_mcp_core::ProvenanceGraphStore> =
        crate::daemon::trifecta_store::PostgresProvenanceGraphStore::connect(postgres_url)?;
    let graph_projection = crate::daemon::trifecta_store::DuckDbGraphProjection::new();
    let projection_source =
        crate::daemon::trifecta_store::PostgresProvenanceGraphStore::connect(postgres_url)?;
    graph_projection.rebuild_from_postgres_rows(projection_source.projection_rows()?)?;

    // Recovery: load snapshot (using kernel's internal snapshot store), restore world, replay tail.
    // Fail closed — if recovery fails, the daemon exits without binding the socket.
    let recovery = kernel.recover()?;

    if recovery.recovery_state == "fresh" {
        // No snapshot found, no commands in database — truly fresh start
        eprintln!("Kernel: fresh start (no recovery data)");
    } else {
        eprintln!(
            "Kernel recovery: {} (snapshot epoch {}, replayed {})",
            recovery.recovery_state, recovery.snapshot_epoch, recovery.replayed_commands,
        );

        if recovery.unresolved_commands > 0 {
            eprintln!(
                "Kernel: {} unresolved commands require attention",
                recovery.unresolved_commands
            );
        }
    }

    // Build and validate production schedule, then register it
    {
        let mut schedule = RuntimeSchedule::new();
        // Register real lifecycle systems for each stage.
        schedule
            .register_system(Box::new(ObserveSystem))
            .expect("register observe");
        schedule
            .register_system(Box::new(PlanSystem))
            .expect("register plan");
        schedule
            .register_system(Box::new(AdmitSystem))
            .expect("register admit");
        schedule
            .register_system(Box::new(LeaseSystem))
            .expect("register lease");
        schedule
            .register_system(Box::new(DispatchSystem))
            .expect("register dispatch");
        schedule
            .register_system(Box::new(CollectSystem))
            .expect("register collect");
        schedule
            .register_system(Box::new(PublishSystem))
            .expect("register publish");
        schedule
            .register_system(Box::new(CleanupSystem))
            .expect("register cleanup");
        schedule.validate().expect("schedule validation");
        // Wire the production compiler dispatcher. On macOS with Metal, this
        // spawns compile_to_cimage threads; without Metal, compilation will
        // fall back to a CPU-only path.
        let has_metal = crate::cli::model::has_metal_gpu();
        let compiler =
            dispatcher::DaemonCompilerDispatcher::new(has_metal, measured_evaluator.clone())
                .with_provenance(dispatcher::CompilerProvenance {
                    artifacts: artifact_store.clone(),
                    evidence: evidence_ledger.clone(),
                    projections: projection_store.clone(),
                });
        let dispatcher: std::sync::Arc<dyn prism_ecs_runtime::WorkDispatcher> =
            std::sync::Arc::new(compiler);
        schedule.set_dispatcher(dispatcher);
        schedule.bind(&kernel_handle);
        kernel.set_schedule(schedule);
    }

    // Register tools
    let mut tools_map = tools::register_basic(model_registry.clone())?;

    // Shared state
    let conn_count = Arc::new(AtomicU64::new(0));
    let idle_gen = Arc::new(AtomicU64::new(0));

    let job_manager: Arc<dyn prism_mcp_core::JobStore> =
        crate::daemon::trifecta_store::PostgresJobStore::connect(postgres_url)?;
    let coordination_store: Option<Arc<dyn prism_mcp_core::CoordinationStore>> =
        Some(crate::daemon::trifecta_store::PostgresCoordinationStore::connect(postgres_url)?);
    let experiment_store: Arc<dyn prism_mcp_core::ExperimentStore> =
        crate::daemon::trifecta_store::PostgresExperimentStore::connect(postgres_url)?;
    let benchmark_store: Arc<dyn prism_mcp_core::BenchmarkStore> =
        crate::daemon::trifecta_store::PostgresBenchmarkStore::connect(postgres_url)?;
    let knowledge_store: Arc<dyn prism_mcp_core::KnowledgeStore> =
        crate::daemon::trifecta_store::PostgresKnowledgeStore::connect(postgres_url)?;

    // Build DaemonState with Phase 1 tools, then register stateful handlers
    let partial_tools = Arc::new(tools_map.clone());
    let mut state = DaemonState {
        semantic_admission: Arc::new(prism_mcp_core::SemanticAdmission::new()),
        coordination_store,
        tools: partial_tools.clone(),
        artifact_store: artifact_store.clone(),
        evidence_ledger: evidence_ledger.clone(),
        conversation_store,
        file_lock,
        work_journal,
        process_cache,
        scheduler_handle: SchedulerHandle {
            release_sender: work_tx_for_handler,
        },
        job_manager,
        resource_leases,
        projection_store: projection_store.clone(),
        provenance_store: provenance_store.clone(),
        experiment_store,
        benchmark_store,
        knowledge_store,
        connection_count: conn_count.clone(),
        idle_generation: idle_gen.clone(),
    };

    // Phase 2: register stateful crate handlers that need &DaemonState
    let stateful_map =
        tools::register_stateful(&state, kernel_handle.clone(), model_registry.clone())?;
    tools_map.extend(stateful_map);
    let tools = Arc::new(tools_map);
    state.tools = tools.clone();
    let state = Arc::new(state);

    // Start scheduler in its own thread
    let sched = Scheduler::new(work_rx, tools, state.clone());
    let health = HealthState {
        artifact_dir: artifact_dir.to_string(),
        scheduler_heartbeat_ms: sched.heartbeat(),
        work_tx: work_tx.clone(),
        connection_count: conn_count.clone(),
        backend_health,
        kernel: kernel_handle.clone(),
    };
    std::thread::spawn(move || {
        sched.run();
    });

    // ── Model registry (shared between MCP handlers and dashboard) ──────
    // The model_registry created at line ~209 is used by tools and dashboard.
    // dashboard thread receives a clone via DashboardState.
    let model_registry = model_registry.clone();

    // ── Dashboard HTTP server ───────────────────────────────────────────
    {
        let registry = model_registry.clone();
        let kh = kernel_handle.clone();
        let dashboard_artifact_dir = artifact_dir.to_string();
        let dashboard_socket_path = paths.socket_path.clone();
        let dashboard_artifact_store = artifact_store.clone();
        let dashboard_evidence_ledger = evidence_ledger.clone();
        let dashboard_projection_store = projection_store.clone();
        let dashboard_provenance_store = provenance_store.clone();
        let dashboard_graph_projection = graph_projection.clone();
        let workflow_store = projection_store.clone();
        let session_threads = Arc::new(parking_lot::Mutex::new(
            std::collections::HashMap::<u64, uuid::Uuid>::new(),
        ));
        let observation_server = ecs_inference.clone();
        std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().expect("dashboard tokio runtime");
            rt.block_on(async move {
                let (model_tx, _) = broadcast::channel::<Vec<String>>(16);
                let (compiler_lab_tx, _) = broadcast::channel::<serde_json::Value>(64);
                let auth_path = std::path::PathBuf::from(format!(
                    "{}/.prism/auth",
                    std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string())
                ));
                let auth_token = std::env::var("PRISM_DASHBOARD_AUTH_TOKEN")
                    .ok()
                    .or_else(|| {
                        std::fs::read_to_string(&auth_path)
                            .ok()
                            .map(|v| v.trim().to_string())
                    })
                    .filter(|v| !v.is_empty())
                    .unwrap_or_else(|| "local-prism-demo".to_string());
                let workflow_client = Arc::new(parking_lot::Mutex::new(WorkflowClient::new(
                    kh.clone(),
                    ProjectionWorkflowStore::new(workflow_store),
                    NoopWorkflowCancellation,
                )));
                let workflow_for_observation = workflow_client.clone();
                let threads_for_observation = session_threads.clone();
                observation_server.scheduler.set_observation_sink(move |receipt| {
                    let Some(thread_id) = threads_for_observation.lock().get(&receipt.session_id.0).copied() else {
                        return;
                    };
                    let _ = workflow_for_observation.lock().publish_runtime_observation(
                        thread_id,
                        receipt.dispatch_id.0,
                        receipt.session_id.0,
                        receipt.model_id,
                        receipt.modality,
                        receipt.status,
                        receipt.output_digest,
                        receipt.output_units,
                    );
                }).expect("install ECS workflow observation sink");
                let state = crate::daemon::dashboard::DashboardState {
                    registry,
                    ecs_inference,
                    session_threads,
                    model_tx,
                    compiler_lab_tx,
                    has_compiler_dispatcher: true,
                    world: kh.clone(),
                    socket_path: std::path::PathBuf::from(dashboard_socket_path),
                    artifact_dir: std::path::PathBuf::from(dashboard_artifact_dir),
                    artifact_store: dashboard_artifact_store,
                    evidence_ledger: dashboard_evidence_ledger,
                    projection_store: dashboard_projection_store,
                    provenance_store: dashboard_provenance_store,
                    graph_projection: dashboard_graph_projection,
                    workflow_client,
                    authorized: Arc::new(AtomicBool::new(auth_path.exists())),
                    auth_token: Arc::new(auth_token),
                };
                let app = crate::daemon::dashboard::router(state);
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

    // ── Kernel tick loop ────────────────────────────────────────────────
    let tick_running = Arc::new(AtomicBool::new(true));
    let tick_stop = tick_running.clone();
    let kernel_for_tick = kernel.clone();
    let tick_daemon_id = daemon_instance_id.clone();

    let tick_handle = std::thread::spawn(move || {
        while tick_running.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(100));
            if let Err(e) = kernel_for_tick.run_kernel_tick(&tick_daemon_id) {
                eprintln!("prism-mcpd: kernel tick error: {e}");
            }
        }
    });

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
    // ── Graceful shutdown ──────────────────────────────────────────────
    tick_stop.store(false, Ordering::Relaxed);
    let _ = tick_handle.join(); // wait for tick thread to fully stop

    // Now safe to capture final snapshot
    if let Err(e) = kernel.shutdown() {
        eprintln!("prism-mcpd: kernel shutdown error: {e}");
    }
    Ok(())
}
