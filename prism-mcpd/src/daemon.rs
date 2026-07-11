use std::io::{BufRead, BufReader, BufWriter, Write};
use std::os::unix::io::AsRawFd;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crossbeam_channel::{bounded, Receiver, Sender};
use prism_mcp_core::{
    ArtifactStore, ConnectionId, DaemonState, DbManager, EvidenceLedger, FileLock, JobManager,
    ProcessCache, RequestEnvelope, ResourceLeaseManager, ResponseFrame, Scheduler, SchedulerHandle,
    WorkJournal,
};

use crate::tools;

struct StatePaths {
    socket_path: String,
    lock_path: String,
    pid_path: String,
    file_lock_path: String,
    db_path: String,
    staging_dir: String,
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

    // Singleton lock
    let singleton_lock = FileLock::new(Path::new(&paths.lock_path));
    let _singleton_guard = singleton_lock.try_lock()?.ok_or_else(|| {
        anyhow::anyhow!(
            "another daemon is already running (lock held at {})",
            paths.lock_path
        )
    })?;
    let _ = std::fs::remove_file(&paths.socket_path);

    // Open database
    let db_path = Path::new(&paths.db_path);
    let schema = include_str!("../migrations/001_schema.sql");
    let db = Arc::new(DbManager::open(db_path, schema, 8)?);

    let artifact_store = ArtifactStore::open(Path::new(artifact_dir))?;
    let evidence_ledger = EvidenceLedger::open(db.clone())?; // Arc clone — shares the DbManager
    let file_lock = FileLock::new(Path::new(&paths.file_lock_path));
    let work_journal = WorkJournal::new(Path::new(&paths.staging_dir));
    let process_cache = ProcessCache::new(32);

    // Global work queue: reader threads push, scheduler pulls
    let (work_tx, work_rx) = bounded::<RequestEnvelope>(256);
    let work_tx_for_handler = work_tx.clone();

    // Register tools
    let mut tools_map = tools::register_basic()?;

    // Shared state
    let conn_count = Arc::new(AtomicU64::new(0));
    let idle_gen = Arc::new(AtomicU64::new(0));

    // Create job manager and resource leases
    let job_manager = JobManager::new(db.clone())?;
    let resource_leases = ResourceLeaseManager::new();

    // Build DaemonState with Phase 1 tools, then register stateful handlers
    let partial_tools = Arc::new(tools_map.clone());
    let mut state = DaemonState {
        tools: partial_tools.clone(),
        db,
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
    std::thread::spawn(move || {
        sched.run();
    });

    // Write PID
    let _ = std::fs::write(&paths.pid_path, format!("{}", std::process::id()));

    // Bind and listen
    let listener = UnixListener::bind(&paths.socket_path)?;
    eprintln!("prism-mcpd: listening on {}", paths.socket_path);

    // Accept loop (non-blocking)
    listener.set_nonblocking(true)?;
    // Track connections to restore blocking mode after accept
    loop {
        if conn_count.load(Ordering::Relaxed) == 0 {
            idle_gen.fetch_add(1, Ordering::Relaxed);
            std::thread::sleep(Duration::from_secs(1));
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
                let conn_id = ConnectionId::new();
                let (response_tx, response_rx) = bounded::<ResponseFrame>(64);

                std::thread::spawn(move || {
                    handle_connection(stream, conn_id, work_tx_clone, response_tx, response_rx);
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
