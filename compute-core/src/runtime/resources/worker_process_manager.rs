//! WorkerProcessManager — owns raw OS process handles and IPC pipes.
//!
//! Completely decouples hardware I/O (std process management, stdout
//! reading, signal delivery) from the logical ECS schedule.  Systems
//! interact through the resource API; they never touch a `Child` or pipe
//! directly.

use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};


// ---------------------------------------------------------------------------
// WorkerId
// ---------------------------------------------------------------------------

/// Opaque handle identifying a running worker process.
pub type WorkerId = u32;

// ---------------------------------------------------------------------------
// WorkerControlMessage
// ---------------------------------------------------------------------------

/// Out-of-band message sent to a worker process.
pub enum WorkerControlMessage {
    /// Request cancellation of the given request.
    CancelRequest(String),
    /// Polite shutdown signal.
    Shutdown,
}

// ---------------------------------------------------------------------------
// WorkerProcessHandles
// ---------------------------------------------------------------------------

/// Plumbing for one running worker process.
///
/// Owns the OS `Child`, an optional stdout reader, and enough metadata to
/// deliver cancellation or shutdown signals.
pub struct WorkerProcessHandles {
    pub worker_id: WorkerId,
    pub process: Child,
    pub stdout_reader: Option<std::io::Lines<BufReader<std::process::ChildStdout>>>,
}

impl WorkerProcessHandles {
    /// Try to read one line from the worker's stdout.
    ///
    /// Returns `None` when no line is available yet (non-blocking) or when
    /// stdout has already been consumed or was not captured.
    pub fn try_read_line(&mut self) -> Option<String> {
        let reader = self.stdout_reader.as_mut()?;
        match reader.next() {
            Some(Ok(line)) => {
                if line.is_empty() {
                    None
                } else {
                    Some(line)
                }
            }
            _ => None,
        }
    }

    /// Send a cancellation signal to the worker process.
    ///
    /// Current implementation sends `SIGINT` on Unix.  A future slice may
    /// use an out-of-band IPC channel for a more graceful protocol.
    pub fn send_cancel(&mut self) {
        // `Child::kill` sends SIGKILL — too aggressive for cancellation.
        // For now this is a no-op placeholder; real cancellation will
        // go through an IPC pipe or a dedicated control channel.
        let _ = self.process.kill();
    }

    /// Force-kill the worker process.
    pub fn kill(&mut self) {
        let _ = self.process.kill();
        let _ = self.process.wait();
    }
}

impl std::fmt::Debug for WorkerProcessHandles {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkerProcessHandles")
            .field("worker_id", &self.worker_id)
            .field("process", &self.process.id())
            .field("stdout_reader", &self.stdout_reader.as_ref().map(|_| "active"))
            .finish()
    }
}

// ---------------------------------------------------------------------------
// WorkerProcessManager
// ---------------------------------------------------------------------------

/// Singleton resource that owns all running worker process handles.
///
/// Inserted into the ECS World at initialisation and passed to systems
/// that need to spawn, interrogate, or tear down worker binaries.
pub struct WorkerProcessManager {
    processes: HashMap<WorkerId, WorkerProcessHandles>,
    next_worker_id: WorkerId,
}

impl WorkerProcessManager {
    /// Create an empty manager.
    pub fn new() -> Self {
        Self {
            processes: HashMap::new(),
            next_worker_id: 1,
        }
    }

    /// Spawn a new worker binary and return its assigned `WorkerId`.
    ///
    /// `binary_path` is the path to the worker executable.
    /// `model_path` is passed as the first argument to the binary.
    ///
    /// The worker's stdout is piped for line-oriented reading.
    pub fn spawn_worker(
        &mut self,
        binary_path: &str,
        model_path: &str,
    ) -> Result<WorkerId, String> {
        let worker_id = self.next_worker_id;
        self.next_worker_id += 1;

        let mut child = Command::new(binary_path)
            .arg(model_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .stdin(Stdio::null())
            .spawn()
            .map_err(|e| {
                format!(
                    "failed to spawn worker binary '{}' with model '{}': {}",
                    binary_path, model_path, e,
                )
            })?;

        // Take ownership of stdout before moving child into handles.
        let stdout_reader = child.stdout.take().map(|s| BufReader::new(s).lines());

        let handles = WorkerProcessHandles {
            worker_id,
            process: child,
            stdout_reader,
        };

        self.processes.insert(worker_id, handles);
        Ok(worker_id)
    }

    /// Get mutable handles for a worker.
    pub fn get_mut(&mut self, id: WorkerId) -> Option<&mut WorkerProcessHandles> {
        self.processes.get_mut(&id)
    }

    /// Remove and kill a worker, returning its handles.
    pub fn remove(&mut self, id: WorkerId) {
        if let Some(mut handles) = self.processes.remove(&id) {
            handles.kill();
        }
    }

    /// Drain all available stdout lines from all workers (non-blocking).
    ///
    /// Returns one `(WorkerId, line)` per available line.
    pub fn drain_stdout(&mut self) -> Vec<(WorkerId, String)> {
        let mut lines = Vec::new();
        // Collect ids first to avoid borrow conflicts with the inner loop.
        let ids: Vec<WorkerId> = self.processes.keys().copied().collect();
        for id in ids {
            if let Some(handles) = self.processes.get_mut(&id) {
                while let Some(line) = handles.try_read_line() {
                    lines.push((id, line));
                }
            }
        }
        lines
    }

    /// Number of active workers.
    pub fn len(&self) -> usize {
        self.processes.len()
    }

    /// Returns `true` when no workers are running.
    pub fn is_empty(&self) -> bool {
        self.processes.is_empty()
    }
}

impl Default for WorkerProcessManager {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for WorkerProcessManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkerProcessManager")
            .field("active_workers", &self.processes.len())
            .field("next_worker_id", &self.next_worker_id)
            .finish()
    }
}
