use parking_lot::Mutex;
use prism_mcp_core::{
    DaemonState, JobId, JobProgress, JobState, JobStore, McpHandler, RequestContext, ToolRequest,
    ToolResult,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

const CACHE_FORMAT_VERSION: u32 = 2;
const MAX_CACHED_STREAM_BYTES: usize = 1024 * 1024;
const MAX_CACHE_ENTRIES: usize = 256;
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(2);

enum ManagedChild {
    Owned(Child),
    Adopted,
}

struct ManagedJob {
    job_id: String,
    child: ManagedChild,
    command: String,
    stdout_path: PathBuf,
    stderr_path: PathBuf,
    exit_path: PathBuf,
    manifest_path: PathBuf,
    cwd: PathBuf,
    fingerprint: String,
    pid: u32,
    process_group: i32,
    process_identity: String,
    started_at: chrono::DateTime<chrono::Utc>,
    deadline_at: Option<chrono::DateTime<chrono::Utc>>,
    last_output_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProcessManifest {
    format_version: u32,
    job_id: String,
    command: String,
    cwd: PathBuf,
    fingerprint: String,
    pid: u32,
    process_group: i32,
    process_identity: String,
    daemon_instance: String,
    stdout_path: PathBuf,
    stderr_path: PathBuf,
    exit_path: PathBuf,
    started_at: chrono::DateTime<chrono::Utc>,
    #[serde(default)]
    deadline_at: Option<chrono::DateTime<chrono::Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CachedResult {
    format_version: u32,
    fingerprint: String,
    command: String,
    cwd: String,
    exit_code: i32,
    stdout: String,
    stderr: String,
    duration_ms: u64,
    created_at: chrono::DateTime<chrono::Utc>,
}

static JOBS: OnceLock<Mutex<HashMap<String, ManagedJob>>> = OnceLock::new();
static SUPERVISOR: OnceLock<()> = OnceLock::new();

fn jobs() -> &'static Mutex<HashMap<String, ManagedJob>> {
    JOBS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn daemon_instance() -> String {
    static INSTANCE: OnceLock<String> = OnceLock::new();
    INSTANCE
        .get_or_init(|| format!("{}-{}", std::process::id(), uuid::Uuid::new_v4()))
        .clone()
}

fn job_state_dir() -> PathBuf {
    let state = std::env::var_os("PRISM_MCPD_STATE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").unwrap_or_else(|| ".".into()))
                .join(".local/state/prism-mcpd")
        });
    state.join("jobs")
}

fn process_identity(pid: u32) -> Option<String> {
    let output = Command::new("/bin/ps")
        .args(["-p", &pid.to_string(), "-o", "lstart="])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let value = String::from_utf8(output.stdout).ok()?.trim().to_string();
    (!value.is_empty()).then_some(value)
}

fn process_is_live(pid: u32, expected_identity: &str) -> bool {
    process_identity(pid).as_deref() == Some(expected_identity)
}

fn terminate_process_group(process_group: i32, pid: u32, identity: &str) {
    unsafe {
        libc::kill(-process_group, libc::SIGTERM);
    }
    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    while std::time::Instant::now() < deadline {
        if !process_is_live(pid, identity) {
            return;
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    unsafe {
        libc::kill(-process_group, libc::SIGKILL);
    }
}

fn atomic_json<T: Serialize>(path: &Path, value: &T) -> anyhow::Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("path has no parent: {}", path.display()))?;
    std::fs::create_dir_all(parent)?;
    let temporary = parent.join(format!(
        ".{}.{}.tmp",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("state"),
        std::process::id()
    ));
    std::fs::write(&temporary, serde_json::to_vec_pretty(value)?)?;
    std::fs::rename(temporary, path)?;
    Ok(())
}

fn read_exit_code(path: &Path) -> Option<i32> {
    std::fs::read_to_string(path).ok()?.trim().parse().ok()
}

fn output_bytes(job: &ManagedJob) -> u64 {
    [&job.stdout_path, &job.stderr_path]
        .iter()
        .filter_map(|path| std::fs::metadata(path).ok())
        .map(|metadata| metadata.len())
        .sum()
}

fn output_summary(command: &str, stdout: &str, stderr: &str) -> Value {
    if !command.starts_with("cargo ") && !command.contains(" cargo ") {
        return json!({
            "format": "plain",
            "stdout_tail": tail(stdout),
            "stderr_tail": tail(stderr),
        });
    }
    let mut artifacts = 0_u64;
    let mut build_scripts = 0_u64;
    let mut diagnostics = Vec::new();
    let mut plain = Vec::new();
    for line in stdout.lines().chain(stderr.lines()) {
        let Ok(value) = serde_json::from_str::<Value>(line) else {
            if !line.trim().is_empty() {
                plain.push(line.to_string());
            }
            continue;
        };
        match value.get("reason").and_then(Value::as_str) {
            Some("compiler-artifact") => artifacts += 1,
            Some("build-script-executed") => build_scripts += 1,
            Some("compiler-message") => {
                let message = &value["message"];
                let level = message["level"].as_str().unwrap_or("unknown");
                if matches!(level, "warning" | "error" | "failure-note") {
                    diagnostics.push(json!({
                        "level": level,
                        "message": message["message"],
                        "rendered": message["rendered"],
                    }));
                }
            }
            _ => {}
        }
    }
    let plain_start = plain.len().saturating_sub(50);
    let diagnostic_start = diagnostics.len().saturating_sub(100);
    json!({
        "format": "cargo-json",
        "compiler_artifacts": artifacts,
        "build_scripts": build_scripts,
        "diagnostic_count": diagnostics.len(),
        "diagnostics": &diagnostics[diagnostic_start..],
        "messages_tail": &plain[plain_start..],
    })
}

fn command_output(command: &str, args: &[&str], cwd: &Path) -> Option<Vec<u8>> {
    let output = Command::new(command)
        .args(args)
        .current_dir(cwd)
        .output()
        .ok()?;
    output.status.success().then_some(output.stdout)
}

fn append_file(material: &mut Vec<u8>, path: &Path) {
    if let Ok(bytes) = std::fs::read(path) {
        material.extend_from_slice(path.to_string_lossy().as_bytes());
        material.push(0);
        material.extend_from_slice(&bytes);
        material.push(0);
    }
}

fn cache_key(command: &str, cwd: &Path) -> anyhow::Result<Option<String>> {
    if !command.starts_with("cargo ") && !command.contains(" cargo ") {
        return Ok(None);
    }
    let cwd = cwd.canonicalize()?;
    let mut material = format!(
        "prism-cargo-cache-v{CACHE_FORMAT_VERSION}\0{command}\0{}\0",
        cwd.display()
    )
    .into_bytes();
    for (program, args) in [
        ("cargo", &["-V"] as &[&str]),
        ("rustc", &["-vV"] as &[&str]),
    ] {
        if let Some(output) = command_output(program, args, &cwd) {
            material.extend_from_slice(&output);
            material.push(0);
        }
    }
    for name in [
        "RUSTFLAGS",
        "RUSTDOCFLAGS",
        "CARGO_BUILD_TARGET",
        "CARGO_TARGET_DIR",
        "CC",
        "CXX",
        "SDKROOT",
        "MACOSX_DEPLOYMENT_TARGET",
    ] {
        if let Some(value) = std::env::var_os(name) {
            material.extend_from_slice(name.as_bytes());
            material.push(b'=');
            material.extend_from_slice(value.to_string_lossy().as_bytes());
            material.push(0);
        }
    }
    let repo_root = command_output("git", &["rev-parse", "--show-toplevel"], &cwd)
        .and_then(|bytes| String::from_utf8(bytes).ok())
        .map(|value| PathBuf::from(value.trim()));
    if let Some(root) = repo_root.as_deref() {
        for args in [
            &["rev-parse", "HEAD"] as &[&str],
            &["status", "--porcelain=v2", "-z", "--untracked-files=all"],
            &["diff", "--binary", "--no-ext-diff", "HEAD", "--"],
            &["submodule", "status", "--recursive"],
        ] {
            if let Some(output) = command_output("git", args, root) {
                material.extend_from_slice(&output);
                material.push(0);
            }
        }
        if let Some(untracked) = command_output(
            "git",
            &["ls-files", "--others", "--exclude-standard", "-z"],
            root,
        ) {
            for relative in untracked
                .split(|byte| *byte == 0)
                .filter(|path| !path.is_empty())
            {
                if let Ok(relative) = std::str::from_utf8(relative) {
                    append_file(&mut material, &root.join(relative));
                }
            }
        }
        for relative in [
            ".cargo/config",
            ".cargo/config.toml",
            "rust-toolchain",
            "rust-toolchain.toml",
        ] {
            append_file(&mut material, &root.join(relative));
        }
    } else {
        for relative in [
            "Cargo.toml",
            "Cargo.lock",
            "build.rs",
            "rust-toolchain.toml",
        ] {
            append_file(&mut material, &cwd.join(relative));
        }
    }
    Ok(Some(blake3::hash(&material).to_hex().to_string()))
}

fn cache_dir() -> PathBuf {
    let state = std::env::var_os("PRISM_MCPD_STATE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").unwrap_or_else(|| ".".into()))
                .join(".local/state/prism-mcpd")
        });
    state.join("build-cache/v2")
}

fn load_cached(fingerprint: &str) -> Option<CachedResult> {
    let bytes = std::fs::read(cache_dir().join(format!("{fingerprint}.json"))).ok()?;
    let entry: CachedResult = serde_json::from_slice(&bytes).ok()?;
    (entry.format_version == CACHE_FORMAT_VERSION
        && entry.fingerprint == fingerprint
        && entry.exit_code == 0)
        .then_some(entry)
}

fn bounded(value: &str) -> String {
    if value.len() <= MAX_CACHED_STREAM_BYTES {
        return value.to_string();
    }
    let mut start = value.len() - MAX_CACHED_STREAM_BYTES;
    while !value.is_char_boundary(start) {
        start += 1;
    }
    value[start..].to_string()
}

fn store_cached(entry: &CachedResult) -> anyhow::Result<()> {
    let dir = cache_dir();
    std::fs::create_dir_all(&dir)?;
    let destination = dir.join(format!("{}.json", entry.fingerprint));
    let temporary = dir.join(format!(".{}.{}.tmp", entry.fingerprint, std::process::id()));
    std::fs::write(&temporary, serde_json::to_vec(entry)?)?;
    std::fs::rename(temporary, destination)?;
    prune_cache(&dir);
    Ok(())
}

fn prune_cache(dir: &Path) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<_> = entries
        .flatten()
        .filter_map(|entry| {
            let metadata = entry.metadata().ok()?;
            metadata
                .is_file()
                .then(|| (metadata.modified().ok(), entry.path()))
        })
        .collect();
    if entries.len() <= MAX_CACHE_ENTRIES {
        return;
    }
    entries.sort_by_key(|(modified, _)| *modified);
    let remove_count = entries.len() - MAX_CACHE_ENTRIES;
    for (_, path) in entries.into_iter().take(remove_count) {
        let _ = std::fs::remove_file(path);
    }
}

fn finish_job(store: &dyn JobStore, job: ManagedJob, exit_code: i32, lost: bool) {
    let id = match uuid::Uuid::parse_str(&job.job_id) {
        Ok(id) => JobId(id),
        Err(error) => {
            tracing::error!(job_id = %job.job_id, %error, "invalid supervised job id");
            return;
        }
    };
    let elapsed = (chrono::Utc::now() - job.started_at)
        .num_milliseconds()
        .max(0) as u64;
    let output = read_output(&job.stdout_path, &job.stderr_path);
    if exit_code == 0 && !job.fingerprint.is_empty() {
        match cache_key(&job.command, &job.cwd) {
            Ok(Some(current)) if current == job.fingerprint => {
                let entry = CachedResult {
                    format_version: CACHE_FORMAT_VERSION,
                    fingerprint: job.fingerprint.clone(),
                    command: job.command.clone(),
                    cwd: job.cwd.display().to_string(),
                    exit_code,
                    stdout: bounded(&output.0),
                    stderr: bounded(&output.1),
                    duration_ms: elapsed,
                    created_at: chrono::Utc::now(),
                };
                if let Err(error) = store_cached(&entry) {
                    tracing::warn!(job_id = %job.job_id, %error, "failed to persist job cache entry");
                }
            }
            Ok(_) => {}
            Err(error) => {
                tracing::warn!(job_id = %job.job_id, %error, "failed to recompute job fingerprint");
            }
        }
    }
    let state = if exit_code == 0 {
        JobState::Succeeded
    } else if lost {
        JobState::Failed("supervised process disappeared without an exit record".into())
    } else {
        JobState::Failed(format!("process exited with code {exit_code}"))
    };
    if let Err(error) = store.update_state(&id, state) {
        tracing::error!(job_id = %job.job_id, %error, "failed to persist terminal job state");
        return;
    }
    let event = if lost { "process_lost" } else { "process_exit" };
    let _ = store.push_event(
        &id,
        event,
        &format!(
            "pid={} exit_code={} elapsed_ms={elapsed}",
            job.pid, exit_code
        ),
    );
    let _ = std::fs::remove_file(&job.manifest_path);
}

fn supervise_once(store: &dyn JobStore) {
    let mut completed = Vec::new();
    let mut heartbeats = Vec::new();
    {
        let mut managed = jobs().lock();
        let keys: Vec<String> = managed.keys().cloned().collect();
        for key in keys {
            let Some(job) = managed.get_mut(&key) else {
                continue;
            };
            let mut completion = read_exit_code(&job.exit_path).map(|code| (code, false));
            if completion.is_none()
                && job
                    .deadline_at
                    .is_some_and(|deadline| chrono::Utc::now() >= deadline)
            {
                terminate_process_group(job.process_group, job.pid, &job.process_identity);
                completion = Some((124, false));
            }
            if completion.is_none() {
                match &mut job.child {
                    ManagedChild::Owned(child) => match child.try_wait() {
                        Ok(Some(status)) => completion = Some((status.code().unwrap_or(-1), false)),
                        Ok(None) => {}
                        Err(error) => {
                            tracing::warn!(job_id = %job.job_id, %error, "failed to poll child");
                        }
                    },
                    ManagedChild::Adopted => {
                        if !process_is_live(job.pid, &job.process_identity) {
                            completion = Some((-1, true));
                        }
                    }
                }
            }
            if let Some((exit_code, lost)) = completion {
                if let Some(job) = managed.remove(&key) {
                    completed.push((job, exit_code, lost));
                }
                continue;
            }
            let bytes = output_bytes(job);
            let changed = bytes != job.last_output_bytes;
            job.last_output_bytes = bytes;
            heartbeats.push((job.job_id.clone(), job.pid, bytes, changed, job.started_at));
        }
    }
    for (raw, pid, bytes, changed, started_at) in heartbeats {
        let Ok(uuid) = uuid::Uuid::parse_str(&raw) else {
            continue;
        };
        let elapsed = (chrono::Utc::now() - started_at).num_seconds().max(0);
        let activity = if changed {
            "output advanced"
        } else {
            "compiler active"
        };
        let _ = store.update_progress(
            &JobId(uuid),
            JobProgress {
                message: format!(
                    "{activity}; pid={pid}; elapsed_secs={elapsed}; output_bytes={bytes}"
                ),
                percent: 0.0,
            },
        );
    }
    for (job, exit_code, lost) in completed {
        finish_job(store, job, exit_code, lost);
    }
}

fn reconcile_jobs(store: &dyn JobStore) {
    let records = match store.list_jobs(Some("run_job")) {
        Ok(records) => records,
        Err(error) => {
            tracing::error!(%error, "failed to list jobs for supervision reconciliation");
            return;
        }
    };
    for record in records {
        if record.state != JobState::Running {
            continue;
        }
        let key = record.id.to_string();
        let manifest_path = job_state_dir().join(&key).join("manifest.json");
        let manifest = std::fs::read(&manifest_path)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<ProcessManifest>(&bytes).ok());
        let Some(manifest) = manifest else {
            let _ = store.update_state(
                &record.id,
                JobState::Failed("running job has no durable process manifest".into()),
            );
            let _ = store.push_event(
                &record.id,
                "reconciliation_failed",
                "durable process manifest missing or invalid",
            );
            continue;
        };
        let has_exit = read_exit_code(&manifest.exit_path).is_some();
        let live = process_is_live(manifest.pid, &manifest.process_identity);
        if !has_exit && !live {
            let _ = store.update_state(
                &record.id,
                JobState::Failed(
                    "supervised process is no longer live and has no exit record".into(),
                ),
            );
            let _ = store.push_event(
                &record.id,
                "reconciliation_failed",
                "pid identity did not match and no exit record was present",
            );
            continue;
        }
        jobs().lock().insert(
            key.clone(),
            ManagedJob {
                job_id: key,
                child: ManagedChild::Adopted,
                command: manifest.command,
                stdout_path: manifest.stdout_path,
                stderr_path: manifest.stderr_path,
                exit_path: manifest.exit_path,
                manifest_path,
                cwd: manifest.cwd,
                fingerprint: manifest.fingerprint,
                pid: manifest.pid,
                process_group: manifest.process_group,
                process_identity: manifest.process_identity,
                started_at: manifest.started_at,
                deadline_at: manifest.deadline_at,
                last_output_bytes: 0,
            },
        );
        let _ = store.push_event(
            &record.id,
            "adopted",
            &format!("daemon {} adopted pid {}", daemon_instance(), manifest.pid),
        );
    }
    supervise_once(store);
}

fn start_supervisor(store: Arc<dyn JobStore>) {
    SUPERVISOR.get_or_init(|| {
        reconcile_jobs(store.as_ref());
        std::thread::Builder::new()
            .name("prism-job-supervisor".into())
            .spawn(move || loop {
                std::thread::sleep(HEARTBEAT_INTERVAL);
                supervise_once(store.as_ref());
            })
            .expect("spawn prism job supervisor");
    });
}

pub struct JobHandler {
    store: Arc<dyn JobStore>,
}

pub struct TestScopeJobHandler {
    jobs: JobHandler,
}

impl TestScopeJobHandler {
    pub fn new(store: Arc<dyn JobStore>) -> Self {
        Self {
            jobs: JobHandler::new(store),
        }
    }
}

fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}

impl McpHandler for TestScopeJobHandler {
    fn name(&self) -> &'static str {
        "test_scope"
    }

    fn description(&self) -> &'static str {
        "Start a daemon-supervised Cargo test scope and return its durable job identity."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "component": {"type": "string"},
                "scope": {"type": "string", "enum": ["auto", "all"], "default": "auto"},
                "timeout_secs": {"type": "integer", "minimum": 1, "maximum": 86400, "default": 600},
                "workspace_root": {"type": "string", "description": "Absolute Cargo workspace root."}
            },
            "additionalProperties": false
        })
    }

    fn call(
        &self,
        request: ToolRequest<'_>,
        context: &RequestContext,
        state: &DaemonState,
    ) -> anyhow::Result<ToolResult> {
        let workspace = request
            .args
            .get("workspace_root")
            .and_then(Value::as_str)
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .parent()
                    .expect("prism-mcpd workspace parent")
                    .to_path_buf()
            })
            .canonicalize()?;
        if !workspace.join("Cargo.toml").is_file() {
            anyhow::bail!(
                "workspace_root does not contain Cargo.toml: {}",
                workspace.display()
            );
        }
        let scope = request
            .args
            .get("scope")
            .and_then(Value::as_str)
            .unwrap_or("auto");
        let command = if scope == "all" {
            "cargo test --workspace --message-format=json --color never".to_string()
        } else if let Some(component) = request.args.get("component").and_then(Value::as_str) {
            format!(
                "cargo test --message-format=json --color never -p {}",
                shell_quote(component)
            )
        } else {
            "cargo test --message-format=json --color never".to_string()
        };
        let args = json!({
            "action": "start",
            "command": command,
            "cwd": workspace,
            "timeout_secs": request.args.get("timeout_secs").and_then(Value::as_u64).unwrap_or(600),
        });
        self.jobs.call(ToolRequest { args: &args }, context, state)
    }
}

impl JobHandler {
    pub fn new(store: Arc<dyn JobStore>) -> Self {
        start_supervisor(store.clone());
        Self { store }
    }

    fn result(value: Value) -> anyhow::Result<ToolResult> {
        Ok(ToolResult::text(serde_json::to_string(&value)?))
    }

    fn status(&self, id: &JobId) -> anyhow::Result<ToolResult> {
        supervise_once(self.store.as_ref());
        let key = id.to_string();
        let record = self.store.get_job(id)?;
        let manifest_path = job_state_dir().join(&key).join("manifest.json");
        let manifest = std::fs::read(&manifest_path)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<ProcessManifest>(&bytes).ok());
        let (stdout_path, stderr_path) = manifest
            .as_ref()
            .map(|manifest| (manifest.stdout_path.clone(), manifest.stderr_path.clone()))
            .unwrap_or_else(|| {
                let dir = job_state_dir().join(&key);
                (dir.join("stdout.log"), dir.join("stderr.log"))
            });
        let output = read_output(&stdout_path, &stderr_path);
        let exit_path = manifest
            .as_ref()
            .map(|manifest| manifest.exit_path.clone())
            .unwrap_or_else(|| job_state_dir().join(&key).join("exit-code"));
        let exit_code = read_exit_code(&exit_path);
        let summary = output_summary(&record.operation, &output.0, &output.1);
        let duration_ms = manifest.as_ref().map(|manifest| {
            (chrono::Utc::now() - manifest.started_at)
                .num_milliseconds()
                .max(0) as u64
        });
        Self::result(
            json!({"job_id":key,"status":record.state.as_str(),"phase":if record.state == JobState::Running { "running" } else { "terminal" },"command":record.operation,"cwd":manifest.as_ref().map(|value| value.cwd.display().to_string()),"fingerprint":manifest.as_ref().map(|value| &value.fingerprint),"pid":manifest.as_ref().map(|value| value.pid),"process_group":manifest.as_ref().map(|value| value.process_group),"daemon_instance":manifest.as_ref().map(|value| &value.daemon_instance),"exit_code":exit_code,"duration_ms":duration_ms,"progress":record.progress.as_ref().map(|value| json!({"message":value.message,"percent":value.percent})),"stdout_bytes":output.0.len(),"stderr_bytes":output.1.len(),"summary":summary,"updated_at":record.updated_at}),
        )
    }
}

impl McpHandler for JobHandler {
    fn name(&self) -> &'static str {
        "run_job"
    }
    fn description(&self) -> &'static str {
        "Start, inspect, or cancel a daemon-supervised shell job."
    }
    fn input_schema(&self) -> Value {
        json!({"type":"object","properties":{"action":{"type":"string","enum":["start","status","cancel"]},"command":{"type":"string"},"job_id":{"type":"string"},"cwd":{"type":"string","description":"Absolute workspace directory. Required for start; the persistent daemon never infers this from its own CWD."},"timeout_secs":{"type":"integer","minimum":1,"maximum":86400,"description":"Optional supervisor-enforced execution deadline."}},"required":["action"]})
    }

    fn call(
        &self,
        request: ToolRequest<'_>,
        _context: &RequestContext,
        state: &DaemonState,
    ) -> anyhow::Result<ToolResult> {
        let action = request
            .args
            .get("action")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("action is required"))?;
        match action {
            "start" => {
                let command = request
                    .args
                    .get("command")
                    .and_then(Value::as_str)
                    .filter(|v| !v.trim().is_empty())
                    .ok_or_else(|| anyhow::anyhow!("command is required"))?;
                let cwd = request
                    .args
                    .get("cwd")
                    .and_then(Value::as_str)
                    .map(std::path::PathBuf::from)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "cwd is required when starting a job; daemon CWD inference is disabled"
                        )
                    })?;
                if !cwd.is_absolute() {
                    anyhow::bail!("cwd must be absolute: {}", cwd.display());
                }
                let cwd = cwd.canonicalize().map_err(|error| {
                    anyhow::anyhow!("cannot resolve cwd {}: {error}", cwd.display())
                })?;
                if !cwd.is_dir() {
                    anyhow::bail!(
                        "cwd does not exist or is not a directory: {}",
                        cwd.display()
                    );
                }
                let cwd_display = cwd.display().to_string();
                let key = cache_key(command, &cwd)?;
                if let Some(key_value) = &key {
                    if let Some(entry) = load_cached(key_value) {
                        let summary = output_summary(command, &entry.stdout, &entry.stderr);
                        return Self::result(
                            json!({"status":"cache_hit","phase":"terminal","cache_key":key_value,"fingerprint":key_value,"command":command,"cwd":cwd_display,"exit_code":entry.exit_code,"duration_ms":entry.duration_ms,"created_at":entry.created_at,"stdout_bytes":entry.stdout.len(),"stderr_bytes":entry.stderr.len(),"summary":summary}),
                        );
                    }
                    if let Some(existing) = jobs()
                        .lock()
                        .values()
                        .find(|job| job.fingerprint == *key_value)
                    {
                        return Self::result(
                            json!({"status":"already_running","phase":"running","job_id":existing.job_id,"command":command,"cwd":cwd_display,"cache_key":key_value,"fingerprint":key_value}),
                        );
                    }
                }
                let id = self.store.create_job("run_job", command)?;
                let dir = job_state_dir().join(id.to_string());
                std::fs::create_dir_all(&dir)?;
                let stdout_path = dir.join("stdout.log");
                let stderr_path = dir.join("stderr.log");
                let exit_path = dir.join("exit-code");
                let manifest_path = dir.join("manifest.json");
                let _ = std::fs::remove_file(&exit_path);
                let stdout = OpenOptions::new()
                    .create(true)
                    .truncate(true)
                    .write(true)
                    .open(&stdout_path)?;
                let stderr = OpenOptions::new()
                    .create(true)
                    .truncate(true)
                    .write(true)
                    .open(&stderr_path)?;
                let wrapper = r#"set +e
/bin/zsh -lc "$2"
code=$?
tmp="$1.tmp.$$"
printf '%s\n' "$code" > "$tmp"
mv -f "$tmp" "$1"
exit "$code""#;
                let mut process = Command::new("/bin/zsh");
                process
                    .args([
                        "-c",
                        wrapper,
                        "prism-job-wrapper",
                        &exit_path.to_string_lossy(),
                        command,
                    ])
                    .current_dir(&cwd)
                    .stdout(Stdio::from(stdout))
                    .stderr(Stdio::from(stderr));
                unsafe {
                    process.pre_exec(|| {
                        if libc::setpgid(0, 0) == -1 {
                            return Err(std::io::Error::last_os_error());
                        }
                        Ok(())
                    });
                }
                let child = process.spawn()?;
                let pid = child.id();
                let Some(identity) = process_identity(pid) else {
                    unsafe {
                        libc::kill(-(pid as i32), libc::SIGKILL);
                    }
                    anyhow::bail!("could not establish process identity for pid {pid}");
                };
                let started_at = chrono::Utc::now();
                let deadline_at = request
                    .args
                    .get("timeout_secs")
                    .and_then(Value::as_u64)
                    .map(|seconds| {
                        started_at + chrono::Duration::seconds(seconds.min(86_400) as i64)
                    });
                let manifest = ProcessManifest {
                    format_version: 1,
                    job_id: id.to_string(),
                    command: command.to_string(),
                    cwd: cwd.clone(),
                    fingerprint: key.clone().unwrap_or_default(),
                    pid,
                    process_group: pid as i32,
                    process_identity: identity.clone(),
                    daemon_instance: daemon_instance(),
                    stdout_path: stdout_path.clone(),
                    stderr_path: stderr_path.clone(),
                    exit_path: exit_path.clone(),
                    started_at,
                    deadline_at,
                };
                if let Err(error) = atomic_json(&manifest_path, &manifest) {
                    unsafe {
                        libc::kill(-(pid as i32), libc::SIGKILL);
                    }
                    return Err(error.context("persist supervised process manifest"));
                }
                self.store.update_state(&id, JobState::Running)?;
                self.store.push_event(
                    &id,
                    "process_start",
                    &format!(
                        "pid={pid} process_group={pid} daemon={}",
                        manifest.daemon_instance
                    ),
                )?;
                jobs().lock().insert(
                    id.to_string(),
                    ManagedJob {
                        job_id: id.to_string(),
                        child: ManagedChild::Owned(child),
                        command: command.to_string(),
                        stdout_path,
                        stderr_path,
                        exit_path,
                        manifest_path,
                        cwd,
                        fingerprint: key.clone().unwrap_or_default(),
                        pid,
                        process_group: pid as i32,
                        process_identity: identity,
                        started_at,
                        deadline_at,
                        last_output_bytes: 0,
                    },
                );
                Self::result(
                    json!({"status":"started","phase":"running","job_id":id.to_string(),"command":command,"cwd":cwd_display,"cache_key":key,"fingerprint":key,"pid":pid,"process_group":pid,"daemon_instance":manifest.daemon_instance}),
                )
            }
            "status" => {
                let raw = request
                    .args
                    .get("job_id")
                    .and_then(Value::as_str)
                    .ok_or_else(|| anyhow::anyhow!("job_id is required"))?;
                let id = JobId(
                    uuid::Uuid::parse_str(raw).map_err(|_| anyhow::anyhow!("invalid job_id"))?,
                );
                self.status(&id)
            }
            "cancel" => {
                let raw = request
                    .args
                    .get("job_id")
                    .and_then(Value::as_str)
                    .ok_or_else(|| anyhow::anyhow!("job_id is required"))?;
                let id = JobId(
                    uuid::Uuid::parse_str(raw).map_err(|_| anyhow::anyhow!("invalid job_id"))?,
                );
                if let Some(mut job) = jobs().lock().remove(raw) {
                    terminate_process_group(job.process_group, job.pid, &job.process_identity);
                    if let ManagedChild::Owned(child) = &mut job.child {
                        let _ = child.wait();
                    }
                    let _ = std::fs::remove_file(&job.manifest_path);
                }
                state.job_manager.update_state(&id, JobState::Cancelled)?;
                state.job_manager.push_event(
                    &id,
                    "process_cancel",
                    "terminated supervised process group",
                )?;
                Self::result(json!({"status":"cancelled","job_id":raw}))
            }
            _ => anyhow::bail!("action must be start, status, or cancel"),
        }
    }
}

fn read_output(stdout: &std::path::Path, stderr: &std::path::Path) -> (String, String) {
    (
        std::fs::read_to_string(stdout).unwrap_or_default(),
        std::fs::read_to_string(stderr).unwrap_or_default(),
    )
}
fn tail(value: &str) -> Vec<&str> {
    let lines: Vec<&str> = value.lines().collect();
    lines.into_iter().rev().take(50).rev().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn git(cwd: &Path, args: &[&str]) {
        let status = Command::new("git")
            .args(args)
            .current_dir(cwd)
            .status()
            .expect("run git");
        assert!(status.success(), "git {args:?} failed");
    }

    #[test]
    fn fingerprint_tracks_untracked_content_and_staging() {
        let repo = tempfile::tempdir().unwrap();
        std::fs::write(
            repo.path().join("Cargo.toml"),
            "[package]\nname='fingerprint-fixture'\nversion='0.1.0'\n",
        )
        .unwrap();
        git(repo.path(), &["init", "-q"]);
        git(repo.path(), &["config", "user.email", "test@prism.local"]);
        git(repo.path(), &["config", "user.name", "Prism Test"]);
        git(repo.path(), &["add", "Cargo.toml"]);
        git(repo.path(), &["commit", "-qm", "fixture"]);

        let clean = cache_key("cargo check", repo.path()).unwrap().unwrap();
        std::fs::write(repo.path().join("untracked.rs"), "const VALUE: u8 = 1;\n").unwrap();
        let untracked_one = cache_key("cargo check", repo.path()).unwrap().unwrap();
        std::fs::write(repo.path().join("untracked.rs"), "const VALUE: u8 = 2;\n").unwrap();
        let untracked_two = cache_key("cargo check", repo.path()).unwrap().unwrap();
        git(repo.path(), &["add", "untracked.rs"]);
        let staged = cache_key("cargo check", repo.path()).unwrap().unwrap();

        assert_ne!(clean, untracked_one);
        assert_ne!(untracked_one, untracked_two);
        assert_ne!(untracked_two, staged);
    }

    #[test]
    fn cached_streams_are_utf8_safe_and_bounded() {
        let value = "é".repeat(MAX_CACHED_STREAM_BYTES);
        let result = bounded(&value);
        assert!(result.len() <= MAX_CACHED_STREAM_BYTES);
        assert!(result.is_char_boundary(0));
        assert!(result.ends_with('é'));
    }

    #[test]
    fn cargo_output_is_compacted_into_structured_diagnostics() {
        let stdout = concat!(
            r#"{"reason":"compiler-artifact","target":{"name":"fixture"}}"#,
            "\n",
            r#"{"reason":"compiler-message","message":{"level":"warning","message":"unused field","rendered":"warning: unused field"}}"#,
            "\n"
        );
        let summary = output_summary("cargo check", stdout, "Finished dev profile");
        assert_eq!(summary["format"], "cargo-json");
        assert_eq!(summary["compiler_artifacts"], 1);
        assert_eq!(summary["diagnostic_count"], 1);
        assert_eq!(summary["diagnostics"][0]["level"], "warning");
        assert_eq!(summary["messages_tail"][0], "Finished dev profile");
    }
}
