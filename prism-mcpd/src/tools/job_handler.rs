use parking_lot::Mutex;
use prism_mcp_core::{
    DaemonState, JobId, JobState, JobStore, McpHandler, RequestContext, ToolRequest, ToolResult,
};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, OnceLock};

struct ManagedJob {
    child: Child,
    command: String,
    stdout_path: std::path::PathBuf,
    stderr_path: std::path::PathBuf,
}

static JOBS: OnceLock<Mutex<HashMap<String, ManagedJob>>> = OnceLock::new();
static CACHE: OnceLock<Mutex<HashMap<String, (i32, String, String)>>> = OnceLock::new();

fn jobs() -> &'static Mutex<HashMap<String, ManagedJob>> {
    JOBS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn cache() -> &'static Mutex<HashMap<String, (i32, String, String)>> {
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn cache_key(command: &str, cwd: &std::path::Path) -> Option<String> {
    if !command.starts_with("cargo ") && !command.contains(" cargo ") {
        return None;
    }
    let mut material = command.as_bytes().to_vec();
    material.extend_from_slice(cwd.to_string_lossy().as_bytes());
    for file in ["Cargo.toml", "Cargo.lock"] {
        if let Ok(bytes) = std::fs::read(cwd.join(file)) {
            material.extend_from_slice(&bytes);
        }
    }
    if let Ok(output) = Command::new("git")
        .args(["status", "--porcelain"])
        .current_dir(cwd)
        .output()
    {
        material.extend_from_slice(&output.stdout);
    }
    if let Ok(output) = Command::new("git")
        .args(["diff", "--no-ext-diff"])
        .current_dir(cwd)
        .output()
    {
        material.extend_from_slice(&output.stdout);
    }
    Some(blake3::hash(&material).to_hex().to_string())
}

pub struct JobHandler {
    store: Arc<dyn JobStore>,
}

impl JobHandler {
    pub fn new(store: Arc<dyn JobStore>) -> Self {
        Self { store }
    }

    fn result(value: Value) -> anyhow::Result<ToolResult> {
        Ok(ToolResult::text(serde_json::to_string(&value)?))
    }

    fn status(&self, id: &JobId) -> anyhow::Result<ToolResult> {
        let key = id.to_string();
        let mut managed = jobs().lock();
        let mut output = (String::new(), String::new());
        if let Some(job) = managed.get_mut(&key) {
            output = read_output(&job.stdout_path, &job.stderr_path);
            if let Some(status) = job.child.try_wait()? {
                if let Some(cache_key_value) = cache_key(&job.command, &std::env::current_dir()?) {
                    cache().lock().insert(
                        cache_key_value,
                        (
                            status.code().unwrap_or(-1),
                            output.0.clone(),
                            output.1.clone(),
                        ),
                    );
                }
                let state = if status.success() {
                    JobState::Succeeded
                } else {
                    JobState::Failed(format!("process exited with {status}"))
                };
                self.store.update_state(id, state)?;
                managed.remove(&key);
            }
        }
        let record = self.store.get_job(id)?;
        Self::result(
            json!({"job_id":key,"status":record.state.as_str(),"command":record.operation,"exit_code":null,"stdout":output.0,"stderr":output.1,"stdout_tail":tail(&output.0),"stderr_tail":tail(&output.1)}),
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
        json!({"type":"object","properties":{"action":{"type":"string","enum":["start","status","cancel"]},"command":{"type":"string"},"job_id":{"type":"string"}},"required":["action"]})
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
                let cwd = std::env::current_dir()?;
                let key = cache_key(command, &cwd);
                if let Some(key_value) = &key {
                    if let Some((exit_code, stdout, stderr)) =
                        cache().lock().get(key_value).cloned()
                    {
                        return Self::result(
                            json!({"status":"cache_hit","cache_key":key_value,"command":command,"exit_code":exit_code,"stdout":stdout,"stderr":stderr,"stdout_tail":tail(&stdout),"stderr_tail":tail(&stderr)}),
                        );
                    }
                    if let Some(existing) =
                        jobs().lock().values().find(|job| job.command == command)
                    {
                        return Self::result(
                            json!({"status":"already_running","job_id":existing.child.id().to_string(),"command":command,"cache_key":key_value}),
                        );
                    }
                }
                let id = self.store.create_job("run_job", command)?;
                let dir = std::env::temp_dir().join("prism-mcpd-jobs");
                std::fs::create_dir_all(&dir)?;
                let stdout_path = dir.join(format!("{}.stdout", id));
                let stderr_path = dir.join(format!("{}.stderr", id));
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
                let child = Command::new("/bin/zsh")
                    .args(["-lc", command])
                    .current_dir(cwd)
                    .stdout(Stdio::from(stdout))
                    .stderr(Stdio::from(stderr))
                    .spawn()?;
                self.store.update_state(&id, JobState::Running)?;
                jobs().lock().insert(
                    id.to_string(),
                    ManagedJob {
                        child,
                        command: command.to_string(),
                        stdout_path,
                        stderr_path,
                    },
                );
                Self::result(json!({"status":"started","job_id":id.to_string(),"command":command}))
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
                    let _ = job.child.kill();
                }
                state.job_manager.cancel_job(&id)?;
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
    value.lines().rev().take(50).collect()
}
