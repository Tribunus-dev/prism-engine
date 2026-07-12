use crossbeam_channel::Receiver;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::protocol::{
    DaemonState, McpHandler, RequestContext, RequestEnvelope, ResponseFrame, ToolRequest,
};

/// Per-tool concurrency limit.
#[derive(Debug, Clone, Copy)]
pub struct ToolLimit {
    pub max_concurrency: usize,
    pub queue_capacity: usize,
    pub timeout: Duration,
}

impl ToolLimit {
    pub const fn new(max: usize, queue: usize, timeout_secs: u64) -> Self {
        Self {
            max_concurrency: max,
            queue_capacity: queue,
            timeout: Duration::from_secs(timeout_secs),
        }
    }
}

/// Policy mapping tool names to their concurrency limits.
#[derive(Debug, Clone)]
pub struct ToolConcurrencyPolicy {
    pub limits: HashMap<&'static str, ToolLimit>,
    pub default: ToolLimit,
}

impl ToolConcurrencyPolicy {
    pub fn new() -> Self {
        let mut limits = HashMap::new();
        limits.insert("search_kb", ToolLimit::new(8, 64, 5));
        limits.insert("get_document", ToolLimit::new(16, 128, 5));
        limits.insert("get_related", ToolLimit::new(16, 64, 5));
        limits.insert("get_by_tag", ToolLimit::new(16, 64, 5));
        limits.insert("list_documents", ToolLimit::new(8, 32, 10));
        limits.insert("ingest_document", ToolLimit::new(1, 8, 30));
        limits.insert("scan_directory", ToolLimit::new(1, 2, 120));
        limits.insert("register_kernel", ToolLimit::new(1, 4, 60));
        limits.insert("cargo_build", ToolLimit::new(1, 4, 300));
        limits.insert("quant_sweep", ToolLimit::new(2, 4, 600));
        limits.insert("cimage_read", ToolLimit::new(8, 32, 10));
        Self {
            limits,
            default: ToolLimit::new(8, 32, 30),
        }
    }

    pub fn get(&self, tool: &str) -> ToolLimit {
        self.limits.get(tool).copied().unwrap_or(self.default)
    }
}

impl Default for ToolConcurrencyPolicy {
    fn default() -> Self {
        Self::new()
    }
}

/// Handle for communicating with the scheduler (permit release, dequeue next).
#[derive(Clone)]
pub struct SchedulerHandle {
    pub release_sender: crossbeam_channel::Sender<RequestEnvelope>,
}

/// The scheduler reads requests from the global work queue, enforces
/// per-tool concurrency limits, and dispatches work to spawned worker
/// threads. Workers release their permit on completion via a channel,
/// allowing the scheduler to dequeue the next waiting request.
pub struct Scheduler {
    work_queue: Receiver<RequestEnvelope>,
    tools: Arc<HashMap<&'static str, Arc<dyn McpHandler + Sync + Send>>>,
    state: Arc<DaemonState>,
    policy: ToolConcurrencyPolicy,
    active: Arc<Mutex<HashMap<String, usize>>>,
}

impl Scheduler {
    pub fn new(
        work_queue: Receiver<RequestEnvelope>,
        tools: Arc<HashMap<&'static str, Arc<dyn McpHandler + Sync + Send>>>,
        state: Arc<DaemonState>,
    ) -> Self {
        Self {
            work_queue,
            tools,
            state,
            policy: ToolConcurrencyPolicy::new(),
            active: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Run the dispatch loop. Frames arrive from the work queue. For each
    /// valid `tools/call`, we try to acquire a permit and spawn a worker.
    pub fn run(&self) {
        loop {
            let envelope = match self.work_queue.recv() {
                Ok(e) => e,
                Err(_) => break,
            };

            let response_tx = &envelope.response_tx;

            // Parse the frame minimally to extract method and id
            let frame: serde_json::Value = match serde_json::from_str(&envelope.frame) {
                Ok(v) => v,
                Err(_) => {
                    self.send(
                        response_tx,
                        envelope.connection_id,
                        crate::McpResponse::parse_error(),
                    );
                    continue;
                }
            };

            let method = frame["method"].as_str().unwrap_or("").to_string();
            let id = frame["id"].clone();

            if method == "initialize" {
                let protocol_version = frame["params"]["protocolVersion"]
                    .as_str()
                    .unwrap_or("2025-03-26");
                self.send(
                    response_tx,
                    envelope.connection_id,
                    crate::McpResponse {
                        jsonrpc: "2.0".into(),
                        id,
                        result: Some(serde_json::json!({
                            "protocolVersion": protocol_version,
                            "capabilities": { "tools": { "listChanged": false } },
                            "serverInfo": { "name": "prism-mcpd", "version": env!("CARGO_PKG_VERSION") }
                        })),
                        error: None,
                    },
                );
                continue;
            }

            if method == "notifications/initialized" || method.starts_with("notifications/") {
                continue;
            }

            if method == "ping" {
                self.send(
                    response_tx,
                    envelope.connection_id,
                    crate::McpResponse {
                        jsonrpc: "2.0".into(),
                        id,
                        result: Some(serde_json::json!({})),
                        error: None,
                    },
                );
                continue;
            }

            if method == "tools/list" {
                self.handle_list(response_tx, envelope.connection_id, id);
                continue;
            }

            if method != "tools/call" {
                self.send(
                    response_tx,
                    envelope.connection_id,
                    crate::McpResponse::method_not_found(id, &method),
                );
                continue;
            }

            let tool_name: String = frame["params"]["name"].as_str().unwrap_or("").to_string();
            let args_value: serde_json::Value = frame["params"]["arguments"]
                .as_object()
                .map(|_| frame["params"]["arguments"].clone())
                .unwrap_or_else(|| serde_json::json!({}));

            // Find handler
            let handler = match self.tools.get(tool_name.as_str()) {
                Some(h) => h.clone(),
                None => {
                    self.send(
                        response_tx,
                        envelope.connection_id,
                        crate::McpResponse::method_not_found(id, &tool_name),
                    );
                    continue;
                }
            };

            let limit = self.policy.get(&tool_name);
            {
                let mut active = self.active.lock().unwrap_or_else(|e| e.into_inner());
                let count = active.entry(tool_name.clone()).or_default();
                if *count >= limit.max_concurrency {
                    self.send(
                        response_tx,
                        envelope.connection_id,
                        crate::McpResponse::tool_error(
                            id,
                            "RATE_LIMITED",
                            &format!("tool '{}' has reached its concurrency limit", tool_name),
                            true,
                        ),
                    );
                    continue;
                }
                *count += 1;
            }

            // Spawn a worker thread for this request
            let response_tx = envelope.response_tx.clone();
            let handler = handler.clone();
            let state = self.state.clone();
            let conn_id = envelope.connection_id;
            let active = self.active.clone();
            let active_tool_name = tool_name.clone();

            std::thread::spawn(move || {
                let (result_tx, result_rx) = crossbeam_channel::bounded(1);
                let deadline = chrono::Utc::now()
                    + chrono::Duration::from_std(limit.timeout)
                        .unwrap_or_else(|_| chrono::Duration::seconds(30));
                std::thread::spawn(move || {
                    let result = handler.call(
                        ToolRequest { args: &args_value },
                        &RequestContext {
                            connection_id: conn_id,
                            deadline: Some(deadline),
                        },
                        &state,
                    );
                    let _ = result_tx.send(result);
                    let mut counts = active.lock().unwrap_or_else(|e| e.into_inner());
                    if let Some(count) = counts.get_mut(&active_tool_name) {
                        *count = count.saturating_sub(1);
                    }
                });
                let resp = match result_rx.recv_timeout(limit.timeout) {
                    Ok(Ok(crate::ToolResult::Text(t))) => crate::McpResponse::success(id, &t),
                    Ok(Err(e)) => {
                        crate::McpResponse::tool_error(id, "TOOL_FAILED", &e.to_string(), false)
                    }
                    Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                        crate::McpResponse::tool_error(
                            id,
                            "TIMEOUT",
                            "tool execution timed out",
                            true,
                        )
                    }
                    Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                        crate::McpResponse::tool_error(
                            id,
                            "WORKER_DISCONNECTED",
                            "tool worker disconnected",
                            true,
                        )
                    }
                };
                let _ = response_tx.send(ResponseFrame {
                    connection_id: conn_id,
                    json: serde_json::to_string(&resp).unwrap(),
                });
            });
        }
    }

    fn send(
        &self,
        response_tx: &crossbeam_channel::Sender<ResponseFrame>,
        conn_id: crate::ConnectionId,
        resp: crate::McpResponse,
    ) {
        let _ = response_tx.send(ResponseFrame {
            connection_id: conn_id,
            json: serde_json::to_string(&resp).unwrap(),
        });
    }

    fn handle_list(
        &self,
        response_tx: &crossbeam_channel::Sender<ResponseFrame>,
        conn_id: crate::ConnectionId,
        id: serde_json::Value,
    ) {
        let tool_list: Vec<serde_json::Value> = self
            .tools
            .iter()
            .map(|(name, h)| {
                serde_json::json!({
                    "name": name,
                    "description": h.description(),
                    "inputSchema": h.input_schema(),
                    "outputSchema": h.output_schema(),
                    "annotations": h.annotations(),
                })
            })
            .collect();

        self.send(
            response_tx,
            conn_id,
            crate::McpResponse {
                jsonrpc: "2.0".into(),
                id,
                result: Some(serde_json::json!({ "tools": tool_list })),
                error: None,
            },
        );
    }
}
