use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

/// Opaque connection identifier for routing responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConnectionId(u64);

static NEXT_CONN_ID: AtomicU64 = AtomicU64::new(1);

impl ConnectionId {
    pub fn new() -> Self {
        Self(NEXT_CONN_ID.fetch_add(1, Ordering::Relaxed))
    }
}

impl fmt::Display for ConnectionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "conn-{}", self.0)
    }
}

/// A raw JSON-RPC frame from a connection, tagged with the connection id.
#[derive(Debug, Clone)]
pub struct RequestEnvelope {
    pub connection_id: ConnectionId,
    /// Sender for routing the response back to this connection's writer.
    pub response_tx: crossbeam_channel::Sender<ResponseFrame>,
    pub frame: String,
}

/// A JSON-RPC response frame ready to write to a connection.
#[derive(Debug, Clone)]
pub struct ResponseFrame {
    pub connection_id: ConnectionId,
    pub json: String,
}

// ── JSON-RPC protocol types ──────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct McpRequest {
    #[serde(rename = "jsonrpc")]
    pub jsonrpc: String,
    pub id: serde_json::Value,
    pub method: String,
    #[serde(default)]
    pub params: serde_json::Value,
}

#[derive(Debug, Serialize)]
pub struct McpResponse {
    #[serde(rename = "jsonrpc")]
    pub jsonrpc: String,
    pub id: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<McpError>,
}

impl McpResponse {
    pub fn success(id: serde_json::Value, text: &str) -> Self {
        let structured = serde_json::from_str::<serde_json::Value>(text).ok();
        let mut result = serde_json::json!({
            "content": [{"type": "text", "text": text}],
            "isError": false,
        });
        result["structuredContent"] =
            structured.unwrap_or_else(|| serde_json::json!({"ok":true,"text":text}));
        Self {
            jsonrpc: "2.0".into(),
            id,
            result: Some(result),
            error: None,
        }
    }

    pub fn tool_error(id: serde_json::Value, code: &str, message: &str, retryable: bool) -> Self {
        Self {
            jsonrpc: "2.0".into(),
            id,
            result: Some(serde_json::json!({
                "content": [{"type":"text","text":message}],
                "structuredContent": {"ok":false,"error":{"code":code,"message":message,"retryable":retryable}},
                "isError": true
            })),
            error: None,
        }
    }

    pub fn error(id: serde_json::Value, code: i32, message: &str) -> Self {
        Self {
            jsonrpc: "2.0".into(),
            id,
            result: None,
            error: Some(McpError {
                code,
                message: message.into(),
            }),
        }
    }

    pub fn parse_error() -> Self {
        Self::error(serde_json::Value::Null, -32700, "Parse error")
    }

    pub fn method_not_found(id: serde_json::Value, method: &str) -> Self {
        Self::error(id, -32601, &format!("Method not found: {method}"))
    }

    pub fn internal_error(id: serde_json::Value, msg: &str) -> Self {
        Self::error(id, -32603, msg)
    }
}

/// JSON-RPC error object.
#[derive(Debug, Serialize)]
pub struct McpError {
    pub code: i32,
    pub message: String,
}

/// Standard JSON-RPC error codes.
pub struct McpStatus;
impl McpStatus {
    pub const PARSE_ERROR: i32 = -32700;
    pub const INVALID_REQUEST: i32 = -32600;
    pub const METHOD_NOT_FOUND: i32 = -32601;
    pub const INTERNAL_ERROR: i32 = -32603;
    pub const TOOL_RATE_LIMITED: i32 = -32001;
    pub const TOOL_TIMEOUT: i32 = -32002;
}

// ── Tool handler types ────────────────────────────────────────────────────

/// Parsed tool invocation arguments.
pub struct ToolRequest<'a> {
    pub args: &'a serde_json::Value,
}

/// Structured result from a tool handler.
pub enum ToolResult {
    Text(String),
}

/// Output contracts advertised to agents.  These are centralized because the
/// handler crates intentionally keep execution logic independent of MCP wire
/// concerns.
pub fn output_schema_for_tool(name: &str) -> serde_json::Value {
    let object = |properties: serde_json::Value| {
        serde_json::json!({
            "type": "object",
            "properties": properties,
            "additionalProperties": true
        })
    };
    match name {
        "inspect_model" => object(
            serde_json::json!({"source":{"type":"string"},"size_bytes":{"type":"integer"},"format":{"type":"string"},"readable":{"type":"boolean"},"tensor_count":{"type":"integer"}}),
        ),
        "list_model_tensors" => object(
            serde_json::json!({"model":{"type":"string"},"count":{"type":"integer"},"tensors":{"type":"array","items":{"type":"object"}}}),
        ),
        "get_model_tensor" => {
            object(serde_json::json!({"name":{"type":"string"},"metadata":{"type":"object"}}))
        }
        "classify_model_tensors" => object(
            serde_json::json!({"model":{"type":"string"},"tensors":{"type":"array","items":{"type":"object","properties":{"name":{"type":"string"},"class":{"type":"string"}}}}}),
        ),
        "compare_models" => object(
            serde_json::json!({"manifest_a":{"type":"string"},"manifest_b":{"type":"string"},"same_tensor_names":{"type":"boolean"},"only_a":{"type":"array","items":{"type":"string"}},"only_b":{"type":"array","items":{"type":"string"}}}),
        ),
        "estimate_model_memory" => object(
            serde_json::json!({"model":{"type":"string"},"estimated_bytes":{"type":"integer"},"estimated_mib":{"type":"number"},"unknown_tensors":{"type":"array","items":{"type":"string"}}}),
        ),
        "validate_model_assets" => object(
            serde_json::json!({"manifest":{"type":"string"},"valid":{"type":"boolean"},"missing_assets":{"type":"array","items":{"type":"string"}}}),
        ),
        "create_benchmark_plan" => object(
            serde_json::json!({"plan_id":{"type":"string"},"name":{"type":"string"},"status":{"type":"string"}}),
        ),
        "run_benchmark" => object(
            serde_json::json!({"report_id":{"type":"string"},"plan_id":{"type":"string"},"elapsed_ms":{"type":"number"},"exit_code":{"type":"integer"},"output":{"type":"string"}}),
        ),
        "compare_benchmarks" => object(
            serde_json::json!({"delta_ms":{"type":"number"},"improved":{"type":"boolean"},"both_succeeded":{"type":"boolean"}}),
        ),
        "detect_performance_regression" => object(
            serde_json::json!({"report_id":{"type":"string"},"baseline":{"type":"string"},"regression":{"type":"boolean"},"delta_percent":{"type":"number"}}),
        ),
        "promote_baseline" => object(
            serde_json::json!({"baseline_name":{"type":"string"},"report_id":{"type":"string"},"status":{"type":"string"}}),
        ),
        "capture_replay" => {
            object(serde_json::json!({"replay_id":{"type":"string"},"status":{"type":"string"}}))
        }
        "run_replay" => object(
            serde_json::json!({"replay_id":{"type":"string"},"status":{"type":"string"},"payload":{"type":"object"}}),
        ),
        "minimize_replay" => object(
            serde_json::json!({"replay_id":{"type":"string"},"source":{"type":"string"},"status":{"type":"string"},"payload":{"type":"object"}}),
        ),
        "compare_replays" => object(
            serde_json::json!({"replay_a":{"type":"string"},"replay_b":{"type":"string"},"same_payload":{"type":"boolean"}}),
        ),
        "export_replay" => object(
            serde_json::json!({"replay_id":{"type":"string"},"destination":{"type":"string"},"bytes":{"type":"integer"},"status":{"type":"string"}}),
        ),
        "import_replay" => object(
            serde_json::json!({"replay_id":{"type":"string"},"source":{"type":"string"},"status":{"type":"string"},"payload":{"type":"object"}}),
        ),
        "start_trace" => object(
            serde_json::json!({"trace_id":{"type":"string"},"scope":{"type":"string"},"label":{"type":"string"},"status":{"type":"string"}}),
        ),
        "stop_trace" => object(
            serde_json::json!({"trace_id":{"type":"string"},"status":{"type":"string"},"stopped_at":{"type":"string","format":"date-time"}}),
        ),
        "capture_operation_trace" => object(
            serde_json::json!({"trace_id":{"type":"string"},"events":{"type":"array","items":{"type":"object"}},"status":{"type":"string"}}),
        ),
        "summarize_trace" => object(
            serde_json::json!({"trace_id":{"type":"string"},"status":{"type":"string"},"event_count":{"type":"integer"},"events":{"type":"array","items":{"type":"object"}}}),
        ),
        "compare_traces" => object(
            serde_json::json!({"trace_a":{"type":"string"},"trace_b":{"type":"string"},"event_count_a":{"type":"integer"},"event_count_b":{"type":"integer"},"same_events":{"type":"boolean"}}),
        ),
        "find_trace_stalls" => object(
            serde_json::json!({"trace_id":{"type":"string"},"threshold_ms":{"type":"integer"},"stalls_found":{"type":"integer"},"stalls":{"type":"array","items":{"type":"object"}}}),
        ),
        "validate_kernel_abi" => object(
            serde_json::json!({"binary_path":{"type":"string"},"abi_compatible":{"type":"boolean"},"missing_entry_points":{"type":"array","items":{"type":"string"}}}),
        ),
        "register_kernel" => object(
            serde_json::json!({"name":{"type":"string"},"backend":{"type":"string"},"registered":{"type":"boolean"},"artifact_hash":{"type":"string"},"persistent":{"type":"boolean"}}),
        ),
        _ => object(serde_json::json!({"ok":{"type":"boolean"},"text":{"type":"string"}})),
    }
}

impl ToolResult {
    pub fn text(s: impl Into<String>) -> Self {
        Self::Text(s.into())
    }
}

/// Context for a single tool invocation.
pub struct RequestContext {
    pub connection_id: ConnectionId,
    pub deadline: Option<chrono::DateTime<chrono::Utc>>,
}

// ── McpHandler trait ──────────────────────────────────────────────────────

use std::collections::HashMap;
use std::sync::Arc;

use crate::file_lock::FileLock;
use crate::scheduler::SchedulerHandle;
use crate::storage::{
    ArtifactRepository, BenchmarkStore, EvidenceStore, ExperimentStore, JobStore, KnowledgeStore,
    LeaseStore, ProjectionStore,
};
use crate::subprocess::ProcessCache;
use crate::work_journal::WorkJournal;

/// Shared state accessible to every tool handler.
pub struct DaemonState {
    pub tools: Arc<HashMap<&'static str, Arc<dyn McpHandler + Sync + Send>>>,
    pub artifact_store: Arc<dyn ArtifactRepository>,
    pub evidence_ledger: Arc<dyn EvidenceStore>,
    pub file_lock: FileLock,
    pub work_journal: WorkJournal,
    pub process_cache: ProcessCache,
    pub scheduler_handle: SchedulerHandle,
    pub job_manager: Arc<dyn JobStore>,
    pub resource_leases: Arc<dyn LeaseStore>,
    pub projection_store: Arc<dyn ProjectionStore>,
    pub experiment_store: Arc<dyn ExperimentStore>,
    pub benchmark_store: Arc<dyn BenchmarkStore>,
    pub knowledge_store: Arc<dyn KnowledgeStore>,
    pub connection_count: Arc<AtomicU64>,
    pub idle_generation: Arc<AtomicU64>,
}

/// A tool handler registered with the daemon.
pub trait McpHandler: Send + Sync {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    fn input_schema(&self) -> serde_json::Value;
    fn output_schema(&self) -> serde_json::Value {
        output_schema_for_tool(self.name())
    }
    fn annotations(&self) -> serde_json::Value {
        serde_json::json!({"readOnlyHint":false,"destructiveHint":false,"idempotentHint":false,"openWorldHint":true})
    }
    fn call(
        &self,
        request: ToolRequest<'_>,
        context: &RequestContext,
        state: &DaemonState,
    ) -> anyhow::Result<ToolResult>;
}
