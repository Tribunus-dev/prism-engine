use chrono::Utc;
use prism_mcp_core::{DaemonState, McpHandler, RequestContext, ToolRequest, ToolResult};
use serde_json::{json, Value};
use uuid::Uuid;

fn init(state: &DaemonState) -> anyhow::Result<()> {
    let _ = state;
    Ok(())
}
fn id<'a>(args: &'a Value, key: &str) -> anyhow::Result<&'a str> {
    args.get(key)
        .and_then(Value::as_str)
        .filter(|v| !v.is_empty())
        .ok_or_else(|| anyhow::anyhow!("{key} required"))
}
fn text(v: Value) -> anyhow::Result<ToolResult> {
    Ok(ToolResult::text(serde_json::to_string_pretty(&v)?))
}

fn save(state: &DaemonState, trace_id: &str, snapshot: &Value) -> anyhow::Result<()> {
    state.projection_store.put_trace(trace_id, snapshot)
}

fn load(state: &DaemonState, trace_id: &str) -> anyhow::Result<Value> {
    state
        .projection_store
        .get_trace(trace_id)?
        .ok_or_else(|| anyhow::anyhow!("trace not found: {trace_id}"))
}

pub struct StartTraceHandler;
impl McpHandler for StartTraceHandler {
    fn name(&self) -> &'static str {
        "start_trace"
    }
    fn description(&self) -> &'static str {
        "Start and persist a trace capture."
    }
    fn input_schema(&self) -> Value {
        json!({"type":"object","properties":{"scope":{"type":"string"},"label":{"type":"string"}},"required":["scope"]})
    }
    fn call(
        &self,
        r: ToolRequest<'_>,
        _: &RequestContext,
        s: &DaemonState,
    ) -> anyhow::Result<ToolResult> {
        init(s)?;
        let scope = id(r.args, "scope")?;
        let label = r.args.get("label").and_then(Value::as_str).unwrap_or(scope);
        let trace = Uuid::new_v4().to_string();
        save(
            s,
            &trace,
            &json!({"scope":scope,"label":label,"status":"capturing","events":[],"started_at":Utc::now().to_rfc3339()}),
        )?;
        text(json!({"trace_id":trace,"scope":scope,"label":label,"status":"capturing"}))
    }
}

pub struct StopTraceHandler;
impl McpHandler for StopTraceHandler {
    fn name(&self) -> &'static str {
        "stop_trace"
    }
    fn description(&self) -> &'static str {
        "Finalize a persisted trace capture."
    }
    fn input_schema(&self) -> Value {
        json!({"type":"object","properties":{"trace_id":{"type":"string"}},"required":["trace_id"]})
    }
    fn call(
        &self,
        r: ToolRequest<'_>,
        _: &RequestContext,
        s: &DaemonState,
    ) -> anyhow::Result<ToolResult> {
        init(s)?;
        let trace = id(r.args, "trace_id")?;
        let now = Utc::now().to_rfc3339();
        let mut snapshot = load(s, trace)?;
        if snapshot.get("status").and_then(Value::as_str) != Some("capturing") {
            anyhow::bail!("active trace not found: {trace}")
        }
        snapshot["status"] = json!("finalized");
        snapshot["stopped_at"] = json!(now.clone());
        save(s, trace, &snapshot)?;
        text(json!({"trace_id":trace,"status":"finalized","stopped_at":now}))
    }
}

pub struct CaptureOperationTraceHandler;
impl McpHandler for CaptureOperationTraceHandler {
    fn name(&self) -> &'static str {
        "capture_operation_trace"
    }
    fn description(&self) -> &'static str {
        "Persist a trace event for an operation."
    }
    fn input_schema(&self) -> Value {
        json!({"type":"object","properties":{"operation":{"type":"string"},"args":{"type":"object"}},"required":["operation"]})
    }
    fn call(
        &self,
        r: ToolRequest<'_>,
        _: &RequestContext,
        s: &DaemonState,
    ) -> anyhow::Result<ToolResult> {
        init(s)?;
        let op = id(r.args, "operation")?;
        let trace = Uuid::new_v4().to_string();
        let event = json!({"operation":op,"args":r.args.get("args").cloned().unwrap_or(json!({})),"started_at":Utc::now().to_rfc3339(),"duration_us":0});
        save(
            s,
            &trace,
            &json!({"scope":"operation","label":op,"status":"finalized","events":[event.clone()],"started_at":Utc::now().to_rfc3339()}),
        )?;
        text(json!({"trace_id":trace,"events":[event],"status":"captured"}))
    }
}

pub struct SummarizeTraceHandler;
impl McpHandler for SummarizeTraceHandler {
    fn name(&self) -> &'static str {
        "summarize_trace"
    }
    fn description(&self) -> &'static str {
        "Summarize persisted trace events."
    }
    fn input_schema(&self) -> Value {
        json!({"type":"object","properties":{"trace_id":{"type":"string"}},"required":["trace_id"]})
    }
    fn call(
        &self,
        r: ToolRequest<'_>,
        _: &RequestContext,
        s: &DaemonState,
    ) -> anyhow::Result<ToolResult> {
        init(s)?;
        let trace = id(r.args, "trace_id")?.to_owned();
        let snapshot = load(s, &trace)?;
        let events = snapshot.get("events").cloned().unwrap_or(json!([]));
        text(
            json!({"trace_id":trace,"status":snapshot.get("status"),"event_count":events.as_array().map_or(0, Vec::len),"events":events}),
        )
    }
}

pub struct CompareTracesHandler;
impl McpHandler for CompareTracesHandler {
    fn name(&self) -> &'static str {
        "compare_traces"
    }
    fn description(&self) -> &'static str {
        "Compare persisted trace event sequences."
    }
    fn input_schema(&self) -> Value {
        json!({"type":"object","properties":{"trace_a":{"type":"string"},"trace_b":{"type":"string"}},"required":["trace_a","trace_b"]})
    }
    fn call(
        &self,
        r: ToolRequest<'_>,
        _: &RequestContext,
        s: &DaemonState,
    ) -> anyhow::Result<ToolResult> {
        init(s)?;
        let a = id(r.args, "trace_a")?.to_owned();
        let b = id(r.args, "trace_b")?.to_owned();
        let ea = load(s, &a)?.get("events").cloned().unwrap_or(json!([]));
        let eb = load(s, &b)?.get("events").cloned().unwrap_or(json!([]));
        text(
            json!({"trace_a":a,"trace_b":b,"event_count_a":ea.as_array().map_or(0, Vec::len),"event_count_b":eb.as_array().map_or(0, Vec::len),"same_events":ea==eb}),
        )
    }
}

pub struct FindTraceStallsHandler;
impl McpHandler for FindTraceStallsHandler {
    fn name(&self) -> &'static str {
        "find_trace_stalls"
    }
    fn description(&self) -> &'static str {
        "Find recorded trace events exceeding a duration threshold."
    }
    fn input_schema(&self) -> Value {
        json!({"type":"object","properties":{"trace_id":{"type":"string"},"threshold_ms":{"type":"integer","default":100}},"required":["trace_id"]})
    }
    fn call(
        &self,
        r: ToolRequest<'_>,
        _: &RequestContext,
        s: &DaemonState,
    ) -> anyhow::Result<ToolResult> {
        let trace = id(r.args, "trace_id")?.to_owned();
        let threshold = r
            .args
            .get("threshold_ms")
            .and_then(Value::as_u64)
            .unwrap_or(100)
            * 1000;
        let events = load(s, &trace)?.get("events").cloned().unwrap_or(json!([]));
        let events: Vec<Value> = serde_json::from_value(events)?;
        let stalls: Vec<Value> = events
            .into_iter()
            .filter(|e| e.get("duration_us").and_then(Value::as_u64).unwrap_or(0) >= threshold)
            .collect();
        text(
            json!({"trace_id":trace,"threshold_ms":threshold/1000,"stalls":stalls,"stalls_found":stalls.len()}),
        )
    }
}
