use chrono::Utc;
use prism_mcp_core::{DaemonState, McpHandler, RequestContext, ToolRequest, ToolResult};
use serde_json::{json, Value};
use uuid::Uuid;

fn init(state: &DaemonState) -> anyhow::Result<()> {
    state.db.with_writer(|c| { c.execute_batch("CREATE TABLE IF NOT EXISTS prism_traces(id TEXT PRIMARY KEY, scope TEXT NOT NULL, label TEXT NOT NULL, status TEXT NOT NULL, events_json TEXT NOT NULL, started_at TEXT NOT NULL, stopped_at TEXT)")?; Ok(()) })
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
        s.db.with_writer(|c| {
            c.execute(
                "INSERT INTO prism_traces VALUES(?1,?2,?3,'capturing','[]',?4,NULL)",
                [&trace, scope, label, &Utc::now().to_rfc3339()],
            )?;
            Ok(())
        })?;
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
        let changed=s.db.with_writer(|c|Ok(c.execute("UPDATE prism_traces SET status='finalized',stopped_at=?1 WHERE id=?2 AND status='capturing'",[&now,trace])?))?;
        if changed == 0 {
            anyhow::bail!("active trace not found: {trace}")
        }
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
        s.db.with_writer(|c| {
            c.execute(
                "INSERT INTO prism_traces VALUES(?1,'operation',?2,'finalized',?3,?4,?4)",
                [
                    &trace,
                    op,
                    &serde_json::to_string(&vec![event.clone()])?,
                    &Utc::now().to_rfc3339(),
                ],
            )?;
            Ok(())
        })?;
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
        let row = s.db.with_reader(|c| {
            let mut q = c.prepare("SELECT events_json,status FROM prism_traces WHERE id=?1")?;
            q.query_row([&trace], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })
            .map_err(Into::into)
        })?;
        let events: Vec<Value> = serde_json::from_str(&row.0)?;
        text(json!({"trace_id":trace,"status":row.1,"event_count":events.len(),"events":events}))
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
        let rows = s.db.with_reader(|c| {
            let mut q = c.prepare("SELECT id,events_json FROM prism_traces WHERE id IN (?1,?2)")?;
            let mut out = Vec::new();
            for row in q.query_map([&a, &b], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
            })? {
                out.push(row?)
            }
            Ok(out)
        })?;
        if rows.len() != 2 {
            anyhow::bail!("both traces must exist")
        }
        let ea: Vec<Value> = serde_json::from_str(&rows.iter().find(|x| x.0 == a).unwrap().1)?;
        let eb: Vec<Value> = serde_json::from_str(&rows.iter().find(|x| x.0 == b).unwrap().1)?;
        text(
            json!({"trace_a":a,"trace_b":b,"event_count_a":ea.len(),"event_count_b":eb.len(),"same_events":ea==eb}),
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
        let raw = s.db.with_reader(|c| {
            Ok(c.query_row(
                "SELECT events_json FROM prism_traces WHERE id=?1",
                [&trace],
                |r| r.get::<_, String>(0),
            )?)
        })?;
        let events: Vec<Value> = serde_json::from_str(&raw)?;
        let stalls: Vec<Value> = events
            .into_iter()
            .filter(|e| e.get("duration_us").and_then(Value::as_u64).unwrap_or(0) >= threshold)
            .collect();
        text(
            json!({"trace_id":trace,"threshold_ms":threshold/1000,"stalls":stalls,"stalls_found":stalls.len()}),
        )
    }
}
