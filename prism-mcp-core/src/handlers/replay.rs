use crate::{DaemonState, McpHandler, RequestContext, ToolRequest, ToolResult};
use serde_json::{json, Value};
use std::fs;
use uuid::Uuid;

fn init(s: &DaemonState) -> anyhow::Result<()> {
    let _ = s;
    Ok(())
}
fn arg<'a>(a: &'a Value, k: &str) -> anyhow::Result<&'a str> {
    a.get(k)
        .and_then(Value::as_str)
        .filter(|v| !v.is_empty())
        .ok_or_else(|| anyhow::anyhow!("{k} required"))
}
fn result(v: Value) -> anyhow::Result<ToolResult> {
    Ok(ToolResult::text(serde_json::to_string_pretty(&v)?))
}
fn stored(s: &DaemonState, id: &str) -> anyhow::Result<(String, String)> {
    let (status, payload) = s
        .projection_store
        .get_replay(id)?
        .ok_or_else(|| anyhow::anyhow!("replay not found: {id}"))?;
    Ok((serde_json::to_string(&payload)?, status))
}

pub struct CaptureReplayHandler;
impl McpHandler for CaptureReplayHandler {
    fn name(&self) -> &'static str {
        "capture_replay"
    }
    fn description(&self) -> &'static str {
        "Capture and persist a replay bundle."
    }
    fn input_schema(&self) -> Value {
        json!({"type":"object","properties":{"invocation_id":{"type":"string"},"payload":{"type":"object"}},"required":["invocation_id"]})
    }
    fn call(
        &self,
        r: ToolRequest<'_>,
        _: &RequestContext,
        s: &DaemonState,
    ) -> anyhow::Result<ToolResult> {
        init(s)?;
        let inv = arg(r.args, "invocation_id")?;
        let id = Uuid::new_v4().to_string();
        let payload = serde_json::to_string(
            &json!({"invocation_id":inv,"payload":r.args.get("payload").cloned().unwrap_or(json!({}))}),
        )?;
        s.projection_store
            .put_replay(&id, "captured", &serde_json::from_str(&payload)?)?;
        result(json!({"replay_id":id,"status":"captured"}))
    }
}
pub struct RunReplayHandler;
impl McpHandler for RunReplayHandler {
    fn name(&self) -> &'static str {
        "run_replay"
    }
    fn description(&self) -> &'static str {
        "Load and validate a persisted replay bundle."
    }
    fn input_schema(&self) -> Value {
        json!({"type":"object","properties":{"replay_id":{"type":"string"}},"required":["replay_id"]})
    }
    fn call(
        &self,
        r: ToolRequest<'_>,
        _: &RequestContext,
        s: &DaemonState,
    ) -> anyhow::Result<ToolResult> {
        init(s)?;
        let id = arg(r.args, "replay_id")?;
        let (p, _) = stored(s, id)?;
        result(
            json!({"replay_id":id,"status":"validated","payload":serde_json::from_str::<Value>(&p)?}),
        )
    }
}
pub struct MinimizeReplayHandler;
impl McpHandler for MinimizeReplayHandler {
    fn name(&self) -> &'static str {
        "minimize_replay"
    }
    fn description(&self) -> &'static str {
        "Remove optional payload fields and persist a minimized replay."
    }
    fn input_schema(&self) -> Value {
        json!({"type":"object","properties":{"replay_id":{"type":"string"}},"required":["replay_id"]})
    }
    fn call(
        &self,
        r: ToolRequest<'_>,
        _: &RequestContext,
        s: &DaemonState,
    ) -> anyhow::Result<ToolResult> {
        init(s)?;
        let id = arg(r.args, "replay_id")?;
        let (p, _) = stored(s, id)?;
        let mut v: Value = serde_json::from_str(&p)?;
        if let Some(obj) = v.get_mut("payload").and_then(Value::as_object_mut) {
            obj.retain(|k, _| k == "input" || k == "operation");
        }
        let out = Uuid::new_v4().to_string();
        s.projection_store.put_replay(&out, "minimized", &v)?;
        result(json!({"replay_id":out,"source":id,"status":"minimized","payload":v}))
    }
}
pub struct CompareReplaysHandler;
impl McpHandler for CompareReplaysHandler {
    fn name(&self) -> &'static str {
        "compare_replays"
    }
    fn description(&self) -> &'static str {
        "Compare two persisted replay payloads."
    }
    fn input_schema(&self) -> Value {
        json!({"type":"object","properties":{"replay_a":{"type":"string"},"replay_b":{"type":"string"}},"required":["replay_a","replay_b"]})
    }
    fn call(
        &self,
        r: ToolRequest<'_>,
        _: &RequestContext,
        s: &DaemonState,
    ) -> anyhow::Result<ToolResult> {
        init(s)?;
        let a = arg(r.args, "replay_a")?;
        let b = arg(r.args, "replay_b")?;
        let (pa, _) = stored(s, a)?;
        let (pb, _) = stored(s, b)?;
        let va: Value = serde_json::from_str(&pa)?;
        let vb: Value = serde_json::from_str(&pb)?;
        result(
            json!({"replay_a":a,"replay_b":b,"same_payload":va==vb,"payload_a":va,"payload_b":vb}),
        )
    }
}
pub struct ExportReplayHandler;
impl McpHandler for ExportReplayHandler {
    fn name(&self) -> &'static str {
        "export_replay"
    }
    fn description(&self) -> &'static str {
        "Export a replay bundle to a file."
    }
    fn input_schema(&self) -> Value {
        json!({"type":"object","properties":{"replay_id":{"type":"string"},"destination":{"type":"string"}},"required":["replay_id","destination"]})
    }
    fn call(
        &self,
        r: ToolRequest<'_>,
        _: &RequestContext,
        s: &DaemonState,
    ) -> anyhow::Result<ToolResult> {
        init(s)?;
        let id = arg(r.args, "replay_id")?;
        let dest = arg(r.args, "destination")?;
        let (p, _) = stored(s, id)?;
        fs::write(dest, &p)?;
        result(json!({"replay_id":id,"destination":dest,"bytes":p.len(),"status":"exported"}))
    }
}
pub struct ImportReplayHandler;
impl McpHandler for ImportReplayHandler {
    fn name(&self) -> &'static str {
        "import_replay"
    }
    fn description(&self) -> &'static str {
        "Import and validate a replay bundle from a file."
    }
    fn input_schema(&self) -> Value {
        json!({"type":"object","properties":{"source":{"type":"string"}},"required":["source"]})
    }
    fn call(
        &self,
        r: ToolRequest<'_>,
        _: &RequestContext,
        s: &DaemonState,
    ) -> anyhow::Result<ToolResult> {
        init(s)?;
        let source = arg(r.args, "source")?;
        let p = fs::read_to_string(source)?;
        let v: Value = serde_json::from_str(&p)?;
        let id = Uuid::new_v4().to_string();
        s.projection_store.put_replay(&id, "imported", &v)?;
        result(json!({"replay_id":id,"source":source,"status":"imported","payload":v}))
    }
}
