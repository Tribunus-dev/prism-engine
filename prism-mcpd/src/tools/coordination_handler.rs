use anyhow::Result;
use prism_mcp_core::{DaemonState, McpHandler, RequestContext, ToolRequest, ToolResult};
use serde_json::{json, Value};

pub struct CoordinationHandler {
    pub name: &'static str,
    pub action: &'static str,
}
impl McpHandler for CoordinationHandler {
    fn name(&self) -> &'static str {
        self.name
    }
    fn description(&self) -> &'static str {
        "Native Prism agent coordination primitive backed by the daemon trifecta."
    }
    fn input_schema(&self) -> Value {
        json!({"type":"object","properties":{"session_id":{"type":"string"},"agent_id":{"type":"string"},"purpose":{"type":"string"},"work_id":{"type":"string"},"work_title":{"type":"string"},"priority":{"type":"integer"},"status":{"type":"string"},"claim_id":{"type":"string"},"lock_id":{"type":"string"},"path":{"type":"string"},"lock_kind":{"type":"string","enum":["read","write"]},"ttl_seconds":{"type":"integer"},"from_session":{"type":"string"},"to_session":{"type":"string"},"context":{"type":"object"},"event_type":{"type":"string"},"payload":{"type":"object"}},"additionalProperties":false})
    }
    fn call(
        &self,
        request: ToolRequest<'_>,
        _context: &RequestContext,
        state: &DaemonState,
    ) -> Result<ToolResult> {
        let store = state
            .coordination_store
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("coordination requires the trifecta storage profile"))?;
        let a = self.action;
        let s = |n: &str| {
            request
                .args
                .get(n)
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow::anyhow!(format!("{n} is required")))
        };
        let ttl = request
            .args
            .get("ttl_seconds")
            .and_then(Value::as_i64)
            .unwrap_or(300);
        let out = match a {
            "start_session" => serde_json::to_value(store.start_session(
                s("session_id")?,
                s("agent_id")?,
                request.args.get("purpose").and_then(Value::as_str),
            )?)?,
            "heartbeat" => {
                store.heartbeat(s("session_id")?)?;
                json!({"ok":true})
            }
            "close_session" => {
                store.close_session(s("session_id")?)?;
                json!({"ok":true})
            }
            "create_work" => serde_json::to_value(
                store.create_work(
                    s("work_id")?,
                    s("work_title")?,
                    request
                        .args
                        .get("priority")
                        .and_then(Value::as_i64)
                        .unwrap_or(0) as i32,
                    request.args.get("session_id").and_then(Value::as_str),
                )?,
            )?,
            "list_work" => serde_json::to_value(
                store.list_work(request.args.get("status").and_then(Value::as_str))?,
            )?,
            "claim_work" => {
                serde_json::to_value(store.claim_work(s("work_id")?, s("session_id")?, ttl)?)?
            }
            "release_claim" => {
                store.release_claim(s("claim_id")?, s("session_id")?)?;
                json!({"ok":true})
            }
            "acquire_path" => serde_json::to_value(
                store.acquire_path(
                    s("session_id")?,
                    s("path")?,
                    request
                        .args
                        .get("lock_kind")
                        .and_then(Value::as_str)
                        .unwrap_or("write"),
                    ttl,
                )?,
            )?,
            "release_path" => {
                store.release_path(s("lock_id")?, s("session_id")?)?;
                json!({"ok":true})
            }
            "recover" => store.recover_expired()?,
            "handoff" => {
                store.handoff(
                    s("work_id")?,
                    s("from_session")?,
                    s("to_session")?,
                    request.args.get("context").unwrap_or(&json!({})),
                )?;
                json!({"ok":true})
            }
            "event" => serde_json::to_value(store.append_event(
                s("event_type")?,
                s("session_id")?,
                request.args.get("payload").unwrap_or(&json!({})),
            )?)?,
            "status" => store.status()?,
            _ => anyhow::bail!("unknown coordination action: {a}"),
        };
        Ok(ToolResult::Text(serde_json::to_string(
            &json!({"ok":true,"result":out}),
        )?))
    }
}
