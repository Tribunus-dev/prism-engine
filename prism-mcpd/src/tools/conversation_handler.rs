use anyhow::Result;
use prism_mcp_core::{
    DaemonState, EvidenceReceipt, EvidenceStatus, McpHandler, MetricSet, RequestContext,
    ToolInvocationId, ToolRequest, ToolResult,
};
use serde_json::{json, Value};

pub struct ConversationHandler;

impl McpHandler for ConversationHandler {
    fn name(&self) -> &'static str {
        "conversation"
    }

    fn description(&self) -> &'static str {
        "Conversation append, load, and search operations backed by the evidence ledger."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["append", "load", "search"],
                    "description": "Conversation operation"
                },
                "role": {
                    "type": "string",
                    "enum": ["user", "assistant", "system"],
                    "description": "Message role (append only)"
                },
                "text": {
                    "type": "string",
                    "description": "Message body (append only)"
                },
                "image_data": {
                    "type": "string",
                    "description": "Optional base64-encoded image data (append only)"
                },
                "offset": {
                    "type": "integer",
                    "description": "Pagination offset, default 0 (load only)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results, default 50 (load only)"
                },
                "query": {
                    "type": "string",
                    "description": "Text to search for within messages (search only)"
                }
            },
            "required": ["command"]
        })
    }

    fn call(
        &self,
        request: ToolRequest<'_>,
        _context: &RequestContext,
        state: &DaemonState,
    ) -> Result<ToolResult> {
        let command = request
            .args
            .get("command")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("command is required"))?;

        match command {
            "append" => handle_append(request, state),
            "load" => handle_load(request, state),
            "search" => handle_search(request, state),
            _ => anyhow::bail!("unknown conversation command: {command}"),
        }
    }
}

/// Append a new message to the conversation ledger.
///
/// We store the message payload as a JSON string in the receipt's
/// `environment` field (which round-trips through the evidence ledger SQLite
/// store).  The `target` field holds the role for lightweight indexing.
fn handle_append(request: ToolRequest<'_>, state: &DaemonState) -> Result<ToolResult> {
    let role = request
        .args
        .get("role")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow::anyhow!("role is required (user/assistant/system)"))?;
    let text = request
        .args
        .get("text")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow::anyhow!("text is required"))?;
    let image_data = request.args.get("image_data").and_then(Value::as_str);

    // Serialise the message payload into the `environment` field so it
    // survives the evidence ledger's persist/load cycle.
    let mut payload = json!({
        "role": role,
        "text": text,
    });
    if let Some(img) = image_data {
        payload["image_data"] = json!(img);
    }

    let invocation_id = ToolInvocationId::new();
    let receipt = EvidenceReceipt {
        invocation_id: invocation_id.clone(),
        tool: "conversation".to_string(),
        operation: "conversation_append".to_string(),
        inputs: vec![],
        outputs: vec![],
        environment: serde_json::to_string(&payload)?,
        target: Some(format!("role={role}")),
        source_revision: None,
        status: EvidenceStatus::Success,
        metrics: MetricSet::new(),
        diagnostics: vec![],
        started_at: chrono::Utc::now(),
        duration_ms: 0,
    };

    state.evidence_ledger.record(&receipt)?;

    let result = json!({
        "message_id": invocation_id.0.to_string(),
        "role": role,
    });
    Ok(ToolResult::Text(serde_json::to_string(
        &json!({"ok": true, "result": result}),
    )?))
}

/// Load conversation messages with optional offset/limit.
///
/// Each receipt carries the full message payload in its `environment` field
/// as JSON, so we can reconstruct {role, text, image_data} without needing
/// artifact reads.
fn handle_load(request: ToolRequest<'_>, state: &DaemonState) -> Result<ToolResult> {
    let limit = request
        .args
        .get("limit")
        .and_then(Value::as_u64)
        .unwrap_or(50) as usize;
    let offset = request
        .args
        .get("offset")
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize;

    let receipts = state
        .evidence_ledger
        .query("conversation", Some("conversation_append"), limit + offset)?;

    // DB returns newest-first — reverse into chronological order, apply offset.
    let messages: Vec<Value> = receipts
        .into_iter()
        .rev()
        .skip(offset)
        .map(|r| {
            // Unpack the message payload from `environment`.
            let (role, text, image_data) = serde_json::from_str::<Value>(&r.environment)
                .ok()
                .map(|v| {
                    let role = v
                        .get("role")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                    let text = v
                        .get("text")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                    let image_data =
                        v.get("image_data").and_then(Value::as_str).map(|s| s.to_string());
                    (role, text, image_data)
                })
                .unwrap_or_default();

            let mut msg = json!({
                "id": r.invocation_id.0.to_string(),
                "role": role,
                "text": text,
                "timestamp": r.started_at.to_rfc3339(),
            });
            if let Some(img) = image_data {
                msg["image_data"] = json!(img);
            }
            msg
        })
        .collect();

    let result = json!({"messages": messages, "count": messages.len()});
    Ok(ToolResult::Text(serde_json::to_string(
        &json!({"ok": true, "result": result}),
    )?))
}

/// Search conversation messages by text (client-side filter over loaded
/// receipts).
fn handle_search(request: ToolRequest<'_>, state: &DaemonState) -> Result<ToolResult> {
    let query = request
        .args
        .get("query")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow::anyhow!("query is required for search"))?;
    let limit = request
        .args
        .get("limit")
        .and_then(Value::as_u64)
        .unwrap_or(50) as usize;

    let query_lower = query.to_lowercase();

    let receipts = state
        .evidence_ledger
        .query("conversation", Some("conversation_append"), 500)?;

    let messages: Vec<Value> = receipts
        .into_iter()
        .rev()
        .filter_map(|r| {
            let v: Value = serde_json::from_str(&r.environment).ok()?;
            let role = v
                .get("role")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let text = v
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let image_data =
                v.get("image_data").and_then(Value::as_str).map(|s| s.to_string());

            // Client-side text filter.
            if !text.to_lowercase().contains(&query_lower) {
                return None;
            }

            let mut msg = json!({
                "id": r.invocation_id.0.to_string(),
                "role": role,
                "text": text,
                "timestamp": r.started_at.to_rfc3339(),
            });
            if let Some(img) = image_data {
                msg["image_data"] = json!(img);
            }
            Some(msg)
        })
        .take(limit)
        .collect();

    let result = json!({"messages": messages, "count": messages.len()});
    Ok(ToolResult::Text(serde_json::to_string(
        &json!({"ok": true, "result": result}),
    )?))
}
