use prism_mcp_core::{DaemonState, McpHandler, RequestContext, ToolRequest, ToolResult};
use serde_json::json;

pub struct AgentTickHandler;

impl AgentTickHandler {
    pub fn new() -> Self {
        Self
    }
}

impl McpHandler for AgentTickHandler {
    fn name(&self) -> &'static str {
        "agent_tick"
    }

    fn description(&self) -> &'static str {
        "Run AgentStateMachine::tick() on the ECS World and return transitions."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["tick"]
                }
            },
            "required": ["command"]
        })
    }

    fn call(
        &self,
        request: ToolRequest<'_>,
        _context: &RequestContext,
        _state: &DaemonState,
    ) -> anyhow::Result<ToolResult> {
        let command = request.args["command"].as_str().unwrap_or("");
        match command {
            "tick" => Ok(ToolResult::text(
                json!({
                    "status": "placeholder",
                    "message": "AgentTickHandler placeholder — ECS World not yet wired to DaemonState.",
                    "guidance": "Add Arc<Mutex<World>> to DaemonState, then call AgentStateMachine::tick(&world) here.",
                    "transitions": []
                })
                .to_string(),
            )),
            _ => Err(anyhow::anyhow!(
                "Unknown command: {command}. Use 'tick'."
            )),
        }
    }
}
