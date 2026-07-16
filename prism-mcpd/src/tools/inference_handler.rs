use prism_mcp_core::{DaemonState, McpHandler, RequestContext, ToolRequest, ToolResult};

pub struct InferenceHandler {
    // registry: Arc<prism_ecs_server::inference::ModelRegistry>,
}

impl InferenceHandler {
    pub fn new() -> Self {
        Self {}
    }
}

impl McpHandler for InferenceHandler {
    fn name(&self) -> &'static str {
        "inference"
    }

    fn description(&self) -> &'static str {
        "Load, list, and run model inference. Sub-commands: load_model, list_models, generate"
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["load_model", "list_models", "generate"]
                },
                "model_path": {
                    "type": "string",
                    "description": "Path to .cimage file (for load_model)"
                },
                "model_name": {
                    "type": "string",
                    "description": "Model name (for generate)"
                },
                "prompt": {
                    "type": "string",
                    "description": "Input text (for generate)"
                },
                "max_tokens": {
                    "type": "integer",
                    "default": 256
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
            "load_model" => {
                let path = request.args["model_path"].as_str().unwrap_or("");
                if path.is_empty() {
                    return Err(anyhow::anyhow!("model_path is required"));
                }
                // TODO: delegate to prism_ecs_server::inference::ModelRegistry
                Ok(ToolResult::text(format!(
                    "Model loading from {path} — not yet wired to runtime"
                )))
            }
            "list_models" => {
                // TODO: read from ModelRegistry
                Ok(ToolResult::text("No models loaded."))
            }
            "generate" => {
                let model_name = request.args["model_name"].as_str().unwrap_or("");
                let prompt = request.args["prompt"].as_str().unwrap_or("");
                let max_tokens = request.args["max_tokens"].as_u64().unwrap_or(256);
                if model_name.is_empty() || prompt.is_empty() {
                    return Err(anyhow::anyhow!("model_name and prompt are required"));
                }
                Ok(ToolResult::text(format!(
                    "Generate from {model_name}: {prompt} (max_tokens={max_tokens}) — engine not yet wired"
                )))
            }
            _ => Err(anyhow::anyhow!(
                "Unknown command: {command}. Use load_model, list_models, or generate"
            )),
        }
    }
}
