use std::sync::Arc;

use parking_lot::Mutex;
use prism_ecs_server::inference::ModelRegistry;
use prism_mcp_core::{DaemonState, McpHandler, RequestContext, ToolRequest, ToolResult};

pub struct InferenceHandler {
    registry: Arc<Mutex<ModelRegistry>>,
}

impl InferenceHandler {
    pub fn new(registry: Arc<Mutex<ModelRegistry>>) -> Self {
        Self { registry }
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
                let model = self
                    .registry
                    .lock()
                    .load_model(std::path::Path::new(path))
                    .map_err(|e| anyhow::anyhow!("failed to load model: {e}"))?;
                Ok(ToolResult::text(format!("Loaded model: {}", model.name)))
            }
            "list_models" => {
                let models = self.registry.lock().list_models();
                Ok(ToolResult::text(if models.is_empty() {
                    "No models loaded.".to_string()
                } else {
                    models.join("\n")
                }))
            }
            "generate" => {
                let model_name = request.args["model_name"].as_str().unwrap_or("");
                let prompt = request.args["prompt"].as_str().unwrap_or("");
                let max_tokens = request.args["max_tokens"].as_u64().unwrap_or(256);
                if model_name.is_empty() || prompt.is_empty() {
                    return Err(anyhow::anyhow!("model_name and prompt are required"));
                }

                let reg = self.registry.lock();
                let _model = reg.get_model(model_name).ok_or_else(|| {
                    anyhow::anyhow!(
                        "Model '{model_name}' not found. Load it first with load_model."
                    )
                })?;
                // Runtime generate is not yet implemented — return status
                Ok(ToolResult::text(format!(
                    "[prism-runtime:generate] model={model_name} prompt='{prompt}' max_tokens={max_tokens}\n\
                     Inference engine not yet implemented — staged for wire-up."
                )))
            }
            _ => Err(anyhow::anyhow!(
                "Unknown command: {command}. Use load_model, list_models, or generate"
            )),
        }
    }
}
