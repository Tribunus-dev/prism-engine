use anyhow::Result;
use prism_ecs_constitutional::lifecycle_command::ENVELOPE_SCHEMA_VERSION;
use prism_ecs_runtime::{CommandEnvelope, CommandResult, KernelHandle};
use prism_ecs_server::inference::ModelRegistry;
use prism_ecs_server::runtime::server::PrefillDecodeRuntime;
use prism_ecs_server::runtime::server_types::SamplingConfig;
use prism_mcp_core::{
    CoordinationStore, DaemonState, JobProgress, JobState, JobStore, LeaseStore, LoopBudget,
    LoopDecision, LoopGuard, McpHandler, ProjectionStore, RequestContext, ToolRequest, ToolResult,
};
use serde_json::{json, Value};

/// Subagent management MCP handler backed by the ECS KernelHandle.
pub struct EcsSubagentHandler {
    kernel: KernelHandle,
    registry: std::sync::Arc<parking_lot::Mutex<ModelRegistry>>,
    projection: std::sync::Arc<dyn ProjectionStore>,
    jobs: std::sync::Arc<dyn JobStore>,
    leases: std::sync::Arc<dyn LeaseStore>,
    coordination: Option<std::sync::Arc<dyn CoordinationStore>>,
}

impl EcsSubagentHandler {
    pub fn new(
        kernel: KernelHandle,
        registry: std::sync::Arc<parking_lot::Mutex<ModelRegistry>>,
        projection: std::sync::Arc<dyn ProjectionStore>,
        jobs: std::sync::Arc<dyn JobStore>,
        leases: std::sync::Arc<dyn LeaseStore>,
        coordination: Option<std::sync::Arc<dyn CoordinationStore>>,
    ) -> Self {
        Self {
            kernel,
            registry,
            projection,
            jobs,
            leases,
            coordination,
        }
    }
}

impl McpHandler for EcsSubagentHandler {
    fn name(&self) -> &'static str {
        "subagent"
    }

    fn description(&self) -> &'static str {
        "Subagent lifecycle backed by ECS KernelHandle. Commands: spawn, list, collect, cancel"
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["spawn", "list", "collect", "cancel"]
                },
                "parent_entity_id": {
                    "type": "integer",
                    "description": "Parent agent entity numeric ID"
                },
                "task_description": {
                    "type": "string",
                    "description": "Task for the subagent"
                },
                "model_name": {"type": "string", "description": "Loaded Prism model to use for local execution"},
                "max_steps": {
                    "type": "integer",
                    "description": "Maximum execution steps"
                },
                "subagent_entity_id": {
                    "type": "integer",
                    "description": "Subagent entity numeric ID to collect/cancel"
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
    ) -> Result<ToolResult> {
        let command = request.args["command"].as_str().unwrap_or("");

        match command {
            "spawn" => {
                let parent_id = request.args["parent_entity_id"]
                    .as_i64()
                    .ok_or_else(|| anyhow::anyhow!("parent_entity_id (integer) required"))?
                    as u64;
                let task_desc = request.args["task_description"]
                    .as_str()
                    .unwrap_or("subtask");
                let max_steps = request.args["max_steps"]
                    .as_u64()
                    .map(|v| v as u32)
                    .unwrap_or(10);

                let envelope = CommandEnvelope {
                    schema_version: ENVELOPE_SCHEMA_VERSION,
                    command_type_id: 0,
                    idempotency_key: uuid::Uuid::new_v4(),
                    expected_epoch: None,
                    authority: "mcp:subagent".to_string(),
                    correlation_id: String::new(),
                    command: prism_ecs_runtime::Command::SpawnAgent {
                        parent_id,
                        task: task_desc.to_string(),
                        max_steps,
                    },
                };
                let result = self.kernel.submit(envelope)?;

                match result.result {
                    CommandResult::Spawned { entity_id } => {
                        let model_name = request.args["model_name"]
                            .as_str()
                            .map(str::to_string)
                            .or_else(|| self.registry.lock().list_models().into_iter().next())
                            .ok_or_else(|| anyhow::anyhow!("no loaded Prism model available"))?;
                        let model =
                            self.registry.lock().get_model(&model_name).ok_or_else(|| {
                                anyhow::anyhow!("loaded model not found: {model_name}")
                            })?;
                        let job_id = self.jobs.create_job("subagent", task_desc)?;
                        self.jobs.update_state(&job_id, JobState::Running)?;
                        let kernel = self.kernel.clone();
                        let jobs = self.jobs.clone();
                        let leases = self.leases.clone();
                        let projection = self.projection.clone();
                        let coordination = self.coordination.clone();
                        let task = task_desc.to_string();
                        let owner = format!("agent-{entity_id}");
                        let lease_key = format!("agent:{entity_id}");
                        let session = format!("agent-session-{entity_id}");
                        std::thread::spawn(move || {
                            let _ = coordination
                                .as_ref()
                                .map(|c| c.start_session(&session, &owner, Some(&task)));
                            let result = if leases.acquire(&lease_key, &owner, 120).unwrap_or(false)
                            {
                                run_local_agent(
                                    &*model.runtime,
                                    &task,
                                    max_steps,
                                    jobs.as_ref(),
                                    &job_id,
                                    &*leases,
                                    &lease_key,
                                    &owner,
                                    coordination.as_deref(),
                                    &session,
                                    &*projection,
                                    entity_id,
                                )
                            } else {
                                Err(anyhow::anyhow!("agent lease unavailable"))
                            };
                            match result {
                                Ok(text) => {
                                    let _ = kernel.submit(CommandEnvelope::new(
                                        prism_ecs_runtime::Command::CompleteAgent {
                                            agent_id: entity_id,
                                            result: text,
                                        },
                                    ));
                                    let _ = jobs.update_state(&job_id, JobState::Succeeded);
                                }
                                Err(error) => {
                                    let error_text = error.to_string();
                                    let _ = kernel.submit(CommandEnvelope::new(
                                        prism_ecs_runtime::Command::FailAgent {
                                            agent_id: entity_id,
                                            error: error_text.clone(),
                                        },
                                    ));
                                    let _ =
                                        jobs.update_state(&job_id, JobState::Failed(error_text));
                                }
                            }
                            let _ = leases.release(&lease_key, &owner);
                            let _ = coordination.as_ref().map(|c| c.close_session(&session));
                        });
                        Ok(ToolResult::Json(
                            json!({"ok":true,"subagent_entity_id":entity_id,"job_id":job_id.to_string(),"model_name":model_name,"phase":"Planning","lifecycle":"Active"}),
                        ))
                    }
                    _ => Err(anyhow::anyhow!("unexpected result from kernel")),
                }
            }

            "list" => {
                let parent_id = request.args["parent_entity_id"]
                    .as_i64()
                    .ok_or_else(|| anyhow::anyhow!("parent_entity_id (integer) required"))?
                    as u64;

                let agents = self.kernel.query_agents();
                let subagents: Vec<Value> = agents
                    .into_iter()
                    .filter(|a| a.parent_id == Some(parent_id))
                    .map(|a| {
                        json!({
                            "entity_id": a.entity_id,
                            "phase": a.phase,
                            "lifecycle": a.lifecycle,
                        })
                    })
                    .collect();

                Ok(ToolResult::Json(json!({
                    "ok": true,
                    "parent_entity_id": parent_id,
                    "subagents": subagents
                })))
            }

            "collect" => {
                let subagent_id = request.args["subagent_entity_id"]
                    .as_i64()
                    .ok_or_else(|| anyhow::anyhow!("subagent_entity_id (integer) required"))?
                    as u64;

                let agents = self.kernel.query_agents();
                let agent = agents.into_iter().find(|a| a.entity_id == subagent_id);

                match agent {
                    Some(a) => Ok(ToolResult::Json(json!({
                        "ok": true,
                        "subagent_entity_id": subagent_id,
                        "phase": a.phase,
                        "lifecycle": a.lifecycle,
                        "result": a.result,
                    }))),
                    None => Ok(ToolResult::Json(json!({
                        "ok": false,
                        "error": "subagent not found"
                    }))),
                }
            }

            "cancel" => {
                let subagent_id = request.args["subagent_entity_id"]
                    .as_i64()
                    .ok_or_else(|| anyhow::anyhow!("subagent_entity_id (integer) required"))?
                    as u64;

                self.kernel.submit(CommandEnvelope {
                    schema_version: ENVELOPE_SCHEMA_VERSION,
                    command_type_id: 0,
                    idempotency_key: uuid::Uuid::new_v4(),
                    expected_epoch: None,
                    authority: "mcp:subagent".to_string(),
                    correlation_id: String::new(),
                    command: prism_ecs_runtime::Command::CancelAgent {
                        agent_id: subagent_id,
                    },
                })?;

                Ok(ToolResult::Json(json!({
                    "ok": true,
                    "subagent_entity_id": subagent_id,
                    "phase": "Failed",
                    "lifecycle": "Completed"
                })))
            }

            _ => Err(anyhow::anyhow!(
                "Unknown command: {command}. Use spawn, list, collect, or cancel"
            )),
        }
    }
}

fn run_local_agent(
    runtime: &dyn PrefillDecodeRuntime,
    task: &str,
    max_steps: u32,
    jobs: &dyn JobStore,
    job_id: &prism_mcp_core::JobId,
    leases: &dyn LeaseStore,
    lease_key: &str,
    owner: &str,
    coordination: Option<&dyn CoordinationStore>,
    session: &str,
    projection: &dyn ProjectionStore,
    agent_id: u64,
) -> Result<String> {
    let tokens = runtime.tokenize(task).map_err(|e| anyhow::anyhow!(e))?;
    if tokens.is_empty() {
        anyhow::bail!("agent task tokenized to empty input");
    }
    let mut logits = runtime
        .run_prefill(&tokens)
        .map_err(|e| anyhow::anyhow!(e))?;
    let mut output = String::new();
    let mut loop_guard = LoopGuard::new(LoopBudget::new(
        max_steps,
        (max_steps as u64).saturating_mul(256),
        std::time::Duration::from_secs(600),
    ));
    let sampling = SamplingConfig::default();
    for step in 0..max_steps {
        let token = runtime
            .sample(&logits, &sampling)
            .map_err(|e| anyhow::anyhow!(e))?;
        if token == runtime.eos_token_id() {
            break;
        }
        output.push_str(&runtime.detokenize(token).map_err(|e| anyhow::anyhow!(e))?);
        logits = runtime.run_decode(token).map_err(|e| anyhow::anyhow!(e))?;
        match loop_guard.observe(&output, 1) {
            LoopDecision::Continue => {}
            LoopDecision::Escalate(reason) => anyhow::bail!("agent escalation required: {reason}"),
            LoopDecision::Stop(reason) => anyhow::bail!("agent loop stopped: {reason}"),
        }
        if !leases.renew(lease_key, owner, 120).unwrap_or(false) {
            anyhow::bail!("agent lease lost during execution");
        }
        if let Some(coordination) = coordination {
            coordination
                .heartbeat(session)
                .map_err(|e| anyhow::anyhow!(e))?;
        }
        let _ = projection.put_trace(
            &format!("agent:{agent_id}"),
            &serde_json::json!({"agent_id":agent_id,"step":step + 1,"output":output}),
        );
        let _ = jobs.update_progress(
            job_id,
            JobProgress {
                message: format!("agent step {}/{}", step + 1, max_steps),
                percent: Some(((step + 1) as f64 / max_steps.max(1) as f64) * 100.0),
            },
        );
    }
    Ok(output)
}
