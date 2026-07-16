use anyhow::Result;
use parking_lot::Mutex;
use prism_ecs_constitutional::{
    AgentConfig, AgentLifecycle, AgentPhase, AgentRun, AgentTask, ParentAgentId, Timestamp,
    WorldTransitExt, WorldTxn,
};
use prism_ecs_core::{Entity, EntityKind, World};
use prism_mcp_core::{DaemonState, McpHandler, RequestContext, ToolRequest, ToolResult};
use serde_json::{json, Value};
use std::sync::Arc;

/// Subagent management MCP handler backed by the ECS World.
pub struct EcsSubagentHandler {
    world: Arc<Mutex<World>>,
}

impl EcsSubagentHandler {
    pub fn new(world: Arc<Mutex<World>>) -> Self {
        Self { world }
    }
}

impl McpHandler for EcsSubagentHandler {
    fn name(&self) -> &'static str {
        "subagent"
    }

    fn description(&self) -> &'static str {
        "Subagent lifecycle backed by ECS World. Commands: spawn, list, collect, cancel"
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
        let mut world = self.world.lock();

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

                // Follow the CreateAgentRunCommand pattern: reserve entity id,
                // build a transaction, stage spawn + durable components, commit.
                let agent_id = WorldTxn::next_entity_id(&world);
                let mut txn = WorldTxn::new(&world);

                txn.stage_spawn(agent_id, EntityKind::Agent);
                txn.put_durable(
                    agent_id,
                    AgentRun {
                        run_id: agent_id.id(),
                        session_entity: 0,
                        name: format!("subagent_of_{}", parent_id),
                        created_at: Timestamp::now(),
                    },
                );
                txn.put_durable(
                    agent_id,
                    AgentTask {
                        task_description: task_desc.to_string(),
                        model_entity: parent_id,
                        max_steps,
                    },
                );
                txn.put_durable(
                    agent_id,
                    AgentConfig {
                        model: "default".to_string(),
                        temperature: 0.7,
                        max_tokens: 4096,
                        tools_enabled: true,
                        max_tool_rounds: 10,
                    },
                );
                txn.put_durable(agent_id, AgentPhase::Planning);
                txn.put_durable(agent_id, AgentLifecycle::Active);
                txn.put_durable(agent_id, ParentAgentId(Entity::new(parent_id, 0)));

                world
                    .transit(txn)
                    .map_err(|e| anyhow::anyhow!("world transit: {e}"))?;

                Ok(ToolResult::text(
                    json!({
                        "ok": true,
                        "subagent_entity_id": agent_id.id(),
                        "phase": "Planning",
                        "lifecycle": "Active"
                    })
                    .to_string(),
                ))
            }

            "list" => {
                let parent_id = request.args["parent_entity_id"]
                    .as_i64()
                    .ok_or_else(|| anyhow::anyhow!("parent_entity_id (integer) required"))?
                    as u64;

                // Collect matching entity ids first to release the query borrow,
                // then read components individually.
                let candidates: Vec<Entity> = world
                    .query::<ParentAgentId>()
                    .filter(|(_, parent)| parent.0.id() == parent_id)
                    .map(|(entity, _)| entity)
                    .collect();

                let subagents: Vec<Value> = candidates
                    .iter()
                    .map(|&entity| {
                        let phase = world.get_component::<AgentPhase>(entity);
                        let lifecycle = world.get_component::<AgentLifecycle>(entity);
                        json!({
                            "entity_id": entity.id(),
                            "phase": format!("{:?}", phase.unwrap_or(&AgentPhase::Planning)),
                            "lifecycle": format!("{:?}", lifecycle.unwrap_or(&AgentLifecycle::Active)),
                        })
                    })
                    .collect();

                Ok(ToolResult::text(
                    json!({
                        "ok": true,
                        "parent_entity_id": parent_id,
                        "subagents": subagents
                    })
                    .to_string(),
                ))
            }

            "collect" => {
                let subagent_id = request.args["subagent_entity_id"]
                    .as_i64()
                    .ok_or_else(|| anyhow::anyhow!("subagent_entity_id (integer) required"))?
                    as u64;
                let entity = Entity::new(subagent_id, 0);

                let phase = world.get_component::<AgentPhase>(entity).copied();
                let lifecycle = world.get_component::<AgentLifecycle>(entity).copied();

                Ok(ToolResult::text(
                    json!({
                        "ok": true,
                        "subagent_entity_id": subagent_id,
                        "phase": format!("{:?}", phase.unwrap_or(AgentPhase::Planning)),
                        "lifecycle": format!("{:?}", lifecycle.unwrap_or(AgentLifecycle::Active)),
                    })
                    .to_string(),
                ))
            }

            "cancel" => {
                let subagent_id = request.args["subagent_entity_id"]
                    .as_i64()
                    .ok_or_else(|| anyhow::anyhow!("subagent_entity_id (integer) required"))?
                    as u64;
                let entity = Entity::new(subagent_id, 0);

                world
                    .add_component(entity, AgentLifecycle::Failed)
                    .map_err(|e| anyhow::anyhow!("failed to cancel subagent: {e}"))?;

                Ok(ToolResult::text(
                    json!({
                        "ok": true,
                        "subagent_entity_id": subagent_id,
                        "phase": "Failed",
                        "lifecycle": "Completed"
                    })
                    .to_string(),
                ))
            }

            _ => Err(anyhow::anyhow!(
                "Unknown command: {command}. Use spawn, list, collect, or cancel"
            )),
        }
    }
}
