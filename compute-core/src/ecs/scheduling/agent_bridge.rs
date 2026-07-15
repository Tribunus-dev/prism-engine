//! Bridge wrapping the constitutional `CreateAgentRunCommand` behind a simple
//! synchronous API for production agent-run callers.
//!
//! Owns an `Arc<RwLock<World>>` (the constitutional world) and a
//! pre-populated `SchemaRegistry`.  Provides synchronous methods that
//! lock the world, construct the constitutional command, then execute it.
//!
//! # Pattern
//!
//! Follows the same design as [`WorkLifecycleBridge`], [`ExecutionLeaseBridge`],
//! and [`CompilationJobBridge`]:
//!
//! 1. Lock the world for writing
//! 2. Construct the relevant constitutional command
//! 3. Call the command's `execute` (which internally runs preflight)
//! 4. Return the result or a string error

use crate::ecs::constitutional::agent_exec::{
    AgentConfig, AgentLifecycle, AgentMessage, AgentPhase, AgentRun, AgentTask,
    CreateAgentRunCommand, ToolInvocation, ToolOutcome, SCHEMA_AGENT_CONFIG,
    SCHEMA_AGENT_LIFECYCLE, SCHEMA_AGENT_MESSAGE, SCHEMA_AGENT_PHASE, SCHEMA_AGENT_RUN,
    SCHEMA_AGENT_TASK, SCHEMA_TOOL_INVOCATION, SCHEMA_TOOL_OUTCOME,
};
use crate::ecs::constitutional::schema::{ComponentDurability, SchemaRegistry};
use crate::ecs::constitutional::types::{ComponentSchemaId, MessageId, SchemaVersion};
use crate::ecs::World;
use std::sync::{Arc, RwLock};
use uuid::Uuid;

// SAFETY: AgentBridge only accesses the World through an RwLock(World) that
// properly synchronizes all access. The `!Sync` nature of `World` comes from
// internal FnOnce closures, not from any data-race-vulnerable state — the
// RwLock ensures mutual exclusion across threads. This is the same pattern
// used by other bridge structures in this codebase that hold Arc<RwLock<World>>.
unsafe impl Sync for AgentBridge {}
// SAFETY: See above for Sync. The RwLock provides mutual exclusion for all
// World access, making it safe to transfer the bridge across thread boundaries.
unsafe impl Send for AgentBridge {}

/// Thin API over constitutional agent run commands.
///
/// Each method:
/// 1. Generates a unique `MessageId` via UUID
/// 2. Locks the world for writing
/// 3. Constructs the constitutional command
/// 4. Executes it (preflight + execute internally)
/// 5. Returns the result
pub struct AgentBridge {
    world: Arc<RwLock<World>>,
    schema_registry: SchemaRegistry,
}

impl AgentBridge {
    /// Wrap an existing `Arc<RwLock<World>>`, registering all agent-exec
    /// schemas on construction so schema validation inside each
    /// constitutional command passes.
    pub fn new(world: Arc<RwLock<World>>) -> Self {
        let mut schema_registry = SchemaRegistry::new();
        Self::register_agent_schemas(&mut schema_registry);
        Self {
            world,
            schema_registry,
        }
    }

    /// Register all agent exec domain schemas into the given registry.
    fn register_agent_schemas(reg: &mut SchemaRegistry) {
        reg.register_for_type::<AgentRun>(
            ComponentSchemaId(SCHEMA_AGENT_RUN),
            SchemaVersion(1),
            "AgentRun",
            "Agent run metadata",
            ComponentDurability::Durable,
        );
        reg.register_for_type::<AgentTask>(
            ComponentSchemaId(SCHEMA_AGENT_TASK),
            SchemaVersion(1),
            "AgentTask",
            "Agent task description",
            ComponentDurability::Durable,
        );
        reg.register_for_type::<AgentPhase>(
            ComponentSchemaId(SCHEMA_AGENT_PHASE),
            SchemaVersion(1),
            "AgentPhase",
            "Agent execution phase",
            ComponentDurability::Durable,
        );
        reg.register_for_type::<ToolInvocation>(
            ComponentSchemaId(SCHEMA_TOOL_INVOCATION),
            SchemaVersion(1),
            "ToolInvocation",
            "Tool invocation record",
            ComponentDurability::Durable,
        );
        reg.register_for_type::<ToolOutcome>(
            ComponentSchemaId(SCHEMA_TOOL_OUTCOME),
            SchemaVersion(1),
            "ToolOutcome",
            "Tool outcome record",
            ComponentDurability::Durable,
        );
        reg.register_for_type::<AgentMessage>(
            ComponentSchemaId(SCHEMA_AGENT_MESSAGE),
            SchemaVersion(1),
            "AgentMessage",
            "Agent message record",
            ComponentDurability::Durable,
        );
        reg.register_for_type::<AgentConfig>(
            ComponentSchemaId(SCHEMA_AGENT_CONFIG),
            SchemaVersion(1),
            "AgentConfig",
            "Agent configuration",
            ComponentDurability::Durable,
        );
        reg.register_for_type::<AgentLifecycle>(
            ComponentSchemaId(SCHEMA_AGENT_LIFECYCLE),
            SchemaVersion(1),
            "AgentLifecycle",
            "Agent lifecycle state",
            ComponentDurability::Durable,
        );
    }

    /// Create a new agent run within a session.
    ///
    /// Constructs a [`CreateAgentRunCommand`], executes it against the
    /// constitutional world, and returns the newly allocated agent entity id
    /// on success.
    pub fn create_agent_run(
        &self,
        session_entity: u64,
        task: AgentTask,
        config: AgentConfig,
    ) -> Result<u64, String> {
        let mut world = self.world.write().map_err(|e| e.to_string())?;

        let uuid = Uuid::new_v4();
        let id = MessageId::compute(uuid.as_bytes());

        let cmd = CreateAgentRunCommand {
            id,
            session_entity,
            task,
            config,
        };

        let (_epoch, event) = cmd
            .execute(&mut world, &self.schema_registry)
            .map_err(|e| e.to_string())?;

        // The domain event carries the new agent entity id.
        let agent_id = event
            .entity_id
            .ok_or_else(|| "no entity_id in domain event".to_string())?
            .0;

        Ok(agent_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::EntityKind;

    /// Helper: create a minimal session entity in the constitutional world.
    fn spawn_session(world: &mut crate::ecs::World) -> u64 {
        use crate::ecs::constitutional::lifecycle::SessionLifecycle;
        use crate::ecs::constitutional::world_txn::WorldTxn;

        let mut txn = WorldTxn::new(world);
        let e = WorldTxn::next_entity_id(world);
        txn.stage_spawn(e, EntityKind::Session);
        txn.put_durable(e, SessionLifecycle::Active);
        txn.put_durable(
            e,
            crate::ecs::constitutional::session::SessionConfig {
                max_tokens: 4096,
                max_input_tokens: 2048,
                max_output_tokens: 2048,
                batch_size: 1,
                priority: 1,
                deadline_epochs: 100,
            },
        );
        world.transit(txn).unwrap();
        e.id()
    }

    #[test]
    fn test_create_agent_run() {
        let mut world = World::new();
        let session_id = spawn_session(&mut world);
        let bridge = AgentBridge::new(Arc::new(RwLock::new(world)));

        let agent_id = bridge
            .create_agent_run(
                session_id,
                AgentTask {
                    task_description: "analyze logs".to_string(),
                    model_entity: 1,
                    max_steps: 10,
                },
                AgentConfig {
                    model: "gemma-4".to_string(),
                    temperature: 0.7,
                    max_tokens: 4096,
                    tools_enabled: true,
                    max_tool_rounds: 5,
                },
            )
            .expect("create_agent_run should succeed");

        assert!(agent_id > 0, "agent entity id should be positive");
    }

    #[test]
    fn test_create_agent_run_session_missing() {
        let world = World::new();
        let bridge = AgentBridge::new(Arc::new(RwLock::new(world)));

        let result = bridge.create_agent_run(
            9999, // non-existent session
            AgentTask {
                task_description: "test".to_string(),
                model_entity: 1,
                max_steps: 10,
            },
            AgentConfig {
                model: "gemma-4".to_string(),
                temperature: 0.7,
                max_tokens: 4096,
                tools_enabled: true,
                max_tool_rounds: 5,
            },
        );

        assert!(result.is_err(), "missing session should fail");
    }
}
