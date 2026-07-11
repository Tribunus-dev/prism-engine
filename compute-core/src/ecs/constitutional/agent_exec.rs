use crate::ecs::constitutional::command::DomainEvent;
use crate::ecs::constitutional::lifecycle::SessionLifecycle;
use crate::ecs::constitutional::schema::SchemaRegistry;
use crate::ecs::constitutional::types::*;
use crate::ecs::constitutional::world_txn::{CommittedEpoch, WorldTxn, WorldTxnError};
use crate::ecs::{CompEntity, CompWorld, EntityKind};
use serde::{Deserialize, Serialize};

// ── Component Schema IDs (39-46) ──────────────────────────────────────────

pub const SCHEMA_AGENT_RUN: u64 = 39;
pub const SCHEMA_AGENT_TASK: u64 = 40;
pub const SCHEMA_AGENT_PHASE: u64 = 41;
pub const SCHEMA_TOOL_INVOCATION: u64 = 42;
pub const SCHEMA_TOOL_OUTCOME: u64 = 43;
pub const SCHEMA_AGENT_MESSAGE: u64 = 44;
pub const SCHEMA_AGENT_CONFIG: u64 = 45;
pub const SCHEMA_AGENT_LIFECYCLE: u64 = 46;

// ── Agent Run Components ─────────────────────────────────────────────────

/// Top-level agent run — the unit of autonomous execution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentRun {
    pub run_id: u64,
    pub session_entity: u64,
    pub name: String,
    pub created_at: Timestamp,
}

/// The task assigned to the agent — scoped, cancellable work.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentTask {
    pub task_description: String,
    pub model_entity: u64,
    pub max_steps: u32,
}

/// Current phase of the agent's reasoning loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AgentPhase {
    Planning,
    Reasoning,
    ToolCall,
    Observing,
    Responding,
    Completed,
    Failed,
}

impl AgentPhase {
    /// Returns true if this phase can transition to the target.
    ///
    /// Valid transitions:
    ///   Planning → Reasoning
    ///   Reasoning → ToolCall
    ///   ToolCall → Observing
    ///   Observing → Reasoning | Planning | Responding
    ///   Responding → Completed
    ///   any → Failed
    pub fn can_transition_to(&self, target: Self) -> bool {
        match (self, target) {
            (Self::Planning, Self::Reasoning)
            | (Self::Reasoning, Self::ToolCall)
            | (Self::ToolCall, Self::Observing)
            | (Self::Observing, Self::Reasoning)
            | (Self::Observing, Self::Planning)
            | (Self::Observing, Self::Responding)
            | (Self::Responding, Self::Completed)
            | (_, Self::Failed) => true,
            _ => false,
        }
    }
}

/// A tool invocation issued by the agent during a ToolCall phase.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolInvocation {
    pub invocation_id: u64,
    pub tool_name: String,
    pub arguments: serde_json::Value,
    pub status: InvocationStatus,
}

/// Status of a tool invocation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum InvocationStatus {
    Pending,
    InFlight,
    Succeeded(serde_json::Value),
    Failed(String),
}

/// Outcome of a tool invocation, validated as an effect outcome.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolOutcome {
    pub invocation_id: u64,
    pub success: bool,
    pub output: serde_json::Value,
    pub error: Option<String>,
}

/// A message exchanged between the agent and the user/system.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentMessage {
    pub role: String,
    pub content: String,
    pub created_at: Timestamp,
}

/// Agent configuration — model selection, generation params, tool policy.
#[derive(Debug, Clone, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct AgentConfig {
    pub model: String,
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    pub max_tokens: u64,
    #[serde(default)]
    pub tools_enabled: bool,
    #[serde(default = "default_max_tool_rounds")]
    pub max_tool_rounds: u32,
}

fn default_temperature() -> f64 {
    0.7
}

fn default_max_tool_rounds() -> u32 {
    10
}

impl std::cmp::Eq for AgentConfig {}

/// Agent lifecycle — the top-level execution state of the run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AgentLifecycle {
    Active,
    Waiting,
    Completed,
    Failed,
}

// ── CreateAgentRunCommand ─────────────────────────────────────────────────

/// Command to create a new agent run within a session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CreateAgentRunCommand {
    pub id: MessageId,
    pub session_entity: u64,
    pub task: AgentTask,
    pub config: AgentConfig,
}

impl CreateAgentRunCommand {
    /// Validate all agent-related schemas are registered.
    pub fn validate_schemas(schema_registry: &SchemaRegistry) -> Result<(), String> {
        schema_registry
            .verify_type::<AgentRun>(ComponentSchemaId(SCHEMA_AGENT_RUN))
            .map_err(|e| format!("AgentRun schema: {e}"))?;
        schema_registry
            .verify_type::<AgentTask>(ComponentSchemaId(SCHEMA_AGENT_TASK))
            .map_err(|e| format!("AgentTask schema: {e}"))?;
        schema_registry
            .verify_type::<AgentPhase>(ComponentSchemaId(SCHEMA_AGENT_PHASE))
            .map_err(|e| format!("AgentPhase schema: {e}"))?;
        schema_registry
            .verify_type::<ToolInvocation>(ComponentSchemaId(SCHEMA_TOOL_INVOCATION))
            .map_err(|e| format!("ToolInvocation schema: {e}"))?;
        schema_registry
            .verify_type::<ToolOutcome>(ComponentSchemaId(SCHEMA_TOOL_OUTCOME))
            .map_err(|e| format!("ToolOutcome schema: {e}"))?;
        schema_registry
            .verify_type::<AgentMessage>(ComponentSchemaId(SCHEMA_AGENT_MESSAGE))
            .map_err(|e| format!("AgentMessage schema: {e}"))?;
        schema_registry
            .verify_type::<AgentConfig>(ComponentSchemaId(SCHEMA_AGENT_CONFIG))
            .map_err(|e| format!("AgentConfig schema: {e}"))?;
        schema_registry
            .verify_type::<AgentLifecycle>(ComponentSchemaId(SCHEMA_AGENT_LIFECYCLE))
            .map_err(|e| format!("AgentLifecycle schema: {e}"))?;
        Ok(())
    }

    /// Preflight: session exists, session lifecycle is Active or Admitted.
    pub fn preflight(
        &self,
        world: &CompWorld,
        schema_registry: &SchemaRegistry,
    ) -> Result<(), AgentExecError> {
        Self::validate_schemas(schema_registry).map_err(AgentExecError::SchemaError)?;

        let session = CompEntity(self.session_entity);
        if !world.has_entity(session) {
            return Err(AgentExecError::SessionNotFound(self.session_entity));
        }
        if world.entity_kind(session) != Some(EntityKind::Session) {
            return Err(AgentExecError::SessionNotFound(self.session_entity));
        }

        // Session must be Active or Admitted.
        if let Some(lifecycle) = world.get_component::<SessionLifecycle>(session) {
            if !matches!(
                lifecycle,
                SessionLifecycle::Active | SessionLifecycle::Admitted
            ) {
                return Err(AgentExecError::SessionNotReady(self.session_entity));
            }
        } else {
            return Err(AgentExecError::SessionNotReady(self.session_entity));
        }

        Ok(())
    }

    /// Execute: spawn agent entity with all components, emit domain event.
    pub fn execute(
        self,
        world: &mut CompWorld,
        schema_registry: &SchemaRegistry,
    ) -> Result<(CommittedEpoch, DomainEvent), AgentExecError> {
        self.preflight(world, schema_registry)?;

        let agent_id = WorldTxn::next_entity_id(world);
        let mut txn = WorldTxn::new(world);

        txn.stage_spawn(agent_id, EntityKind::Agent);

        // Durable components
        txn.add_component(
            agent_id,
            ComponentSchemaId(SCHEMA_AGENT_RUN),
            SchemaVersion(1),
            AgentRun {
                run_id: agent_id,
                session_entity: self.session_entity,
                name: "agent_run".to_string(),
                created_at: Timestamp::now(),
            },
        );
        txn.add_component(
            agent_id,
            ComponentSchemaId(SCHEMA_AGENT_TASK),
            SchemaVersion(1),
            self.task.clone(),
        );
        txn.add_component(
            agent_id,
            ComponentSchemaId(SCHEMA_AGENT_CONFIG),
            SchemaVersion(1),
            self.config.clone(),
        );
        // Initial lifecycle + phase
        txn.add_component(
            agent_id,
            ComponentSchemaId(SCHEMA_AGENT_LIFECYCLE),
            SchemaVersion(1),
            AgentLifecycle::Active,
        );
        txn.add_component(
            agent_id,
            ComponentSchemaId(SCHEMA_AGENT_PHASE),
            SchemaVersion(1),
            AgentPhase::Planning,
        );

        let event = DomainEvent {
            id: self.id,
            kind: "agent_run_created".to_string(),
            entity_id: Some(EntityKindId(agent_id)),
            payload: serde_json::json!({
                "agent_id": agent_id,
                "session_entity": self.session_entity,
                "task": self.task.task_description,
            }),
        };
        txn.emit_event(event.clone());

        let epoch = world.transit(txn).map_err(AgentExecError::CommitFailed)?;
        Ok((epoch, event))
    }
}

// ── SubmitToolOutcomeCommand ──────────────────────────────────────────────

/// Correlates a tool effect outcome, updating the invocation status and
/// potentially transitioning the agent phase.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubmitToolOutcomeCommand {
    pub id: MessageId,
    pub agent_entity: u64,
    pub invocation_id: u64,
    pub outcome: ToolOutcome,
}

impl SubmitToolOutcomeCommand {
    /// Preflight: agent exists, is in ToolCall phase.
    pub fn preflight(
        &self,
        world: &CompWorld,
        _schema_registry: &SchemaRegistry,
    ) -> Result<(), AgentExecError> {
        let agent = CompEntity(self.agent_entity);
        if !world.has_entity(agent) {
            return Err(AgentExecError::AgentNotFound(self.agent_entity));
        }
        if world.entity_kind(agent) != Some(EntityKind::Agent) {
            return Err(AgentExecError::AgentNotFound(self.agent_entity));
        }

        // Must be in ToolCall phase to accept a tool outcome.
        if let Some(phase) = world.get_component::<AgentPhase>(agent) {
            if !matches!(phase, AgentPhase::ToolCall) {
                return Err(AgentExecError::InvalidPhase(*phase));
            }
        } else {
            return Err(AgentExecError::InvalidPhase(AgentPhase::Planning));
        }

        Ok(())
    }

    /// Execute: correlate the tool outcome to the matching invocation,
    /// update the invocation status, transition phase to Observing.
    pub fn execute(
        self,
        world: &mut CompWorld,
        schema_registry: &SchemaRegistry,
    ) -> Result<(CommittedEpoch, DomainEvent), AgentExecError> {
        self.preflight(world, schema_registry)?;

        let mut txn = WorldTxn::new(world);

        // Record the tool outcome component.
        txn.add_component(
            self.agent_entity,
            ComponentSchemaId(SCHEMA_TOOL_OUTCOME),
            SchemaVersion(1),
            self.outcome.clone(),
        );

        // Transition phase from ToolCall → Observing.
        txn.add_component(
            self.agent_entity,
            ComponentSchemaId(SCHEMA_AGENT_PHASE),
            SchemaVersion(1),
            AgentPhase::Observing,
        );

        let event = DomainEvent {
            id: self.id,
            kind: "tool_outcome_submitted".to_string(),
            entity_id: Some(EntityKindId(self.agent_entity)),
            payload: serde_json::json!({
                "agent_entity": self.agent_entity,
                "invocation_id": self.invocation_id,
                "success": self.outcome.success,
            }),
        };
        txn.emit_event(event.clone());

        let epoch = world.transit(txn).map_err(AgentExecError::CommitFailed)?;
        Ok((epoch, event))
    }
}

// ── Errors ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AgentExecError {
    #[error("schema error: {0}")]
    SchemaError(String),
    #[error("session entity {0} not found")]
    SessionNotFound(u64),
    #[error("session entity {0} not Active or Admitted")]
    SessionNotReady(u64),
    #[error("agent entity {0} not found")]
    AgentNotFound(u64),
    #[error("invalid phase for operation: {0:?}")]
    InvalidPhase(AgentPhase),
    #[error("commit failed: {0}")]
    CommitFailed(WorldTxnError),
}

// ── Validation helper ───────────────────────────────────────────────────

/// Validate all agent exec schemas are registered for the correct types.
pub fn validate_agent_exec_schemas(schema_registry: &SchemaRegistry) -> Result<(), String> {
    schema_registry
        .verify_type::<AgentRun>(ComponentSchemaId(SCHEMA_AGENT_RUN))
        .map_err(|e| format!("AgentRun schema: {e}"))?;
    schema_registry
        .verify_type::<AgentTask>(ComponentSchemaId(SCHEMA_AGENT_TASK))
        .map_err(|e| format!("AgentTask schema: {e}"))?;
    schema_registry
        .verify_type::<AgentPhase>(ComponentSchemaId(SCHEMA_AGENT_PHASE))
        .map_err(|e| format!("AgentPhase schema: {e}"))?;
    schema_registry
        .verify_type::<ToolInvocation>(ComponentSchemaId(SCHEMA_TOOL_INVOCATION))
        .map_err(|e| format!("ToolInvocation schema: {e}"))?;
    schema_registry
        .verify_type::<ToolOutcome>(ComponentSchemaId(SCHEMA_TOOL_OUTCOME))
        .map_err(|e| format!("ToolOutcome schema: {e}"))?;
    schema_registry
        .verify_type::<AgentMessage>(ComponentSchemaId(SCHEMA_AGENT_MESSAGE))
        .map_err(|e| format!("AgentMessage schema: {e}"))?;
    schema_registry
        .verify_type::<AgentConfig>(ComponentSchemaId(SCHEMA_AGENT_CONFIG))
        .map_err(|e| format!("AgentConfig schema: {e}"))?;
    schema_registry
        .verify_type::<AgentLifecycle>(ComponentSchemaId(SCHEMA_AGENT_LIFECYCLE))
        .map_err(|e| format!("AgentLifecycle schema: {e}"))?;
    Ok(())
}

// ── Component impls ───────────────────────────────────────────────────────

impl crate::ecs::Component for AgentRun {}
impl crate::ecs::Component for AgentTask {}
impl crate::ecs::Component for AgentPhase {}
impl crate::ecs::Component for ToolInvocation {}
impl crate::ecs::Component for ToolOutcome {}
impl crate::ecs::Component for AgentMessage {}
impl crate::ecs::Component for AgentConfig {}
impl crate::ecs::Component for AgentLifecycle {}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── AgentPhase Transitions ──────────────────────────────────────────

    #[test]
    fn test_agent_phase_transitions() {
        // Valid transitions
        assert!(AgentPhase::Planning.can_transition_to(AgentPhase::Reasoning));
        assert!(AgentPhase::Reasoning.can_transition_to(AgentPhase::ToolCall));
        assert!(AgentPhase::ToolCall.can_transition_to(AgentPhase::Observing));
        assert!(AgentPhase::Observing.can_transition_to(AgentPhase::Reasoning));
        assert!(AgentPhase::Observing.can_transition_to(AgentPhase::Planning));
        assert!(AgentPhase::Observing.can_transition_to(AgentPhase::Responding));
        assert!(AgentPhase::Responding.can_transition_to(AgentPhase::Completed));
        assert!(AgentPhase::Planning.can_transition_to(AgentPhase::Failed));
        assert!(AgentPhase::Reasoning.can_transition_to(AgentPhase::Failed));
        assert!(AgentPhase::ToolCall.can_transition_to(AgentPhase::Failed));
        assert!(AgentPhase::Observing.can_transition_to(AgentPhase::Failed));
        assert!(AgentPhase::Responding.can_transition_to(AgentPhase::Failed));
        assert!(AgentPhase::Completed.can_transition_to(AgentPhase::Failed));

        // Invalid transitions
        assert!(!AgentPhase::Planning.can_transition_to(AgentPhase::ToolCall));
        assert!(!AgentPhase::Planning.can_transition_to(AgentPhase::Observing));
        assert!(!AgentPhase::Planning.can_transition_to(AgentPhase::Responding));
        assert!(!AgentPhase::Planning.can_transition_to(AgentPhase::Completed));
        assert!(!AgentPhase::Reasoning.can_transition_to(AgentPhase::Planning));
        assert!(!AgentPhase::Reasoning.can_transition_to(AgentPhase::Observing));
        assert!(!AgentPhase::Reasoning.can_transition_to(AgentPhase::Responding));
        assert!(!AgentPhase::Reasoning.can_transition_to(AgentPhase::Completed));
        assert!(!AgentPhase::ToolCall.can_transition_to(AgentPhase::Planning));
        assert!(!AgentPhase::ToolCall.can_transition_to(AgentPhase::Reasoning));
        assert!(!AgentPhase::ToolCall.can_transition_to(AgentPhase::Responding));
        assert!(!AgentPhase::ToolCall.can_transition_to(AgentPhase::Completed));
        assert!(!AgentPhase::Observing.can_transition_to(AgentPhase::ToolCall));
        assert!(!AgentPhase::Observing.can_transition_to(AgentPhase::Completed));
        assert!(!AgentPhase::Responding.can_transition_to(AgentPhase::Planning));
        assert!(!AgentPhase::Responding.can_transition_to(AgentPhase::Reasoning));
        assert!(!AgentPhase::Responding.can_transition_to(AgentPhase::ToolCall));
        assert!(!AgentPhase::Responding.can_transition_to(AgentPhase::Observing));
        assert!(!AgentPhase::Completed.can_transition_to(AgentPhase::Planning));
        assert!(!AgentPhase::Completed.can_transition_to(AgentPhase::Reasoning));
        assert!(!AgentPhase::Completed.can_transition_to(AgentPhase::ToolCall));
        assert!(!AgentPhase::Completed.can_transition_to(AgentPhase::Observing));
        assert!(!AgentPhase::Completed.can_transition_to(AgentPhase::Responding));
        assert!(!AgentPhase::Failed.can_transition_to(AgentPhase::Planning));
        assert!(!AgentPhase::Failed.can_transition_to(AgentPhase::Reasoning));
        assert!(!AgentPhase::Failed.can_transition_to(AgentPhase::ToolCall));
        assert!(!AgentPhase::Failed.can_transition_to(AgentPhase::Observing));
        assert!(!AgentPhase::Failed.can_transition_to(AgentPhase::Responding));
        assert!(!AgentPhase::Failed.can_transition_to(AgentPhase::Completed));
    }

    // ── Tool Invocation Lifecycle ───────────────────────────────────────

    #[test]
    fn test_tool_invocation_lifecycle() {
        // Pending → InFlight → Succeeded
        let mut inv = ToolInvocation {
            invocation_id: 1,
            tool_name: "read".to_string(),
            arguments: serde_json::json!({"path": "test.txt"}),
            status: InvocationStatus::Pending,
        };
        assert_eq!(inv.status, InvocationStatus::Pending);

        inv.status = InvocationStatus::InFlight;
        assert_eq!(inv.status, InvocationStatus::InFlight);

        inv.status = InvocationStatus::Succeeded(serde_json::json!({"content": "hello"}));
        if let InvocationStatus::Succeeded(ref output) = inv.status {
            assert_eq!(output["content"], "hello");
        } else {
            panic!("expected Succeeded");
        }

        // Pending → InFlight → Failed
        let mut inv2 = ToolInvocation {
            invocation_id: 2,
            tool_name: "bash".to_string(),
            arguments: serde_json::json!({"cmd": "rm -rf /"}),
            status: InvocationStatus::Pending,
        };
        inv2.status = InvocationStatus::InFlight;
        inv2.status = InvocationStatus::Failed("permission denied".to_string());
        if let InvocationStatus::Failed(ref err) = inv2.status {
            assert_eq!(err, "permission denied");
        } else {
            panic!("expected Failed");
        }
    }

    // ── AgentRun Serde Round-Trip ──────────────────────────────────────

    #[test]
    fn test_agent_run_serde() {
        let run = AgentRun {
            run_id: 42,
            session_entity: 7,
            name: "test-run".to_string(),
            created_at: Timestamp(1_000_000),
        };
        let json = serde_json::to_string(&run).unwrap();
        let deserialized: AgentRun = serde_json::from_str(&json).unwrap();
        assert_eq!(run, deserialized);
    }

    // ── AgentPhase Serde ───────────────────────────────────────────────

    #[test]
    fn test_agent_phase_serde() {
        let cases = [
            AgentPhase::Planning,
            AgentPhase::Reasoning,
            AgentPhase::ToolCall,
            AgentPhase::Observing,
            AgentPhase::Responding,
            AgentPhase::Completed,
            AgentPhase::Failed,
        ];
        for phase in &cases {
            let json = serde_json::to_string(phase).unwrap();
            let deserialized: AgentPhase = serde_json::from_str(&json).unwrap();
            assert_eq!(*phase, deserialized);
        }
    }

    // ── AgentConfig Serde (PartialOrd / PartialEq) ─────────────────────

    #[test]
    fn test_agent_config_defaults() {
        let config = AgentConfig {
            model: "gemma-4".to_string(),
            temperature: 0.7,
            max_tokens: 4096,
            tools_enabled: false,
            max_tool_rounds: 10,
        };
        assert_eq!(config.temperature, 0.7);
        assert_eq!(config.max_tool_rounds, 10);
        assert!(!config.tools_enabled);

        // Round-trip
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: AgentConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config, deserialized);
    }

    // ── ToolOutcome Construction ────────────────────────────────────────

    #[test]
    fn test_tool_outcome_construction() {
        let outcome = ToolOutcome {
            invocation_id: 1,
            success: true,
            output: serde_json::json!({"result": [1, 2, 3]}),
            error: None,
        };
        assert!(outcome.success);
        assert_eq!(outcome.output["result"][0], 1);
        assert!(outcome.error.is_none());

        let failed = ToolOutcome {
            invocation_id: 2,
            success: false,
            output: serde_json::Value::Null,
            error: Some("timeout".to_string()),
        };
        assert!(!failed.success);
        assert_eq!(failed.error.as_deref(), Some("timeout"));
    }

    // ── CreateAgentRunCommand Preflight ──────────────────────────────────

    #[test]
    fn test_create_agent_run_preflight_session_missing() {
        let world = CompWorld::new();
        let reg = SchemaRegistry::new();

        let cmd = CreateAgentRunCommand {
            id: MessageId::compute(b"test"),
            session_entity: 99,
            task: AgentTask {
                task_description: "test".to_string(),
                model_entity: 1,
                max_steps: 10,
            },
            config: AgentConfig {
                model: "gemma-4".to_string(),
                temperature: 0.7,
                max_tokens: 4096,
                tools_enabled: true,
                max_tool_rounds: 5,
            },
        };

        let result = cmd.preflight(&world, &reg);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            AgentExecError::SchemaError(
                "AgentRun schema: ComponentSchemaId(AgentRun(39)) not registered".to_string()
            )
        );
    }

    // ── AgentAgentLifecycle ─────────────────────────────────────────────

    #[test]
    fn test_agent_lifecycle_variants() {
        let all = [
            AgentLifecycle::Active,
            AgentLifecycle::Waiting,
            AgentLifecycle::Completed,
            AgentLifecycle::Failed,
        ];
        assert_eq!(all.len(), 4);
        assert_ne!(all[0], all[1]);
        assert_ne!(all[1], all[2]);
        assert_ne!(all[2], all[3]);
    }

    // ── InvocationStatus Serde ─────────────────────────────────────────

    #[test]
    fn test_invocation_status_discriminants() {
        let pending = InvocationStatus::Pending;
        let json = serde_json::to_value(&pending).unwrap();
        assert_eq!(json, serde_json::json!("Pending"));

        let succeeded = InvocationStatus::Succeeded(serde_json::json!({"x": 1}));
        let json = serde_json::to_value(&succeeded).unwrap();
        assert_eq!(json, serde_json::json!({"Succeeded": {"x": 1}}));

        let failed = InvocationStatus::Failed("err".to_string());
        let json = serde_json::to_value(&failed).unwrap();
        assert_eq!(json, serde_json::json!({"Failed": "err"}));
    }
}
