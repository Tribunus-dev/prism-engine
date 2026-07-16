//! Agent planning subsystem — plans, tool calls, subagent tracking.
//!
//! Components for agent planning and reflection:
//! - `AgentPlan` / `PlanStep` — the plan itself with typed steps
//! - `ToolCall` — a tool invocation record (shared with reflection)
//! - `ParentAgentId` — parent of a subagent entity
//! - `SubagentResult` — result of a subagent execution
use crate::types::{SchemaKey, Timestamp};
use crate::world_txn::{ClassifiedComponent, DurableClass, DurableComponent};
use prism_ecs_core::{Component, Entity};
use serde::{Deserialize, Serialize};

// ── Component Schema IDs (47-50) ──────────────────────────────────────────

pub const SCHEMA_AGENT_PLAN: u64 = 47;
pub const SCHEMA_PARENT_AGENT: u64 = 49;
pub const SCHEMA_SUBAGENT_RESULT: u64 = 50;

// ── AgentPlan ─────────────────────────────────────────────────────────────

/// An agent's plan — a sequence of steps with reasoning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentPlan {
    pub plan_steps: Vec<PlanStep>,
    pub reasoning: String,
    pub created_at: Timestamp,
}

/// A single step in an agent plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PlanStep {
    /// Invoke a registered tool.
    ToolCall {
        tool: String,
        args: serde_json::Value,
    },
    /// Delegate work to a subagent.
    SubTask {
        task_description: String,
        max_steps: u32,
    },
    /// Generate a model inference.
    Inference { prompt: String, max_tokens: u32 },
}

// ── ToolCall (shared between plan and reflection) ────────────────────────

/// A tool call record — tool name and JSON arguments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub tool: String,
    pub args: serde_json::Value,
}

// ── ParentAgentId (subagent tracking) ────────────────────────────────────

/// Marks a subagent's parent agent entity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParentAgentId(pub Entity);

// ── SubagentResult ───────────────────────────────────────────────────────

/// Result of a subagent execution, delivered to the parent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubagentResult(pub String);

// ── Component impls ──────────────────────────────────────────────────────

impl Component for AgentPlan {}
impl ClassifiedComponent for AgentPlan {
    type Class = DurableClass;
}
impl DurableComponent for AgentPlan {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.agent",
        id: 47,
        version: 1,
    };
}

impl Component for ParentAgentId {}
impl ClassifiedComponent for ParentAgentId {
    type Class = DurableClass;
}
impl DurableComponent for ParentAgentId {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.agent",
        id: 49,
        version: 1,
    };
}

impl Component for SubagentResult {}
impl ClassifiedComponent for SubagentResult {
    type Class = DurableClass;
}
impl DurableComponent for SubagentResult {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.agent",
        id: 50,
        version: 1,
    };
}
