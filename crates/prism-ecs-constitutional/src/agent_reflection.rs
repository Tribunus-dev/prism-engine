use crate::types::*;
use crate::world_txn::{
    ClassifiedComponent, DurableClass, DurableComponent,
};
use prism_ecs_core::Component;
use serde::{Deserialize, Serialize};

// ── Component Schema IDs (48) ──────────────────────────────────────────

pub const SCHEMA_REFLECTION_RESULT: u64 = 48;

// ── ReflectionResult ───────────────────────────────────────────────────

/// Result of agent reflection — inference on accumulated observations
/// to decide the next action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReflectionResult {
    pub reasoning: String,
    pub decision: ReflectionDecision,
    pub tokens_used: u32,
}

// ── ReflectionDecision ─────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReflectionDecision {
    /// More tool calls are needed — continue execution
    Continue(Vec<crate::agent_plan::ToolCall>),
    /// Task is complete — final summary
    Complete(String),
    /// Re-plan with new context
    RePlan(String),
}

// ── Component impls ────────────────────────────────────────────────────

impl Component for ReflectionResult {}

impl ClassifiedComponent for ReflectionResult {
    type Class = DurableClass;
}

impl DurableComponent for ReflectionResult {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.agent",
        id: 48,
        version: 1,
    };
}
