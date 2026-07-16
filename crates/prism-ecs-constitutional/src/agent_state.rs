//! Agent state machine — orchestrates agent entities through high-level states.
//!
//! States: `Idle → Planning → Executing → Observing → Reflecting → Finished/Error`.
//! The `tick()` system scans all active agent entities and records `TransitionRecord`
//! entries when preconditions are met (components available, invocations resolved).

use crate::agent_exec::{AgentLifecycle, AgentPhase, InvocationStatus, ToolInvocation};
use crate::agent_plan::AgentPlan;
use crate::agent_reflection::{ReflectionDecision, ReflectionResult};
use prism_ecs_core::{Entity, World};
use serde::{Deserialize, Serialize};

// ── AgentState ────────────────────────────────────────────────────────────

/// Top-level state in the agent state machine.
///
/// Each state maps to (but is not identical to) the lower-level
/// [`AgentPhase`] used at the execution-command layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AgentState {
    /// Initial — no plan assigned yet.
    Idle,
    /// A plan is being formulated.
    Planning,
    /// Tools / model calls are being executed.
    Executing,
    /// Observing tool outcomes.
    Observing,
    /// Reflecting on outcomes to decide next step.
    Reflecting,
    /// Task completed successfully.
    Finished,
    /// Unrecoverable error.
    Error,
}

impl AgentState {
    /// Returns `true` if a transition from `self` to `next` is valid.
    ///
    /// # Valid transitions
    /// - `Idle → Planning`
    /// - `Planning → Executing | Error`
    /// - `Executing → Observing | Error`
    /// - `Observing → Reflecting | Planning | Executing | Error`
    /// - `Reflecting → Finished | Planning | Executing | Error`
    /// - `Finished → Planning` (replan)
    /// - All other pairs are invalid.
    pub fn can_transition_to(&self, next: &AgentState) -> bool {
        matches!(
            (self, next),
            (AgentState::Idle, AgentState::Planning)
                | (AgentState::Planning, AgentState::Executing)
                | (AgentState::Planning, AgentState::Error)
                | (AgentState::Executing, AgentState::Observing)
                | (AgentState::Executing, AgentState::Error)
                | (AgentState::Observing, AgentState::Reflecting)
                | (AgentState::Observing, AgentState::Planning)
                | (AgentState::Observing, AgentState::Executing)
                | (AgentState::Observing, AgentState::Error)
                | (AgentState::Reflecting, AgentState::Finished)
                | (AgentState::Reflecting, AgentState::Planning)
                | (AgentState::Reflecting, AgentState::Executing)
                | (AgentState::Reflecting, AgentState::Error)
                | (AgentState::Finished, AgentState::Planning)
        )
    }
}

// ── TransitionRecord ─────────────────────────────────────────────────────

/// A recorded state transition for one agent entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionRecord {
    /// The agent entity that transitioned.
    pub agent_entity: Entity,
    /// State before the transition.
    pub from_state: AgentState,
    /// State after the transition.
    pub to_state: AgentState,
    /// Human-readable reason for the transition.
    pub reason: String,
    /// Wall-clock timestamp (ms since epoch) when the transition was recorded.
    pub timestamp: f64,
}

// ── State mapping helpers ─────────────────────────────────────────────────

/// Map an [`AgentPhase`] to the corresponding high-level [`AgentState`].
fn phase_to_state(phase: AgentPhase) -> AgentState {
    match phase {
        AgentPhase::Planning => AgentState::Planning,
        AgentPhase::Reasoning => AgentState::Reflecting,
        AgentPhase::ToolCall => AgentState::Executing,
        AgentPhase::Observing => AgentState::Observing,
        AgentPhase::Responding => AgentState::Reflecting,
        AgentPhase::Completed => AgentState::Finished,
        AgentPhase::Failed => AgentState::Error,
    }
}

// ── Tick ──────────────────────────────────────────────────────────────────

/// Scan all active agent entities and return transition records for any that
/// have met the precondition to advance to the next state.
///
/// This function is idempotent — it only *records* transitions; the caller
/// is responsible for applying them to the underlying world state.
pub fn tick(world: &World) -> Result<Vec<TransitionRecord>, String> {
    let mut transitions = Vec::new();
    let now = now_ms();

    // Collect all entities that have BOTH AgentPhase and AgentLifecycle.
    let agents: Vec<(Entity, AgentPhase, AgentLifecycle)> = world
        .query2::<AgentPhase, AgentLifecycle>()
        .map(|(e, p, l)| (e, *p, *l))
        .collect();

    for (entity, phase, lifecycle) in &agents {
        if *lifecycle != AgentLifecycle::Active {
            continue;
        }
        let from_state = phase_to_state(*phase);

        match phase {
            AgentPhase::Planning => {
                // A plan has been produced → transition to Executing (ToolCall phase).
                if world.has_component::<AgentPlan>(*entity) {
                    transitions.push(TransitionRecord {
                        agent_entity: *entity,
                        from_state,
                        to_state: AgentState::Executing,
                        reason: "plan_produced".to_string(),
                        timestamp: now,
                    });
                }
            }

            AgentPhase::ToolCall => {
                // All invocations have been resolved → transition to Observing.
                let all_done = world
                    .query::<ToolInvocation>()
                    .filter(|(inv_entity, inv)| {
                        // In the current schema, each ToolInvocation is stored on its own
                        // entity.  The invocation entity's `invocation_id` links it to the
                        // SubmitToolOutcomeCommand that created it; without an `agent_entity`
                        // field we check every invocation entity in the world.
                        //
                        // TODO: add an `agent_entity` field to ToolInvocation so filtering
                        //       is entity-specific rather than global.
                        inv.status != InvocationStatus::Pending && inv_entity.id() != 0
                    })
                    .count();
                let total = world.query::<ToolInvocation>().count();

                if all_done > 0 && all_done == total {
                    transitions.push(TransitionRecord {
                        agent_entity: *entity,
                        from_state,
                        to_state: AgentState::Observing,
                        reason: "tool_outcomes_collected".to_string(),
                        timestamp: now,
                    });
                }
            }

            AgentPhase::Observing | AgentPhase::Reasoning => {
                // A reflection result has been produced → decide next state.
                if let Some(result) = world.get_component::<ReflectionResult>(*entity) {
                    let (to_state, reason) = match &result.decision {
                        ReflectionDecision::Complete(_) => {
                            (AgentState::Finished, "task_complete".to_string())
                        }
                        ReflectionDecision::RePlan(_) => {
                            (AgentState::Planning, "replan_needed".to_string())
                        }
                        ReflectionDecision::Continue(_) => {
                            (AgentState::Executing, "more_tools".to_string())
                        }
                    };
                    transitions.push(TransitionRecord {
                        agent_entity: *entity,
                        from_state,
                        to_state,
                        reason,
                        timestamp: now,
                    });
                }
            }

            // Terminal phases — nothing to advance.
            AgentPhase::Responding => {}
            AgentPhase::Completed | AgentPhase::Failed => {}
        }
    }

    Ok(transitions)
}

// ── Helpers ───────────────────────────────────────────────────────────────

fn now_ms() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
        * 1000.0
}
