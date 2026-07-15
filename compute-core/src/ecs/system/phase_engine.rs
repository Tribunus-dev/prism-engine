
use crate::ecs::Entity;
use crate::ecs::{World, CompilerSystem, Component, EntityKind, SchedulePhase};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// PhaseState — lifecycle state for a phase
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum PhaseState {
    Dormant,
    Ready,
    ResidencyPending,
    LeasePending,
    Admitted,
    Dispatched,
    AwaitingCompletion,
    Validating,
    Publishing,
    Complete,
    Rejected,
    Cancelled,
    TimedOut,
    FailedBeforePublication,
    FailedAfterTentativeState,
    RolledBack,
    FallbackPending,
    FallbackComplete,
    Quarantined,
}

// ---------------------------------------------------------------------------
// PhaseLifecycleComponent
// ---------------------------------------------------------------------------

/// Tracks the lifecycle of a single phase through the phase engine state
/// machine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct PhaseLifecycleComponent {
    pub phase: PhaseState,
    pub started_at: u64,
}
impl Component for PhaseLifecycleComponent {}

/// Manages phase lifecycle transitions — advances entities through
/// the phase engine state machine (Dormant → Ready → Admitted → …
/// → Complete / Failed).
///
/// Runs every tick of the `Execution` phase.
pub struct PhaseEngineSystem;
impl CompilerSystem for PhaseEngineSystem {
    fn name(&self) -> &str {
        "PhaseEngineSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::Execution
    }
    fn run(&self, world: &mut World) -> anyhow::Result<()> {
        let entities: Vec<Entity> = world.entities_of_kind(EntityKind::Tensor);

        for entity in &entities {
            let Some(phase) = world.get_component_mut::<PhaseLifecycleComponent>(*entity) else {
                continue;
            };

            phase.phase = match phase.phase {
                PhaseState::Dormant => PhaseState::Ready,
                PhaseState::Ready => PhaseState::ResidencyPending,
                PhaseState::ResidencyPending => PhaseState::LeasePending,
                PhaseState::LeasePending => PhaseState::Admitted,
                PhaseState::Admitted => PhaseState::Dispatched,
                PhaseState::Dispatched => PhaseState::AwaitingCompletion,
                PhaseState::AwaitingCompletion => PhaseState::Validating,
                PhaseState::Validating => PhaseState::Publishing,
                PhaseState::Publishing => PhaseState::Complete,
                other => other,
            };
        }

        Ok(())
    }
}
