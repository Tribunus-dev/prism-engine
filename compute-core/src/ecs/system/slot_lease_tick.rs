use crate::ecs::Entity;
use crate::ecs::{CompilerSystem, Component, EntityKind, SchedulePhase, World};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// SlotState — lifecycle state for a slot lease
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum SlotState {
    Free,
    WriteActive,
    ReadActive,
    OutputReady,
    Consumed,
    Poisoned,
}

// ---------------------------------------------------------------------------
// SlotLeaseComponent
// ---------------------------------------------------------------------------

/// Tracks a single slot lease — the reservation of an IOSurface/arena slot
/// for a lane executor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct SlotLeaseComponent {
    pub slot_id: u64,
    pub state: SlotState,
    pub lease_holder: String,
}
impl Component for SlotLeaseComponent {}

/// Ticks the slot lease state machine — advances leases through
/// their lifecycle (Free → WriteActive → OutputReady → Consumed).
pub struct SlotLeaseTickSystem;
impl CompilerSystem for SlotLeaseTickSystem {
    fn name(&self) -> &str {
        "SlotLeaseTickSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::Execution
    }
    fn run(&self, world: &mut World) -> anyhow::Result<()> {
        let entities: Vec<Entity> = world.entities_of_kind(EntityKind::Tensor);

        for entity in &entities {
            let Some(lease) = world.get_component_mut::<SlotLeaseComponent>(*entity) else {
                continue;
            };

            lease.state = match lease.state {
                SlotState::Free => SlotState::WriteActive,
                SlotState::WriteActive => SlotState::OutputReady,
                SlotState::OutputReady => SlotState::Consumed,
                other => other,
            };
        }

        Ok(())
    }
}
