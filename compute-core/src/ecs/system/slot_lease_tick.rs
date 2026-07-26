use crate::ecs::Entity;
use crate::ecs::runtime::constitutional_world_txn::ConstitutionalWorldTxn;
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

        // Stage every per-entity `SlotLeaseComponent` mutation on a
        // single `ConstitutionalWorldTxn` and commit atomically. Direct
        // `world.get_component_mut` calls outside the WorldTxn seam
        // are forbidden. Extract-mutate-insert pattern.
        let mut txn = ConstitutionalWorldTxn::new();
        for entity in &entities {
            let Some(lease) = world.get_component::<SlotLeaseComponent>(*entity).cloned()
            else {
                continue;
            };

            let mut updated = lease;
            updated.state = match updated.state {
                SlotState::Free => SlotState::WriteActive,
                SlotState::WriteActive => SlotState::OutputReady,
                SlotState::OutputReady => SlotState::Consumed,
                other => other,
            };
            if let Err(e) = txn.stage_insert(*entity, updated) {
                tracing::warn!(entity = ?entity, error = %e, "slot_lease_tick: stage_insert SlotLeaseComponent");
            }
        }
        let _ = txn.commit(world).map_err(|e| {
            tracing::error!(error = %e, "slot_lease_tick: ConstitutionalWorldTxn commit failed");
            anyhow::anyhow!("slot_lease_tick: ConstitutionalWorldTxn commit failed: {e}")
        })?;

        Ok(())
    }
}
