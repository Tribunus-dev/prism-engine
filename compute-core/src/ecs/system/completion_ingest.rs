use crate::ecs::component::scheduling::{WorkRegistryComponent, WorkState};
#[allow(unused_imports)]
use crate::ecs::Entity;
use crate::ecs::{CompWorld, CompilerSystem, Component, EntityKind, SchedulePhase};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// ExecutionReceiptComponent
// ---------------------------------------------------------------------------

/// An execution receipt produced by a backend after completing work.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ExecutionReceiptComponent {
    pub receipt_data: Vec<u8>,
    pub signature: Option<Vec<u8>>,
}
impl Component for ExecutionReceiptComponent {}

/// Ingests completion receipts — matches receipts against the
/// work registry and advances work items to `Complete`.
///
/// Runs every tick of the `Execution` phase.
pub struct CompletionIngestSystem;
impl CompilerSystem for CompletionIngestSystem {
    fn name(&self) -> &str {
        "CompletionIngestSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::Execution
    }
    fn run(&self, world: &mut CompWorld) -> anyhow::Result<()> {
        let entities: Vec<Entity> = world.entities_of_kind(EntityKind::Tensor);

        for entity in &entities {
            let receipt = world.get_component::<ExecutionReceiptComponent>(*entity);
            match receipt {
                None => continue,
                Some(r) if r.receipt_data.is_empty() => continue,
                _ => {}
            }

            let Some(work) = world.get_component_mut::<WorkRegistryComponent>(*entity) else {
                continue;
            };
            if work.state == WorkState::Running {
                work.state = WorkState::Complete;
            }
        }

        Ok(())
    }
}
