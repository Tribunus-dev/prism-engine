use crate::ecs::component::scheduling::{WorkRegistryComponent, WorkState};
use crate::ecs::runtime::constitutional_world_txn::ConstitutionalWorldTxn;

use crate::ecs::Entity;
use crate::ecs::{CompilerSystem, Component, EntityKind, SchedulePhase, World};
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
    fn run(&self, world: &mut World) -> anyhow::Result<()> {
        let entities: Vec<Entity> = world.entities_of_kind(EntityKind::Tensor);

        // Stage every per-entity `WorkRegistryComponent` mutation on a
        // single `ConstitutionalWorldTxn` and commit atomically. Direct
        // `world.get_component_mut` calls outside the WorldTxn seam
        // are forbidden. Extract-mutate-insert pattern.
        let mut txn = ConstitutionalWorldTxn::new();
        for entity in &entities {
            let receipt = world.get_component::<ExecutionReceiptComponent>(*entity);
            match receipt {
                None => continue,
                Some(r) if r.receipt_data.is_empty() => continue,
                _ => {}
            }

            let Some(work) = world.get_component::<WorkRegistryComponent>(*entity).cloned()
            else {
                continue;
            };
            if work.state == WorkState::Running {
                let mut updated = work;
                updated.state = WorkState::Complete;
                if let Err(e) = txn.stage_insert(*entity, updated) {
                    tracing::warn!(entity = ?entity, error = %e, "completion_ingest: stage_insert WorkRegistryComponent");
                }
            }
        }
        let _ = txn.commit(world).map_err(|e| {
            tracing::error!(error = %e, "completion_ingest: ConstitutionalWorldTxn commit failed");
            anyhow::anyhow!("completion_ingest: ConstitutionalWorldTxn commit failed: {e}")
        })?;

        Ok(())
    }
}
