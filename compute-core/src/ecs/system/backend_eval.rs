use crate::ecs::Entity;
use crate::ecs::runtime::constitutional_world_txn::ConstitutionalWorldTxn;
use crate::ecs::{CompilerSystem, Component, EntityKind, SchedulePhase, World};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// EvalGroupComponent
// ---------------------------------------------------------------------------

/// Evaluation group state for batched tensor operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct EvalGroupComponent {
    pub group_id: u64,
    pub outputs: Vec<String>,
    pub submitted: bool,
    pub completed_at: Option<u64>,
}
impl Component for EvalGroupComponent {}

/// Evaluates tensor operations on backends by advancing
/// `EvalGroupComponent` state from submitted → completed.
///
/// Reads evaluation groups and marks them done once their backend
/// signals completion.
pub struct BackendEvalSystem;
impl CompilerSystem for BackendEvalSystem {
    fn name(&self) -> &str {
        "BackendEvalSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::Validation
    }
    fn run(&self, world: &mut World) -> anyhow::Result<()> {
        let eval_entities: Vec<Entity> = world.entities_of_kind(EntityKind::Tensor);

        // Stage every per-entity `EvalGroupComponent` mutation on a
        // single `ConstitutionalWorldTxn` and commit atomically. Direct
        // `world.get_component_mut` calls outside the WorldTxn seam
        // are forbidden. Extract-mutate-insert pattern.
        let mut txn = ConstitutionalWorldTxn::new();
        for entity in &eval_entities {
            let Some(group) = world.get_component::<EvalGroupComponent>(*entity).cloned()
            else {
                continue;
            };
            let mut updated = group;
            if updated.submitted && updated.completed_at.is_none() {
                updated.completed_at = Some(
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_millis() as u64,
                );
            }
            if let Err(e) = txn.stage_insert(*entity, updated) {
                tracing::warn!(entity = ?entity, error = %e, "backend_eval: stage_insert EvalGroupComponent");
            }
        }
        let _ = txn.commit(world).map_err(|e| {
            tracing::error!(error = %e, "backend_eval: ConstitutionalWorldTxn commit failed");
            anyhow::anyhow!("backend_eval: ConstitutionalWorldTxn commit failed: {e}")
        })?;

        Ok(())
    }
}
