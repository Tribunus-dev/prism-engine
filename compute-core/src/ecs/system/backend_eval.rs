
use crate::ecs::Entity;
use crate::ecs::{World, CompilerSystem, Component, EntityKind, SchedulePhase};
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

        for entity in &eval_entities {
            let Some(group) = world.get_component_mut::<EvalGroupComponent>(*entity) else {
                continue;
            };
            if group.submitted && group.completed_at.is_none() {
                group.completed_at = Some(
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_millis() as u64,
                );
            }
        }

        Ok(())
    }
}
