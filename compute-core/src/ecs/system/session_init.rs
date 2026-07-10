use crate::ecs::component::scheduling::SessionState;
use crate::ecs::{CompEntity, CompWorld, CompilerSystem, EntityKind, SchedulePhase};

/// Initializes an inference session — spawns a session entity with
/// `SessionState` components.
///
/// Runs once during `SchedulePhase::ModelLoading`.
pub struct SessionInitSystem;
impl CompilerSystem for SessionInitSystem {
    fn name(&self) -> &str {
        "SessionInitSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::ModelLoading
    }
    fn run(&self, world: &mut CompWorld) -> anyhow::Result<()> {
        // Check if a session already exists.
        let existing: Vec<CompEntity> = world.entities_of_kind(EntityKind::CommandBuffer);
        for entity in &existing {
            if world.get_component::<SessionState>(*entity).is_some() {
                return Ok(());
            }
        }

        // Spawn a session entity.
        let entity = world.spawn(EntityKind::CommandBuffer, Some("session".into()));
        world.add_component(
            entity,
            SessionState {
                decode_step: 0,
                active_model: "".into(),
                generation_params_json: "{}".into(),
            },
        );

        Ok(())
    }
}
