use crate::ecs::component::scheduling::SessionState;
#[allow(unused_imports)]
use crate::ecs::Entity;
use crate::ecs::{CompWorld, CompilerSystem, EntityKind, SchedulePhase};

/// Advances one decode step — increments `SessionState.decode_step`
/// on each tick of the `Execution` phase.
///
/// Runs every tick of the `Execution` phase.
pub struct SessionDecodeTickSystem;
impl CompilerSystem for SessionDecodeTickSystem {
    fn name(&self) -> &str {
        "SessionDecodeTickSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::Execution
    }
    fn run(&self, world: &mut CompWorld) -> anyhow::Result<()> {
        let entities: Vec<Entity> = world.entities_of_kind(EntityKind::CommandBuffer);

        for entity in &entities {
            let Some(session) = world.get_component_mut::<SessionState>(*entity) else {
                continue;
            };
            session.decode_step += 1;
        }

        Ok(())
    }
}
