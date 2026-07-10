use crate::ecs::component::scheduling::{PhaseDagState, ReadyQueueState};
use crate::ecs::{CompEntity, CompWorld, CompilerSystem, EntityKind, SchedulePhase};

/// Initializes the phase DAG — spawns a backend entity with
/// `PhaseDagState` and `ReadyQueueState` components.
///
/// Runs once during `SchedulePhase::ModelLoading`.
pub struct PhaseEngineInitSystem;
impl CompilerSystem for PhaseEngineInitSystem {
    fn name(&self) -> &str {
        "PhaseEngineInitSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::ModelLoading
    }
    fn run(&self, world: &mut CompWorld) -> anyhow::Result<()> {
        // Check if phase DAG already exists.
        let existing: Vec<CompEntity> = world.entities_of_kind(EntityKind::Executable);
        for entity in &existing {
            if world.get_component::<PhaseDagState>(*entity).is_some() {
                return Ok(());
            }
        }

        // Spawn a new engine entity with phase DAG and ready queue state.
        let entity = world.spawn(EntityKind::Executable, Some("phase-engine".into()));
        world.add_component(
            entity,
            PhaseDagState {
                phase_names: vec![
                    "model_load".into(),
                    "quantize".into(),
                    "memory_plan".into(),
                    "fusion_dispatch".into(),
                    "kernel_gen".into(),
                    "compile".into(),
                    "package".into(),
                    "validate".into(),
                    "execute".into(),
                ],
                edges: vec![
                    ("model_load".into(), "quantize".into()),
                    ("quantize".into(), "memory_plan".into()),
                    ("memory_plan".into(), "fusion_dispatch".into()),
                    ("fusion_dispatch".into(), "kernel_gen".into()),
                    ("kernel_gen".into(), "compile".into()),
                    ("compile".into(), "package".into()),
                    ("package".into(), "validate".into()),
                    ("validate".into(), "execute".into()),
                ],
                current_phase: "model_load".into(),
            },
        );
        world.add_component(
            entity,
            ReadyQueueState {
                pending_items: Vec::new(),
            },
        );

        Ok(())
    }
}
