use crate::ecs::component::scheduling::{PhaseDagState, ReadyQueueState};
use crate::ecs::runtime::constitutional_world_txn::ConstitutionalWorldTxn;

use crate::ecs::Entity;
use crate::ecs::{CompilerSystem, EntityKind, SchedulePhase, World};

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
    fn run(&self, world: &mut World) -> anyhow::Result<()> {
        // Check if phase DAG already exists.
        let existing: Vec<Entity> = world.entities_of_kind(EntityKind::Executable);
        for entity in &existing {
            if world.get_component::<PhaseDagState>(*entity).is_some() {
                return Ok(());
            }
        }

        // Spawn a new engine entity with phase DAG and ready queue state.
        //
        // Stage the spawn + both inserts on a single
        // `ConstitutionalWorldTxn` and commit atomically. Direct
        // `world.spawn` / `world.add_component` calls outside the
        // WorldTxn seam are forbidden.
        let mut txn = ConstitutionalWorldTxn::new();
        let token = txn.stage_spawn(EntityKind::Executable, Some("phase_engine".into()));
        if let Err(e) = txn.stage_insert_on(
            token,
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
        ) {
            tracing::warn!(error = %e, "phase_engine_init: stage_insert_on PhaseDagState");
        }
        if let Err(e) = txn.stage_insert_on(
            token,
            ReadyQueueState {
                pending_items: Vec::new(),
            },
        ) {
            tracing::warn!(error = %e, "phase_engine_init: stage_insert_on ReadyQueueState");
        }
        let _ = txn.commit(world).map_err(|e| {
            tracing::error!(error = %e, "phase_engine_init: ConstitutionalWorldTxn commit failed");
            anyhow::anyhow!("phase_engine_init: ConstitutionalWorldTxn commit failed: {e}")
        })?;

        Ok(())
    }
}
