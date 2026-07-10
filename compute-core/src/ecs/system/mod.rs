pub mod backend_compile;
pub mod buffer_lifetime;
pub mod fusion;
pub mod kernel_gen;
pub mod memory_plan;
pub mod model_load;
pub mod moe_budget;
pub mod package;
pub mod quant_plan;
pub mod tuning;
pub mod validation;

use crate::ecs::{CompWorld, SchedulePhase};

/// Run all systems in the given phase.
pub fn run_phase(world: &mut CompWorld, phase: SchedulePhase) -> anyhow::Result<()> {
    world.run_phase(phase)
}
