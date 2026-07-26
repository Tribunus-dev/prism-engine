pub mod archive;
pub mod backend_compile;
pub mod backend_dispatch;
pub mod backend_eval;
pub mod backend_residency;
pub mod backpressure_tick;
pub mod capability_registry_sys;
pub mod catalog_validation;
pub mod compiler_systems;
pub mod completion_ingest;
pub mod download;
pub mod draft_model;
pub mod execution_graph;
pub mod fusion;
pub mod int4_pack;
pub mod kernel_catalog;
pub mod memory_plan;
pub mod moe_budget;
pub mod package;
pub mod phase_engine;
pub mod portfolio;
pub mod profile;
pub mod quant_plan;
pub mod slot_lease_tick;
pub mod source_load;
pub mod ternary_pipeline;
pub mod token_budget_tick;
pub mod tts;
pub mod validation;
pub mod validation_matrix;
pub mod variant_gen;
pub mod variant_select;
pub mod work_dispatch;

// ── Runtime backend & scheduling systems ────────────────────────
pub mod executor_systems;
pub mod metal_cleanup;
pub mod metal_dispatch;
pub mod metal_init;
pub mod metal_transfer;
pub mod phase_engine_cleanup;
pub mod phase_engine_init;
pub mod phase_engine_tick;
pub mod session_cleanup;
pub mod session_decode_tick;
pub mod session_init;
pub mod work_dispatch_tick;

use crate::ecs::{SchedulePhase, World, WorldSystemsExt};

/// Run all systems in the given phase.
pub fn run_phase(world: &mut World, phase: SchedulePhase) -> anyhow::Result<()> {
    world.run_phase(phase)
}
