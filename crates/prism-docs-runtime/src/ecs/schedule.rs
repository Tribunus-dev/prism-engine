//! Schedule — ordered system execution.
//!
//! A schedule is a list of named system ids. The schedule runs them
//! in order on the SSG path. The hydration path uses a separate
//! schedule that excludes static-only systems.

use prism_ecs_core::World;

use crate::ecs::world_bootstrap::seal_for_runtime;
use crate::error::RuntimeError;
use crate::systems::{
    chapter_presentation_system, claim_validation_system, nav_projection_system,
    render_coordinator_system,
};

/// Identifiers for systems in the docs site schedule. Adding a
/// system is a constitutional change — the new id must be added to
/// the right schedule below.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SystemId {
    ChapterPresentation,
    ClaimValidation,
    NavProjection,
    RenderCoordinator,
}

impl SystemId {
    pub fn as_str(self) -> &'static str {
        match self {
            SystemId::ChapterPresentation => "chapter_presentation",
            SystemId::ClaimValidation => "claim_validation",
            SystemId::NavProjection => "nav_projection",
            SystemId::RenderCoordinator => "render_coordinator",
        }
    }
}

/// The static schedule. Runs at SSG time. No visitor state, no DOM.
pub const STATIC_SCHEDULE: &[SystemId] = &[
    SystemId::ChapterPresentation,
    SystemId::ClaimValidation,
    SystemId::NavProjection,
    SystemId::RenderCoordinator,
];

/// Run the static schedule on the world. The world must be in
/// `Bootstrap` policy on entry; this function seals it for runtime
/// at the start of the schedule, so any further direct mutation
/// fails.
pub fn run_static(world: &mut World) -> Result<(), RuntimeError> {
    seal_for_runtime(world);
    for sys in STATIC_SCHEDULE {
        run_system(*sys, world)?;
    }
    Ok(())
}

fn run_system(id: SystemId, world: &mut World) -> Result<(), RuntimeError> {
    match id {
        SystemId::ChapterPresentation => chapter_presentation_system::run(world),
        SystemId::ClaimValidation => claim_validation_system::run(world),
        SystemId::NavProjection => nav_projection_system::run(world),
        SystemId::RenderCoordinator => render_coordinator_system::run(world),
    }
}
