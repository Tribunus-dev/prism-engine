//! Canonical agent projection — the read model of agent state for the runtime.
//!
//! Authority: this module owns the canonical read-side projection over
//! agent entities. `AgentSnapshot` is the projected shape surfaced through
//! `KernelHandle::query_agents`; it is rebuilt from the world on every call
//! and never persisted. There is no engine counterpart: agent state lives
//! only in the constitutional world.
//!
//! Classification: canonical (no hardware, no `unsafe`, no process-local
//! state, no FFI). The engine's `compute-core/src/ecs/core/executor.rs`
//! and `executor_projection.rs` are execution-boundary math code (MLX
//! arrays, hardware calls) and are not absorbed here.

use crate::ports::RuntimeError;
use prism_ecs_constitutional::agent_exec::{AgentLifecycle, AgentPhase};
use prism_ecs_constitutional::agent_plan::ParentAgentId;
use prism_ecs_core::World;

/// Read-side projection of an agent entity.
///
/// `phase` and `lifecycle` are rendered as `Debug` strings because the
/// underlying enums are not `Serialize` and the snapshot is the user-facing
/// shape. `parent_id` is `None` when the agent has no `ParentAgentId`
/// component.
#[derive(Debug, Clone, serde::Serialize)]
pub struct AgentSnapshot {
    pub entity_id: u64,
    pub phase: String,
    pub lifecycle: String,
    pub parent_id: Option<u64>,
}

/// Project every agent entity into a snapshot.
///
/// Iterates `world.all_entities()` and selects those that carry an
/// `AgentPhase` component (the canonical "this is an agent" mark). The
/// resulting `Vec` is rebuilt on every call — there is no caching layer
/// that can drift from canonical state.
///
/// Returns an empty `Vec` when no agents have been spawned.
pub fn query_agents(world: &World) -> Result<Vec<AgentSnapshot>, RuntimeError> {
    let mut agents = Vec::new();
    for entity in world.all_entities() {
        if let Some(phase) = world.get_component::<AgentPhase>(entity) {
            let lifecycle = world.get_component::<AgentLifecycle>(entity);
            let parent = world.get_component::<ParentAgentId>(entity);
            agents.push(AgentSnapshot {
                entity_id: entity.id(),
                phase: format!("{phase:?}"),
                lifecycle: format!(
                    "{:?}",
                    lifecycle.unwrap_or(&AgentLifecycle::Active)
                ),
                parent_id: parent.map(|p| p.0.id()),
            });
        }
    }
    Ok(agents)
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ecs_constitutional::agent_exec::AgentLifecycle;
    use prism_ecs_core::{EntityKind, World};

    /// An empty world yields an empty agent list — no agents, no projection.
    #[test]
    fn empty_world_yields_no_agents() {
        let world = World::new();
        let agents = query_agents(&world).expect("query");
        assert!(agents.is_empty());
    }

    /// Spawning an agent makes it visible through the projection.
    #[test]
    fn spawned_agent_is_projected_with_planning_phase() {
        let mut world = World::new();
        let spawned = world.spawn(EntityKind::Agent, None).expect("spawn");
        let entity = spawned.entity;
        world
            .add_component(entity, AgentPhase::Planning)
            .expect("attach phase");

        let agents = query_agents(&world).expect("query");
        assert_eq!(agents.len(), 1);
        assert_eq!(agents[0].entity_id, entity.id());
        assert_eq!(agents[0].phase, "Planning");
        // Default lifecycle is Active when no component is present.
        assert_eq!(agents[0].lifecycle, "Active");
        assert_eq!(agents[0].parent_id, None);
    }

    /// An agent with a `ParentAgentId` exposes the parent's numeric id.
    #[test]
    fn parent_id_round_trips_through_projection() {
        use prism_ecs_constitutional::agent_plan::ParentAgentId;
        use prism_ecs_core::Entity;

        let mut world = World::new();
        let child = world.spawn(EntityKind::Agent, None).expect("spawn child");
        let parent = Entity::new(42, 0);
        world
            .add_component(child.entity, AgentPhase::Planning)
            .expect("phase");
        world
            .add_component(child.entity, ParentAgentId(parent))
            .expect("parent");

        let agents = query_agents(&world).expect("query");
        assert_eq!(agents.len(), 1);
        assert_eq!(agents[0].parent_id, Some(42));
    }

    /// An agent with an explicit `AgentLifecycle` is projected as that value.
    #[test]
    fn explicit_lifecycle_overrides_default() {
        let mut world = World::new();
        let entity = world.spawn(EntityKind::Agent, None).expect("spawn").entity;
        world
            .add_component(entity, AgentPhase::Planning)
            .expect("phase");
        world
            .add_component(entity, AgentLifecycle::Completed)
            .expect("lifecycle");

        let agents = query_agents(&world).expect("query");
        assert_eq!(agents[0].lifecycle, "Completed");
    }
}
