//! Canonical lifecycle marker components for the kernel's 8-stage schedule.
//!
//! Authority: this module owns the canonical marker components attached to
//! work entities as they move through the Plan → Admit → Publish pipeline.
//! Each marker is a zero-sized `Component` whose presence is itself the
//! recorded fact — no payload, no behavior, no execution state.

use prism_ecs_core::Component;

/// Marker component attached to a work entity when its plan has been
/// recorded (`MarkObservedCommand` / `RecordWorkPlanCommand`).
///
/// Attachment is the recorded fact; there is no other canonical store of
/// "this entity has been observed/planned."
#[derive(Debug, Clone, Copy)]
pub struct PlannedMarker;
impl Component for PlannedMarker {}

/// Marker component attached to a work entity when admission has succeeded
/// (`AdmitWorkCommand`).
///
/// Pairs with `WorkState::Ready`; the marker is the explicit observation
/// record, while the `WorkState` carries the worker's view of the same fact.
#[derive(Debug, Clone, Copy)]
pub struct AdmittedMarker;
impl Component for AdmittedMarker {}

/// Marker component attached to a work entity when its result has been
/// published (`PublishResultCommand`).
///
/// Pairs with `ResultPayload`; the marker is the explicit observation
/// record, the payload carries the published bytes.
#[derive(Debug, Clone, Copy)]
pub struct PublishedMarker;
impl Component for PublishedMarker {}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ecs_core::World;

    /// Markers are zero-sized and `Copy`; attaching them is the recorded fact.
    #[test]
    fn markers_attach_and_query_as_components() {
        let mut world = World::new();
        let spawned = world
            .spawn(prism_ecs_core::EntityKind::WorkUnit, None)
            .expect("spawn work unit");
        let entity = spawned.entity;

        // All three markers are independent components.
        world
            .add_component(entity, PlannedMarker)
            .expect("attach PlannedMarker");
        world
            .add_component(entity, AdmittedMarker)
            .expect("attach AdmittedMarker");
        world
            .add_component(entity, PublishedMarker)
            .expect("attach PublishedMarker");

        assert!(world.get_component::<PlannedMarker>(entity).is_some());
        assert!(world.get_component::<AdmittedMarker>(entity).is_some());
        assert!(world.get_component::<PublishedMarker>(entity).is_some());
    }

    /// A marker attached to one entity is not visible on another — the
    /// canonical fact is per-entity, never global.
    #[test]
    fn markers_are_per_entity() {
        let mut world = World::new();
        let a = world
            .spawn(prism_ecs_core::EntityKind::WorkUnit, None)
            .expect("spawn a")
            .entity;
        let b = world
            .spawn(prism_ecs_core::EntityKind::WorkUnit, None)
            .expect("spawn b")
            .entity;
        world.add_component(a, PlannedMarker).expect("a plan");
        assert!(world.get_component::<PlannedMarker>(a).is_some());
        assert!(world.get_component::<PlannedMarker>(b).is_none());
    }
}
