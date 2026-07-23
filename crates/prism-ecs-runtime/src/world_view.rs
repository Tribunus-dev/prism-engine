//! Read-only world view for schedule systems.
//!
//! [`WorldViewImpl`] is the concrete type that schedule systems receive through
//! [`SystemContext`](crate::SystemContext).  It owns the world mutex guard for
//! the duration of a tick, providing safe read-only access.

use prism_ecs_core::component::Component;
use prism_ecs_core::entity::Entity;
use prism_ecs_core::query::Query;
use prism_ecs_core::World;

/// Read-only projection of the ECS world.
///
/// Wraps a standard read lock guard and exposes only query/read operations.
/// Systems receive a reference to this through `SystemContext`.
pub struct WorldViewImpl<'a> {
    guard: std::sync::RwLockReadGuard<'a, World>,
}

impl<'a> WorldViewImpl<'a> {
    /// Construct a view from an acquired world lock guard.
    pub fn new(guard: std::sync::RwLockReadGuard<'a, World>) -> Self {
        Self { guard }
    }

    /// Current world epoch — increments on every committed transaction.
    pub fn epoch(&self) -> u64 {
        self.guard.current_epoch().0
    }

    /// Check whether `entity` is currently alive.
    pub fn is_alive(&self, entity: Entity) -> bool {
        self.guard.has_entity(entity)
    }

    /// Borrow a component of type `T` from `entity`, if present.
    pub fn get<T: Component>(&self, entity: Entity) -> Option<&T> {
        self.guard.get_component(entity)
    }

    /// Iterate all entities that have component `A`.
    ///
    /// The caller **must** bound iteration — use `.take(max_results)`.
    pub fn query<A: Component>(&self) -> Query<'_, A> {
        self.guard.query()
    }

    /// Count entities that have component `A`.
    pub fn count<A: Component>(&self) -> usize {
        self.guard.query::<A>().count()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ecs_core::{EntityKind, World};

    #[derive(Debug)]
    struct Health {
        hp: u32,
    }
    impl Component for Health {}

    #[test]
    fn view_captures_epoch() {
        let world = World::new();
        let epoch_before = world.current_epoch().0;
        let mtx = std::sync::RwLock::new(world);
        let view = WorldViewImpl::new(mtx.read().unwrap());
        assert_eq!(view.epoch(), epoch_before);
    }

    #[test]
    fn view_sees_spawned_entities() {
        let mut world = World::new();
        let spawned = world.spawn(EntityKind::Node, None).expect("spawn");
        let entity = spawned.entity;
        let mtx = std::sync::RwLock::new(world);
        let view = WorldViewImpl::new(mtx.read().unwrap());
        assert!(view.is_alive(entity), "entity should be alive after spawn");
    }

    #[test]
    fn view_sees_components() {
        let mut world = World::new();
        let spawned = world.spawn(EntityKind::Node, None).expect("spawn");
        let entity = spawned.entity;
        world
            .add_component(entity, Health { hp: 42 })
            .expect("add_component");

        let mtx = std::sync::RwLock::new(world);
        let view = WorldViewImpl::new(mtx.read().unwrap());
        let hp = view.get::<Health>(entity).expect("should have health");
        assert_eq!(hp.hp, 42);
    }

    #[test]
    fn view_query_finds_matching_entities() {
        let mut world = World::new();
        let e1 = world.spawn(EntityKind::Node, None).unwrap().entity;
        let e2 = world.spawn(EntityKind::Node, None).unwrap().entity;
        world.add_component(e1, Health { hp: 10 }).unwrap();
        world.add_component(e2, Health { hp: 20 }).unwrap();

        let mtx = std::sync::RwLock::new(world);
        let view = WorldViewImpl::new(mtx.read().unwrap());
        let results: Vec<_> = view.query::<Health>().collect();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn view_count_matches() {
        let mut world = World::new();
        let e = world.spawn(EntityKind::Node, None).unwrap().entity;
        world.add_component(e, Health { hp: 99 }).unwrap();

        let mtx = std::sync::RwLock::new(world);
        let view = WorldViewImpl::new(mtx.read().unwrap());
        assert_eq!(view.count::<Health>(), 1);
    }

    #[test]
    fn dead_entity_not_alive() {
        let mut world = World::new();
        let spawned = world.spawn(EntityKind::Node, None).unwrap();
        let entity = spawned.entity;
        assert!(world.despawn(entity), "despawn should succeed");
        let mtx = std::sync::RwLock::new(world);
        let view = WorldViewImpl::new(mtx.read().unwrap());
        assert!(!view.is_alive(entity));
    }
}
