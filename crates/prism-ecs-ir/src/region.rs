//! Region entity framework for the ECS-native IR.
//!
//! A Region contains a sequence of Blocks. Each Block contains operations.
//! Regions are themselves entities in the ECS World, enabling query-based
//! traversal of the IR graph.

use prism_ecs_core::{Component, Entity};
use serde::{Deserialize, Serialize};

// ── RegionKind ──────────────────────────────────────────────────────────────

/// The kind of a region (determines verification rules).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegionKind {
    /// Graph region: no block arguments, single-block.
    Graph,
    /// SSA CFG region: multiple blocks with block arguments and terminators.
    SSACFG,
}

// ── Core components ─────────────────────────────────────────────────────────

/// Marker component identifying an entity as a region.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RegionMarker;
impl Component for RegionMarker {}

/// The kind of this region.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RegionKindComp(pub RegionKind);
impl Component for RegionKindComp {}

/// Blocks contained in this region — ordered list of Block entities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionBlocks(pub Vec<Entity>);
impl Component for RegionBlocks {}

// ── Helper functions ────────────────────────────────────────────────────────

/// Check whether an entity is a region.
pub fn is_region(world: &prism_ecs_core::World, entity: Entity) -> bool {
    world.get_component::<RegionMarker>(entity).is_some()
}

/// Get the kind of a region.
pub fn region_kind(world: &prism_ecs_core::World, entity: Entity) -> Option<RegionKind> {
    world.get_component::<RegionKindComp>(entity).map(|k| k.0)
}

/// Get the blocks in a region.
pub fn region_blocks(world: &prism_ecs_core::World, entity: Entity) -> Vec<Entity> {
    world
        .get_component::<RegionBlocks>(entity)
        .map(|b| b.0.clone())
        .unwrap_or_default()
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ecs_core::{EntityKind, World};

    #[test]
    fn create_region() {
        let mut world = World::new();
        let region: Entity = world
            .spawn(EntityKind::Node, Some("test_region".into()))
            .expect("spawn failed")
            .into();

        world
            .add_component(region, RegionMarker)
            .expect("add RegionMarker");
        world
            .add_component(region, RegionKindComp(RegionKind::SSACFG))
            .expect("add RegionKindComp");
        world
            .add_component(region, RegionBlocks(vec![]))
            .expect("add RegionBlocks");

        assert!(is_region(&world, region));
        assert_eq!(region_kind(&world, region), Some(RegionKind::SSACFG));
        assert!(region_blocks(&world, region).is_empty());
    }
}
