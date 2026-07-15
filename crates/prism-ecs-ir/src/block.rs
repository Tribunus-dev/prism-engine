//! Block entity framework for the ECS-native IR.
//!
//! A Block is a basic block in the IR graph: a sequence of operations with
//! block arguments (entry values) and an optional terminator operation.
//! Blocks are entities in the ECS World.

use prism_ecs_core::{Component, Entity};
use serde::{Deserialize, Serialize};

// ── Core components ─────────────────────────────────────────────────────────

/// Marker component identifying an entity as a block.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BlockMarker;
impl Component for BlockMarker {}

/// Block arguments — Value entities for this block's entry values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockArguments(pub Vec<Entity>);
impl Component for BlockArguments {}

/// The terminator operation of this block (if any).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TerminatorOp(pub Option<Entity>);
impl Component for TerminatorOp {}

/// Operations contained in this block — ordered list of Op entities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockOps(pub Vec<Entity>);
impl Component for BlockOps {}

// ── Helper functions ────────────────────────────────────────────────────────

/// Check whether an entity is a block.
pub fn is_block(world: &prism_ecs_core::World, entity: Entity) -> bool {
    world.get_component::<BlockMarker>(entity).is_some()
}

/// Get the block arguments.
pub fn block_args(world: &prism_ecs_core::World, entity: Entity) -> Vec<Entity> {
    world
        .get_component::<BlockArguments>(entity)
        .map(|a| a.0.clone())
        .unwrap_or_default()
}

/// Get the operations in a block.
pub fn block_ops(world: &prism_ecs_core::World, entity: Entity) -> Vec<Entity> {
    world
        .get_component::<BlockOps>(entity)
        .map(|o| o.0.clone())
        .unwrap_or_default()
}

/// Get the terminator operation of a block.
pub fn block_terminator(world: &prism_ecs_core::World, entity: Entity) -> Option<Entity> {
    world
        .get_component::<TerminatorOp>(entity)
        .and_then(|t| t.0)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ecs_core::{EntityKind, World};

    #[test]
    fn create_block() {
        let mut world = World::new();
        let block: Entity = world
            .spawn(EntityKind::Node, Some("entry_block".into()))
            .expect("spawn failed")
            .into();

        world
            .add_component(block, BlockMarker)
            .expect("add BlockMarker");
        world
            .add_component(block, BlockArguments(vec![]))
            .expect("add BlockArguments");
        world
            .add_component(block, BlockOps(vec![]))
            .expect("add BlockOps");
        world
            .add_component(block, TerminatorOp(None))
            .expect("add TerminatorOp");

        assert!(is_block(&world, block));
        assert!(block_args(&world, block).is_empty());
        assert!(block_ops(&world, block).is_empty());
        assert!(block_terminator(&world, block).is_none());
    }
}
