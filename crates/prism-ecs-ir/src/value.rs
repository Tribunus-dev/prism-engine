//! SSA value representation for the ECS-native IR.
//!
//! Values are either OpResults (produced by an operation) or BlockArguments
//! (entry values of a block). Each value is an Entity in the World.
//!
//! The use-list tracks every operation that consumes this value as an operand,
//! enabling efficient SSA traversal and replacement.

use prism_ecs_core::{Component, Entity};
use serde::{Deserialize, Serialize};

use crate::ir_types::Type;

// ── ValueKind ───────────────────────────────────────────────────────────────

/// Discriminator for where a value is defined.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValueKind {
    /// Value is produced by an operation.
    OpResult,
    /// Value is a block argument (entry value of a basic block).
    BlockArgument,
}

// ── ValueDef: defines provenance and semantics of a value entity ────────────

/// Provenance: which op or block defines this value, and its index.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ValueDef {
    /// What kind of definition.
    pub kind: ValueKind,
    /// The defining entity (operation or block).
    pub defining_entity: Entity,
    /// Index among this definer's results/block_args.
    pub index: u32,
}
impl Component for ValueDef {}

impl ValueDef {
    pub fn op_result(op: Entity, index: u32) -> Self {
        Self {
            kind: ValueKind::OpResult,
            defining_entity: op,
            index,
        }
    }

    pub fn block_argument(block: Entity, index: u32) -> Self {
        Self {
            kind: ValueKind::BlockArgument,
            defining_entity: block,
            index,
        }
    }
}

/// The type of this value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValueType(pub Type);
impl Component for ValueType {}

/// Use-list: entities (operations) that consume this value as an operand.
/// Updated when operands are attached to operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Uses(pub Vec<Entity>);
impl Component for Uses {}

// ── Value helper functions ──────────────────────────────────────────────────

/// Check whether an entity is a value.
pub fn is_value(world: &prism_ecs_core::World, entity: Entity) -> bool {
    world.get_component::<ValueDef>(entity).is_some()
}

/// Get the type of a value.
pub fn value_type(world: &prism_ecs_core::World, entity: Entity) -> Option<Type> {
    world
        .get_component::<ValueType>(entity)
        .map(|vt| vt.0.clone())
}

/// Get all users of a value (ops that consume it).
pub fn value_users(world: &prism_ecs_core::World, entity: Entity) -> Vec<Entity> {
    world
        .get_component::<Uses>(entity)
        .map(|u| u.0.clone())
        .unwrap_or_default()
}

/// Count users.
pub fn value_use_count(world: &prism_ecs_core::World, entity: Entity) -> usize {
    value_users(world, entity).len()
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn create_value_entity() {
        use prism_ecs_core::{EntityKind, World};
        let mut world = World::new();
        let value: Entity = world
            .spawn(EntityKind::Node, Some("test_val".into()))
            .expect("spawn failed")
            .into();
        world
            .add_component(value, ValueDef::op_result(Entity(0, 1), 0))
            .expect("add ValueDef");
        world
            .add_component(value, ValueType(Type::f32()))
            .expect("add ValueType");
        world.add_component(value, Uses(vec![])).expect("add Uses");

        assert!(is_value(&world, value));
        assert_eq!(value_type(&world, value), Some(Type::f32()));
        assert_eq!(value_use_count(&world, value), 0);
    }

    #[test]
    fn value_provenance() {
        let def = ValueDef::op_result(Entity(42, 1), 1);
        assert_eq!(def.defining_entity, Entity(42, 1));
        assert_eq!(def.index, 1);
        assert_eq!(def.kind, ValueKind::OpResult);

        let block_def = ValueDef::block_argument(Entity(7, 1), 0);
        assert_eq!(block_def.defining_entity, Entity(7, 1));
        assert_eq!(block_def.kind, ValueKind::BlockArgument);
    }

    #[test]
    fn use_count_tracking() {
        use prism_ecs_core::{EntityKind, World};
        let mut world = World::new();
        let val: Entity = world
            .spawn(EntityKind::Node, Some("val".into()))
            .expect("spawn")
            .into();
        world
            .add_component(val, ValueDef::op_result(Entity(0, 1), 0))
            .expect("add ValueDef");
        world
            .add_component(val, ValueType(Type::f32()))
            .expect("add ValueType");
        world
            .add_component(val, Uses(vec![Entity(1, 1), Entity(2, 1)]))
            .expect("add Uses");

        assert_eq!(value_use_count(&world, val), 2);
    }
}
