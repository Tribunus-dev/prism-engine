//! OpBuilder — entity-scoped operation construction in the ECS World.
//!
//! Provides a builder interface for creating ops, blocks, and regions
//! within a World transaction. Maintains an insertion point (current block)
//! for sequential op creation.

use prism_ecs_core::{Entity, EntityKind, World, WorldError};

use crate::block::{BlockArguments, BlockMarker, BlockOps};
use crate::ir_attrs::Attribute;
use crate::ir_types::Type;
use crate::op::{OpAttributes, OpMarker, OpName, Operands, Results};
use crate::region::{RegionBlocks, RegionKind, RegionKindComp, RegionMarker};
use crate::value::{Uses, ValueDef, ValueType};

/// Builder for constructing IR entities in a World.
pub struct OpBuilder<'w> {
    world: &'w mut World,
    insertion_block: Option<Entity>,
}

impl<'w> OpBuilder<'w> {
    pub fn new(world: &'w mut World) -> Self {
        Self {
            world,
            insertion_block: None,
        }
    }

    pub fn set_insertion_point(&mut self, block: Entity) {
        self.insertion_block = Some(block);
    }

    /// Create an operation with name, operands, attributes, and result types.
    pub fn create_op(
        &mut self,
        name: &str,
        operands: &[Entity],
        attributes: &[Attribute],
        result_types: &[Type],
    ) -> Result<Entity, WorldError> {
        let entity: Entity = self
            .world
            .spawn(EntityKind::Node, Some(format!("op_{}", name)))?
            .into();
        self.world.add_component(entity, OpMarker)?;
        self.world.add_component(entity, OpName(name.to_string()))?;
        self.world
            .add_component(entity, Operands(operands.to_vec()))?;
        self.world
            .add_component(entity, OpAttributes(attributes.to_vec()))?;

        let mut result_entities = Vec::new();
        for (i, ty) in result_types.iter().enumerate() {
            let val: Entity = self
                .world
                .spawn(EntityKind::Node, Some(format!("{}.r{}", name, i)))?
                .into();
            self.world
                .add_component(val, ValueDef::op_result(entity, i as u32))?;
            self.world.add_component(val, ValueType(ty.clone()))?;
            self.world.add_component(val, Uses(vec![]))?;
            result_entities.push(val);
        }
        self.world.add_component(entity, Results(result_entities))?;

        for &op in operands {
            if let Some(u) = self.world.get_component_mut::<Uses>(op) {
                u.0.push(entity);
            }
        }

        if let Some(block) = self.insertion_block {
            if let Some(bops) = self.world.get_component_mut::<BlockOps>(block) {
                bops.0.push(entity);
            }
        }
        Ok(entity)
    }

    /// Create a block. Returns (block_entity, argument_value_entities).
    pub fn create_block(
        &mut self,
        arg_types: &[Type],
    ) -> Result<(Entity, Vec<Entity>), WorldError> {
        let entity: Entity = self
            .world
            .spawn(EntityKind::Node, Some("block".into()))?
            .into();
        self.world.add_component(entity, BlockMarker)?;

        let mut arg_values = Vec::new();
        for (i, ty) in arg_types.iter().enumerate() {
            let val: Entity = self
                .world
                .spawn(EntityKind::Node, Some(format!("arg_{}", i)))?
                .into();
            self.world
                .add_component(val, ValueDef::block_argument(entity, i as u32))?;
            self.world.add_component(val, ValueType(ty.clone()))?;
            self.world.add_component(val, Uses(vec![]))?;
            arg_values.push(val);
        }
        self.world
            .add_component(entity, BlockArguments(arg_values.clone()))?;
        self.world.add_component(entity, BlockOps(vec![]))?;
        Ok((entity, arg_values))
    }

    /// Create a region.
    pub fn create_region(&mut self, kind: RegionKind) -> Result<Entity, WorldError> {
        let entity: Entity = self
            .world
            .spawn(EntityKind::Node, Some(format!("{:?}_region", kind)))?
            .into();
        self.world.add_component(entity, RegionMarker)?;
        self.world.add_component(entity, RegionKindComp(kind))?;
        self.world.add_component(entity, RegionBlocks(vec![]))?;
        Ok(entity)
    }

    /// Add a block to a region.
    pub fn add_block_to_region(&mut self, region: Entity, block: Entity) -> Result<(), WorldError> {
        if let Some(blocks) = self.world.get_component_mut::<RegionBlocks>(region) {
            blocks.0.push(block);
        }
        Ok(())
    }

    pub fn world(&mut self) -> &mut World {
        self.world
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::{op_name, operands, results};
    use crate::value::value_users;

    #[test]
    fn create_op_with_operands() {
        let mut world = World::new();
        let mut builder = OpBuilder::new(&mut world);

        // Create a producer op that yields two results
        let producer = builder
            .create_op("test.produce", &[], &[], &[Type::f32(), Type::f32()])
            .unwrap();
        drop(builder);

        let vals = results(&world, producer);
        assert_eq!(vals.len(), 2);
        let v1 = vals[0];
        let v2 = vals[1];

        // Create a consumer
        let mut builder = OpBuilder::new(&mut world);
        let consumer = builder
            .create_op("arith.addf", &[v1, v2], &[], &[Type::f32()])
            .unwrap();
        drop(builder);

        assert_eq!(op_name(&world, consumer), Some("arith.addf".into()));
        assert_eq!(operands(&world, consumer), vec![v1, v2]);
        assert_eq!(value_users(&world, v1), vec![consumer]);
        assert_eq!(value_users(&world, v2), vec![consumer]);
    }

    #[test]
    fn create_block_with_args() {
        let mut world = World::new();
        let mut builder = OpBuilder::new(&mut world);
        let (block, args) = builder.create_block(&[Type::i32(), Type::f32()]).unwrap();
        drop(builder);

        assert!(crate::block::is_block(&world, block));
        assert_eq!(args.len(), 2);
        assert_eq!(crate::value::value_type(&world, args[0]), Some(Type::i32()));
        assert_eq!(crate::value::value_type(&world, args[1]), Some(Type::f32()));
    }

    #[test]
    fn create_region_with_blocks() {
        let mut world = World::new();
        let mut builder = OpBuilder::new(&mut world);
        let region = builder.create_region(RegionKind::SSACFG).unwrap();
        let (block, _) = builder.create_block(&[]).unwrap();
        builder.add_block_to_region(region, block).unwrap();
        drop(builder);

        assert!(crate::region::is_region(&world, region));
        let blocks = crate::region::region_blocks(&world, region);
        assert_eq!(blocks, vec![block]);
    }
}
