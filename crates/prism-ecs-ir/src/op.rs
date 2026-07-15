//! Operation framework for the ECS-native IR.
//!
//! Every operation is an Entity in the World with:
//! - An `OpMarker` component (marks the entity as an operation)
//! - An `OpName` component (string name, e.g. "arith.addf")
//! - An `Operands` component (references to Value entities)
//! - A `Results` component (Value entities this op produces)
//! - An `OpAttributes` component (attribute list)
//! - Optionally a `Region` component for region-bearing ops
//! - Optionally a `Successors` component for terminator ops
//! - Optionally a `RegionRef` component for region-bearing ops
//!
//! The `OpaqueOp` trait allows dyn-safe trait-object access to typed ops.

use prism_ecs_core::{Component, Entity};
use serde::{Deserialize, Serialize};

use crate::ir_attrs::Attribute;
use crate::ir_types::Type;

// ── Core components ─────────────────────────────────────────────────────────

/// Marker component identifying an entity as an operation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct OpMarker;
impl Component for OpMarker {}

/// Operation name (e.g. "arith.addf", "func.return").
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpName(pub String);
impl Component for OpName {}

/// Operand references — Value entities consumed by this operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Operands(pub Vec<Entity>);
impl Component for Operands {}

/// Result Value entities produced by this operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Results(pub Vec<Entity>);
impl Component for Results {}

/// Attributes attached to this operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpAttributes(pub Vec<Attribute>);
impl Component for OpAttributes {}

/// Successor blocks for terminator operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Successors(pub Vec<Entity>);
impl Component for Successors {}

/// Regions owned by this operation (for region-bearing ops like func, scf.for).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionRef(pub Vec<Entity>);
impl Component for RegionRef {}

// ── OpaqueOp trait ──────────────────────────────────────────────────────────

/// Dyn-safe trait for all operations. Enables heterogeneous operation
/// traversal without knowing the concrete op type.
pub trait OpaqueOp: Component + std::fmt::Debug {
    fn op_name(&self) -> &'static str;
    fn verify(&self, _context: &OpVerifierContext) -> Result<(), Vec<String>> {
        Ok(())
    }
    fn infer_result_types(
        &self,
        _operand_types: &[Type],
        _attributes: &[Attribute],
    ) -> Option<Vec<Type>> {
        None
    }
}

/// Context passed to op verify() methods.
#[derive(Debug, Default)]
pub struct OpVerifierContext {
    pub operand_types: Vec<Type>,
    pub result_types: Vec<Type>,
    pub attributes: Vec<Attribute>,
}

// ── OpInfo registry ─────────────────────────────────────────────────────────

#[derive(Default)]
pub struct OpRegistry {
    ops: std::collections::HashMap<&'static str, OpInfo>,
}

pub struct OpInfo {
    pub name: &'static str,
    pub description: &'static str,
    pub verify_fn: Option<fn(&OpVerifierContext) -> Result<(), Vec<String>>>,
    pub infer_fn: Option<fn(&[Type], &[Attribute]) -> Option<Vec<Type>>>,
}

impl OpRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, info: OpInfo) {
        self.ops.insert(info.name, info);
    }

    pub fn get(&self, name: &str) -> Option<&OpInfo> {
        self.ops.get(name)
    }

    pub fn verify(&self, name: &str, context: &OpVerifierContext) -> Result<(), Vec<String>> {
        match self.ops.get(name) {
            Some(info) => {
                if let Some(f) = info.verify_fn {
                    f(context)
                } else {
                    Ok(())
                }
            }
            None => Err(vec![format!("unknown operation: {}", name)]),
        }
    }

    pub fn infer_result_types(
        &self,
        name: &str,
        operand_types: &[Type],
        attributes: &[Attribute],
    ) -> Option<Vec<Type>> {
        self.ops
            .get(name)
            .and_then(|info| info.infer_fn)
            .and_then(|f| f(operand_types, attributes))
    }
}

// ── Helper functions ────────────────────────────────────────────────────────

pub fn is_op(world: &prism_ecs_core::World, entity: Entity) -> bool {
    world.get_component::<OpMarker>(entity).is_some()
}

pub fn op_name(world: &prism_ecs_core::World, entity: Entity) -> Option<String> {
    world.get_component::<OpName>(entity).map(|n| n.0.clone())
}

pub fn operands(world: &prism_ecs_core::World, entity: Entity) -> Vec<Entity> {
    world
        .get_component::<Operands>(entity)
        .map(|o| o.0.clone())
        .unwrap_or_default()
}

pub fn results(world: &prism_ecs_core::World, entity: Entity) -> Vec<Entity> {
    world
        .get_component::<Results>(entity)
        .map(|r| r.0.clone())
        .unwrap_or_default()
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ecs_core::{EntityKind, World};

    #[test]
    fn create_op_entity() {
        let mut world = World::new();
        let entity: Entity = world
            .spawn(EntityKind::Node, Some("test_addf".into()))
            .expect("spawn failed")
            .into();
        world.add_component(entity, OpMarker).expect("add OpMarker");
        world
            .add_component(entity, OpName("arith.addf".into()))
            .expect("add OpName");
        world
            .add_component(entity, Operands(vec![]))
            .expect("add Operands");
        world
            .add_component(entity, Results(vec![]))
            .expect("add Results");
        assert!(is_op(&world, entity));
        assert_eq!(op_name(&world, entity), Some("arith.addf".into()));
    }

    #[test]
    fn op_registry_verify_ok() {
        let mut registry = OpRegistry::new();
        registry.register(OpInfo {
            name: "arith.addf",
            description: "Floating-point addition",
            verify_fn: Some(|ctx| {
                if ctx.operand_types.len() != 2 {
                    return Err(vec!["arith.addf requires exactly 2 operands".into()]);
                }
                Ok(())
            }),
            infer_fn: None,
        });

        let ctx = OpVerifierContext {
            operand_types: vec![Type::f32(), Type::f32()],
            result_types: vec![Type::f32()],
            attributes: vec![],
        };
        assert!(registry.verify("arith.addf", &ctx).is_ok());

        let bad_ctx = OpVerifierContext {
            operand_types: vec![Type::f32(), Type::f32(), Type::f32()],
            result_types: vec![Type::f32()],
            attributes: vec![],
        };
        assert!(registry.verify("arith.addf", &bad_ctx).is_err());
    }
}
