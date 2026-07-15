//! Func dialect — function definition, return, and call operations.
//!
//! Provides operations for defining functions with region bodies (`func.func`),
//! returning values (`func.return`), and calling functions (`func.call`).
//!
//! All operations define a `FuncOp` component on the entity alongside
//! the standard `OpMarker`, `OpName`, `Operands`, `Results`, and
//! `OpAttributes` components.
//!
//! # Operation semantics
//!
//! - `func.func`: Defines a function. Has exactly one SSACFG region containing
//!   blocks terminated by `func.return`. Carries a `function_type` attribute
//!   describing the function signature. Zero operands and zero results.
//!
//! - `func.return`: Terminator that returns control flow and optional values
//!   to the caller. Its operands are the return values. Zero results.
//!   Carries a `Successors` component (typically empty).
//!
//! - `func.call`: Calls a function by name. Operands are the call arguments.
//!   Exactly one result (the call's return value). Carries a `callee`
//!   attribute (the function name string).

use prism_ecs_core::Component;
use serde::{Deserialize, Serialize};

use crate::ir_attrs::Attribute;
use crate::ir_types::Type;
use crate::op::{OpInfo, OpRegistry, OpVerifierContext};

// ── Op kind ──────────────────────────────────────────────────────────────────

/// Specific func operation variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FuncOpKind {
    /// Function definition with region body.
    /// Name: "func.func"
    Func,
    /// Return from a function (terminator).
    /// Name: "func.return"
    Return,
    /// Call a function.
    /// Name: "func.call"
    Call,
}

impl FuncOpKind {
    /// Returns the MLIR-style operation name for this kind.
    pub fn op_name(&self) -> &'static str {
        match self {
            FuncOpKind::Func => "func.func",
            FuncOpKind::Return => "func.return",
            FuncOpKind::Call => "func.call",
        }
    }

    /// Returns a human-readable description for this kind.
    pub fn description(&self) -> &'static str {
        match self {
            FuncOpKind::Func => "Function definition with region body",
            FuncOpKind::Return => "Return from function (terminator)",
            FuncOpKind::Call => "Call function by name",
        }
    }
}

// ── Component ────────────────────────────────────────────────────────────────

/// Component attaching a func op kind to an operation entity.
///
/// Every entity representing a func dialect operation carries this component
/// so dialects and passes can discriminate func operations from other dialects.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FuncOp(pub FuncOpKind);
impl Component for FuncOp {}

// ── Verifiers ────────────────────────────────────────────────────────────────

/// Verify a func.func operation: 0 operands, 0 results, must have a
/// `function_type` attribute describing the function signature.
pub fn verify_func(ctx: &OpVerifierContext) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();

    if !ctx.operand_types.is_empty() {
        errors.push("func.func expects 0 operands".to_string());
    }
    if !ctx.result_types.is_empty() {
        errors.push("func.func expects 0 results".to_string());
    }
    if ctx.attributes.is_empty() {
        errors.push("func.func requires a function_type attribute".to_string());
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Verify a func.return operation: 0 results.
/// Operands (return values) can be any number of any types.
pub fn verify_return(ctx: &OpVerifierContext) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();

    if !ctx.result_types.is_empty() {
        errors.push("func.return expects 0 results".to_string());
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Verify a func.call operation: operands = call arguments, 1 result.
/// Must have a callee attribute (string).
pub fn verify_call(ctx: &OpVerifierContext) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();

    if ctx.result_types.len() != 1 {
        errors.push("func.call expects exactly 1 result".to_string());
    }
    let has_callee = ctx
        .attributes
        .iter()
        .any(|a| matches!(a, Attribute::String(_)));
    if !has_callee {
        errors.push("func.call requires a callee attribute (string)".to_string());
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

// ── Type inference ───────────────────────────────────────────────────────────

/// Infer result types for func.func: no results.
pub fn infer_func(_operand_types: &[Type], _attributes: &[Attribute]) -> Option<Vec<Type>> {
    Some(vec![])
}

/// Infer result types for func.return: no results.
pub fn infer_return(_operand_types: &[Type], _attributes: &[Attribute]) -> Option<Vec<Type>> {
    Some(vec![])
}

/// Infer result types for func.call: cannot determine from operands alone
/// without the function signature. Returns None (caller must supply types).
pub fn infer_call(_operand_types: &[Type], _attributes: &[Attribute]) -> Option<Vec<Type>> {
    None
}

// ── Registration ─────────────────────────────────────────────────────────────

/// Register all func dialect operations into the given OpRegistry.
pub fn register_func_ops(registry: &mut OpRegistry) {
    registry.register(OpInfo {
        name: "func.func",
        description: "Function definition with region body",
        verify_fn: Some(verify_func as fn(&OpVerifierContext) -> Result<(), Vec<String>>),
        infer_fn: Some(infer_func as fn(&[Type], &[Attribute]) -> Option<Vec<Type>>),
    });
    registry.register(OpInfo {
        name: "func.return",
        description: "Return from function (terminator)",
        verify_fn: Some(verify_return as fn(&OpVerifierContext) -> Result<(), Vec<String>>),
        infer_fn: Some(infer_return as fn(&[Type], &[Attribute]) -> Option<Vec<Type>>),
    });
    registry.register(OpInfo {
        name: "func.call",
        description: "Call function by name",
        verify_fn: Some(verify_call as fn(&OpVerifierContext) -> Result<(), Vec<String>>),
        infer_fn: Some(infer_call as fn(&[Type], &[Attribute]) -> Option<Vec<Type>>),
    });
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::{BlockArguments, BlockMarker, BlockOps, TerminatorOp};
    use crate::builder::OpBuilder;
    use crate::ir_types::Type;
    use crate::op::{op_name, results, OpMarker, RegionRef};
    use crate::region::{RegionBlocks, RegionKind, RegionKindComp, RegionMarker};
    use crate::serde::{from_json, to_json};
    use prism_ecs_core::{Entity, EntityKind, World};

    #[test]
    fn func_op_kind_op_name() {
        assert_eq!(FuncOpKind::Func.op_name(), "func.func");
        assert_eq!(FuncOpKind::Return.op_name(), "func.return");
        assert_eq!(FuncOpKind::Call.op_name(), "func.call");
    }

    #[test]
    fn func_op_component_attached() {
        let mut world = World::new();
        let entity: Entity = world
            .spawn(EntityKind::Node, Some("test_func".into()))
            .expect("spawn")
            .into();
        world
            .add_component(entity, FuncOp(FuncOpKind::Func))
            .expect("add FuncOp");
        let retrieved = world.get_component::<FuncOp>(entity);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().0, FuncOpKind::Func);
    }

    #[test]
    fn func_op_serialization_roundtrip() {
        let op = FuncOp(FuncOpKind::Return);
        let json = serde_json::to_string(&op).unwrap();
        let back: FuncOp = serde_json::from_str(&json).unwrap();
        assert_eq!(op.0, back.0);

        let kind = FuncOpKind::Call;
        let json = serde_json::to_string(&kind).unwrap();
        let back: FuncOpKind = serde_json::from_str(&json).unwrap();
        assert_eq!(kind, back);
    }

    // ── Verification tests ───────────────────────────────────────────────────

    #[test]
    fn verify_func_ok() {
        let ctx = OpVerifierContext {
            operand_types: vec![],
            result_types: vec![],
            attributes: vec![Attribute::String("function_type".into())],
        };
        assert!(verify_func(&ctx).is_ok());
    }

    #[test]
    fn verify_func_with_operands() {
        let ctx = OpVerifierContext {
            operand_types: vec![Type::i32()],
            result_types: vec![],
            attributes: vec![Attribute::String("function_type".into())],
        };
        assert!(verify_func(&ctx).is_err());
    }

    #[test]
    fn verify_func_with_results() {
        let ctx = OpVerifierContext {
            operand_types: vec![],
            result_types: vec![Type::f32()],
            attributes: vec![Attribute::String("function_type".into())],
        };
        assert!(verify_func(&ctx).is_err());
    }

    #[test]
    fn verify_func_no_attributes() {
        let ctx = OpVerifierContext {
            operand_types: vec![],
            result_types: vec![],
            attributes: vec![],
        };
        assert!(verify_func(&ctx).is_err());
    }

    #[test]
    fn verify_return_ok() {
        let ctx = OpVerifierContext {
            operand_types: vec![Type::f32(), Type::f32()],
            result_types: vec![],
            attributes: vec![],
        };
        assert!(verify_return(&ctx).is_ok());
    }

    #[test]
    fn verify_return_with_results() {
        let ctx = OpVerifierContext {
            operand_types: vec![],
            result_types: vec![Type::f32()],
            attributes: vec![],
        };
        assert!(verify_return(&ctx).is_err());
    }

    #[test]
    fn verify_call_ok() {
        let ctx = OpVerifierContext {
            operand_types: vec![Type::f32()],
            result_types: vec![Type::f32()],
            attributes: vec![Attribute::String("my_func".into())],
        };
        assert!(verify_call(&ctx).is_ok());
    }

    #[test]
    fn verify_call_no_results() {
        let ctx = OpVerifierContext {
            operand_types: vec![Type::f32()],
            result_types: vec![],
            attributes: vec![Attribute::String("my_func".into())],
        };
        assert!(verify_call(&ctx).is_err());
    }

    #[test]
    fn verify_call_no_callee() {
        let ctx = OpVerifierContext {
            operand_types: vec![Type::f32()],
            result_types: vec![Type::f32()],
            attributes: vec![],
        };
        assert!(verify_call(&ctx).is_err());
    }

    // ── Type inference tests ─────────────────────────────────────────────────

    #[test]
    fn infer_func_no_results() {
        let result = infer_func(&[], &[]);
        assert_eq!(result, Some(vec![]));
    }

    #[test]
    fn infer_return_no_results() {
        let result = infer_return(&[Type::f32()], &[]);
        assert_eq!(result, Some(vec![]));
    }

    #[test]
    fn infer_call_returns_none() {
        let result = infer_call(&[Type::f32()], &[]);
        assert_eq!(result, None);
    }

    // ── Registry integration tests ───────────────────────────────────────────

    #[test]
    fn register_all_func_ops() {
        let mut registry = crate::op::OpRegistry::new();
        register_func_ops(&mut registry);

        assert!(registry.get("func.func").is_some());
        assert!(registry.get("func.return").is_some());
        assert!(registry.get("func.call").is_some());

        let expected = ["func.func", "func.return", "func.call"];
        for name in &expected {
            assert!(
                registry.get(name).is_some(),
                "missing registration for {}",
                name
            );
        }
    }

    #[test]
    fn registry_verify_func_ok() {
        let mut registry = crate::op::OpRegistry::new();
        register_func_ops(&mut registry);

        let ctx = OpVerifierContext {
            operand_types: vec![],
            result_types: vec![],
            attributes: vec![Attribute::String("function_type".into())],
        };
        assert!(registry.verify("func.func", &ctx).is_ok());
    }

    #[test]
    fn registry_verify_func_bad() {
        let mut registry = crate::op::OpRegistry::new();
        register_func_ops(&mut registry);

        let ctx = OpVerifierContext {
            operand_types: vec![Type::i32()],
            result_types: vec![],
            attributes: vec![],
        };
        assert!(registry.verify("func.func", &ctx).is_err());
    }

    #[test]
    fn registry_infer_func() {
        let mut registry = crate::op::OpRegistry::new();
        register_func_ops(&mut registry);

        let result = registry.infer_result_types("func.func", &[], &[]);
        assert_eq!(result, Some(vec![]));
    }

    // ── Builder integration tests ─────────────────────────────────────────────

    /// Build a func.func with a region containing one block ending with
    /// func.return, then round-trip through serde.
    #[test]
    fn func_with_return_roundtrip() {
        let mut world = World::new();

        // ── Create func.func ──────────────────────────────────────────────
        let func_op: Entity = {
            let mut builder = OpBuilder::new(&mut world);
            let func_op = builder
                .create_op(
                    "func.func",
                    &[], // no operands
                    &[], // no attributes (serde doesn't round-trip OpName/attributes)
                    &[], // no results
                )
                .unwrap();
            drop(builder);
            func_op
        };

        // Attach FuncOp component.
        world
            .add_component(func_op, FuncOp(FuncOpKind::Func))
            .unwrap();

        // ── Create region (SSACFG) and add to func ────────────────────────
        let region: Entity = world
            .spawn(EntityKind::Node, Some("func_body".into()))
            .unwrap()
            .into();
        world.add_component(region, RegionMarker).unwrap();
        world
            .add_component(region, RegionKindComp(RegionKind::SSACFG))
            .unwrap();
        world.add_component(region, RegionBlocks(vec![])).unwrap();

        world
            .add_component(func_op, RegionRef(vec![region]))
            .unwrap();

        // ── Create entry block in the region ──────────────────────────────
        let block: Entity = world
            .spawn(EntityKind::Node, Some("entry".into()))
            .unwrap()
            .into();
        world.add_component(block, BlockMarker).unwrap();
        world.add_component(block, BlockArguments(vec![])).unwrap();
        world.add_component(block, BlockOps(vec![])).unwrap();
        world.add_component(block, TerminatorOp(None)).unwrap();

        world
            .add_component(region, RegionBlocks(vec![block]))
            .unwrap();

        // ── Create a return op (func.return) in the block ─────────────────
        let return_op: Entity = {
            let mut builder = OpBuilder::new(&mut world);
            builder.set_insertion_point(block);
            let return_op = builder
                .create_op(
                    "func.return",
                    &[], // no return values
                    &[], // no attributes
                    &[], // no results
                )
                .unwrap();
            drop(builder);
            return_op
        };

        // Attach FuncOp component and mark as terminator.
        world
            .add_component(return_op, FuncOp(FuncOpKind::Return))
            .unwrap();
        world
            .add_component(block, TerminatorOp(Some(return_op)))
            .unwrap();

        // ── Verify structure ────────────────────────────────────────────────
        assert_eq!(op_name(&world, func_op), Some("func.func".into()));
        assert_eq!(op_name(&world, return_op), Some("func.return".into()));

        assert!(world.get_component::<FuncOp>(func_op).is_some());
        assert_eq!(
            world.get_component::<FuncOp>(func_op).unwrap().0,
            FuncOpKind::Func
        );
        assert!(world.get_component::<FuncOp>(return_op).is_some());
        assert_eq!(
            world.get_component::<FuncOp>(return_op).unwrap().0,
            FuncOpKind::Return
        );

        // Verify region structure.
        let region_ref = world
            .get_component::<RegionRef>(func_op)
            .expect("func.func should have RegionRef");
        assert_eq!(region_ref.0.len(), 1);
        let region_entity = region_ref.0[0];
        assert!(world.get_component::<RegionMarker>(region_entity).is_some());
        assert_eq!(
            world
                .get_component::<RegionKindComp>(region_entity)
                .unwrap()
                .0,
            RegionKind::SSACFG
        );

        let region_blocks_comp = world
            .get_component::<RegionBlocks>(region_entity)
            .expect("region should have RegionBlocks");
        assert_eq!(region_blocks_comp.0.len(), 1);
        assert_eq!(region_blocks_comp.0[0], block);

        // Verify block has our return op as terminator.
        let term = world
            .get_component::<TerminatorOp>(block)
            .expect("block should have TerminatorOp");
        assert_eq!(term.0, Some(return_op));

        let block_ops = world
            .get_component::<BlockOps>(block)
            .expect("block should have BlockOps");
        assert_eq!(block_ops.0, vec![return_op]);

        // ── Round-trip via serde ─────────────────────────────────────────
        let json = to_json(func_op, &world).expect("to_json should succeed");

        let mut world2 = World::new();
        let restored = from_json(&json, &mut world2).expect("from_json should succeed");

        // Verify restored structure.
        assert!(world2.is_alive(restored));
        assert!(world2.get_component::<OpMarker>(restored).is_some());

        // FuncOp component is NOT preserved by serde (only core structural
        // components like OpMarker, RegionRef, RegionBlocks, BlockOps are).
        // The FuncOp dialect component must be reapplied when building.

        // Verify region was restored.
        let restored_region_ref = world2
            .get_component::<RegionRef>(restored)
            .expect("restored func.func should have RegionRef");
        assert_eq!(restored_region_ref.0.len(), 1);

        let restored_region = restored_region_ref.0[0];
        assert!(world2
            .get_component::<RegionMarker>(restored_region)
            .is_some());
        assert_eq!(
            world2
                .get_component::<RegionKindComp>(restored_region)
                .unwrap()
                .0,
            RegionKind::SSACFG
        );

        // Verify block was restored.
        let restored_region_blocks = world2
            .get_component::<RegionBlocks>(restored_region)
            .expect("restored region should have RegionBlocks");
        assert_eq!(restored_region_blocks.0.len(), 1);
        let restored_block = restored_region_blocks.0[0];
        assert!(world2
            .get_component::<BlockMarker>(restored_block)
            .is_some());

        // Verify return op survived round-trip.
        let restored_block_ops = world2
            .get_component::<BlockOps>(restored_block)
            .expect("restored block should have BlockOps");
        assert!(
            !restored_block_ops.0.is_empty(),
            "restored block should have operations"
        );

        let restored_return = restored_block_ops.0[0];
        assert!(world2.get_component::<OpMarker>(restored_return).is_some());

        // Verify return op has 0 results.
        let restored_return_results = results(&world2, restored_return);
        assert!(restored_return_results.is_empty());
    }
}
