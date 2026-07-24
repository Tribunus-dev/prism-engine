//! SCF dialect — structured control flow operations as ECS components.
//!
//! Provides the core structured control flow operations: scf.for (loop),
//! scf.if (conditional), scf.yield (terminator for loop/if bodies), and
//! scf.while (while loop with before/after regions).
//!
//! All operations define an `ScfOp` component on the entity alongside
//! the standard `OpMarker`, `OpName`, `Operands`, `Results`, and
//! `OpAttributes` components. Region-bearing ops also carry `RegionRef`.

use prism_ecs_core::Component;
use serde::{Deserialize, Serialize};

use crate::ir_attrs::Attribute;
use crate::ir_types::Type;
use crate::op::{OpInfo, OpRegistry, OpVerifierContext};

// ── Op kind ──────────────────────────────────────────────────────────────────

/// Specific SCF operation variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScfOpKind {
    /// `scf.for` — for loop with lb, ub, step, and iteration-carried values.
    For,
    /// `scf.if` — conditional with then and optional else regions.
    If,
    /// `scf.yield` — terminator yielding values from a region.
    Yield,
    /// `scf.while` — while loop with before (condition) and after (body) regions.
    While,
}

impl ScfOpKind {
    /// MLIR-style operation name for this kind.
    pub fn op_name(&self) -> &'static str {
        match self {
            ScfOpKind::For => "scf.for",
            ScfOpKind::If => "scf.if",
            ScfOpKind::Yield => "scf.yield",
            ScfOpKind::While => "scf.while",
        }
    }
}

// ── Component ────────────────────────────────────────────────────────────────

/// Component attaching an SCF op kind to an operation entity.
///
/// Every entity representing an SCF operation carries this component
/// so dialects and passes can discriminate typed SCF operations from
/// other operations or from general `OpName`-only queries.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ScfOp(pub ScfOpKind);
impl Component for ScfOp {}

// ── Verifiers ────────────────────────────────────────────────────────────────

/// Verify `scf.for`: at least 3 operands (lb, ub, step, ...iter_args).
pub fn verify_for(ctx: &OpVerifierContext) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    if ctx.operand_types.len() < 3 {
        errors.push(format!(
            "scf.for requires at least 3 operands (lb, ub, step, ...iter_args), got {}",
            ctx.operand_types.len()
        ));
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Verify `scf.if`: exactly 1 operand (condition).
pub fn verify_if(ctx: &OpVerifierContext) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    if ctx.operand_types.len() != 1 {
        errors.push(format!(
            "scf.if requires exactly 1 operand (condition), got {}",
            ctx.operand_types.len()
        ));
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Verify `scf.yield`: must not produce results.
pub fn verify_yield(ctx: &OpVerifierContext) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    if !ctx.result_types.is_empty() {
        errors.push(format!(
            "scf.yield must not produce results, got {}",
            ctx.result_types.len()
        ));
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Verify `scf.while`: 0 operands.
pub fn verify_while(ctx: &OpVerifierContext) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    if !ctx.operand_types.is_empty() {
        errors.push(format!(
            "scf.while requires 0 operands, got {}",
            ctx.operand_types.len()
        ));
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

// ── Type inference ───────────────────────────────────────────────────────────

/// Infer result types for `scf.for`: cannot infer from operands alone
/// (depends on iter_args block arguments).
pub fn infer_for(_operand_types: &[Type], _attributes: &[Attribute]) -> Option<Vec<Type>> {
    None
}

/// Infer result types for `scf.if`: cannot infer from operands alone
/// (depends on yield ops in then/else regions).
pub fn infer_if(_operand_types: &[Type], _attributes: &[Attribute]) -> Option<Vec<Type>> {
    None
}

/// Infer result types for `scf.yield`: always produces no results.
pub fn infer_yield(_operand_types: &[Type], _attributes: &[Attribute]) -> Option<Vec<Type>> {
    Some(vec![])
}

/// Infer result types for `scf.while`: cannot infer from operands alone
/// (depends on iteration-carried value types).
pub fn infer_while(_operand_types: &[Type], _attributes: &[Attribute]) -> Option<Vec<Type>> {
    None
}

// ── Registration ─────────────────────────────────────────────────────────────

/// Register all SCF dialect operations into the given OpRegistry.
pub fn register_scf_ops(registry: &mut OpRegistry) {
    // scf.for
    registry.register(OpInfo {
        name: "scf.for",
        description: "For loop with lb, ub, step, and iteration-carried values",
        verify_fn: Some(verify_for as fn(&OpVerifierContext) -> Result<(), Vec<String>>),
        infer_fn: Some(infer_for as fn(&[Type], &[Attribute]) -> Option<Vec<Type>>),
    });
    // scf.if
    registry.register(OpInfo {
        name: "scf.if",
        description: "Conditional with then and optional else regions",
        verify_fn: Some(verify_if as fn(&OpVerifierContext) -> Result<(), Vec<String>>),
        infer_fn: Some(infer_if as fn(&[Type], &[Attribute]) -> Option<Vec<Type>>),
    });
    // scf.yield
    registry.register(OpInfo {
        name: "scf.yield",
        description: "Region terminator yielding values",
        verify_fn: Some(verify_yield as fn(&OpVerifierContext) -> Result<(), Vec<String>>),
        infer_fn: Some(infer_yield as fn(&[Type], &[Attribute]) -> Option<Vec<Type>>),
    });
    // scf.while
    registry.register(OpInfo {
        name: "scf.while",
        description: "While loop with before (condition) and after (body) regions",
        verify_fn: Some(verify_while as fn(&OpVerifierContext) -> Result<(), Vec<String>>),
        infer_fn: Some(infer_while as fn(&[Type], &[Attribute]) -> Option<Vec<Type>>),
    });
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::OpBuilder;
    use crate::op::{op_name, RegionRef};
    use crate::region::RegionKind;
    use prism_ecs_core::{Entity, EntityKind, World};

    // ── Component tests ──────────────────────────────────────────────────────

    #[test]
    fn scf_op_kind_op_name() {
        assert_eq!(ScfOpKind::For.op_name(), "scf.for");
        assert_eq!(ScfOpKind::If.op_name(), "scf.if");
        assert_eq!(ScfOpKind::Yield.op_name(), "scf.yield");
        assert_eq!(ScfOpKind::While.op_name(), "scf.while");
    }

    #[test]
    fn scf_op_component_attached() {
        let mut world = World::new();
        let entity: Entity = world
            .spawn(EntityKind::Node, Some("test_scf".into()))
            .expect("spawn")
            .into();
        world
            .add_component(entity, ScfOp(ScfOpKind::For))
            .expect("add ScfOp");
        let retrieved = world.get_component::<ScfOp>(entity);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().0, ScfOpKind::For);
    }

    #[test]
    fn scf_op_serialization_roundtrip() {
        let op = ScfOp(ScfOpKind::While);
        let json = serde_json::to_string(&op).unwrap();
        let back: ScfOp = serde_json::from_str(&json).unwrap();
        assert_eq!(op.0, back.0);

        let kind = ScfOpKind::If;
        let json = serde_json::to_string(&kind).unwrap();
        let back: ScfOpKind = serde_json::from_str(&json).unwrap();
        assert_eq!(kind, back);
    }

    // ── Builder integration: create scf.for with region body ────────────────

    #[test]
    fn create_scf_for_with_region() {
        let mut world = World::new();

        // Create three index-typed operands: lb, ub, step
        let (lb, ub, step) = {
            let mut b = OpBuilder::new(&mut world);
            let op0 = b
                .create_op("test.produce", &[], &[], &[Type::Index])
                .unwrap();
            let op1 = b
                .create_op("test.produce", &[], &[], &[Type::Index])
                .unwrap();
            let op2 = b
                .create_op("test.produce", &[], &[], &[Type::Index])
                .unwrap();
            (
                crate::op::results(&world, op0)[0],
                crate::op::results(&world, op1)[0],
                crate::op::results(&world, op2)[0],
            )
        };

        // Create the body block for the loop, with block args: induction var + iter_args
        let body_block = {
            let mut b = OpBuilder::new(&mut world);
            let (block, _args) = b.create_block(&[]).expect("create body block");
            block
        };

        // Create the region containing the body block
        let body_region = {
            let mut b = OpBuilder::new(&mut world);
            let region = b.create_region(RegionKind::SSACFG).expect("create region");
            b.add_block_to_region(region, body_block)
                .expect("add block to region");
            region
        };

        // Create scf.for operation
        let scf_for = {
            let mut b = OpBuilder::new(&mut world);
            b.create_op("scf.for", &[lb, ub, step], &[], &[])
                .expect("create scf.for")
        };

        // Attach RegionRef to the scf.for entity
        world
            .add_component(scf_for, RegionRef(vec![body_region]))
            .expect("add RegionRef");

        // Attach ScfOp component
        world
            .add_component(scf_for, ScfOp(ScfOpKind::For))
            .expect("add ScfOp");

        // Verify via get_component
        assert_eq!(op_name(&world, scf_for), Some("scf.for".into()));
        let retrieved = world.get_component::<ScfOp>(scf_for);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().0, ScfOpKind::For);

        // Verify operands
        let ops = crate::op::operands(&world, scf_for);
        assert_eq!(ops, vec![lb, ub, step]);

        // Verify RegionRef attached
        let regions = world.get_component::<RegionRef>(scf_for);
        assert!(regions.is_some());
        assert_eq!(regions.unwrap().0, vec![body_region]);
    }

    // ── Verification tests ───────────────────────────────────────────────────

    #[test]
    fn verify_for_ok() {
        let ctx = OpVerifierContext {
            operand_types: vec![Type::Index, Type::Index, Type::Index],
            result_types: vec![],
            attributes: vec![],
        };
        assert!(verify_for(&ctx).is_ok());
    }

    #[test]
    fn verify_for_with_iter_args_ok() {
        let ctx = OpVerifierContext {
            operand_types: vec![Type::Index, Type::Index, Type::Index, Type::f32()],
            result_types: vec![Type::f32()],
            attributes: vec![],
        };
        assert!(verify_for(&ctx).is_ok());
    }

    #[test]
    fn verify_for_too_few_operands() {
        let ctx = OpVerifierContext {
            operand_types: vec![Type::Index, Type::Index],
            result_types: vec![],
            attributes: vec![],
        };
        assert!(verify_for(&ctx).is_err());
    }

    #[test]
    fn verify_if_ok() {
        let ctx = OpVerifierContext {
            operand_types: vec![Type::Index],
            result_types: vec![],
            attributes: vec![],
        };
        assert!(verify_if(&ctx).is_ok());
    }

    #[test]
    fn verify_if_wrong_operand_count() {
        let ctx = OpVerifierContext {
            operand_types: vec![],
            result_types: vec![],
            attributes: vec![],
        };
        assert!(verify_if(&ctx).is_err());
    }

    #[test]
    fn verify_yield_ok() {
        let ctx = OpVerifierContext {
            operand_types: vec![],
            result_types: vec![],
            attributes: vec![],
        };
        assert!(verify_yield(&ctx).is_ok());
    }

    #[test]
    fn verify_yield_with_operands_ok() {
        let ctx = OpVerifierContext {
            operand_types: vec![Type::f32()],
            result_types: vec![],
            attributes: vec![],
        };
        assert!(verify_yield(&ctx).is_ok());
    }

    #[test]
    fn verify_yield_with_results() {
        let ctx = OpVerifierContext {
            operand_types: vec![],
            result_types: vec![Type::f32()],
            attributes: vec![],
        };
        assert!(verify_yield(&ctx).is_err());
    }

    #[test]
    fn verify_while_ok() {
        let ctx = OpVerifierContext {
            operand_types: vec![],
            result_types: vec![],
            attributes: vec![],
        };
        assert!(verify_while(&ctx).is_ok());
    }

    #[test]
    fn verify_while_with_operands() {
        let ctx = OpVerifierContext {
            operand_types: vec![Type::Index],
            result_types: vec![],
            attributes: vec![],
        };
        assert!(verify_while(&ctx).is_err());
    }

    // ── Type inference tests ─────────────────────────────────────────────────

    #[test]
    fn infer_for_returns_none() {
        let result = infer_for(&[Type::Index, Type::Index, Type::Index], &[]);
        assert_eq!(result, None);
    }

    #[test]
    fn infer_if_returns_none() {
        let result = infer_if(&[Type::Index], &[]);
        assert_eq!(result, None);
    }

    #[test]
    fn infer_yield_returns_empty() {
        let result = infer_yield(&[Type::f32()], &[]);
        assert_eq!(result, Some(vec![]));
    }

    #[test]
    fn infer_while_returns_none() {
        let result = infer_while(&[], &[]);
        assert_eq!(result, None);
    }

    // ── Registry integration tests ───────────────────────────────────────────

    #[test]
    fn register_all_scf_ops() {
        let mut registry = crate::op::OpRegistry::new();
        register_scf_ops(&mut registry);

        assert!(registry.get("scf.for").is_some());
        assert!(registry.get("scf.if").is_some());
        assert!(registry.get("scf.yield").is_some());
        assert!(registry.get("scf.while").is_some());

        let expected = ["scf.for", "scf.if", "scf.yield", "scf.while"];
        for name in &expected {
            assert!(
                registry.get(name).is_some(),
                "missing registration for {}",
                name
            );
        }
    }

    #[test]
    fn registry_verify_scf_for() {
        let mut registry = crate::op::OpRegistry::new();
        register_scf_ops(&mut registry);

        let ctx = OpVerifierContext {
            operand_types: vec![Type::Index, Type::Index, Type::Index],
            result_types: vec![],
            attributes: vec![],
        };
        assert!(registry.verify("scf.for", &ctx).is_ok());

        let bad_ctx = OpVerifierContext {
            operand_types: vec![Type::Index, Type::Index],
            result_types: vec![],
            attributes: vec![],
        };
        assert!(registry.verify("scf.for", &bad_ctx).is_err());
    }

    #[test]
    fn registry_verify_scf_if() {
        let mut registry = crate::op::OpRegistry::new();
        register_scf_ops(&mut registry);

        let ctx = OpVerifierContext {
            operand_types: vec![Type::Index],
            result_types: vec![],
            attributes: vec![],
        };
        assert!(registry.verify("scf.if", &ctx).is_ok());

        let bad_ctx = OpVerifierContext {
            operand_types: vec![],
            result_types: vec![],
            attributes: vec![],
        };
        assert!(registry.verify("scf.if", &bad_ctx).is_err());
    }

    #[test]
    fn registry_verify_scf_yield() {
        let mut registry = crate::op::OpRegistry::new();
        register_scf_ops(&mut registry);

        let ctx = OpVerifierContext {
            operand_types: vec![Type::f32()],
            result_types: vec![],
            attributes: vec![],
        };
        assert!(registry.verify("scf.yield", &ctx).is_ok());

        let bad_ctx = OpVerifierContext {
            operand_types: vec![],
            result_types: vec![Type::f32()],
            attributes: vec![],
        };
        assert!(registry.verify("scf.yield", &bad_ctx).is_err());
    }

    #[test]
    fn registry_verify_scf_while() {
        let mut registry = crate::op::OpRegistry::new();
        register_scf_ops(&mut registry);

        let ctx = OpVerifierContext {
            operand_types: vec![],
            result_types: vec![],
            attributes: vec![],
        };
        assert!(registry.verify("scf.while", &ctx).is_ok());

        let bad_ctx = OpVerifierContext {
            operand_types: vec![Type::Index],
            result_types: vec![],
            attributes: vec![],
        };
        assert!(registry.verify("scf.while", &bad_ctx).is_err());
    }

    #[test]
    fn registry_infer_for_unknown_op() {
        let mut registry = crate::op::OpRegistry::new();
        register_scf_ops(&mut registry);

        let result = registry.infer_result_types("scf.for", &[], &[]);
        assert_eq!(result, None);

        let result = registry.infer_result_types("unknown.op", &[], &[]);
        assert_eq!(result, None);
    }
}
