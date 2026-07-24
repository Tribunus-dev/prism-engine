//! Lowering patterns — ECS-native IR-to-IR lowering.
//!
//! Provides rewrite patterns that lower high-level dialect ops (linalg.matmul)
//! to lower-level ops (scf.for loops with vector operations).
//!
//! Each lowering is a RewritePattern that can be registered with the RewriteDriver.

use prism_ecs_core::{Entity, World};

use crate::builder::OpBuilder;
use crate::op::{Operands, Results};

/// Lower a linalg.matmul to scf.for loops.
///
/// linalg.matmul(A[M,K], B[K,N], C[M,N]) is lowered to:
/// ```ignore
/// scf.for %i = 0 to M:
///   scf.for %j = 0 to N:
///     scf.for %k = 0 to K:
///       %val = arith.mulf A[%i][%k] B[%k][%j]
///       %acc = arith.addf C[%i][%j] %val
/// ```
pub fn lower_matmul(world: &mut World, matmul_op: Entity) -> Result<Entity, String> {
    // Read the matmul operand entities
    let operands = world
        .get_component::<Operands>(matmul_op)
        .map(|o| o.0.clone())
        .ok_or("matmul op has no Operands")?;

    if operands.len() != 3 {
        return Err(format!(
            "matmul expected 3 operands, got {}",
            operands.len()
        ));
    }

    let _a = operands[0];
    let _b = operands[1];
    let _c = operands[2];

    // Read the result entity
    let results = world
        .get_component::<Results>(matmul_op)
        .map(|r| r.0.clone())
        .ok_or("matmul op has no Results")?;

    if results.is_empty() {
        return Err("matmul op has no results".into());
    }

    let result_ty = world
        .get_component::<crate::value::ValueType>(results[0])
        .map(|vt| vt.0.clone())
        .ok_or("matmul result has no ValueType")?;

    // Create scf.for loop structure using OpBuilder
    let mut builder = OpBuilder::new(world);

    // Create scf.for loop op
    let loop_op = builder
        .create_op("scf.for", &[], &[], &[result_ty])
        .map_err(|e| format!("failed to create scf.for: {:?}", e))?;

    // Create a region with blocks inside the loop
    let _region = builder
        .create_region(crate::region::RegionKind::SSACFG)
        .map_err(|e| format!("failed to create region: {:?}", e))?;

    let (_block, _args) = builder
        .create_block(&[])
        .map_err(|e| format!("failed to create block: {:?}", e))?;

    // The full lowering would wire up operands, create scf.yield, etc.
    // For now, this is a structural scaffold.

    Ok(loop_op)
}

#[cfg(test)]
mod tests {
    use crate::ir_types::Type;
    use prism_ecs_core::{EntityKind, World};

    use super::*;
    use crate::op::{OpAttributes, OpMarker, OpName, Operands, Results};
    use crate::value::{Uses, ValueDef, ValueType};

    #[test]
    fn lower_matmul_creates_loop() {
        let mut world = World::new();

        // Create matmul operands
        let a: Entity = world
            .spawn(EntityKind::Node, Some("A".into()))
            .unwrap()
            .into();
        world
            .add_component(a, ValueDef::op_result(Entity(0, 1), 0))
            .unwrap();
        world.add_component(a, ValueType(Type::f32())).unwrap();
        world.add_component(a, Uses(vec![])).unwrap();

        let b: Entity = world
            .spawn(EntityKind::Node, Some("B".into()))
            .unwrap()
            .into();
        world
            .add_component(b, ValueDef::op_result(Entity(0, 1), 1))
            .unwrap();
        world.add_component(b, ValueType(Type::f32())).unwrap();
        world.add_component(b, Uses(vec![])).unwrap();

        let c: Entity = world
            .spawn(EntityKind::Node, Some("C".into()))
            .unwrap()
            .into();
        world
            .add_component(c, ValueDef::op_result(Entity(0, 1), 2))
            .unwrap();
        world.add_component(c, ValueType(Type::f32())).unwrap();
        world.add_component(c, Uses(vec![])).unwrap();

        // Create matmul result value
        let res: Entity = world
            .spawn(EntityKind::Node, Some("result".into()))
            .unwrap()
            .into();
        world
            .add_component(res, ValueDef::op_result(Entity(0, 1), 0))
            .unwrap();
        world.add_component(res, ValueType(Type::f32())).unwrap();
        world.add_component(res, Uses(vec![])).unwrap();

        // Create the matmul op
        let matmul: Entity = world
            .spawn(EntityKind::Node, Some("matmul".into()))
            .unwrap()
            .into();
        world.add_component(matmul, OpMarker).unwrap();
        world
            .add_component(matmul, OpName("linalg.matmul".into()))
            .unwrap();
        world
            .add_component(matmul, Operands(vec![a, b, c]))
            .unwrap();
        world.add_component(matmul, Results(vec![res])).unwrap();
        world.add_component(matmul, OpAttributes(vec![])).unwrap();

        // Lower
        let loop_op = lower_matmul(&mut world, matmul).expect("lowering failed");

        // Verify the loop op was created
        let name = world.get_component::<OpName>(loop_op).map(|n| n.0.clone());
        assert_eq!(name, Some("scf.for".into()));
    }
}
