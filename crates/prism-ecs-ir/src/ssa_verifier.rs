//! SSA dominance verifier.
//!
//! Walks all operations in the IR tree rooted at a given op and verifies
//! that every operand use satisfies SSA dominance: the defining entity
//! (operation or block) must dominate all its uses.
//!
//! Also checks that uses don't cross region boundaries for values that
//! aren't block arguments of the target region.
//!
//! # Checking strategy
//!
//! 1. Recursively walk the IR tree from the root op, collecting all
//!    operations and their containing blocks/regions.
//! 2. For each region, compute the dominator tree.
//! 3. For each operation, inspect every operand value and verify:
//!    - OpResult: defining op must dominate the using op.
//!    - BlockArgument: defining block must dominate the using block.
//!    - Same-region: values must not cross region boundaries.

use std::collections::HashMap;

use prism_ecs_core::{Entity, World};

use crate::block::block_ops;
use crate::dominance::DominanceAnalyzer;
use crate::op::operands;
use crate::region::region_blocks;
use crate::value::{ValueDef, ValueKind};

// ---------------------------------------------------------------------------
// SsaViolation
// ---------------------------------------------------------------------------

/// A detected SSA violation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SsaViolation {
    /// A value is used before its defining operation executes.
    UseBeforeDef {
        /// The operation consuming the value.
        user: Entity,
        /// The value entity that is not dominated by its definition.
        value: Entity,
        /// The operation that defines this value (or the defining block for
        /// block arguments).
        defining_op: Entity,
    },
    /// A value is used in a different region than where it was defined,
    /// without being a captured/implicit value.
    DefInDifferentRegion {
        /// The value entity.
        value: Entity,
        /// The region where the value is defined.
        def_region: Entity,
        /// The region where the value is used.
        use_region: Entity,
    },
}

// ---------------------------------------------------------------------------
// SsaVerifier
// ---------------------------------------------------------------------------

/// SSA dominance verifier.
///
/// # Example
///
/// ```ignore
/// let violations = SsaVerifier::verify(&world, root_op);
/// assert!(violations.is_empty(), "SSA violations: {:?}", violations);
/// ```
pub struct SsaVerifier;

impl SsaVerifier {
    /// Create a new SSA verifier.
    pub fn new() -> Self {
        Self
    }

    /// Verify SSA dominance for all ops in the IR tree under `root_op`.
    ///
    /// Walks every region, block, and operation reachable from the root.
    /// Returns the list of violations found. An empty list means the IR is
    /// clean with respect to SSA dominance.
    pub fn verify(world: &World, root_op: Entity) -> Vec<SsaViolation> {
        let mut violations = Vec::new();

        // Map each operation to its containing block entity.
        let mut op_to_block: HashMap<Entity, Entity> = HashMap::new();
        // Map each block to its containing region entity.
        let mut block_to_region: HashMap<Entity, Entity> = HashMap::new();
        // Collect operations per region — one list per region.
        let mut region_ops: HashMap<Entity, Vec<Entity>> = HashMap::new();

        // Walk the IR tree from the root op, populating maps.
        walk_ir(
            world,
            root_op,
            &mut op_to_block,
            &mut block_to_region,
            &mut region_ops,
        );

        // For each region, compute dominance and check every operand.
        for (&region, ops) in &region_ops {
            if ops.is_empty() {
                continue;
            }

            let analyzer = DominanceAnalyzer::new();
            let dom_tree = analyzer.compute_dominators(region, world);

            for &user_op in ops {
                let op_operands = operands(world, user_op);
                for &value in &op_operands {
                    let Some(value_def) = world.get_component::<ValueDef>(value) else {
                        // Value without a ValueDef — malformed, but don't crash.
                        continue;
                    };

                    match value_def.kind {
                        ValueKind::OpResult => {
                            let defining_op = value_def.defining_entity;

                            // Locate the defining op's block and region.
                            let Some(&def_block) = op_to_block.get(&defining_op) else {
                                continue;
                            };
                            let Some(&def_region) = block_to_region.get(&def_block) else {
                                continue;
                            };

                            // Locate the using op's block and region.
                            let Some(&use_block) = op_to_block.get(&user_op) else {
                                continue;
                            };
                            let Some(&use_region) = block_to_region.get(&use_block) else {
                                continue;
                            };

                            // Cross-region value use is always a violation:
                            // OpResult values are scoped to their defining region.
                            if def_region != use_region {
                                violations.push(SsaViolation::DefInDifferentRegion {
                                    value,
                                    def_region,
                                    use_region,
                                });
                                continue;
                            }

                            // Dominance check.
                            if def_block == use_block {
                                // Same block: defining op must appear before the using op
                                // in the block's operation list (or be the same op in the
                                // degenerate case of a result used in its defining op,
                                // which is valid).
                                let block_ops_list = block_ops(world, def_block);
                                let def_pos = block_ops_list.iter().position(|&e| e == defining_op);
                                let use_pos = block_ops_list.iter().position(|&e| e == user_op);

                                match (def_pos, use_pos) {
                                    (Some(dp), Some(up)) if dp > up => {
                                        // Use before definition in the same block.
                                        violations.push(SsaViolation::UseBeforeDef {
                                            user: user_op,
                                            value,
                                            defining_op,
                                        });
                                    }
                                    _ => {
                                        // Valid: dp <= up, or one of the ops isn't in
                                        // the block's op list (edge case).
                                    }
                                }
                            } else {
                                // Different blocks: check block-level dominance.
                                if !block_dominates(&dom_tree, def_block, use_block) {
                                    violations.push(SsaViolation::UseBeforeDef {
                                        user: user_op,
                                        value,
                                        defining_op,
                                    });
                                }
                            }
                        }

                        ValueKind::BlockArgument => {
                            let defining_block = value_def.defining_entity;

                            let Some(&use_block) = op_to_block.get(&user_op) else {
                                continue;
                            };
                            let Some(&def_region) = block_to_region.get(&defining_block) else {
                                continue;
                            };
                            let Some(&use_region) = block_to_region.get(&use_block) else {
                                continue;
                            };

                            // Cross-region block argument use is a violation.
                            if def_region != use_region {
                                violations.push(SsaViolation::DefInDifferentRegion {
                                    value,
                                    def_region,
                                    use_region,
                                });
                                continue;
                            }

                            // A block argument is considered "defined" at the block
                            // entry, so it dominates the block itself and all blocks
                            // reachable from it. Only flag if the defining block
                            // doesn't dominate the using block.
                            if defining_block != use_block
                                && !block_dominates(&dom_tree, defining_block, use_block)
                            {
                                violations.push(SsaViolation::UseBeforeDef {
                                    user: user_op,
                                    value,
                                    defining_op: defining_block,
                                });
                            }
                        }
                    }
                }
            }
        }

        violations
    }
}

impl Default for SsaVerifier {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Check whether `def_block` dominates `use_block` in the given dominator
/// tree by climbing the immediate-dominator chain from `use_block` up to
/// the entry.
fn block_dominates(
    tree: &crate::dominance::DominatorTree,
    def_block: Entity,
    use_block: Entity,
) -> bool {
    // Climb the idom chain from use_block toward the entry.
    // If we ever reach def_block, dominance holds.
    let mut current = use_block;
    loop {
        if current == def_block {
            return true;
        }
        match tree.immediate_dominators.get(&current) {
            Some(&idom) if idom == current => {
                // Reached entry (its own dominator) without finding def_block.
                break;
            }
            Some(&idom) => {
                current = idom;
            }
            None => {
                // Block not in the dominator tree — can't dominate anything.
                break;
            }
        }
    }
    false
}

/// Recursively walk the IR tree starting from `op`, populating:
///
/// - `op_to_block`: maps each operation entity to its containing block.
/// - `block_to_region`: maps each block entity to its containing region.
/// - `region_ops`: maps each region to the list of ops it contains.
///
/// The walk descends into regions owned by any op encountered.
fn walk_ir(
    world: &World,
    op: Entity,
    op_to_block: &mut HashMap<Entity, Entity>,
    block_to_region: &mut HashMap<Entity, Entity>,
    region_ops: &mut HashMap<Entity, Vec<Entity>>,
) {
    // Check if this op owns any regions.
    let Some(region_ref) = world.get_component::<crate::op::RegionRef>(op) else {
        return;
    };

    for &region in &region_ref.0 {
        let blocks = region_blocks(world, region);
        for &block in &blocks {
            block_to_region.insert(block, region);
            let ops = block_ops(world, block);
            for &bop in &ops {
                op_to_block.insert(bop, block);
                region_ops.entry(region).or_default().push(bop);
                // Recurse into child ops that own their own regions.
                walk_ir(world, bop, op_to_block, block_to_region, region_ops);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::{BlockArguments, BlockMarker, BlockOps, TerminatorOp};
    use crate::ir_types::Type;
    use crate::op::{OpAttributes, OpMarker, OpName, Operands, RegionRef, Results};
    use crate::region::{RegionBlocks, RegionKind, RegionKindComp, RegionMarker};
    use crate::value::{Uses, ValueDef, ValueType};
    use prism_ecs_core::{EntityKind, World};

    /// Helper: create a block entity with the required components.
    fn create_block(world: &mut World, name: &str) -> Entity {
        let entity: Entity = world
            .spawn(EntityKind::Node, Some(name.into()))
            .expect("spawn failed")
            .into();
        world
            .add_component(entity, BlockMarker)
            .expect("add BlockMarker");
        world
            .add_component(entity, BlockArguments(vec![]))
            .expect("add BlockArguments");
        world
            .add_component(entity, BlockOps(vec![]))
            .expect("add BlockOps");
        world
            .add_component(entity, TerminatorOp(None))
            .expect("add TerminatorOp");
        entity
    }

    /// Helper: create a region with an ordered list of blocks.
    fn create_region(world: &mut World, blocks: Vec<Entity>) -> Entity {
        let entity: Entity = world
            .spawn(EntityKind::Node, Some("test_region".into()))
            .expect("spawn failed")
            .into();
        world
            .add_component(entity, RegionMarker)
            .expect("add RegionMarker");
        world
            .add_component(entity, RegionKindComp(RegionKind::SSACFG))
            .expect("add RegionKindComp");
        world
            .add_component(entity, RegionBlocks(blocks))
            .expect("add RegionBlocks");
        entity
    }

    /// Helper: create a value entity.
    fn create_value(world: &mut World, def: ValueDef, ty: Type) -> Entity {
        let entity: Entity = world
            .spawn(EntityKind::Node, None)
            .expect("spawn failed")
            .into();
        world.add_component(entity, def).expect("add ValueDef");
        world
            .add_component(entity, ValueType(ty))
            .expect("add ValueType");
        world.add_component(entity, Uses(vec![])).expect("add Uses");
        entity
    }

    /// Helper: create an operation entity.
    fn create_op(
        world: &mut World,
        name: &str,
        op_operands: Vec<Entity>,
        results: Vec<Entity>,
    ) -> Entity {
        let entity: Entity = world
            .spawn(EntityKind::Node, Some(name.into()))
            .expect("spawn failed")
            .into();
        world.add_component(entity, OpMarker).expect("add OpMarker");
        world
            .add_component(entity, OpName(name.into()))
            .expect("add OpName");
        world
            .add_component(entity, Operands(op_operands))
            .expect("add Operands");
        world
            .add_component(entity, Results(results))
            .expect("add Results");
        world
            .add_component(entity, OpAttributes(vec![]))
            .expect("add OpAttributes");
        entity
    }

    // -----------------------------------------------------------------------
    // valid_ssa_chain — op defines, later op uses (same block)
    // -----------------------------------------------------------------------

    #[test]
    fn valid_ssa_chain() {
        let mut world = World::new();

        // Build: region -> block -> [def_op, use_op]
        let block = create_block(&mut world, "entry");
        let region = create_region(&mut world, vec![block]);

        // Root op (like func) owns the region.
        let root_op = create_op(&mut world, "func", vec![], vec![]);
        world
            .add_component(root_op, RegionRef(vec![region]))
            .expect("add RegionRef");

        // Value defined by def_op.
        let val = create_value(
            &mut world,
            ValueDef::op_result(Entity::new(0, 0), 0), // placeholder, updated below
            Type::f32(),
        );

        // def_op: produces val
        let def_op = create_op(&mut world, "test.def", vec![], vec![val]);

        // Fix the placeholder ValueDef to point to the real def_op.
        world
            .add_component(val, ValueDef::op_result(def_op, 0))
            .expect("fix ValueDef");

        // use_op: consumes val
        let use_op = create_op(&mut world, "test.use", vec![val], vec![]);

        // Both ops in the same block, def before use.
        world
            .add_component(block, BlockOps(vec![def_op, use_op]))
            .expect("add BlockOps");

        let violations = SsaVerifier::verify(&world, root_op);
        assert!(
            violations.is_empty(),
            "valid SSA chain should have no violations, got: {:?}",
            violations
        );
    }

    // -----------------------------------------------------------------------
    // use_before_def — op uses a value whose definition appears later (same block)
    // -----------------------------------------------------------------------

    #[test]
    fn use_before_def_same_block() {
        let mut world = World::new();

        let block = create_block(&mut world, "entry");
        let region = create_region(&mut world, vec![block]);

        let root_op = create_op(&mut world, "func", vec![], vec![]);
        world
            .add_component(root_op, RegionRef(vec![region]))
            .expect("add RegionRef");

        // Create ops and value in reversed order.
        let val = create_value(
            &mut world,
            ValueDef::op_result(Entity::new(0, 0), 0),
            Type::f32(),
        );

        // use_op appears BEFORE def_op in the block.
        let use_op = create_op(&mut world, "test.use", vec![val], vec![]);
        let def_op = create_op(&mut world, "test.def", vec![], vec![val]);

        world
            .add_component(val, ValueDef::op_result(def_op, 0))
            .expect("fix ValueDef");

        // use_op before def_op in block ordering.
        world
            .add_component(block, BlockOps(vec![use_op, def_op]))
            .expect("add BlockOps");

        let violations = SsaVerifier::verify(&world, root_op);
        assert_eq!(
            violations.len(),
            1,
            "expected one violation, got: {:?}",
            violations
        );
        match &violations[0] {
            SsaViolation::UseBeforeDef {
                user,
                value,
                defining_op,
            } => {
                assert_eq!(*user, use_op, "violation should name the using op");
                assert_eq!(*value, val, "violation should name the value");
                assert_eq!(*defining_op, def_op, "violation should name the def op");
            }
            other => panic!("expected UseBeforeDef, got: {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // use_before_def — cross-block: def in later block, use in entry
    // -----------------------------------------------------------------------

    #[test]
    fn use_before_def_cross_block() {
        let mut world = World::new();

        // Two blocks: entry does NOT dominate block2 in a linear chain?
        // In our order-based predecessor model:
        //   preds(entry)  = {}
        //   preds(block2) = {entry}
        //   idom(entry)  = entry
        //   idom(block2) = entry
        // So entry dominates block2, but block2 does NOT dominate entry.
        //
        // Therefore: def in block2, use in entry => use-before-def violation.
        let entry = create_block(&mut world, "entry");
        let later = create_block(&mut world, "later");
        let region = create_region(&mut world, vec![entry, later]);

        let root_op = create_op(&mut world, "func", vec![], vec![]);
        world
            .add_component(root_op, RegionRef(vec![region]))
            .expect("add RegionRef");

        let val = create_value(
            &mut world,
            ValueDef::op_result(Entity::new(0, 0), 0),
            Type::f32(),
        );

        // def_op in later block.
        let def_op = create_op(&mut world, "test.def", vec![], vec![val]);
        world
            .add_component(val, ValueDef::op_result(def_op, 0))
            .expect("fix ValueDef");
        world
            .add_component(later, BlockOps(vec![def_op]))
            .expect("add BlockOps");

        // use_op in entry block — entry doesn't post-dominate later.
        let use_op = create_op(&mut world, "test.use", vec![val], vec![]);
        world
            .add_component(entry, BlockOps(vec![use_op]))
            .expect("add BlockOps");

        let violations = SsaVerifier::verify(&world, root_op);
        assert_eq!(
            violations.len(),
            1,
            "expected one cross-block violation, got: {:?}",
            violations
        );
        match &violations[0] {
            SsaViolation::UseBeforeDef {
                user,
                value,
                defining_op,
            } => {
                assert_eq!(*user, use_op);
                assert_eq!(*value, val);
                assert_eq!(*defining_op, def_op);
            }
            other => panic!("expected UseBeforeDef, got: {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // cross_region_use — value defined in one region, used in another
    // -----------------------------------------------------------------------

    #[test]
    fn cross_region_use() {
        let mut world = World::new();

        // Two independent regions, each with one block.
        let block_a = create_block(&mut world, "block_a");
        let region_a = create_region(&mut world, vec![block_a]);

        let block_b = create_block(&mut world, "block_b");
        let region_b = create_region(&mut world, vec![block_b]);

        // Root op owns both regions (e.g., a parent with two child regions).
        let root_op = create_op(&mut world, "multi_region", vec![], vec![]);
        world
            .add_component(root_op, RegionRef(vec![region_a, region_b]))
            .expect("add RegionRef");

        // Value defined in region_a.
        let val = create_value(
            &mut world,
            ValueDef::op_result(Entity::new(0, 0), 0),
            Type::f32(),
        );

        let def_op = create_op(&mut world, "test.def", vec![], vec![val]);
        world
            .add_component(val, ValueDef::op_result(def_op, 0))
            .expect("fix ValueDef");
        world
            .add_component(block_a, BlockOps(vec![def_op]))
            .expect("add BlockOps");

        // Use in region_b — cross-region violation.
        let use_op = create_op(&mut world, "test.use", vec![val], vec![]);
        world
            .add_component(block_b, BlockOps(vec![use_op]))
            .expect("add BlockOps");

        let violations = SsaVerifier::verify(&world, root_op);
        assert_eq!(
            violations.len(),
            1,
            "expected one cross-region violation, got: {:?}",
            violations
        );
        match &violations[0] {
            SsaViolation::DefInDifferentRegion {
                value,
                def_region,
                use_region,
            } => {
                assert_eq!(*value, val);
                assert_eq!(*def_region, region_a);
                assert_eq!(*use_region, region_b);
            }
            other => panic!("expected DefInDifferentRegion, got: {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // empty_region — no violations in an empty region
    // -----------------------------------------------------------------------

    #[test]
    fn empty_region_clean() {
        let mut world = World::new();

        let region = create_region(&mut world, vec![]);

        let root_op = create_op(&mut world, "func", vec![], vec![]);
        world
            .add_component(root_op, RegionRef(vec![region]))
            .expect("add RegionRef");

        let violations = SsaVerifier::verify(&world, root_op);
        assert!(
            violations.is_empty(),
            "empty region should have no violations"
        );
    }

    // -----------------------------------------------------------------------
    // block_argument_use — valid use of a block argument
    // -----------------------------------------------------------------------

    #[test]
    fn block_argument_valid_use() {
        let mut world = World::new();

        // Block with one argument.
        let block: Entity = world
            .spawn(EntityKind::Node, Some("entry".into()))
            .expect("spawn failed")
            .into();
        world
            .add_component(block, BlockMarker)
            .expect("add BlockMarker");

        let arg_val = create_value(&mut world, ValueDef::block_argument(block, 0), Type::i32());

        world
            .add_component(block, BlockArguments(vec![arg_val]))
            .expect("add BlockArguments");
        world
            .add_component(block, TerminatorOp(None))
            .expect("add TerminatorOp");

        let region = create_region(&mut world, vec![block]);
        let root_op = create_op(&mut world, "func", vec![], vec![]);
        world
            .add_component(root_op, RegionRef(vec![region]))
            .expect("add RegionRef");

        // Op in the same block uses the block argument — valid.
        let use_op = create_op(&mut world, "test.use", vec![arg_val], vec![]);
        world
            .add_component(block, BlockOps(vec![use_op]))
            .expect("add BlockOps");

        let violations = SsaVerifier::verify(&world, root_op);
        assert!(
            violations.is_empty(),
            "block argument use in same region should be valid, got: {:?}",
            violations
        );
    }

    // -----------------------------------------------------------------------
    // op_result_used_by_itself — self-use is valid (degenerate case)
    // -----------------------------------------------------------------------

    #[test]
    fn op_uses_own_result() {
        let mut world = World::new();

        let block = create_block(&mut world, "entry");
        let region = create_region(&mut world, vec![block]);

        let root_op = create_op(&mut world, "func", vec![], vec![]);
        world
            .add_component(root_op, RegionRef(vec![region]))
            .expect("add RegionRef");

        // The op uses its own result — degenerate but allowed (the def dominates
        // the use since def_pos == use_pos; the op exists at a point where it
        // has already "produced" its results from the user's perspective in SSA).
        let val = create_value(
            &mut world,
            ValueDef::op_result(Entity::new(0, 0), 0),
            Type::f32(),
        );

        let def_op = create_op(&mut world, "test.selfref", vec![val], vec![val]);
        world
            .add_component(val, ValueDef::op_result(def_op, 0))
            .expect("fix ValueDef");
        world
            .add_component(block, BlockOps(vec![def_op]))
            .expect("add BlockOps");

        let violations = SsaVerifier::verify(&world, root_op);
        assert!(
            violations.is_empty(),
            "self-use should be valid, got: {:?}",
            violations
        );
    }
}
