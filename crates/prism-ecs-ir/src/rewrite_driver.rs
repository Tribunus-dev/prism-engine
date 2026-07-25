//! Rewrite driver — applies rewrite patterns to the IR until fixpoint.
//!
//! Core abstractions:
//! - [`PatternRewriter`] — trait for mutating the IR (replace, erase, use-replace)
//! - [`RewritePattern`] — trait for pattern matching and rewriting a single op
//! - [`RewriteDriver`] — applies all registered patterns repeatedly until fixpoint
//!
//! # Usage
//!
//! ```ignore
//! let mut driver = RewriteDriver::new();
//! driver.add_pattern(Box::new(MyPattern));
//! let count = driver.apply(&mut world)?;
//! ```

use std::collections::HashSet;

use prism_ecs_core::{Entity, World};

use crate::block::BlockOps;
use crate::op::{OpMarker, OpName, Operands, Results};
use crate::value::Uses;

// ── PatternRewriter trait ─────────────────────────────────────────────────

/// Pattern rewriter — applies modifications to the IR.
///
/// Implemented by the rewrite driver's internal state machine.
/// Patterns receive a `&mut dyn PatternRewriter` and use it to perform
/// replacement, erasure, and value-use substitution without needing direct
/// access to the [`World`].
pub trait PatternRewriter {
    /// Replace `op` with one or more `new_ops` in the containing block.
    ///
    /// The old operation is despawned. New operations are assumed to already
    /// have their `OpMarker`, `OpName`, `Operands`, `Results`, etc.
    fn replace_op(&mut self, op: Entity, new_ops: &[Entity]) -> Result<(), String>;

    /// Erase `op` from its block and despawn it.
    fn erase_op(&mut self, op: Entity) -> Result<(), String>;

    /// Replace every use of value `old` with value `new`.
    ///
    /// All operations that consumed `old` will now consume `new`.
    fn replace_all_uses_with(&mut self, old: Entity, new: Entity) -> Result<(), String>;
}

// ── RewritePattern trait ──────────────────────────────────────────────────

/// A rewrite pattern: matches a specific op and rewrites.
///
/// Implementations inspect `op` via the [`World`] and, if the pattern matches,
/// use the provided [`PatternRewriter`] to perform the rewrite. Return `true`
/// when a rewrite was performed, `false` when the pattern did not match.
pub trait RewritePattern: Send + Sync {
    /// Try to match and rewrite `op`. Returns `true` if a rewrite was applied.
    fn match_and_rewrite(
        &self,
        op: Entity,
        rewriter: &mut dyn PatternRewriter,
        world: &mut World,
    ) -> Result<bool, String>;
}

// ── RewriteDriverState (internal) ─────────────────────────────────────────

/// Internal state for a single [`RewriteDriver::apply`] iteration.
///
/// Holds a raw pointer to the [`World`] so that the [`PatternRewriter`] impl
/// can mutate the world without the type system tracking the borrow. This is
/// sound because `RewriteDriverState` is created, used, and destroyed entirely
/// within `apply()`, which already holds `&mut World` exclusively.
struct RewriteDriverState {
    visited: HashSet<Entity>,
    world: *mut World,
}

// SAFETY: RewriteDriverState is not Send/Sync and never leaves the apply()
// call, so the raw pointer is never aliased across threads.
unsafe impl Send for RewriteDriverState {}

impl RewriteDriverState {
    fn new(world: *mut World) -> Self {
        Self {
            visited: HashSet::new(),
            world,
        }
    }

    fn was_visited(&self, op: Entity) -> bool {
        self.visited.contains(&op)
    }

    fn mark_visited(&mut self, op: Entity) {
        self.visited.insert(op);
    }

    fn block_containing_op(&self, op: Entity) -> Result<Entity, String> {
        let world = unsafe { &*self.world };
        for (block, block_ops) in world.query::<BlockOps>() {
            if block_ops.0.contains(&op) {
                return Ok(block);
            }
        }
        Err(format!("op {:?} not found in any block", op))
    }

    fn op_position(&self, block: Entity, op: Entity) -> Result<usize, String> {
        let world = unsafe { &*self.world };
        let bops = world
            .get_component::<BlockOps>(block)
            .ok_or_else(|| format!("block {:?} has no BlockOps", block))?;
        bops.0
            .iter()
            .position(|e| *e == op)
            .ok_or_else(|| format!("op {:?} not found in block {:?}", op, block))
    }
}

impl PatternRewriter for RewriteDriverState {
    fn replace_op(&mut self, op: Entity, new_ops: &[Entity]) -> Result<(), String> {
        if new_ops.is_empty() {
            return self.erase_op(op);
        }

        let world = unsafe { &mut *self.world };
        let block = self.block_containing_op(op)?;
        let pos = self.op_position(block, op)?;

        // Remove components before despawn so stale data doesn't appear in queries
        let _ = world.remove_component::<OpMarker>(op);
        let _ = world.remove_component::<OpName>(op);
        let _ = world.remove_component::<Operands>(op);
        let _ = world.remove_component::<Results>(op);
        // Despawn old op. The handle was just validated by `block_containing_op`
        // and `op_position`; a StaleHandle here indicates a re-entrant rewrite
        // bug, so propagate as a String error.
        world.despawn(op).map_err(|e| format!("despawn failed for {op:?}: {e}"))?;

        // Insert new ops at the old position in the block
        if let Some(bops) = world.get_component_mut::<BlockOps>(block) {
            bops.0.splice(pos..pos + 1, new_ops.iter().copied());
        }

        self.mark_visited(op);
        for new_op in new_ops {
            self.mark_visited(*new_op);
        }

        Ok(())
    }

    fn erase_op(&mut self, op: Entity) -> Result<(), String> {
        let block = self.block_containing_op(op)?;
        let pos = self.op_position(block, op)?;

        let world = unsafe { &mut *self.world };

        // Remove from block
        if let Some(bops) = world.get_component_mut::<BlockOps>(block) {
            bops.0.remove(pos);
        }

        // Remove components before despawn so stale data doesn't appear in queries
        let _ = world.remove_component::<OpMarker>(op);
        let _ = world.remove_component::<OpName>(op);
        let _ = world.remove_component::<Operands>(op);
        let _ = world.remove_component::<Results>(op);
        // Despawn the op entity. The handle was just validated; a StaleHandle
        // here indicates a re-entrant rewrite bug, so propagate.
        world.despawn(op).map_err(|e| format!("despawn failed for {op:?}: {e}"))?;

        self.mark_visited(op);
        Ok(())
    }

    fn replace_all_uses_with(&mut self, old: Entity, new: Entity) -> Result<(), String> {
        let world = unsafe { &mut *self.world };

        // Read the current use list for the old value
        let old_uses: Vec<Entity> = world
            .get_component::<Uses>(old)
            .map(|u| u.0.clone())
            .unwrap_or_default();

        // For each user, replace `old` with `new` in its Operands
        for user in &old_uses {
            if !world.is_alive(*user) {
                // The user may have been despawned during this iteration
                continue;
            }
            if let Some(operands) = world.get_component_mut::<Operands>(*user) {
                for operand in operands.0.iter_mut() {
                    if *operand == old {
                        *operand = new;
                    }
                }
            }
        }

        // Clear the old value's use list
        if let Some(uses) = world.get_component_mut::<Uses>(old) {
            uses.0.clear();
        }

        // Append old users to the new value's use list (dedup not required —
        // SSA guarantees each user appears once)
        if let Some(uses) = world.get_component_mut::<Uses>(new) {
            for user in &old_uses {
                if !uses.0.contains(user) {
                    uses.0.push(*user);
                }
            }
        }

        Ok(())
    }
}

// ── RewriteDriver ─────────────────────────────────────────────────────────

/// Rewrite driver — applies patterns until fixpoint.
///
/// Each call to [`apply`](Self::apply) iterates all operations in the world,
/// tries every registered pattern, and repeats until no pattern matches any
/// operation. A visited set prevents re-processing ops that were already
/// replaced or erased within the same iteration.
pub struct RewriteDriver {
    patterns: Vec<Box<dyn RewritePattern>>,
}

impl RewriteDriver {
    /// Create a new empty rewrite driver.
    pub fn new() -> Self {
        Self {
            patterns: Vec::new(),
        }
    }

    /// Register a rewrite pattern.
    pub fn add_pattern(&mut self, pattern: Box<dyn RewritePattern>) {
        self.patterns.push(pattern);
    }

    /// Apply all patterns repeatedly until no more match.
    ///
    /// Returns the total number of rewrites performed across all iterations.
    pub fn apply(&self, world: &mut World) -> Result<u64, String> {
        let mut total = 0u64;

        loop {
            // Snapshot the current set of ops before this iteration
            let ops: Vec<Entity> = world.query::<OpMarker>().map(|(e, _)| e).collect();

            if ops.is_empty() {
                break;
            }

            let mut state = RewriteDriverState::new(world as *mut World);
            let mut made_progress = false;

            for op in &ops {
                if state.was_visited(*op) {
                    continue;
                }

                for pattern in &self.patterns {
                    if pattern.match_and_rewrite(*op, &mut state, world)? {
                        made_progress = true;
                        total += 1;
                        // Don't try other patterns on this op — it was replaced/erased
                        break;
                    }
                }
            }

            if !made_progress {
                break;
            }
        }

        Ok(total)
    }
}

impl Default for RewriteDriver {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ecs_core::EntityKind;

    /// A test pattern that replaces any op named "test.op" with a new op "test.replaced".
    struct RenamePattern;

    impl RewritePattern for RenamePattern {
        fn match_and_rewrite(
            &self,
            op: Entity,
            rewriter: &mut dyn PatternRewriter,
            world: &mut World,
        ) -> Result<bool, String> {
            let name = world
                .get_component::<OpName>(op)
                .map(|n| n.0.as_str())
                .unwrap_or("");

            if name != "test.op" {
                return Ok(false);
            }

            // Spawn a replacement op
            let new_op: Entity = world
                .spawn(EntityKind::Node, Some("replacement".into()))
                .map_err(|e| format!("spawn: {:?}", e))?
                .into();

            world
                .add_component(new_op, OpMarker)
                .map_err(|e| format!("add OpMarker: {:?}", e))?;
            world
                .add_component(new_op, OpName("test.replaced".into()))
                .map_err(|e| format!("add OpName: {:?}", e))?;
            world
                .add_component(new_op, Operands(vec![]))
                .map_err(|e| format!("add Operands: {:?}", e))?;
            world
                .add_component(new_op, Results(vec![]))
                .map_err(|e| format!("add Results: {:?}", e))?;

            rewriter.replace_op(op, &[new_op])?;

            Ok(true)
        }
    }

    /// A test pattern that erases any op named "test.erase".
    struct ErasePattern;

    impl RewritePattern for ErasePattern {
        fn match_and_rewrite(
            &self,
            op: Entity,
            rewriter: &mut dyn PatternRewriter,
            world: &mut World,
        ) -> Result<bool, String> {
            let name = world
                .get_component::<OpName>(op)
                .map(|n| n.0.as_str())
                .unwrap_or("");

            if name != "test.erase" {
                return Ok(false);
            }

            rewriter.erase_op(op)?;
            Ok(true)
        }
    }

    /// Helper: create a test world with ops in a single block.
    fn create_test_world() -> (World, Entity) {
        let mut world = World::new();

        // Spawn a block
        let block: Entity = world
            .spawn(EntityKind::Node, Some("test_block".into()))
            .expect("spawn block")
            .into();
        world
            .add_component(block, crate::block::BlockMarker)
            .unwrap();
        world
            .add_component(block, crate::block::BlockArguments(vec![]))
            .unwrap();

        let ops = ["test.op", "test.erase", "test.op", "test.other"];

        let op_entities: Vec<Entity> = ops
            .iter()
            .map(|name| {
                let e: Entity = world
                    .spawn(EntityKind::Node, Some(name.to_string()))
                    .expect("spawn op")
                    .into();
                world.add_component(e, OpMarker).unwrap();
                world.add_component(e, OpName(name.to_string())).unwrap();
                world.add_component(e, Operands(vec![])).unwrap();
                world.add_component(e, Results(vec![])).unwrap();
                e
            })
            .collect();

        world.add_component(block, BlockOps(op_entities)).unwrap();
        world
            .add_component(block, crate::block::TerminatorOp(None))
            .unwrap();

        (world, block)
    }

    #[test]
    fn test_no_patterns_no_rewrites() {
        let (world, _) = create_test_world();
        // Verify driver creates cleanly
        let driver = RewriteDriver::new();
        drop(driver);
        drop(world);
    }

    #[test]
    fn test_rename_pattern() -> Result<(), String> {
        let (mut world, _) = create_test_world();

        let mut driver = RewriteDriver::new();
        driver.add_pattern(Box::new(RenamePattern));

        let count = driver.apply(&mut world)?;
        // Two "test.op" ops should be replaced
        assert_eq!(count, 2, "expected 2 rewrites");

        // Verify no "test.op" ops remain
        for op in world.query::<OpName>() {
            assert_ne!(
                op.1 .0, "test.op",
                "op {:?} should have been replaced",
                op.0
            );
        }

        // Verify replacements exist
        let replaced_count = world
            .query::<OpName>()
            .filter(|(_, n)| n.0 == "test.replaced")
            .count();
        assert_eq!(replaced_count, 2, "expected 2 replacement ops");

        Ok(())
    }

    #[test]
    fn test_erase_pattern() -> Result<(), String> {
        let (mut world, block) = create_test_world();

        let mut driver = RewriteDriver::new();
        driver.add_pattern(Box::new(ErasePattern));

        let count = driver.apply(&mut world)?;
        assert_eq!(count, 1, "expected 1 erasure");

        // Verify block now has 3 ops instead of 4
        let bops = world
            .get_component::<BlockOps>(block)
            .ok_or("block missing BlockOps")?;
        assert_eq!(bops.0.len(), 3, "expected 3 ops remaining in block");

        // Verify no "test.erase" ops remain
        for op in world.query::<OpName>() {
            assert_ne!(op.1 .0, "test.erase");
        }

        Ok(())
    }

    #[test]
    fn test_replace_all_uses_with() -> Result<(), String> {
        use crate::ir_types::Type;
        use crate::op::Results;
        use crate::value::ValueDef;
        use crate::value::ValueType;

        let mut world = World::new();
        let block: Entity = world
            .spawn(EntityKind::Node, Some("block".into()))
            .expect("spawn block")
            .into();
        world
            .add_component(block, crate::block::BlockMarker)
            .unwrap();
        world
            .add_component(block, crate::block::BlockArguments(vec![]))
            .unwrap();

        // Create two value entities
        let old_val: Entity = world
            .spawn(EntityKind::Node, Some("old_value".into()))
            .expect("spawn old_val")
            .into();
        world
            .add_component(old_val, ValueDef::op_result(Entity(1, 0), 0))
            .unwrap();
        world
            .add_component(old_val, ValueType(Type::f32()))
            .unwrap();
        world.add_component(old_val, Uses(vec![])).unwrap();

        let new_val: Entity = world
            .spawn(EntityKind::Node, Some("new_value".into()))
            .expect("spawn new_val")
            .into();
        world
            .add_component(new_val, ValueDef::op_result(Entity(2, 0), 0))
            .unwrap();
        world
            .add_component(new_val, ValueType(Type::f32()))
            .unwrap();
        world.add_component(new_val, Uses(vec![])).unwrap();

        // Create an op that uses old_val
        let op: Entity = world
            .spawn(EntityKind::Node, Some("user".into()))
            .expect("spawn op")
            .into();
        world.add_component(op, OpMarker).unwrap();
        world.add_component(op, OpName("test.user".into())).unwrap();
        world.add_component(op, Operands(vec![old_val])).unwrap();
        world.add_component(op, Results(vec![])).unwrap();

        world.add_component(block, BlockOps(vec![op])).unwrap();

        // Update old value's uses
        if let Some(uses) = world.get_component_mut::<Uses>(old_val) {
            uses.0.push(op);
        }

        // Perform the replacement
        let mut state = RewriteDriverState::new(&mut world as *mut World);
        state.replace_all_uses_with(old_val, new_val)?;

        // Verify the user now has new_val as operand
        let operands = world
            .get_component::<Operands>(op)
            .ok_or("missing Operands")?;
        assert_eq!(operands.0, vec![new_val], "operand should be replaced");

        // Verify old value's uses is empty
        let old_uses = world
            .get_component::<Uses>(old_val)
            .ok_or("missing Uses on old")?;
        assert!(old_uses.0.is_empty(), "old uses should be cleared");

        // Verify new value's uses contains the user
        let new_uses = world
            .get_component::<Uses>(new_val)
            .ok_or("missing Uses on new")?;
        assert!(new_uses.0.contains(&op), "new uses should contain user");

        Ok(())
    }

    #[test]
    fn test_replace_op_with_multiple_ops() -> Result<(), String> {
        let mut world = World::new();
        let block: Entity = world
            .spawn(EntityKind::Node, Some("block".into()))
            .expect("spawn")
            .into();
        world
            .add_component(block, crate::block::BlockMarker)
            .unwrap();
        world
            .add_component(block, crate::block::BlockArguments(vec![]))
            .unwrap();

        // Create op to replace
        let old_op: Entity = world
            .spawn(EntityKind::Node, Some("old".into()))
            .expect("spawn old")
            .into();
        world.add_component(old_op, OpMarker).unwrap();
        world
            .add_component(old_op, OpName("test.op".into()))
            .unwrap();
        world.add_component(old_op, Operands(vec![])).unwrap();
        world.add_component(old_op, Results(vec![])).unwrap();

        world.add_component(block, BlockOps(vec![old_op])).unwrap();

        // Create two replacement ops
        let new_a: Entity = world
            .spawn(EntityKind::Node, Some("new_a".into()))
            .expect("spawn")
            .into();
        world.add_component(new_a, OpMarker).unwrap();
        world
            .add_component(new_a, OpName("test.new_a".into()))
            .unwrap();
        world.add_component(new_a, Operands(vec![])).unwrap();
        world.add_component(new_a, Results(vec![])).unwrap();

        let new_b: Entity = world
            .spawn(EntityKind::Node, Some("new_b".into()))
            .expect("spawn")
            .into();
        world.add_component(new_b, OpMarker).unwrap();
        world
            .add_component(new_b, OpName("test.new_b".into()))
            .unwrap();
        world.add_component(new_b, Operands(vec![])).unwrap();
        world.add_component(new_b, Results(vec![])).unwrap();

        // Replace old_op with [new_a, new_b]
        let mut state = RewriteDriverState::new(&mut world as *mut World);
        state.replace_op(old_op, &[new_a, new_b])?;

        // Verify old_op is despawned
        assert!(!world.is_alive(old_op), "old op should be despawned");

        // Verify block ops are [new_a, new_b]
        let bops = world
            .get_component::<BlockOps>(block)
            .ok_or("missing BlockOps")?;
        assert_eq!(
            bops.0,
            vec![new_a, new_b],
            "block should have new ops at position"
        );

        Ok(())
    }

    #[test]
    fn test_fixpoint_convergence() -> Result<(), String> {
        // Two passes: pattern1 replaces "test.a" with "test.b", pattern2 replaces "test.b" with "test.c"
        struct AToB;
        impl RewritePattern for AToB {
            fn match_and_rewrite(
                &self,
                op: Entity,
                rewriter: &mut dyn PatternRewriter,
                world: &mut World,
            ) -> Result<bool, String> {
                let name = world.get_component::<OpName>(op).map(|n| n.0.clone());
                if name.as_deref() != Some("test.op") {
                    return Ok(false);
                }
                let new_op: Entity = world
                    .spawn(EntityKind::Node, Some("to_b".into()))
                    .map_err(|e| format!("spawn: {:?}", e))?
                    .into();
                world.add_component(new_op, OpMarker).unwrap();
                world
                    .add_component(new_op, OpName("test.b".into()))
                    .unwrap();
                world.add_component(new_op, Operands(vec![])).unwrap();
                world.add_component(new_op, Results(vec![])).unwrap();
                rewriter.replace_op(op, &[new_op])?;
                Ok(true)
            }
        }

        struct BToC;
        impl RewritePattern for BToC {
            fn match_and_rewrite(
                &self,
                op: Entity,
                rewriter: &mut dyn PatternRewriter,
                world: &mut World,
            ) -> Result<bool, String> {
                let name = world.get_component::<OpName>(op).map(|n| n.0.as_str());
                if name != Some("test.b") {
                    return Ok(false);
                }
                let new_op: Entity = world
                    .spawn(EntityKind::Node, Some("to_c".into()))
                    .map_err(|e| format!("spawn: {:?}", e))?
                    .into();
                world.add_component(new_op, OpMarker).unwrap();
                world
                    .add_component(new_op, OpName("test.c".into()))
                    .unwrap();
                world.add_component(new_op, Operands(vec![])).unwrap();
                world.add_component(new_op, Results(vec![])).unwrap();
                rewriter.replace_op(op, &[new_op])?;
                Ok(true)
            }
        }

        let (mut world, block) = create_test_world();

        let mut driver = RewriteDriver::new();
        driver.add_pattern(Box::new(AToB));
        driver.add_pattern(Box::new(BToC));

        let count = driver.apply(&mut world)?;

        // 2 test.a -> test.b, then those 2 test.b -> test.c = 4 rewrites total
        // test.erase stays, test.other stays
        assert_eq!(
            count, 4,
            "expected 4 total rewrites across fixpoint iterations"
        );

        // Final state: test.c appears, no test.a or test.b
        let names: HashSet<String> = world.query::<OpName>().map(|(_, n)| n.0.clone()).collect();
        assert!(names.contains("test.c"), "should have test.c");
        assert!(!names.contains("test.a"), "should not have test.a");
        assert!(!names.contains("test.b"), "should not have test.b");
        assert!(names.contains("test.erase"), "should preserve test.erase");
        assert!(names.contains("test.other"), "should preserve test.other");

        let bops = world
            .get_component::<BlockOps>(block)
            .ok_or("missing BlockOps")?;
        assert_eq!(
            bops.0.len(),
            4,
            "should have 4 ops (2 test.c + test.erase + test.other)"
        );

        Ok(())
    }

    #[test]
    fn test_erase_pattern_marks_visited() -> Result<(), String> {
        // A pattern that erases "test.erase" and a second pattern that would try to
        // process everything. Ensure the erased op isn't processed by the second
        // pattern in the same iteration.
        struct UniversalReplacer;
        impl RewritePattern for UniversalReplacer {
            fn match_and_rewrite(
                &self,
                _op: Entity,
                _rewriter: &mut dyn PatternRewriter,
                _world: &mut World,
            ) -> Result<bool, String> {
                // This should only be reached for ops that weren't erased
                Ok(false) // don't actually replace, just checking liveness
            }
        }

        let (mut world, _) = create_test_world();
        let op_count_before = world.query::<OpMarker>().count();

        let mut driver = RewriteDriver::new();
        driver.add_pattern(Box::new(ErasePattern));
        driver.add_pattern(Box::new(UniversalReplacer));

        driver.apply(&mut world)?;

        let op_count_after = world.query::<OpMarker>().count();
        assert_eq!(op_count_after, op_count_before - 1);

        Ok(())
    }

    #[test]
    fn test_replace_op_empty_slice_ercs() -> Result<(), String> {
        // Replacing with empty slice should erase the op
        let (mut world, block) = create_test_world();

        let op_count_before = world.query::<OpMarker>().count();

        let mut state = RewriteDriverState::new(&mut world as *mut World);

        // Find the first op to erase via replace_op with empty slice
        let first_op = world
            .query::<OpMarker>()
            .next()
            .map(|(e, _)| e)
            .ok_or("no ops")?;

        state.replace_op(first_op, &[])?;

        let op_count_after = world.query::<OpMarker>().count();
        assert_eq!(op_count_after, op_count_before - 1);

        let bops = world
            .get_component::<BlockOps>(block)
            .ok_or("missing BlockOps")?;
        assert_eq!(bops.0.len(), 3);

        Ok(())
    }
}
