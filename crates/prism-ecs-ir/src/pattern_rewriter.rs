//! Pattern rewriter — match-and-rewrite over the ECS-native IR.
//!
//! This module provides a higher-level rewrite abstraction than the
//! `rewrite_driver` module. Patterns return match results (an `OpRewrite`
//! action or actions) rather than mutating the IR inline, and a worklist
//! loop drives fixpoint convergence with canonicalization, constant folding,
//! and CSE.
//!
//! # Design
//!
//! Each [`RewritePattern`] implements `match(root_op) -> Option<SmallVec<OpRewrite>>`.
//! The [`PatternRewriter`] applies the returned actions to the ECS [`World`]
//! using a worklist-based driver. Actions are first-class operations:
//!
//! - `ReplaceOp` — replace an op with zero or more new ops
//! - `EraseOp` — erase an op entirely
//! - `CreateOp` — insert a new op at a given insertion point

use prism_ecs_core::{Entity, World};
use smallvec::SmallVec;

use crate::block::BlockOps;
use crate::ir_attrs::Attribute;
use crate::ir_types::Type;
use crate::ir_types::{FloatKind, FloatType};
use crate::op::{OpAttributes, OpMarker, OpName, Operands, Results};
use crate::value::ValueKind;
use crate::value::{Uses, ValueDef, ValueType};

// ── OpRewrite ────────────────────────────────────────────────────────────────

/// A single rewrite action produced by matching a [`RewritePattern`].
///
/// Multiple actions are batched into a `SmallVec` and applied as an atomic
/// group by the [`PatternRewriter`] worklist driver.
#[derive(Debug, Clone)]
pub enum OpRewrite {
    /// Replace `target` op with one or more new ops.
    ///
    /// The new ops' definitions are given as (`name`, `operands`,
    /// `attributes`, `result_types`) tuples.
    ReplaceOp {
        /// Entity of the operation to replace.
        target: Entity,
        /// Descriptions of the replacement operations.
        replacements: SmallVec<[NewOpDesc; 4]>,
    },
    /// Erase `target` op from the IR.
    EraseOp {
        /// Entity of the operation to erase.
        target: Entity,
    },
    /// Create a new operation at a logical insertion point.
    CreateOp {
        /// Description of the new operation.
        desc: NewOpDesc,
        /// Block to insert after this entity. When `None`, appends to
        /// the insertion block default.
        after: Option<Entity>,
    },
}

/// Describes a single new operation to create.
#[derive(Debug, Clone)]
pub struct NewOpDesc {
    /// Operation name (e.g. `"arith.addf"`).
    pub name: &'static str,
    /// Operand value entities.
    pub operands: Vec<Entity>,
    /// Attribute list.
    pub attributes: Vec<Attribute>,
    /// Result types.
    pub result_types: Vec<Type>,
}

impl NewOpDesc {
    /// Create a new operation description.
    pub fn new(
        name: &'static str,
        operands: Vec<Entity>,
        attributes: Vec<Attribute>,
        result_types: Vec<Type>,
    ) -> Self {
        Self {
            name,
            operands,
            attributes,
            result_types,
        }
    }
}

// ── RewritePattern trait ─────────────────────────────────────────────────────

/// A rewrite pattern using the match-return model.
///
/// Unlike the stateful `RewritePattern` in `rewrite_driver`, this trait
/// returns actions declaratively. The [`PatternRewriter`] applies them.
pub trait RewritePattern: Send + Sync {
    /// Attempt to match `root_op` and, on success, return one or more
    /// [`OpRewrite`] actions.
    ///
    /// Return `None` when the pattern does not apply.
    fn match_op(
        &self,
        root_op: Entity,
        world: &World,
    ) -> Result<Option<SmallVec<[OpRewrite; 4]>>, String>;
}

// ── FoldableOp trait ─────────────────────────────────────────────────────────

/// An operation that can be folded to a constant attribute value.
///
/// Mirrors MLIR's `Op::fold()` hook. When all operand values can be
/// resolved to constant attributes, producing a single constant result
/// via `fold()` lets the pattern rewriter bypass the op entirely.
pub trait FoldableOp {
    /// Fold this op given its constant-folded operands.
    ///
    /// Returns `Some(Attribute)` when the op can be fully folded to a
    /// constant result, or `None` when folding is not possible.
    fn fold(&self, operands: &[Attribute]) -> Option<Vec<Attribute>>;
}

// ── PatternRewriter (worklist driver) ────────────────────────────────────────

/// Applies [`RewritePattern`]s in a worklist-driven fixpoint loop.
///
/// The rewriter drives canonicalization (pattern application), constant
/// folding, and limited CSE (common subexpression elimination) to
/// convergence.
pub struct PatternRewriter {
    patterns: Vec<Box<dyn RewritePattern>>,
    foldable_ops: Vec<Box<dyn FoldableOp>>,
    max_iterations: usize,
}

impl PatternRewriter {
    /// Create a new rewriter with default settings.
    pub fn new() -> Self {
        Self {
            patterns: Vec::new(),
            foldable_ops: Vec::new(),
            max_iterations: 10,
        }
    }

    /// Register a rewrite pattern.
    pub fn add_pattern(&mut self, pattern: Box<dyn RewritePattern>) {
        self.patterns.push(pattern);
    }

    /// Register a foldable operation descriptor.
    ///
    /// TODO: In a full ODS-style implementation this would be generated
    /// from op definitions. For now, foldable ops are registered manually.
    pub fn add_foldable_op(&mut self, foldable: Box<dyn FoldableOp>) {
        self.foldable_ops.push(foldable);
    }

    /// Set the maximum number of fixpoint iterations.
    pub fn set_max_iterations(&mut self, max: usize) {
        self.max_iterations = max;
    }

    /// Run the rewriter to fixpoint on `world`.
    ///
    /// The worklist loop:
    /// 1. Collect all ops from the world into a worklist.
    /// 2. For each op, try every registered pattern.
    /// 3. Apply the returned actions and add affected ops back to the
    ///    worklist.
    /// 4. Attempt constant folding on ops whose operands are all constants.
    /// 5. Repeat until no changes or max_iterations reached.
    ///
    /// Returns the total number of rewrites applied.
    pub fn apply_all(&mut self, world: &mut World) -> Result<u64, String> {
        let mut total_applied: u64 = 0;

        for iteration in 0..self.max_iterations {
            let mut any_changed = false;

            // Collect the initial worklist: all entites with OpMarker.
            let worklist: Vec<Entity> = world
                .query::<OpMarker>()
                .map(|(entity, _)| entity)
                .collect();

            for &op in &worklist {
                // Skip erased ops (no longer exist in world).
                if world.get_component::<OpMarker>(op).is_none() {
                    continue;
                }

                // Try constant folding first.
                if let Some(folded) = self.try_fold_op(world, op)? {
                    Self::do_replace(world, op, &folded)?;
                    any_changed = true;
                    total_applied += 1;
                    continue;
                }

                // Try each pattern.
                for pattern in &self.patterns {
                    let actions = pattern.match_op(op, world)?;
                    if let Some(actions) = actions {
                        for action in actions {
                            Self::apply_action(world, action)?;
                        }
                        any_changed = true;
                        total_applied += 1;
                        break; // only one pattern per op per iteration
                    }
                }
            }

            // Try CSE pass: dedup ops with identical structure.
            if self.try_cse(world)? {
                any_changed = true;
                total_applied += 1;
            }

            if !any_changed {
                // Fixpoint reached.
                break;
            }

            // Safety valve: last iteration may leave some unconverged
            // patterns untouched, which is acceptable.
            if iteration == self.max_iterations - 1 {
                eprintln!(
                    "[prism-ecs-ir] PatternRewriter: fixpoint not reached after {} iterations",
                    self.max_iterations
                );
            }
        }

        Ok(total_applied)
    }

    // ── Single action application ──────────────────────────────────────────

    /// Apply a single [`OpRewrite`] action to the world.
    fn apply_action(world: &mut World, action: OpRewrite) -> Result<(), String> {
        match action {
            OpRewrite::ReplaceOp {
                target,
                replacements,
            } => Self::do_replace(world, target, &replacements),
            OpRewrite::EraseOp { target } => Self::do_erase(world, target),
            OpRewrite::CreateOp { desc, after } => Self::do_create(world, &desc, after),
        }
    }

    /// Replace `target` with one or more new ops.
    fn do_replace(
        world: &mut World,
        target: Entity,
        replacements: &[NewOpDesc],
    ) -> Result<(), String> {
        // Record all uses of target's results before we erase.
        let old_results: Vec<Entity> = world
            .get_component::<Results>(target)
            .map(|r| r.0.clone())
            .ok_or_else(|| format!("replace target {:?} has no Results", target))?;

        // Create replacement ops.
        let mut new_ops: Vec<Entity> = Vec::with_capacity(replacements.len());
        for desc in replacements {
            let entity = Self::build_op_in_world(world, desc)?;
            new_ops.push(entity);
        }

        // Redirect all uses of old results to the first replacement's results.
        if let Some(first_new) = new_ops.first() {
            let new_results: Vec<Entity> = world
                .get_component::<Results>(*first_new)
                .map(|r| r.0.clone())
                .unwrap_or_default();

            for (i, old_val) in old_results.iter().enumerate() {
                if let Some(new_val) = new_results.get(i) {
                    let uses = world
                        .get_component::<Uses>(*old_val)
                        .map(|u| u.0.clone())
                        .unwrap_or_default();
                    for user in &uses {
                        if let Some(operands) = world.get_component_mut::<Operands>(*user) {
                            for opnd in &mut operands.0 {
                                if *opnd == *old_val {
                                    *opnd = *new_val;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Swap target in its block with the new ops.
        Self::swap_in_block(world, target, &new_ops)?;

        // Despawn the old target.
        if !world.despawn(target) {
            return Err(format!("despawn failed for {:?}", target));
        }

        Ok(())
    }

    /// Erase `target` from its block and despawn it.
    fn do_erase(world: &mut World, target: Entity) -> Result<(), String> {
        // Remove from block's op list.
        Self::remove_from_block(world, target)?;

        // Despawn the entity and all its values.
        if !world.despawn(target) {
            return Err(format!("despawn failed for {:?}", target));
        }
        Ok(())
    }
    /// Create a new operation in the world.
    fn do_create(world: &mut World, desc: &NewOpDesc, after: Option<Entity>) -> Result<(), String> {
        let entity = Self::build_op_in_world(world, desc)?;

        // Insert into block.
        if let Some(anchor) = after {
            if let Some(bops) = world.get_component_mut::<BlockOps>(anchor) {
                let pos = bops
                    .0
                    .iter()
                    .position(|e| *e == anchor)
                    .map(|p| p + 1)
                    .unwrap_or(bops.0.len());
                bops.0.insert(pos, entity);
            }
        }

        Ok(())
    }

    // ── Internal helpers ───────────────────────────────────────────────────

    /// Build an op entity in the world from a NewOpDesc.
    fn build_op_in_world(world: &mut World, desc: &NewOpDesc) -> Result<Entity, String> {
        let entity: Entity = world
            .spawn(
                prism_ecs_core::EntityKind::Node,
                Some(format!("op_{}", desc.name)),
            )
            .map_err(|e| format!("spawn failed: {e}"))?
            .into();

        world
            .add_component(entity, OpMarker)
            .map_err(|e| format!("add OpMarker: {e}"))?;
        world
            .add_component(entity, OpName(desc.name.to_string()))
            .map_err(|e| format!("add OpName: {e}"))?;
        world
            .add_component(entity, Operands(desc.operands.clone()))
            .map_err(|e| format!("add Operands: {e}"))?;
        world
            .add_component(entity, OpAttributes(desc.attributes.clone()))
            .map_err(|e| format!("add OpAttributes: {e}"))?;

        // Create result Value entities.
        let mut result_entities = Vec::new();
        for (i, ty) in desc.result_types.iter().enumerate() {
            let val: Entity = world
                .spawn(
                    prism_ecs_core::EntityKind::Node,
                    Some(format!("{}.r{}", desc.name, i)),
                )
                .map_err(|e| format!("spawn result value: {e}"))?
                .into();
            world
                .add_component(val, ValueDef::op_result(entity, i as u32))
                .map_err(|e| format!("add ValueDef: {e}"))?;
            world
                .add_component(val, ValueType(ty.clone()))
                .map_err(|e| format!("add ValueType: {e}"))?;
            world
                .add_component(val, Uses(vec![]))
                .map_err(|e| format!("add Uses: {e}"))?;
            result_entities.push(val);
        }
        world
            .add_component(entity, Results(result_entities))
            .map_err(|e| format!("add Results: {e}"))?;

        // Wire operand uses.
        for &opnd in &desc.operands {
            if let Some(uses) = world.get_component_mut::<Uses>(opnd) {
                uses.0.push(entity);
            }
        }

        Ok(entity)
    }

    /// Replace `target` in its block with `new_ops`.
    fn swap_in_block(world: &mut World, target: Entity, new_ops: &[Entity]) -> Result<(), String> {
        for (_block, bops) in world.query_mut::<BlockOps>() {
            if let Some(pos) = bops.0.iter().position(|e| *e == target) {
                bops.0.remove(pos);
                for (offset, &new_op) in new_ops.iter().enumerate() {
                    bops.0.insert(pos + offset, new_op);
                }
                return Ok(());
            }
        }
        // Not in any block (e.g. floating ops) — fine.
        Ok(())
    }

    /// Remove `target` from its containing block.
    fn remove_from_block(world: &mut World, target: Entity) -> Result<(), String> {
        for (_block, bops) in world.query_mut::<BlockOps>() {
            bops.0.retain(|e| *e != target);
        }
        Ok(())
    }

    // ── Constant folding ──────────────────────────────────────────────────

    /// Try to fold `op` to a constant. Returns replacement op descriptions
    /// when folding succeeds, or `None`.
    fn try_fold_op(
        &self,
        world: &World,
        op: Entity,
    ) -> Result<Option<SmallVec<[NewOpDesc; 4]>>, String> {
        // Resolve all operands to attributes if they are constant-foldable.
        let operands = world
            .get_component::<Operands>(op)
            .map(|o| o.0.clone())
            .unwrap_or_default();

        let mut folded_operands: Vec<Attribute> = Vec::new();
        for &opnd in &operands {
            if let Some(val_def) = world.get_component::<ValueDef>(opnd) {
                if val_def.kind == ValueKind::OpResult {
                    let def_op = val_def.defining_entity;
                    let attr = world.get_component::<OpAttributes>(def_op).and_then(|a| {
                        // Check if this is a constant-like op with a single "value" attribute.
                        a.0.iter()
                            .find(|attr| {
                                matches!(attr, Attribute::Float(..) | Attribute::Integer(..))
                            })
                            .cloned()
                    });
                    if let Some(attr) = attr {
                        folded_operands.push(attr);
                    } else {
                        // Non-foldable operand.
                        return Ok(None);
                    }
                } else {
                    // Block argument — not foldable.
                    return Ok(None);
                }
            } else {
                return Ok(None);
            }
        }

        // Try each registered foldable op for a match.
        for foldable in &self.foldable_ops {
            if let Some(result_attrs) = foldable.fold(&folded_operands) {
                let result_types: Vec<Type> = result_attrs
                    .iter()
                    .map(|_a| {
                        Type::Float(FloatType {
                            kind: FloatKind::F32,
                        })
                    }) // Default
                    .collect();
                let mut replacements: SmallVec<[NewOpDesc; 4]> = SmallVec::new();
                replacements.push(NewOpDesc {
                    name: "arith.constant",
                    operands: vec![],
                    attributes: result_attrs,
                    result_types,
                });
                return Ok(Some(replacements));
            }
        }

        Ok(None)
    }

    // ── CSE ─────────────────────────────────────────────────────────────────

    /// Try common subexpression elimination.
    ///
    /// Deduplicates ops that have the same name, operands, and attributes.
    /// Currently a simple linear scan; a production CSE would hash
    /// structure.
    fn try_cse(&mut self, world: &mut World) -> Result<bool, String> {
        let mut changed = false;

        let ops: Vec<Entity> = world.query::<OpMarker>().map(|(e, _)| e).collect();

        for i in 0..ops.len() {
            let a = ops[i];
            if world.get_component::<OpMarker>(a).is_none() {
                continue;
            }

            let a_name = match world.get_component::<OpName>(a) {
                Some(n) => n.0.clone(),
                None => continue,
            };
            let a_operands: Vec<Entity> = world
                .get_component::<Operands>(a)
                .map(|o| o.0.clone())
                .unwrap_or_default();
            let a_attrs: Vec<Attribute> = world
                .get_component::<OpAttributes>(a)
                .map(|a| a.0.clone())
                .unwrap_or_default();
            let a_results: Vec<Entity> = world
                .get_component::<Results>(a)
                .map(|r| r.0.clone())
                .unwrap_or_default();

            for j in (i + 1)..ops.len() {
                let b = ops[j];
                if world.get_component::<OpMarker>(b).is_none() {
                    continue;
                }

                let b_name = match world.get_component::<OpName>(b) {
                    Some(n) => n.0.clone(),
                    None => continue,
                };
                if a_name != b_name {
                    continue;
                }

                let b_operands: Vec<Entity> = world
                    .get_component::<Operands>(b)
                    .map(|o| o.0.clone())
                    .unwrap_or_default();
                if a_operands != b_operands {
                    continue;
                }

                let b_attrs: Vec<Attribute> = world
                    .get_component::<OpAttributes>(b)
                    .map(|a| a.0.clone())
                    .unwrap_or_default();
                if a_attrs != b_attrs {
                    continue;
                }

                let b_results: Vec<Entity> = world
                    .get_component::<Results>(b)
                    .map(|r| r.0.clone())
                    .unwrap_or_default();
                if a_results.len() != b_results.len() {
                    continue;
                }

                // Dedup: replace all uses of b's results with a's results.
                for (k, b_val) in b_results.iter().enumerate() {
                    if let Some(a_val) = a_results.get(k) {
                        let uses = world
                            .get_component::<Uses>(*b_val)
                            .map(|u| u.0.clone())
                            .unwrap_or_default();
                        for user in &uses {
                            if let Some(operands) = world.get_component_mut::<Operands>(*user) {
                                for opnd in &mut operands.0 {
                                    if *opnd == *b_val {
                                        *opnd = *a_val;
                                    }
                                }
                            }
                        }
                    }
                }

                // Erase b.
                Self::remove_from_block(world, b)?;
                let _ = world.despawn(b);
                changed = true;
            }
        }

        Ok(changed)
    }
}

impl Default for PatternRewriter {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ecs_core::{EntityKind, World};

    /// A pattern that replaces `test.replaceme` with `test.replacement`.
    struct ReplacePattern;

    impl RewritePattern for ReplacePattern {
        fn match_op(
            &self,
            root_op: Entity,
            world: &World,
        ) -> Result<Option<SmallVec<[OpRewrite; 4]>>, String> {
            let name = world
                .get_component::<OpName>(root_op)
                .map(|n| n.0.as_str() == "test.replaceme")
                .unwrap_or(false);
            if !name {
                return Ok(None);
            }

            let operands = world
                .get_component::<Operands>(root_op)
                .map(|o| o.0.clone())
                .unwrap_or_default();
            let attrs = world
                .get_component::<OpAttributes>(root_op)
                .map(|a| a.0.clone())
                .unwrap_or_default();
            let result_types = world
                .get_component::<Results>(root_op)
                .map(|r| {
                    r.0.iter()
                        .filter_map(|v| world.get_component::<ValueType>(*v).map(|t| t.0.clone()))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();

            let mut actions: SmallVec<[OpRewrite; 4]> = SmallVec::new();
            actions.push(OpRewrite::ReplaceOp {
                target: root_op,
                replacements: smallvec::smallvec![NewOpDesc {
                    name: "test.replacement",
                    operands,
                    attributes: attrs,
                    result_types,
                }],
            });
            Ok(Some(actions))
        }
    }

    #[test]
    fn pattern_rewriter_replace() {
        let mut world = World::new();
        let mut rewriter = PatternRewriter::new();
        rewriter.add_pattern(Box::new(ReplacePattern));

        // Create an op to replace.
        let op: Entity = world
            .spawn(EntityKind::Node, Some("op_replaceme".into()))
            .unwrap()
            .into();
        world.add_component(op, OpMarker).unwrap();
        world
            .add_component(op, OpName("test.replaceme".into()))
            .unwrap();
        world.add_component(op, Operands(vec![])).unwrap();
        world.add_component(op, OpAttributes(vec![])).unwrap();
        let val: Entity = world
            .spawn(EntityKind::Node, Some("r0".into()))
            .unwrap()
            .into();
        world
            .add_component(val, ValueDef::op_result(op, 0))
            .unwrap();
        world
            .add_component(val, ValueType(Type::Float(FloatType::new(FloatKind::F32))))
            .unwrap();
        world.add_component(val, Uses(vec![])).unwrap();
        world.add_component(op, Results(vec![val])).unwrap();

        let count = rewriter.apply_all(&mut world).unwrap();
        assert!(count > 0, "rewriter should have applied the pattern");

        // The old op should be gone.
        assert!(
            world.get_component::<OpName>(op).is_none(),
            "old op should be despawned"
        );

        // A new op with the replacement name should exist.
        let has_replacement = world
            .query::<OpName>()
            .any(|(_, n)| n.0 == "test.replacement");
        assert!(has_replacement, "replacement op should exist");
    }

    #[test]
    fn pattern_rewriter_no_match() {
        let mut world = World::new();
        let mut rewriter = PatternRewriter::new();
        rewriter.add_pattern(Box::new(ReplacePattern));

        let op: Entity = world
            .spawn(EntityKind::Node, Some("op_other".into()))
            .unwrap()
            .into();
        world.add_component(op, OpMarker).unwrap();
        world
            .add_component(op, OpName("test.other".into()))
            .unwrap();

        let count = rewriter.apply_all(&mut world).unwrap();
        assert_eq!(count, 0, "no pattern should match");
    }

    #[test]
    fn pattern_rewriter_erase() {
        struct ErasePattern;

        impl RewritePattern for ErasePattern {
            fn match_op(
                &self,
                root_op: Entity,
                world: &World,
            ) -> Result<Option<SmallVec<[OpRewrite; 4]>>, String> {
                let name = world
                    .get_component::<OpName>(root_op)
                    .map(|n| n.0.as_str() == "test.dead")
                    .unwrap_or(false);
                if !name {
                    return Ok(None);
                }
                Ok(Some(smallvec::smallvec![OpRewrite::EraseOp {
                    target: root_op,
                }]))
            }
        }

        let mut world = World::new();
        let mut rewriter = PatternRewriter::new();
        rewriter.add_pattern(Box::new(ErasePattern));

        let op: Entity = world
            .spawn(EntityKind::Node, Some("op_dead".into()))
            .unwrap()
            .into();
        world.add_component(op, OpMarker).unwrap();
        world.add_component(op, OpName("test.dead".into())).unwrap();

        let count = rewriter.apply_all(&mut world).unwrap();
        assert!(count > 0);
        assert!(
            world.get_component::<OpMarker>(op).is_none(),
            "erased op should be gone"
        );
    }
}
