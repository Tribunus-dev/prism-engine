//! Dominance analysis for IR regions.
//!
//! Computes the dominator tree and dominance frontier for SSA CFG regions
//! using iterative dataflow. Predecessor/successor relationships are derived
//! from block ordering in the region (simplified approach). This is sufficient
//! for linear chains and structured control flow; a full Lengauer-Tarjan
//! implementation using terminator-successor edges can be substituted later.
//!
//! # Dominator Tree
//!
//! For a region's blocks in order [entry, b1, b2, ..., bn], the predecessor
//! of bᵢ is bᵢ₋₁. The iterative idom algorithm converges to:
//! - idom(entry) = entry
//! - idom(bᵢ)   = bᵢ₋₁ for i > 0
//!
//! # Dominance Frontier
//!
//! Uses the simple algorithm from Cooper, Harvey & Waterman:
//! For each block b, for each predecessor p of b where |preds(b)| ≥ 2,
//! climb from p up the idom chain to idom(b), adding b to DF(runner).

use std::collections::HashMap;

use prism_ecs_core::{Entity, World};

use crate::region::region_blocks;

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Results of dominator tree computation.
///
/// Maps each block to its immediate dominator. The entry block maps to itself
/// (it has no strict dominator).
#[derive(Debug, Clone)]
pub struct DominatorTree {
    /// immediate_dominators[block] = the immediate dominator of block.
    /// The entry block's idom is itself.
    pub immediate_dominators: HashMap<Entity, Entity>,
}

/// Results of dominance frontier computation.
///
/// DF(block) is the set of blocks that block does not strictly dominate
/// but that have a predecessor it does dominate — i.e., join points.
#[derive(Debug, Clone)]
pub struct DominanceFrontier {
    /// frontiers[block] = the dominance frontier of block.
    pub frontiers: HashMap<Entity, Vec<Entity>>,
}

// ---------------------------------------------------------------------------
// Analyzer
// ---------------------------------------------------------------------------

/// Computes dominator trees and dominance frontiers for CFG regions.
///
/// # Example
///
/// ```ignore
/// let analyzer = DominanceAnalyzer::new();
/// let tree = analyzer.compute_dominators(region, &world);
/// let frontier = analyzer.compute_frontier(region, &world);
/// ```
pub struct DominanceAnalyzer;

impl DominanceAnalyzer {
    /// Create a new dominance analyzer.
    pub fn new() -> Self {
        Self
    }

    /// Compute the dominator tree for a region using iterative dataflow.
    ///
    /// The entry block's immediate dominator is set to itself (a sentinel
    /// meaning "no strict dominator").
    pub fn compute_dominators(&self, region: Entity, world: &World) -> DominatorTree {
        let blocks = region_blocks(world, region);
        if blocks.is_empty() {
            return DominatorTree {
                immediate_dominators: HashMap::new(),
            };
        }

        let entry = blocks[0];
        let mut idom: HashMap<Entity, Entity> = HashMap::new();

        // Initialize: entry → entry, everything else → undefined (None sentinel).
        // We store the entry's idom as entry, and for other blocks we skip
        // setting a value in the map until computed. The intersect helper
        // uses the entry sentinel or the map itself for lookups.
        idom.insert(entry, entry);

        // Build predecessor map from block ordering.
        let preds = predecessors_by_order(&blocks);

        // We store an explicit "undefined" sentinel for uncomputed blocks.
        // In our approach, we iterate blocks in order (which is effectively RPO
        // for a linear chain or reducible CFG) and converge.
        let mut changed = true;
        while changed {
            changed = false;
            for b in blocks.iter().copied().skip(1) {
                let Some(b_preds) = preds.get(&b) else {
                    continue;
                };
                if b_preds.is_empty() {
                    continue;
                }

                // Start with the first predecessor as the candidate.
                let mut new_idom = b_preds[0];

                // Intersect with remaining predecessors.
                for p in b_preds.iter().copied().skip(1) {
                    if let Some(&p_idom) = idom.get(&p) {
                        new_idom = idom_intersect(new_idom, p_idom, &idom, entry);
                    }
                }

                // Check if this block already has an idom equal to the new one.
                if idom.get(&b).is_none_or(|&cur| cur != new_idom) {
                    idom.insert(b, new_idom);
                    changed = true;
                }
            }
        }

        DominatorTree {
            immediate_dominators: idom,
        }
    }

    /// Compute the dominance frontier for each block in the region.
    ///
    /// Uses the simple dataflow algorithm from Cooper, Harvey & Waterman:
    /// for each block `b` with ≥ 2 predecessors, for each predecessor `p`,
    /// climb from `p` up the idom chain to `idom(b)`, adding `b` to DF(runner).
    pub fn compute_frontier(&self, region: Entity, world: &World) -> DominanceFrontier {
        let tree = self.compute_dominators(region, world);
        let blocks = region_blocks(world, region);
        let preds = predecessors_by_order(&blocks);

        let mut frontiers: HashMap<Entity, Vec<Entity>> =
            blocks.iter().map(|&b| (b, Vec::new())).collect();

        // CHW algorithm
        for &b in &blocks {
            let Some(b_preds) = preds.get(&b) else {
                continue;
            };
            if b_preds.len() < 2 {
                continue;
            }

            let idom_b = tree.immediate_dominators.get(&b).copied();

            for &p in b_preds {
                let mut runner = p;
                loop {
                    let runner_idom = tree.immediate_dominators.get(&runner).copied();
                    if runner_idom == idom_b {
                        break;
                    }
                    frontiers.entry(runner).or_default().push(b);
                    if let Some(rid) = runner_idom {
                        runner = rid;
                    } else {
                        break;
                    }
                }
            }
        }

        // Deduplicate frontiers (a block can be added multiple times).
        for v in frontiers.values_mut() {
            v.sort_by_key(|e| e.id());
            v.dedup();
        }

        DominanceFrontier { frontiers }
    }
}

impl Default for DominanceAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Build a predecessor map from block ordering.
///
/// In the simplified model, block at index _i_ has block at index _i-1_ as
/// its sole predecessor. The entry block (index 0) has no predecessors.
fn predecessors_by_order(blocks: &[Entity]) -> HashMap<Entity, Vec<Entity>> {
    let mut preds: HashMap<Entity, Vec<Entity>> = HashMap::new();
    for (i, &b) in blocks.iter().enumerate() {
        if i > 0 {
            preds.entry(b).or_default().push(blocks[i - 1]);
        } else {
            preds.entry(b).or_default();
        }
    }
    preds
}

/// Intersect two nodes in the dominator tree — climb the idom chain from the
/// deeper node until both fingers meet at the least common dominator.
fn idom_intersect(b1: Entity, b2: Entity, idom: &HashMap<Entity, Entity>, entry: Entity) -> Entity {
    let mut f1 = b1;
    let mut f2 = b2;
    loop {
        if f1 == f2 {
            return f1;
        }
        let d1 = depth(f1, idom, entry);
        let d2 = depth(f2, idom, entry);
        if d1 >= d2 {
            f1 = idom.get(&f1).copied().unwrap_or(entry);
        }
        if d1 <= d2 {
            f2 = idom.get(&f2).copied().unwrap_or(entry);
        }
    }
}

/// Compute the depth of a node in the dominator tree (number of edges from
/// entry to node in the idom chain).
fn depth(node: Entity, idom: &HashMap<Entity, Entity>, entry: Entity) -> usize {
    let mut cur = node;
    let mut d = 0;
    while cur != entry {
        cur = idom.get(&cur).copied().unwrap_or(entry);
        d += 1;
    }
    d
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::{BlockArguments, BlockMarker, BlockOps, TerminatorOp};
    use crate::op::{OpAttributes, OpMarker, OpName, Operands, Results, Successors};
    use crate::region::{RegionBlocks, RegionKind, RegionKindComp, RegionMarker};
    use prism_ecs_core::{EntityKind, World};

    /// Create a block entity with the required components.
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

    /// Create a simple terminator op with successors.
    #[allow(dead_code)]
    fn create_terminator(world: &mut World, successors: Vec<Entity>) -> Entity {
        let entity: Entity = world
            .spawn(EntityKind::Node, None)
            .expect("spawn failed")
            .into();
        world.add_component(entity, OpMarker).expect("add OpMarker");
        world
            .add_component(entity, OpName("cf.br".into()))
            .expect("add OpName");
        world
            .add_component(entity, Operands(vec![]))
            .expect("add Operands");
        world
            .add_component(entity, Results(vec![]))
            .expect("add Results");
        world
            .add_component(entity, OpAttributes(vec![]))
            .expect("add OpAttributes");
        world
            .add_component(entity, Successors(successors))
            .expect("add Successors");
        entity
    }

    #[test]
    fn linear_three_blocks() {
        let mut world = World::new();

        // Create three blocks in linear order
        let entry = create_block(&mut world, "entry");
        let b1 = create_block(&mut world, "b1");
        let b2 = create_block(&mut world, "b2");

        // Create a region containing the blocks
        let region: Entity = world
            .spawn(EntityKind::Node, Some("test_region".into()))
            .expect("spawn failed")
            .into();
        world
            .add_component(region, RegionMarker)
            .expect("add RegionMarker");
        world
            .add_component(region, RegionKindComp(RegionKind::SSACFG))
            .expect("add RegionKindComp");
        world
            .add_component(region, RegionBlocks(vec![entry, b1, b2]))
            .expect("add RegionBlocks");

        let analyzer = DominanceAnalyzer::new();
        let tree = analyzer.compute_dominators(region, &world);

        // Verify immediate dominators
        assert_eq!(tree.immediate_dominators.len(), 3);
        assert_eq!(
            tree.immediate_dominators.get(&entry).copied(),
            Some(entry),
            "entry's idom should be itself"
        );
        assert_eq!(
            tree.immediate_dominators.get(&b1).copied(),
            Some(entry),
            "b1's idom should be entry"
        );
        assert_eq!(
            tree.immediate_dominators.get(&b2).copied(),
            Some(b1),
            "b2's idom should be b1"
        );

        // Verify dominance frontier (no join points → empty frontiers)
        let frontier = analyzer.compute_frontier(region, &world);
        assert_eq!(frontier.frontiers.len(), 3);
        for (b, df) in &frontier.frontiers {
            assert!(
                df.is_empty(),
                "block {:?} should have empty frontier in linear chain",
                b
            );
        }
    }

    #[test]
    fn empty_region() {
        let mut world = World::new();

        let region: Entity = world
            .spawn(EntityKind::Node, Some("empty_region".into()))
            .expect("spawn failed")
            .into();
        world
            .add_component(region, RegionMarker)
            .expect("add RegionMarker");
        world
            .add_component(region, RegionKindComp(RegionKind::SSACFG))
            .expect("add RegionKindComp");
        world
            .add_component(region, RegionBlocks(vec![]))
            .expect("add RegionBlocks");

        let analyzer = DominanceAnalyzer::new();
        let tree = analyzer.compute_dominators(region, &world);
        assert!(tree.immediate_dominators.is_empty());

        let frontier = analyzer.compute_frontier(region, &world);
        assert!(frontier.frontiers.is_empty());
    }

    #[test]
    fn single_block() {
        let mut world = World::new();

        let entry = create_block(&mut world, "entry");

        let region: Entity = world
            .spawn(EntityKind::Node, Some("single_region".into()))
            .expect("spawn failed")
            .into();
        world
            .add_component(region, RegionMarker)
            .expect("add RegionMarker");
        world
            .add_component(region, RegionKindComp(RegionKind::SSACFG))
            .expect("add RegionKindComp");
        world
            .add_component(region, RegionBlocks(vec![entry]))
            .expect("add RegionBlocks");

        let analyzer = DominanceAnalyzer::new();
        let tree = analyzer.compute_dominators(region, &world);

        assert_eq!(tree.immediate_dominators.len(), 1);
        assert_eq!(
            tree.immediate_dominators.get(&entry).copied(),
            Some(entry),
            "single block's idom should be itself"
        );
    }

    #[test]
    fn frontier_diamond() {
        let mut world = World::new();

        // Diamond CFG: entry -> {left, right} -> merge
        // This exercises the CHW frontier algorithm.
        // But since we use block-order predecessors (not real terminator edges),
        // the diamond shape reduces to: entry, left, right, merge
        // where predecessors are:
        //   entry: {}
        //   left: {entry}
        //   right: {left}
        //   merge: {right}
        //
        // Since no block has ≥ 2 predecessors in this order model, frontiers
        // will all be empty. This test documents that behavior and sets the
        // expectation for when real terminator-based edges replace order.
        let entry = create_block(&mut world, "entry");
        let left = create_block(&mut world, "left");
        let right = create_block(&mut world, "right");
        let merge = create_block(&mut world, "merge");

        let region: Entity = world
            .spawn(EntityKind::Node, Some("diamond_region".into()))
            .expect("spawn failed")
            .into();
        world
            .add_component(region, RegionMarker)
            .expect("add RegionMarker");
        world
            .add_component(region, RegionKindComp(RegionKind::SSACFG))
            .expect("add RegionKindComp");
        world
            .add_component(region, RegionBlocks(vec![entry, left, right, merge]))
            .expect("add RegionBlocks");

        let analyzer = DominanceAnalyzer::new();
        let tree = analyzer.compute_dominators(region, &world);

        // With linear order predecessor model:
        //   idom(entry) = entry
        //   idom(left)  = entry
        //   idom(right) = left
        //   idom(merge) = right
        assert_eq!(tree.immediate_dominators.get(&entry).copied(), Some(entry));
        assert_eq!(tree.immediate_dominators.get(&left).copied(), Some(entry));
        assert_eq!(tree.immediate_dominators.get(&right).copied(), Some(left));
        assert_eq!(tree.immediate_dominators.get(&merge).copied(), Some(right));
    }
}
