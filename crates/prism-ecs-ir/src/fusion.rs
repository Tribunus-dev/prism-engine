//! Fusion/partition system — dataflow analysis for op fusion.
//!
//! Analyzes data dependencies between operations (linalg.matmul + surrounding
//! arith ops) and creates [`FusionGroup`] components that mark which ops
//! should be fused together into one kernel.
//!
//! The fusion strategy is controlled by an optional [`FusionPolicy`] parameter
//! — see [`FusionStrategy`] for the available modes.  When no policy is given
//! the default is [`FusionStrategy::Aggressive`] (full graph transitive fusion).
//!
//! # Usage
//!
//! ```ignore
//! let groups = analyze_dataflow(&world, root_op, None);
//! let groups = analyze_dataflow(&world, root_op, Some(FusionPolicy(FusionStrategy::Conservative)));
//! let count = partition_fusion_groups(&mut world, root_op, None)?;
//! ```
//!
//! After partitioning, each fused op carries a [`FusionGroup`] component whose
//! vector contains every op in the same fusion group.

use std::collections::{HashSet, VecDeque};

use prism_ecs_core::{Component, Entity, World};

use crate::op::{operands, results};
use crate::value::{value_users, ValueDef};

// ── FusionGroup component ──────────────────────────────────────────────────

/// Marks a group of ops that should be fused together into one kernel.
///
/// Every op in a fusion group carries an identical copy of this component
/// listing every member of the group.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FusionGroup(pub Vec<Entity>);
impl Component for FusionGroup {}

// ── Fusion strategy ─────────────────────────────────────────────────────────

/// How aggressively to fuse operations into groups.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum FusionStrategy {
    /// No fusion — each op is returned in its own singleton group.
    None,
    /// Only fuse operations that are direct (depth-1) producers or consumers
    /// of the root op.  Transitive chains are not followed.
    Conservative,
    /// Full dataflow-connected-component fusion across all transitive
    /// dependencies (the default behaviour when no policy is supplied).
    Aggressive,
}

/// ECS component controlling the fusion strategy for dataflow analysis.
///
/// Attached to a CompilePlan entity to drive how [`analyze_dataflow`] and
/// [`partition_fusion_groups`] build operation groups.  When no `FusionPolicy`
/// is present the analysis defaults to [`FusionStrategy::Aggressive`].
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct FusionPolicy(pub FusionStrategy);
impl Component for FusionPolicy {}

// ── Dataflow analysis ──────────────────────────────────────────────────────

/// Walk the dataflow graph from `root_op` and collect connected ops into
/// fusion groups, controlled by an optional [`FusionPolicy`].
///
/// The behaviour per strategy:
///   - [`FusionStrategy::None`] — returns one singleton group containing only
///     `root_op`.
///   - [`FusionStrategy::Conservative`] — only direct (depth-1) producers and
///     consumers of `root_op` are included.  Transitive chains are not
///     followed beyond the immediate neighbours.
///   - [`FusionStrategy::Aggressive`] — full connected-component BFS through
///     all transitive data dependencies (the default when `policy` is `None`).
///
/// **Forward traversal** — for each op, finds the ops that consume its result
/// values (via [`Uses`]).
///
/// **Backward traversal** — for each op, finds the ops that produce its
/// operand values (via [`ValueDef`]).
///
/// Returns one group per connected component reachable from `root_op`.  For a
/// single chain of element-wise + matmul ops this typically returns one group
/// under aggressive fusion.
pub fn analyze_dataflow(
    world: &World,
    root_op: Entity,
    policy: Option<FusionPolicy>,
) -> Vec<Vec<Entity>> {
    let strategy = policy.map(|p| p.0).unwrap_or(FusionStrategy::Aggressive);

    match strategy {
        FusionStrategy::None => {
            // Every op is its own singleton group.
            vec![vec![root_op]]
        }

        FusionStrategy::Conservative => {
            // Only direct neighbours of root_op at depth 1.
            let mut visited: HashSet<Entity> = HashSet::new();
            let mut group: Vec<Entity> = Vec::new();

            visited.insert(root_op);
            group.push(root_op);

            // ── Forward: direct consumers of root_op's results ────────
            for val in results(world, root_op).iter() {
                for &consumer in value_users(world, *val).iter() {
                    if visited.insert(consumer) {
                        group.push(consumer);
                    }
                }
            }

            // ── Backward: direct producers of root_op's operands ──────
            for &val in operands(world, root_op).iter() {
                if let Some(vd) = world.get_component::<ValueDef>(val) {
                    if visited.insert(vd.defining_entity) {
                        group.push(vd.defining_entity);
                    }
                }
            }

            vec![group]
        }

        FusionStrategy::Aggressive => {
            // Full transitive-closure BFS (original behaviour).
            let mut visited: HashSet<Entity> = HashSet::new();
            let mut queue: VecDeque<Entity> = VecDeque::new();
            let mut component: Vec<Entity> = Vec::new();

            queue.push_back(root_op);
            visited.insert(root_op);

            while let Some(op) = queue.pop_front() {
                component.push(op);

                // ── Forward: follow Uses edges on every result ──────
                for val in results(world, op).iter() {
                    for &consumer in value_users(world, *val).iter() {
                        if visited.insert(consumer) {
                            queue.push_back(consumer);
                        }
                    }
                }

                // ── Backward: find producers of this op's operands ──
                for &val in operands(world, op).iter() {
                    if let Some(vd) = world.get_component::<ValueDef>(val) {
                        if visited.insert(vd.defining_entity) {
                            queue.push_back(vd.defining_entity);
                        }
                    }
                }
            }

            if component.is_empty() {
                vec![]
            } else {
                vec![component]
            }
        }
    }
}

// ── Partition ──────────────────────────────────────────────────────────────

/// Run dataflow analysis from `root_op` with the given optional
/// [`FusionPolicy`] and write [`FusionGroup`] components onto every op in
/// each discovered group.
///
/// Every op in a group receives a [`FusionGroup`] component whose vector
/// contains all members of that group.
///
/// Returns the number of fusion groups written, or an error if any component
/// could not be added.
pub fn partition_fusion_groups(
    world: &mut World,
    root_op: Entity,
    policy: Option<FusionPolicy>,
) -> Result<usize, String> {
    let groups = analyze_dataflow(world, root_op, policy);
    let count = groups.len();

    for group in &groups {
        let fg = FusionGroup(group.clone());
        for &op in group {
            world
                .add_component(op, fg.clone())
                .map_err(|e| format!("failed to add FusionGroup to {op:?}: {e}"))?;
        }
    }

    Ok(count)
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir_types::Type;
    use crate::op::{OpMarker, OpName, Operands, Results};
    use crate::value::{Uses, ValueDef, ValueType};

    /// Helper: create a value entity with the given producer op and index.
    fn make_value(world: &mut World, producer: Entity, idx: u32) -> Entity {
        let val: Entity = world
            .spawn(prism_ecs_core::EntityKind::Node, None)
            .unwrap()
            .into();
        world
            .add_component(val, ValueDef::op_result(producer, idx))
            .unwrap();
        world.add_component(val, ValueType(Type::f32())).unwrap();
        world.add_component(val, Uses(vec![])).unwrap();
        val
    }

    /// Helper: create an op entity with name, operands, and a result value.
    fn make_op(world: &mut World, name: &str, op_operands: &[Entity]) -> (Entity, Entity) {
        let op: Entity = world
            .spawn(prism_ecs_core::EntityKind::Node, None)
            .unwrap()
            .into();
        world.add_component(op, OpMarker).unwrap();
        world.add_component(op, OpName(name.to_string())).unwrap();
        world
            .add_component(op, Operands(op_operands.to_vec()))
            .unwrap();

        // Create one result value
        let val = make_value(world, op, 0);

        // Record the result on the op
        world.add_component(op, Results(vec![val])).unwrap();

        // Update Uses on operand values to include this op
        for &operand in op_operands {
            if let Some(u) = world.get_component_mut::<Uses>(operand) {
                u.0.push(op);
            }
        }

        (op, val)
    }

    // ── analyze_dataflow tests ───────────────────────────────────────────

    #[test]
    fn analyze_single_op() {
        let mut world = World::new();
        let (_op, _val) = make_op(&mut world, "linalg.matmul", &[]);

        let groups = analyze_dataflow(&world, _op, None);
        assert_eq!(groups.len(), 1, "should produce one group");
        assert_eq!(groups[0].len(), 1, "single op group has one member");
        assert!(groups[0].contains(&_op));
    }

    #[test]
    fn analyze_chain_forward() {
        let mut world = World::new();

        // addf → mulf → matmul
        let (addf_op, addf_val) = make_op(&mut world, "arith.addf", &[]);
        let (mulf_op, mulf_val) = make_op(&mut world, "arith.mulf", &[addf_val]);
        let (_matmul_op, _matmul_val) = make_op(&mut world, "linalg.matmul", &[mulf_val]);

        // analyze from the start of the chain
        let groups = analyze_dataflow(&world, addf_op, None);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].len(), 3, "all three ops in one group");
        assert!(groups[0].contains(&addf_op));
        assert!(groups[0].contains(&mulf_op));
        assert!(groups[0].contains(&_matmul_op));
    }

    #[test]
    fn analyze_chain_backward() {
        let mut world = World::new();

        // addf → mulf → matmul, but analyze from matmul
        let (addf_op, addf_val) = make_op(&mut world, "arith.addf", &[]);
        let (mulf_op, mulf_val) = make_op(&mut world, "arith.mulf", &[addf_val]);
        let (matmul_op, _matmul_val) = make_op(&mut world, "linalg.matmul", &[mulf_val]);

        let groups = analyze_dataflow(&world, matmul_op, None);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].len(), 3);
        assert!(groups[0].contains(&addf_op));
        assert!(groups[0].contains(&mulf_op));
        assert!(groups[0].contains(&matmul_op));
    }

    #[test]
    fn analyze_fan_in() {
        let mut world = World::new();

        // Two independent producers feed into one matmul
        let (addf_op, addf_val) = make_op(&mut world, "arith.addf", &[]);
        let (mulf_op, mulf_val) = make_op(&mut world, "arith.mulf", &[]);
        let (matmul_op, _matmul_val) = make_op(&mut world, "linalg.matmul", &[addf_val, mulf_val]);

        let groups = analyze_dataflow(&world, matmul_op, None);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].len(), 3);
        assert!(groups[0].contains(&addf_op));
        assert!(groups[0].contains(&mulf_op));
        assert!(groups[0].contains(&matmul_op));
    }

    #[test]
    fn analyze_fan_out() {
        let mut world = World::new();

        // One producer feeds two consumers
        let (addf_op, addf_val) = make_op(&mut world, "arith.addf", &[]);
        let (mulf_op, _mulf_val) = make_op(&mut world, "arith.mulf", &[addf_val]);
        let (divf_op, _divf_val) = make_op(&mut world, "arith.divf", &[addf_val]);

        let groups = analyze_dataflow(&world, addf_op, None);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].len(), 3);
        assert!(groups[0].contains(&addf_op));
        assert!(groups[0].contains(&mulf_op));
        assert!(groups[0].contains(&divf_op));
    }

    // ── FusionPolicy tests ───────────────────────────────────────────────

    #[test]
    fn none_policy_returns_singleton_groups() {
        let mut world = World::new();

        let (addf_op, addf_val) = make_op(&mut world, "arith.addf", &[]);
        let (mulf_op, mulf_val) = make_op(&mut world, "arith.mulf", &[addf_val]);
        let (matmul_op, _) = make_op(&mut world, "linalg.matmul", &[mulf_val]);

        let policy = Some(FusionPolicy(FusionStrategy::None));

        // Analyze from the middle of the chain — each op should be alone
        let groups = analyze_dataflow(&world, mulf_op, policy);
        assert_eq!(groups.len(), 1, "None policy returns one group");
        assert_eq!(groups[0].len(), 1, "None policy: group has exactly one op");
        assert_eq!(
            groups[0][0], mulf_op,
            "None policy: group contains the root op"
        );
    }

    #[test]
    fn conservative_policy_limits_to_direct_neighbours() {
        let mut world = World::new();

        // Chain: addf → mulf → matmul
        // When analyzing from the middle (mulf) with Conservative:
        //   - addf (direct producer) IS included
        //   - matmul (direct consumer) IS included
        let (addf_op, addf_val) = make_op(&mut world, "arith.addf", &[]);
        let (mulf_op, mulf_val) = make_op(&mut world, "arith.mulf", &[addf_val]);
        let (matmul_op, _) = make_op(&mut world, "linalg.matmul", &[mulf_val]);

        let policy = Some(FusionPolicy(FusionStrategy::Conservative));

        let groups = analyze_dataflow(&world, mulf_op, policy);
        assert_eq!(groups.len(), 1, "Conservative: one group");
        assert_eq!(
            groups[0].len(),
            3,
            "Conservative: root + direct producer + direct consumer"
        );
        assert!(
            groups[0].contains(&addf_op),
            "Conservative: includes direct producer (addf)"
        );
        assert!(
            groups[0].contains(&mulf_op),
            "Conservative: includes root (mulf)"
        );
        assert!(
            groups[0].contains(&matmul_op),
            "Conservative: includes direct consumer (matmul)"
        );
    }

    #[test]
    fn conservative_does_not_follow_transitive_chain() {
        let mut world = World::new();

        // Chain: subf → addf → mulf, and then addf also → divf → expf
        // Analyzing from addf with Conservative:
        //   - subf (direct producer) IS included
        //   - mulf (direct consumer) IS included
        //   - divf (direct consumer) IS included
        //   - BUT expf (transitive via divf) is NOT included
        let (subf_op, subf_val) = make_op(&mut world, "arith.subf", &[]);
        let (addf_op, addf_val) = make_op(&mut world, "arith.addf", &[subf_val]);
        let (mulf_op, _) = make_op(&mut world, "arith.mulf", &[addf_val]);
        let (divf_op, divf_val) = make_op(&mut world, "arith.divf", &[addf_val]);
        let (_expf_op, _) = make_op(&mut world, "arith.expf", &[divf_val]);

        let policy = Some(FusionPolicy(FusionStrategy::Conservative));

        let groups = analyze_dataflow(&world, addf_op, policy);
        assert_eq!(groups.len(), 1, "Conservative: one group");
        assert_eq!(
            groups[0].len(),
            4,
            "Conservative: 4 direct neighbours (1 producer + 2 consumers + root)"
        );
        assert!(
            groups[0].contains(&subf_op),
            "includes direct producer (subf)"
        );
        assert!(groups[0].contains(&addf_op), "includes root (addf)");
        assert!(
            groups[0].contains(&mulf_op),
            "includes direct consumer (mulf)"
        );
        assert!(
            groups[0].contains(&divf_op),
            "includes direct consumer (divf)"
        );
        assert!(
            !groups[0].contains(&_expf_op),
            "Conservative: does NOT include transitive consumer (expf)"
        );
    }

    #[test]
    fn aggressive_policy_includes_transitive_deps() {
        let mut world = World::new();

        // Chain: subf → addf, addf → mulf, addf → divf → expf
        // Analyzing from addf with Aggressive: everything connected.
        let (subf_op, subf_val) = make_op(&mut world, "arith.subf", &[]);
        let (addf_op, addf_val) = make_op(&mut world, "arith.addf", &[subf_val]);
        let (mulf_op, _) = make_op(&mut world, "arith.mulf", &[addf_val]);
        let (divf_op, divf_val) = make_op(&mut world, "arith.divf", &[addf_val]);
        let (expf_op, _) = make_op(&mut world, "arith.expf", &[divf_val]);

        let policy = Some(FusionPolicy(FusionStrategy::Aggressive));

        let groups = analyze_dataflow(&world, addf_op, policy);
        assert_eq!(groups.len(), 1, "Aggressive: one group");
        assert_eq!(
            groups[0].len(),
            5,
            "Aggressive: all 5 ops in one connected component"
        );
        assert!(groups[0].contains(&subf_op));
        assert!(groups[0].contains(&addf_op));
        assert!(groups[0].contains(&mulf_op));
        assert!(groups[0].contains(&divf_op));
        assert!(groups[0].contains(&expf_op));
    }

    // ── partition_fusion_groups tests ────────────────────────────────────

    #[test]
    fn partition_writes_fusion_group() {
        let mut world = World::new();

        let (addf_op, addf_val) = make_op(&mut world, "arith.addf", &[]);
        let (mulf_op, mulf_val) = make_op(&mut world, "arith.mulf", &[addf_val]);
        let (matmul_op, _matmul_val) = make_op(&mut world, "linalg.matmul", &[mulf_val]);

        let count = partition_fusion_groups(&mut world, matmul_op, None).unwrap();
        assert_eq!(count, 1, "one fusion group");

        // Every op should have FusionGroup with all three members
        for (label, op) in [("addf", addf_op), ("mulf", mulf_op), ("matmul", matmul_op)] {
            let fg = world
                .get_component::<FusionGroup>(op)
                .unwrap_or_else(|| panic!("{label} op should have FusionGroup"));
            assert_eq!(fg.0.len(), 3, "{label}: group has 3 members");
            assert!(fg.0.contains(&addf_op), "{label}: group has addf");
            assert!(fg.0.contains(&mulf_op), "{label}: group has mulf");
            assert!(fg.0.contains(&matmul_op), "{label}: group has matmul");
        }

        // A separate unrelated op should NOT get a FusionGroup
        let (unrelated, _) = make_op(&mut world, "func.return", &[]);
        assert!(
            world.get_component::<FusionGroup>(unrelated).is_none(),
            "unrelated op should not have FusionGroup"
        );
    }

    #[test]
    fn partition_isolated_op() {
        let mut world = World::new();
        let (op, _) = make_op(&mut world, "linalg.matmul", &[]);

        let count = partition_fusion_groups(&mut world, op, None).unwrap();
        assert_eq!(count, 1);

        let fg = world.get_component::<FusionGroup>(op).unwrap();
        assert_eq!(fg.0.len(), 1);
        assert_eq!(fg.0[0], op);
    }

    #[test]
    fn partition_no_double_write() {
        let mut world = World::new();
        let (addf_op, addf_val) = make_op(&mut world, "arith.addf", &[]);
        let (matmul_op, _matmul_val) = make_op(&mut world, "linalg.matmul", &[addf_val]);

        // Partition once
        partition_fusion_groups(&mut world, matmul_op, None).unwrap();

        // Adding again should overwrite — or at least not panic
        let count2 = partition_fusion_groups(&mut world, matmul_op, None).unwrap();
        assert_eq!(count2, 1);

        let fg = world.get_component::<FusionGroup>(addf_op).unwrap();
        assert_eq!(fg.0.len(), 2);
    }

    #[test]
    fn partition_none_policy_singleton() {
        let mut world = World::new();

        let (addf_op, addf_val) = make_op(&mut world, "arith.addf", &[]);
        let (mulf_op, _) = make_op(&mut world, "arith.mulf", &[addf_val]);

        let policy = Some(FusionPolicy(FusionStrategy::None));

        let count = partition_fusion_groups(&mut world, mulf_op, policy).unwrap();
        assert_eq!(count, 1, "None policy: one group");

        // Only the root op should carry FusionGroup
        let fg_mulf = world.get_component::<FusionGroup>(mulf_op).unwrap();
        assert_eq!(fg_mulf.0.len(), 1, "None policy: group has one member");
        assert_eq!(fg_mulf.0[0], mulf_op, "None policy: root op is sole member");

        // addf should NOT be in a fusion group
        assert!(
            world.get_component::<FusionGroup>(addf_op).is_none(),
            "None policy: producer should NOT be fused"
        );
    }

    #[test]
    fn partition_conservative_policy() {
        let mut world = World::new();

        // Chain: addf → mulf → matmul
        let (addf_op, addf_val) = make_op(&mut world, "arith.addf", &[]);
        let (mulf_op, mulf_val) = make_op(&mut world, "arith.mulf", &[addf_val]);
        let (matmul_op, _) = make_op(&mut world, "linalg.matmul", &[mulf_val]);

        let policy = Some(FusionPolicy(FusionStrategy::Conservative));

        // Analyze from mulf — should get all three (direct neighbours)
        let count = partition_fusion_groups(&mut world, mulf_op, policy).unwrap();
        assert_eq!(count, 1, "Conservative: one fusion group");

        for (label, op) in [("addf", addf_op), ("mulf", mulf_op), ("matmul", matmul_op)] {
            let fg = world
                .get_component::<FusionGroup>(op)
                .unwrap_or_else(|| panic!("{label} op should have FusionGroup under Conservative"));
            assert_eq!(fg.0.len(), 3, "{label}: group has 3 direct neighbours");
        }
    }
}
