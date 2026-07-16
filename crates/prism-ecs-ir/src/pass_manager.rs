//! Pass framework for the ECS-native IR.
//!
//! Defines compiler passes: transformations that run over the IR in a World,
//! reporting what they modified. Passes are registered into a PassPipeline
//! and executed in order.

use crate::op::{OpMarker, Results};
use crate::value::Uses;
use prism_ecs_core::{Entity, World};

// ── PassResult ───────────────────────────────────────────────────────────────

/// Outcome of running a single pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PassResult {
    /// Pass made changes; carries the number of ops modified.
    Modified(u64),
    /// Pass ran and made no changes.
    Unchanged,
    /// Pass failed with an error message.
    Failure(String),
}

// ── Pass trait ───────────────────────────────────────────────────────────────

/// A single compiler pass — a transformation over the IR in a World.
///
/// Passes are `Send + Sync` so they can be composed safely in pipelines
/// (the pipeline serialises execution; the bounds enable future parallel
/// pass scheduling).
pub trait Pass: Send + Sync {
    /// Short, unique name for the pass (e.g. `"erase-unused-ops"`).
    fn name(&self) -> &'static str;

    /// Human-readable description of what the pass does.
    fn description(&self) -> &'static str;

    /// Execute the pass against `world`.
    ///
    /// The pass receives mutable access to the entire World and returns
    /// a [`PassResult`] summarising the effect.
    fn run(&self, world: &mut World) -> PassResult;
}

// ── PassStatistics ───────────────────────────────────────────────────────────

/// Aggregate statistics across all passes in a pipeline run.
#[derive(Debug, Clone, Default)]
pub struct PassStatistics {
    /// Total number of passes that were executed (including failed ones).
    pub passes_run: u64,
    /// Cumulative number of ops modified by all passes.
    pub ops_modified: u64,
    /// Number of passes that returned [`PassResult::Failure`].
    pub failures: u64,
}

impl PassStatistics {
    /// Reset all counters to zero.
    pub fn reset(&mut self) {
        self.passes_run = 0;
        self.ops_modified = 0;
        self.failures = 0;
    }
}

// ── PassPipeline ─────────────────────────────────────────────────────────────

/// An ordered sequence of compiler passes.
///
/// Passes are registered with [`add_pass`](PassPipeline::add_pass) and
/// executed in registration order via [`run`](PassPipeline::run), which
/// collects per-pass statistics.
pub struct PassPipeline {
    passes: Vec<Box<dyn Pass>>,
    statistics: PassStatistics,
}

impl PassPipeline {
    /// Create an empty pipeline.
    pub fn new() -> Self {
        Self {
            passes: Vec::new(),
            statistics: PassStatistics::default(),
        }
    }

    /// Append a pass to the end of the pipeline.
    pub fn add_pass(&mut self, pass: Box<dyn Pass>) {
        self.passes.push(pass);
    }

    /// Run every registered pass in order against `world`.
    ///
    /// Returns a cumulative result:
    /// - [`PassResult::Modified(n)`] if any pass modified ops (n = total).
    /// - [`PassResult::Unchanged`] if no pass made changes.
    /// - [`PassResult::Failure(msg)`] if a pass failed (stops early).
    ///
    /// Call [`print_statistics`](PassPipeline::print_statistics) after
    /// to inspect per-run aggregates.
    pub fn run(&mut self, world: &mut World) -> PassResult {
        let mut total_modified: u64 = 0;
        let mut any_modified = false;

        for pass in &self.passes {
            self.statistics.passes_run += 1;

            match pass.run(world) {
                PassResult::Modified(n) => {
                    total_modified += n;
                    self.statistics.ops_modified += n;
                    any_modified = true;
                }
                PassResult::Unchanged => {
                    // Nothing to accumulate.
                }
                PassResult::Failure(msg) => {
                    self.statistics.failures += 1;
                    return PassResult::Failure(format!("pass '{}' failed: {}", pass.name(), msg,));
                }
            }
        }

        if any_modified {
            PassResult::Modified(total_modified)
        } else {
            PassResult::Unchanged
        }
    }

    /// Print a summary of the statistics collected during the last run.
    pub fn print_statistics(&self) {
        println!(
            "PassPipeline statistics: {} passes run, {} ops modified, {} failures",
            self.statistics.passes_run, self.statistics.ops_modified, self.statistics.failures,
        );
    }

    /// Return a reference to the current statistics.
    pub fn statistics(&self) -> &PassStatistics {
        &self.statistics
    }

    /// Reset all statistics counters.
    pub fn reset_statistics(&mut self) {
        self.statistics.reset();
    }
}

impl Default for PassPipeline {
    fn default() -> Self {
        Self::new()
    }
}

// ── Built-in passes ──────────────────────────────────────────────────────────

/// A pass that removes operations whose results have no consumers.
///
/// For each op in the world, the pass inspects every result Value produced by
/// the op. If *all* result values have an empty use list ([`Uses`] component
/// containing zero entries, or no `Uses` component at all), the op is
/// considered dead and is despawned from the world.
///
/// This pass is local (does not follow Regions into nested ops) and runs
/// to a fixed point: operations whose results become dead only after a
/// consumer is erased in the same pass are NOT removed. Call the pass
/// repeatedly or rely on iteration order when needed.
pub struct EraseUnusedOpsPass;

impl Pass for EraseUnusedOpsPass {
    fn name(&self) -> &'static str {
        "erase-unused-ops"
    }

    fn description(&self) -> &'static str {
        "Erase operations whose results have no uses"
    }

    fn run(&self, world: &mut World) -> PassResult {
        // Collect candidates first to avoid borrow issues with mutation.
        let candidates: Vec<Entity> = world
            .query::<OpMarker>()
            .filter_map(|(entity, _)| {
                let results = world.get_component::<Results>(entity);
                match results {
                    Some(results) if results.0.is_empty() => {
                        // No results at all — op is dead by definition.
                        Some(entity)
                    }
                    Some(results) => {
                        // Op has results — check every result's use count.
                        let all_unused = results.0.iter().all(|val_entity| {
                            if let Some(uses) = world.get_component::<Uses>(*val_entity) {
                                uses.0.is_empty()
                            } else {
                                // No Uses component = no consumers.
                                true
                            }
                        });
                        if all_unused {
                            Some(entity)
                        } else {
                            None
                        }
                    }
                    None => {
                        // Op-like entity without Results — treat as dead.
                        Some(entity)
                    }
                }
            })
            .collect();

        let count = candidates.len() as u64;
        if count == 0 {
            return PassResult::Unchanged;
        }

        for entity in candidates {
            // Remove components from column store BEFORE despawn, because
            // despawn only advances the slot generation without removing
            // component data — Query would still find the ghost entity.
            let _ = world.remove_component::<OpMarker>(entity);
            let _ = world.remove_component::<Results>(entity);
            let _ = world.despawn(entity);
        }

        PassResult::Modified(count)
    }
}

impl Default for EraseUnusedOpsPass {
    fn default() -> Self {
        Self
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ecs_core::EntityKind;

    /// Helper: create an op entity with the given results.
    fn create_op(world: &mut World, name: &str, result_values: Vec<Entity>) -> Entity {
        let entity: Entity = world
            .spawn(EntityKind::Node, Some(name.into()))
            .expect("spawn")
            .into();
        world.add_component(entity, OpMarker).expect("OpMarker");
        world
            .add_component(entity, crate::op::OpName(name.into()))
            .expect("OpName");
        world
            .add_component(entity, Results(result_values))
            .expect("Results");
        entity
    }

    /// Helper: create a Value entity with the given use-list.
    fn create_value(world: &mut World, uses: Vec<Entity>) -> Entity {
        let val: Entity = world
            .spawn(EntityKind::Node, Some("val".into()))
            .expect("spawn")
            .into();
        world
            .add_component(val, crate::value::ValueDef::op_result(Entity(0, 1), 0))
            .expect("ValueDef");
        world
            .add_component(val, crate::value::Uses(uses))
            .expect("Uses");
        val
    }

    // ── EraseUnusedOpsPass tests ─────────────────────────────────────────

    #[test]
    fn erase_op_with_no_results() {
        let mut world = World::new();
        create_op(&mut world, "test.noop", vec![]);

        let pass = EraseUnusedOpsPass;
        let result = pass.run(&mut world);

        assert_eq!(result, PassResult::Modified(1));
        // The op should no longer exist.
        let remaining: Vec<_> = world.query::<OpMarker>().collect();
        assert!(
            remaining.is_empty(),
            "expected all ops removed, got {}",
            remaining.len()
        );
    }

    #[test]
    fn erase_op_with_unused_result() {
        let mut world = World::new();
        let val = create_value(&mut world, vec![]); // no uses
        create_op(&mut world, "test.producer", vec![val]);

        let pass = EraseUnusedOpsPass;
        let result = pass.run(&mut world);

        assert_eq!(result, PassResult::Modified(1));
        let remaining: Vec<_> = world.query::<OpMarker>().collect();
        assert!(
            remaining.is_empty(),
            "expected all ops removed, got {}",
            remaining.len()
        );
    }

    #[test]
    fn keep_op_with_used_result() {
        let mut world = World::new();
        let val = create_value(&mut world, vec![Entity(99, 1)]); // has a user
        let op = create_op(&mut world, "test.producer", vec![val]);

        let pass = EraseUnusedOpsPass;
        let result = pass.run(&mut world);

        assert_eq!(result, PassResult::Unchanged);
        let remaining: Vec<_> = world.query::<OpMarker>().collect();
        assert_eq!(remaining.len(), 1, "op should still be present");
        assert_eq!(remaining[0].0, op, "same entity");
    }

    #[test]
    fn keep_op_when_one_of_multiple_results_is_used() {
        let mut world = World::new();
        let used_val = create_value(&mut world, vec![Entity(42, 1)]);
        let unused_val = create_value(&mut world, vec![]);
        let op = create_op(&mut world, "test.multi", vec![used_val, unused_val]);

        let pass = EraseUnusedOpsPass;
        let result = pass.run(&mut world);

        assert_eq!(result, PassResult::Unchanged);
        let remaining: Vec<_> = world.query::<OpMarker>().collect();
        assert_eq!(remaining.len(), 1, "op still in use");
        assert_eq!(remaining[0].0, op);
    }

    #[test]
    fn erase_op_without_uses_component() {
        // A value without any Uses component is treated as having no consumers.
        let mut world = World::new();
        let val: Entity = world
            .spawn(EntityKind::Node, Some("val".into()))
            .expect("spawn")
            .into();
        world
            .add_component(val, crate::value::ValueDef::op_result(Entity(0, 1), 0))
            .expect("ValueDef");
        // Intentionally no Uses component.
        create_op(&mut world, "test.producer", vec![val]);

        let pass = EraseUnusedOpsPass;
        let result = pass.run(&mut world);

        assert_eq!(result, PassResult::Modified(1));
        let remaining: Vec<_> = world.query::<OpMarker>().collect();
        assert!(remaining.is_empty());
    }

    // ── PassPipeline tests ────────────────────────────────────────────────

    #[test]
    fn pipeline_empty_run_is_unchanged() {
        let mut pipeline = PassPipeline::new();
        let mut world = World::new();

        let result = pipeline.run(&mut world);
        assert_eq!(result, PassResult::Unchanged);
    }

    #[test]
    fn pipeline_single_pass_modifies() {
        let mut pipeline = PassPipeline::new();
        pipeline.add_pass(Box::new(EraseUnusedOpsPass));

        let mut world = World::new();
        create_op(&mut world, "test.unused", vec![]);

        let result = pipeline.run(&mut world);
        assert_eq!(result, PassResult::Modified(1));
    }

    #[test]
    fn pipeline_multiple_passes_accumulate() {
        let mut pipeline = PassPipeline::new();
        // Register the same pass twice; each run removes whatever it finds.
        pipeline.add_pass(Box::new(EraseUnusedOpsPass));
        pipeline.add_pass(Box::new(EraseUnusedOpsPass));

        let mut world = World::new();
        // Two unused ops — first pass removes both, second finds nothing.
        create_op(&mut world, "test.a", vec![]);
        create_op(&mut world, "test.b", vec![]);

        let result = pipeline.run(&mut world);
        assert_eq!(result, PassResult::Modified(2));
    }

    #[test]
    fn pipeline_statistics_accurate() {
        let mut pipeline = PassPipeline::new();
        pipeline.add_pass(Box::new(EraseUnusedOpsPass));
        pipeline.add_pass(Box::new(EraseUnusedOpsPass));

        let mut world = World::new();
        create_op(&mut world, "test.a", vec![]);
        create_op(&mut world, "test.b", vec![]);

        let _ = pipeline.run(&mut world);

        assert_eq!(
            pipeline.statistics().passes_run,
            2,
            "both passes should have run"
        );
        assert_eq!(
            pipeline.statistics().ops_modified,
            2,
            "should have modified 2 ops total"
        );
        assert_eq!(pipeline.statistics().failures, 0, "no failures expected");
    }

    #[test]
    fn pipeline_statistics_unchanged() {
        let mut pipeline = PassPipeline::new();
        pipeline.add_pass(Box::new(EraseUnusedOpsPass));

        let mut world = World::new();
        let val = create_value(&mut world, vec![Entity(1, 1)]);
        create_op(&mut world, "test.used", vec![val]);

        let _ = pipeline.run(&mut world);

        assert_eq!(pipeline.statistics().passes_run, 1);
        assert_eq!(pipeline.statistics().ops_modified, 0);
        assert_eq!(pipeline.statistics().failures, 0);
    }

    #[test]
    fn pipeline_statistics_reset() {
        let mut pipeline = PassPipeline::new();
        pipeline.add_pass(Box::new(EraseUnusedOpsPass));

        let mut world = World::new();
        create_op(&mut world, "test.unused", vec![]);
        let _ = pipeline.run(&mut world);
        assert_eq!(pipeline.statistics().ops_modified, 1);

        pipeline.reset_statistics();
        assert_eq!(pipeline.statistics().ops_modified, 0);
        assert_eq!(pipeline.statistics().passes_run, 0);
    }

    #[test]
    fn pipeline_default_impl() {
        let pipeline = PassPipeline::default();
        assert!(pipeline.passes.is_empty());
        assert_eq!(pipeline.statistics().passes_run, 0);
    }
}
