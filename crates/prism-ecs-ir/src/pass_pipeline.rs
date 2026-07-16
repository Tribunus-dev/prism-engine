//! Pass pipeline registry and runner DSL.
//!
//! Extends the existing [`PassPipeline`] in `pass_manager` with a registry
//! of known passes and a pipeline DSL for composing them by name.
//!
//! # Design
//!
//! Mirrors MLIR's `PassPipeline` and `PassManager` patterns — a pipeline
//! is an ordered sequence of named passes, each implementing the [`Pass`]
//! trait. The [`PassPipelineRegistry`] provides a central catalog, and
//! [`PipelineRunner`] provides a builder DSL with statistics collection.
//!
//! # Example
//!
//! ```ignore
//! let mut registry = PassPipelineRegistry::new();
//! registry.register::<EraseUnusedOpsPass>();
//!
//! let mut runner = PipelineRunner::new(&registry);
//! runner.add("erase-unused-ops");
//! runner.run(&mut world)?;
//! runner.print_statistics();
//! ```

use prism_ecs_core::World;

use crate::pass_manager::{Pass, PassPipeline, PassResult, PassStatistics};

// ── PassRegistration ─────────────────────────────────────────────────────────

/// A named pass registration that can be instantiated on demand.
pub struct PassRegistration {
    /// Canonical pass name (e.g. `"erase-unused-ops"`).
    pub name: &'static str,
    /// One-line description.
    pub description: &'static str,
    /// Factory function that creates a new instance of this pass.
    pub factory: fn() -> Box<dyn Pass>,
}

impl std::fmt::Debug for PassRegistration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PassRegistration")
            .field("name", &self.name)
            .field("description", &self.description)
            .finish()
    }
}

// ── PassPipelineRegistry ─────────────────────────────────────────────────────

/// A registry of known compiler passes, keyed by name.
///
/// Passes are registered by type and instantiated lazily when a
/// [`PipelineRunner`] requests them by name.
pub struct PassPipelineRegistry {
    passes: Vec<PassRegistration>,
}

impl PassPipelineRegistry {
    pub fn new() -> Self {
        Self { passes: Vec::new() }
    }

    /// Register a pass type so it can be referenced by name in pipelines.
    ///
    /// Uses the pass's `name()` and `description()` methods to populate the
    /// registration metadata. The pass type must implement [`Pass`] + `Default`.
    pub fn register<P: Pass + Default + 'static>(&mut self) {
        let instance = P::default();
        let name = instance.name();
        let description = instance.description();
        drop(instance); // drop the temporary; factory creates a fresh one
        self.register_named(name, description, || Box::new(P::default()));
    }

    /// Register a pass with explicit factory.
    pub fn register_named(
        &mut self,
        name: &'static str,
        description: &'static str,
        factory: fn() -> Box<dyn Pass>,
    ) {
        // Ensure no duplicate names.
        if let Some(pos) = self.passes.iter().position(|r| r.name == name) {
            self.passes[pos] = PassRegistration {
                name,
                description,
                factory,
            };
            return;
        }
        self.passes.push(PassRegistration {
            name,
            description,
            factory,
        });
    }

    /// Look up a pass registration by name.
    pub fn get(&self, name: &str) -> Option<&PassRegistration> {
        self.passes.iter().find(|r| r.name == name)
    }

    /// Instantiate a pass by name.
    pub fn instantiate(&self, name: &str) -> Option<Box<dyn Pass>> {
        self.get(name).map(|r| (r.factory)())
    }

    /// List all registered pass names.
    pub fn names(&self) -> Vec<&'static str> {
        self.passes.iter().map(|r| r.name).collect()
    }

    /// The number of registered passes.
    pub fn len(&self) -> usize {
        self.passes.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.passes.is_empty()
    }
}

impl Default for PassPipelineRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ── Pipeline DSL ─────────────────────────────────────────────────────────────

/// A step in a pipeline, either a named pass or a sub-pipeline.
#[derive(Debug, Clone)]
pub enum PipelineStep {
    /// A named pass to instantiate and run.
    Pass(&'static str),
    /// A sub-pipeline to run as a single step.
    SubPipeline(Vec<PipelineStep>),
}

/// A DSL for composing pass pipelines by name.
///
/// ```ignore
/// let mut runner = PipelineRunner::new(&registry);
/// runner
///     .then("erase-unused-ops")
///     .then("canonicalize")
///     .then("cse")
///     .run(&mut world)?;
/// ```
pub struct PipelineRunner<'a> {
    registry: &'a PassPipelineRegistry,
    pipeline: PassPipeline,
    steps: Vec<PipelineStep>,
}

impl<'a> PipelineRunner<'a> {
    /// Create a new pipeline runner referencing a [`PassPipelineRegistry`].
    pub fn new(registry: &'a PassPipelineRegistry) -> Self {
        Self {
            registry,
            pipeline: PassPipeline::new(),
            steps: Vec::new(),
        }
    }

    /// Append a named pass to the pipeline.
    ///
    /// # Panics
    ///
    /// Panics if `name` is not registered. Use [`try_then`](Self::try_then)
    /// for a non-panicking variant.
    pub fn then(&mut self, name: &'static str) -> &mut Self {
        assert!(
            self.registry.get(name).is_some(),
            "pass '{}' not registered in PassPipelineRegistry",
            name
        );
        self.steps.push(PipelineStep::Pass(name));
        self
    }

    /// Append a named pass, returning `None` if unregistered.
    pub fn try_then(&mut self, name: &'static str) -> Option<&mut Self> {
        if self.registry.get(name).is_some() {
            self.steps.push(PipelineStep::Pass(name));
            Some(self)
        } else {
            None
        }
    }

    /// Append a sub-pipeline step.
    pub fn then_pipeline(&mut self, steps: Vec<PipelineStep>) -> &mut Self {
        self.steps.push(PipelineStep::SubPipeline(steps));
        self
    }

    /// Build the internal pass pipeline from the registered steps.
    fn build(&mut self) {
        for step in &self.steps {
            match step {
                PipelineStep::Pass(name) => {
                    if let Some(pass) = self.registry.instantiate(name) {
                        self.pipeline.add_pass(pass);
                    }
                }
                PipelineStep::SubPipeline(sub_steps) => {
                    for sub_step in sub_steps {
                        if let PipelineStep::Pass(name) = sub_step {
                            if let Some(pass) = self.registry.instantiate(name) {
                                self.pipeline.add_pass(pass);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Run the pipeline against `world`.
    pub fn run(&mut self, world: &mut World) -> PassResult {
        self.build();
        self.pipeline.run(world)
    }

    /// Return the current statistics.
    pub fn statistics(&self) -> &PassStatistics {
        self.pipeline.statistics()
    }

    /// Print a statistics summary.
    pub fn print_statistics(&self) {
        self.pipeline.print_statistics();
    }

    /// Reset statistics.
    pub fn reset_statistics(&mut self) {
        self.pipeline.reset_statistics();
    }

    /// Clear all steps and reset the pipeline.
    pub fn clear(&mut self) {
        self.steps.clear();
        self.pipeline = PassPipeline::new();
    }

    /// Access the underlying pipeline for advanced use.
    pub fn pipeline(&mut self) -> &mut PassPipeline {
        &mut self.pipeline
    }
}

// ── Standard pass registrations ──────────────────────────────────────────────

/// Register the standard built-in passes into a registry.
///
/// This is a convenience function that registers all passes defined in
/// `pass_manager`.
pub fn register_standard_passes(registry: &mut PassPipelineRegistry) {
    registry.register_named(
        "erase-unused-ops",
        "Remove operations whose results have no consumers",
        || Box::new(crate::pass_manager::EraseUnusedOpsPass),
    );
    // Future passes can be added here:
    // registry.register_named("canonicalize", "Canonicalize operations", || ...);
    // registry.register_named("cse", "Common subexpression elimination", || ...);
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pass_manager::{EraseUnusedOpsPass, Pass, PassPipeline};

    #[test]
    fn register_and_instantiate() {
        let mut registry = PassPipelineRegistry::new();
        registry.register_named(
            "erase-unused-ops",
            "Remove dead ops",
            || Box::new(EraseUnusedOpsPass),
        );

        assert!(registry.get("erase-unused-ops").is_some());
        assert!(registry.get("nonexistent").is_none());

        let pass = registry.instantiate("erase-unused-ops");
        assert!(pass.is_some());
        assert_eq!(pass.unwrap().name(), "erase-unused-ops");
    }

    #[test]
    fn pipeline_runner_empty() {
        let registry = PassPipelineRegistry::new();
        let mut runner = PipelineRunner::new(&registry);

        let mut world = World::new();
        let result = runner.run(&mut world);
        assert_eq!(result, PassResult::Unchanged);
    }

    #[test]
    fn pipeline_runner_valid() {
        let mut registry = PassPipelineRegistry::new();
        registry.register_named(
            "erase-unused-ops",
            "Remove dead ops",
            || Box::new(EraseUnusedOpsPass),
        );

        let mut world = World::new();
        let mut runner = PipelineRunner::new(&registry);
        runner.then("erase-unused-ops");
        let result = runner.run(&mut world);
        // In an empty world, pass should report Unchanged.
        assert_eq!(result, PassResult::Unchanged);
        assert_eq!(runner.statistics().passes_run, 1);
    }

    #[test]
    fn register_standard_passes_test() {
        let mut registry = PassPipelineRegistry::new();
        register_standard_passes(&mut registry);

        assert!(registry.get("erase-unused-ops").is_some());
        assert_eq!(registry.names().len(), 1);
    }

    #[test]
    #[should_panic(expected = "not registered")]
    fn pipeline_runner_unknown_pass() {
        let registry = PassPipelineRegistry::new();
        let mut runner = PipelineRunner::new(&registry);
        runner.then("nonexistent-pass");
    }
}
