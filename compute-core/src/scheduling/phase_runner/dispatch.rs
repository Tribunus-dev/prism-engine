//! Phase dispatch — [`PhaseRunnerRegistry`] maps each [`PhaseKind`] to a
//! concrete [`PhaseRunner`].
//!
//! The registry is populated at construction with every built-in runner.
//! Unrecognized phase kinds fall through to the fallback handler.

use crate::compute_image::phase_dag::{EmittedPhase, PhaseKind};
use crate::scheduling::execution_context::ExecutionContext;
use crate::scheduling::phase_runner::execution::{
    AccelElementWiseRunner, AccelMatMulRunner, ArenaAllocRunner, CoreMlGraphRunner,
    LegacyMlxEpilogueRunner, LegacyMlxLayerRunner, LegacyMlxPrologueRunner,
    MetalFusedKernelRunner, MlxDecodeRunner, PhaseRunner, ResidualRmsNormRunner, SamplingRunner,
    SyncBarrierRunner, TransferRunner, WeightResidencyRunner,
};
use crate::scheduling::phase_runner::fallback;

/// Registry that maps [`PhaseKind`] to a concrete [`PhaseRunner`].
pub struct PhaseRunnerRegistry {
    runners: std::collections::HashMap<PhaseKind, Box<dyn PhaseRunner>>,
}

impl PhaseRunnerRegistry {
    pub fn new() -> Self {
        let mut runners: std::collections::HashMap<PhaseKind, Box<dyn PhaseRunner>> =
            std::collections::HashMap::new();

        let default_runners: Vec<Box<dyn PhaseRunner>> = vec![
            Box::new(MlxDecodeRunner),
            Box::new(MetalFusedKernelRunner),
            Box::new(CoreMlGraphRunner),
            Box::new(AccelMatMulRunner),
            Box::new(AccelElementWiseRunner),
            Box::new(ArenaAllocRunner),
            Box::new(SyncBarrierRunner),
            Box::new(TransferRunner),
            Box::new(ResidualRmsNormRunner),
            Box::new(LegacyMlxLayerRunner),
            Box::new(LegacyMlxPrologueRunner),
            Box::new(LegacyMlxEpilogueRunner),
            Box::new(SamplingRunner),
            Box::new(WeightResidencyRunner),
        ];

        for r in default_runners {
            runners.insert(r.kind(), r);
        }

        Self { runners }
    }

    /// Dispatch a phase to its registered runner, or to the
    /// [`fallback`] handler when no runner is registered.
    pub fn dispatch(&self, phase: &EmittedPhase, ctx: &mut ExecutionContext) -> Result<(), String> {
        match self.runners.get(&phase.kind) {
            Some(runner) => runner.run(phase, ctx),
            None => fallback::run_fallback(phase),
        }
    }
}

impl Default for PhaseRunnerRegistry {
    fn default() -> Self {
        Self::new()
    }
}
