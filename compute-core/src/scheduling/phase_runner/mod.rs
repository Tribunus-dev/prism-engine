//! Phase runners — dispatch logic for each [`PhaseKind`].
//!
//! Each phase kind maps to a concrete runner.  The [`PhaseRunnerRegistry`]
//! provides dispatch-by-kind lookup.

pub mod dispatch;
pub mod execution;
pub mod fallback;

pub use dispatch::PhaseRunnerRegistry;
pub use execution::{
    AccelElementWiseRunner, AccelMatMulRunner, ArenaAllocRunner, CoreMlGraphRunner,
    LegacyMlxEpilogueRunner, LegacyMlxLayerRunner, LegacyMlxPrologueRunner,
    MetalFusedKernelRunner, MlxDecodeRunner, PhaseResult, PhaseRunner, ResidualRmsNormRunner,
    SamplingRunner, SyncBarrierRunner, TransferRunner, WeightResidencyRunner,
};
pub use fallback::run_fallback;
