//! Canonical authority for buffer-lifetime planning system types (lifetime analysis, scratch planning) that the engine's compile_session.rs references. The engine file is no longer present in the engine source.

pub struct LifetimeAnalysisSystem;

impl Default for LifetimeAnalysisSystem {
    fn default() -> Self { Self }
}

impl LifetimeAnalysisSystem {
    pub fn new() -> Self { Self }
}

pub struct ScratchPlanningSystem;

impl Default for ScratchPlanningSystem {
    fn default() -> Self { Self }
}
