//! Tri-lane orchestration system (constitutional home, runtime half).
//!
//! Per the inventory v2.1 row 52, the engine's `tri_lane_orchestrator.rs`
//! is split during absorption:
//! - Runtime orchestration logic (this file)
//! - Per-lane dispatch consumers (move to `prism-ecs-kernel::backend::*`)
//!
//! The orchestrator coordinates three lanes (Metal/GPU, ANE, CPU)
//! and selects the best variant for each phase.

use std::collections::BTreeMap;

use crate::scheduling::state::lane_work::ExecutionLane;
use crate::scheduling::state::phase::PhaseId;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriLaneStrategy {
    /// One lane only.
    Single(ExecutionLane),
    /// Split across multiple lanes.
    Split,
    /// Run sequentially: GPU → ANE → CPU fallback chain.
    Sequential,
}

#[derive(Debug, Clone, Default)]
pub struct TriLaneOrchestrator {
    /// Per-phase strategy map. BTreeMap for stable iteration order.
    strategies: BTreeMap<PhaseId, TriLaneStrategy>,
}

impl TriLaneOrchestrator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the strategy for a phase.
    pub fn set_strategy(&mut self, phase_id: PhaseId, strategy: TriLaneStrategy) {
        self.strategies.insert(phase_id, strategy);
    }

    /// Get the strategy for a phase. Returns `Sequential` as the
    /// default for unregistered phases.
    pub fn strategy(&self, phase_id: &PhaseId) -> TriLaneStrategy {
        self.strategies
            .get(phase_id)
            .copied()
            .unwrap_or(TriLaneStrategy::Sequential)
    }

    /// Number of phases with explicit strategies.
    pub fn len(&self) -> usize {
        self.strategies.len()
    }

    pub fn is_empty(&self) -> bool {
        self.strategies.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_orchestrator_is_empty() {
        let o = TriLaneOrchestrator::new();
        assert!(o.is_empty());
    }

    #[test]
    fn default_strategy_is_sequential() {
        // Architectural invariant: an unregistered phase defaults
        // to Sequential (the conservative GPU → ANE → CPU chain).
        let o = TriLaneOrchestrator::new();
        let pid = PhaseId("p1".into());
        assert_eq!(o.strategy(&pid), TriLaneStrategy::Sequential);
    }

    #[test]
    fn set_strategy_overrides_default() {
        // Architectural invariant: setting a strategy for a phase
        // overrides the default.
        let mut o = TriLaneOrchestrator::new();
        let pid = PhaseId("p1".into());
        o.set_strategy(pid.clone(), TriLaneStrategy::Single(ExecutionLane::CoreAiAne));
        assert_eq!(
            o.strategy(&pid),
            TriLaneStrategy::Single(ExecutionLane::CoreAiAne)
        );
    }

    #[test]
    fn strategies_partition_correctly() {
        // Architectural invariant: the three TriLaneStrategy variants
        // are mutually exclusive. A reader can dispatch on the
        // variant without forgetting a case.
        let strategies = [
            TriLaneStrategy::Single(ExecutionLane::MlxGpu),
            TriLaneStrategy::Split,
            TriLaneStrategy::Sequential,
        ];
        for s in strategies {
            let count = strategies.iter().filter(|&&v| v == s).count();
            assert_eq!(count, 1);
        }
    }
}
