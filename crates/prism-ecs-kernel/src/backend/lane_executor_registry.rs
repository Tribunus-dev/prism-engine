//! Lane executor registry (constitutional home).
//!
//! Per the inventory v2.1 row 21, this replaces the engine's
//! `lane_executors.rs` (97 LOC). The registry maps lanes to
//! their per-backend executors.

use std::collections::BTreeMap;

use crate::execution_lane::ExecutionLane;
use crate::backend::accelerate::lane_executor::AccelerateLaneExecutor;
use crate::backend::ane::lane_executor::AneLaneExecutor;
use crate::backend::metal::lane_executor::MetalLaneExecutor;
use crate::backend::dispatcher::heterogeneous::Completion;

#[derive(Debug, Default)]
pub struct LaneExecutorRegistry {
    metal: Option<MetalLaneExecutor>,
    ane: Option<AneLaneExecutor>,
    accelerate: Option<AccelerateLaneExecutor>,
    /// Per-lane executor map. Uses BTreeMap for stable iteration
    /// order — a future dispatcher round-robin iterates this map
    /// and the result must be deterministic.
    lanes: BTreeMap<ExecutionLane, LaneKind>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LaneKind {
    Metal,
    Ane,
    Accelerate,
}

impl LaneExecutorRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_metal(&mut self) {
        self.metal = Some(MetalLaneExecutor::new());
        self.lanes.insert(ExecutionLane::MlxGpu, LaneKind::Metal);
        self.lanes.insert(ExecutionLane::Tensix, LaneKind::Metal);
    }

    pub fn register_ane(&mut self) {
        self.ane = Some(AneLaneExecutor::new());
        self.lanes.insert(ExecutionLane::CoreAiAne, LaneKind::Ane);
    }

    pub fn register_accelerate(&mut self) {
        self.accelerate = Some(AccelerateLaneExecutor::new());
        self.lanes
            .insert(ExecutionLane::AccelerateCpu, LaneKind::Accelerate);
        self.lanes
            .insert(ExecutionLane::CandleCpu, LaneKind::Accelerate);
        self.lanes
            .insert(ExecutionLane::IntelLevelZero, LaneKind::Accelerate);
    }

    /// Submit work to the executor registered for `lane`.
    /// Returns None if no executor is registered for the lane.
    pub fn submit(&self, lane: ExecutionLane) -> Option<Result<Completion, String>> {
        let kind = self.lanes.get(&lane)?;
        Some(match kind {
            LaneKind::Metal => self.metal.as_ref()?.submit(),
            LaneKind::Ane => self.ane.as_ref()?.submit(),
            LaneKind::Accelerate => self.accelerate.as_ref()?.submit(),
        }
        // The completion's lane is set by the executor; the
        // registry forwards the request lane so the dispatch
        // policy can route per-lane. The test verifies the
        // lane round-trips.
        .map(|mut c| {
            c.lane = lane;
            c
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_registry_submits_nothing() {
        // Architectural invariant: a fresh registry has no
        // registered executors; submit returns None for any lane.
        let r = LaneExecutorRegistry::new();
        assert!(r.submit(ExecutionLane::MlxGpu).is_none());
    }

    #[test]
    fn register_metal_makes_metal_lane_routable() {
        // Architectural invariant: registering the metal executor
        // makes Metal-family lanes routable.
        let mut r = LaneExecutorRegistry::new();
        r.register_metal();
        let result = r.submit(ExecutionLane::MlxGpu);
        assert!(result.is_some());
        let c = result.unwrap().expect("submit succeeds");
        assert_eq!(c.lane, ExecutionLane::MlxGpu);
    }

    #[test]
    fn register_ane_makes_ane_lane_routable() {
        let mut r = LaneExecutorRegistry::new();
        r.register_ane();
        let result = r.submit(ExecutionLane::CoreAiAne);
        assert!(result.is_some());
        let c = result.unwrap().expect("submit succeeds");
        assert_eq!(c.lane, ExecutionLane::CoreAiAne);
    }

    #[test]
    fn register_accelerate_routes_all_cpu_lanes() {
        // Architectural invariant: the accelerate executor routes
        // all CPU-family lanes (AccelerateCpu, CandleCpu,
        // IntelLevelZero). This is the classification invariant
        // from ExecutionLane.
        let mut r = LaneExecutorRegistry::new();
        r.register_accelerate();
        for lane in [
            ExecutionLane::AccelerateCpu,
            ExecutionLane::CandleCpu,
            ExecutionLane::IntelLevelZero,
        ] {
            let c = r.submit(lane).unwrap().expect("submit succeeds");
            assert_eq!(c.lane, lane);
        }
    }
}
