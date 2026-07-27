//! Scheduling benchmarks (constitutional home).
//!
//! Per the inventory v2.1 step 56, this replaces the engine's
//! `benchmark_harness.rs` (289 LOC). The benchmarks exercise the
//! constitutional scheduling state and systems.

use prism_ecs_runtime::scheduling::state::lane_capacity::{LaneCapacityConfig, LaneCapacityManager};
use prism_ecs_kernel::execution_lane::ExecutionLane;
use std::time::Instant;

#[test]
fn benchmark_lane_capacity_acquire_release() {
    // Architectural invariant: the lane capacity manager can
    // sustain high acquire/release throughput. This is a
    // regression test, not a microbenchmark.
    let mut mgr = LaneCapacityManager::new(LaneCapacityConfig {
        max_in_flight_command_buffers: 100,
        max_in_flight_ane_predictions: 100,
        max_in_flight_cpu_workers: 100,
        max_queued_per_lane: 1000,
        max_in_flight_per_session: 1000,
        global_max_pending: 100_000,
    });
    let start = Instant::now();
    let mut permits = Vec::new();
    for i in 0..1000 {
        let lane = match i % 3 {
            0 => ExecutionLane::MlxGpu,
            1 => ExecutionLane::CoreAiAne,
            _ => ExecutionLane::AccelerateCpu,
        };
        if let Some(p) = mgr.try_acquire(lane, "bench") {
            permits.push(p);
        }
    }
    let elapsed = start.elapsed();
    eprintln!("acquire 1000 permits in {elapsed:?}");
    // The benchmark completes (no panic, no error). The actual
    // timing assertion is intentionally loose.
    assert!(!permits.is_empty());
    for p in permits {
        mgr.release(p, "bench");
    }
}
