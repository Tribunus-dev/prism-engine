//! Memory subsystem data types and pure abstractions.
//!
//! This module owns the canonical authority for the engine-independent
//! memory data types: pressure levels, statistics, engine pool, memory
//! enforcer, pre-computed memory plan data types, unified telemetry
//! snapshot, and the Core ML ANE warmup helper. Execution-plane code
//! that depends on engine-internal `Arena`, `ExternalStorage`,
//! `MappedSegment`, and `worker_memory` stays engine-side at
//! `compute-core/src/ecs/memory_impl/`.

pub mod ane_warmup_mil;
pub mod coreai_warmup;
pub mod enforcer;
pub mod monitor;
pub mod plan;
pub mod pool;
pub mod telemetry;

pub use enforcer::{MemoryAction, MemoryEnforcer};
pub use monitor::{MemoryMonitor, MemoryStats};
pub use plan::{MemoryPlan, MemoryPlanSlot};
pub use pool::{EngineEntry, EngineLifecycle, EnginePool};

/// Memory pressure level.
///
/// Ordered; ascending values indicate escalating pressure. Use
/// [`MemoryMonitor::pressure`](monitor::MemoryMonitor::pressure) to
/// obtain a current value from a stats snapshot, and
/// [`MemoryEnforcer::enforce`](enforcer::MemoryEnforcer::enforce) to
/// dispatch mitigation actions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MemoryPressure {
    Normal = 0,
    Warning = 1,
    Critical = 2,
    Severe = 3,
    Oom = 4,
}
