//! Runtime configuration for ECS worker supervision (Slice 2).
//!
//! Mode changes are process-start-only in Slice 2 — the resource is
//! inserted once at World initialisation and read by supervision systems
//! during each tick.

/// Supervision mode for ECS worker management.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EcsWorkerSupervisionMode {
    /// Worker supervision is completely disabled — no workers are created
    /// or managed.
    Disabled,
    /// Full worker supervision is active — workers are created, monitored,
    /// and lifecycle events are enforced.
    Enabled,
}

/// Configuration resource inserted into the ECS World at startup.
///
/// Read by supervision systems (watchdog, event drain, etc.) to determine
/// the level of worker lifecycle enforcement.
#[derive(Debug, Clone)]
pub struct EcsWorkerSupervisionConfig {
    pub mode: EcsWorkerSupervisionMode,
}

impl Default for EcsWorkerSupervisionConfig {
    fn default() -> Self {
        Self {
            mode: EcsWorkerSupervisionMode::Disabled,
        }
    }
}
