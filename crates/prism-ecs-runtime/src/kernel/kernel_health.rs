//! Canonical kernel health snapshot.
//!
//! Authority: this module owns the canonical read-only health summary of
//! the runtime kernel — entity count, world epoch, journal sequence,
//! receipt sequence, last snapshot watermark, and runtime status string.
//! It is the single shape the daemon reports to operators.
//!
//! Classification: canonical. The computation reads the world under its
//! read-lock and never mutates canonical state. No hardware, no `unsafe`,
//! no process-local state.

use crate::ports::RuntimeError;
use prism_ecs_core::WorldEpoch;
use std::sync::atomic::Ordering;
use std::sync::Arc;

/// Canonical health summary of a `RuntimeKernel` instance.
///
/// All fields are derived from the kernel's authoritative state at the
/// moment of the call. `status` is the coarse runtime status string
/// (currently always `"running"`).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct KernelHealth {
    pub entity_count: usize,
    pub world_epoch: u64,
    pub journal_sequence: u64,
    pub receipt_sequence: u64,
    pub last_snapshot_epoch: u64,
    pub last_snapshot_sequence: u64,
    pub status: String,
}

/// Compute the canonical health of the kernel over a borrowed world and
/// sequence counter.
///
/// `last_snapshot_epoch` and `last_snapshot_sequence` are not tracked by
/// the kernel at the time of this decomposition; they are exposed as
/// fields because the operational contract reserves them for the snapshot
/// store to populate. The default value of `0` is the documented
/// "no snapshot taken yet" sentinel.
pub fn compute_health(
    world: &prism_ecs_core::World,
    sequence: &std::sync::atomic::AtomicU64,
) -> Result<KernelHealth, RuntimeError> {
    let epoch: WorldEpoch = world.current_epoch();
    let entity_count = world.all_entities().len();
    let seq = sequence.load(Ordering::Relaxed);
    Ok(KernelHealth {
        entity_count,
        world_epoch: epoch.0,
        journal_sequence: seq,
        receipt_sequence: seq,
        last_snapshot_epoch: 0,
        last_snapshot_sequence: 0,
        status: "running".to_string(),
    })
}

/// Compute the kernel's health over a `World` that is shared via
/// `Arc<RwLock<…>>` — the shape used by `RuntimeKernelInner`.
///
/// Acquires a read lock on the world for the duration of the projection.
/// Returns the same error type as the world lock would (a poisoned
/// `RwLock` is reported as `RuntimeError::Entity`).
pub fn compute_health_locked(
    world_lock: &Arc<std::sync::RwLock<prism_ecs_core::World>>,
    sequence: &std::sync::atomic::AtomicU64,
) -> Result<KernelHealth, RuntimeError> {
    let world = world_lock
        .read()
        .map_err(|e| RuntimeError::Entity(format!("world read lock poisoned: {e}")))?;
    compute_health(&world, sequence)
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ecs_core::{EntityKind, World};
    use std::sync::atomic::AtomicU64;
    use std::sync::Arc;

    /// A fresh world reports zero entities and epoch 0.
    #[test]
    fn fresh_world_reports_zero_state() {
        let world = World::new();
        let seq = AtomicU64::new(0);
        let health = compute_health(&world, &seq).expect("compute");
        assert_eq!(health.entity_count, 0);
        assert_eq!(health.world_epoch, 0);
        assert_eq!(health.journal_sequence, 0);
        assert_eq!(health.receipt_sequence, 0);
        assert_eq!(health.status, "running");
        assert_eq!(health.last_snapshot_epoch, 0);
    }

    /// Entity count grows as entities are spawned.
    #[test]
    fn entity_count_grows_with_spawns() {
        let mut world = World::new();
        world.spawn(EntityKind::WorkUnit, None).expect("spawn 1");
        world.spawn(EntityKind::WorkUnit, None).expect("spawn 2");
        let seq = AtomicU64::new(7);
        let health = compute_health(&world, &seq).expect("compute");
        assert_eq!(health.entity_count, 2);
        assert_eq!(health.journal_sequence, 7);
    }

    /// The locked variant observes the same facts as the unlocked one.
    #[test]
    fn locked_health_matches_unlocked() {
        let mut world = World::new();
        world.spawn(EntityKind::Agent, None).expect("spawn");
        let seq = AtomicU64::new(3);
        let lock = Arc::new(std::sync::RwLock::new(world));
        let health = compute_health_locked(&lock, &seq).expect("compute");
        assert_eq!(health.entity_count, 1);
        assert_eq!(health.journal_sequence, 3);
    }
}
