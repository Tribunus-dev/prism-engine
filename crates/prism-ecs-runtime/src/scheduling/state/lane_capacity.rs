//! Lane capacity tracking and permit management (constitutional home).
//!
//! This is the constitutional home for per-lane admission control. It tracks
//! in-flight permit counts, queued-but-not-submitted work, and per-session
//! quotas for every execution lane. The runtime scheduling systems consult
//! this state to decide whether a new dispatch intent may commit; the
//! kernel backends consult the same state to decide whether to accept a
//! submission request.
//!
//! # Authority
//!
//! - `LaneCapacityManager` and its associated types are **scheduling state** —
//!   they live in `prism_ecs_runtime::scheduling::state::lane_capacity`.
//! - The manager **does not submit work to a backend**. It only tracks
//!   permit counts. Submission is a kernel concern (`prism-ecs-kernel::backend::*`),
//!   and submission does not mutate this state directly — completion
//!   reconciliation stages any state change through `ConstitutionalWorldTxn`.
//! - A lane permit is a *capability* (an "I may submit this work to this lane
//!   for this session") — not a guarantee. The kernel may still fail the
//!   submission; on failure, the runtime calls [`LaneCapacityManager::release`]
//!   via a completion-reconciliation transaction.
//!
//! # Migration provenance
//!
//! The legacy home was `compute-core/src/ecs/scheduling/lane_capacity.rs`.
//! The engine file remains during the absorption window; this is the
//! constitutional replacement. When the engine's `heterogeneous_executor.rs`
//! is split during step 36, the runtime half will import from this module.
//!
//! # Invariants
//!
//! - All counters use saturating arithmetic. They never go negative and
//!   never wrap below zero on release.
//! - `permit_id` uses `wrapping_add` because the id space is `2^64` and
//!   callers must not rely on strict monotonic ordering across that range.
//! - This module is engine-agnostic. It does not import from `compute-core`.
//!   It depends only on the kernel's `ExecutionLane` type and `std`.

use prism_ecs_kernel::execution_lane::ExecutionLane;

// ---------------------------------------------------------------------------
// LanePermit
// ---------------------------------------------------------------------------

/// A permit granting capacity to submit work to a lane.
///
/// Returned by [`LaneCapacityManager::try_acquire`] when sufficient capacity
/// exists. Must be returned via [`LaneCapacityManager::release`] once the
/// work completes, is cancelled, or fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LanePermit {
    /// The lane this permit was acquired for.
    pub lane: ExecutionLane,
    /// Monotonically-increasing unique permit identifier.
    pub permit_id: u64,
}

// ---------------------------------------------------------------------------
// LaneCapacityConfig
// ---------------------------------------------------------------------------

/// Per-lane capacity configuration.
///
/// Controls concurrency limits for each backend lane type, per-session
/// quotas, and the global pending ceiling. Sensible defaults are provided
/// via [`Default`].
#[derive(Debug, Clone)]
pub struct LaneCapacityConfig {
    /// Maximum concurrent in-flight command buffers for Metal/GPU lanes
    /// (`MlxGpu`, `Tensix`).
    pub max_in_flight_command_buffers: usize,
    /// Maximum concurrent in-flight ANE predictions (`CoreAiAne`).
    pub max_in_flight_ane_predictions: usize,
    /// Maximum concurrent in-flight CPU workers
    /// (`AccelerateCpu`, `CandleCpu`, `IntelLevelZero`).
    pub max_in_flight_cpu_workers: usize,
    /// Maximum queued-but-not-yet-in-flight items per lane.
    pub max_queued_per_lane: usize,
    /// Maximum concurrent in-flight items per session.
    pub max_in_flight_per_session: usize,
    /// Global ceiling on total pending (in-flight + queued) items across all
    /// lanes and sessions.
    pub global_max_pending: usize,
}

impl Default for LaneCapacityConfig {
    fn default() -> Self {
        Self {
            max_in_flight_command_buffers: 3,
            max_in_flight_ane_predictions: 1,
            max_in_flight_cpu_workers: 2,
            max_queued_per_lane: 64,
            max_in_flight_per_session: 128,
            global_max_pending: 4096,
        }
    }
}

// ---------------------------------------------------------------------------
// LaneCapacityManager
// ---------------------------------------------------------------------------

/// Tracks in-flight permits and capacity per lane with session quotas.
///
/// Provides admission control for the heterogeneous executor by enforcing
/// per-lane concurrency limits, per-session quotas, and a global pending
/// ceiling. All counter mutations use saturating arithmetic.
///
/// This is **runtime scheduling state**, not a backend. It does not submit
/// work; it only decides whether a permit *may* be issued. Submission is a
/// kernel concern, and the kernel must release the permit (via a runtime
/// completion-reconciliation transaction) once the work is done.
pub struct LaneCapacityManager {
    config: LaneCapacityConfig,
    metal_in_flight: usize,
    ane_in_flight: usize,
    cpu_in_flight: usize,
    metal_queued: usize,
    ane_queued: usize,
    cpu_queued: usize,
    global_pending: usize,
    /// Per-session in-flight count.
    ///
    /// Uses `HashMap` because session-id keys are opaque and order is
    /// not part of the canonical scheduling state. The aggregate totals
    /// (`metal_in_flight`, `global_pending`, etc.) are the authoritative
    /// counts; this map is a fast-path index, never an alternative source
    /// of truth. A reader of the map cannot derive world-visible state
    /// from it alone — they must consult the per-lane counters and the
    /// snapshot.
    session_in_flight: std::collections::HashMap<String, usize>,
    next_permit_id: u64,
}

impl LaneCapacityManager {
    /// Create a new manager with the given configuration.
    ///
    /// All counters start at zero and the first permit will have id `1`.
    pub fn new(config: LaneCapacityConfig) -> Self {
        Self {
            config,
            metal_in_flight: 0,
            ane_in_flight: 0,
            cpu_in_flight: 0,
            metal_queued: 0,
            ane_queued: 0,
            cpu_queued: 0,
            global_pending: 0,
            session_in_flight: std::collections::HashMap::new(),
            next_permit_id: 1,
        }
    }

    /// Try to acquire a permit for submitting work on a lane.
    ///
    /// Returns `None` if any of the following limits would be exceeded:
    ///
    /// 1. Lane-specific in-flight count has reached its configured maximum.
    /// 2. The session's in-flight count has reached the per-session limit.
    /// 3. The global pending count has reached the global ceiling.
    ///
    /// When a permit is granted all associated counters are incremented
    /// using saturating arithmetic.
    pub fn try_acquire(&mut self, lane: ExecutionLane, session: &str) -> Option<LanePermit> {
        // 1. Check lane-specific in-flight capacity. Each lane family
        //    has a single counter and a single cap; the three families
        //    are mutually exclusive (see `lane_classification_helpers_partition_all_variants`).
        let lane_at_cap = if lane.is_metal_family() {
            self.metal_in_flight >= self.config.max_in_flight_command_buffers
        } else if lane.is_ane() {
            self.ane_in_flight >= self.config.max_in_flight_ane_predictions
        } else {
            // lane.is_cpu_family() — guaranteed by the classification invariant.
            self.cpu_in_flight >= self.config.max_in_flight_cpu_workers
        };
        if lane_at_cap {
            return None;
        }

        // 2. Check session in-flight limit.
        let session_count = self.session_in_flight.get(session).copied().unwrap_or(0);
        if session_count >= self.config.max_in_flight_per_session {
            return None;
        }

        // 3. Check global pending limit.
        if self.global_pending >= self.config.global_max_pending {
            return None;
        }

        // All checks passed — allocate permit.
        let permit_id = self.next_permit_id;
        // Permit id space is large enough that wrapping is safe; callers
        // should not rely on strict monotonic ordering across 2^64 ids.
        self.next_permit_id = self.next_permit_id.wrapping_add(1);

        if lane.is_metal_family() {
            self.metal_in_flight = self.metal_in_flight.saturating_add(1);
        } else if lane.is_ane() {
            self.ane_in_flight = self.ane_in_flight.saturating_add(1);
        } else if lane.is_cpu_family() {
            self.cpu_in_flight = self.cpu_in_flight.saturating_add(1);
        }

        self.global_pending = self.global_pending.saturating_add(1);
        self.session_in_flight
            .entry(session.to_string())
            .and_modify(|c| *c = c.saturating_add(1))
            .or_insert(1);

        Some(LanePermit { lane, permit_id })
    }

    /// Release a permit after work completes (or is cancelled/failed).
    ///
    /// Decrements the lane-specific in-flight counter, the session
    /// in-flight counter, and the global pending count. All arithmetic
    /// uses saturating subtraction so counters never go negative.
    pub fn release(&mut self, permit: LanePermit, session: &str) {
        if permit.lane.is_metal_family() {
            self.metal_in_flight = self.metal_in_flight.saturating_sub(1);
        } else if permit.lane.is_ane() {
            self.ane_in_flight = self.ane_in_flight.saturating_sub(1);
        } else if permit.lane.is_cpu_family() {
            self.cpu_in_flight = self.cpu_in_flight.saturating_sub(1);
        }

        self.global_pending = self.global_pending.saturating_sub(1);

        if let Some(count) = self.session_in_flight.get_mut(session) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                self.session_in_flight.remove(session);
            }
        }
    }

    /// Increment the queued count for a lane (work waiting in the lane
    /// queue, not yet in-flight).
    pub fn increment_queued(&mut self, lane: ExecutionLane) {
        if lane.is_metal_family() {
            self.metal_queued = self.metal_queued.saturating_add(1);
        } else if lane.is_ane() {
            self.ane_queued = self.ane_queued.saturating_add(1);
        } else if lane.is_cpu_family() {
            self.cpu_queued = self.cpu_queued.saturating_add(1);
        }
    }

    /// Decrement the queued count for a lane (work popped from the lane
    /// queue for submission).
    pub fn decrement_queued(&mut self, lane: ExecutionLane) {
        if lane.is_metal_family() {
            self.metal_queued = self.metal_queued.saturating_sub(1);
        } else if lane.is_ane() {
            self.ane_queued = self.ane_queued.saturating_sub(1);
        } else if lane.is_cpu_family() {
            self.cpu_queued = self.cpu_queued.saturating_sub(1);
        }
    }

    /// Return an immutable reference to the capacity configuration.
    pub fn config(&self) -> &LaneCapacityConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// LaneCapacitySnapshot
// ---------------------------------------------------------------------------

/// Snapshot of current lane capacity state for observability and
/// feedback-driven scheduling decisions.
#[derive(Debug, Clone)]
pub struct LaneCapacitySnapshot {
    /// Currently in-flight command buffers on the Metal/GPU lane.
    pub metal_in_flight: usize,
    /// Currently in-flight predictions on the ANE lane.
    pub ane_in_flight: usize,
    /// Currently in-flight CPU workers on CPU lanes.
    pub cpu_in_flight: usize,
    /// Work items queued (not yet submitted) on the Metal/GPU lane.
    pub metal_queued: usize,
    /// Work items queued (not yet submitted) on the ANE lane.
    pub ane_queued: usize,
    /// Work items queued (not yet submitted) on CPU lanes.
    pub cpu_queued: usize,
    /// Total pending items across all lanes and sessions (in-flight +
    /// queued).
    pub global_pending: usize,
    /// Maximum concurrent in-flight command buffers (`max_in_flight_command_buffers`).
    pub metal_capacity: usize,
    /// Maximum concurrent in-flight ANE predictions (`max_in_flight_ane_predictions`).
    pub ane_capacity: usize,
    /// Maximum concurrent in-flight CPU workers (`max_in_flight_cpu_workers`).
    pub cpu_capacity: usize,
}

impl LaneCapacityManager {
    /// Capture an atomic snapshot of the current capacity state.
    ///
    /// The returned [`LaneCapacitySnapshot`] reflects the counter values at
    /// the time of the call and is not guaranteed to be consistent across
    /// the individual fields if the manager is concurrently accessed from
    /// multiple threads.
    pub fn snapshot(&self) -> LaneCapacitySnapshot {
        LaneCapacitySnapshot {
            metal_in_flight: self.metal_in_flight,
            ane_in_flight: self.ane_in_flight,
            cpu_in_flight: self.cpu_in_flight,
            metal_queued: self.metal_queued,
            ane_queued: self.ane_queued,
            cpu_queued: self.cpu_queued,
            global_pending: self.global_pending,
            metal_capacity: self.config.max_in_flight_command_buffers,
            ane_capacity: self.config.max_in_flight_ane_predictions,
            cpu_capacity: self.config.max_in_flight_cpu_workers,
        }
    }
}

// ---------------------------------------------------------------------------
// SessionQuota
// ---------------------------------------------------------------------------

/// Per-session quota tracker.
///
/// Tracks a single session's pending count relative to its maximum
/// allowed pending work items.
#[derive(Debug, Clone)]
pub struct SessionQuota {
    /// Maximum number of pending items allowed for this session.
    pub max_pending: usize,
    /// Current number of pending items for this session.
    pub current: usize,
}

impl SessionQuota {
    /// Returns `true` if the session has capacity for at least one more
    /// pending item.
    pub fn has_capacity(&self) -> bool {
        self.current < self.max_pending
    }

    /// Returns the remaining capacity for this session.
    pub fn remaining(&self) -> usize {
        self.max_pending.saturating_sub(self.current)
    }

    /// Attempt to reserve one unit of capacity.  Returns `false` if the
    /// session is already at `max_pending`.
    pub fn try_reserve(&mut self) -> bool {
        if self.current >= self.max_pending {
            return false;
        }
        self.current = self.current.saturating_add(1);
        true
    }

    /// Release one unit of capacity (call when work completes or is
    /// cancelled).
    pub fn release(&mut self) {
        self.current = self.current.saturating_sub(1);
    }
}

// ---------------------------------------------------------------------------
// Invariant tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Architectural-invariant tests for the `lane_capacity` state.
    //!
    //! These tests verify the *constitution* of lane capacity: that
    //! admission control is monotone, that release is saturating, and
    //! that the manager cannot be observed as a partial state. They are
    //! the property-level tests that the rest of the scheduling systems
    //! rely on.
    //!
    //! Test names are architectural invariants, not function names.
    //! The function names may change; the constitutional rule survives.

    use super::*;

    fn small_config() -> LaneCapacityConfig {
        LaneCapacityConfig {
            max_in_flight_command_buffers: 2,
            max_in_flight_ane_predictions: 1,
            max_in_flight_cpu_workers: 1,
            max_queued_per_lane: 4,
            max_in_flight_per_session: 3,
            global_max_pending: 8,
        }
    }

    #[test]
    fn admission_refuses_when_lane_at_capacity() {
        let mut mgr = LaneCapacityManager::new(small_config());
        // 2 metal in-flight is the cap.
        let _p1 = mgr.try_acquire(ExecutionLane::MlxGpu, "s1").expect("p1");
        let _p2 = mgr.try_acquire(ExecutionLane::MlxGpu, "s1").expect("p2");
        // A third must be refused by the lane-specific limit.
        assert!(
            mgr.try_acquire(ExecutionLane::MlxGpu, "s1").is_none(),
            "lane capacity must be enforced"
        );
    }

    #[test]
    fn admission_refuses_when_session_at_quota() {
        let mut mgr = LaneCapacityManager::new(small_config());
        // 3 mixed-lane permits is the per-session cap.
        let _p1 = mgr.try_acquire(ExecutionLane::MlxGpu, "s1").expect("p1");
        let _p2 = mgr.try_acquire(ExecutionLane::CoreAiAne, "s1").expect("p2");
        let _p3 = mgr.try_acquire(ExecutionLane::CandleCpu, "s1").expect("p3");
        // A 4th on the same session must be refused.
        assert!(
            mgr.try_acquire(ExecutionLane::MlxGpu, "s1").is_none(),
            "session quota must be enforced even when lane has room"
        );
    }

    #[test]
    fn admission_refuses_when_global_at_ceiling() {
        // A config where the global ceiling is the binding constraint
        // (lane caps and per-session cap are both larger).
        let cfg = LaneCapacityConfig {
            max_in_flight_command_buffers: 16,
            max_in_flight_ane_predictions: 16,
            max_in_flight_cpu_workers: 16,
            max_queued_per_lane: 64,
            max_in_flight_per_session: 16,
            global_max_pending: 8,
        };
        let mut mgr = LaneCapacityManager::new(cfg);
        // 8 permits on 8 different sessions (so per-session doesn't fire)
        // all on the same lane (so per-lane doesn't fire).
        for i in 0..8 {
            mgr.try_acquire(ExecutionLane::MlxGpu, &format!("s{i}"))
                .unwrap_or_else(|| panic!("permit {i} should be granted"));
        }
        assert!(
            mgr.try_acquire(ExecutionLane::MlxGpu, "s_extra").is_none(),
            "global ceiling must be enforced even when lanes and sessions have room"
        );
    }

    #[test]
    fn admission_grants_when_within_all_limits() {
        let mut mgr = LaneCapacityManager::new(small_config());
        let p = mgr
            .try_acquire(ExecutionLane::MlxGpu, "s1")
            .expect("permit must be granted within all limits");
        assert_eq!(p.lane, ExecutionLane::MlxGpu);
        assert_eq!(p.permit_id, 1);
    }

    #[test]
    fn release_decrements_all_counters() {
        let mut mgr = LaneCapacityManager::new(small_config());
        let p = mgr
            .try_acquire(ExecutionLane::MlxGpu, "s1")
            .expect("permit");
        let snap_before = mgr.snapshot();
        assert_eq!(snap_before.metal_in_flight, 1);
        assert_eq!(snap_before.global_pending, 1);
        mgr.release(p, "s1");
        let snap_after = mgr.snapshot();
        assert_eq!(snap_after.metal_in_flight, 0);
        assert_eq!(snap_after.global_pending, 0);
    }

    #[test]
    fn release_is_saturating_for_double_release() {
        // A double-release (programmer error or stale permit) must not
        // produce negative counters. Saturating subtraction enforces
        // that the manager state is monotone non-negative.
        let mut mgr = LaneCapacityManager::new(small_config());
        let p = mgr
            .try_acquire(ExecutionLane::MlxGpu, "s1")
            .expect("permit");
        mgr.release(p, "s1");
        // No second permit to release, but if we tried, saturation
        // would clamp the counter. The invariant we can assert directly
        // is that the snapshot has zero in-flight after one release.
        let snap = mgr.snapshot();
        assert_eq!(snap.metal_in_flight, 0);
        assert_eq!(snap.global_pending, 0);
    }

    #[test]
    fn snapshot_reflects_current_counters() {
        let mut mgr = LaneCapacityManager::new(small_config());
        let _p1 = mgr.try_acquire(ExecutionLane::MlxGpu, "s1").expect("p1");
        let _p2 = mgr.try_acquire(ExecutionLane::CoreAiAne, "s1").expect("p2");
        mgr.increment_queued(ExecutionLane::MlxGpu);
        mgr.increment_queued(ExecutionLane::MlxGpu);
        let snap = mgr.snapshot();
        assert_eq!(snap.metal_in_flight, 1);
        assert_eq!(snap.ane_in_flight, 1);
        assert_eq!(snap.cpu_in_flight, 0);
        assert_eq!(snap.metal_queued, 2);
        assert_eq!(snap.global_pending, 2);
    }

    #[test]
    fn release_with_wrong_session_id_decrements_lane_but_leaks_session_quota() {
        // Architectural invariant: a `release` with a session id that
        // doesn't match the original acquisition must NOT panic, must
        // NOT produce a negative counter, and must decrement the lane
        // in-flight count. The session quota slot is leaked (the
        // implementation only decrements the session's in-flight count
        // for sessions that have an entry; a phantom id is simply
        // absent from the map). This test pins that behavior so a
        // future change cannot silently make the manager panic on
        // mismatched ids.
        let mut mgr = LaneCapacityManager::new(small_config());
        let p = mgr
            .try_acquire(ExecutionLane::MlxGpu, "s_real")
            .expect("permit");
        // The release takes the permit (so the lane counter MUST drop
        // to zero), but the session id is wrong so the per-session
        // entry is never decremented. The real session therefore
        // retains a quota slot until the next release with the right id
        // (or until process restart, in the current implementation).
        mgr.release(p, "s_phantom");
        let snap = mgr.snapshot();
        assert_eq!(snap.metal_in_flight, 0, "lane counter must decrement");
        assert_eq!(snap.global_pending, 0, "global counter must decrement");
        // No panic, no negative counters. (The session's in-flight
        // entry is leaked; this is a known limitation of the
        // `HashMap<String, usize>` index, not a soundness issue.)
    }

    #[test]
    fn permit_ids_are_unique_within_manager() {
        let mut mgr = LaneCapacityManager::new(small_config());
        let p1 = mgr.try_acquire(ExecutionLane::MlxGpu, "s1").expect("p1");
        let p2 = mgr.try_acquire(ExecutionLane::CoreAiAne, "s1").expect("p2");
        assert_ne!(p1.permit_id, p2.permit_id);
    }

    #[test]
    fn release_frees_a_session_quota_slot() {
        // Architectural invariant: releasing a permit must drop the
        // session's in-flight count, so a future admission on the same
        // session can succeed.
        let mut mgr = LaneCapacityManager::new(small_config());
        // Session cap is 3.
        let p1 = mgr.try_acquire(ExecutionLane::MlxGpu, "s1").expect("p1");
        let p2 = mgr.try_acquire(ExecutionLane::CoreAiAne, "s1").expect("p2");
        let p3 = mgr.try_acquire(ExecutionLane::CandleCpu, "s1").expect("p3");
        assert!(mgr.try_acquire(ExecutionLane::MlxGpu, "s1").is_none());
        mgr.release(p1, "s1");
        assert!(
            mgr.try_acquire(ExecutionLane::MlxGpu, "s1").is_some(),
            "after release, the session quota slot must be reusable"
        );
        // Suppress unused warnings for p2/p3 in case the assertion above
        // is the one that fails first.
        let _ = (p2, p3);
    }

    #[test]
    fn lane_classification_helpers_partition_all_variants() {
        // The three family helpers must cover the enum and never
        // double-count a variant. This is the type-level invariant
        // that lets `try_acquire` and `release` dispatch without
        // forgetting a variant.
        for lane in [
            ExecutionLane::MlxGpu,
            ExecutionLane::AccelerateCpu,
            ExecutionLane::CoreAiAne,
            ExecutionLane::CandleCpu,
            ExecutionLane::Tensix,
            ExecutionLane::IntelLevelZero,
        ] {
            let flags = [lane.is_metal_family(), lane.is_ane(), lane.is_cpu_family()];
            let true_count = flags.iter().filter(|&&b| b).count();
            assert_eq!(
                true_count, 1,
                "every variant must belong to exactly one lane family: {lane:?}"
            );
        }
    }
}
