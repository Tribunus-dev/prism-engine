//! PRISM-PRODUCTION-HETEROGENEOUS-EXECUTOR-0001 — Lane capacity tracking and permit management.
//!
//! Tracks capacity and permits for each execution lane. Manages slot-based
//! concurrency limits, command-buffer permits, and per-session quotas.
//! All counters are conservative using saturating arithmetic — they never
//! go negative or wrap below zero.

use crate::backend::placement::ExecutionLane;

// ---------------------------------------------------------------------------
// LanePermit
// ---------------------------------------------------------------------------

/// A permit granting capacity to submit work to a lane.
///
/// Returned by [`LaneCapacityManager::try_acquire`] when sufficient capacity
/// exists.  Must be returned via [`LaneCapacityManager::release`] once the
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
/// quotas, and the global pending ceiling.  Sensible defaults are provided
/// via [`Default`].
#[derive(Debug, Clone)]
pub struct LaneCapacityConfig {
    /// Maximum concurrent in-flight command buffers for Metal/GPU lanes
    /// (`MlxGpu`, `Tensix`).
    pub max_in_flight_command_buffers: usize,
    /// Maximum concurrent in-flight ANE predictions (`CoreMlAne`).
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
/// ceiling.  All counter mutations use saturating arithmetic.
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
        // 1. Check lane-specific in-flight capacity.
        match lane {
            ExecutionLane::MlxGpu | ExecutionLane::Tensix => {
                if self.metal_in_flight >= self.config.max_in_flight_command_buffers {
                    return None;
                }
            }
            ExecutionLane::CoreMlAne => {
                if self.ane_in_flight >= self.config.max_in_flight_ane_predictions {
                    return None;
                }
            }
            ExecutionLane::AccelerateCpu
            | ExecutionLane::CandleCpu
            | ExecutionLane::IntelLevelZero => {
                if self.cpu_in_flight >= self.config.max_in_flight_cpu_workers {
                    return None;
                }
            }
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

        match lane {
            ExecutionLane::MlxGpu | ExecutionLane::Tensix => {
                self.metal_in_flight = self.metal_in_flight.saturating_add(1);
            }
            ExecutionLane::CoreMlAne => {
                self.ane_in_flight = self.ane_in_flight.saturating_add(1);
            }
            ExecutionLane::AccelerateCpu
            | ExecutionLane::CandleCpu
            | ExecutionLane::IntelLevelZero => {
                self.cpu_in_flight = self.cpu_in_flight.saturating_add(1);
            }
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
    /// in-flight counter, and the global pending count.  All arithmetic
    /// uses saturating subtraction so counters never go negative.
    pub fn release(&mut self, permit: LanePermit, session: &str) {
        match permit.lane {
            ExecutionLane::MlxGpu | ExecutionLane::Tensix => {
                self.metal_in_flight = self.metal_in_flight.saturating_sub(1);
            }
            ExecutionLane::CoreMlAne => {
                self.ane_in_flight = self.ane_in_flight.saturating_sub(1);
            }
            ExecutionLane::AccelerateCpu
            | ExecutionLane::CandleCpu
            | ExecutionLane::IntelLevelZero => {
                self.cpu_in_flight = self.cpu_in_flight.saturating_sub(1);
            }
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
        match lane {
            ExecutionLane::MlxGpu | ExecutionLane::Tensix => {
                self.metal_queued = self.metal_queued.saturating_add(1);
            }
            ExecutionLane::CoreMlAne => {
                self.ane_queued = self.ane_queued.saturating_add(1);
            }
            ExecutionLane::AccelerateCpu
            | ExecutionLane::CandleCpu
            | ExecutionLane::IntelLevelZero => {
                self.cpu_queued = self.cpu_queued.saturating_add(1);
            }
        }
    }

    /// Decrement the queued count for a lane (work popped from the lane
    /// queue for submission).
    pub fn decrement_queued(&mut self, lane: ExecutionLane) {
        match lane {
            ExecutionLane::MlxGpu | ExecutionLane::Tensix => {
                self.metal_queued = self.metal_queued.saturating_sub(1);
            }
            ExecutionLane::CoreMlAne => {
                self.ane_queued = self.ane_queued.saturating_sub(1);
            }
            ExecutionLane::AccelerateCpu
            | ExecutionLane::CandleCpu
            | ExecutionLane::IntelLevelZero => {
                self.cpu_queued = self.cpu_queued.saturating_sub(1);
            }
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
