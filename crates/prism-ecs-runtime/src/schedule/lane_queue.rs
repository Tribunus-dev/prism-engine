//! Per-lane bounded work queue with priority ordering.
//!
//! Authority: this module owns the canonical admission queue for one
//! execution lane, enforcing the priority order, bounded depth, and
//! typed backpressure reason that the heterogeneous executor observes
//! before dispatch. It does not own lane capacity counters (those
//! live in `lane_capacity`), the lane-executor trait (that lives in
//! `lane_work`), or the orchestrator actor (that lives in
//! `heterogeneous_executor`).
//!
//! Constitutional notes:
//!
//! - The lane is identified by [`LaneId`], a typed newtype that is
//!   independent of the engine's `ExecutionLane` enum. Adapters
//!   convert at the boundary.
//! - All canonical collections use `BTreeMap` per the
//!   "no HashMap/HashSet for canonical collections" rule.
//! - All fallible operations return [`Result<_, LaneQueueError>`],
//!   a `thiserror`-derived enum classified as `Rejected` (queue
//!   full → admission gate refused) and `NotFound` (work id
//!   absent from the queue). The error is the constitutional
//!   surface, not a `String`.

use std::collections::{BTreeMap, VecDeque};

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ── Lane identity ─────────────────────────────────────────────────────────

/// Typed identity for an execution lane. The constitutional side does
/// not import the engine's `ExecutionLane` enum; adapters convert at
/// the boundary. Newtype around `u32` so the wire format is a single
/// machine word.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct LaneId(pub u32);

impl LaneId {
    /// Construct a [`LaneId`] from a raw `u32`.
    pub const fn new(raw: u32) -> Self {
        Self(raw)
    }

    /// Borrow the raw lane ordinal.
    pub const fn raw(self) -> u32 {
        self.0
    }
}

/// Typed identity for one queued work item. Newtype around `u64`; the
/// work-id allocator (`next_work_id`) hands out monotonic ids. The
/// type says "this is a work id, not a lease id and not a command
/// id," which is the constitutional rule from `references/rust-quality.md`.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct WorkId(pub u64);

impl WorkId {
    /// Construct a [`WorkId`] from a raw `u64`.
    pub const fn new(raw: u64) -> Self {
        Self(raw)
    }

    /// Borrow the raw work ordinal.
    pub const fn raw(self) -> u64 {
        self.0
    }
}

// ── Priority ──────────────────────────────────────────────────────────────

/// Priority level for queued work.
///
/// Higher-priority entries are dequeued first. Within the same
/// priority level the queue preserves FIFO order.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
pub enum WorkPriority {
    /// Background compilation tasks (model load, pipeline compilation).
    Compilation,
    /// Warmup / pre-heat runs before latency-critical requests.
    Warmup,
    /// Low-priority background inference.
    Low,
    /// Normal request (default).
    #[default]
    Normal,
    /// Elevated priority for interactive use.
    High,
    /// Highest priority — user-facing interactive sessions.
    Interactive,
}

impl WorkPriority {
    /// True if this priority is at or above `Interactive` (i.e. user
    /// must not be throttled under backpressure).
    pub const fn is_user_facing(self) -> bool {
        matches!(self, Self::Interactive | Self::High)
    }
}

// ── Backpressure reason ───────────────────────────────────────────────────

/// Typed reason why a `try_push` was rejected. The orchestrator
/// observes this and adjusts the admission gate. The mapping from
/// [`LaneId`] to reason is the lane-class decision (metal/ane/cpu);
/// the queue itself just carries the reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum BackpressureReason {
    /// Metal/GPU lane is at capacity.
    MetalCapacity,
    /// ANE / Core ML lane is at capacity.
    AneCapacity,
    /// Accelerate / CPU lane is at capacity.
    CpuCapacity,
    /// Activation slot reservation limit reached.
    ActivationSlots,
    /// IOSurface pool exhausted.
    IOSurfacePool,
    /// Session-level quota exceeded.
    SessionQuota,
    /// Global orchestrator queue full.
    GlobalQueue,
    /// Artifact cold — weights not resident.
    ArtifactCold,
}

impl BackpressureReason {
    /// Map a [`LaneId`] to the most specific reason for that lane
    /// class. Lanes that are not first-class in the queue set
    /// fall through to a conservative CPU capacity reason.
    pub const fn for_lane(lane: LaneId) -> Self {
        // The lane-id space is a flat `u32`; the lane-class decision
        // is the orchestrator's job (it knows the engine's
        // `ExecutionLane` → LaneId mapping). For the canonical
        // first-class lanes, the orchestrator passes:
        //   0 → MetalCapacity, 1 → AneCapacity, 2 → CpuCapacity.
        // Higher ordinals map to ActivationSlots (the conservative
        // shared-pool reason).
        match lane.0 {
            0 => Self::MetalCapacity,
            1 => Self::AneCapacity,
            2..=5 => Self::CpuCapacity,
            _ => Self::ActivationSlots,
        }
    }

    /// Whether this reason is a transient (recoverable) condition.
    /// All built-in reasons are transient; the orchestrator can retry.
    pub const fn is_transient(self) -> bool {
        true
    }
}

// ── Queue entry ───────────────────────────────────────────────────────────

/// A single entry in a lane queue, carrying priority, deadline, and
/// the lane-specific work descriptor. The `request` field is the
/// opaque work descriptor the orchestrator hands to the lane
/// executor; the constitutional side treats it as a typed envelope.
///
/// Note: `deadline` is `#[serde(skip)]` because `std::time::Instant`
/// does not implement `Serialize`/`Deserialize` (the wire format is
/// the `Instant`'s nanos-since-epoch or the monotonic delta, neither
/// of which is round-trippable across processes). The deadline is
/// an in-process advisory field; admission time is the durable
/// record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueueEntry {
    /// Unique work identifier (matches the `WorkId` the orchestrator
    /// allocated at admission time).
    pub work_id: WorkId,
    /// Priority (higher = dequeued sooner).
    pub priority: WorkPriority,
    /// Optional deadline — used for timeout checks, not enforced by
    /// the queue itself. Not serialized; see struct-level docs.
    #[serde(skip)]
    pub deadline: Option<std::time::Instant>,
    /// Lane this entry is bound to (must match the queue's lane).
    pub lane: LaneId,
    /// Opaque per-entry tag the orchestrator uses to route the entry
    /// on dequeue. Constitutional side stores it as a `u64` to keep
    /// the queue type-agnostic; the orchestrator can wrap a typed
    /// `EpochId` / `SessionId` / etc.
    pub tag: u64,
}

// ── Errors ────────────────────────────────────────────────────────────────

/// Errors emitted by the lane queue. Classified as `Rejected` (queue
/// full → admission gate refused) and `NotFound` (work id absent
/// from the queue). Per the `no anyhow` rule from
/// `references/rust-quality.md`, every fallible API returns this
/// typed error.
#[derive(Debug, Error, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LaneQueueError {
    /// The queue is at capacity; the entry was not enqueued.
    #[error("lane queue at capacity: {reason:?}")]
    Rejected {
        /// The specific capacity reason for the rejection.
        reason: BackpressureReason,
        /// The lane whose queue was full.
        lane: LaneId,
    },
    /// The work id was not present in the queue (caller passed a
    /// stale or unknown id).
    #[error("work id {work_id:?} not found in lane {lane:?}")]
    NotFound {
        /// The work id that was searched for.
        work_id: WorkId,
        /// The lane that was searched.
        lane: LaneId,
    },
}

// ── Lane queue ────────────────────────────────────────────────────────────

/// Bounded per-lane queue with priority ordering.
///
/// Maintains a fixed maximum depth (`max_depth`).  Pushing beyond
/// capacity returns [`Err(LaneQueueError::Rejected)`] with the
/// reason mapped from the lane id, allowing the orchestrator to
/// apply backpressure to the upstream pipeline.
///
/// Priority ordering: higher [`WorkPriority`] values are popped
/// first; ties are resolved in FIFO order. The `deadline` field is
/// advisory (for external timeout checks) and is not enforced by
/// the queue.
#[derive(Debug)]
pub struct LaneQueue {
    lane: LaneId,
    max_depth: usize,
    entries: VecDeque<QueueEntry>,
}

impl LaneQueue {
    /// Create a new bounded lane queue.
    ///
    /// `max_depth` sets the maximum number of entries allowed. Zero
    /// produces an always-full queue (every push is rejected); this
    /// is a useful way to disable a lane without removing it from
    /// the lane-queue set.
    pub fn new(lane: LaneId, max_depth: usize) -> Self {
        Self {
            lane,
            max_depth,
            entries: VecDeque::new(),
        }
    }

    /// Try to push an entry onto the queue.
    ///
    /// Returns `Ok(())` if space is available, or
    /// `Err(LaneQueueError::Rejected)` when the queue is full. The
    /// reason is derived from the queue's lane via
    /// [`BackpressureReason::for_lane`].
    pub fn try_push(&mut self, entry: QueueEntry) -> Result<(), LaneQueueError> {
        if self.entries.len() >= self.max_depth {
            return Err(LaneQueueError::Rejected {
                reason: BackpressureReason::for_lane(self.lane),
                lane: self.lane,
            });
        }
        // Sanity: the entry's lane must match the queue's lane.
        // Mismatched pushes are a programmer error; the constitutional
        // rule prefers an explicit error over a silent misroute.
        if entry.lane != self.lane {
            return Err(LaneQueueError::Rejected {
                reason: BackpressureReason::GlobalQueue,
                lane: self.lane,
            });
        }
        self.entries.push_back(entry);
        Ok(())
    }

    /// Pop the highest-priority entry.
    ///
    /// Returns `None` if the queue is empty. Among entries of equal
    /// priority the oldest (earliest-pushed) is returned.
    pub fn pop(&mut self) -> Option<QueueEntry> {
        let idx = self.highest_priority_index()?;
        self.entries.remove(idx)
    }

    /// Peek at the highest-priority entry without removing it.
    pub fn peek(&self) -> Option<&QueueEntry> {
        let idx = self.highest_priority_index()?;
        self.entries.get(idx)
    }

    /// Number of entries currently in the queue.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the queue contains no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Maximum number of entries this queue can hold.
    pub const fn capacity(&self) -> usize {
        self.max_depth
    }

    /// Remaining capacity before the queue rejects pushes.
    pub fn remaining(&self) -> usize {
        // Saturating because `len() <= max_depth` is a class invariant
        // maintained by `try_push`.
        self.max_depth.saturating_sub(self.entries.len())
    }

    /// The execution lane this queue serves.
    pub const fn lane(&self) -> LaneId {
        self.lane
    }

    /// Remove an entry by [`WorkId`] (cancellation path).
    ///
    /// Performs a linear scan — acceptable because per-lane queues
    /// are deliberately small (typically single-digit depths).
    /// Returns the entry if found, or
    /// `Err(LaneQueueError::NotFound)` if the id is absent.
    pub fn remove(&mut self, work_id: WorkId) -> Result<QueueEntry, LaneQueueError> {
        let pos = self
            .entries
            .iter()
            .position(|e| e.work_id == work_id)
            .ok_or(LaneQueueError::NotFound {
                work_id,
                lane: self.lane,
            })?;
        // `pos` was returned by `iter().position()` and the
        // single-threaded `&mut self` borrow makes the index
        // valid until the next mutation. `VecDeque::remove`
        // returns `Option<T>`; the `None` case is unreachable
        // here (the index is in `[0, entries.len())`) so the
        // result is mapped to a typed error to avoid an `expect`.
        self.entries.remove(pos).ok_or(LaneQueueError::NotFound {
            work_id,
            lane: self.lane,
        })
    }

    /// Remove every entry from the queue.
    ///
    /// Returns the count of removed items.
    pub fn drain(&mut self) -> usize {
        let count = self.entries.len();
        self.entries.clear();
        count
    }

    /// Find the index of the highest-priority entry.
    ///
    /// Lower index wins ties (FIFO for equal priority).
    fn highest_priority_index(&self) -> Option<usize> {
        // `max_by` is O(n); at the per-lane queue depths the engine
        // expects (≤ 64 entries), the linear scan is faster than any
        // heap-based alternative (heap dequeue is O(log n) but
        // constant factor is much higher than a tight loop over ≤ 64
        // entries).
        self.entries
            .iter()
            .enumerate()
            .max_by(|(i, a), (j, b)| {
                // Compare priority first, then by index (FIFO wins
                // for equal priority: smaller index = earlier = wins
                // in `max_by`).
                a.priority
                    .cmp(&b.priority)
                    .then_with(|| j.cmp(i))
            })
            .map(|(idx, _)| idx)
    }
}

// ── Lane queue set ────────────────────────────────────────────────────────

/// Backpressured lane queue manager — owns queues for the registered
/// execution lanes.
///
/// Provides lookup by [`LaneId`], aggregate pending counts, and a
/// snapshot for observability. Lanes are added explicitly via
/// [`LaneQueueSet::with_lane`] (no implicit metal/ane/cpu defaults
/// — the orchestrator declares its topology).
#[derive(Debug, Default)]
pub struct LaneQueueSet {
    /// Per-lane queues. `BTreeMap` because iteration order is part
    /// of the observability snapshot.
    queues: BTreeMap<LaneId, LaneQueue>,
}

impl LaneQueueSet {
    /// Create an empty lane queue set. Use [`Self::with_lane`] to
    /// register lanes.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a lane with the given maximum depth. Returns the
    /// previous queue for the lane if one already existed (and
    /// replaces it).
    pub fn with_lane(&mut self, lane: LaneId, max_depth: usize) -> Option<LaneQueue> {
        self.queues.insert(lane, LaneQueue::new(lane, max_depth))
    }

    /// Mutable access to the queue for a given lane. Returns `None`
    /// if the lane is not registered.
    pub fn queue_for(&mut self, lane: LaneId) -> Option<&mut LaneQueue> {
        self.queues.get_mut(&lane)
    }

    /// Immutable access to the queue for a given lane.
    pub fn queue_for_lane(&self, lane: LaneId) -> Option<&LaneQueue> {
        self.queues.get(&lane)
    }

    /// Total number of pending entries across all registered lane
    /// queues.
    pub fn total_pending(&self) -> usize {
        self.queues.values().map(LaneQueue::len).sum()
    }

    /// Snapshot of per-lane queue depths. The map is ordered by
    /// [`LaneId`] (the `Ord` impl) so the snapshot is deterministic.
    /// Only lanes with non-zero depth appear in the map.
    pub fn snapshot(&self) -> BTreeMap<LaneId, usize> {
        self.queues
            .iter()
            .filter(|(_, q)| !q.is_empty())
            .map(|(lane, queue)| (*lane, queue.len()))
            .collect()
    }

    /// Number of registered lanes.
    pub fn lane_count(&self) -> usize {
        self.queues.len()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn entry(priority: WorkPriority, work_id: WorkId, lane: LaneId) -> QueueEntry {
        QueueEntry {
            work_id,
            priority,
            deadline: None,
            lane,
            tag: 0,
        }
    }

    // ── LaneQueue tests ─────────────────────────────────────────────

    #[test]
    fn basic_push_pop() {
        let mut q = LaneQueue::new(LaneId(0), 4);
        assert!(q.is_empty());
        assert_eq!(q.len(), 0);
        assert_eq!(q.capacity(), 4);
        assert_eq!(q.remaining(), 4);

        let wid = WorkId(1);
        assert!(q.try_push(entry(WorkPriority::Normal, wid, LaneId(0))).is_ok());
        assert!(!q.is_empty());
        assert_eq!(q.len(), 1);
        assert_eq!(q.remaining(), 3);

        let popped = q.pop().expect("should have an entry");
        assert_eq!(popped.work_id, wid);
        assert!(q.is_empty());
    }

    #[test]
    fn backpressure_when_full() {
        let mut q = LaneQueue::new(LaneId(1), 2);
        assert!(q.try_push(entry(WorkPriority::Normal, WorkId(1), LaneId(1))).is_ok());
        assert!(q.try_push(entry(WorkPriority::Normal, WorkId(2), LaneId(1))).is_ok());
        let err = q
            .try_push(entry(WorkPriority::Normal, WorkId(3), LaneId(1)))
            .expect_err("queue should be full");
        match err {
            LaneQueueError::Rejected { reason, lane } => {
                assert_eq!(reason, BackpressureReason::AneCapacity);
                assert_eq!(lane, LaneId(1));
            }
            other => panic!("expected Rejected, got {other:?}"),
        }
    }

    #[test]
    fn priority_ordering() {
        let mut q = LaneQueue::new(LaneId(0), 10);
        let low = WorkId(1);
        let high = WorkId(2);
        let normal = WorkId(3);

        // Insert in non-priority order.
        q.try_push(entry(WorkPriority::Low, low, LaneId(0))).unwrap();
        q.try_push(entry(WorkPriority::High, high, LaneId(0))).unwrap();
        q.try_push(entry(WorkPriority::Normal, normal, LaneId(0))).unwrap();

        // Pop order: High → Normal → Low
        assert_eq!(q.pop().unwrap().work_id, high);
        assert_eq!(q.pop().unwrap().work_id, normal);
        assert_eq!(q.pop().unwrap().work_id, low);
        assert!(q.pop().is_none());
    }

    #[test]
    fn fifo_within_same_priority() {
        let mut q = LaneQueue::new(LaneId(0), 10);
        let a = WorkId(1);
        let b = WorkId(2);
        let c = WorkId(3);

        q.try_push(entry(WorkPriority::Normal, a, LaneId(0))).unwrap();
        q.try_push(entry(WorkPriority::Normal, b, LaneId(0))).unwrap();
        q.try_push(entry(WorkPriority::Normal, c, LaneId(0))).unwrap();

        // All same priority → FIFO.
        assert_eq!(q.pop().unwrap().work_id, a);
        assert_eq!(q.pop().unwrap().work_id, b);
        assert_eq!(q.pop().unwrap().work_id, c);
    }

    #[test]
    fn priority_within_fifo_interleaved() {
        let mut q = LaneQueue::new(LaneId(0), 10);
        // Push order: Low, High, Low, High
        let l1 = WorkId(1);
        let h1 = WorkId(2);
        let l2 = WorkId(3);
        let h2 = WorkId(4);

        q.try_push(entry(WorkPriority::Low, l1, LaneId(0))).unwrap();
        q.try_push(entry(WorkPriority::High, h1, LaneId(0))).unwrap();
        q.try_push(entry(WorkPriority::Low, l2, LaneId(0))).unwrap();
        q.try_push(entry(WorkPriority::High, h2, LaneId(0))).unwrap();

        // High entries first (FIFO among themselves).
        assert_eq!(q.pop().unwrap().work_id, h1);
        assert_eq!(q.pop().unwrap().work_id, h2);

        // Then Low entries (FIFO among themselves).
        assert_eq!(q.pop().unwrap().work_id, l1);
        assert_eq!(q.pop().unwrap().work_id, l2);
    }

    #[test]
    fn peek_does_not_remove() {
        let mut q = LaneQueue::new(LaneId(0), 4);
        let wid = WorkId(1);
        q.try_push(entry(WorkPriority::High, wid, LaneId(0))).unwrap();

        let peeked = q.peek().expect("peek should return an entry");
        assert_eq!(peeked.work_id, wid);
        assert_eq!(q.len(), 1, "peek should not remove");

        let popped = q.pop().expect("pop after peek should work");
        assert_eq!(popped.work_id, wid);
    }

    #[test]
    fn remove_by_work_id() {
        let mut q = LaneQueue::new(LaneId(0), 10);
        let a = WorkId(1);
        let b = WorkId(2);
        let c = WorkId(3);

        q.try_push(entry(WorkPriority::Normal, a, LaneId(0))).unwrap();
        q.try_push(entry(WorkPriority::Normal, b, LaneId(0))).unwrap();
        q.try_push(entry(WorkPriority::Normal, c, LaneId(0))).unwrap();

        // Remove the middle entry.
        let removed = q.remove(b).expect("b should be found");
        assert_eq!(removed.work_id, b);
        assert_eq!(q.len(), 2);

        // Remaining entries are a and c, still in FIFO order.
        assert_eq!(q.pop().unwrap().work_id, a);
        assert_eq!(q.pop().unwrap().work_id, c);
        assert!(q.pop().is_none());
    }

    #[test]
    fn remove_nonexistent_is_error() {
        let mut q = LaneQueue::new(LaneId(0), 4);
        q.try_push(entry(WorkPriority::Normal, WorkId(1), LaneId(0)))
            .unwrap();
        let missing = WorkId(99);
        let err = q.remove(missing).expect_err("missing id should error");
        match err {
            LaneQueueError::NotFound { work_id, lane } => {
                assert_eq!(work_id, missing);
                assert_eq!(lane, LaneId(0));
            }
            other => panic!("expected NotFound, got {other:?}"),
        }
        assert_eq!(q.len(), 1, "queue should be unchanged after not-found");
    }

    #[test]
    fn drain_clears_all() {
        let mut q = LaneQueue::new(LaneId(0), 10);
        q.try_push(entry(WorkPriority::Normal, WorkId(1), LaneId(0)))
            .unwrap();
        q.try_push(entry(WorkPriority::Normal, WorkId(2), LaneId(0)))
            .unwrap();
        q.try_push(entry(WorkPriority::Normal, WorkId(3), LaneId(0)))
            .unwrap();

        assert_eq!(q.drain(), 3);
        assert!(q.is_empty());
        assert_eq!(q.drain(), 0);
    }

    #[test]
    fn empty_queue_pop_and_peek() {
        let q = LaneQueue::new(LaneId(0), 4);
        // Pop on an immutable queue would need `&mut`; this is a
        // presence test.
        assert!(q.peek().is_none());
        assert!(q.is_empty());
    }

    #[test]
    fn zero_capacity_is_always_full() {
        let mut q = LaneQueue::new(LaneId(0), 0);
        assert_eq!(q.remaining(), 0);
        let err = q
            .try_push(entry(WorkPriority::Normal, WorkId(1), LaneId(0)))
            .expect_err("zero-capacity queue should reject all pushes");
        match err {
            LaneQueueError::Rejected { reason, lane } => {
                assert_eq!(reason, BackpressureReason::MetalCapacity);
                assert_eq!(lane, LaneId(0));
            }
            other => panic!("expected Rejected, got {other:?}"),
        }
    }

    #[test]
    fn lane_roundtrip() {
        let q = LaneQueue::new(LaneId(1), 8);
        assert_eq!(q.lane(), LaneId(1));
        assert_eq!(q.capacity(), 8);
    }

    #[test]
    fn push_to_wrong_lane_is_rejected() {
        // The orchestrator must not push an entry tagged for one
        // lane onto another lane's queue. The queue rejects with
        // a typed error.
        let mut q = LaneQueue::new(LaneId(0), 4);
        let bad = entry(WorkPriority::Normal, WorkId(1), LaneId(1));
        let err = q.try_push(bad).expect_err("mismatched lane should reject");
        assert!(matches!(err, LaneQueueError::Rejected { .. }));
        assert_eq!(q.len(), 0);
    }

    #[test]
    fn deadline_round_trips() {
        let mut q = LaneQueue::new(LaneId(0), 1);
        let deadline = Instant::now();
        q.try_push(QueueEntry {
            work_id: WorkId(1),
            priority: WorkPriority::High,
            deadline: Some(deadline),
            lane: LaneId(0),
            tag: 42,
        })
        .unwrap();
        let popped = q.pop().unwrap();
        assert_eq!(popped.deadline, Some(deadline));
        assert_eq!(popped.tag, 42);
    }

    // ── LaneQueueSet tests ───────────────────────────────────────────

    #[test]
    fn queue_set_new_and_pending() {
        let set = LaneQueueSet::new();
        assert_eq!(set.total_pending(), 0);
        assert!(set.snapshot().is_empty());
        assert_eq!(set.lane_count(), 0);
    }

    #[test]
    fn queue_set_queue_for_each_lane() {
        let mut set = LaneQueueSet::new();
        set.with_lane(LaneId(0), 2);
        set.with_lane(LaneId(1), 2);
        set.with_lane(LaneId(2), 2);

        let mq = set.queue_for(LaneId(0)).unwrap();
        assert_eq!(mq.lane(), LaneId(0));
        mq.try_push(entry(WorkPriority::Normal, WorkId(1), LaneId(0)))
            .unwrap();

        let aq = set.queue_for(LaneId(1)).unwrap();
        assert_eq!(aq.lane(), LaneId(1));
        aq.try_push(entry(WorkPriority::Normal, WorkId(2), LaneId(1)))
            .unwrap();

        let cq = set.queue_for(LaneId(2)).unwrap();
        assert_eq!(cq.lane(), LaneId(2));
        cq.try_push(entry(WorkPriority::Normal, WorkId(3), LaneId(2)))
            .unwrap();

        assert_eq!(set.total_pending(), 3);
        let snap = set.snapshot();
        assert_eq!(snap.len(), 3);
        assert_eq!(*snap.get(&LaneId(0)).unwrap(), 1);
        assert_eq!(*snap.get(&LaneId(1)).unwrap(), 1);
        assert_eq!(*snap.get(&LaneId(2)).unwrap(), 1);
    }

    #[test]
    fn queue_set_immutable_access() {
        let mut set = LaneQueueSet::new();
        set.with_lane(LaneId(0), 2);
        set.queue_for(LaneId(0))
            .unwrap()
            .try_push(entry(WorkPriority::Normal, WorkId(1), LaneId(0)))
            .unwrap();

        let q = set.queue_for_lane(LaneId(0)).unwrap();
        assert_eq!(q.len(), 1);
        assert_eq!(q.lane(), LaneId(0));
    }

    #[test]
    fn queue_set_backpressure_each_lane() {
        // Each lane has depth 1; second push per lane fails.
        let mut set = LaneQueueSet::new();
        set.with_lane(LaneId(0), 1);
        set.with_lane(LaneId(1), 1);
        set.with_lane(LaneId(2), 1);

        assert!(set
            .queue_for(LaneId(0))
            .unwrap()
            .try_push(entry(WorkPriority::Normal, WorkId(1), LaneId(0)))
            .is_ok());
        let err = set
            .queue_for(LaneId(0))
            .unwrap()
            .try_push(entry(WorkPriority::Normal, WorkId(2), LaneId(0)))
            .expect_err("metal should be full");
        assert!(matches!(
            err,
            LaneQueueError::Rejected {
                reason: BackpressureReason::MetalCapacity,
                ..
            }
        ));

        assert!(set
            .queue_for(LaneId(1))
            .unwrap()
            .try_push(entry(WorkPriority::Normal, WorkId(3), LaneId(1)))
            .is_ok());
        let err = set
            .queue_for(LaneId(1))
            .unwrap()
            .try_push(entry(WorkPriority::Normal, WorkId(4), LaneId(1)))
            .expect_err("ane should be full");
        assert!(matches!(
            err,
            LaneQueueError::Rejected {
                reason: BackpressureReason::AneCapacity,
                ..
            }
        ));

        assert!(set
            .queue_for(LaneId(2))
            .unwrap()
            .try_push(entry(WorkPriority::Normal, WorkId(5), LaneId(2)))
            .is_ok());
        let err = set
            .queue_for(LaneId(2))
            .unwrap()
            .try_push(entry(WorkPriority::Normal, WorkId(6), LaneId(2)))
            .expect_err("cpu should be full");
        assert!(matches!(
            err,
            LaneQueueError::Rejected {
                reason: BackpressureReason::CpuCapacity,
                ..
            }
        ));
    }

    #[test]
    fn queue_set_snapshot_only_nonzero() {
        let mut set = LaneQueueSet::new();
        set.with_lane(LaneId(0), 5);
        set.with_lane(LaneId(1), 5);
        set.with_lane(LaneId(2), 5);
        set.queue_for(LaneId(0))
            .unwrap()
            .try_push(entry(WorkPriority::Normal, WorkId(1), LaneId(0)))
            .unwrap();

        let snap = set.snapshot();
        assert_eq!(snap.len(), 1);
        assert_eq!(*snap.get(&LaneId(0)).unwrap(), 1);
        assert!(snap.get(&LaneId(1)).is_none());
        assert!(snap.get(&LaneId(2)).is_none());
    }

    #[test]
    fn queue_set_unknown_lane_returns_none() {
        let mut set = LaneQueueSet::new();
        set.with_lane(LaneId(0), 4);
        assert!(set.queue_for(LaneId(99)).is_none());
        assert!(set.queue_for_lane(LaneId(99)).is_none());
    }

    #[test]
    fn backpressure_reason_for_lane_mapping() {
        // Verify the lane-id → reason mapping is consistent.
        assert_eq!(
            BackpressureReason::for_lane(LaneId(0)),
            BackpressureReason::MetalCapacity
        );
        assert_eq!(
            BackpressureReason::for_lane(LaneId(1)),
            BackpressureReason::AneCapacity
        );
        assert_eq!(
            BackpressureReason::for_lane(LaneId(2)),
            BackpressureReason::CpuCapacity
        );
        // Higher ordinals fall through to ActivationSlots.
        assert_eq!(
            BackpressureReason::for_lane(LaneId(99)),
            BackpressureReason::ActivationSlots
        );
    }

    #[test]
    fn work_priority_user_facing() {
        assert!(WorkPriority::Interactive.is_user_facing());
        assert!(WorkPriority::High.is_user_facing());
        assert!(!WorkPriority::Normal.is_user_facing());
        assert!(!WorkPriority::Low.is_user_facing());
        assert!(!WorkPriority::Compilation.is_user_facing());
    }

    #[test]
    fn newtypes_serde_transparent() {
        // Round-trip through JSON to confirm `#[serde(transparent)]`
        // is honored. A `LaneId(7)` should serialize to `7`, not
        // `{"0": 7}`.
        let lane = LaneId(7);
        let json = serde_json::to_string(&lane).unwrap();
        assert_eq!(json, "7");
        let back: LaneId = serde_json::from_str(&json).unwrap();
        assert_eq!(back, LaneId(7));

        let work = WorkId(123);
        let json = serde_json::to_string(&work).unwrap();
        assert_eq!(json, "123");
        let back: WorkId = serde_json::from_str(&json).unwrap();
        assert_eq!(back, WorkId(123));
    }
}
