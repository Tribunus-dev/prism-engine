//! PRISM-PRODUCTION-HETEROGENEOUS-EXECUTOR-0001 — Bounded per-lane execution queues.
//!
//! Provides a fixed-capacity, priority-ordered queue per execution lane
//! with backpressure signalling and a [`LaneQueueSet`] that owns the three
//! primary lane queues (Metal GPU, Core ML ANE, Accelerate CPU).

use std::collections::HashMap;
use std::collections::VecDeque;
use std::time::Instant;

use crate::backend::placement::ExecutionLane;
use crate::compilation::phase_ir::PhaseId;
use crate::scheduling::lane_work::{LaneWorkRequest, WorkId};

// ── Priority level ─────────────────────────────────────────────────────────

/// Priority level for queued work.
///
/// Higher-priority entries are dequeued first.  Within the same priority
/// level the queue preserves FIFO order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum WorkPriority {
    /// Background compilation tasks (model load, pipeline compilation).
    Compilation,
    /// Warmup / pre-heat runs before latency-critical requests.
    Warmup,
    /// Low-priority background inference.
    Low,
    /// Normal request (default).
    Normal,
    /// Elevated priority for interactive use.
    High,
    /// Highest priority — user-facing interactive sessions.
    Interactive,
}

impl Default for WorkPriority {
    fn default() -> Self {
        Self::Normal
    }
}

// ── Queue entry ────────────────────────────────────────────────────────────

/// A single entry in a lane queue, carrying priority, deadline, and the
/// full [`LaneWorkRequest`] for the backend.
#[derive(Debug, Clone)]
pub struct QueueEntry {
    /// Unique work identifier (matches [`LaneWorkRequest::work_id`]).
    pub work_id: WorkId,
    /// Compilation phase this work belongs to.
    pub phase_id: PhaseId,
    /// Dispatch priority (higher = dequeued sooner).
    pub priority: WorkPriority,
    /// Optional deadline — used for timeout checks, not enforced by the queue.
    pub deadline: Option<Instant>,
    /// Full work request descriptor for the backend executor.
    pub request: LaneWorkRequest,
}

// ── Backpressure reason ────────────────────────────────────────────────────

/// Reason why a [`LaneQueue::try_push`] was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

// ── LaneQueue ──────────────────────────────────────────────────────────────

/// Bounded per-lane queue with priority ordering.
///
/// Maintains a fixed maximum depth (`max_depth`).  Pushing beyond capacity
/// returns [`Err(BackpressureReason)`] with the reason mapped from the lane
/// type, allowing the caller to apply backpressure to the upstream pipeline.
///
/// Priority ordering: higher [`WorkPriority`] values are popped first; ties
/// are resolved in FIFO order.  [`deadline`](QueueEntry::deadline) is
/// advisory (for external timeout checks) and ǃis not enforced by the queue.
#[derive(Debug)]
pub struct LaneQueue {
    lane: ExecutionLane,
    max_depth: usize,
    entries: VecDeque<QueueEntry>,
}

impl LaneQueue {
    /// Create a new bounded lane queue.
    ///
    /// `max_depth` sets the maximum number of entries allowed.
    /// Zero or negative values produce an always-full queue.
    pub fn new(lane: ExecutionLane, max_depth: usize) -> Self {
        Self {
            lane,
            max_depth,
            entries: VecDeque::with_capacity(max_depth),
        }
    }

    /// Try to push an entry onto the queue.
    ///
    /// Returns `Ok(())` if space is available, or
    /// `Err(BackpressureReason)` when the queue is full.  The
    /// reason is derived from the lane type.
    pub fn try_push(&mut self, entry: QueueEntry) -> Result<(), BackpressureReason> {
        if self.entries.len() >= self.max_depth {
            return Err(backpressure_for(self.lane));
        }
        self.entries.push_back(entry);
        Ok(())
    }

    /// Pop the highest-priority entry.
    ///
    /// Returns `None` if the queue is empty.  Among entries of equal
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
    pub fn capacity(&self) -> usize {
        self.max_depth
    }

    /// Remaining capacity before the queue rejects pushes.
    pub fn remaining(&self) -> usize {
        self.max_depth.saturating_sub(self.entries.len())
    }

    /// The execution lane this queue serves.
    pub fn lane(&self) -> ExecutionLane {
        self.lane
    }

    /// Remove an entry by [`WorkId`] (cancellation path).
    ///
    /// Performs a linear scan — acceptable because per-lane queues
    /// are deliberately small (typically single-digit depths).
    /// Returns the entry if found, `None` otherwise.
    pub fn remove(&mut self, work_id: WorkId) -> Option<QueueEntry> {
        let pos = self.entries.iter().position(|e| e.work_id == work_id)?;
        self.entries.remove(pos)
    }

    /// Remove every entry from the queue.
    ///
    /// Returns the count of removed items.
    pub fn drain(&mut self) -> usize {
        let count = self.entries.len();
        self.entries.clear();
        count
    }

    // ── helpers ─────────────────────────────────────────────────────────

    /// Find the index of the highest-priority entry.
    ///
    /// Lower index wins ties (FIFO for equal priority).
    fn highest_priority_index(&self) -> Option<usize> {
        self.entries
            .iter()
            .enumerate()
            .max_by(|(i, a), (j, b)| {
                a.priority.cmp(&b.priority).then_with(|| j.cmp(i)) // smaller index = earlier = wins in max_by
            })
            .map(|(idx, _)| idx)
    }
}

// ── LaneQueueSet ───────────────────────────────────────────────────────────

/// Backpressured lane queue manager — owns queues for the three primary
/// execution lanes (Metal GPU, Core ML ANE, Accelerate CPU).
///
/// Provides lookup by [`ExecutionLane`], aggregate pending counts,
/// and a snapshot for observability.
#[derive(Debug)]
pub struct LaneQueueSet {
    metal: LaneQueue,
    ane: LaneQueue,
    accelerate: LaneQueue,
}

impl LaneQueueSet {
    /// Build a [`LaneQueueSet`] with per-lane depth limits.
    pub fn new(metal_depth: usize, ane_depth: usize, accel_depth: usize) -> Self {
        Self {
            metal: LaneQueue::new(ExecutionLane::MlxGpu, metal_depth),
            ane: LaneQueue::new(ExecutionLane::CoreMlAne, ane_depth),
            accelerate: LaneQueue::new(ExecutionLane::AccelerateCpu, accel_depth),
        }
    }

    /// Mutable access to the queue for a given lane.
    ///
    /// Lanes beyond the three primary ones fall through to the CPU queue
    /// (conservative — those lanes are handled externally).
    pub fn queue_for(&mut self, lane: ExecutionLane) -> &mut LaneQueue {
        match lane {
            ExecutionLane::MlxGpu => &mut self.metal,
            ExecutionLane::CoreMlAne => &mut self.ane,
            ExecutionLane::AccelerateCpu
            | ExecutionLane::CandleCpu
            | ExecutionLane::Tensix
            | ExecutionLane::IntelLevelZero => &mut self.accelerate,
        }
    }

    /// Immutable access to the queue for a given lane.
    pub fn queue_for_lane(&self, lane: ExecutionLane) -> &LaneQueue {
        match lane {
            ExecutionLane::MlxGpu => &self.metal,
            ExecutionLane::CoreMlAne => &self.ane,
            ExecutionLane::AccelerateCpu
            | ExecutionLane::CandleCpu
            | ExecutionLane::Tensix
            | ExecutionLane::IntelLevelZero => &self.accelerate,
        }
    }

    /// Total number of pending entries across all three lane queues.
    pub fn total_pending(&self) -> usize {
        self.metal.len() + self.ane.len() + self.accelerate.len()
    }

    /// Snapshot of per-lane queue depths as a [`HashMap`].
    ///
    /// Only lanes with non-zero depth appear in the map.
    pub fn snapshot(&self) -> HashMap<ExecutionLane, usize> {
        let mut map = HashMap::new();
        let metal_len = self.metal.len();
        let ane_len = self.ane.len();
        let accel_len = self.accelerate.len();

        if metal_len > 0 {
            map.insert(ExecutionLane::MlxGpu, metal_len);
        }
        if ane_len > 0 {
            map.insert(ExecutionLane::CoreMlAne, ane_len);
        }
        if accel_len > 0 {
            map.insert(ExecutionLane::AccelerateCpu, accel_len);
        }
        map
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────

/// Map an [`ExecutionLane`] to the most specific [`BackpressureReason`].
fn backpressure_for(lane: ExecutionLane) -> BackpressureReason {
    match lane {
        ExecutionLane::MlxGpu => BackpressureReason::MetalCapacity,
        ExecutionLane::CoreMlAne => BackpressureReason::AneCapacity,
        ExecutionLane::AccelerateCpu | ExecutionLane::CandleCpu => BackpressureReason::CpuCapacity,
        ExecutionLane::Tensix => BackpressureReason::ActivationSlots,
        ExecutionLane::IntelLevelZero => BackpressureReason::MetalCapacity,
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compilation::activation_abi::{ActivationAbi, MetalOnlyParams, SlotLeaseId};
    use crate::compilation::phase_ir::TensorDtype;
    use crate::scheduling::lane_work::{next_work_id, CompletionClock, StreamId};

    // Helper to build a minimal queue entry at a given priority.
    fn entry(priority: WorkPriority, work_id: WorkId) -> QueueEntry {
        QueueEntry {
            work_id,
            phase_id: PhaseId(0),
            priority,
            deadline: None,
            request: crate::scheduling::lane_work::LaneWorkRequest {
                work_id,
                session_id: StreamId(0),
                epoch_id: 0,
                phase_id: PhaseId(0),
                variant_id: 0,
                lane: ExecutionLane::MlxGpu,
                input_slots: Vec::new(),
                output_slot: SlotLeaseId(0),
                input_abi: ActivationAbi::MetalOnly(MetalOnlyParams {
                    name: String::new(),
                    dtype: TensorDtype::Float16,
                    byte_count: 0,
                }),
                output_abi: ActivationAbi::MetalOnly(MetalOnlyParams {
                    name: String::new(),
                    dtype: TensorDtype::Float16,
                    byte_count: 0,
                }),
                artifact_key: None,
                metal_pipeline: None,
                completion_clock: CompletionClock::new(0),
            },
        }
    }

    #[test]
    fn basic_push_pop() {
        let mut q = LaneQueue::new(ExecutionLane::MlxGpu, 4);
        assert!(q.is_empty());
        assert_eq!(q.len(), 0);
        assert_eq!(q.capacity(), 4);
        assert_eq!(q.remaining(), 4);

        let wid = next_work_id();
        assert!(q.try_push(entry(WorkPriority::Normal, wid)).is_ok());
        assert!(!q.is_empty());
        assert_eq!(q.len(), 1);
        assert_eq!(q.remaining(), 3);

        let popped = q.pop().expect("should have an entry");
        assert_eq!(popped.work_id, wid);
        assert!(q.is_empty());
    }

    #[test]
    fn backpressure_when_full() {
        let mut q = LaneQueue::new(ExecutionLane::CoreMlAne, 2);
        assert!(q
            .try_push(entry(WorkPriority::Normal, next_work_id()))
            .is_ok());
        assert!(q
            .try_push(entry(WorkPriority::Normal, next_work_id()))
            .is_ok());
        // Third push should fail with AneCapacity.
        let err = q
            .try_push(entry(WorkPriority::Normal, next_work_id()))
            .expect_err("queue should be full");
        assert_eq!(err, BackpressureReason::AneCapacity);
    }

    #[test]
    fn priority_ordering() {
        let mut q = LaneQueue::new(ExecutionLane::MlxGpu, 10);
        let low = next_work_id();
        let high = next_work_id();
        let normal = next_work_id();

        // Insert in non-priority order.
        q.try_push(entry(WorkPriority::Low, low)).unwrap();
        q.try_push(entry(WorkPriority::High, high)).unwrap();
        q.try_push(entry(WorkPriority::Normal, normal)).unwrap();

        // Pop order: High → Normal → Low
        assert_eq!(q.pop().unwrap().work_id, high);
        assert_eq!(q.pop().unwrap().work_id, normal);
        assert_eq!(q.pop().unwrap().work_id, low);
        assert!(q.pop().is_none());
    }

    #[test]
    fn fifo_within_same_priority() {
        let mut q = LaneQueue::new(ExecutionLane::MlxGpu, 10);
        let a = next_work_id();
        let b = next_work_id();
        let c = next_work_id();

        q.try_push(entry(WorkPriority::Normal, a)).unwrap();
        q.try_push(entry(WorkPriority::Normal, b)).unwrap();
        q.try_push(entry(WorkPriority::Normal, c)).unwrap();

        // All same priority → FIFO.
        assert_eq!(q.pop().unwrap().work_id, a);
        assert_eq!(q.pop().unwrap().work_id, b);
        assert_eq!(q.pop().unwrap().work_id, c);
    }

    #[test]
    fn priority_within_fifo_interleaved() {
        let mut q = LaneQueue::new(ExecutionLane::MlxGpu, 10);
        // Push order: Low, High, Low, High
        let l1 = next_work_id();
        let h1 = next_work_id();
        let l2 = next_work_id();
        let h2 = next_work_id();

        q.try_push(entry(WorkPriority::Low, l1)).unwrap();
        q.try_push(entry(WorkPriority::High, h1)).unwrap();
        q.try_push(entry(WorkPriority::Low, l2)).unwrap();
        q.try_push(entry(WorkPriority::High, h2)).unwrap();

        // High entries first (FIFO among themselves).
        assert_eq!(q.pop().unwrap().work_id, h1);
        assert_eq!(q.pop().unwrap().work_id, h2);

        // Then Low entries (FIFO among themselves).
        assert_eq!(q.pop().unwrap().work_id, l1);
        assert_eq!(q.pop().unwrap().work_id, l2);
    }

    #[test]
    fn peek_does_not_remove() {
        let mut q = LaneQueue::new(ExecutionLane::MlxGpu, 4);
        let wid = next_work_id();
        q.try_push(entry(WorkPriority::High, wid)).unwrap();

        let peeked = q.peek().expect("peek should return an entry");
        assert_eq!(peeked.work_id, wid);
        assert_eq!(q.len(), 1, "peek should not remove");

        let popped = q.pop().expect("pop after peek should work");
        assert_eq!(popped.work_id, wid);
    }

    #[test]
    fn remove_by_work_id() {
        let mut q = LaneQueue::new(ExecutionLane::MlxGpu, 10);
        let a = next_work_id();
        let b = next_work_id();
        let c = next_work_id();

        q.try_push(entry(WorkPriority::Normal, a)).unwrap();
        q.try_push(entry(WorkPriority::Normal, b)).unwrap();
        q.try_push(entry(WorkPriority::Normal, c)).unwrap();

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
    fn remove_nonexistent() {
        let mut q = LaneQueue::new(ExecutionLane::MlxGpu, 4);
        let wid = next_work_id();
        q.try_push(entry(WorkPriority::Normal, wid)).unwrap();
        let missing = next_work_id();
        assert!(q.remove(missing).is_none());
        assert_eq!(q.len(), 1);
    }

    #[test]
    fn drain_clears_all() {
        let mut q = LaneQueue::new(ExecutionLane::MlxGpu, 10);
        q.try_push(entry(WorkPriority::Normal, next_work_id()))
            .unwrap();
        q.try_push(entry(WorkPriority::Normal, next_work_id()))
            .unwrap();
        q.try_push(entry(WorkPriority::Normal, next_work_id()))
            .unwrap();

        assert_eq!(q.drain(), 3);
        assert!(q.is_empty());
        assert_eq!(q.drain(), 0);
    }

    #[test]
    fn empty_queue_pop_and_peek() {
        let mut q = LaneQueue::new(ExecutionLane::MlxGpu, 4);
        assert!(q.pop().is_none());
        assert!(q.peek().is_none());
    }

    #[test]
    fn zero_capacity_is_always_full() {
        let mut q = LaneQueue::new(ExecutionLane::MlxGpu, 0);
        assert_eq!(q.remaining(), 0);
        let err = q
            .try_push(entry(WorkPriority::Normal, next_work_id()))
            .expect_err("zero-capacity queue should reject all pushes");
        assert_eq!(err, BackpressureReason::MetalCapacity);
    }

    #[test]
    fn lane_roundtrip() {
        let q = LaneQueue::new(ExecutionLane::CoreMlAne, 8);
        assert_eq!(q.lane(), ExecutionLane::CoreMlAne);
        assert_eq!(q.capacity(), 8);
    }

    // ── LaneQueueSet tests ──────────────────────────────────────────────

    #[test]
    fn queue_set_new_and_pending() {
        let set = LaneQueueSet::new(4, 3, 8);
        assert_eq!(set.total_pending(), 0);
        assert!(set.snapshot().is_empty());
    }

    #[test]
    fn queue_set_queue_for_each_lane() {
        let mut set = LaneQueueSet::new(2, 2, 2);

        // Each queue is independent.
        let mq = set.queue_for(ExecutionLane::MlxGpu);
        assert_eq!(mq.lane(), ExecutionLane::MlxGpu);
        mq.try_push(entry(WorkPriority::Normal, next_work_id()))
            .unwrap();

        let aq = set.queue_for(ExecutionLane::CoreMlAne);
        assert_eq!(aq.lane(), ExecutionLane::CoreMlAne);
        aq.try_push(entry(WorkPriority::Normal, next_work_id()))
            .unwrap();

        let cq = set.queue_for(ExecutionLane::AccelerateCpu);
        assert_eq!(cq.lane(), ExecutionLane::AccelerateCpu);
        cq.try_push(entry(WorkPriority::Normal, next_work_id()))
            .unwrap();

        assert_eq!(set.total_pending(), 3);
        let snap = set.snapshot();
        assert_eq!(snap.len(), 3);
        assert_eq!(*snap.get(&ExecutionLane::MlxGpu).unwrap(), 1);
        assert_eq!(*snap.get(&ExecutionLane::CoreMlAne).unwrap(), 1);
        assert_eq!(*snap.get(&ExecutionLane::AccelerateCpu).unwrap(), 1);
    }

    #[test]
    fn queue_set_immutable_access() {
        let mut set = LaneQueueSet::new(2, 2, 2);
        set.queue_for(ExecutionLane::MlxGpu)
            .try_push(entry(WorkPriority::Normal, next_work_id()))
            .unwrap();

        let q = set.queue_for_lane(ExecutionLane::MlxGpu);
        assert_eq!(q.len(), 1);
        assert_eq!(q.lane(), ExecutionLane::MlxGpu);
    }

    #[test]
    fn queue_set_backpressure_each_lane() {
        // Each lane has depth 1; second push per lane fails.
        let mut set = LaneQueueSet::new(1, 1, 1);

        assert!(set
            .queue_for(ExecutionLane::MlxGpu)
            .try_push(entry(WorkPriority::Normal, next_work_id()))
            .is_ok());
        let err = set
            .queue_for(ExecutionLane::MlxGpu)
            .try_push(entry(WorkPriority::Normal, next_work_id()))
            .expect_err("metal should be full");
        assert_eq!(err, BackpressureReason::MetalCapacity);

        assert!(set
            .queue_for(ExecutionLane::CoreMlAne)
            .try_push(entry(WorkPriority::Normal, next_work_id()))
            .is_ok());
        let err = set
            .queue_for(ExecutionLane::CoreMlAne)
            .try_push(entry(WorkPriority::Normal, next_work_id()))
            .expect_err("ane should be full");
        assert_eq!(err, BackpressureReason::AneCapacity);

        assert!(set
            .queue_for(ExecutionLane::AccelerateCpu)
            .try_push(entry(WorkPriority::Normal, next_work_id()))
            .is_ok());
        let err = set
            .queue_for(ExecutionLane::AccelerateCpu)
            .try_push(entry(WorkPriority::Normal, next_work_id()))
            .expect_err("cpu should be full");
        assert_eq!(err, BackpressureReason::CpuCapacity);
    }

    #[test]
    fn queue_set_fallback_lane_goes_to_cpu() {
        let mut set = LaneQueueSet::new(1, 1, 1);
        // CandleCpu, Tensix, IntelLevelZero all go to accelerate.
        let q = set.queue_for(ExecutionLane::CandleCpu);
        assert_eq!(q.lane(), ExecutionLane::AccelerateCpu);
        let q = set.queue_for(ExecutionLane::Tensix);
        assert_eq!(q.lane(), ExecutionLane::AccelerateCpu);
        let q = set.queue_for(ExecutionLane::IntelLevelZero);
        assert_eq!(q.lane(), ExecutionLane::AccelerateCpu);
    }

    #[test]
    fn queue_set_snapshot_only_nonzero() {
        let mut set = LaneQueueSet::new(5, 5, 5);
        set.queue_for(ExecutionLane::MlxGpu)
            .try_push(entry(WorkPriority::Normal, next_work_id()))
            .unwrap();

        let snap = set.snapshot();
        assert_eq!(snap.len(), 1);
        assert_eq!(*snap.get(&ExecutionLane::MlxGpu).unwrap(), 1);
        assert!(snap.get(&ExecutionLane::CoreMlAne).is_none());
        assert!(snap.get(&ExecutionLane::AccelerateCpu).is_none());
    }

    #[test]
    fn backpressure_reason_mapping() {
        // Verify that lane -> reason mapping is consistent.
        // This tests the module-level `backpressure_for` via `try_push`.
        let mut q = LaneQueue::new(ExecutionLane::Tensix, 0);
        let err = q
            .try_push(entry(WorkPriority::Normal, next_work_id()))
            .expect_err("zero-cap Tensix");
        assert_eq!(err, BackpressureReason::ActivationSlots);

        let mut q = LaneQueue::new(ExecutionLane::IntelLevelZero, 0);
        let err = q
            .try_push(entry(WorkPriority::Normal, next_work_id()))
            .expect_err("zero-cap IntelLevelZero");
        assert_eq!(err, BackpressureReason::MetalCapacity);

        let mut q = LaneQueue::new(ExecutionLane::CandleCpu, 0);
        let err = q
            .try_push(entry(WorkPriority::Normal, next_work_id()))
            .expect_err("zero-cap CandleCpu");
        assert_eq!(err, BackpressureReason::CpuCapacity);
    }
}
