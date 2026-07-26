//! Per-lane slot lease lifecycle and access enforcement.
//!
//! Authority: this module owns the canonical lease lifecycle for one
//! output slot, including the read/write access protocol, the
//! reader-count tracking, and the force-release on cancellation. It
//! does not own lane capacity counters (those live in
//! `lane_capacity`), the lane queue (that lives in `lane_queue`), the
//! constitutional `ExecutionLease` (that lives in
//! `prism_ecs_constitutional::execution`), or the orchestrator
//! (that lives in `heterogeneous_executor`).
//!
//! Constitutional notes:
//!
//! - Lease identity is [`LeaseId`], a typed newtype that is
//!   independent of the engine's `SlotLeaseId`. The constitutional
//!   side does not import `activation_abi`.
//! - All canonical collections use `BTreeMap` per the
//!   "no HashMap/HashSet for canonical collections" rule.
//! - All fallible operations return [`Result<_, LeaseError>`], a
//!   `thiserror`-derived enum classified as `Conflict` (slot already
//!   has a writer / not yet ready), `NotFound` (lease id absent),
//!   and `Poisoned` (slot was force-marked and can no longer be
//!   acquired).
//!
//! Lease state machine:
//!
//! ```text
//!                         ┌──────────┐
//!                         │   Free   │
//!                         └────┬─────┘
//!                   ┌──────────┼──────────┐
//!                   │          │          │
//!              acquire    acquire      Poison
//!               write      read       (error)
//!                   │          │          │
//!              ┌────▼────┐ ┌──▼───┐  ┌────▼─────┐
//!              │ Write   │ │ Read │  │ Poisoned │
//!              │ Active  │ │Active│  └──────────┘
//!              └────┬────┘ └──┬───┘
//!                   │         │
//!          mark_    │    release
//!       output_ready│   (last reader)
//!                   │         │
//!              ┌────▼────┐   │
//!              │ Output  │   │
//!              │  Ready  │   │
//!              └────┬────┘   │
//!                   │        │
//!         acquire   │        │
//!          read ────┘        │
//!                   │        │
//!              ┌────▼────┐   │
//!              │Consumed │───┘
//!              └─────────┘
//! ```
//!
//! A slot can have multiple concurrent readers but only one writer.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ── Identity ──────────────────────────────────────────────────────────────

/// Typed identity for one output slot. The constitutional side does
/// not import the engine's `activation_abi::SlotLeaseId`; adapters
/// convert at the boundary.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct SlotId(pub u64);

impl SlotId {
    /// Construct a [`SlotId`] from a raw `u64`.
    pub const fn new(raw: u64) -> Self {
        Self(raw)
    }

    /// Borrow the raw slot ordinal.
    pub const fn raw(self) -> u64 {
        self.0
    }
}

/// Typed identity for one lease. The lease id is allocated by
/// [`SlotLeaseManager`] when a slot is acquired; it is the
/// constitutional surface the orchestrator uses to mark output ready
/// or release.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct LeaseId(pub u64);

impl LeaseId {
    /// Construct a [`LeaseId`] from a raw `u64`.
    pub const fn new(raw: u64) -> Self {
        Self(raw)
    }

    /// Borrow the raw lease ordinal.
    pub const fn raw(self) -> u64 {
        self.0
    }
}

/// Typed identity for a work item. Newtype around `u64`; the work-id
/// allocator hands out monotonic ids. Distinct from [`LeaseId`] so a
/// stale work id cannot accidentally reference a lease.
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

// ── Access mode ───────────────────────────────────────────────────────────

/// Access mode granted to a lease holder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SlotAccess {
    /// Read-only access to the slot contents.
    Read,
    /// Exclusive write access to the slot contents.
    Write,
    /// Unrestricted read + write access.
    ReadWrite,
}

// ── Lease state ───────────────────────────────────────────────────────────

/// Current lifecycle state of a slot lease.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LeaseState {
    /// Slot is free and available for any acquisition.
    Free,
    /// Reserved for writing (exclusive). Transition to [`WriteActive`]
    /// once the writer begins.
    WriteReserved,
    /// Actively being written by exactly one writer.
    WriteActive,
    /// Data is fully written and ready to be consumed by readers.
    OutputReady,
    /// Being read by one or more consumers.
    ReadActive,
    /// Writer released but readers are still active. Transitions to
    /// [`Free`] when the last consumer releases.
    Consumed,
    /// Slot is poisoned due to an unrecoverable failure and must be
    /// force-released.
    Poisoned,
}

// ── Errors ────────────────────────────────────────────────────────────────

/// Errors emitted by the lease manager. Classified as `Conflict` (a
/// write lease already exists, or a read cannot proceed because the
/// writer has not yet marked output ready), `NotFound` (lease id or
/// slot id absent from the manager), and `Poisoned` (the slot was
/// force-marked and can no longer be acquired).
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum LeaseError {
    /// The slot already has an active write lease; the new acquire
    /// is rejected.
    #[error("slot {slot:?} already has an active write lease (lease {existing:?})")]
    WriteConflict {
        /// The slot that was being acquired.
        slot: SlotId,
        /// The lease id of the existing writer.
        existing: LeaseId,
    },
    /// The slot is being actively written (lease in `WriteActive`);
    /// a read cannot proceed until `mark_output_ready` is called.
    #[error("slot {slot:?} is being written (lease {existing:?}); output not ready")]
    NotYetReady {
        /// The slot that was being read.
        slot: SlotId,
        /// The lease id of the still-writing lease.
        existing: LeaseId,
    },
    /// The lease id is not present in the manager.
    #[error("lease {lease:?} not found")]
    NotFound {
        /// The lease id that was searched for.
        lease: LeaseId,
    },
    /// The lease exists but is not in the expected state for this
    /// operation (e.g. `mark_output_ready` on a `ReadActive` lease).
    #[error("lease {lease:?} is in state {state:?}, expected {expected:?}")]
    BadState {
        /// The lease id.
        lease: LeaseId,
        /// The current state.
        state: LeaseState,
        /// The expected state for this operation.
        expected: LeaseState,
    },
    /// The slot is poisoned; no new acquisitions are allowed.
    #[error("slot {slot:?} is poisoned; cannot acquire")]
    Poisoned {
        /// The slot that was being acquired.
        slot: SlotId,
    },
}

// ── Slot lease ────────────────────────────────────────────────────────────

/// A single lease on a slot, tracking ownership, access mode,
/// lifecycle state, and consumer references.
///
/// Note: the `acquired_at` and `last_transition` fields are
/// in-process `Instant` values (not serializable as `Instant` does
/// not implement `Serialize`). The struct therefore derives
/// `Serialize` only — it is runtime-only state, and the durable
/// record is the `acquired` / `released` / `marked_ready` events in
/// the event store. The orchestrator reconstructs the lease from
/// the event history on replay.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SlotLease {
    /// Unique lease identifier.
    pub lease_id: LeaseId,
    /// Logical slot index within the arena.
    pub slot_id: SlotId,
    /// Work item that currently holds this lease.
    pub owner_work_id: WorkId,
    /// Session that owns this lease (typed as a `String`; the
    /// constitutional side accepts a string for session names — the
    /// orchestrator decides whether to use a typed `SessionId` or
    /// an opaque string at the call site).
    pub owner_session: String,
    /// Access mode granted.
    pub access: SlotAccess,
    /// Current lifecycle state.
    pub state: LeaseState,
    /// Wall-clock time when the lease was acquired. Not serialized.
    #[serde(skip)]
    pub acquired_at: Instant,
    /// Wall-clock time of the most recent state transition. Not
    /// serialized.
    #[serde(skip)]
    pub last_transition: Instant,
    /// Number of outstanding consumers (readers) on this slot.
    ///
    /// For a write lease this tracks how many readers are still
    /// active; the slot only transitions to [`LeaseState::Free`]
    /// once this reaches zero and the writer has released.
    pub consumer_count: u32,
}

impl SlotLease {
    /// True if this lease grants write access.
    pub const fn is_writer(&self) -> bool {
        matches!(self.access, SlotAccess::Write | SlotAccess::ReadWrite)
    }
}

// ── Slot lease manager ────────────────────────────────────────────────────

/// Manages the full lifecycle of slot leases across heterogeneous
/// backends.
///
/// # Invariants
///
/// * A slot can have at most one active write lease.
/// * A slot can have zero or more concurrent read leases.
/// * `acquire_read` fails if the slot has an active write lease in
///   [`LeaseState::WriteActive`] (data not yet ready).
/// * `acquire_read` succeeds if the slot has no active write lease,
///   or if the write lease is in [`LeaseState::OutputReady`].
/// * `mark_output_ready` may only be called by the write lease holder
///   and only when the lease is in [`LeaseState::WriteActive`].
/// * A write lease's `consumer_count` tracks the number of
///   outstanding readers. The slot transitions back to
///   [`LeaseState::Free`] only when all readers have released.
#[derive(Debug)]
pub struct SlotLeaseManager {
    /// All active leases keyed by lease id.
    leases: BTreeMap<LeaseId, SlotLease>,

    /// Map from slot id to the lease id of its active write lease.
    ///
    /// `None` if no writer is currently active.
    slot_write_lease: BTreeMap<SlotId, LeaseId>,

    /// Number of active readers per slot id.
    slot_readers: BTreeMap<SlotId, u32>,

    /// Set of slots that have been force-marked as poisoned; no
    /// new acquisitions are allowed for these slots.
    poisoned_slots: BTreeMap<SlotId, Instant>,

    /// Monotonically increasing lease-id generator.
    next_lease_id: AtomicU64,
}

impl Default for SlotLeaseManager {
    fn default() -> Self {
        Self::new()
    }
}

impl SlotLeaseManager {
    /// Create a new empty lease manager.
    pub fn new() -> Self {
        Self {
            leases: BTreeMap::new(),
            slot_write_lease: BTreeMap::new(),
            slot_readers: BTreeMap::new(),
            poisoned_slots: BTreeMap::new(),
            next_lease_id: AtomicU64::new(1),
        }
    }

    /// True if the slot has been marked poisoned (no acquisitions
    /// allowed).
    pub fn is_poisoned(&self, slot: SlotId) -> bool {
        self.poisoned_slots.contains_key(&slot)
    }

    // ── Write path ───────────────────────────────────────────────────

    /// Reserve a slot for exclusive writing.
    ///
    /// Returns [`LeaseError::WriteConflict`] if the slot already has
    /// an active write lease, or [`LeaseError::Poisoned`] if the slot
    /// has been force-marked. On success returns the new lease id.
    pub fn acquire_write(
        &mut self,
        slot_id: SlotId,
        work_id: WorkId,
        session: &str,
    ) -> Result<LeaseId, LeaseError> {
        if self.poisoned_slots.contains_key(&slot_id) {
            return Err(LeaseError::Poisoned { slot: slot_id });
        }
        if let Some(existing) = self.slot_write_lease.get(&slot_id) {
            return Err(LeaseError::WriteConflict {
                slot: slot_id,
                existing: *existing,
            });
        }

        let lease_id = LeaseId(self.next_lease_id.fetch_add(1, Ordering::Relaxed));
        let now = Instant::now();

        let lease = SlotLease {
            lease_id,
            slot_id,
            owner_work_id: work_id,
            owner_session: session.to_string(),
            access: SlotAccess::Write,
            state: LeaseState::WriteActive,
            acquired_at: now,
            last_transition: now,
            consumer_count: 0,
        };

        self.leases.insert(lease_id, lease);
        self.slot_write_lease.insert(slot_id, lease_id);

        Ok(lease_id)
    }

    // ── Read path ────────────────────────────────────────────────────

    /// Reserve a slot for reading.
    ///
    /// Multiple concurrent readers are allowed. Returns
    /// [`LeaseError::NotYetReady`] if the slot has an active write
    /// lease in [`LeaseState::WriteActive`]. Returns
    /// [`LeaseError::Poisoned`] if the slot has been force-marked.
    pub fn acquire_read(
        &mut self,
        slot_id: SlotId,
        work_id: WorkId,
        session: &str,
    ) -> Result<LeaseId, LeaseError> {
        if self.poisoned_slots.contains_key(&slot_id) {
            return Err(LeaseError::Poisoned { slot: slot_id });
        }

        // Reject if the slot has a write lease still writing.
        if let Some(write_lease_id) = self.slot_write_lease.get(&slot_id) {
            if let Some(write_lease) = self.leases.get(write_lease_id) {
                if write_lease.state == LeaseState::WriteActive {
                    return Err(LeaseError::NotYetReady {
                        slot: slot_id,
                        existing: *write_lease_id,
                    });
                }
            }
        }

        let lease_id = LeaseId(self.next_lease_id.fetch_add(1, Ordering::Relaxed));
        let now = Instant::now();

        let lease = SlotLease {
            lease_id,
            slot_id,
            owner_work_id: work_id,
            owner_session: session.to_string(),
            access: SlotAccess::Read,
            state: LeaseState::ReadActive,
            acquired_at: now,
            last_transition: now,
            consumer_count: 0,
        };

        self.leases.insert(lease_id, lease);

        // Track the reader on the slot and on the write lease (if any).
        let reader_count = self.slot_readers.entry(slot_id).or_insert(0);
        *reader_count = reader_count.saturating_add(1);

        if let Some(write_lease_id) = self.slot_write_lease.get(&slot_id) {
            if let Some(write_lease) = self.leases.get_mut(write_lease_id) {
                write_lease.consumer_count = write_lease.consumer_count.saturating_add(1);
            }
        }

        Ok(lease_id)
    }

    // ── State transitions ────────────────────────────────────────────

    /// Mark a write lease as having produced ready output.
    ///
    /// Only the write lease holder may call this, and the lease must
    /// be in [`LeaseState::WriteActive`].
    pub fn mark_output_ready(&mut self, lease_id: LeaseId) -> Result<(), LeaseError> {
        let lease = self
            .leases
            .get_mut(&lease_id)
            .ok_or(LeaseError::NotFound { lease: lease_id })?;

        if lease.state != LeaseState::WriteActive {
            return Err(LeaseError::BadState {
                lease: lease_id,
                state: lease.state,
                expected: LeaseState::WriteActive,
            });
        }

        lease.state = LeaseState::OutputReady;
        lease.last_transition = Instant::now();
        Ok(())
    }

    // ── Release ──────────────────────────────────────────────────────

    /// Release a lease.
    ///
    /// * **Writer** release transitions to [`LeaseState::Consumed`]
    ///   if there are outstanding readers, or cleans up the slot
    ///   immediately if there are none.
    /// * **Reader** release decrements the reader count. When the
    ///   last reader releases and the writer has already released
    ///   (state is [`LeaseState::Consumed`]), the slot is cleaned up.
    pub fn release(&mut self, lease_id: LeaseId) -> Result<(), LeaseError> {
        let lease = self
            .leases
            .get(&lease_id)
            .ok_or(LeaseError::NotFound { lease: lease_id })?
            .clone();

        if lease.is_writer() {
            self.release_writer(lease_id, lease.slot_id, lease.state)
        } else {
            self.release_reader(lease_id, lease.slot_id)
        }
    }

    fn release_writer(
        &mut self,
        lease_id: LeaseId,
        slot_id: SlotId,
        _state: LeaseState,
    ) -> Result<(), LeaseError> {
        let now = Instant::now();

        // Determine how many readers are still outstanding.
        let reader_count = self.slot_readers.get(&slot_id).copied().unwrap_or(0);

        if reader_count == 0 {
            // No active readers — clean up immediately.
            self.leases.remove(&lease_id);
            self.slot_write_lease.remove(&slot_id);
            self.slot_readers.remove(&slot_id);
        } else {
            // Readers still active — transition to Consumed and update
            // the lease state so the last reader's release completes
            // cleanup.
            if let Some(lease) = self.leases.get_mut(&lease_id) {
                lease.state = LeaseState::Consumed;
                lease.last_transition = now;
            }
        }

        Ok(())
    }

    fn release_reader(&mut self, lease_id: LeaseId, slot_id: SlotId) -> Result<(), LeaseError> {
        self.leases.remove(&lease_id);

        // Decrement slot-level reader count.
        let remaining = if let Some(count) = self.slot_readers.get_mut(&slot_id) {
            *count = count.saturating_sub(1);
            *count
        } else {
            0
        };

        // Decrement consumer_count on the write lease if one exists.
        if let Some(write_lease_id) = self.slot_write_lease.get(&slot_id) {
            if let Some(write_lease) = self.leases.get_mut(write_lease_id) {
                write_lease.consumer_count = write_lease.consumer_count.saturating_sub(1);
            }
        }

        // If the writer already released (Consumed) and no more
        // readers, clean up the slot.
        if remaining == 0 {
            let should_cleanup = self
                .slot_write_lease
                .get(&slot_id)
                .and_then(|wid| self.leases.get(wid))
                .map(|wl| wl.state == LeaseState::Consumed)
                .unwrap_or(false);

            if should_cleanup {
                if let Some(wid) = self.slot_write_lease.remove(&slot_id) {
                    self.leases.remove(&wid);
                }
            }

            self.slot_readers.remove(&slot_id);
        }

        Ok(())
    }

    // ── Bulk operations ──────────────────────────────────────────────

    /// Force-release all leases owned by a session (cancellation
    /// path). Returns the list of lease ids that were
    /// force-released.
    pub fn release_session(&mut self, session_id: &str) -> Vec<LeaseId> {
        let to_remove: Vec<LeaseId> = self
            .leases
            .iter()
            .filter(|(_, l)| l.owner_session == session_id)
            .map(|(id, _)| *id)
            .collect();

        for id in &to_remove {
            // `release` may fail (e.g. NotFound) if the lease was
            // already removed; we are best-effort on the cancellation
            // path. The cancel-then-release race is not a correctness
            // issue because the slot ownership invariants are
            // maintained by the per-lease state machine.
            let _ = self.release(*id);
        }

        to_remove
    }

    /// Mark a slot as poisoned. No new acquisitions are allowed for
    /// this slot. Existing leases are not released by this call —
    /// callers should pair it with [`Self::release_session`] or
    /// [`Self::release`] on each existing lease.
    pub fn poison_slot(&mut self, slot_id: SlotId) {
        self.poisoned_slots.insert(slot_id, Instant::now());
    }

    /// Un-poison a slot (recovery path). Use only when the cause of
    /// the poisoning has been resolved.
    pub fn unpoison_slot(&mut self, slot_id: SlotId) {
        self.poisoned_slots.remove(&slot_id);
    }

    // ── Observability ────────────────────────────────────────────────

    /// Return the number of active readers on a slot.
    pub fn reader_count(&self, slot_id: SlotId) -> u32 {
        self.slot_readers.get(&slot_id).copied().unwrap_or(0)
    }

    /// Return the active write lease for a slot, if any.
    pub fn current_writer(&self, slot_id: SlotId) -> Option<LeaseId> {
        self.slot_write_lease.get(&slot_id).copied()
    }

    /// Return a reference to the lease with the given id, if it
    /// exists.
    pub fn lease(&self, lease_id: LeaseId) -> Option<&SlotLease> {
        self.leases.get(&lease_id)
    }

    /// Number of active leases in the manager.
    pub fn active_lease_count(&self) -> usize {
        self.leases.len()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Identity tests ───────────────────────────────────────────────

    #[test]
    fn slot_id_serde_transparent() {
        let id = SlotId(42);
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(json, "42");
        let back: SlotId = serde_json::from_str(&json).unwrap();
        assert_eq!(back, SlotId(42));
    }

    #[test]
    fn lease_id_serde_transparent() {
        let id = LeaseId(7);
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(json, "7");
        let back: LeaseId = serde_json::from_str(&json).unwrap();
        assert_eq!(back, LeaseId(7));
    }

    #[test]
    fn work_id_serde_transparent() {
        let id = WorkId(99);
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(json, "99");
        let back: WorkId = serde_json::from_str(&json).unwrap();
        assert_eq!(back, WorkId(99));
    }

    // ── Construction tests ───────────────────────────────────────────

    #[test]
    fn new_manager_is_empty() {
        let mgr = SlotLeaseManager::new();
        assert_eq!(mgr.active_lease_count(), 0);
        assert_eq!(mgr.reader_count(SlotId(0)), 0);
        assert_eq!(mgr.current_writer(SlotId(0)), None);
    }

    // ── Write path tests ─────────────────────────────────────────────

    #[test]
    fn acquire_write_succeeds_for_free_slot() {
        let mut mgr = SlotLeaseManager::new();
        let lease = mgr
            .acquire_write(SlotId(0), WorkId(1), "session-1")
            .expect("free slot should accept writer");
        let entry = mgr.lease(lease).expect("lease should exist");
        assert_eq!(entry.state, LeaseState::WriteActive);
        assert_eq!(entry.access, SlotAccess::Write);
        assert_eq!(entry.owner_work_id, WorkId(1));
        assert_eq!(entry.owner_session, "session-1");
    }

    #[test]
    fn acquire_write_rejects_when_writer_active() {
        let mut mgr = SlotLeaseManager::new();
        let _first = mgr
            .acquire_write(SlotId(0), WorkId(1), "session-1")
            .unwrap();
        let err = mgr
            .acquire_write(SlotId(0), WorkId(2), "session-2")
            .expect_err("second write should fail");
        assert!(matches!(err, LeaseError::WriteConflict { .. }));
    }

    // ── Read path tests ──────────────────────────────────────────────

    #[test]
    fn acquire_read_succeeds_for_free_slot() {
        let mut mgr = SlotLeaseManager::new();
        let lease = mgr
            .acquire_read(SlotId(0), WorkId(1), "session-1")
            .expect("free slot should accept reader");
        let entry = mgr.lease(lease).expect("lease should exist");
        assert_eq!(entry.state, LeaseState::ReadActive);
        assert_eq!(entry.access, SlotAccess::Read);
        assert_eq!(mgr.reader_count(SlotId(0)), 1);
    }

    #[test]
    fn acquire_read_rejects_when_writing() {
        let mut mgr = SlotLeaseManager::new();
        let _writer = mgr
            .acquire_write(SlotId(0), WorkId(1), "session-1")
            .unwrap();
        let err = mgr
            .acquire_read(SlotId(0), WorkId(2), "session-2")
            .expect_err("read should fail while writer is active");
        assert!(matches!(err, LeaseError::NotYetReady { .. }));
    }

    #[test]
    fn acquire_read_succeeds_after_mark_output_ready() {
        let mut mgr = SlotLeaseManager::new();
        let writer = mgr
            .acquire_write(SlotId(0), WorkId(1), "session-1")
            .unwrap();
        mgr.mark_output_ready(writer).unwrap();
        let _reader = mgr
            .acquire_read(SlotId(0), WorkId(2), "session-2")
            .expect("read should succeed after output is ready");
        assert_eq!(mgr.reader_count(SlotId(0)), 1);
    }

    #[test]
    fn multiple_concurrent_readers_allowed() {
        let mut mgr = SlotLeaseManager::new();
        mgr.acquire_read(SlotId(0), WorkId(1), "session-1").unwrap();
        mgr.acquire_read(SlotId(0), WorkId(2), "session-2").unwrap();
        mgr.acquire_read(SlotId(0), WorkId(3), "session-3").unwrap();
        assert_eq!(mgr.reader_count(SlotId(0)), 3);
    }

    // ── mark_output_ready tests ──────────────────────────────────────

    #[test]
    fn mark_output_ready_succeeds_for_active_writer() {
        let mut mgr = SlotLeaseManager::new();
        let writer = mgr
            .acquire_write(SlotId(0), WorkId(1), "session-1")
            .unwrap();
        mgr.mark_output_ready(writer).unwrap();
        let entry = mgr.lease(writer).unwrap();
        assert_eq!(entry.state, LeaseState::OutputReady);
    }

    #[test]
    fn mark_output_ready_rejects_when_not_active_writer() {
        let mut mgr = SlotLeaseManager::new();
        let writer = mgr
            .acquire_write(SlotId(0), WorkId(1), "session-1")
            .unwrap();
        mgr.mark_output_ready(writer).unwrap();
        // Calling mark_output_ready a second time is a BadState error.
        let err = mgr
            .mark_output_ready(writer)
            .expect_err("second mark should fail");
        assert!(matches!(err, LeaseError::BadState { .. }));
    }

    #[test]
    fn mark_output_ready_rejects_unknown_lease() {
        let mut mgr = SlotLeaseManager::new();
        let err = mgr
            .mark_output_ready(LeaseId(99))
            .expect_err("unknown lease should fail");
        assert!(matches!(err, LeaseError::NotFound { .. }));
    }

    #[test]
    fn mark_output_ready_rejects_read_lease() {
        let mut mgr = SlotLeaseManager::new();
        let reader = mgr
            .acquire_read(SlotId(0), WorkId(1), "session-1")
            .unwrap();
        let err = mgr
            .mark_output_ready(reader)
            .expect_err("read lease should not accept mark_output_ready");
        assert!(matches!(err, LeaseError::BadState { .. }));
    }

    // ── Release tests ────────────────────────────────────────────────

    #[test]
    fn release_writer_with_no_readers_clears_slot() {
        let mut mgr = SlotLeaseManager::new();
        let writer = mgr
            .acquire_write(SlotId(0), WorkId(1), "session-1")
            .unwrap();
        mgr.release(writer).unwrap();
        assert_eq!(mgr.current_writer(SlotId(0)), None);
        assert_eq!(mgr.active_lease_count(), 0);
    }

    #[test]
    fn release_writer_with_readers_transitions_to_consumed() {
        let mut mgr = SlotLeaseManager::new();
        let writer = mgr
            .acquire_write(SlotId(0), WorkId(1), "session-1")
            .unwrap();
        mgr.mark_output_ready(writer).unwrap();
        let _r1 = mgr
            .acquire_read(SlotId(0), WorkId(2), "session-2")
            .unwrap();
        let _r2 = mgr
            .acquire_read(SlotId(0), WorkId(3), "session-3")
            .unwrap();

        mgr.release(writer).unwrap();
        // Slot still has readers; writer lease is now Consumed.
        let entry = mgr.lease(writer).unwrap();
        assert_eq!(entry.state, LeaseState::Consumed);
        // Reader count is still 2.
        assert_eq!(mgr.reader_count(SlotId(0)), 2);
    }

    #[test]
    fn last_reader_release_cleans_up_slot() {
        let mut mgr = SlotLeaseManager::new();
        let writer = mgr
            .acquire_write(SlotId(0), WorkId(1), "session-1")
            .unwrap();
        mgr.mark_output_ready(writer).unwrap();
        let r1 = mgr
            .acquire_read(SlotId(0), WorkId(2), "session-2")
            .unwrap();

        mgr.release(writer).unwrap();
        mgr.release(r1).unwrap();
        // Both reader and writer are gone; slot is clean.
        assert_eq!(mgr.current_writer(SlotId(0)), None);
        assert_eq!(mgr.reader_count(SlotId(0)), 0);
        assert_eq!(mgr.active_lease_count(), 0);
    }

    #[test]
    fn release_unknown_lease_returns_not_found() {
        let mut mgr = SlotLeaseManager::new();
        let err = mgr
            .release(LeaseId(99))
            .expect_err("unknown lease should fail");
        assert!(matches!(err, LeaseError::NotFound { .. }));
    }

    // ── Bulk operation tests ─────────────────────────────────────────

    #[test]
    fn release_session_returns_released_leases() {
        let mut mgr = SlotLeaseManager::new();
        let w1 = mgr
            .acquire_write(SlotId(0), WorkId(1), "session-1")
            .unwrap();
        let w2 = mgr
            .acquire_write(SlotId(1), WorkId(2), "session-1")
            .unwrap();
        let _w3 = mgr
            .acquire_write(SlotId(2), WorkId(3), "session-2")
            .unwrap();

        let released = mgr.release_session("session-1");
        assert_eq!(released.len(), 2);
        assert!(released.contains(&w1));
        assert!(released.contains(&w2));
        // session-2's writer is untouched.
        assert_eq!(mgr.current_writer(SlotId(2)), Some(LeaseId(3)));
    }

    #[test]
    fn release_session_no_match_returns_empty() {
        let mut mgr = SlotLeaseManager::new();
        let _w = mgr
            .acquire_write(SlotId(0), WorkId(1), "session-1")
            .unwrap();
        let released = mgr.release_session("unknown-session");
        assert!(released.is_empty());
    }

    // ── Poison tests ─────────────────────────────────────────────────

    #[test]
    fn poison_slot_blocks_new_acquisitions() {
        let mut mgr = SlotLeaseManager::new();
        mgr.poison_slot(SlotId(0));
        assert!(mgr.is_poisoned(SlotId(0)));

        let err = mgr
            .acquire_write(SlotId(0), WorkId(1), "session-1")
            .expect_err("poisoned slot should reject writer");
        assert!(matches!(err, LeaseError::Poisoned { .. }));

        let err = mgr
            .acquire_read(SlotId(0), WorkId(2), "session-2")
            .expect_err("poisoned slot should reject reader");
        assert!(matches!(err, LeaseError::Poisoned { .. }));
    }

    #[test]
    fn unpoison_slot_allows_new_acquisitions() {
        let mut mgr = SlotLeaseManager::new();
        mgr.poison_slot(SlotId(0));
        mgr.unpoison_slot(SlotId(0));
        assert!(!mgr.is_poisoned(SlotId(0)));
        mgr.acquire_write(SlotId(0), WorkId(1), "session-1")
            .expect("unpoisoned slot should accept writer");
    }

    // ── Observability tests ──────────────────────────────────────────

    #[test]
    fn current_writer_returns_active_write_lease() {
        let mut mgr = SlotLeaseManager::new();
        let writer = mgr
            .acquire_write(SlotId(0), WorkId(1), "session-1")
            .unwrap();
        assert_eq!(mgr.current_writer(SlotId(0)), Some(writer));
        assert_eq!(mgr.current_writer(SlotId(1)), None);
    }

    #[test]
    fn lease_returns_none_for_unknown_id() {
        let mgr = SlotLeaseManager::new();
        assert!(mgr.lease(LeaseId(99)).is_none());
    }

    #[test]
    fn slot_lease_is_writer() {
        let mut mgr = SlotLeaseManager::new();
        let writer = mgr
            .acquire_write(SlotId(0), WorkId(1), "session-1")
            .unwrap();
        let reader = mgr
            .acquire_read(SlotId(1), WorkId(2), "session-2")
            .unwrap();
        assert!(mgr.lease(writer).unwrap().is_writer());
        assert!(!mgr.lease(reader).unwrap().is_writer());
    }
}
