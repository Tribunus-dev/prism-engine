//! PRISM-PRODUCTION-HETEROGENEOUS-EXECUTOR-0001 — Slot lease lifecycle and access enforcement.
//!
//! Manages IOSurface / arena slot lease lifecycle — acquire, release,
//! transfer ownership. Extends the existing [`SlotLeaseManager`] in
//! `tri_lane_orchestrator` with access modes and consumer tracking.
//!
//! # Lease states
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

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::compilation::activation_abi::SlotLeaseId;
use crate::compilation::phase_ir::PhaseId;
use crate::scheduling::lane_work::WorkId;

// ── Access mode ─────────────────────────────────────────────────────────────

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

// ── Lease state ─────────────────────────────────────────────────────────────

/// Current lifecycle state of a slot lease.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LeaseState {
    /// Slot is free and available for any acquisition.
    Free,
    /// Reserved for writing (exclusive). Transition to [`WriteActive`] once
    /// the writer begins.
    WriteReserved,
    /// Actively being written by exactly one writer.
    WriteActive,
    /// Data is fully written and ready to be consumed by readers.
    OutputReady,
    /// Being read by one or more consumers.
    ReadActive,
    /// Writer released but readers are still active. Transitions to [`Free`]
    /// when the last consumer releases.
    Consumed,
    /// Slot is poisoned due to an unrecoverable failure and must be
    /// force-released.
    Poisoned,
}

// ── Slot lease ──────────────────────────────────────────────────────────────

/// A single lease on an IOSurface / arena slot, tracking ownership,
/// access mode, lifecycle state, and consumer references.
#[derive(Debug, Clone)]
pub struct SlotLease {
    /// Unique lease identifier.
    pub lease_id: SlotLeaseId,
    /// Logical slot index within the arena.
    pub slot_id: u64,
    /// Arena identifier (if backed by an arena).
    pub arena_id: Option<String>,
    /// IOSurface identifier (if backed by an IOSurface).
    pub iosurface_id: Option<u64>,
    /// Work item that currently holds this lease.
    pub owner_work_id: WorkId,
    /// Session that owns this lease.
    pub owner_session_id: String,
    /// Compilation phase that owns this lease.
    pub owner_phase_id: PhaseId,
    /// Access mode granted.
    pub access: SlotAccess,
    /// Current lifecycle state.
    pub state: LeaseState,
    /// Wall-clock time when the lease was acquired.
    pub acquired_at: Instant,
    /// Wall-clock time of the most recent state transition.
    pub last_transition: Instant,
    /// Number of outstanding consumers (readers) on this slot.
    ///
    /// For a write lease this tracks how many readers are still active;
    /// the slot only transitions to [`Free`] once this reaches zero
    /// and the writer has released.
    pub consumer_count: u32,
}

// ── Slot lease manager ──────────────────────────────────────────────────────

/// Manages the full lifecycle of slot leases across heterogeneous backends.
///
/// # Invariants
///
/// * A slot can have at most one active write lease.
/// * A slot can have zero or more concurrent read leases.
/// * `acquire_read` fails if the slot has an active write lease in
///   [`WriteActive`] (data not yet ready).
/// * `acquire_read` succeeds if the slot has no active write lease, or if
///   the write lease is in [`OutputReady`].
/// * `mark_output_ready` may only be called by the write lease holder.
/// * A write lease's [`consumer_count`] tracks the number of outstanding
///   readers. The slot transitions back to [`Free`] only when all readers
///   have released.
pub struct SlotLeaseManager {
    /// All active leases keyed by lease id.
    leases: HashMap<SlotLeaseId, SlotLease>,

    /// Map from slot id to the lease id of its active write lease.
    ///
    /// `None` if no writer is currently active.
    slot_write_lease: HashMap<u64, SlotLeaseId>,

    /// Number of active readers per slot id.
    slot_readers: HashMap<u64, u32>,

    /// Monotonically increasing lease-id generator.
    next_lease_id: AtomicU64,
}

impl SlotLeaseManager {
    /// Create a new empty lease manager.
    pub fn new() -> Self {
        Self {
            leases: HashMap::new(),
            slot_write_lease: HashMap::new(),
            slot_readers: HashMap::new(),
            next_lease_id: AtomicU64::new(1),
        }
    }

    // ── Write path ───────────────────────────────────────────────────────

    /// Reserve a slot for exclusive writing.
    ///
    /// Returns an error if the slot already has an active write lease.
    pub fn acquire_write(
        &mut self,
        slot_id: u64,
        work_id: WorkId,
        session: &str,
        phase: PhaseId,
    ) -> Result<SlotLeaseId, String> {
        if self.slot_write_lease.contains_key(&slot_id) {
            return Err(format!(
                "slot {} already has an active write lease",
                slot_id
            ));
        }

        let lease_id = SlotLeaseId(self.next_lease_id.fetch_add(1, Ordering::Relaxed));
        let now = Instant::now();

        let lease = SlotLease {
            lease_id,
            slot_id,
            arena_id: None,
            iosurface_id: None,
            owner_work_id: work_id,
            owner_session_id: session.to_string(),
            owner_phase_id: phase,
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

    // ── Read path ─────────────────────────────────────────────────────────

    /// Reserve a slot for reading.
    ///
    /// Multiple concurrent readers are allowed. Returns an error if the slot
    /// has an active write lease in [`WriteActive`] (data not yet written).
    pub fn acquire_read(
        &mut self,
        slot_id: u64,
        work_id: WorkId,
        session: &str,
        phase: PhaseId,
    ) -> Result<SlotLeaseId, String> {
        // Reject if the slot has a write lease still writing.
        if let Some(write_lease_id) = self.slot_write_lease.get(&slot_id) {
            if let Some(write_lease) = self.leases.get(write_lease_id) {
                if write_lease.state == LeaseState::WriteActive {
                    return Err(format!(
                        "slot {} is still being written (lease {})",
                        slot_id, write_lease.lease_id.0
                    ));
                }
            }
        }

        let lease_id = SlotLeaseId(self.next_lease_id.fetch_add(1, Ordering::Relaxed));
        let now = Instant::now();

        let lease = SlotLease {
            lease_id,
            slot_id,
            arena_id: None,
            iosurface_id: None,
            owner_work_id: work_id,
            owner_session_id: session.to_string(),
            owner_phase_id: phase,
            access: SlotAccess::Read,
            state: LeaseState::ReadActive,
            acquired_at: now,
            last_transition: now,
            consumer_count: 0,
        };

        self.leases.insert(lease_id, lease);

        // Track the reader on the slot and on the write lease (if any).
        *self.slot_readers.entry(slot_id).or_insert(0) += 1;
        if let Some(write_lease_id) = self.slot_write_lease.get(&slot_id) {
            if let Some(write_lease) = self.leases.get_mut(write_lease_id) {
                write_lease.consumer_count += 1;
            }
        }

        Ok(lease_id)
    }

    // ── State transitions ─────────────────────────────────────────────────

    /// Mark a write lease as having produced ready output.
    ///
    /// Only the write lease holder may call this, and the lease must be in
    /// [`WriteActive`] state.
    pub fn mark_output_ready(&mut self, lease_id: SlotLeaseId) -> Result<(), String> {
        let lease = self
            .leases
            .get_mut(&lease_id)
            .ok_or_else(|| format!("lease {} not found", lease_id.0))?;

        if lease.state != LeaseState::WriteActive {
            return Err(format!(
                "lease {} is {:?}, expected WriteActive to mark output ready",
                lease_id.0, lease.state
            ));
        }

        lease.state = LeaseState::OutputReady;
        lease.last_transition = Instant::now();
        Ok(())
    }

    // ── Release ───────────────────────────────────────────────────────────

    /// Release a lease.
    ///
    /// * **Writer** release transitions to [`Consumed`] if there are
    ///   outstanding readers, or to [`Free`] (slot reclaimed) if there
    ///   are none.
    /// * **Reader** release decrements the reader count. When the last
    ///   reader releases and the writer has already released (state is
    ///   [`Consumed`]), the slot transitions to [`Free`].
    pub fn release(&mut self, lease_id: SlotLeaseId) -> Result<(), String> {
        let lease = self
            .leases
            .get(&lease_id)
            .ok_or_else(|| format!("lease {} not found", lease_id.0))?;

        let slot_id = lease.slot_id;
        let is_writer = matches!(lease.access, SlotAccess::Write | SlotAccess::ReadWrite);
        let state = lease.state;
        let _ = lease; // borrow ends — we need &mut self below.

        if is_writer {
            self.release_writer(lease_id, slot_id, state)
        } else {
            self.release_reader(lease_id, slot_id)
        }
    }

    /// Internal: release a write lease.
    fn release_writer(
        &mut self,
        lease_id: SlotLeaseId,
        slot_id: u64,
        _state: LeaseState,
    ) -> Result<(), String> {
        let now = Instant::now();

        // Determine how many readers are still outstanding.
        let reader_count = self.slot_readers.get(&slot_id).copied().unwrap_or(0);

        if reader_count == 0 {
            // No active readers — clean up immediately.
            self.leases.remove(&lease_id);
            self.slot_write_lease.remove(&slot_id);
            self.slot_readers.remove(&slot_id);
        } else {
            // Readers still active — transition to Consumed and update the
            // lease state so the last reader's release completes cleanup.
            if let Some(lease) = self.leases.get_mut(&lease_id) {
                lease.state = LeaseState::Consumed;
                lease.last_transition = now;
            }
        }

        Ok(())
    }

    /// Internal: release a read lease.
    fn release_reader(&mut self, lease_id: SlotLeaseId, slot_id: u64) -> Result<(), String> {
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

        // If the writer already released (Consumed) and no more readers,
        // clean up the slot.
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

    // ── Bulk operations ─────────────────────────────────────────────────

    /// Force-release all leases owned by a session (cancellation path).
    ///
    /// Returns the list of lease ids that were force-released.
    pub fn release_session(&mut self, session_id: &str) -> Vec<SlotLeaseId> {
        // Collect lease ids to remove.
        let to_remove: Vec<SlotLeaseId> = self
            .leases
            .iter()
            .filter(|(_, l)| l.owner_session_id == session_id)
            .map(|(id, _)| *id)
            .collect();

        let mut released = Vec::with_capacity(to_remove.len());

        for lease_id in &to_remove {
            if let Some(lease) = self.leases.remove(lease_id) {
                released.push(*lease_id);

                // Clean up slot-level bookkeeping.
                let slot_id = lease.slot_id;
                if matches!(lease.access, SlotAccess::Write | SlotAccess::ReadWrite) {
                    self.slot_write_lease.remove(&slot_id);
                }
                if let Some(count) = self.slot_readers.get_mut(&slot_id) {
                    *count = count.saturating_sub(1);
                    if *count == 0 {
                        self.slot_readers.remove(&slot_id);
                    }
                }
            }
        }

        released
    }

    // ── Accessors ────────────────────────────────────────────────────────

    /// Look up a lease by its id.
    pub fn get(&self, lease_id: SlotLeaseId) -> Option<&SlotLease> {
        self.leases.get(&lease_id)
    }

    /// Mutably look up a lease by its id.
    pub fn get_mut(&mut self, lease_id: SlotLeaseId) -> Option<&mut SlotLease> {
        self.leases.get_mut(&lease_id)
    }

    /// Number of currently active leases.
    pub fn active_count(&self) -> usize {
        self.leases.len()
    }

    /// Number of currently free (unleased) slots.
    ///
    /// This counts slots that have a lease entry in one of the tracking
    /// maps; slots with no entry at all are implicitly free.
    pub fn free_count(&self) -> usize {
        let tracked: HashSet<u64> = self
            .leases
            .values()
            .map(|l| l.slot_id)
            .chain(self.slot_readers.keys().copied())
            .collect();
        // Return the number of tracked slots that have no active lease
        // of any kind.  This is meaningful only when the caller knows
        // the total pool size; without it we report the tracked-free set.
        tracked
            .iter()
            .filter(|sid| {
                !self.slot_write_lease.contains_key(sid)
                    && self.slot_readers.get(sid).copied().unwrap_or(0) == 0
            })
            .count()
    }

    /// Return all currently active lease ids.
    pub fn all_leases(&self) -> Vec<SlotLeaseId> {
        self.leases.keys().copied().collect()
    }
}

impl Default for SlotLeaseManager {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    fn sample_work_id(id: u64) -> WorkId {
        WorkId(id)
    }

    fn sample_phase_id(id: u64) -> PhaseId {
        PhaseId(id)
    }

    #[test]
    fn test_acquire_write_creates_active_lease() {
        let mut mgr = SlotLeaseManager::new();
        let lid = mgr
            .acquire_write(1, sample_work_id(10), "session-a", sample_phase_id(100))
            .unwrap();

        let lease = mgr.get(lid).unwrap();
        assert_eq!(lease.slot_id, 1);
        assert_eq!(lease.access, SlotAccess::Write);
        assert_eq!(lease.state, LeaseState::WriteActive);
        assert_eq!(lease.owner_session_id, "session-a");
        assert_eq!(mgr.active_count(), 1);
    }

    #[test]
    fn test_acquire_write_rejects_duplicate_writer() {
        let mut mgr = SlotLeaseManager::new();
        mgr.acquire_write(1, sample_work_id(10), "session-a", sample_phase_id(100))
            .unwrap();

        let err = mgr
            .acquire_write(1, sample_work_id(20), "session-b", sample_phase_id(200))
            .unwrap_err();
        assert!(err.contains("already has an active write lease"));
    }

    #[test]
    fn test_acquire_read_rejects_during_write_active() {
        let mut mgr = SlotLeaseManager::new();
        mgr.acquire_write(1, sample_work_id(10), "writer", sample_phase_id(100))
            .unwrap();

        let err = mgr
            .acquire_read(1, sample_work_id(11), "reader", sample_phase_id(101))
            .unwrap_err();
        assert!(err.contains("still being written"));
    }

    #[test]
    fn test_acquire_read_succeeds_after_output_ready() {
        let mut mgr = SlotLeaseManager::new();
        let wlid = mgr
            .acquire_write(1, sample_work_id(10), "writer", sample_phase_id(100))
            .unwrap();
        mgr.mark_output_ready(wlid).unwrap();

        let rlid = mgr
            .acquire_read(1, sample_work_id(11), "reader", sample_phase_id(101))
            .unwrap();
        let lease = mgr.get(rlid).unwrap();
        assert_eq!(lease.access, SlotAccess::Read);
        assert_eq!(lease.state, LeaseState::ReadActive);

        // Writer's consumer_count should be 1.
        let write_lease = mgr.get(wlid).unwrap();
        assert_eq!(write_lease.consumer_count, 1);
    }

    #[test]
    fn test_multiple_concurrent_readers() {
        let mut mgr = SlotLeaseManager::new();
        let wlid = mgr
            .acquire_write(1, sample_work_id(10), "writer", sample_phase_id(100))
            .unwrap();
        mgr.mark_output_ready(wlid).unwrap();

        let r1 = mgr
            .acquire_read(1, sample_work_id(11), "r1", sample_phase_id(101))
            .unwrap();
        let _r2 = mgr
            .acquire_read(1, sample_work_id(12), "r2", sample_phase_id(102))
            .unwrap();

        assert_eq!(mgr.active_count(), 3); // writer + 2 readers
        assert_eq!(mgr.get(wlid).unwrap().consumer_count, 2);
        assert_eq!(mgr.slot_readers.get(&1).copied().unwrap_or(0), 2);

        // Release one reader.
        mgr.release(r1).unwrap();
        assert_eq!(mgr.active_count(), 2);
        assert_eq!(mgr.get(wlid).unwrap().consumer_count, 1);
    }

    #[test]
    fn test_writer_release_consumed_then_reader_release_frees_slot() {
        let mut mgr = SlotLeaseManager::new();
        let wlid = mgr
            .acquire_write(1, sample_work_id(10), "writer", sample_phase_id(100))
            .unwrap();
        mgr.mark_output_ready(wlid).unwrap();

        let rlid = mgr
            .acquire_read(1, sample_work_id(11), "reader", sample_phase_id(101))
            .unwrap();

        // Writer releases first — goes to Consumed.
        mgr.release(wlid).unwrap();
        assert_eq!(mgr.get(wlid).map(|l| l.state), Some(LeaseState::Consumed));

        // Reader releases — slot freed, write lease also cleaned up.
        mgr.release(rlid).unwrap();
        assert!(mgr.get(wlid).is_none());
        assert!(mgr.slot_write_lease.get(&1).is_none());
        assert!(mgr.slot_readers.get(&1).is_none());
    }

    #[test]
    fn test_writer_release_no_readers_frees_immediately() {
        let mut mgr = SlotLeaseManager::new();
        let wlid = mgr
            .acquire_write(1, sample_work_id(10), "writer", sample_phase_id(100))
            .unwrap();
        mgr.mark_output_ready(wlid).unwrap();

        mgr.release(wlid).unwrap();
        assert!(mgr.get(wlid).is_none());
        assert!(mgr.slot_write_lease.get(&1).is_none());
    }

    #[test]
    fn test_mark_output_ready_fails_for_non_writer() {
        let mut mgr = SlotLeaseManager::new();
        let _wlid = mgr
            .acquire_write(1, sample_work_id(10), "writer", sample_phase_id(100))
            .unwrap();

        // A non-existent lease id.
        let fake = SlotLeaseId(999);
        let err = mgr.mark_output_ready(fake).unwrap_err();
        assert!(err.contains("not found"));
    }

    #[test]
    fn test_mark_output_ready_fails_from_wrong_state() {
        let mut mgr = SlotLeaseManager::new();
        let wlid = mgr
            .acquire_write(1, sample_work_id(10), "writer", sample_phase_id(100))
            .unwrap();
        mgr.mark_output_ready(wlid).unwrap();

        // Calling it again from OutputReady should fail.
        let err = mgr.mark_output_ready(wlid).unwrap_err();
        assert!(err.contains("expected WriteActive"));
    }

    #[test]
    fn test_release_session_force_clears_all() {
        let mut mgr = SlotLeaseManager::new();
        let wlid = mgr
            .acquire_write(
                1,
                sample_work_id(10),
                "session-cancel",
                sample_phase_id(100),
            )
            .unwrap();
        mgr.mark_output_ready(wlid).unwrap();

        let rlid = mgr
            .acquire_read(
                1,
                sample_work_id(11),
                "session-cancel",
                sample_phase_id(101),
            )
            .unwrap();

        let released = mgr.release_session("session-cancel");
        assert_eq!(released.len(), 2);
        assert!(released.contains(&wlid));
        assert!(released.contains(&rlid));

        assert_eq!(mgr.active_count(), 0);
        assert!(mgr.slot_write_lease.get(&1).is_none());
        assert!(mgr.slot_readers.get(&1).is_none());
    }

    #[test]
    fn test_release_session_does_not_affect_other_sessions() {
        let mut mgr = SlotLeaseManager::new();
        let wlid = mgr
            .acquire_write(1, sample_work_id(10), "cancel-me", sample_phase_id(100))
            .unwrap();
        mgr.mark_output_ready(wlid).unwrap();

        let other = mgr
            .acquire_read(1, sample_work_id(11), "keep-me", sample_phase_id(101))
            .unwrap();

        let released = mgr.release_session("cancel-me");
        assert_eq!(released.len(), 1);
        assert!(mgr.get(other).is_some()); // other session still valid
        assert_eq!(mgr.active_count(), 1);
    }

    #[test]
    fn test_release_twice_errors() {
        let mut mgr = SlotLeaseManager::new();
        let wlid = mgr
            .acquire_write(1, sample_work_id(10), "s", sample_phase_id(100))
            .unwrap();
        mgr.release(wlid).unwrap();
        let err = mgr.release(wlid).unwrap_err();
        assert!(err.contains("not found"));
    }

    #[test]
    fn test_all_leases_and_active_count() {
        let mut mgr = SlotLeaseManager::new();
        assert_eq!(mgr.active_count(), 0);
        assert!(mgr.all_leases().is_empty());

        let wlid = mgr
            .acquire_write(1, sample_work_id(10), "s", sample_phase_id(100))
            .unwrap();
        assert_eq!(mgr.active_count(), 1);
        assert_eq!(mgr.all_leases(), vec![wlid]);
    }

    #[test]
    fn test_lease_timestamps_advance() {
        let mut mgr = SlotLeaseManager::new();
        let wlid = mgr
            .acquire_write(1, sample_work_id(10), "s", sample_phase_id(100))
            .unwrap();

        let acquired = mgr.get(wlid).unwrap().acquired_at;
        let first_transition = mgr.get(wlid).unwrap().last_transition;

        // Small sleep to ensure time advances.
        thread::sleep(std::time::Duration::from_millis(1));

        mgr.mark_output_ready(wlid).unwrap();
        let second_transition = mgr.get(wlid).unwrap().last_transition;
        assert!(second_transition > first_transition);
        assert_eq!(mgr.get(wlid).unwrap().acquired_at, acquired); // unchanged
    }

    #[test]
    fn test_acquire_read_on_unleased_slot_succeeds() {
        // Reading from a slot with no write lease at all should be fine
        // (e.g. pre-populated data).
        let mut mgr = SlotLeaseManager::new();
        let rlid = mgr
            .acquire_read(42, sample_work_id(10), "s", sample_phase_id(100))
            .unwrap();

        let lease = mgr.get(rlid).unwrap();
        assert_eq!(lease.slot_id, 42);
        assert_eq!(lease.access, SlotAccess::Read);
        assert_eq!(lease.state, LeaseState::ReadActive);
    }

    #[test]
    fn test_read_then_release_on_unleased_slot() {
        let mut mgr = SlotLeaseManager::new();
        let rlid = mgr
            .acquire_read(42, sample_work_id(10), "s", sample_phase_id(100))
            .unwrap();
        mgr.release(rlid).unwrap();
        assert_eq!(mgr.active_count(), 0);
    }

    #[test]
    fn test_free_count_no_leases() {
        let mgr = SlotLeaseManager::new();
        // With no tracked slots at all, free_count returns 0 because there
        // is nothing to count.
        assert_eq!(mgr.free_count(), 0);
    }

    #[test]
    fn test_release_session_reuses_slot_ids() {
        let mut mgr = SlotLeaseManager::new();
        mgr.acquire_write(1, sample_work_id(10), "s1", sample_phase_id(100))
            .unwrap();
        mgr.release_session("s1");

        // After cancelling session s1, slot 1 should be reusable by s2.
        let wlid2 = mgr
            .acquire_write(1, sample_work_id(20), "s2", sample_phase_id(200))
            .unwrap();
        assert!(mgr.get(wlid2).is_some());
        assert_eq!(mgr.get(wlid2).unwrap().slot_id, 1);
    }
}
