//! Ring-buffered staging regions for cross-lane data transfer.
//!
//! `StagingRing<T>` is a fixed-depth (4-slot) ring buffer that mediates
//! ownership of data slots between CPU, ANE, and GPU lanes via CAS-based
//! state transitions.  Each slot is an independent state machine; producers
//! and consumers coordinate through atomic transitions rather than mutual
//! exclusion.
//!
//! # State machine
//!
//! ```text
//!   Empty ──────────→ CpuFilled
//!   CpuFilled ──────→ AneSubmitted
//!   CpuFilled ──────→ GpuSubmitted
//!   AneSubmitted ───→ AneComplete
//!   GpuSubmitted ───→ GpuComplete
//!   AneComplete ────→ CpuValidated
//!   GpuComplete ────→ CpuValidated
//!   CpuValidated ───→ Empty      (via try_pop)
//!   Reclaimable ────→ Empty      (cleanup / time-out recovery)
//! ```
//!
//! # Safety
//!
//! - `T: Send` is required because a value written by one thread may be read
//!   by a different thread after a successful slot transition.
//! - All data access is gated by CAS on the slot's `AtomicU8` state.
//!   The consumer transitions the slot to [`SlotState::Reclaimable`] before
//!   reading — this guarantees exclusive ownership and prevents a concurrent
//!   producer from writing into the same slot.

use core::cell::UnsafeCell;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};

// ---------------------------------------------------------------------------
// SlotState
// ---------------------------------------------------------------------------

/// States in the cross-lane staging slot lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SlotState {
    Empty = 0,
    CpuFilled = 1,
    AneSubmitted = 2,
    AneComplete = 3,
    GpuSubmitted = 4,
    GpuComplete = 5,
    CpuValidated = 6,
    Reclaimable = 7,
}

impl SlotState {
    /// Convert a raw `u8` discriminant back to a `SlotState`.
    ///
    /// # Panics
    ///
    /// Panics if `v` does not correspond to a valid variant.
    #[inline]
    fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Empty,
            1 => Self::CpuFilled,
            2 => Self::AneSubmitted,
            3 => Self::AneComplete,
            4 => Self::GpuSubmitted,
            5 => Self::GpuComplete,
            6 => Self::CpuValidated,
            7 => Self::Reclaimable,
            _ => panic!("invalid SlotState discriminant: {v}"),
        }
    }
}

// ---------------------------------------------------------------------------
// StagingRing
// ---------------------------------------------------------------------------

/// A fixed-depth (4-slot) ring buffer for cross-lane data transfer.
///
/// Every slot is a mini state machine.  Data is written into a slot that
/// is in [`SlotState::Empty`] and progresses through CPU-fill, lane-submit,
/// lane-complete, validation, and finally back to Empty.
///
/// Thread safety is provided by CAS on each slot's state byte; the `head`
/// and `tail` cursors give a hint about where to start scanning but never
/// gate access — the CAS does that.
pub struct StagingRing<T> {
    /// Per-slot state bytes (one per slot).
    slot_states: [AtomicU8; 4],
    /// Slot data.  Only the thread that owns the slot (via a successful CAS
    /// state transition) is allowed to read or write the corresponding cell.
    data: [UnsafeCell<Option<T>>; 4],
    /// HINT for producers — next slot to try first when looking for `Empty`.
    head: AtomicUsize,
    /// HINT for consumers — next slot to try first when looking for a
    /// consumable state.
    tail: AtomicUsize,
}

// SAFETY: `StagingRing<T>` manages all data access through CAS on slot
// states.  `T: Send` suffices because values are moved between threads
// (never shared).  The struct itself is `Sync` so that `&StagingRing<T>`
// can be passed across threads.
unsafe impl<T: Send> Sync for StagingRing<T> {}

impl<T: Send> StagingRing<T> {
    /// Fixed ring depth (4 slots).
    #[inline]
    pub const fn depth() -> usize {
        4
    }

    /// Construct an empty staging ring with all slots in `Empty` state.
    pub fn new() -> Self {
        Self {
            slot_states: [
                AtomicU8::new(SlotState::Empty as u8),
                AtomicU8::new(SlotState::Empty as u8),
                AtomicU8::new(SlotState::Empty as u8),
                AtomicU8::new(SlotState::Empty as u8),
            ],
            data: [
                UnsafeCell::new(None),
                UnsafeCell::new(None),
                UnsafeCell::new(None),
                UnsafeCell::new(None),
            ],
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
        }
    }

    /// Try to push a value into the ring.
    ///
    /// Scans up to `depth()` slots looking for one in `Empty` state,
    /// starting from the current `head` hint.  When found, transitions
    /// the slot to `CpuFilled` and stores the value.
    ///
    /// Returns the slot index on success, or `Err("ring full")` if every
    /// slot is occupied.
    pub fn try_push(&self, value: T) -> Result<usize, String> {
        let hint = self.head.load(Ordering::Relaxed);
        for i in 0..4 {
            let idx = (hint + i) % 4;
            // Try to claim the slot.  CAS Empty→CpuFilled with AcqRel so
            // that subsequent data writes are visible to the thread that
            // observes CpuFilled.
            let prev = self.slot_states[idx].compare_exchange(
                SlotState::Empty as u8,
                SlotState::CpuFilled as u8,
                Ordering::AcqRel,
                Ordering::Relaxed,
            );
            if prev.is_ok() {
                // We own this slot — safe to write.
                unsafe { *self.data[idx].get() = Some(value) };
                // Advance the producer hint.
                self.head.store((idx + 1) % 4, Ordering::Relaxed);
                return Ok(idx);
            }
        }
        Err("ring full".into())
    }

    /// Try to pop a completed value from the ring.
    ///
    /// Looks for a slot whose state is `GpuComplete` or `CpuValidated`,
    /// starting from the `tail` hint.  The slot is first CAS'd to
    /// `Reclaimable` (exclusive consumer ownership), the value is read,
    /// and finally the state is set to `Empty`.
    pub fn try_pop(&self) -> Option<(usize, T)> {
        let hint = self.tail.load(Ordering::Relaxed);
        for i in 0..4 {
            let idx = (hint + i) % 4;
            let state = self.slot_state(idx);
            if matches!(state, SlotState::GpuComplete | SlotState::CpuValidated) {
                // Claim the slot by transitioning to Reclaimable.
                if self
                    .slot_states[idx]
                    .compare_exchange(
                        state as u8,
                        SlotState::Reclaimable as u8,
                        Ordering::AcqRel,
                        Ordering::Relaxed,
                    )
                    .is_ok()
                {
                    // Exclusive ownership — safe to read.
                    let value = unsafe { (*self.data[idx].get()).take().unwrap() };
                    // Release the slot.  The CAS that put us here is the
                    // only way Reclaimable is reached, so a plain store
                    // with Release is sufficient.
                    self.slot_states[idx].store(SlotState::Empty as u8, Ordering::Release);
                    self.tail.store((idx + 1) % 4, Ordering::Relaxed);
                    return Some((idx, value));
                }
            }
        }
        None
    }

    /// Atomically transition a slot from one state to another.
    ///
    /// Returns `Ok(())` on success, or `Err` with the actual current state
    /// on mismatch.
    pub fn transition(&self, idx: usize, from: SlotState, to: SlotState) -> Result<(), String> {
        let actual = self.slot_states[idx].compare_exchange(
            from as u8,
            to as u8,
            Ordering::AcqRel,
            Ordering::Relaxed,
        );
        match actual {
            Ok(_) => Ok(()),
            Err(v) => Err(format!(
                "slot {idx}: expected {:?} but found {:?}",
                from,
                SlotState::from_u8(v)
            )),
        }
    }

    /// Read the current state of a slot.
    #[inline]
    pub fn slot_state(&self, idx: usize) -> SlotState {
        SlotState::from_u8(self.slot_states[idx].load(Ordering::Acquire))
    }
}

impl<T: Send> Default for StagingRing<T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    // -----------------------------------------------------------------------
    // 1. State transitions — valid and invalid
    // -----------------------------------------------------------------------

    #[test]
    fn test_valid_state_transitions() {
        let ring = StagingRing::<u32>::new();

        // All slots start Empty.
        for i in 0..4 {
            assert_eq!(ring.slot_state(i), SlotState::Empty);
        }

        // Empty → CpuFilled
        assert!(ring.transition(0, SlotState::Empty, SlotState::CpuFilled).is_ok());
        assert_eq!(ring.slot_state(0), SlotState::CpuFilled);

        // CpuFilled → AneSubmitted
        assert!(ring.transition(0, SlotState::CpuFilled, SlotState::AneSubmitted).is_ok());
        assert_eq!(ring.slot_state(0), SlotState::AneSubmitted);

        // AneSubmitted → AneComplete
        assert!(ring.transition(0, SlotState::AneSubmitted, SlotState::AneComplete).is_ok());
        assert_eq!(ring.slot_state(0), SlotState::AneComplete);

        // AneComplete → CpuValidated
        assert!(ring.transition(0, SlotState::AneComplete, SlotState::CpuValidated).is_ok());
        assert_eq!(ring.slot_state(0), SlotState::CpuValidated);

        // CpuValidated → Empty (via pop path — done manually via transition)
        assert!(ring.transition(0, SlotState::CpuValidated, SlotState::Reclaimable).is_ok());
        assert!(ring.transition(0, SlotState::Reclaimable, SlotState::Empty).is_ok());
        assert_eq!(ring.slot_state(0), SlotState::Empty);

        // Now test the GPU path on slot 1.
        assert!(ring.transition(1, SlotState::Empty, SlotState::CpuFilled).is_ok());
        assert!(ring.transition(1, SlotState::CpuFilled, SlotState::GpuSubmitted).is_ok());
        assert!(ring.transition(1, SlotState::GpuSubmitted, SlotState::GpuComplete).is_ok());
        assert!(ring.transition(1, SlotState::GpuComplete, SlotState::CpuValidated).is_ok());
        assert_eq!(ring.slot_state(1), SlotState::CpuValidated);
    }

    // -----------------------------------------------------------------------
    // 2. Invalid transitions
    // -----------------------------------------------------------------------

    #[test]
    fn test_invalid_transition() {
        let ring = StagingRing::<u32>::new();

        // Cannot skip states: Empty → AneSubmitted is invalid.
        let err = ring.transition(0, SlotState::Empty, SlotState::AneSubmitted);
        assert!(err.is_err(), "skipping states should fail");
        assert!(err.unwrap_err().contains("Empty"));

        // Empty → CpuFilled, then try duplicate transition.
        assert!(ring.transition(0, SlotState::Empty, SlotState::CpuFilled).is_ok());
        let err = ring.transition(0, SlotState::Empty, SlotState::CpuFilled);
        assert!(err.is_err(), "double-claim should fail");
        assert!(err.unwrap_err().contains("CpuFilled"));
    }

    // -----------------------------------------------------------------------
    // 3. Push / pop lifecycle
    // -----------------------------------------------------------------------

    #[test]
    fn test_push_pop_lifecycle() {
        let ring = StagingRing::new();

        // Push four values.
        for v in 0..4 {
            let idx = ring.try_push(v).expect("push should succeed");
            assert_eq!(ring.slot_state(idx), SlotState::CpuFilled);
        }

        // Ring is full.
        assert!(ring.try_push(99).is_err());

        // Simulate lane processing for every slot: CpuFilled → GpuSubmitted → GpuComplete.
        for i in 0..4 {
            assert!(ring
                .transition(i, SlotState::CpuFilled, SlotState::GpuSubmitted)
                .is_ok());
            assert!(ring
                .transition(i, SlotState::GpuSubmitted, SlotState::GpuComplete)
                .is_ok());
        }

        // Pop all four.
        for v in 0..4 {
            let popped = ring.try_pop().expect("pop should succeed");
            assert_eq!(popped.1, v, "value mismatch at pop index {v}");
            // Slot is now Empty.
            assert_eq!(
                ring.slot_state(popped.0),
                SlotState::Empty,
                "slot {} should be Empty after pop",
                popped.0
            );
        }

        // Ring is empty.
        assert!(ring.try_pop().is_none());
    }

    // -----------------------------------------------------------------------
    // 4. Concurrent push/pop (stress)
    // -----------------------------------------------------------------------

    #[test]
    fn test_concurrent_push_pop() {
        let ring = Arc::new(StagingRing::new());
        let n_items = 100;

        let ring_prod = Arc::clone(&ring);
        let producer = thread::spawn(move || {
            for v in 0..n_items {
                loop {
                    if ring_prod.try_push(v).is_ok() {
                        break;
                    }
                    // Spin — ring may be full.
                    thread::yield_now();
                }
            }
        });

        let ring_cons = Arc::clone(&ring);
        let consumer = thread::spawn(move || {
            let mut seen = vec![false; n_items];
            let mut popped = 0;
            while popped < n_items {
                if let Some((_idx, v)) = ring_cons.try_pop() {
                    assert!(
                        !std::mem::replace(&mut seen[v as usize], true),
                        "duplicate value {v}"
                    );
                    popped += 1;
                } else {
                    thread::yield_now();
                }
            }
        });

        producer.join().expect("producer panicked");
        consumer.join().expect("consumer panicked");
    }

    // -----------------------------------------------------------------------
    // 5. Concurrent state transitions (multiple threads claiming the same
    //    slot should never double-count)
    // -----------------------------------------------------------------------

    #[test]
    fn test_concurrent_transition_raf() {
        let ring = Arc::new(StagingRing::new());

        // Set up one slot as CpuFilled.
        ring.try_push(42).unwrap();
        // Set up another as GpuComplete.
        ring.transition(0, SlotState::CpuFilled, SlotState::GpuSubmitted).unwrap();
        ring.transition(0, SlotState::GpuSubmitted, SlotState::GpuComplete).unwrap();

        let n_threads = 8;
        let mut handles = Vec::with_capacity(n_threads);

        for _ in 0..n_threads {
            let r = Arc::clone(&ring);
            handles.push(thread::spawn(move || {
                // All threads try to pop the same GpuComplete slot.
                r.try_pop()
            }));
        }

        let mut success_count = 0;
        for h in handles {
            if h.join().unwrap().is_some() {
                success_count += 1;
            }
        }

        // Exactly one thread should have claimed the slot.
        assert_eq!(
            success_count, 1,
            "exactly one thread should pop a slot; {success_count} succeeded"
        );
    }

    // -----------------------------------------------------------------------
    // 6. Ring wrap-around (head/tail advancing past index 4 → 0)
    // -----------------------------------------------------------------------

    #[test]
    fn test_ring_wrap_around() {
        let ring = StagingRing::new();

        // Fill and drain the ring twice to force head/tail wrap.
        for cycle in 0..2 {
            for v in 0..4 {
                ring.try_push(v + cycle * 100).unwrap();
            }
            for v in 0..4 {
                let (idx, val) = ring.try_pop().unwrap();
                assert_eq!(val, v + cycle * 100);
                assert!(idx < 4);
            }
        }

        // Third cycle — verify fresh push also lands in a valid slot.
        ring.try_push(200).unwrap();
        assert_eq!(ring.slot_state(0), SlotState::CpuFilled);
        // Advance to GpuComplete, pop, confirm value.
        ring.transition(0, SlotState::CpuFilled, SlotState::GpuSubmitted).unwrap();
        ring.transition(0, SlotState::GpuSubmitted, SlotState::GpuComplete).unwrap();
        let (_, val) = ring.try_pop().unwrap();
        assert_eq!(val, 200);
    }
}
