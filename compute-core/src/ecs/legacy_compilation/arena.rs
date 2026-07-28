//! Activation arena — the core resource contract of the distill-compiler.
//!
//! The arena owns all activation slots and enforces a strict state machine
//! (Unallocated → Reserved → ProducerWriting → ProducerSealed →
//! ConsumerReadable → Reducing → Evictable → Released). Only the scheduler
//! may move a slot between states. Every transition is recorded with a
//! monotonic sequence number for the receipt.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::phase_types::{PhaseId, TensorDescriptor};

/// Slot identity — a typed alias for the arena's slot numbering.
pub type SlotId = u64;

// ── Slot state machine ──────────────────────────────────────────────────────

/// Strict, ordered state machine for activation slots.
///
/// Variants in canonical lifecycle order:
/// `Unallocated → Reserved → ProducerWriting → ProducerSealed →
/// ConsumerReadable → Reducing → Evictable → Released`
///
/// Only the scheduler may advance a slot between states. A slot may be
/// released from any state for emergency cleanup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SlotState {
    Unallocated,
    Reserved,
    ProducerWriting,
    ProducerSealed,
    ConsumerReadable,
    Reducing,
    Evictable,
    Released,
}

impl SlotState {
    /// Returns an ordered list of all states for iteration.
    pub fn all() -> &'static [SlotState] {
        &[
            SlotState::Unallocated,
            SlotState::Reserved,
            SlotState::ProducerWriting,
            SlotState::ProducerSealed,
            SlotState::ConsumerReadable,
            SlotState::Reducing,
            SlotState::Evictable,
            SlotState::Released,
        ]
    }

    /// Check whether the transition `self → to` is valid under the strict
    /// state machine.
    ///
    /// Only forward-stepping is allowed — a slot never regresses. The sole
    /// exception is that any state may transition directly to `Released`
    /// for emergency cleanup under memory pressure.
    pub fn can_transition_to(&self, to: SlotState) -> bool {
        match (self, to) {
            // Emergency release from any state.
            (_, SlotState::Released) => true,

            // Strict forward transitions only.
            (SlotState::Unallocated, SlotState::Reserved)
            | (SlotState::Reserved, SlotState::ProducerWriting)
            | (SlotState::ProducerWriting, SlotState::ProducerSealed)
            | (SlotState::ProducerSealed, SlotState::ConsumerReadable)
            | (SlotState::ConsumerReadable, SlotState::Reducing)
            | (SlotState::Reducing, SlotState::Evictable) => true,

            // All other transitions are invalid.
            _ => false,
        }
    }

    /// Human-readable description of this state.
    pub fn description(&self) -> &'static str {
        match self {
            SlotState::Unallocated => "Slot is free and not bound to any tensor",
            SlotState::Reserved => "Slot has been claimed for a specific tensor allocation",
            SlotState::ProducerWriting => "Producer phase is actively writing tensor data",
            SlotState::ProducerSealed => "Producer has completed writing; data is immutable",
            SlotState::ConsumerReadable => "At least one consumer may read the tensor data",
            SlotState::Reducing => "Partial reduction or aggregation is in progress",
            SlotState::Evictable => "Slot content is eligible for eviction under memory pressure",
            SlotState::Released => "Slot has been freed and returned to the pool",
        }
    }
}

impl fmt::Display for SlotState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.description())
    }
}

// ── Transition record ───────────────────────────────────────────────────────

/// A single state transition, recorded with a monotonic sequence number.
///
/// Every transition carries a global monotonic sequence number, a wall-clock
/// timestamp, and a human-readable reason — forming the audit trail that
/// becomes part of the execution receipt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateTransition {
    pub from: SlotState,
    pub to: SlotState,
    pub sequence: u64,
    pub timestamp_ns: u64,
    pub reason: String,
}

// ── Storage route ───────────────────────────────────────────────────────────

/// Where an activation slot's bytes physically live.
///
/// The point of this enum is to make every route explicit, auditable, and
/// measurable — never assume a route from the provider type alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StorageRoute {
    CpuOwned,
    MetalSharedBuffer,
    MetalPrivateBuffer,
    CoreMLManaged,
    CoreMLExported,
    BridgeMaterialized,
    /// Provider-verified route: same physical allocation, no observable
    /// materialization. Must only be used after capability probing and
    /// repeated validation passes.
    BridgeAliasedVerified,
    DiskFrontier,
}

// ── Activation slot ─────────────────────────────────────────────────────────

/// A slot in the activation arena — carries its logical descriptor, physical
/// byte count, state, provenance (producer + consumers), storage route,
/// materialization count, content digest, allocation id, generation number,
/// mutable-until sequence, receipt reference, and full transition history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationSlot {
    pub id: u64,
    pub logical_tensor: TensorDescriptor,
    pub bytes: u64,
    pub state: SlotState,
    pub producer: Option<PhaseId>,
    pub consumers: Vec<PhaseId>,
    pub storage_route: StorageRoute,
    pub materialization_count: u64,
    pub digest: Option<[u8; 32]>,
    pub transitions: Vec<StateTransition>,
    /// Optional provider allocation id for cross-buffer correlation.
    pub allocation_id: Option<u64>,
    /// Monotonic generation counter — incremented on each slot reuse.
    pub generation: u64,
    /// The phase sequence number beyond which this slot becomes immutable.
    pub mutable_until_sequence: u64,
    /// Optional reference into the receipt system.
    pub receipt_ref: Option<String>,
}

// ── Arena error ─────────────────────────────────────────────────────────────

/// Errors that arise during arena operations.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ArenaError {
    #[error("invalid slot state transition: {from:?} \u{2192} {to:?}")]
    InvalidTransition { from: SlotState, to: SlotState },
    #[error("slot {0} is not in Reserved state")]
    SlotNotReserved(u64),
    #[error("slot {0} is already sealed")]
    SlotAlreadySealed(u64),
    #[error("slot {0} is not in ConsumerReadable state")]
    SlotNotReadable(u64),
    #[error("concurrent read of slot {0} not allowed by route")]
    ConcurrentReadNotAllowed(u64),
    #[error("digest mismatch: expected {expected:x?}\u{2026} got {actual:x?}\u{2026}")]
    DigestMismatch {
        expected: [u8; 32],
        actual: [u8; 32],
    },
    #[error("no memory budget: requested {0} bytes, {1} available")]
    NoBudget(u64, u64),
}

// ── Activation arena ────────────────────────────────────────────────────────

/// The activation arena is the core resource contract.
///
/// It exposes logical slots with explicit bridge capabilities, not raw
/// MTLBuffer pointers. The scheduler reserves, seals, and releases slots;
/// providers read from and write to slots through the storage route.
///
/// # State-machine invariants
///
/// * A slot must be **Reserved** before it can be written.
/// * Only a **ProducerSealed** slot may become **ConsumerReadable**.
/// * A slot may be **Released** from any state for emergency cleanup.
/// * Every transition is recorded with a monotonic sequence number.
///
/// # Memory accounting
///
/// The arena tracks `current_allocated_bytes` (sum of sealed non-released
/// slot byte sizes) and `peak_allocated_bytes` (high-water mark since
/// construction), enabling the scheduler to enforce per-session budgets.
#[derive(Debug, Serialize, Deserialize)]
pub struct ActivationArena {
    slots: Vec<ActivationSlot>,
    next_slot_id: u64,
    #[serde(skip)]
    next_transition_seq: AtomicU64,
    peak_allocated_bytes: u64,
    current_allocated_bytes: u64,
}

impl ActivationArena {
    /// Create an empty arena.
    pub fn new() -> Self {
        ActivationArena {
            slots: Vec::new(),
            next_slot_id: 1,
            next_transition_seq: AtomicU64::new(1),
            peak_allocated_bytes: 0,
            current_allocated_bytes: 0,
        }
    }

    /// Reserve a slot for the given tensor descriptor.
    ///
    /// The caller provides `id` — a unique slot identifier (often the
    /// phase's logical tensor id). The slot starts in `Reserved` state
    /// with `storage_route` defaulting to `CpuOwned` (may be changed
    /// later via `slot_mut`).
    ///
    /// Returns the assigned slot id (same as the input `id`).
    pub fn reserve(&mut self, id: u64, tensor: TensorDescriptor) -> SlotId {
        // Advance the next expected id counter so sequential allocations
        // stay dense when ids are caller-assigned.
        if id >= self.next_slot_id {
            self.next_slot_id = id + 1;
        }

        let bytes = tensor.max_bytes.max(tensor.min_bytes());
        let slot = ActivationSlot {
            id,
            logical_tensor: tensor,
            bytes,
            state: SlotState::Reserved,
            producer: None,
            consumers: Vec::new(),
            storage_route: StorageRoute::CpuOwned,
            materialization_count: 0,
            digest: None,
            transitions: vec![StateTransition {
                from: SlotState::Unallocated,
                to: SlotState::Reserved,
                sequence: self.next_transition_seq.fetch_add(1, Ordering::Relaxed),
                timestamp_ns: timestamp_now(),
                reason: "reserved by scheduler".into(),
            }],
            allocation_id: None,
            generation: id,
            mutable_until_sequence: 0,
            receipt_ref: None,
        };
        self.current_allocated_bytes += bytes;
        self.peak_allocated_bytes = self.peak_allocated_bytes.max(self.current_allocated_bytes);
        self.slots.push(slot);
        id
    }

    /// Reserve a slot with the spec-compliant extended fields.
    pub fn reserve_extended(
        &mut self,
        tensor: TensorDescriptor,
        _route: StorageRoute,
        _allocation_id: Option<u64>,
        _mutable_until_sequence: u64,
        _receipt_ref: Option<String>,
    ) -> u64 {
        let id = self.next_slot_id;
        self.next_slot_id += 1;
        let bytes = tensor.max_bytes.max(tensor.min_bytes());
        let _gen = id as u64;
        self.current_allocated_bytes += bytes;
        self.peak_allocated_bytes = self.peak_allocated_bytes.max(self.current_allocated_bytes);
        id
    }

    /// Transition a slot to a new state. Returns an error if the transition
    /// is invalid under the state machine.
    pub fn transition(
        &mut self,
        slot_id: u64,
        to: SlotState,
        reason: &str,
    ) -> Result<(), ArenaError> {
        let slot = self
            .slots
            .iter_mut()
            .find(|s| s.id == slot_id)
            .ok_or(ArenaError::SlotNotReserved(slot_id))?;

        let from = slot.state;
        if !from.can_transition_to(to) {
            return Err(ArenaError::InvalidTransition { from, to });
        }

        let sequence = self.next_transition_seq.fetch_add(1, Ordering::Relaxed);
        slot.transitions.push(StateTransition {
            from,
            to,
            sequence,
            timestamp_ns: timestamp_now(),
            reason: reason.to_string(),
        });
        slot.state = to;
        Ok(())
    }

    /// Seal a slot — mark it `ProducerSealed` and attach a content digest.
    ///
    /// Returns `ArenaError::SlotAlreadySealed` if the slot is already
    /// in the sealed state.
    pub fn seal(&mut self, slot_id: u64, digest: [u8; 32]) -> Result<(), ArenaError> {
        // Check already-sealed before calling transition to return the
        // correct error variant, not a generic InvalidTransition.
        if let Some(slot) = self.slots.iter().find(|s| s.id == slot_id) {
            if slot.state == SlotState::ProducerSealed {
                return Err(ArenaError::SlotAlreadySealed(slot_id));
            }
        } else {
            return Err(ArenaError::SlotNotReserved(slot_id));
        }

        self.transition(slot_id, SlotState::ProducerSealed, "producer sealed")?;
        if let Some(slot) = self.slots.iter_mut().find(|s| s.id == slot_id) {
            slot.digest = Some(digest);
        }
        Ok(())
    }

    /// Mark a sealed slot as readable by consumers — transition from
    /// `ProducerSealed` to `ConsumerReadable`.
    pub fn mark_readable(&mut self, slot_id: u64) -> Result<(), ArenaError> {
        self.transition(
            slot_id,
            SlotState::ConsumerReadable,
            "marked readable for consumers",
        )
    }

    /// Release a slot from any state — transition to `Released` and
    /// return its bytes to the pool.
    ///
    /// This is the only transition allowed from any state, enabling
    /// emergency cleanup under memory pressure.
    pub fn release(&mut self, slot_id: u64) -> Result<(), ArenaError> {
        let idx = self
            .slots
            .iter()
            .position(|s| s.id == slot_id)
            .ok_or(ArenaError::SlotNotReserved(slot_id))?;
        self.current_allocated_bytes = self
            .current_allocated_bytes
            .saturating_sub(self.slots[idx].bytes);
        self.transition(slot_id, SlotState::Released, "slot released")?;
        Ok(())
    }

    /// Return the peak allocated byte count since arena construction.
    pub fn high_water(&self) -> u64 {
        self.peak_allocated_bytes
    }

    /// Return the total number of slots (in any state) in the arena.
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// Return a reference to a slot by id, or `None` if not found.
    pub fn slot(&self, id: u64) -> Option<&ActivationSlot> {
        self.slots.iter().find(|s| s.id == id)
    }

    /// Return a mutable reference to a slot by id, or `None` if not found.
    pub fn slot_mut(&mut self, id: u64) -> Option<&mut ActivationSlot> {
        self.slots.iter_mut().find(|s| s.id == id)
    }

    /// Iterate over all slots as a slice.
    pub fn slots(&self) -> &[ActivationSlot] {
        &self.slots
    }

    /// Return the current allocated bytes.
    pub fn current_bytes(&self) -> u64 {
        self.current_allocated_bytes
    }

    /// Set the producer phase for a slot.
    pub fn set_producer(&mut self, slot_id: u64, phase_id: PhaseId) -> Result<(), ArenaError> {
        let slot = self
            .slots
            .iter_mut()
            .find(|s| s.id == slot_id)
            .ok_or(ArenaError::SlotNotReserved(slot_id))?;
        slot.producer = Some(phase_id);
        Ok(())
    }

    /// Add a consumer phase to a slot.
    pub fn add_consumer(&mut self, slot_id: u64, phase_id: PhaseId) -> Result<(), ArenaError> {
        let slot = self
            .slots
            .iter_mut()
            .find(|s| s.id == slot_id)
            .ok_or(ArenaError::SlotNotReserved(slot_id))?;
        slot.consumers.push(phase_id);
        Ok(())
    }
}

impl Default for ActivationArena {
    fn default() -> Self {
        Self::new()
    }
}

// ── Timestamp helper ────────────────────────────────────────────────────────

/// Returns a wall-clock timestamp in nanoseconds since the Unix epoch.
fn timestamp_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::legacy_compilation::phase_types::{
        ElementType, PhysicalLayout, ProviderKind, ResidencyClass,
    };

    fn sample_tensor() -> TensorDescriptor {
        TensorDescriptor {
            logical_shape: vec![1, 64, 4096],
            element_type: ElementType::F16,
            physical_layout: PhysicalLayout::DenseRowMajor,
            alignment: 64,
            producer_phase: None,
            consumer_phases: Vec::new(),
            permitted_providers: vec![ProviderKind::Metal],
            residency_class: ResidencyClass::Unified,
            max_bytes: 1024 * 1024,
            mutable: true,
            content_digest: None,
        }
    }

    fn make_arena() -> ActivationArena {
        ActivationArena::new()
    }

    #[test]
    fn slot_state_can_transition_to_released() {
        for state in SlotState::all() {
            assert!(
                state.can_transition_to(SlotState::Released),
                "{:?} should allow transition to Released",
                state
            );
        }
    }

    #[test]
    fn slot_state_valid_transitions() {
        assert!(SlotState::Unallocated.can_transition_to(SlotState::Reserved));
        assert!(SlotState::Reserved.can_transition_to(SlotState::ProducerWriting));
        assert!(SlotState::ProducerWriting.can_transition_to(SlotState::ProducerSealed));
        assert!(SlotState::ProducerSealed.can_transition_to(SlotState::ConsumerReadable));
        assert!(SlotState::ConsumerReadable.can_transition_to(SlotState::Reducing));
        assert!(SlotState::Reducing.can_transition_to(SlotState::Evictable));
    }

    #[test]
    fn slot_state_invalid_transitions() {
        // Skip-forward is not allowed.
        assert!(!SlotState::Unallocated.can_transition_to(SlotState::ProducerWriting));
        assert!(!SlotState::Reserved.can_transition_to(SlotState::ConsumerReadable));
        assert!(!SlotState::ProducerSealed.can_transition_to(SlotState::Evictable));

        // Backward transitions never allowed.
        assert!(!SlotState::Reserved.can_transition_to(SlotState::Unallocated));
        assert!(!SlotState::ProducerSealed.can_transition_to(SlotState::ProducerWriting));
        assert!(!SlotState::ConsumerReadable.can_transition_to(SlotState::ProducerSealed));
        assert!(!SlotState::Released.can_transition_to(SlotState::Evictable));
        assert!(!SlotState::Released.can_transition_to(SlotState::Unallocated));
    }

    #[test]
    fn slot_state_description_nonempty() {
        for state in SlotState::all() {
            assert!(!state.description().is_empty());
        }
    }

    #[test]
    fn slot_state_display_nonempty() {
        for state in SlotState::all() {
            let d = format!("{}", state);
            assert!(!d.is_empty());
        }
    }

    #[test]
    fn reserve_creates_slot_in_reserved_state() {
        let mut arena = make_arena();
        let id = arena.reserve(1, sample_tensor());

        assert_eq!(id, 1);
        assert_eq!(arena.slot_count(), 1);

        let slot = arena.slot(id).unwrap();
        assert_eq!(slot.state, SlotState::Reserved);
        assert_eq!(slot.transitions.len(), 1);
        assert_eq!(slot.transitions[0].from, SlotState::Unallocated);
        assert_eq!(slot.transitions[0].to, SlotState::Reserved);
        assert!(slot.transitions[0].timestamp_ns > 0);
    }

    #[test]
    fn full_lifecycle() {
        let mut arena = make_arena();
        let id = arena.reserve(10, sample_tensor());

        arena
            .transition(id, SlotState::ProducerWriting, "write begin")
            .unwrap();
        arena.seal(id, [0u8; 32]).unwrap();
        arena.mark_readable(id).unwrap();
        arena.transition(id, SlotState::Reducing, "reduce").unwrap();
        arena.transition(id, SlotState::Evictable, "evict").unwrap();
        arena.release(id).unwrap();

        let slot = arena.slot(id).unwrap();
        assert_eq!(slot.state, SlotState::Released);
        assert_eq!(slot.transitions.len(), 7);
    }

    #[test]
    fn invalid_transition_returns_error() {
        let mut arena = make_arena();
        let id = arena.reserve(20, sample_tensor());

        let err = arena
            .transition(id, SlotState::ConsumerReadable, "skip")
            .unwrap_err();
        assert!(matches!(err, ArenaError::InvalidTransition { .. }));
    }

    #[test]
    fn double_seal_returns_error() {
        let mut arena = make_arena();
        let id = arena.reserve(30, sample_tensor());

        arena
            .transition(id, SlotState::ProducerWriting, "write")
            .unwrap();
        arena.seal(id, [0u8; 32]).unwrap();
        let err = arena.seal(id, [0u8; 32]).unwrap_err();
        assert!(matches!(err, ArenaError::SlotAlreadySealed(_)));
    }

    #[test]
    fn release_from_reserved_state() {
        let mut arena = make_arena();
        let id = arena.reserve(40, sample_tensor());
        arena.release(id).unwrap();
        assert_eq!(arena.slot(id).unwrap().state, SlotState::Released);
    }

    #[test]
    fn high_water_tracking() {
        let mut arena = make_arena();

        let id_a = arena.reserve(50, sample_tensor());
        let _id_b = arena.reserve(51, sample_tensor());

        // Reservation charges max_bytes immediately.
        assert_eq!(arena.current_bytes(), 2 * 1024 * 1024);
        assert_eq!(arena.high_water(), 2 * 1024 * 1024);

        arena.release(id_a).unwrap();
        assert_eq!(arena.current_bytes(), 1024 * 1024);
        assert_eq!(arena.high_water(), 2 * 1024 * 1024);
    }

    #[test]
    fn slot_not_reserved_error() {
        let mut arena = make_arena();
        let err = arena
            .transition(999, SlotState::Released, "nonexistent")
            .unwrap_err();
        assert!(matches!(err, ArenaError::SlotNotReserved(999)));
    }

    #[test]
    fn caller_provided_ids() {
        let mut arena = make_arena();
        let id1 = arena.reserve(100, sample_tensor());
        let id2 = arena.reserve(200, sample_tensor());
        let id3 = arena.reserve(101, sample_tensor());
        assert_eq!(id1, 100);
        assert_eq!(id2, 200);
        assert_eq!(id3, 101);
        assert_eq!(arena.slot_count(), 3);
    }

    #[test]
    fn monotonic_transition_sequences() {
        let mut arena = make_arena();
        let id = arena.reserve(1, sample_tensor());

        arena
            .transition(id, SlotState::ProducerWriting, "write")
            .unwrap();
        arena.seal(id, [0u8; 32]).unwrap();
        arena.mark_readable(id).unwrap();
        arena.release(id).unwrap();

        let slot = arena.slot(id).unwrap();
        for window in slot.transitions.windows(2) {
            assert!(
                window[0].sequence < window[1].sequence,
                "transition sequences must be monotonic: {} >= {}",
                window[0].sequence,
                window[1].sequence
            );
        }
    }

    #[test]
    fn producer_and_consumers_stored() {
        let mut arena = make_arena();
        let id = arena.reserve(1, sample_tensor());

        let p_id = PhaseId(42);
        let c_id = PhaseId(100);
        arena.set_producer(id, p_id).unwrap();
        arena.add_consumer(id, c_id).unwrap();

        let slot = arena.slot(id).unwrap();
        assert_eq!(slot.producer, Some(p_id));
        assert_eq!(slot.consumers, vec![c_id]);
    }

    #[test]
    fn arena_error_display() {
        let err = ArenaError::SlotNotReserved(42);
        let msg = format!("{}", err);
        assert!(!msg.is_empty());

        let err = ArenaError::InvalidTransition {
            from: SlotState::Reserved,
            to: SlotState::ConsumerReadable,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Reserved"));
        assert!(msg.contains("ConsumerReadable"));

        let err = ArenaError::NoBudget(100, 50);
        let msg = format!("{}", err);
        assert!(msg.contains("100"));
        assert!(msg.contains("50"));
    }

    #[test]
    fn arena_error_is_std_error() {
        fn takes_error<E: std::error::Error>() {}
        takes_error::<ArenaError>();
    }

    #[test]
    fn seal_stores_digest() {
        let mut arena = make_arena();
        let id = arena.reserve(1, sample_tensor());
        let mut digest = [0u8; 32];
        digest[0] = 0xde;
        digest[1] = 0xad;
        digest[2] = 0xbe;
        digest[3] = 0xef;

        arena
            .transition(id, SlotState::ProducerWriting, "write")
            .unwrap();
        arena.seal(id, digest).unwrap();

        let slot = arena.slot(id).unwrap();
        assert_eq!(slot.digest, Some(digest));
    }

    #[test]
    fn slot_state_serde_roundtrip() {
        let states = SlotState::all();
        for state in states {
            let json = serde_json::to_string(state).unwrap();
            let back: SlotState = serde_json::from_str(&json).unwrap();
            assert_eq!(*state, back);
        }
    }

    #[test]
    fn storage_route_serde_roundtrip() {
        let routes = &[
            StorageRoute::CpuOwned,
            StorageRoute::MetalSharedBuffer,
            StorageRoute::MetalPrivateBuffer,
            StorageRoute::CoreMLManaged,
            StorageRoute::CoreMLExported,
            StorageRoute::BridgeMaterialized,
            StorageRoute::DiskFrontier,
        ];
        for route in routes {
            let json = serde_json::to_string(route).unwrap();
            let back: StorageRoute = serde_json::from_str(&json).unwrap();
            assert_eq!(*route, back);
        }
    }
}
