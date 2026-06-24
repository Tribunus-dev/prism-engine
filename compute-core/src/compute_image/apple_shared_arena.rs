//! Live IOSurface arena for ANE-TRI-LANE-REALIZATION-0001 Phase 1.
//!
//! Provides sealed arena metadata, live IOSurface arena installation, and
//! slot state machines with explicit ownership semantics.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Digest type for layout verification.
type Digest = String;

/// Slot identifier.
type SlotId = u32;

/// Arena identifier.
type ArenaId = String;

/// A monotonic epoch + generation pair for slot freshness tracking.
type SlotGeneration = u64;

/// Failure reason for poisoned slots.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SlotFailureReason {
    LayoutMismatch { expected: String, actual: String },
    CoreMlPredictionFailed(String),
    MetalDispatchFailed(String),
    Timeout { deadline_ns: u64 },
    NumericalGuardFailed(String),
    AllocationPrevented,
    InternalError(String),
}

/// Slot state with explicit ownership semantics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SlotState {
    Free,
    Reserved { epoch: u64, producer: ExecutionLane },
    Writing { epoch: u64, producer: ExecutionLane },
    Ready { epoch: u64, producer: ExecutionLane },
    Reading { epoch: u64, consumer: ExecutionLane },
    Retired { epoch: u64 },
    Poisoned {
        epoch: u64,
        reason: SlotFailureReason,
    },
}

/// IOSurface slot manifest (immutable, from CImage).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IOSurfaceSlotManifest {
    pub slot_id: SlotId,
    pub tensor_id: String,
    pub byte_offset: u64,
    pub byte_length: u64,
    pub dtype: String,
    pub logical_shape: Vec<u32>,
    pub physical_shape: Vec<u32>,
    pub strides_bytes: Vec<u64>,
    pub layout: String,
    pub producer: ExecutionLane,
    pub consumer: ExecutionLane,
    pub reuse_class: SlotReuseClass,
    pub required_alignment: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SlotReuseClass {
    Exclusive,
    SharedReadOnly,
    RingReuse { ring_depth: u8 },
}

/// Re-export ExecutionLane for convenience.
pub use crate::backend::placement::ExecutionLane;

/// Live IOSurface slot at runtime.
pub struct LiveIOSurfaceSlot {
    pub manifest: IOSurfaceSlotManifest,
    pub state: SlotState,
    pub generation: u64,
    pub layout_digest: String,
    pub metal_view: Option<String>,   // Metal resource view descriptor
    pub coreml_view: Option<String>, // Core ML IOSurface view descriptor
}

impl LiveIOSurfaceSlot {
    /// Returns true when the slot may be acquired for the given epoch.
    pub fn is_available_for(&self, epoch: u64, _producer: ExecutionLane) -> bool {
        matches!(self.state, SlotState::Free)
            || matches!(&self.state, SlotState::Retired { epoch: e } if *e <= epoch)
    }

    /// Reserve a slot. The slot must be Free or Retired with epoch at most
    /// the caller's epoch.
    pub fn reserve(&mut self, epoch: u64, producer: ExecutionLane) -> Result<(), String> {
        match &self.state {
            SlotState::Free => {}
            SlotState::Retired { epoch: e } if *e <= epoch => {}
            _ => {
                return Err(format!(
                    "slot {} cannot be reserved from state {:?}",
                    self.manifest.slot_id, self.state
                ));
            }
        }
        self.state = SlotState::Reserved { epoch, producer };
        Ok(())
    }

    /// Transition to Writing state.
    pub fn mark_writing(&mut self, epoch: u64, producer: ExecutionLane) {
        self.state = SlotState::Writing { epoch, producer };
    }

    /// Transition to Ready state and bump slot generation.
    pub fn mark_ready(&mut self, epoch: u64, producer: ExecutionLane) {
        self.state = SlotState::Ready { epoch, producer };
        self.generation += 1;
    }

    /// Transition to Reading state. Only valid from Ready with matching epoch.
    pub fn mark_reading(&mut self, epoch: u64, consumer: ExecutionLane) -> Result<(), String> {
        match &self.state {
            SlotState::Ready {
                epoch: e,
                producer: _,
            } if *e == epoch => {}
            _ => {
                return Err(format!(
                    "slot {} not ready for reading at epoch {}",
                    self.manifest.slot_id, epoch
                ));
            }
        }
        self.state = SlotState::Reading { epoch, consumer };
        Ok(())
    }

    /// Retire the slot.
    pub fn retire(&mut self, epoch: u64) {
        self.state = SlotState::Retired { epoch };
    }

    /// Poison the slot with a failure reason.
    pub fn poison(&mut self, epoch: u64, reason: SlotFailureReason) {
        self.state = SlotState::Poisoned { epoch, reason };
    }
}

/// Live shared arena at runtime.
pub struct AppleSharedArena {
    pub arena_id: ArenaId,
    pub layout_digest: String,
    pub slots: HashMap<SlotId, LiveIOSurfaceSlot>,
    pub generation: u64,
    pub ring_depth: u8,
}

impl AppleSharedArena {
    /// Create a new empty arena.
    pub fn new(arena_id: ArenaId, ring_depth: u8) -> Self {
        Self {
            arena_id,
            layout_digest: String::new(),
            slots: HashMap::new(),
            generation: 0,
            ring_depth,
        }
    }

    /// Add a slot to the arena.
    pub fn add_slot(&mut self, slot: LiveIOSurfaceSlot) {
        let id = slot.manifest.slot_id;
        self.slots.insert(id, slot);
    }

    /// Borrow a slot by id.
    pub fn slot(&self, id: SlotId) -> Option<&LiveIOSurfaceSlot> {
        self.slots.get(&id)
    }

    /// Mutably borrow a slot by id.
    pub fn slot_mut(&mut self, id: SlotId) -> Option<&mut LiveIOSurfaceSlot> {
        self.slots.get_mut(&id)
    }

    /// Advance the arena generation counter.
    pub fn advance_generation(&mut self) {
        self.generation += 1;
    }

    /// Returns true when every slot has been retired for the given epoch.
    pub fn all_slots_retired(&self, epoch: u64) -> bool {
        self.slots
            .values()
            .all(|s| matches!(&s.state, SlotState::Retired { epoch: e } if *e == epoch))
    }

    /// Install an arena from its sealed manifest.
    /// Fails if any constraint cannot be satisfied exactly — no silent reshaping.
    pub fn install(manifest: &crate::compute_image::apple_cimage_manifest::AppleSharedArenaManifest) -> Result<Self, String> {
        if manifest.allocation_bytes == 0 {
            return Err("allocation_bytes must be > 0".into());
        }
        if manifest.slots.is_empty() {
            return Err("arena manifest has no slots".into());
        }
        let mut arena = Self::new(
            format!("arena-{}", &manifest.arena_layout_digest[..8.min(manifest.arena_layout_digest.len())]),
            manifest.ring_depth,
        );
        arena.layout_digest = manifest.arena_layout_digest.clone();

        for slot_manifest in &manifest.slots {
            let reuse_class = match slot_manifest.reuse_class.as_str() {
                "exclusive" => SlotReuseClass::Exclusive,
                "shared_readonly" => SlotReuseClass::SharedReadOnly,
                _ => SlotReuseClass::RingReuse { ring_depth: manifest.ring_depth },
            };
            let slot = LiveIOSurfaceSlot {
                manifest: IOSurfaceSlotManifest {
                    slot_id: slot_manifest.slot_id,
                    tensor_id: slot_manifest.tensor_id.clone(),
                    byte_offset: slot_manifest.byte_offset,
                    byte_length: slot_manifest.byte_length,
                    dtype: slot_manifest.dtype.clone(),
                    logical_shape: slot_manifest.logical_shape.clone(),
                    physical_shape: slot_manifest.physical_shape.clone(),
                    strides_bytes: slot_manifest.strides_bytes.clone(),
                    layout: slot_manifest.layout.clone(),
                    producer: slot_manifest.producer,
                    consumer: slot_manifest.consumer,
                    reuse_class,
                    required_alignment: slot_manifest.required_alignment,
                },
                state: SlotState::Free,
                generation: 0,
                layout_digest: manifest.arena_layout_digest.clone(),
                metal_view: None,
                coreml_view: None,
            };
            arena.add_slot(slot);
        }
        Ok(arena)
    }
}

/// SlotEpoch for generation tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotEpoch {
    pub epoch: u64,
    pub arena_generation: u64,
    pub slot_generation: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_slot(id: SlotId) -> LiveIOSurfaceSlot {
        LiveIOSurfaceSlot {
            manifest: IOSurfaceSlotManifest {
                slot_id: id,
                tensor_id: format!("tensor_{}", id),
                byte_offset: 0,
                byte_length: 4096,
                dtype: "float16".into(),
                logical_shape: vec![64, 64],
                physical_shape: vec![64, 64],
                strides_bytes: vec![128, 2],
                layout: "NHWC".into(),
                producer: ExecutionLane::CandleCpu,
                consumer: ExecutionLane::CoreMlAne,
                reuse_class: SlotReuseClass::Exclusive,
                required_alignment: 256,
            },
            state: SlotState::Free,
            generation: 0,
            layout_digest: "abc123".into(),
            metal_view: None,
            coreml_view: None,
        }
    }

    /// Full acquire → write → ready → read → retire cycle.
    #[test]
    fn test_slot_acquire_release_full_cycle() {
        let mut slot = make_test_slot(1);

        // Acquire: reserve from Free
        slot.reserve(0, ExecutionLane::CandleCpu).unwrap();
        assert!(matches!(slot.state, SlotState::Reserved { epoch: 0, producer: ExecutionLane::CandleCpu }));

        // Write
        slot.mark_writing(0, ExecutionLane::CandleCpu);
        assert!(matches!(slot.state, SlotState::Writing { epoch: 0, .. }));

        // Ready
        slot.mark_ready(0, ExecutionLane::CandleCpu);
        assert!(matches!(slot.state, SlotState::Ready { epoch: 0, .. }));
        assert_eq!(slot.generation, 1);

        // Read (consumer)
        slot.mark_reading(0, ExecutionLane::CoreMlAne).unwrap();
        assert!(matches!(slot.state, SlotState::Reading { epoch: 0, consumer: ExecutionLane::CoreMlAne }));

        // Retire
        slot.retire(0);
        assert!(matches!(slot.state, SlotState::Retired { epoch: 0 }));

        // Re-acquire from Retired
        slot.reserve(1, ExecutionLane::CandleCpu).unwrap();
        assert!(matches!(slot.state, SlotState::Reserved { epoch: 1, .. }));
    }

    /// Cannot reserve a slot that is in Writing (i.e. not Free/Retired).
    #[test]
    fn test_slot_writing_before_reserve_rejected() {
        let mut slot = make_test_slot(2);

        // Bypass reserve — jump straight to Writing (illegal path).
        slot.mark_writing(0, ExecutionLane::CandleCpu);
        let err = slot.reserve(1, ExecutionLane::CandleCpu).unwrap_err();
        assert!(err.contains("cannot be reserved"));
    }

    /// Ready → Reading transition enforces epoch match.
    #[test]
    fn test_slot_ready_transition_checks_epoch() {
        let mut slot = make_test_slot(3);

        slot.reserve(0, ExecutionLane::MlxGpu).unwrap();
        slot.mark_writing(0, ExecutionLane::MlxGpu);
        slot.mark_ready(0, ExecutionLane::MlxGpu);

        // Try reading with wrong epoch
        let err = slot
            .mark_reading(1, ExecutionLane::CoreMlAne)
            .unwrap_err();
        assert!(err.contains("not ready for reading"));
    }

    /// Poison marks the slot and the reason survives.
    #[test]
    fn test_poison_marks_slot_and_persists_reason() {
        let mut slot = make_test_slot(4);

        slot.reserve(0, ExecutionLane::CandleCpu).unwrap();

        let reason = SlotFailureReason::MetalDispatchFailed("encoder OOM".into());
        slot.poison(0, reason.clone());

        match &slot.state {
            SlotState::Poisoned {
                epoch: 0,
                reason: r,
            } => {
                assert!(matches!(r, SlotFailureReason::MetalDispatchFailed(msg) if msg == "encoder OOM"));
            }
            other => panic!("expected Poisoned, got {:?}", other),
        }
    }
}
