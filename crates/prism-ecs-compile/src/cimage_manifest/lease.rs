//! Segment / tensor lease state machine — the runtime view of which bytes are
//! resident and in what storage backend.
//!
//! This module owns the constitutional authority for the lease state
//! machine ([`LeaseState`], [`StorageBackend`], [`SegmentLease`],
//! [`TensorLease`]). The lease is the runtime's typed contract for
//! "this segment is open / bound / active / retiring / released in
//! backend X." It is the read-side counterpart to the manifest's
//! per-segment `Segment` type.
//!
//! The module does **not** own the manifest itself (see
//! [`super::header`]), the per-tensor table (see [`super::types`]),
//! or the kernel dispatch recipes (see [`super::kernel`]).

use serde::{Deserialize, Serialize};

// ── Copy classification ───────────────────────────────────────────────────

/// How tensor bytes were moved from storage into the runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CopyClassification {
    /// Direct mmap view, no application copy. MLX may still copy internally.
    MappedNoCopy,
    /// Copied from mmap into an application-side buffer before MLX construction.
    CopiedFallback,
    /// MLX created a contiguous temporary (reshape, transpose, dtype cast, repeat).
    MaterializedContiguous,
    /// BF16 -> F32 or other dtype promotion.
    MaterializedDtypeConversion,
    /// K/V physically repeated for grouped-query attention.
    MaterializedRepeat,
}

/// Which storage backend owns the segment bytes.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum StorageBackend {
    Copied,
    MappedNoCopy,
}

// ── Lease state machine ───────────────────────────────────────────────────

/// Five-state lease lifecycle.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum LeaseState {
    /// Lease handle exists but no resources are reserved.
    Opened,
    /// Resources reserved; segment loaded into backend memory.
    Bound,
    /// Runtime is actively reading tensors from the lease.
    Active,
    /// Lease is in the process of being torn down.
    Retiring,
    /// Resources released; lease is stale.
    Released,
}

impl LeaseState {
    /// Return true if the state is terminal (no further transitions).
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Released)
    }

    /// Return true if a transition from `self` to `next` is permitted by
    /// the canonical state machine:
    ///
    /// Opened → Bound → Active → Retiring → Released.
    pub const fn can_transition_to(self, next: Self) -> bool {
        use LeaseState::*;
        match (self, next) {
            (Opened, Bound) => true,
            (Bound, Active) => true,
            (Active, Retiring) => true,
            (Retiring, Released) => true,
            _ => false,
        }
    }
}

/// Segment-level lease record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentLease {
    pub segment_id: String,
    pub filename: String,
    pub backend: StorageBackend,
    pub state: LeaseState,
    pub tensor_handles: Vec<u64>,
    pub byte_size: u64,
}

/// Tensor-level lease record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorLease {
    pub name: String,
    pub handle: u64,
    pub segment_id: String,
    pub state: LeaseState,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lease_state_transitions_are_sequential_only() {
        use LeaseState::*;
        // Forward transitions are permitted.
        assert!(Opened.can_transition_to(Bound));
        assert!(Bound.can_transition_to(Active));
        assert!(Active.can_transition_to(Retiring));
        assert!(Retiring.can_transition_to(Released));

        // No skipping a state.
        assert!(!Opened.can_transition_to(Active));
        assert!(!Bound.can_transition_to(Retiring));
        assert!(!Active.can_transition_to(Released));

        // No going backwards.
        assert!(!Bound.can_transition_to(Opened));
        assert!(!Active.can_transition_to(Bound));
        assert!(!Retiring.can_transition_to(Active));
        assert!(!Released.can_transition_to(Retiring));
    }

    #[test]
    fn lease_state_only_released_is_terminal() {
        use LeaseState::*;
        assert!(!Opened.is_terminal());
        assert!(!Bound.is_terminal());
        assert!(!Active.is_terminal());
        assert!(!Retiring.is_terminal());
        assert!(Released.is_terminal());
    }

    #[test]
    fn copy_classification_round_trip_preserves_variant() {
        let original = CopyClassification::MaterializedDtypeConversion;
        let json = serde_json::to_string(&original).unwrap();
        let parsed: CopyClassification = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, original);
    }

    #[test]
    fn storage_backend_round_trip_preserves_variant() {
        let original = StorageBackend::MappedNoCopy;
        let json = serde_json::to_string(&original).unwrap();
        let parsed: StorageBackend = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, original);
    }

    #[test]
    fn segment_lease_default_state_is_opened() {
        // The default state of a freshly-constructed segment lease is
        // `Opened` — the lease handle exists but no resources are
        // reserved yet.
        let lease = SegmentLease {
            segment_id: "layer_0".into(),
            filename: "segment_001.bin".into(),
            backend: StorageBackend::Copied,
            state: LeaseState::Opened,
            tensor_handles: Vec::new(),
            byte_size: 1024,
        };
        assert_eq!(lease.state, LeaseState::Opened);
        assert_eq!(lease.segment_id, "layer_0");
    }
}
