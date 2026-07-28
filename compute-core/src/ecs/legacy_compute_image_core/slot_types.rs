#![cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
//! Pure data types for IOSurface arena slot state — no system dependencies.
//!
//! These types are used by both the macOS-only IOSurface arena code and by
//! unconditional cross-platform modules (activation_abi, phase_ir, etc.).
//! Separating them avoids macOS-only import chains on iOS.

use serde::{Deserialize, Serialize};

use prism_ecs_kernel::backend::placement::ExecutionLane;

/// Failure reason for poisoned slots.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SlotFailureReason {
    LayoutMismatch { expected: String, actual: String },
    CoreAiPredictionFailed(String),
    MetalDispatchFailed(String),
    Timeout { deadline_ns: u64 },
    NumericalGuardFailed(String),
    AllocationPrevented,
    InternalError(String),
}

/// Slot state with explicit ownership semantics.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SlotState {
    Free,
    Reserved {
        epoch: u64,
        producer: ExecutionLane,
    },
    Writing {
        epoch: u64,
        producer: ExecutionLane,
    },
    Ready {
        epoch: u64,
        producer: ExecutionLane,
    },
    Reading {
        epoch: u64,
        consumer: ExecutionLane,
    },
    Retired {
        epoch: u64,
    },
    Poisoned {
        epoch: u64,
        reason: SlotFailureReason,
    },
}

/// Reuse policy for a slot within the IOSurface arena.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SlotReuseClass {
    Exclusive,
    SharedReadOnly,
    RingReuse { ring_depth: u8 },
}
