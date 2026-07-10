//! Shared Metal-event coordination metadata for IOSurface-backed execution.
//!
//! This module carries the execution contract that lets Core ML and Metal
//! rendezvous around the same IOSurface-resident tensors without copies.

/// The direction of a shared-event dependency for one executable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SharedEventAccess {
    /// The executable must wait until the event reaches `value`.
    Wait,
    /// The executable signals the event after it publishes its outputs.
    Signal,
}

/// One executable's participation in a shared-event contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedEventBinding {
    /// Logical event identifier, stable across install/runtime.
    pub event_id: String,
    /// IOSurface slot protected by this event.
    pub slot_id: u32,
    /// Whether the executable waits or signals.
    pub access: SharedEventAccess,
    /// Fence value used for the wait/signal operation.
    pub value: u64,
}

/// Compile-time/runtime description of one shared-event handoff.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedEventContract {
    /// Logical event identifier, used to look up the live Metal event.
    pub event_id: String,
    /// IOSurface slot shared between producer and consumer.
    pub slot_id: u32,
    /// Producer artifact identifier.
    pub producer_artifact_id: String,
    /// Consumer artifact identifier.
    pub consumer_artifact_id: String,
    /// Signal value emitted by the producer.
    pub signal_value: u64,
    /// Value the consumer must wait for before reading the slot.
    pub wait_value: u64,
}
