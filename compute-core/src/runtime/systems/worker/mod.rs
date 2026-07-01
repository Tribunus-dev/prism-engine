//! Worker supervision systems (Slice 2).
//!
//! These systems implement request intake, worker dispatch, event drain,
//! liveness watchdog, and the legacy bridge shim.

pub mod ingress;
pub mod stream_observer;
pub mod event_drain;
pub mod watchdog;
pub mod bridge;
pub mod spawn;

pub use stream_observer::StreamObservationSystem;
