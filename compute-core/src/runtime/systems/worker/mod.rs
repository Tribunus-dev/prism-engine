//! Worker supervision systems (Slice 2).
//!
//! These systems implement request intake, worker dispatch, event drain,
//! liveness watchdog, and the legacy bridge shim.

pub mod bridge;
pub mod event_drain;
pub mod ingress;
pub mod spawn;
pub mod stream_observer;
pub mod watchdog;

pub use stream_observer::StreamObservationSystem;
