//! In-flight work tracking and state machine.
//!
//! Shared types from the legacy work registry.  The [`WorkRegistry`] write path
//! has been replaced by the constitutional [`WorkLifecycleBridge`]; the remaining
//! types ([`WorkKey`], [`WorkStatus`]) are kept for compat in downstream consumers
//! such as [`completion_bridge`] and [`receipt`].

pub mod registry;
pub mod scheduling;

pub use registry::WorkStatus;
pub use scheduling::WorkKey;
