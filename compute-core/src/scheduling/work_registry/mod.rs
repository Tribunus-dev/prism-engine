//! In-flight work tracking and state machine.
//!
//! Maintains a registry of all in-flight work items across all execution lanes,
//! managing the work state machine and providing indexed access by lane, session,
//! and phase.  The [`WorkRegistry`] is the single source of truth for the lifecycle
//! of every submitted work item.

pub mod registry;
pub mod scheduling;

pub use registry::*;
pub use scheduling::*;
