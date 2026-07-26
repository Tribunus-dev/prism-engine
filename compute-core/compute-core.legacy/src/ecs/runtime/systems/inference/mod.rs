//! Inference ECS systems — per-step inference execution, session management,
//! and telemetry observation.
//!
//! Moved from `profiled_executor.rs` as part of the Parallel Monolith Purge.
//! All public types remain accessible via `crate::profiled_executor::*`.

pub mod session;
pub mod step;
pub mod telemetry;

// Re-export the session types so they can be reached through the
// runtime::systems::inference path, while the legacy profiled_executor
// shim maintains backward compatibility.
pub use session::*;
