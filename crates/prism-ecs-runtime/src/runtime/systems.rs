//! Re-export of the constitutional engine-system surface.
//!
//! The constitutional [`crate::systems`] module is the canonical home for
//! engine-system data types (archive, backend dispatch, residency,
//! backpressure tick, scratch planning, etc.). This module re-exports it
//! under the `runtime::systems` path so the canonical "runtime" surface
//! is grep-able from a single import path.
//!
//! Migration map: engine callers of the legacy
//! `ecs::runtime::systems::*` use the constitutional `runtime::systems::*`
//! types directly. The engine-coupled adapter for the `CompilerSystem`
//! trait lives in the engine's `system_adapters` module.

pub use crate::systems::*;
