//! Re-export of the constitutional ports surface.
//!
//! The constitutional [`crate::ports`] module is the canonical home for the
//! runtime port surface (dispatcher, lease coordinator, snapshot store,
//! evidence sink, command store, kernel clock, etc.). This module re-exports
//! it under the `runtime::ports` path so the canonical "runtime" surface
//! is grep-able from a single import path.
//!
//! Migration map: engine callers of the legacy
//! `ecs::runtime::*::work_dispatcher` / `lease_coordinator` etc. use the
//! constitutional `runtime::ports::*` types directly.

pub use crate::ports::*;
