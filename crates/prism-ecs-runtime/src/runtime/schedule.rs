//! Re-export of the constitutional schedule module.
//!
//! The constitutional [`crate::schedule`] module is the canonical home for
//! the runtime schedule, `System` trait, and command envelope types. This
//! module re-exports it under the `runtime::schedule` path so the canonical
//! "runtime" surface is grep-able from a single import path.
//!
//! Migration map: the engine's `ecs::runtime::scheduling::schedule::*`
//! (engine legacy) is the engine-coupled schedule compiler; the
//! constitutional `runtime::schedule` is the canonical surface.

pub use crate::schedule::*;
