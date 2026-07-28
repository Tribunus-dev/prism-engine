//! Re-export of the constitutional scheduling subsystem.
//!
//! The constitutional [`crate::scheduling`] module is the canonical home for
//! runtime scheduling state, deterministic systems, evidence, and metrics.
//! This module re-exports it under the `runtime::scheduling` path so the
//! canonical "runtime" surface is grep-able from a single import path.
//!
//! Migration map: `compute-core/src/ecs/runtime::scheduling::*` (engine
//! legacy) → `prism_ecs_runtime::runtime::scheduling::*` (constitutional).
//!
//! Note: the engine's `scheduling/` is a self-contained schedule compiler
//! that depends on engine-internal `World` / `Entity` types. The
//! engine-coupled schedule compiler lives in the engine's
//! `legacy_runtime::scheduling` and is the only place where the
//! engine's `World` / `Entity` types are touched. The constitutional
//! scheduling types do not import those engine types and are the
//! canonical home for the runtime schedule.

pub use crate::scheduling::*;
