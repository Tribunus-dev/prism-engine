//! Re-export shim — `state::phase` is the canonical home.
//!
//! Per inventory v2.1 step 7, the engine's `phase_engine_state.rs` is
//! merged into `state::phase`. This file exists as a transitional
//! re-export so the rest of the constitutional code (and the engine
//! file's callers, when they migrate) can continue to refer to
//! `state::phase_engine_state` for one step. The re-exports here
//! will be deleted when the engine file's callers are updated.

pub use super::phase::{PhaseId, RuntimeWorkItemHandle};
