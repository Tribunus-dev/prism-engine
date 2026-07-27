//! Phase engine state (placeholder for the eventual `state::phase`).
//!
//! Per inventory v2.1 step 7, `phase_engine_state.rs` merges into
//! `state::phase.rs`. This file is a temporary home that ships the
//! types the rest of the state migration depends on. Step 7 will
//! move these types into `state::phase` and delete this file.
//!
//! # Migration provenance
//!
//! The legacy home was `compute-core/src/ecs/scheduling/phase_engine_state.rs`.
//! The engine file is the legacy duplicate; step 58 deletes it.

/// Placeholder for `compute-core::ecs::scheduling::phase_engine_state::RuntimeWorkItemHandle`.
/// Replaced when the engine's `phase_engine_state.rs` types move into
/// `state::phase` in step 7.
#[derive(Debug, Clone, Default)]
pub struct RuntimeWorkItemHandle {
    /// Opaque work-item identifier.
    pub id: u64,
}
