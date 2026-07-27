//! Phase state (placeholder for the eventual merged phase state).
//!
//! Per inventory v2.1 step 7, this file will become the home of all
//! types from `phase_engine.rs` and `phase_engine_state.rs`. For now
//! it ships the `EmittedPhase` placeholder that other state files
//! depend on. Step 7 will move the full phase state here and delete
//! `state::phase_engine_state`.
//!
//! # Migration provenance
//!
//! The legacy home for the engine's phase types was
//! `compute-core/src/ecs/scheduling/phase_engine.rs` and
//! `phase_engine_state.rs`. The engine files are the legacy
//! duplicates; step 58 deletes them.

/// Placeholder for `compute-core::ecs::compute_image::phase_dag::EmittedPhase`.
/// Replaced when `phase_dag` migrates. For now it carries only the
/// minimum data `phase_invocation` and `ready_queue` need.
#[derive(Debug, Clone, Default)]
pub struct EmittedPhasePlaceholder {
    pub phase_id: String,
}

/// Re-export of the placeholder under the engine's name. Replaced
/// when `phase_dag` migrates.
pub type EmittedPhase = EmittedPhasePlaceholder;
