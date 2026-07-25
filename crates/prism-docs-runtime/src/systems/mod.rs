//! Systems — pure functions that query the world and stage events.
//!
//! Each system is a single authority. Systems are not allowed to
//! touch the DOM; they may only read components, stage component
//! mutations, and emit typed events.

pub mod chapter_presentation_system;
pub mod claim_validation_system;
pub mod nav_projection_system;
pub mod render_coordinator_system;
