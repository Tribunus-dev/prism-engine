//! Scheduling systems (S bucket) — deterministic behavior that transforms state.
//!
//! Every system in this module reads staged + committed state, computes
//! its decision, and stages its own mutations through
//! `ConstitutionalWorldTxn`. A system is non-authoritative: its output
//! is visible only after the transaction commits.
//!
//! # Migration status
//!
//! Per the inventory v2.1 (steps 15-33), the systems move here from
//! the engine's `compute-core/src/ecs/scheduling/` directory. As of
//! 2026-07-27, the systems are being added one slice at a time; the
//! `mod` declarations in this file are the migration contract.

pub mod compilation_job_bridge;
pub mod execution_lease_bridge;
pub mod fallback;
pub mod phase_readiness;
pub mod work_lifecycle_bridge;
pub use super::state::phase::EmittedPhasePlaceholder;
