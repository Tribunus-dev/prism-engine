//! Scheduling subsystem — constitutional home for runtime scheduling state, systems, and metrics.
//!
//! This module is the migration target for `compute-core/src/ecs/scheduling/`.
//! It owns the authoritative scheduling state (lanes, leases, queues, capacity,
//! phase state), the deterministic systems that advance scheduling decisions,
//! and the advisory metrics produced by those systems. The kernel backends
//! live in `prism-ecs-kernel::backend::*` and operate on this state only
//! through typed, non-authoritative completion values that the runtime
//! reconciliation system stages through `ConstitutionalWorldTxn`.
//!
//! # Submodules
//!
//! - [`state`] — authoritative scheduling data (Components, C bucket).
//!   Mutations are staged through `ConstitutionalWorldTxn`.
//! - [`systems`] — deterministic scheduling behavior (Systems, S bucket).
//!   Read staged + committed state; stage their own mutations through
//!   `ConstitutionalWorldTxn`.
//! - [`metrics`] — advisory metrics (M bucket). Rebuildable projection;
//!   not evidence.
//!
//! # Migration status
//!
//! Phase 0 inventory: `changelogs/2026-07-27-scheduling-migration-inventory.md`.
//! First code slice: `state::lane_capacity` (`lane_capacity.rs` moved from
//! the engine on 2026-07-27, with `ExecutionLane` precursor in
//! `prism-ecs-kernel::execution_lane`).

pub mod state;
