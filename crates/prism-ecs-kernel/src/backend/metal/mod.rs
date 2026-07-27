//! Metal backend (constitutional home).
//!
//! The kernel-side adapter for the Metal/GPU lane. The Metal
//! backend owns the command queue, pipeline state, and per-lane
//! resource cache. It exposes the `LaneExecutor` contract (or its
//! per-backend equivalent) and produces typed, non-authoritative
//! completion values.
//!
//! # Migration status
//!
//! Per the inventory v2.1, the engine's `metal_lane_executor.rs`,
//! `metal_decoder.rs`, `weight_residency.rs`, `activation_arena.rs`,
//! `activation_binding.rs`, `activation_transaction.rs` (FFI half),
//! and `memory_pool.rs` move here. As of 2026-07-27, the
//! submodules are scaffolded with placeholder types; the full
//! implementations arrive when the engine's heterogeneous_executor
//! splits in step 36.

pub mod activation_arena;
pub mod activation_binding;
pub mod activation_transaction;
pub mod lane_executor;
pub mod memory_pool;
pub mod weight_residency;
pub mod decoder;
