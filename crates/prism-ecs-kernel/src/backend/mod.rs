//! Backend executors and adapters — the kernel-side hardware FFI surface.
//!
//! This module is the constitutional home for the per-backend
//! executors (Metal, ANE, Accelerate, CPU, legacy) and the
//! per-backend state records (memory pools, weight residency,
//! activation arenas, etc.). The runtime scheduling systems
//! produce typed, non-authoritative completion values; the runtime
//! reconciliation system stages the resulting state transition
//! through `ConstitutionalWorldTxn`.
//!
//! # Migration status
//!
//! Per the inventory v2.1 steps 36-50, the engine's adapter files
//! move here. As of 2026-07-27, the per-backend submodules are
//! scaffolded with placeholder executors; the full implementations
//! arrive when the engine's heterogeneous_executor splits in step 36.

pub mod accelerate;
pub mod ane;
pub mod cpu;
pub mod dispatcher;
pub mod legacy;
pub mod metal;
pub mod lane_executor_registry;
