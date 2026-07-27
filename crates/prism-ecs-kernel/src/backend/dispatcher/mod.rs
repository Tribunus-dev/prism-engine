//! Heterogeneous dispatcher (constitutional home).
//!
//! Per the inventory v2.1 row 17, the engine's
//! `heterogeneous_executor.rs` is split: runtime half is
//! `prism-ecs-runtime::scheduling::systems::heterogeneous_orchestration`,
//! kernel half is here.
//!
//! The dispatcher is the kernel-side coordinator of the per-lane
//! executors. It receives a typed, non-authoritative dispatch
//! request from the runtime, routes it to the appropriate backend,
//! and produces a typed, non-authoritative completion value.

pub mod heterogeneous;
