//! ANE backend (constitutional home).
//!
//! The kernel-side adapter for the Apple Neural Engine. The ANE
//! backend owns the Core ML model cache, the lane executor, and
//! the agent bridge. It produces typed, non-authoritative
//! completion values.
//!
//! # Migration status
//!
//! Per the inventory v2.1, the engine's `ane_lane_executor.rs`,
//! `ane_artifact_cache.rs`, and the kernel half of
//! `agent_bridge.rs` move here.

pub mod agent_bridge;
pub mod artifact_cache;
pub mod lane_executor;
