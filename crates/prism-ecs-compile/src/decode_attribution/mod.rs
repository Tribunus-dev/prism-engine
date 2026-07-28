//! `prism_ecs_compile::decode_attribution` — decode-time evidence and
//! attribution surface.
//!
//! This module owns the canonical authority for the engine's
//! `ecs::decode_attribution/` subsystem: structured decode-time
//! receipts (per-layer / per-step evidence), environment capture,
//! timing calibration, and the pure-Rust conformance metrics
//! comparing backend outputs against the reference evaluator.
//!
//! Higher-leverage, engine-coupled adapter code (Core ML harness,
//! MLX adapter, compute plan inspection, defect clustering, KV-cache
//! phase contracts) is engine-internal at
//! `compute-core/src/ecs/legacy_decode_attribution/` because it
//! depends on engine FFI bridges and the per-backend ANE/CoreAI/MLX
//! executor stack. This surface is the cross-platform, constitutional
//! home for the data types, statistics, and host-environment
//! capture.
//!
//! Submodules:
//! - [`artifact_hash`] — deterministic directory hashing
//! - [`backend_adapters`] — backend-agnostic conformance metrics
//! - [`breadcrumb`] — append-only fsynced breadcrumb writer
//! - [`compute_plan`] — optional MLComputePlan stub
//! - [`environment`] — host identity capture (chip, OS, toolchain)
//! - [`receipt`] — `DecodeAttributionReceipt` data structure
//! - [`shape_profiles`] — canonical shape profiles
//! - [`statistics`] — distribution statistics
//! - [`timer_calibration`] — timer overhead calibration
//!
//! # Authority
//!
//! The `decode_attribution` surface is the canonical home for
//! decode-time evidence types. The legacy engine surface at
//! `compute-core/src/ecs/legacy_decode_attribution/` re-exports
//! these types and hosts the engine-coupled adapter code.

pub mod artifact_hash;
pub mod backend_adapters;
pub mod breadcrumb;
pub mod compute_plan;
pub mod environment;
pub mod receipt;
pub mod shape_profiles;
pub mod statistics;
pub mod timer_calibration;
