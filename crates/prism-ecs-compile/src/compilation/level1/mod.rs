//! Level 1 of the distill-compiler — Metal + Accelerate only (data-only).
//!
//! This module owns the canonical authority for the std-only
//! knowledge-distillation gate logic (token-by-token teacher/student
//! scoring, calibration stream handling, KD divergence and top-1
//! agreement). The Metal/Accelerate-coupled Level 1 submodules
//! (checkpoint, gates, reducer, scheduler, student, teacher) are
//! engine-internal implementation in the engine's
//! `compute-core/src/ecs/legacy_compilation/level1/` directory;
//! engine callers reach them via `crate::ecs::legacy_compilation::level1::*`.
//!
//! ## Migration status
//!
//! This module absorbed the engine's `compute-core/src/ecs/compilation/level1/kd_gate.rs`
//! (968 LOC) into the constitutional compile crate on 2026-07-27.
//! The remaining level1/* files (checkpoint, gates, reducer, scheduler,
//! student, teacher) depend on engine-internal types (Metal
//! device/compute_image/calibration/system_adapters) and remain
//! engine-side at `compute-core/src/ecs/legacy_compilation/level1/`.

pub mod kd_gate;
