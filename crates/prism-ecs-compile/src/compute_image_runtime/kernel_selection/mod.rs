//! Runtime kernel selection — pure data types and pure algorithms for
//! kernel variant selection, candidate benchmark evidence, and proof
//! seals.
//!
//! This module owns the data-only types absorbed from the engine's
//! `compute-core/src/ecs/compute_image/kernel_selection/` directory
//! on 2026-07-27. The engine-coupled implementations (those that
//! touch Metal/MLX/ANE runtime plumbing) stay at
//! `compute-core/src/ecs/legacy_compute_image_runtime/kernel_selection/`.

pub mod evidence;
pub mod proof_seal;
pub mod selection;
