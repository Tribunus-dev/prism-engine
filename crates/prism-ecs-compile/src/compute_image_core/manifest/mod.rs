//! Constitutional `manifest` surface.
//!
//! Re-exports data-only manifest types and shape extension
//! helpers. The engine-coupled `Manifest` struct (depends on
//! engine-internal config types) and the `runtime.rs` (depends on
//! `mlx_rs`) stay at
//! `compute-core/src/ecs/legacy_compute_image_core/manifest/`.
//!
//! # Authority
//!
//! This module owns the canonical data types for tensor entries,
//! segments, aliases, residency plans, and quantization metadata
//! that flow through a ComputeImage manifest. The full `Manifest`
//! struct (the one that ties config + segments + tensors together)
//! lives in the engine's legacy path.

pub mod shape_ext;
pub mod types;

pub use shape_ext::*;
pub use types::*;
