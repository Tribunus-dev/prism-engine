//! Re-exports from the main compute-image compilation pipeline.
//!
//! Makes selected functions from `crate::ecs::legacy_compute_image_core::compile::pipeline`
//! accessible under `crate::ecs::compile::pipeline`.

/// Re-export archive_ane_modelc from the compute-image compile pipeline.
///
/// The source function is re-exported at `crate::ecs::legacy_compute_image_core::compile::archive_ane_modelc`
/// via `pub use pipeline::*;` in compute_image/compile/mod.rs.
pub(crate) use crate::ecs::legacy_compute_image_core::compile::archive_ane_modelc;
