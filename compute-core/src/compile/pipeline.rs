//! Re-exports from the main compute-image compilation pipeline.
//!
//! Makes selected functions from `crate::compute_image::compile::pipeline`
//! accessible under `crate::compile::pipeline`.

/// Re-export archive_ane_modelc from the compute-image compile pipeline.
///
/// The source function is re-exported at `crate::compute_image::compile::archive_ane_modelc`
/// via `pub use pipeline::*;` in compute_image/compile/mod.rs.
pub(crate) use crate::compute_image::compile::archive_ane_modelc;
