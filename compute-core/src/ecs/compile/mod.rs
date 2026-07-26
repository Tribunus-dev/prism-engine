//! Audio compilation pipeline — standalone cimage builder for audio models.
//!
//! Sub-modules:
//! - `audio`: The audio model compilation entry point.
//! - `pipeline`: Re-exports from the compute-image pipeline.
//! - `vision`: The vision model compilation entry point.

#[cfg(all(feature = "mlx-backend", feature = "prism-backend"))]
pub mod audio;
#[cfg(all(feature = "mlx-backend", feature = "prism-backend"))]
pub mod pipeline;
#[cfg(all(feature = "mlx-backend", feature = "prism-backend"))]
pub mod vision;
