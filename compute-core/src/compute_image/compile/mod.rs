//! ComputeImage compilation pipeline — source loading, quantization,
//! sequential/differential compilation, diagnostics, and publishing.

pub mod source;
mod quantize;
mod emit;
mod pipeline;
mod download;

#[cfg(all(
    target_os = "macos",
    any(feature = "mlx-backend", feature = "prism-backend")
))]
pub mod coreml;
pub mod hardware;
pub mod portfolio;
pub mod ternary;
pub mod int4_pack;
#[cfg(feature = "tensix")]
pub mod tensix;

pub use source::*;
pub use quantize::*;
pub(crate) use emit::*;
pub use pipeline::*;
pub use download::*;
