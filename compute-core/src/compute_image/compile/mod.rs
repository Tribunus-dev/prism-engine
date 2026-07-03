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
pub mod draft_loader;
#[cfg(feature = "tensix")]
pub mod tensix;
pub mod execution_graph;
pub mod ternary_pipeline;
pub mod kernel_types;
#[cfg(feature = "prism-backend")]
pub mod kernel_dispatch;
#[cfg(feature = "prism-backend")]
pub mod kernel_registry;
#[cfg(feature = "prism-backend")]
pub mod validation_matrix;

pub use source::*;
pub use quantize::*;
pub(crate) use emit::*;
pub use pipeline::*;
pub use download::*;
