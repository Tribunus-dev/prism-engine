//! ComputeImage compilation pipeline — source loading, quantization,
//! sequential/differential compilation, diagnostics, and publishing.

pub mod archive;
pub mod capability_registry;
mod download;
#[cfg(feature = "mlx-backend")] // research surface: MLX compile lane
mod emit;
#[cfg(feature = "mlx-backend")] // research surface: MLX compile lane
mod pipeline;
mod quantize;
pub mod source;

#[cfg(all(
    target_os = "macos",
    any(feature = "mlx-backend", feature = "prism-backend")
))]
pub mod coreai;
pub mod draft_loader;
pub mod execution_graph;
pub mod hardware;
pub mod int4_pack;
#[cfg(feature = "prism-backend")]
pub mod kernel_dispatch;
#[cfg(feature = "prism-backend")]
pub mod kernel_registry;
pub mod kernel_types;
#[cfg(all(
    target_os = "macos",
    any(feature = "mlx-backend", feature = "prism-backend")
))]
pub mod portfolio;
#[cfg(feature = "tensix")]
pub mod tensix;
pub mod ternary;
pub mod ternary_pipeline;
pub mod tts_compile;
#[cfg(feature = "prism-backend")]
pub mod validation_matrix;

pub use download::*;
#[cfg(feature = "mlx-backend")] // research surface: MLX compile lane
pub(crate) use emit::*;
#[cfg(feature = "mlx-backend")] // research surface: MLX compile lane
pub use pipeline::*;
pub use quantize::*;
pub use source::*;
