//! ComputeImage compilation pipeline — source loading, quantization,
//! sequential/differential compilation, diagnostics, and publishing.

pub mod archive;
pub mod capability_registry;
mod download;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
mod emit;
#[cfg(feature = "mlx-backend")] // research surface: MLX compile lane
mod pipeline;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
mod quantize;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub mod source;

#[cfg(all(
    target_os = "macos",
    any(feature = "mlx-backend", feature = "prism-backend")
))]
pub mod coreai;
pub mod draft_loader;
pub mod execution_graph;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub mod hardware;
#[cfg(feature = "amd-rocm")]
pub mod hip_dispatch;
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
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub mod tts_compile;
#[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
pub mod validation_matrix;

pub use download::*;
#[cfg(feature = "mlx-backend")] // research surface: MLX compile lane
pub(crate) use emit::*;
#[cfg(feature = "mlx-backend")]
// research surface: MLX compile lane — pipeline requires mlx-rs
pub use pipeline::*;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub use quantize::*;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub use source::*;
