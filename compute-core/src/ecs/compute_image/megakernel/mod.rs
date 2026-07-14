//! Gemma 4 full-transformer GPU megakernel.
//!
//! Splits the original monolithic `megakernel.rs` into three concerns:
//! - [`kernels`] — architecture constants, Metal shader source, on-the-fly compilation
//! - [`kv`] — ternary KV cache block constants
//! - [`pipeline`] — persistent dispatch, work queue, buffer management, host API
//!
//! ## Megakernel vs decomposed kernel strategy
//!
//! The Gemma 4 transformer can be executed via two Metal code paths:
//!
//! - **Megakernel** (`prism.transformer.gemma4.decode.v1`): a single monolithic
//!   GPU compute shader (`gemma4_full_decode_persistent`) that performs the entire
//!   decode step for one layer in one dispatch. Registered by
//!   `MetalImplementationCatalogue::register_megakernel()`.
//!
//! - **Decomposed per-op kernels** (`prism.gemma4.rms_norm`, `prism.gemma4.rope`,
//!   ..., `prism.gemma4.mtp_output`): individual Metal shaders, one per transformer
//!   sub-operation, dispatched sequentially. Registered by
//!   `MetalImplementationCatalogue::register_gemma4_decomposed()` with source paths
//!   under `src/ecs/compute_image/gemma4/`.
//!
//! The decomposed path is intended for (a) compositional debugging / profiling,
//! (b) partial model execution (e.g. single-op validation), and (c) future
//! just-in-time compilation of individual shards. The megakernel is the
//! production entry point for full decode throughput.

#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub mod gather_kernel;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub mod kernels;
pub mod kv;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub mod pipeline;

#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub use kernels::*;
pub use kv::*;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub use pipeline::*;
