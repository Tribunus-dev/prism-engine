//! Gemma 4 full-transformer GPU megakernel.
//!
//! Splits the original monolithic `megakernel.rs` into three concerns:
//! - [`kernels`] — architecture constants, Metal shader source, on-the-fly compilation
//! - [`kv`] — ternary KV cache block constants
//! - [`pipeline`] — persistent dispatch, work queue, buffer management, host API

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
