//! Types re-exported from `prism-kv-cache` and `runtime::resources::kv_cache_coordinator`.
//!
//! All primary KV cache types now live in `prism-kv-cache`. This shim
//! preserves existing import paths for the transition.
//! See: `crate::runtime::resources::kv_cache_coordinator`

pub use prism_kv_cache::*;

#[cfg(feature = "mlx-backend")]
pub use crate::runtime::resources::kv_cache_coordinator::*;
