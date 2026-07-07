//! Re-exported from `runtime::resources::kv_cache_coordinator`.
//!
//! All types have been promoted to the runtime resource layer.  This shim
//! preserves existing import paths for the transition.
//!
//! See: `crate::runtime::resources::kv_cache_coordinator`

#[cfg(feature = "mlx-backend")]
pub use crate::runtime::resources::kv_cache_coordinator::*;

pub mod block_table;
pub mod grid_decimation;
pub mod sliding_window;

pub use block_table::BlockTable;
