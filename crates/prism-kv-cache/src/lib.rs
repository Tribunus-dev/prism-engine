//! KV cache types extracted from `tribunus-compute-core`.
//!
//! Provides block-pool, layered prefix-caching, sliding-window temporal,
//! and spatial grid-decimation KV cache implementations. The
//! `prism_kv_cache::arena` module is the canonical home for the engine's
//! absorbed paged KV-cache arena surface (physical blocks, backend
//! residency, prefix hashing, refcount/eviction, and the aggregator).

pub mod arena;
pub mod block_table;
pub mod grid_decimation;
pub mod layered_cache;
pub mod sliding_window;

pub use arena::{
    AdmissionReceipt, ArenaError, KvBlockArena, KvCachePlan, LogicalBlockTable, SequenceId,
};
pub use block_table::*;
pub use grid_decimation::*;
pub use layered_cache::*;
pub use sliding_window::*;
