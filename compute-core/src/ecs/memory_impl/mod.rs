//! Engine-internal execution-plane memory subsystem.
//!
//! The constitutional surface for engine-independent memory data
//! types and pure abstractions lives in `prism_ecs_data::memory`.
//! This module is the engine-internal home for the execution-plane
//! code that depends on engine-internal `Arena`, `ExternalStorage`,
//! `MappedSegment`, `TensorEntry`, `worker_memory`, and the
//! `tribunus_arena_alloc` / `mlx_set_memory_plan` C FFI bridges.
//!
//! # Re-exports
//!
//! The data types that are now in `prism_ecs_data::memory` are
//! re-exported here so existing engine callers that import
//! `crate::memory_impl::MemoryEnforcer` etc. continue to work.
//! New engine code should prefer the
//! `prism_ecs_data::memory::...` import path; the re-exports here
//! are the migration bridge.
//!
//! See `docs/compute-image-memory-architecture.md` and
//! `docs/unified-memory-island.md` for the full design.

#[cfg(feature = "mlx-backend")]
pub mod allocator;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub mod candle_bridge;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub mod compute_image_bridge;
#[cfg(feature = "mlx-backend")]
pub mod iosurface_storage;
#[cfg(feature = "mlx-backend")]
pub mod plan;
#[cfg(feature = "mlx-backend")]
pub mod telemetry;

// Re-exports of the constitutional data types (see
// `prism_ecs_data::memory`). Existing engine callers that import
// `crate::memory_impl::MemoryEnforcer` etc. continue to work;
// new code should prefer the constitutional path.
pub use prism_ecs_data::memory::{
    EngineEntry, EngineLifecycle, EnginePool, MemoryEnforcer, MemoryMonitor, MemoryPressure,
};

// Re-exports of the engine-internal execution-plane types
// (defined in the submodules of this module).
#[cfg(feature = "mlx-backend")]
pub use allocator::{BlockHandle, IosurfaceAllocator, KvCacheBlockAllocator, PagedIosurfaceAllocator};
