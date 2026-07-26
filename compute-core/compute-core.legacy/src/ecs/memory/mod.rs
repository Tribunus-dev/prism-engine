//! Unified memory island for Tribunus Compute.
//!
//! See `docs/compute-image-memory-architecture.md` and
//! `docs/unified-memory-island.md`.

#[cfg(feature = "mlx-backend")]
pub mod allocator;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub mod candle_bridge;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub mod compute_image_bridge;
pub mod coreai_warmup;
pub mod enforcer;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub mod iosurface_storage;
pub mod monitor;
pub mod plan;
pub mod pool;
#[cfg(feature = "mlx-backend")]
pub mod telemetry;

#[cfg(feature = "mlx-backend")]
pub use allocator::BlockHandle;
#[cfg(feature = "mlx-backend")]
pub use allocator::IosurfaceAllocator;
#[cfg(feature = "mlx-backend")]
pub use allocator::KvCacheBlockAllocator;
#[cfg(feature = "mlx-backend")]
pub use allocator::PagedIosurfaceAllocator;
pub use enforcer::MemoryEnforcer;
pub use monitor::MemoryMonitor;
pub use pool::EnginePool;

/// Memory pressure level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MemoryPressure {
    Normal = 0,
    Warning = 1,
    Critical = 2,
    Severe = 3,
    Oom = 4,
}
