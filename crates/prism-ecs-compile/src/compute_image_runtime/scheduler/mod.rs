//! Decode-batch scheduling — pure data types and algorithms for batching
//! inference slots by system load level.
//!
//! The scheduler is a CPU-side helper: it manages a queue of ready slot
//! ids and selects a batch of slots sized to the current load. It does
//! not perform dispatch; dispatch lives in the engine-side executor.

pub mod batch_scheduler;
pub mod load_monitor;

pub use batch_scheduler::BatchScheduler;
pub use load_monitor::LoadMonitor;

/// Describes the current system load level as observed by the scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LoadLevel {
    /// Low load — small batches.
    Low,
    /// Medium load — moderate batches.
    Medium,
    /// High load — large batches up to the configured maximum.
    High,
}
