pub mod batch_scheduler;
pub mod load_monitor;

pub use batch_scheduler::BatchScheduler;
pub use load_monitor::LoadMonitor;

/// Describes the current system load level as observed by the scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadLevel {
    Low,
    Medium,
    High,
}
