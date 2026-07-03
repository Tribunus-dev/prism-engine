//! Server operational mode and memory allocation broker.
//!
//! The `MemoryAllocationBroker` enforces a 10.5 GB ceiling for the Prism
//! process (reserving 3.5 GB for macOS / system services on a 16 GB M1).
//! It is a *policy* object — consulted by the distillation worker and
//! inference engine — not a pre-allocator.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Server operational mode — controls how the memory broker adjusts
/// allocation ceilings between inference and distillation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServerOperationalMode {
    /// No active workloads.
    Idle,
    /// Serving inference requests.
    Serving,
    /// Running a distillation pass (may throttle or pause inference).
    Distilling,
    /// Serving and distilling concurrently (microbatch_size may be reduced).
    Hybrid,
}

/// Memory allocation broker — enforces process-level memory ceilings.
///
/// **System margin**: 3.5 GB reserved for macOS / system services.
/// **Prism global ceiling**: 10.5 GB (11,274,289,152 bytes).
/// **Distillation sub-ceiling**: 6.0 GB (compiler gets 6 GB max).
/// **Inference sub-ceiling**: 4.5 GB (runtime + KV cache).
pub struct MemoryAllocationBroker {
    /// Current operational mode.
    mode: AtomicU64, // 0=Idle, 1=Serving, 2=Distilling, 3=Hybrid
    /// Bytes currently allocated (estimate).
    allocated_bytes: AtomicU64,
    /// Whether distillation is active (fast-check flag).
    distilling_active: AtomicBool,
}

impl MemoryAllocationBroker {
    /// System margin reserved for macOS.
    pub const SYSTEM_MARGIN_BYTES: u64 = 3_758_096_384; // 3.5 GB
    /// Absolute max for the Prism process.
    pub const PRISM_CEILING_BYTES: u64 = 11_274_289_152; // 10.5 GB
    /// Maximum for the distillation compiler.
    pub const DISTILL_SUB_CEILING_BYTES: u64 = 6_442_173_952; // 6.0 GB
    /// Maximum for inference serving.
    pub const INFERENCE_SUB_CEILING_BYTES: u64 = 4_831_630_464; // 4.5 GB

    /// Create a new broker with `Idle` mode and zero allocation.
    pub fn new() -> Self {
        MemoryAllocationBroker {
            mode: AtomicU64::new(0),
            allocated_bytes: AtomicU64::new(0),
            distilling_active: AtomicBool::new(false),
        }
    }

    /// Current operational mode.
    pub fn mode(&self) -> ServerOperationalMode {
        match self.mode.load(Ordering::Acquire) {
            0 => ServerOperationalMode::Idle,
            1 => ServerOperationalMode::Serving,
            2 => ServerOperationalMode::Distilling,
            3 => ServerOperationalMode::Hybrid,
            _ => ServerOperationalMode::Idle,
        }
    }

    /// Transition to a new mode. Returns the previous mode.
    pub fn set_mode(&self, new: ServerOperationalMode) -> ServerOperationalMode {
        let prev = self.mode.swap(new as u64, Ordering::AcqRel);
        self.distilling_active
            .store(new == ServerOperationalMode::Distilling, Ordering::Release);
        match prev {
            0 => ServerOperationalMode::Idle,
            1 => ServerOperationalMode::Serving,
            2 => ServerOperationalMode::Distilling,
            3 => ServerOperationalMode::Hybrid,
            _ => ServerOperationalMode::Idle,
        }
    }

    /// Whether a distillation pass is running.
    pub fn is_distilling(&self) -> bool {
        self.distilling_active.load(Ordering::Acquire)
    }

    /// Record an allocation (add to counter).
    pub fn declare(&self, bytes: u64) {
        self.allocated_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record a release (subtract from counter).
    pub fn release(&self, bytes: u64) {
        self.allocated_bytes.fetch_sub(bytes, Ordering::Relaxed);
    }

    /// Current estimated allocation in bytes.
    pub fn allocated(&self) -> u64 {
        self.allocated_bytes.load(Ordering::Relaxed)
    }

    /// Available budget in bytes (PRISM_CEILING - allocated).
    pub fn available(&self) -> u64 {
        Self::PRISM_CEILING_BYTES.saturating_sub(self.allocated())
    }

    /// Available budget for distillation (DISTILL_SUB_CEILING - allocated
    /// when mode == Distilling, otherwise 0).
    pub fn distill_available(&self) -> u64 {
        if self.is_distilling() {
            Self::DISTILL_SUB_CEILING_BYTES.saturating_sub(self.allocated())
        } else {
            0
        }
    }
}

impl Default for MemoryAllocationBroker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broker_defaults() {
        let broker = MemoryAllocationBroker::new();
        assert_eq!(broker.mode(), ServerOperationalMode::Idle);
        assert!(!broker.is_distilling());
        assert_eq!(broker.allocated(), 0);
        assert_eq!(broker.available(), MemoryAllocationBroker::PRISM_CEILING_BYTES);
    }

    #[test]
    fn test_mode_transitions() {
        let broker = MemoryAllocationBroker::new();
        broker.set_mode(ServerOperationalMode::Distilling);
        assert_eq!(broker.mode(), ServerOperationalMode::Distilling);
        assert!(broker.is_distilling());
        broker.set_mode(ServerOperationalMode::Idle);
        assert!(!broker.is_distilling());
    }

    #[test]
    fn test_declare_release() {
        let broker = MemoryAllocationBroker::new();
        broker.declare(1_000_000);
        assert_eq!(broker.allocated(), 1_000_000);
        broker.release(500_000);
        assert_eq!(broker.allocated(), 500_000);
    }

    #[test]
    fn test_distill_available() {
        let broker = MemoryAllocationBroker::new();
        assert_eq!(broker.distill_available(), 0); // not distilling
        broker.set_mode(ServerOperationalMode::Distilling);
        assert_eq!(
            broker.distill_available(),
            MemoryAllocationBroker::DISTILL_SUB_CEILING_BYTES
        );
        broker.declare(1_000_000_000);
        assert!(broker.distill_available() < MemoryAllocationBroker::DISTILL_SUB_CEILING_BYTES);
    }
}
