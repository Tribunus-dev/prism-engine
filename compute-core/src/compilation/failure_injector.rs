//! Fallback dispatch infrastructure.
//!
//! Provides the [`FailureInjector`] trait for controlled epoch-level
//! failure injection during CI/soak testing, the concrete
//! [`NoopFailureInjector`] and [`EpochFailureInjector`] implementations,
//! and the [`FallbackExecutionReceipt`] record for observability of
//! actual fallback dispatch events.

use serde::{Deserialize, Serialize};

// ── Failure injection trait ──────────────────────────────────────────────

/// Strategy for injecting failures before a Core ML prediction epoch.
///
/// Implementations are used during the decode loop to decide whether to
/// skip the primary (Core ML ANE) lane and route execution through the
/// fallback lane instead.
pub trait FailureInjector: Send + Sync {
    /// Returns `true` when the given `epoch` should be treated as a
    /// primary-lane failure, triggering fallback dispatch.
    fn should_fail_before_prediction(&self, epoch: u64) -> bool;
}

// ── Noop injector ────────────────────────────────────────────────────────

/// Injector that never triggers fallback — production default.
pub struct NoopFailureInjector;

impl FailureInjector for NoopFailureInjector {
    fn should_fail_before_prediction(&self, _epoch: u64) -> bool {
        false
    }
}

// ── Epoch counter injector ───────────────────────────────────────────────

/// Injector that fails every `fail_every` epochs (starting with epoch 1).
///
/// Epoch 0 is never failed so the first epoch always runs the primary
/// lane, establishing a baseline.
pub struct EpochFailureInjector {
    pub fail_every: u64,
}

impl FailureInjector for EpochFailureInjector {
    fn should_fail_before_prediction(&self, epoch: u64) -> bool {
        epoch > 0 && epoch % self.fail_every == 0
    }
}

// ── Fallback execution receipt ───────────────────────────────────────────

/// Observability record for a fallback dispatch event.
///
/// Produced when the decode loop detects a primary-lane failure (either
/// from the injector or a real Core ML error) and successfully routes
/// the epoch through an alternative execution lane.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FallbackExecutionReceipt {
    /// Epoch index at which fallback was triggered.
    pub epoch: u64,
    /// Name of the lane that failed (e.g. `"coreml_ane"`).
    pub failed_primary_lane: String,
    /// Name of the lane that executed instead (e.g. `"metal_gpu"`,
    /// `"cpu"`).
    pub fallback_lane: String,
    /// IO-arena input slot used for the fallback dispatch.
    pub input_slot_id: u32,
    /// IO-arena output slot used for the fallback dispatch.
    pub output_slot_id: u32,
    /// Generation counter of the input surface at fallback time.
    pub input_generation: u64,
    /// Generation counter of the output surface at fallback time.
    pub output_generation: u64,
    /// Digest identifying the fallback kernel that executed.
    pub fallback_kernel_digest: String,
    /// Whether the Metal (or CPU) command buffer completed.
    pub command_buffer_completed: bool,
    /// Whether the fallback output was published to the IO arena.
    pub output_published: bool,
}
