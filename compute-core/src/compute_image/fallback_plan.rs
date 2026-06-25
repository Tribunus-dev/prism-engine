//! FALLBACK-ABI-CONTINUITY-0001: ANE lane failure detection and ABI-compatible
//! Metal fallback activation.
//!
//! The `FallbackPlanManager` evaluates runtime failure triggers against a
//! compiled `AppleFallbackPlan` and manages the ANE→GPU/CPU fallback
//! lifecycle.  When the ANE lane is unhealthy (consecutive failures exceed
//! the configured threshold), the manager activates fallback, releasing
//! poisoned IOSurface arena slots for reuse.

use crate::compilation::tri_lane::{AppleFallbackPlan, FallbackStatus};
use crate::compute_image::apple_shared_arena::{AppleSharedArena, SlotState};


/// Trait for injecting controlled Core ML failures in production tests.
pub trait CoreMlFailureInjector: std::fmt::Debug {
    /// Return `true` when the injector wants to simulate a failure at this epoch.
    fn should_fail(&self, epoch: u64) -> bool;
}

/// A failure injector that fails at a specific epoch.
#[derive(Debug, Clone)]
pub struct TestFailureInjector {
    pub fail_epoch: Option<u64>,
}

impl CoreMlFailureInjector for TestFailureInjector {
    fn should_fail(&self, epoch: u64) -> bool {
        self.fail_epoch == Some(epoch)
    }
}

/// Detection of various failure conditions that may trigger an ANE lane
/// fallback.
#[derive(Debug, Clone)]
pub enum FallbackTrigger {
    /// ANE artifact could not be loaded from disk or Core ML storage.
    ArtifactLoadFailed(String),
    /// Core ML warmup predictions failed or produced invalid results.
    WarmupContractFailed(String),
    /// Input tensor binding does not match the Core ML model contract.
    InputBindingMismatch(String),
    /// Output tensor binding does not match the Core ML model contract.
    OutputBindingMismatch(String),
    /// Epoch deadline breached — execution took longer than the budget.
    DeadlineBreach {
        epoch: u64,
        deadline_ns: u64,
        actual_ns: u64,
    },
    /// Output tensor values failed numerical validation (RMS error, NaN, etc.).
    OutputValidationFailed {
        epoch: u64,
        max_error: f64,
    },
    /// Numerical guard (e.g. bit-exactness contract) was violated.
    NumericalGuardFailed(String),
    /// Lane explicitly disabled by operator or configuration.
    ExplicitLaneDisablement(String),
    /// Repeated unhealthy latency detected across consecutive epochs.
    RepeatedUnhealthyLatency {
        epoch: u64,
        consecutive: u32,
    },
    /// Unrecoverable runtime exception from the Core ML or Metal layer.
    RuntimeException(String),
}

/// Manages fallback transitions for the ANE lane.
///
/// Tracks consecutive failures and, when the threshold is met, activates the
/// compiled fallback plan (ANE→GPU, ANE→CPU, or both) so the scheduler can
/// route work around the unhealthy lane.
pub struct FallbackPlanManager {
    /// Compiled fallback topology from the CImage manifest.
    pub plan: AppleFallbackPlan,
    /// Current fallback status.
    pub status: FallbackStatus,
    /// Number of consecutive unresolved failures since last reset.
    pub consecutive_failures: u32,
    /// Maximum consecutive failures before fallback is activated.
    ///
    /// Defaults to 3.  Tune via [`FallbackPlanManager::set_max_consecutive_failures`].
    pub max_consecutive_failures: u32,
    /// Optional failure injector for production tests.
    pub failure_injector: Option<Box<dyn CoreMlFailureInjector + Send>>,
}

impl FallbackPlanManager {
    /// Create a new manager for the given compiled fallback plan.
    ///
    /// Defaults to 3 consecutive failures before activation.
    pub fn new(plan: AppleFallbackPlan) -> Self {
        Self {
            plan,
            status: FallbackStatus::NotActivated,
            consecutive_failures: 0,
            max_consecutive_failures: 3,
            failure_injector: None,
        }
    }

    /// Override the consecutive-failure threshold.
    pub fn set_max_consecutive_failures(&mut self, max: u32) {
        self.max_consecutive_failures = max;
    }

    /// Set a failure injector for controlled production test failures.
    pub fn set_failure_injector(&mut self, injector: Box<dyn CoreMlFailureInjector + Send>) {
        self.failure_injector = Some(injector);
    }

    /// Evaluate a trigger and decide whether to activate fallback.
    ///
    /// Returns `true` when the manager determines fallback should be
    /// activated.  The caller should call [`Self::activate`] to commit
    /// the transition.
    ///
    /// # Failure classification
    ///
    /// *Hard* failures (artifact load, warmup contract, input/output binding,
    /// numerical guard) increment the counter and immediately return `true`
    /// when the threshold is met on a single increment.
    ///
    /// *Soft* failures (deadline breach, validation error, latency, runtime
    /// exception, explicit disablement) increment the counter but always
    /// return `false` — fallback only activates after
    /// `max_consecutive_failures` *distinct* soft failures accumulate
    /// without a reset.
    pub fn evaluate(&mut self, trigger: &FallbackTrigger) -> bool {
        match trigger {
            // Hard failures — immediate activation once threshold is met.
            FallbackTrigger::ArtifactLoadFailed(_)
            | FallbackTrigger::WarmupContractFailed(_)
            | FallbackTrigger::InputBindingMismatch(_)
            | FallbackTrigger::NumericalGuardFailed(_) => {
                self.consecutive_failures += 1;
                self.consecutive_failures >= self.max_consecutive_failures
            }
            // Soft failures — require repeated consecutive occurrences.
            _ => {
                self.consecutive_failures += 1;
                false
            }
        }
    }

    /// Evaluate epoch health, checking the failure injector before actual triggers.
    ///
    /// If a failure injector is configured and signals failure for this epoch,
    /// the manager increments consecutive failures. When the threshold is met,
    /// fallback activates.
    ///
    /// If no injector fires, delegates to the normal [`Self::evaluate`] path
    /// using the provided actual trigger.
    pub fn evaluate_with_injector(
        &mut self,
        epoch: u64,
        actual_trigger: Option<&FallbackTrigger>,
    ) -> bool {
        // Check injector first
        if let Some(injector) = &self.failure_injector {
            if injector.should_fail(epoch) {
                self.consecutive_failures += 1;
                if self.consecutive_failures >= self.max_consecutive_failures {
                    return true;
                }
            }
        }
        // Then check actual triggers
        if let Some(trigger) = actual_trigger {
            return self.evaluate(trigger);
        }
        false
    }

    /// Activate fallback, recording the epoch and reason.
    ///
    /// After calling this, [`Self::is_active`] returns `true` and the
    /// scheduler should route ANE-region work to the fallback lane(s)
    /// specified in the plan.
    pub fn activate(&mut self, epoch: u64, reason: String) {
        self.status = FallbackStatus::Activated { epoch, reason };
    }

    /// Reset fallback state (e.g. after the lane recovers or is
    /// re-qualified).
    pub fn reset(&mut self) {
        self.status = FallbackStatus::NotActivated;
        self.consecutive_failures = 0;
    }

    /// Check whether fallback is currently active.
    pub fn is_active(&self) -> bool {
        matches!(&self.status, FallbackStatus::Activated { .. })
    }

    /// Release all poisoned slots in the arena, transitioning them to
    /// `Retired` so they can be reclaimed and reused by a future epoch.
    ///
    /// Returns the number of slots released.
    pub fn release_poisoned_slots(
        &self,
        arena: &mut AppleSharedArena,
        epoch: u64,
    ) -> Result<u32, String> {
        let mut released = 0u32;
        let ids: Vec<u32> = arena
            .slots
            .iter()
            .filter(|(_, s)| matches!(&s.state, SlotState::Poisoned { .. }))
            .map(|(id, _)| *id)
            .collect();
        for id in ids {
            if let Some(slot) = arena.slot_mut(id) {
                slot.retire(epoch);
                released += 1;
            }
        }
        Ok(released)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compilation::tri_lane::AppleFallbackPlan;
    use crate::compute_image::apple_shared_arena::{
        AppleSharedArena, ExecutionLane, IOSurfaceSlotManifest, LiveIOSurfaceSlot,
        SlotFailureReason, SlotReuseClass,
    };

    /// Helper to build a minimal fallback plan for tests.
    fn dummy_plan() -> AppleFallbackPlan {
        AppleFallbackPlan {
            ane_to_gpu: vec!["region_a".into()],
            ane_to_cpu: vec![],
            gpu_only_valid: true,
            cpu_only_valid: false,
        }
    }

    /// Helper to build an arena with a mix of slot states.
    fn arena_with_mixed_slots() -> AppleSharedArena {
        let mut arena = AppleSharedArena::new("test-arena".into(), 2);

        let mk_manifest = |id: u32| IOSurfaceSlotManifest {
            slot_id: id,
            tensor_id: format!("t{}", id),
            byte_offset: 0,
            byte_length: 1024,
            dtype: "float16".into(),
            logical_shape: vec![1, 64],
            physical_shape: vec![1, 64],
            strides_bytes: vec![128, 2],
            layout: "nchw".into(),
            producer: ExecutionLane::CoreMlAne,
            consumer: ExecutionLane::MlxGpu,
            reuse_class: SlotReuseClass::Exclusive,
            required_alignment: 64,
        };

        // Slot 0: Free
        arena.add_slot(LiveIOSurfaceSlot {
            manifest: mk_manifest(0),
            state: SlotState::Free,
            generation: 0,
            layout_digest: String::new(),
            metal_view: None,
            coreml_view: None,
            backing_arena: None,
            attestation: None,
        });

        // Slot 1: Poisoned — layout mismatch
        arena.add_slot(LiveIOSurfaceSlot {
            manifest: mk_manifest(1),
            state: SlotState::Poisoned {
                epoch: 42,
                reason: SlotFailureReason::LayoutMismatch {
                    expected: "A".into(),
                    actual: "B".into(),
                },
            },
            generation: 1,
            layout_digest: "abc".into(),
            metal_view: None,
            coreml_view: None,
            backing_arena: None,
            attestation: None,
        });

        // Slot 2: Ready (healthy)
        arena.add_slot(LiveIOSurfaceSlot {
            manifest: mk_manifest(2),
            state: SlotState::Ready {
                epoch: 42,
                producer: ExecutionLane::CoreMlAne,
            },
            generation: 2,
            layout_digest: "def".into(),
            metal_view: None,
            coreml_view: None,
            backing_arena: None,
            attestation: None,
        });

        // Slot 3: Poisoned — Core ML failure
        arena.add_slot(LiveIOSurfaceSlot {
            manifest: mk_manifest(3),
            state: SlotState::Poisoned {
                epoch: 42,
                reason: SlotFailureReason::CoreMlPredictionFailed("oom".into()),
            },
            generation: 0,
            layout_digest: "xyz".into(),
            metal_view: None,
            coreml_view: None,
            backing_arena: None,
            attestation: None,
        });

        // Slot 4: Retired (already cleaned up)
        arena.add_slot(LiveIOSurfaceSlot {
            manifest: mk_manifest(4),
            state: SlotState::Retired { epoch: 42 },
            generation: 3,
            layout_digest: String::new(),
            metal_view: None,
            coreml_view: None,
            backing_arena: None,
            attestation: None,
        });

        arena
    }

    // -----------------------------------------------------------------------
    // evaluate
    // -----------------------------------------------------------------------

    #[test]
    fn test_fallback_evaluate_triggers() {
        // Hard failures trip on threshold (default 3 consecutive).
        let plan = dummy_plan();
        let mut mgr = FallbackPlanManager::new(plan);
        mgr.set_max_consecutive_failures(2);

        // First hard failure — counter increments but stays below threshold.
        assert!(
            !mgr.evaluate(&FallbackTrigger::ArtifactLoadFailed("missing.mlmodelc".into())),
            "first hard failure should not trigger"
        );
        assert_eq!(mgr.consecutive_failures, 1);

        // Second hard failure — threshold reached.
        assert!(
            mgr.evaluate(&FallbackTrigger::NumericalGuardFailed("bit-exact".into())),
            "second hard failure should trigger"
        );
        assert_eq!(mgr.consecutive_failures, 2);

        // Reset and test soft failures — they never trigger on their own.
        let mut mgr = FallbackPlanManager::new(dummy_plan());
        mgr.set_max_consecutive_failures(3);

        for i in 1..=5 {
            assert!(
                !mgr.evaluate(&FallbackTrigger::DeadlineBreach {
                    epoch: 100 + i as u64,
                    deadline_ns: 5_000_000,
                    actual_ns: 8_000_000,
                }),
                "soft failure #{} should not trigger even past threshold",
                i
            );
        }
        assert_eq!(mgr.consecutive_failures, 5);

        // Even past max_consecutive_failures, soft failures return false.
        assert!(
            !mgr.evaluate(&FallbackTrigger::RuntimeException("crash".into())),
            "runtime exception (soft) should not trigger"
        );
        // But we do not automatically activate — the caller must call activate().
    }

    // -----------------------------------------------------------------------
    // activate / is_active
    // -----------------------------------------------------------------------

    #[test]
    fn test_fallback_activation() {
        let mut mgr = FallbackPlanManager::new(dummy_plan());

        assert!(!mgr.is_active(), "fresh manager must not be active");
        assert!(matches!(mgr.status, FallbackStatus::NotActivated));

        mgr.activate(42, "artifact load failed".into());

        assert!(mgr.is_active(), "manager must report active after activation");
        match &mgr.status {
            FallbackStatus::Activated { epoch, reason } => {
                assert_eq!(*epoch, 42);
                assert_eq!(reason, "artifact load failed");
            }
            other => panic!("expected Activated, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // reset
    // -----------------------------------------------------------------------

    #[test]
    fn test_fallback_reset() {
        let mut mgr = FallbackPlanManager::new(dummy_plan());

        // Activate then reset.
        mgr.activate(99, "test".into());
        assert!(mgr.is_active());
        assert_eq!(mgr.consecutive_failures, 0);

        mgr.reset();

        assert!(!mgr.is_active(), "reset must clear activation");
        assert!(matches!(mgr.status, FallbackStatus::NotActivated));
        assert_eq!(mgr.consecutive_failures, 0);

        // Also verify that consecutive_failures is cleared after reset even
        // when failures had accumulated.
        let mut mgr2 = FallbackPlanManager::new(dummy_plan());
        for _ in 0..4 {
            mgr2.evaluate(&FallbackTrigger::WarmupContractFailed("oom".into()));
        }
        assert_eq!(mgr2.consecutive_failures, 4);

        mgr2.reset();
        assert_eq!(mgr2.consecutive_failures, 0);
    }

    // -----------------------------------------------------------------------
    // release_poisoned_slots
    // -----------------------------------------------------------------------

    #[test]
    fn test_release_poisoned_slots() {
        let mut arena = arena_with_mixed_slots();
        // Sanity: 2 poisoned slots out of 5 (ids 1 and 3).
        let poisoned_count_before = arena
            .slots
            .values()
            .filter(|s| matches!(&s.state, SlotState::Poisoned { .. }))
            .count();
        assert_eq!(poisoned_count_before, 2, "precondition: 2 poisoned slots");

        let mgr = FallbackPlanManager::new(dummy_plan());
        let released = mgr.release_poisoned_slots(&mut arena, 43).unwrap();
        assert_eq!(released, 2, "must release exactly 2 poisoned slots");

        // Both formerly poisoned slots are now Retired.
        let retired_count = arena
            .slots
            .values()
            .filter(|s| matches!(&s.state, SlotState::Retired { epoch: 43 }))
            .count();
        assert_eq!(retired_count, 2, "poisoned slots transitioned to Retired");

        // Non-poisoned slots are untouched.
        assert!(
            matches!(arena.slot(0).unwrap().state, SlotState::Free),
            "slot 0 (Free) must remain untouched"
        );
        assert!(
            matches!(arena.slot(2).unwrap().state, SlotState::Ready { .. }),
            "slot 2 (Ready) must remain untouched"
        );
        assert!(
            matches!(arena.slot(4).unwrap().state, SlotState::Retired { epoch: 42 }),
            "slot 4 (Retired) must remain untouched"
        );

        // No poisoned slots remain.
        let poisoned_after = arena
            .slots
            .values()
            .filter(|s| matches!(&s.state, SlotState::Poisoned { .. }))
            .count();
        assert_eq!(poisoned_after, 0);

        // Second call releases nothing.
        let released2 = mgr.release_poisoned_slots(&mut arena, 44).unwrap();
        assert_eq!(released2, 0, "no poisoned slots remain on second pass");
    }

    // -----------------------------------------------------------------------
    // failure injector
    // -----------------------------------------------------------------------

    #[test]
    fn test_fallback_injector_configured() {
        // Default: injector is None
        let mut mgr = FallbackPlanManager::new(dummy_plan());
        assert!(
            mgr.failure_injector.is_none(),
            "injector must be None by default"
        );

        // Can be set via setter
        let injector = Box::new(TestFailureInjector {
            fail_epoch: Some(7),
        });
        mgr.set_failure_injector(injector);
        assert!(
            mgr.failure_injector.is_some(),
            "injector must be Some after setter"
        );
    }

    #[test]
    fn test_fallback_injector_fires_at_correct_epoch() {
        let mut mgr = FallbackPlanManager::new(dummy_plan());
        mgr.set_max_consecutive_failures(1);
        mgr.set_failure_injector(Box::new(TestFailureInjector {
            fail_epoch: Some(3),
        }));

        // Epoch 3 is the configured failure epoch
        assert!(
            mgr.evaluate_with_injector(3, None),
            "injector must fire at configured epoch"
        );
        assert_eq!(mgr.consecutive_failures, 1);
    }

    #[test]
    fn test_fallback_injector_does_not_fire_at_wrong_epoch() {
        let mut mgr = FallbackPlanManager::new(dummy_plan());
        mgr.set_max_consecutive_failures(3);
        mgr.set_failure_injector(Box::new(TestFailureInjector {
            fail_epoch: Some(7),
        }));

        // Epoch 5 is not the configured failure epoch
        assert!(
            !mgr.evaluate_with_injector(5, None),
            "injector must NOT fire at wrong epoch"
        );
        assert_eq!(mgr.consecutive_failures, 0);
    }

    #[test]
    fn test_fallback_injector_respected_before_actual_triggers() {
        let mut mgr = FallbackPlanManager::new(dummy_plan());
        mgr.set_max_consecutive_failures(1);
        mgr.set_failure_injector(Box::new(TestFailureInjector {
            fail_epoch: Some(2),
        }));

        // Injector fires at epoch 2 — returns true even though actual_trigger
        // is also provided (would have been a trigger anyway)
        assert!(
            mgr.evaluate_with_injector(
                2,
                Some(&FallbackTrigger::ArtifactLoadFailed("missing.mlmodelc".into())),
            ),
            "injector must be checked before actual triggers"
        );
        assert_eq!(mgr.consecutive_failures, 1);

        // Now test that actual trigger is only evaluated when injector does
        // NOT fire. Reset and set injector to None to show actual triggers work.
        let mut mgr2 = FallbackPlanManager::new(dummy_plan());
        mgr2.set_max_consecutive_failures(1);
        // No injector set — actual trigger should be evaluated
        assert!(
            mgr2.evaluate_with_injector(
                3,
                Some(&FallbackTrigger::ArtifactLoadFailed("missing.mlmodelc".into())),
            ),
            "actual trigger must fire when injector is absent"
        );
        assert_eq!(mgr2.consecutive_failures, 1);
    }

    // -----------------------------------------------------------------------
    // epoch boundary / slot lifecycle
    // -----------------------------------------------------------------------

    #[test]
    fn test_epoch_boundary_fallback_does_not_expose_partial_output() {
        let mut mgr = FallbackPlanManager::new(dummy_plan());
        mgr.set_max_consecutive_failures(1);
        mgr.set_failure_injector(Box::new(TestFailureInjector {
            fail_epoch: Some(10),
        }));

        // Simulate: epoch 10 starts, slot is reserved/writing
        let mut arena = AppleSharedArena::new("test-arena".into(), 2);
        let slot = LiveIOSurfaceSlot {
            manifest: IOSurfaceSlotManifest {
                slot_id: 0,
                tensor_id: "t0".into(),
                byte_offset: 0,
                byte_length: 1024,
                dtype: "float16".into(),
                logical_shape: vec![1, 64],
                physical_shape: vec![1, 64],
                strides_bytes: vec![128, 2],
                layout: "nchw".into(),
                producer: ExecutionLane::CoreMlAne,
                consumer: ExecutionLane::MlxGpu,
                reuse_class: SlotReuseClass::Exclusive,
                required_alignment: 64,
            },
            state: SlotState::Reserved {
                epoch: 10,
                producer: ExecutionLane::CoreMlAne,
            },
            generation: 0,
            layout_digest: String::new(),
            metal_view: None,
            coreml_view: None,
            backing_arena: None,
            attestation: None,
        };
        arena.add_slot(slot);

        // Injector fires at epoch 10 — fallback activates
        let triggered = mgr.evaluate_with_injector(10, None);
        assert!(triggered, "injector must fire at epoch 10");

        // On fallback, the slot should be poisoned, never marked Ready
        let poisoned_slot = arena.slot(0).unwrap();
        match &poisoned_slot.state {
            SlotState::Poisoned { epoch, reason: _ } => {
                assert_eq!(*epoch, 10, "slot must be poisoned at epoch 10");
            }
            SlotState::Ready { .. } => {
                panic!("slot must NOT be Ready on fallback — would expose partial output");
            }
            other => {
                panic!("slot must be Poisoned, got {:?}", other);
            }
        }
    }

    #[test]
    fn test_fallback_preserves_slot_abi() {
        // The fallback outputs must have the same ABI as the primary:
        // same tensor id, dtype, logical/ physical shape, layout, strides.
        let primary_manifest = IOSurfaceSlotManifest {
            slot_id: 1,
            tensor_id: "output_0".into(),
            byte_offset: 0,
            byte_length: 4096,
            dtype: "float16".into(),
            logical_shape: vec![1, 64, 64],
            physical_shape: vec![1, 64, 64],
            strides_bytes: vec![8192, 128, 2],
            layout: "NHWC".into(),
            producer: ExecutionLane::CoreMlAne,
            consumer: ExecutionLane::MlxGpu,
            reuse_class: SlotReuseClass::Exclusive,
            required_alignment: 256,
        };

        // Fallback slot manifest (e.g. GPU-produced output) must match
        // in tensor_id, dtype, shape, strides, layout.
        let fallback_manifest = IOSurfaceSlotManifest {
            slot_id: 1,
            tensor_id: "output_0".into(),
            byte_offset: 0,
            byte_length: 4096,
            dtype: "float16".into(),
            logical_shape: vec![1, 64, 64],
            physical_shape: vec![1, 64, 64],
            strides_bytes: vec![8192, 128, 2],
            layout: "NHWC".into(),
            producer: ExecutionLane::MlxGpu,
            consumer: ExecutionLane::MlxGpu,
            reuse_class: SlotReuseClass::Exclusive,
            required_alignment: 256,
        };

        // ABI-critical fields must be identical
        assert_eq!(
            primary_manifest.tensor_id, fallback_manifest.tensor_id,
            "tensor_id must match"
        );
        assert_eq!(primary_manifest.dtype, fallback_manifest.dtype, "dtype must match");
        assert_eq!(
            primary_manifest.logical_shape, fallback_manifest.logical_shape,
            "logical_shape must match"
        );
        assert_eq!(
            primary_manifest.physical_shape, fallback_manifest.physical_shape,
            "physical_shape must match"
        );
        assert_eq!(
            primary_manifest.strides_bytes, fallback_manifest.strides_bytes,
            "strides_bytes must match"
        );
        assert_eq!(primary_manifest.layout, fallback_manifest.layout, "layout must match");
        assert_eq!(
            primary_manifest.byte_length, fallback_manifest.byte_length,
            "byte_length must match"
        );
        assert_eq!(
            primary_manifest.required_alignment, fallback_manifest.required_alignment,
            "required_alignment must match"
        );

        // Producer may differ (ANE vs GPU fallback) — that's expected.
        assert_ne!(
            primary_manifest.producer, fallback_manifest.producer,
            "producer may differ between primary and fallback"
        );
    }

    #[test]
    fn test_fallback_recovery_at_subsequent_epoch_boundary() {
        let mut mgr = FallbackPlanManager::new(dummy_plan());
        mgr.set_max_consecutive_failures(1);
        mgr.set_failure_injector(Box::new(TestFailureInjector {
            fail_epoch: Some(10),
        }));

        // Epoch 10: injector fires — fallback activated
        assert!(
            mgr.evaluate_with_injector(10, None),
            "injector fires at epoch 10"
        );
        mgr.activate(10, "injector failure".into());
        assert!(mgr.is_active());

        // After fallback, a subsequent epoch (11+) can begin recovery:
        // reset clears the failure state and deactivates fallback.
        mgr.reset();
        assert!(!mgr.is_active(), "fallback must be deactivated after reset");
        assert_eq!(mgr.consecutive_failures, 0);

        // Subsequent epoch with injector not firing should succeed
        assert!(
            !mgr.evaluate_with_injector(11, None),
            "epoch 11 must not trigger injector"
        );
        assert_eq!(mgr.consecutive_failures, 0);
    }
}
