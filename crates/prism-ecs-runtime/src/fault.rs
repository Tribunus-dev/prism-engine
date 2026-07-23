//! Deterministic fault-injection framework for runtime certification.
//!
//! Wraps production adapters with configurable failure points for testing
//! every crash boundary in the command lifecycle.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::ports::{
    Admission, AdmittedCommand, CommandStore, CommandWatermarks, CompletedCommand,
    LeaseCoordinator, RuntimeError, SnapshotStore, WorldSnapshot,
};

/// Points in the runtime where faults can be injected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FaultPoint {
    CommandAdmit,
    CommandComplete,
    CommandCompletedAfter,
    CommandUnresolved,
    SnapshotSave,
    SnapshotLoad,
    LeaseAcquire,
    LeaseRenew,
    LeaseRelease,
    CommandTransitionState,
    TickSave,
    ReplayApply,
}

/// How a fault manifests.
#[derive(Debug, Clone)]
pub enum FaultMode {
    ReturnError(String),
    Delay(std::time::Duration),
    CorruptPayload(String),
}

/// Configuration for deterministic fault injection.
///
/// Tracks per-point invocation counters with `AtomicU64` so the plan
/// can be shared across threads without external synchronisation.
/// `Clone` is deliberately *not* derived — atomic counters cannot be
/// cheaply cloned. Use [`FaultPlan::clone_empty`] when a fresh plan
/// with the same trigger schedule is needed.
#[derive(Debug, Default)]
pub struct FaultPlan {
    triggers: HashMap<FaultPoint, Vec<FaultMode>>,
    counts: HashMap<FaultPoint, AtomicU64>,
}

impl FaultPlan {
    /// Create an empty fault plan with no triggers.
    pub fn new() -> Self {
        Self {
            triggers: HashMap::new(),
            counts: HashMap::new(),
        }
    }

    /// Inject a fault at the given point on the N-th invocation (0-based).
    pub fn inject_at(&mut self, point: FaultPoint, mode: FaultMode, invocation: u64) {
        let entry = self.triggers.entry(point).or_default();
        // Extend vec to fit
        while entry.len() <= invocation as usize {
            entry.push(FaultMode::Delay(std::time::Duration::ZERO)); // no-op placeholder
        }
        entry[invocation as usize] = mode;
        // Ensure counts entry exists so check() increments correctly
        self.counts
            .entry(point)
            .or_insert_with(|| AtomicU64::new(0));
    }

    /// Check whether a fault should fire at the given point.
    ///
    /// Each call increments the invocation counter for `point`. When the
    /// counter matches a configured trigger, the corresponding fault mode
    /// is executed.
    pub fn check(&self, point: FaultPoint) -> Result<(), RuntimeError> {
        let count = self
            .counts
            .get(&point)
            .map(|c| c.fetch_add(1, Ordering::Relaxed))
            .unwrap_or(0);
        if let Some(modes) = self.triggers.get(&point) {
            if let Some(mode) = modes.get(count as usize) {
                match mode {
                    FaultMode::ReturnError(msg) => Err(RuntimeError::Journal(msg.clone())),
                    FaultMode::Delay(d) => {
                        std::thread::sleep(*d);
                        Ok(())
                    }
                    FaultMode::CorruptPayload(_) => {
                        Err(RuntimeError::Journal("payload corrupted by fault".into()))
                    }
                }
            } else {
                Ok(())
            }
        } else {
            Ok(())
        }
    }

    /// Return a copy of this plan with the same trigger schedule but fresh
    /// (zeroed) invocation counters.
    pub fn clone_empty(&self) -> Self {
        let mut cloned = Self::new();
        for (point, modes) in &self.triggers {
            cloned.triggers.insert(*point, modes.clone());
        }
        cloned
    }
}

/// Wraps a [`CommandStore`] with configurable fault injection.
pub struct FaultingCommandStore {
    inner: Box<dyn CommandStore>,
    plan: FaultPlan,
}

impl FaultingCommandStore {
    pub fn new(inner: Box<dyn CommandStore>) -> Self {
        Self {
            inner,
            plan: FaultPlan::new(),
        }
    }

    pub fn with_plan(mut self, plan: FaultPlan) -> Self {
        self.plan = plan;
        self
    }

    pub fn plan(&self) -> &FaultPlan {
        &self.plan
    }
}

impl CommandStore for FaultingCommandStore {
    fn admit(&self, key: &uuid::Uuid, envelope: &str) -> Result<Admission, RuntimeError> {
        self.plan.check(FaultPoint::CommandAdmit)?;
        self.inner.admit(key, envelope)
    }

    fn complete(&self, seq: u64, result: &str, epoch: u64) -> Result<(), RuntimeError> {
        self.plan.check(FaultPoint::CommandComplete)?;
        self.inner.complete(seq, result, epoch)
    }

    fn lookup(&self, key: &uuid::Uuid) -> Result<Option<String>, RuntimeError> {
        self.inner.lookup(key)
    }

    fn completed_after(&self, seq: u64) -> Result<Vec<CompletedCommand>, RuntimeError> {
        self.plan.check(FaultPoint::CommandCompletedAfter)?;
        self.inner.completed_after(seq)
    }

    fn unresolved(&self) -> Result<Vec<AdmittedCommand>, RuntimeError> {
        self.plan.check(FaultPoint::CommandUnresolved)?;
        self.inner.unresolved()
    }

    fn high_water_marks(&self) -> Result<CommandWatermarks, RuntimeError> {
        self.inner.high_water_marks()
    }

    fn transition_state(&self, sequence: u64, target_state: &str) -> Result<(), RuntimeError> {
        self.plan.check(FaultPoint::CommandTransitionState)?;
        self.inner.transition_state(sequence, target_state)
    }
}

/// Wraps a [`SnapshotStore`] with configurable fault injection.
pub struct FaultingSnapshotStore {
    inner: Box<dyn SnapshotStore>,
    plan: FaultPlan,
}

impl FaultingSnapshotStore {
    pub fn new(inner: Box<dyn SnapshotStore>) -> Self {
        Self {
            inner,
            plan: FaultPlan::new(),
        }
    }

    pub fn plan(&self) -> &FaultPlan {
        &self.plan
    }
}

impl SnapshotStore for FaultingSnapshotStore {
    fn save(&self, snapshot: &WorldSnapshot) -> Result<(), RuntimeError> {
        self.plan.check(FaultPoint::SnapshotSave)?;
        self.inner.save(snapshot)
    }

    fn load_latest(&self) -> Result<Option<WorldSnapshot>, RuntimeError> {
        self.plan.check(FaultPoint::SnapshotLoad)?;
        self.inner.load_latest()
    }
}

/// Wraps a [`LeaseCoordinator`] with configurable fault injection.
pub struct FaultingLeaseCoordinator {
    inner: Box<dyn LeaseCoordinator>,
    plan: FaultPlan,
}

impl FaultingLeaseCoordinator {
    pub fn new(inner: Box<dyn LeaseCoordinator>) -> Self {
        Self {
            inner,
            plan: FaultPlan::new(),
        }
    }
}

impl LeaseCoordinator for FaultingLeaseCoordinator {
    fn acquire(&self, key: &str, ttl: u64) -> Result<bool, RuntimeError> {
        self.plan.check(FaultPoint::LeaseAcquire)?;
        self.inner.acquire(key, ttl)
    }

    fn renew(&self, key: &str, ttl: u64) -> Result<bool, RuntimeError> {
        self.plan.check(FaultPoint::LeaseRenew)?;
        self.inner.renew(key, ttl)
    }

    fn release(&self, key: &str) -> Result<(), RuntimeError> {
        self.plan.check(FaultPoint::LeaseRelease)?;
        self.inner.release(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_adapters::InMemoryCommandStore;

    #[test]
    fn test_fault_plan_injects_error() {
        let mut plan = FaultPlan::new();
        plan.inject_at(
            FaultPoint::CommandAdmit,
            FaultMode::ReturnError("injected fault".into()),
            0,
        );
        let inner = Box::new(InMemoryCommandStore::new());
        let store = FaultingCommandStore::new(inner).with_plan(plan);
        let key = uuid::Uuid::new_v4();
        let result = store.admit(&key, r##"{"test":true}"##);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("injected fault"),
            "expected injected fault error, got: {err}"
        );
    }

    #[test]
    fn test_fault_only_triggers_at_specified_invocation() {
        let mut plan = FaultPlan::new();
        plan.inject_at(
            FaultPoint::CommandAdmit,
            FaultMode::ReturnError("fail on third".into()),
            2,
        );
        let inner = Box::new(InMemoryCommandStore::new());
        let store = FaultingCommandStore::new(inner).with_plan(plan);
        let key1 = uuid::Uuid::new_v4();
        let key2 = uuid::Uuid::new_v4();
        let key3 = uuid::Uuid::new_v4();
        assert!(store.admit(&key1, "{}").is_ok()); // call 0
        assert!(store.admit(&key2, "{}").is_ok()); // call 1
        assert!(store.admit(&key3, "{}").is_err()); // call 2 = fault
    }
}
