use crate::work::{Prerequisite, WorkKind, WorkState};
use crate::types::*;
use serde::{Deserialize, Serialize};

/// Resource claim — what resources this work item needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ResourceClaim {
    pub memory_bytes: u64,
    pub compute_units: u32,
    pub priority: Priority,
}

/// Priority level for scheduling fairness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Priority {
    Low,
    Normal,
    High,
    Critical,
}

impl Priority {
    pub fn as_u8(&self) -> u8 {
        match self {
            Self::Low => 0,
            Self::Normal => 1,
            Self::High => 2,
            Self::Critical => 3,
        }
    }
}

// ── WorkItem ─────────────────────────────────────────────────────────────

/// A schedulable unit of work. ECS entities carry this as a component.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkItem {
    pub kind: WorkKind,
    pub target_entity: u64,
    pub prerequisites: Vec<Prerequisite>,
    pub priority: Priority,
    pub deadline: Option<Timestamp>,
    pub attempt: u32,
    pub cancellation_epoch: WorldEpoch,
    pub resource_claim: ResourceClaim,
    pub state: WorkState,
}

impl WorkItem {
    pub fn new(kind: WorkKind, target: u64) -> Self {
        Self {
            kind,
            target_entity: target,
            prerequisites: Vec::new(),
            priority: Priority::Normal,
            deadline: None,
            attempt: 0,
            cancellation_epoch: WorldEpoch(0),
            resource_claim: ResourceClaim {
                memory_bytes: 0,
                compute_units: 0,
                priority: Priority::Normal,
            },
            state: WorkState::Pending,
        }
    }

    /// Returns true if all prerequisites are satisfied (checked externally).
    pub fn is_ready(&self) -> bool {
        self.state == WorkState::Pending && self.prerequisites.is_empty()
    }
}

// ── WorkLease ─────────────────────────────────────────────────────────────

/// A leased work identity — returned by Scheduler::drain.
/// Contains enough info to execute the work without re-borrowing from the world.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkLease {
    pub work_entity: u64,
    pub kind: WorkKind,
    pub lease_generation: u64,
    pub attempt: u32,
    pub cancellation_epoch: WorldEpoch,
    pub expiry: Timestamp,
    pub resource_claim: ResourceClaim,
}

// ── Scheduler ────────────────────────────────────────────────────────────

/// In-memory scheduler index over work items.
/// Maintains readiness indexes by kind and priority.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scheduler {
    /// Readiness index: work_entity ids grouped by WorkKind
    ready_by_kind: std::collections::HashMap<WorkKind, Vec<u64>>,
    /// Total ready count
    ready_count: usize,
    /// Next lease generation counter
    lease_gen: u64,
}

impl Scheduler {
    pub fn new() -> Self {
        Self {
            ready_by_kind: std::collections::HashMap::new(),
            ready_count: 0,
            lease_gen: 0,
        }
    }

    /// Register a work item as pending (not yet ready).
    pub fn register_pending(&mut self, _entity: u64) {
        // Work item is pending; will be moved to ready when prerequisites are met
    }

    /// Transition a work item to ready.
    pub fn mark_ready(&mut self, entity: u64, kind: WorkKind) {
        self.ready_by_kind.entry(kind).or_default().push(entity);
        self.ready_count += 1;
    }

    /// Drain up to `max_items` ready work items, returning leases.
    pub fn drain(&mut self, max_items: usize) -> Vec<WorkLease> {
        let mut leases = Vec::new();
        for (_kind, entities) in self.ready_by_kind.iter_mut() {
            while leases.len() < max_items {
                match entities.pop() {
                    Some(entity) => {
                        self.lease_gen += 1;
                        leases.push(WorkLease {
                            work_entity: entity,
                            kind: *_kind,
                            lease_generation: self.lease_gen,
                            attempt: 0,
                            cancellation_epoch: WorldEpoch(0),
                            expiry: Timestamp::now(),
                            resource_claim: ResourceClaim {
                                memory_bytes: 0,
                                compute_units: 0,
                                priority: Priority::Normal,
                            },
                        });
                        self.ready_count -= 1;
                    }
                    None => break,
                }
            }
            if leases.len() >= max_items {
                break;
            }
        }
        leases
    }

    pub fn ready_count(&self) -> usize {
        self.ready_count
    }

    pub fn lease_generation(&self) -> u64 {
        self.lease_gen
    }
}
