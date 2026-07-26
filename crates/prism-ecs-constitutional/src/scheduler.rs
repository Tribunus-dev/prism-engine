use crate::types::*;
use crate::work::{Prerequisite, WorkKind, WorkState};
use prism_ecs_core::Entity;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Resource claim — what resources this work item needs.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ResourceClaim {
    pub memory_bytes: u64,
    pub compute_units: u32,
    pub priority: Priority,
    /// Optional inference-specific hints (prompt tokens, max new tokens,
    /// KV cache configuration, deadline). Captured here so the B-2 typed
    /// boundary still carries the inference metadata that legacy
    /// `CreateWorkCommand` constructors passed as opaque JSON.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inference_hint: Option<InferenceHint>,
}

impl Default for ResourceClaim {
    fn default() -> Self {
        Self {
            memory_bytes: 0,
            compute_units: 0,
            priority: Priority::Normal,
            inference_hint: None,
        }
    }
}

/// Inference hints carried alongside a `ResourceClaim`. Mirrors the JSON
/// fields the legacy free-form `resource_claim: String` used to smuggle
/// into the executor; the B-2 refactor promotes them to a typed optional.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct InferenceHint {
    pub prompt_tokens: u32,
    pub max_new_tokens: u32,
    pub prefill_chunk_tokens: u32,
    pub kv_epoch: u64,
    pub kv_tokens: u32,
    pub kv_capacity_tokens: u32,
    pub deadline_ms: u64,
    pub priority: u32,
}

impl Default for InferenceHint {
    fn default() -> Self {
        Self {
            prompt_tokens: 0,
            max_new_tokens: 1,
            prefill_chunk_tokens: 1,
            kv_epoch: 0,
            kv_tokens: 0,
            kv_capacity_tokens: 0,
            deadline_ms: 0,
            priority: 0,
        }
    }
}

impl InferenceHint {
    /// Parse a JSON blob (the legacy `resource_claim: String` shape) into
    /// a typed hint. Unknown fields are ignored; missing fields use
    /// safe defaults. The `from_resource_claim` helper below is the
    /// inverse of the legacy `InferenceWorkMetadata::from_resource_claim`.
    pub fn from_json_str(s: &str) -> Self {
        #[derive(Debug, Deserialize, Default)]
        struct Raw {
            #[serde(default)]
            prompt_tokens: u32,
            #[serde(default)]
            max_new_tokens: u32,
            #[serde(default)]
            prefill_chunk_tokens: u32,
            #[serde(default)]
            kv_epoch: u64,
            #[serde(default)]
            kv_tokens: u32,
            #[serde(default)]
            kv_capacity_tokens: u32,
            #[serde(default)]
            deadline_ms: u64,
            #[serde(default)]
            priority: u32,
        }
        let raw: Raw = serde_json::from_str(s).unwrap_or_default();
        Self {
            prompt_tokens: raw.prompt_tokens,
            max_new_tokens: raw.max_new_tokens,
            prefill_chunk_tokens: raw.prefill_chunk_tokens,
            kv_epoch: raw.kv_epoch,
            kv_tokens: raw.kv_tokens,
            kv_capacity_tokens: raw.kv_capacity_tokens,
            deadline_ms: raw.deadline_ms,
            priority: raw.priority,
        }
    }
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
            resource_claim: ResourceClaim::default(),
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
    /// Readiness index: work entity handles grouped by WorkKind.
    ///
    /// `BTreeMap` (not `HashMap`): the per-kind iteration order is part
    /// of the canonical schedule and must be deterministic across
    /// processes for replay. See AGENTS.md "no HashMap/HashSet for
    /// canonical collections whose order is observable."
    ready_by_kind: BTreeMap<WorkKind, Vec<Entity>>,
    /// Total ready count
    ready_count: usize,
    /// Next lease generation counter
    lease_gen: u64,
}

impl Scheduler {
    pub fn new() -> Self {
        Self {
            ready_by_kind: BTreeMap::new(),
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
        // NOTE: the input is a raw `u64` entity id; the stored key is
        // the typed `Entity` newtype so the readiness index participates
        // in deterministic BTreeMap iteration. A typed handle is
        // preferred but is gated on B-2 (the cmd! macro newtype refactor).
        self.ready_by_kind
            .entry(kind)
            .or_default()
            .push(Entity::new(entity, 0));
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
                            work_entity: entity.id(),
                            kind: *_kind,
                            lease_generation: self.lease_gen,
                            attempt: 0,
                            cancellation_epoch: WorldEpoch(0),
                            expiry: Timestamp::now(),
                            resource_claim: ResourceClaim::default(),
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
