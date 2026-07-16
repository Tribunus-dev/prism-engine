use crate::ecs::constitutional::command::DomainEvent;
use crate::ecs::constitutional::types::*;
use crate::ecs::World;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single entry in the durable event log.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventLogEntry {
    pub epoch: WorldEpoch,
    pub sequence: u64,
    pub event: DomainEvent,
    pub world_digest: [u8; 32],
}

/// Metadata for a world snapshot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Snapshot {
    pub epoch: WorldEpoch,
    pub world_digest: [u8; 32],
    pub entity_count: u32,
    pub component_count: u32,
    pub created_at: Timestamp,
}

/// Projection checkpoint — where a projector has consumed up to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProjectionCheckpoint {
    pub last_epoch: WorldEpoch,
    pub last_sequence: u64,
}

/// Trait for the durable event store.
pub trait EventStore: Send + Sync {
    /// Append events atomically at the given epoch.
    fn append_events(&mut self, epoch: WorldEpoch, events: &[EventLogEntry]) -> Result<(), String>;
    /// Get all events from a starting epoch.
    fn get_events_from(&self, from_epoch: WorldEpoch) -> Vec<EventLogEntry>;
    /// Store a snapshot.
    fn store_snapshot(&mut self, snapshot: Snapshot) -> Result<(), String>;
    /// Get the latest snapshot at or before the given epoch.
    fn latest_snapshot(&self) -> Option<Snapshot>;
    /// Get total event count.
    fn event_count(&self) -> u64;
    /// Get the highest stored epoch.
    fn latest_epoch(&self) -> Option<WorldEpoch>;
}

/// Reconstructs world state from the event log.
pub struct ReplayEngine;

impl ReplayEngine {
    /// Replay events from the store into an existing world using a registry of appliers.
    pub fn replay_into(
        world: &mut World,
        store: &dyn EventStore,
        from_epoch: WorldEpoch,
        registry: &ReplayRegistry,
    ) -> Result<ReplayResult, String> {
        let events = store.get_events_from(from_epoch);
        let event_count = events.len() as u64;
        for entry in &events {
            registry.apply(world, &entry.event)?;
        }
        Ok(ReplayResult {
            events_replayed: event_count,
            last_epoch: events.last().map(|e| e.epoch).unwrap_or(from_epoch),
            final_digest: [0u8; 32],
        })
    }

    /// Replay events from the store starting at `from_epoch` and return reconstructed state.
    /// This is the recovery path — process restarts, replays the event log, rebuilds the world.
    pub fn replay(store: &dyn EventStore, from_epoch: WorldEpoch) -> ReplayResult {
        let events = store.get_events_from(from_epoch);
        let event_count = events.len() as u64;
        ReplayResult {
            events_replayed: event_count,
            last_epoch: events.last().map(|e| e.epoch).unwrap_or(from_epoch),
            final_digest: [0u8; 32], // computed from actual replay in full implementation
        }
    }
}

/// Function signature for replaying a single event.
///
/// Internally processes entity data via `World`. The canonical entity type
/// [`Entity`](crate::ecs::Entity) `(u64, u32)` is preferred for new replay
/// code outside the constitutional domain.
pub type ReplayApplier = fn(&mut World, &DomainEvent) -> Result<(), String>;

/// Registry mapping event kind strings to replay applier functions.
///
/// Internally dispatches to `ReplayApplier` functions that operate on
/// `CompEntity` handles. For new code, prefer the canonical
/// [`Entity`](crate::ecs::Entity) type.
pub struct ReplayRegistry {
    appliers: HashMap<String, ReplayApplier>,
}

impl ReplayRegistry {
    pub fn new() -> Self {
        Self {
            appliers: HashMap::new(),
        }
    }

    pub fn register(&mut self, event_kind: &str, applier: ReplayApplier) {
        self.appliers.insert(event_kind.to_string(), applier);
    }

    pub fn apply(&self, world: &mut World, event: &DomainEvent) -> Result<(), String> {
        let applier = self.appliers.get(&event.kind).ok_or_else(|| {
            format!(
                "no replay applier registered for event kind: {}",
                event.kind
            )
        })?;
        applier(world, event)
    }

    pub fn register_all() -> Self {
        let mut reg = Self::new();
        reg.register("artifact_loaded", |w, e| {
            crate::ecs::constitutional::artifact::replay_artifact_loaded(w, e)
                .map(|_| ())
                .map_err(|err| format!("{err}"))
        });
        reg.register("device_discovered", |w, e| {
            crate::ecs::constitutional::device::replay_device_discovered(w, e)
                .map(|_| ())
                .map_err(|err| format!("{err}"))
        });
        reg.register("model_deployed", |w, e| {
            crate::ecs::constitutional::residency::replay_model_deployed(w, e)
                .map(|_| ())
                .map_err(|err| format!("{err}"))
        });
        reg.register("session_admitted", |w, e| {
            crate::ecs::constitutional::session::replay_session_admitted(w, e)
                .map(|_| ())
                .map_err(|err| format!("{err}"))
        });
        reg.register("work_created", |w, e| {
            crate::ecs::constitutional::work::replay_work_created(w, e)
        });
        reg.register("compilation_job_created", |w, e| {
            crate::ecs::constitutional::compilation::replay_compilation_job_created(w, e)
                .map(|_| ())
                .map_err(|err| format!("{err}"))
        });
        reg.register("lease_acquired", |w, e| {
            crate::ecs::constitutional::execution::replay_lease_acquired(w, e)
                .map(|_| ())
                .map_err(|err| format!("{err}"))
        });
        reg.register("lease_completed", |w, e| {
            crate::ecs::constitutional::execution::replay_lease_completed(w, e)
                .map(|_| ())
                .map_err(|err| format!("{err}"))
        });
        reg.register("pipeline_created", |w, e| {
            crate::ecs::constitutional::multimodal::replay_pipeline_created(w, e)
                .map(|_| ())
                .map_err(|err| format!("{err}"))
        });
        reg.register("agent_run_created", |w, e| {
            crate::ecs::constitutional::agent_exec::replay_agent_run_created(w, e)
                .map(|_| ())
                .map_err(|err| format!("{err}"))
        });
        reg.register("peer_registered", |w, e| {
            crate::ecs::constitutional::distributed::replay_peer_registered(w, e)
                .map(|_| ())
                .map_err(|err| format!("{err}"))
        });
        reg.register("ingress_request_submitted", |w, e| {
            crate::ecs::constitutional::ingress::replay_ingress_request_submitted(w, e)
                .map(|_| ())
                .map_err(|err| format!("{err}"))
        });
        reg
    }
}

/// Result of replaying events.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplayResult {
    pub events_replayed: u64,
    pub last_epoch: WorldEpoch,
    pub final_digest: [u8; 32],
}

/// In-memory implementation of EventStore for testing.
#[derive(Debug, Clone)]
pub struct InMemoryEventStore {
    events: Vec<EventLogEntry>,
    snapshots: Vec<Snapshot>,
}

impl InMemoryEventStore {
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            snapshots: Vec::new(),
        }
    }
}

impl EventStore for InMemoryEventStore {
    fn append_events(
        &mut self,
        epoch: WorldEpoch,
        entries: &[EventLogEntry],
    ) -> Result<(), String> {
        for entry in entries {
            if entry.epoch != epoch {
                return Err(format!(
                    "epoch mismatch: entry {:?} != batch {:?}",
                    entry.epoch, epoch
                ));
            }
            self.events.push(entry.clone());
        }
        Ok(())
    }

    fn get_events_from(&self, from_epoch: WorldEpoch) -> Vec<EventLogEntry> {
        self.events
            .iter()
            .filter(|e| e.epoch >= from_epoch)
            .cloned()
            .collect()
    }

    fn store_snapshot(&mut self, snapshot: Snapshot) -> Result<(), String> {
        self.snapshots.push(snapshot);
        Ok(())
    }

    fn latest_snapshot(&self) -> Option<Snapshot> {
        self.snapshots.last().cloned()
    }

    fn event_count(&self) -> u64 {
        self.events.len() as u64
    }

    fn latest_epoch(&self) -> Option<WorldEpoch> {
        self.events.last().map(|e| e.epoch)
    }
}
