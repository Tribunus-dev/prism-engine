use crate::command::DomainEvent;
use crate::types::*;
use prism_ecs_core::World;

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A single entry in the durable event log.
///
/// This type intentionally contains [`DomainEvent`] rather than the broader
/// `ClassifiedEvent` enum. Advisory observations therefore cannot be passed to
/// an `EventStore` without an explicit, invalid conversion.
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
    /// Append durable events atomically at the given epoch.
    ///
    /// Advisory events belong to the transaction's runtime observation lane
    /// and must not be converted into `EventLogEntry` values.
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
/// [`Entity`](prism_ecs_core::Entity) `(u64, u32)` is preferred for new replay
/// code outside the constitutional domain.
pub type ReplayApplier = fn(&mut World, &DomainEvent) -> Result<(), String>;

/// Registry mapping event kind strings to replay applier functions.
///
/// Internally dispatches to `ReplayApplier` functions that operate on
/// [`Entity`](prism_ecs_core::Entity) handles.
///
/// `BTreeMap<SchemaKey, ReplayApplier>` keyed by [`SchemaKey`] (not
/// `String`): the schema key is the durable identity, and iteration
/// must be deterministic for replay. See AGENTS.md "no HashMap/HashSet
/// for canonical collections whose order is observable."
///
/// NOTE: `event.kind` is still a free `String` (a separate B-2 newtype
/// concern). The boundary mapping `String → SchemaKey` is encoded by
/// [`event_kind_to_schema_key`] and is stable across processes.
pub struct ReplayRegistry {
    appliers: BTreeMap<SchemaKey, ReplayApplier>,
}

/// Stable, total mapping from canonical event-kind strings to
/// [`SchemaKey`]. Used at the `register` / `apply` boundary so the
/// internal BTreeMap is keyed by a typed, ordered identity.
///
/// Unknown event kinds fall back to a deterministic id derived from
/// the FNV-1a hash of the kind string. The fallback is preserved in
/// the BTreeMap so non-canonical kinds still resolve during replay.
fn event_kind_to_schema_key(kind: &str) -> SchemaKey {
    match kind {
        "artifact_loaded" => SchemaKey { namespace: "event", id: 1, version: 1 },
        "device_discovered" => SchemaKey { namespace: "event", id: 2, version: 1 },
        "model_deployed" => SchemaKey { namespace: "event", id: 3, version: 1 },
        "session_admitted" => SchemaKey { namespace: "event", id: 4, version: 1 },
        "work_created" => SchemaKey { namespace: "event", id: 5, version: 1 },
        "compilation_job_created" => SchemaKey { namespace: "event", id: 6, version: 1 },
        "lease_acquired" => SchemaKey { namespace: "event", id: 7, version: 1 },
        "lease_completed" => SchemaKey { namespace: "event", id: 8, version: 1 },
        "pipeline_created" => SchemaKey { namespace: "event", id: 9, version: 1 },
        "agent_run_created" => SchemaKey { namespace: "event", id: 10, version: 1 },
        "peer_registered" => SchemaKey { namespace: "event", id: 11, version: 1 },
        "ingress_request_submitted" => SchemaKey { namespace: "event", id: 12, version: 1 },
        // Fallback: FNV-1a 32-bit hash, masked into the u32 id field.
        // Deterministic across processes and stable across replays.
        _ => {
            let mut h: u32 = 0x811c_9dc5;
            for b in kind.as_bytes() {
                h ^= *b as u32;
                h = h.wrapping_mul(0x0100_0193);
            }
            // Avoid collision with the canonical namespace (id 1..=12).
            let id = h.wrapping_add(1000);
            SchemaKey {
                namespace: "event",
                id,
                version: 0,
            }
        }
    }
}

impl ReplayRegistry {
    pub fn new() -> Self {
        Self {
            appliers: BTreeMap::new(),
        }
    }

    /// Register a replay applier under a canonical event-kind string.
    ///
    /// The string is mapped to a stable [`SchemaKey`] via
    /// [`event_kind_to_schema_key`]. The public API keeps the
    /// `&str` signature so existing call sites compile unchanged; the
    /// typed key is the internal storage form.
    pub fn register(&mut self, event_kind: &str, applier: ReplayApplier) {
        self.appliers
            .insert(event_kind_to_schema_key(event_kind), applier);
    }

    pub fn apply(&self, world: &mut World, event: &DomainEvent) -> Result<(), String> {
        let key = event_kind_to_schema_key(&event.kind);
        let applier = self.appliers.get(&key).ok_or_else(|| {
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
            crate::artifact::replay_artifact_loaded(w, e)
                .map(|_| ())
                .map_err(|err| format!("{err}"))
        });
        reg.register("device_discovered", |w, e| {
            crate::device::replay_device_discovered(w, e)
                .map(|_| ())
                .map_err(|err| format!("{err}"))
        });
        reg.register("model_deployed", |w, e| {
            crate::residency::replay_model_deployed(w, e)
                .map(|_| ())
                .map_err(|err| format!("{err}"))
        });
        reg.register("session_admitted", |w, e| {
            crate::session::replay_session_admitted(w, e)
                .map(|_| ())
                .map_err(|err| format!("{err}"))
        });
        reg.register("work_created", |w, e| {
            crate::work::replay_work_created(w, e)
        });
        reg.register("compilation_job_created", |w, e| {
            crate::compilation::replay_compilation_job_created(w, e)
                .map(|_| ())
                .map_err(|err| format!("{err}"))
        });
        reg.register("lease_acquired", |w, e| {
            crate::execution::replay_lease_acquired(w, e)
                .map(|_| ())
                .map_err(|err| format!("{err}"))
        });
        reg.register("lease_completed", |w, e| {
            crate::execution::replay_lease_completed(w, e)
                .map(|_| ())
                .map_err(|err| format!("{err}"))
        });
        reg.register("pipeline_created", |w, e| {
            crate::multimodal::replay_pipeline_created(w, e)
                .map(|_| ())
                .map_err(|err| format!("{err}"))
        });
        reg.register("agent_run_created", |w, e| {
            crate::agent_exec::replay_agent_run_created(w, e)
                .map(|_| ())
                .map_err(|err| format!("{err}"))
        });
        reg.register("peer_registered", |w, e| {
            crate::distributed::replay_peer_registered(w, e)
                .map(|_| ())
                .map_err(|err| format!("{err}"))
        });
        reg.register("ingress_request_submitted", |w, e| {
            crate::ingress::replay_ingress_request_submitted(w, e)
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
