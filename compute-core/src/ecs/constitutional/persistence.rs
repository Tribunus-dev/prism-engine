use crate::ecs::constitutional::command::DomainEvent;
use crate::ecs::constitutional::types::*;
use serde::{Deserialize, Serialize};

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
