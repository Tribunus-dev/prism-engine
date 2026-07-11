use crate::ecs::constitutional::command::{DomainEvent, EffectKind, EffectOutcome, EffectRequest};
use crate::ecs::constitutional::types::*;
use crate::ecs::constitutional::world_txn::{CommittedEpoch, WorldTxn, WorldTxnError};
use crate::ecs::CompWorld;
use serde::{Deserialize, Serialize};

/// Format a `[u8; 32]` digest as a lowercase hex string.
fn hex_str(bytes: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

/// Command to load and validate an artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoadArtifactCommand {
    pub id: MessageId,
    pub artifact_path: String,
    pub expected_digest: [u8; 32],
}

impl LoadArtifactCommand {
    /// Create the effect request for loading an artifact from filesystem.
    pub fn to_effect_request(&self) -> EffectRequest {
        EffectRequest {
            id: MessageId::compute(format!("load:{}", self.artifact_path).as_bytes()),
            kind: EffectKind::LoadFile,
            params: serde_json::json!({
                "path": self.artifact_path,
            }),
        }
    }

    /// Validate the outcome, spawn the artifact entity, emit a domain event, and commit.
    ///
    /// Returns the committed epoch and the emitted domain event.
    pub fn execute(
        self,
        world: &mut CompWorld,
        outcome: EffectOutcome,
    ) -> Result<(CommittedEpoch, DomainEvent), ArtifactError> {
        // 1. Validate the outcome
        if !outcome.success {
            return Err(ArtifactError::EffectFailed(outcome.output.to_string()));
        }

        // 2. Create the artifact entity outside the txn (WorldTxn does not support
        //    entity spawning yet — that comes with the executable schema registry).
        let artifact_entity = world.spawn_entity(crate::ecs::EntityKind::Artifact);

        // 3. Build a transaction that records the domain event
        let mut txn = WorldTxn::new(world);
        txn.emit_event(DomainEvent {
            id: self.id,
            kind: "artifact_loaded".to_string(),
            entity_id: None,
            payload: serde_json::json!({
                "artifact_path": self.artifact_path,
                "digest": hex_str(&self.expected_digest),
                "entity_id": artifact_entity.0,
            }),
        });

        // 4. Commit — advances the epoch and moves events into world.committed_events
        let epoch = world.transit(txn).map_err(ArtifactError::CommitFailed)?;
        let events = world.last_committed_events().to_vec();
        Ok((epoch, events.last().unwrap().clone()))
    }
}

/// Artifact ingestion errors.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ArtifactError {
    #[error("artifact effect failed: {0}")]
    EffectFailed(String),
    #[error("digest mismatch")]
    DigestMismatch,
    #[error("commit failed: {0}")]
    CommitFailed(WorldTxnError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::constitutional::persistence::{EventLogEntry, EventStore, InMemoryEventStore};
    use crate::ecs::EntityKind;

    // ── test_artifact_slice_success ──────────────────────────────────────────
    //
    // Happy path: command → outcome → txn → event → commit

    #[test]
    fn test_artifact_slice_success() {
        let mut world = CompWorld::new();
        let id = MessageId::compute(b"artifact-1");
        let path = "/models/test.cimage".to_string();
        let digest = [0u8; 32];

        let cmd = LoadArtifactCommand {
            id,
            artifact_path: path.clone(),
            expected_digest: digest,
        };

        let outcome = EffectOutcome {
            id,
            request_id: id,
            success: true,
            output: serde_json::json!({"bytes": 4096}),
        };

        let prev_epoch = world.current_epoch();
        let (epoch, event) = cmd.execute(&mut world, outcome).unwrap();

        // Epoch advanced
        assert!(epoch.0 > prev_epoch);
        assert_eq!(world.current_epoch(), epoch.0);

        // Event committed
        let events = world.last_committed_events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].kind, "artifact_loaded");
        assert_eq!(events[0].id, id);

        // Entity created
        let entity_id = event.payload["entity_id"].as_u64().unwrap();
        let entity = crate::ecs::CompEntity(entity_id);
        assert!(world.has_entity(entity));
        assert_eq!(world.entity_kind(entity), Some(EntityKind::Artifact));
    }

    // ── test_artifact_slice_effect_failure ───────────────────────────────────
    //
    // Effect fails → error returned, epoch NOT advanced, no events committed.

    #[test]
    fn test_artifact_slice_effect_failure() {
        let mut world = CompWorld::new();
        let id = MessageId::compute(b"artifact-fail");
        let path = "/models/missing.cimage".to_string();
        let digest = [0u8; 32];

        let cmd = LoadArtifactCommand {
            id,
            artifact_path: path.clone(),
            expected_digest: digest,
        };

        let outcome = EffectOutcome {
            id,
            request_id: id,
            success: false,
            output: serde_json::json!({"error": "file not found"}),
        };

        let prev_epoch = world.current_epoch();
        let err = cmd.execute(&mut world, outcome).unwrap_err();

        assert!(matches!(err, ArtifactError::EffectFailed(_)));
        assert_eq!(
            err.to_string(),
            "artifact effect failed: {\"error\":\"file not found\"}"
        );

        // Epoch must NOT advance
        assert_eq!(world.current_epoch(), prev_epoch);
        assert!(world.last_committed_events().is_empty());
    }

    // ── test_artifact_slice_replay_no_file ───────────────────────────────────
    //
    // Simulate replay from event store without the source file:
    //   1. Execute command → store domain event in InMemoryEventStore
    //   2. Create a new CompWorld (restart)
    //   3. Replay the event → reconstruct artifact entity

    #[test]
    fn test_artifact_slice_replay_no_file() {
        // --- Phase 1: Execute and persist ---
        let mut world = CompWorld::new();
        let id = MessageId::compute(b"artifact-replay");
        let path = "/models/persistent.cimage".to_string();
        let digest = [0xde; 32];

        let cmd = LoadArtifactCommand {
            id,
            artifact_path: path.clone(),
            expected_digest: digest,
        };
        let outcome = EffectOutcome {
            id,
            request_id: id,
            success: true,
            output: serde_json::json!({"bytes": 8192}),
        };
        let (_epoch, event) = cmd.execute(&mut world, outcome).unwrap();
        let original_entity_id = event.payload["entity_id"].as_u64().unwrap();

        // Store the event
        let mut store = InMemoryEventStore::new();
        store
            .append_events(
                _epoch.0,
                &[EventLogEntry {
                    epoch: _epoch.0,
                    sequence: 0,
                    event: event.clone(),
                    world_digest: [0u8; 32],
                }],
            )
            .unwrap();

        // --- Phase 2: Restart and replay ---
        let mut replay_world = CompWorld::new();
        let stored_events = store.get_events_from(replay_world.current_epoch());
        for entry in &stored_events {
            let kind = &entry.event.kind[..];
            if kind == "artifact_loaded" {
                let entity = replay_world.spawn_entity(EntityKind::Artifact);
                // The replayed entity gets a fresh id, but the kind is what matters
                assert_eq!(replay_world.entity_kind(entity), Some(EntityKind::Artifact));
            }
        }

        // The replayed world has one artifact entity
        let artifacts = replay_world.entities_of_kind(EntityKind::Artifact);
        assert_eq!(artifacts.len(), 1);
        assert!(replay_world.has_entity(artifacts[0]));

        // The original event payload is recoverable
        let replayed_path = stored_events[0].event.payload["artifact_path"]
            .as_str()
            .unwrap()
            .to_string();
        assert_eq!(replayed_path, path);
    }
}
