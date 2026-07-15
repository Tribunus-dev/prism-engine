use crate::ecs::constitutional::command::{DomainEvent, EffectOutcome, EffectRequest};
use crate::ecs::constitutional::lifecycle::ArtifactLifecycle;
use crate::ecs::constitutional::schema::SchemaRegistry;
use crate::ecs::constitutional::types::*;
use crate::ecs::constitutional::world_txn::{
    ClassifiedComponent, CommittedEpoch, DurableClass, DurableComponent, WorldTxn, WorldTxnError,
};
use crate::ecs::World;


use serde::{Deserialize, Serialize};

// ── Component Types ───────────────────────────────────────────────────────

/// Component: artifact file path.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactPath(pub String);
impl crate::ecs::Component for ArtifactPath {}
impl ClassifiedComponent for ArtifactPath {
    type Class = DurableClass;
}
impl DurableComponent for ArtifactPath {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.artifact",
        id: 2,
        version: 1,
    };
}

/// Component: artifact content digest (blake3 / sha-256).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactDigest(pub [u8; 32]);
impl crate::ecs::Component for ArtifactDigest {}
impl ClassifiedComponent for ArtifactDigest {
    type Class = DurableClass;
}
impl DurableComponent for ArtifactDigest {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.artifact",
        id: 3,
        version: 1,
    };
}

/// Component: artifact metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactMetadata {
    pub length: u64,
    pub path: String,
}
impl crate::ecs::Component for ArtifactMetadata {}
impl ClassifiedComponent for ArtifactMetadata {
    type Class = DurableClass;
}
impl DurableComponent for ArtifactMetadata {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.artifact",
        id: 4,
        version: 1,
    };
}

impl crate::ecs::Component for ArtifactLifecycle {}
impl ClassifiedComponent for ArtifactLifecycle {
    type Class = DurableClass;
}
impl DurableComponent for ArtifactLifecycle {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.artifact",
        id: 1,
        version: 1,
    };
}

// ── Format helpers ────────────────────────────────────────────────────────

/// Format a `[u8; 32]` digest as a lowercase hex string.
fn hex_str(bytes: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

// ── Command ───────────────────────────────────────────────────────────────

/// Command to load and validate an artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoadArtifactCommand {
    pub id: MessageId,
    pub artifact_path: String,
    pub expected_digest: Option<[u8; 32]>,
}

impl LoadArtifactCommand {
    /// Create the effect request for loading an artifact from filesystem.
    pub fn to_effect_request(&self) -> EffectRequest {
        EffectRequest {
            id: MessageId::compute(format!("load:{}", self.artifact_path).as_bytes()),
            kind: crate::ecs::constitutional::command::EffectKind::LoadFile,
            params: serde_json::json!({"path": self.artifact_path}),
        }
    }

    /// Execute the artifact load: validate outcome, spawn entity transactionally,
    /// attach components, emit domain event, commit.
    ///
    /// Returns the committed epoch and the emitted domain event.
    pub fn execute(
        self,
        world: &mut World,
        schema_registry: &SchemaRegistry,
        outcome: EffectOutcome,
    ) -> Result<(CommittedEpoch, DomainEvent), ArtifactError> {
        // 1. Validate outcome: must succeed and be correlated with the request
        if !outcome.success {
            return Err(ArtifactError::EffectFailed(
                outcome
                    .output
                    .get("error")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string(),
            ));
        }

        let expected_request_id = self.to_effect_request().id;
        if outcome.request_id != expected_request_id {
            return Err(ArtifactError::RequestMismatch {
                expected: expected_request_id,
                got: outcome.request_id,
            });
        }

        // 2. Extract and validate content digest
        let observed_digest: [u8; 32] = {
            let digest_str = outcome
                .output
                .get("digest")
                .and_then(|v| v.as_str())
                .ok_or(ArtifactError::MissingDigest)?;
            let bytes = digest_str.as_bytes();
            if bytes.len() != 64 {
                return Err(ArtifactError::InvalidDigestEncoding(
                    "digest must be 64 hex characters".into(),
                ));
            }
            let mut arr = [0u8; 32];
            for i in 0..32 {
                if !bytes[2 * i].is_ascii_hexdigit() || !bytes[2 * i + 1].is_ascii_hexdigit() {
                    return Err(ArtifactError::InvalidDigestEncoding(format!(
                        "invalid hex character at position {}",
                        2 * i
                    )));
                }
                let high = hex_char(bytes[2 * i]);
                let low = hex_char(bytes[2 * i + 1]);
                if high > 15 || low > 15 {
                    return Err(ArtifactError::InvalidDigestEncoding(format!(
                        "invalid hex at position {}",
                        2 * i
                    )));
                }
                arr[i] = high << 4 | low;
            }
            arr
        };

        if let Some(expected) = self.expected_digest {
            if observed_digest != expected {
                return Err(ArtifactError::DigestMismatch {
                    expected,
                    got: observed_digest,
                });
            }
        }

        // 3. Reserve entity ID (transactional — spawn happens inside commit)
        let entity_id = WorldTxn::next_entity_id(world);

        // 4. Validate all schemas are registered for the correct types
        schema_registry
            .verify_type::<ArtifactLifecycle>(crate::ecs::constitutional::types::ComponentSchemaId(
                1,
            ))
            .map_err(|e| ArtifactError::SchemaError(e))?;
        schema_registry
            .verify_type::<ArtifactPath>(crate::ecs::constitutional::types::ComponentSchemaId(2))
            .map_err(|e| ArtifactError::SchemaError(e))?;
        schema_registry
            .verify_type::<ArtifactDigest>(crate::ecs::constitutional::types::ComponentSchemaId(3))
            .map_err(|e| ArtifactError::SchemaError(e))?;
        schema_registry
            .verify_type::<ArtifactMetadata>(crate::ecs::constitutional::types::ComponentSchemaId(
                4,
            ))
            .map_err(|e| ArtifactError::SchemaError(e))?;

        // 5. Build transaction with spawn + components + event
        let mut txn = WorldTxn::new(world);

        txn.stage_spawn(entity_id, crate::ecs::EntityKind::Artifact);
        txn.put_durable(entity_id, ArtifactLifecycle::Loaded);
        txn.put_durable(entity_id, ArtifactPath(self.artifact_path.clone()));
        txn.put_durable(entity_id, ArtifactDigest(observed_digest));

        let file_length: u64 = outcome
            .output
            .get("length")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        txn.put_durable(
            entity_id,
            ArtifactMetadata {
                length: file_length,
                path: self.artifact_path.clone(),
            },
        );

        let event = DomainEvent {
            id: self.id,
            kind: "artifact_loaded".to_string(),
            entity_id: Some(crate::ecs::constitutional::types::EntityKindId(entity_id)),
            payload: serde_json::json!({
                "artifact_path": self.artifact_path,
                "expected_digest": hex_str(&self.expected_digest.unwrap_or([0u8; 32])),
                "observed_digest": hex_str(&observed_digest),
                "file_length": file_length,
                "entity_type": "artifact",
            }),
        };
        txn.emit_event(event.clone());

        // 6. Commit (entity spawn + components + event happen atomically)
        let epoch = world.transit(txn).map_err(ArtifactError::CommitFailed)?;

        Ok((epoch, event))
    }
}

// ── Errors ────────────────────────────────────────────────────────────────

// ── Replay ────────────────────────────────────────────────────────────────

/// Replay an `artifact_loaded` event to reconstruct an artifact entity.
///
/// This function uses `CompEntity` for entity identity internally.
/// The canonical entity type [`Entity`](crate::ecs::Entity) `(u64, u32)`
/// is preferred for new code outside this module.
pub fn replay_artifact_loaded(
    world: &mut World,
    event: &DomainEvent,
) -> Result<(CommittedEpoch, u64), ArtifactError> {
    let entity_id = event.entity_id.ok_or(ArtifactError::MissingDigest)?.0;
    let path = event
        .payload
        .get("artifact_path")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let mut txn = WorldTxn::new(world);
    if !world.has_entity(crate::ecs::CompEntity(entity_id)) {
        txn.stage_spawn(entity_id, crate::ecs::EntityKind::Artifact);
    }
    let file_length = event
        .payload
        .get("file_length")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    txn.add_component(
        entity_id,
        ComponentSchemaId(2),
        SchemaVersion(1),
        crate::ecs::constitutional::artifact::ArtifactPath(path.to_string()),
    );
    txn.add_component(
        entity_id,
        ComponentSchemaId(1),
        SchemaVersion(1),
        crate::ecs::constitutional::artifact::ArtifactLifecycle::Loaded,
    );
    txn.add_component(
        entity_id,
        ComponentSchemaId(4),
        SchemaVersion(1),
        crate::ecs::constitutional::artifact::ArtifactMetadata {
            length: file_length,
            path: path.to_string(),
        },
    );
    let epoch = world.transit(txn).map_err(ArtifactError::CommitFailed)?;
    Ok((epoch, entity_id))
}

/// Artifact ingestion errors.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ArtifactError {
    #[error("artifact effect failed: {0}")]
    EffectFailed(String),
    #[error("request ID mismatch")]
    RequestMismatch { expected: MessageId, got: MessageId },
    #[error("digest mismatch")]
    DigestMismatch { expected: [u8; 32], got: [u8; 32] },
    #[error("missing content digest in effect outcome")]
    MissingDigest,
    #[error("invalid digest encoding: {0}")]
    InvalidDigestEncoding(String),
    #[error("schema error: {0}")]
    SchemaError(String),
    #[error("commit failed: {0}")]
    CommitFailed(WorldTxnError),
}

// ── Hex char helper ───────────────────────────────────────────────────────

/// Decode a single hex character (0-9, a-f, A-F) to its nybble value.
fn hex_char(c: u8) -> u8 {
    match c {
        b'0'..=b'9' => c - b'0',
        b'a'..=b'f' => c - b'a' + 10,
        b'A'..=b'F' => c - b'A' + 10,
        _ => 0xFF,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::constitutional::persistence::{
        EventLogEntry, EventStore, InMemoryEventStore, ReplayEngine,
    };
    use crate::ecs::constitutional::schema::ComponentDurability;
    use crate::ecs::EntityKind;

    // ── test_artifact_slice_success ──────────────────────────────────────────

    #[test]
    fn test_artifact_slice_success() {
        let mut world = World::new();
        let mut schema_registry = SchemaRegistry::new();
        schema_registry.register_for_type::<ArtifactLifecycle>(
            ComponentSchemaId(1),
            SchemaVersion(1),
            "artifact_lifecycle",
            "",
            ComponentDurability::default(),
        );
        schema_registry.register_for_type::<ArtifactPath>(
            ComponentSchemaId(2),
            SchemaVersion(1),
            "artifact_path",
            "",
            ComponentDurability::default(),
        );
        schema_registry.register_for_type::<ArtifactDigest>(
            ComponentSchemaId(3),
            SchemaVersion(1),
            "artifact_digest",
            "",
            ComponentDurability::default(),
        );
        schema_registry.register_for_type::<ArtifactMetadata>(
            ComponentSchemaId(4),
            SchemaVersion(1),
            "artifact_metadata",
            "",
            ComponentDurability::default(),
        );
        let id = MessageId::compute(b"artifact-1");
        let path = "/models/test.cimage".to_string();
        let digest = [0xab; 32];
        let digest_hex = hex_str(&digest);

        let cmd = LoadArtifactCommand {
            id,
            artifact_path: path.clone(),
            expected_digest: Some(digest),
        };

        let outcome = EffectOutcome {
            id,
            request_id: cmd.to_effect_request().id,
            success: true,
            output: serde_json::json!({
                "bytes": 4096,
                "digest": digest_hex,
                "length": 4096,
            }),
        };

        let prev_epoch = world.current_epoch();
        let (epoch, event) = cmd.execute(&mut world, &schema_registry, outcome).unwrap();

        // Epoch advanced
        assert!(epoch.0 > prev_epoch);
        assert_eq!(world.current_epoch(), epoch.0);

        // Event committed
        let events = world.last_committed_events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].kind, "artifact_loaded");
        assert_eq!(events[0].id, id);

        // Event payload has correct observed_digest and entity_type
        assert_eq!(
            event.payload["observed_digest"].as_str().unwrap(),
            digest_hex
        );
        assert_eq!(event.payload["entity_type"].as_str().unwrap(), "artifact");
        assert_eq!(event.payload["file_length"].as_u64().unwrap(), 4096);

        // Entity exists after commit
        let entity_id = event.entity_id.unwrap().0;
        let entity = crate::ecs::CompEntity(entity_id);
        assert!(world.has_entity(entity));
        assert_eq!(world.entity_kind(entity), Some(EntityKind::Artifact));

        // Components attached
        let lifecycle = world.get_component::<ArtifactLifecycle>(entity);
        assert_eq!(lifecycle, Some(&ArtifactLifecycle::Loaded));

        let stored_path = world.get_component::<ArtifactPath>(entity);
        assert_eq!(stored_path, Some(&ArtifactPath(path.clone())));

        let stored_digest = world.get_component::<ArtifactDigest>(entity);
        assert_eq!(stored_digest, Some(&ArtifactDigest(digest)));

        let stored_meta = world.get_component::<ArtifactMetadata>(entity);
        assert!(stored_meta.is_some());
        assert_eq!(stored_meta.unwrap().length, 4096);
    }

    // ── test_artifact_slice_effect_failure ───────────────────────────────────

    #[test]
    fn test_artifact_slice_effect_failure() {
        let mut world = World::new();
        let mut schema_registry = SchemaRegistry::new();
        schema_registry.register_for_type::<ArtifactLifecycle>(
            ComponentSchemaId(1),
            SchemaVersion(1),
            "artifact_lifecycle",
            "",
            ComponentDurability::default(),
        );
        schema_registry.register_for_type::<ArtifactPath>(
            ComponentSchemaId(2),
            SchemaVersion(1),
            "artifact_path",
            "",
            ComponentDurability::default(),
        );
        schema_registry.register_for_type::<ArtifactDigest>(
            ComponentSchemaId(3),
            SchemaVersion(1),
            "artifact_digest",
            "",
            ComponentDurability::default(),
        );
        schema_registry.register_for_type::<ArtifactMetadata>(
            ComponentSchemaId(4),
            SchemaVersion(1),
            "artifact_metadata",
            "",
            ComponentDurability::default(),
        );
        let id = MessageId::compute(b"artifact-fail");
        let path = "/models/missing.cimage".to_string();

        let cmd = LoadArtifactCommand {
            id,
            artifact_path: path.clone(),
            expected_digest: None,
        };

        let outcome = EffectOutcome {
            id,
            request_id: cmd.to_effect_request().id,
            success: false,
            output: serde_json::json!({"error": "file not found"}),
        };

        let prev_epoch = world.current_epoch();
        let err = cmd
            .execute(&mut world, &schema_registry, outcome)
            .unwrap_err();

        assert!(matches!(err, ArtifactError::EffectFailed(_)));
        assert_eq!(err.to_string(), "artifact effect failed: file not found");

        // Epoch must NOT advance
        assert_eq!(world.current_epoch(), prev_epoch);
        assert!(world.last_committed_events().is_empty());
    }

    // ── test_artifact_slice_digest_mismatch ──────────────────────────────────

    #[test]
    fn test_artifact_slice_digest_mismatch() {
        let mut world = World::new();
        let mut schema_registry = SchemaRegistry::new();
        schema_registry.register_for_type::<ArtifactLifecycle>(
            ComponentSchemaId(1),
            SchemaVersion(1),
            "artifact_lifecycle",
            "",
            ComponentDurability::default(),
        );
        schema_registry.register_for_type::<ArtifactPath>(
            ComponentSchemaId(2),
            SchemaVersion(1),
            "artifact_path",
            "",
            ComponentDurability::default(),
        );
        schema_registry.register_for_type::<ArtifactDigest>(
            ComponentSchemaId(3),
            SchemaVersion(1),
            "artifact_digest",
            "",
            ComponentDurability::default(),
        );
        schema_registry.register_for_type::<ArtifactMetadata>(
            ComponentSchemaId(4),
            SchemaVersion(1),
            "artifact_metadata",
            "",
            ComponentDurability::default(),
        );
        let id = MessageId::compute(b"artifact-digest");
        let path = "/models/tampered.cimage".to_string();
        let expected = [0x01; 32];
        let actual = [0x02; 32];

        let cmd = LoadArtifactCommand {
            id,
            artifact_path: path.clone(),
            expected_digest: Some(expected),
        };

        let outcome = EffectOutcome {
            id,
            request_id: cmd.to_effect_request().id,
            success: true,
            output: serde_json::json!({
                "bytes": 1024,
                "digest": hex_str(&actual),
            }),
        };

        let prev_epoch = world.current_epoch();
        let err = cmd
            .execute(&mut world, &schema_registry, outcome)
            .unwrap_err();

        assert!(matches!(err, ArtifactError::DigestMismatch { .. }));

        // Epoch must NOT advance on validation failure
        assert_eq!(world.current_epoch(), prev_epoch);
    }

    // ── test_artifact_slice_request_mismatch ─────────────────────────────────

    #[test]
    fn test_artifact_slice_request_mismatch() {
        let mut world = World::new();
        let mut schema_registry = SchemaRegistry::new();
        schema_registry.register_for_type::<ArtifactLifecycle>(
            ComponentSchemaId(1),
            SchemaVersion(1),
            "artifact_lifecycle",
            "",
            ComponentDurability::default(),
        );
        schema_registry.register_for_type::<ArtifactPath>(
            ComponentSchemaId(2),
            SchemaVersion(1),
            "artifact_path",
            "",
            ComponentDurability::default(),
        );
        schema_registry.register_for_type::<ArtifactDigest>(
            ComponentSchemaId(3),
            SchemaVersion(1),
            "artifact_digest",
            "",
            ComponentDurability::default(),
        );
        schema_registry.register_for_type::<ArtifactMetadata>(
            ComponentSchemaId(4),
            SchemaVersion(1),
            "artifact_metadata",
            "",
            ComponentDurability::default(),
        );
        let id = MessageId::compute(b"artifact-req");
        let path = "/models/req_mismatch.cimage".to_string();

        let cmd = LoadArtifactCommand {
            id,
            artifact_path: path.clone(),
            expected_digest: None,
        };

        // outcome has a different request_id than expected
        let wrong_request_id = MessageId::compute(b"some-other-request");
        let outcome = EffectOutcome {
            id,
            request_id: wrong_request_id,
            success: true,
            output: serde_json::json!({
                "bytes": 512,
                "digest": hex_str(&[0x00; 32]),
            }),
        };

        let prev_epoch = world.current_epoch();
        let err = cmd
            .execute(&mut world, &schema_registry, outcome)
            .unwrap_err();

        assert!(matches!(err, ArtifactError::RequestMismatch { .. }));
        assert_eq!(world.current_epoch(), prev_epoch);
    }

    // ── test_artifact_slice_replay_no_file ───────────────────────────────────

    #[test]
    fn test_artifact_slice_replay_no_file() {
        // --- Phase 1: Execute and persist ---
        let mut world = World::new();
        let mut schema_registry = SchemaRegistry::new();
        schema_registry.register_for_type::<ArtifactLifecycle>(
            ComponentSchemaId(1),
            SchemaVersion(1),
            "artifact_lifecycle",
            "",
            ComponentDurability::default(),
        );
        schema_registry.register_for_type::<ArtifactPath>(
            ComponentSchemaId(2),
            SchemaVersion(1),
            "artifact_path",
            "",
            ComponentDurability::default(),
        );
        schema_registry.register_for_type::<ArtifactDigest>(
            ComponentSchemaId(3),
            SchemaVersion(1),
            "artifact_digest",
            "",
            ComponentDurability::default(),
        );
        schema_registry.register_for_type::<ArtifactMetadata>(
            ComponentSchemaId(4),
            SchemaVersion(1),
            "artifact_metadata",
            "",
            ComponentDurability::default(),
        );
        let id = MessageId::compute(b"artifact-replay");
        let path = "/models/persistent.cimage".to_string();
        let digest = [0xde; 32];

        let cmd = LoadArtifactCommand {
            id,
            artifact_path: path.clone(),
            expected_digest: Some(digest),
        };
        let outcome = EffectOutcome {
            id,
            request_id: cmd.to_effect_request().id,
            success: true,
            output: serde_json::json!({
                "bytes": 8192,
                "digest": hex_str(&digest),
                "length": 8192,
            }),
        };
        let (epoch, event) = cmd.execute(&mut world, &schema_registry, outcome).unwrap();

        // Store the event
        let mut store = InMemoryEventStore::new();
        store
            .append_events(
                epoch.0,
                &[EventLogEntry {
                    epoch: epoch.0,
                    sequence: 0,
                    event: event.clone(),
                    world_digest: [0u8; 32],
                }],
            )
            .unwrap();

        // --- Phase 2: Replay via ReplayEngine ---
        let replay_result = ReplayEngine::replay(&store, WorldEpoch(1));
        assert_eq!(
            replay_result.events_replayed, 1,
            "should replay exactly one event"
        );
        assert_eq!(replay_result.last_epoch, epoch.0);

        // --- Phase 3: Create new world and reconstruct entity from event ---
        let mut replay_world = World::new();
        let stored_events = store.get_events_from(replay_world.current_epoch());

        for entry in &stored_events {
            let kind = &entry.event.kind[..];
            if kind == "artifact_loaded" {
                let entity_id = WorldTxn::next_entity_id(&replay_world);
                let mut txn = WorldTxn::new(&replay_world);
                txn.stage_spawn(entity_id, EntityKind::Artifact);
                replay_world.transit(txn).unwrap();
                let entity = crate::ecs::CompEntity(entity_id);
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

        // Observed digest survives in the event payload
        let replayed_digest = stored_events[0].event.payload["observed_digest"]
            .as_str()
            .unwrap()
            .to_string();
        assert_eq!(replayed_digest, hex_str(&digest));

        // entity_type survives
        let replayed_entity_type = stored_events[0].event.payload["entity_type"]
            .as_str()
            .unwrap()
            .to_string();
        assert_eq!(replayed_entity_type, "artifact");
    }
}
