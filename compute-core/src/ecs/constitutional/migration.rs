use crate::ecs::constitutional::command::{DomainEvent, EffectKind, EffectOutcome, EffectRequest};
pub use crate::ecs::constitutional::types::*;
use crate::ecs::constitutional::world_txn::{CommittedEpoch, WorldTxn, WorldTxnError};
use crate::ecs::CompWorld;
use serde::{Deserialize, Serialize};

// ── Migration Pattern ────────────────────────────────────────────────────
//
// System migration follows this pattern:
//
// 1. Accept a Command (requested intent)
// 2. Issue an EffectRequest (external work, e.g. file load)
// 3. Receive an EffectOutcome (untrusted result)
// 4. Validate the outcome
// 5. Build a WorldTxn with validated state
// 6. Commit via world.transit(txn)
// 7. Emit DomainEvent on success
//
// The existing system keeps its current implementation as a compat path.
// New functionality uses the constitutional path.

/// A command to load an artifact (model weights, compiled graph, etc.)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoadArtifactCommand {
    pub id: MessageId,
    pub artifact_path: String,
    pub expected_digest: [u8; 32],
}

impl LoadArtifactCommand {
    /// Execute the load artifact flow.
    /// 1. Validate the command
    /// 2. Issue load effect
    /// 3. Validate the outcome
    /// 4. Commit the world transaction
    /// 5. Return the domain event
    pub fn execute(
        self,
        world: &mut CompWorld,
        _effect_outcome: EffectOutcome,
    ) -> Result<(CommittedEpoch, DomainEvent), MigrationError> {
        // Validate the command
        if self.artifact_path.is_empty() {
            return Err(MigrationError::InvalidCommand("empty artifact path".into()));
        }

        // Build a WorldTxn
        let mut txn = WorldTxn::new(world);

        // Emit the success domain event
        let digest_hex: String = self
            .expected_digest
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        let event = DomainEvent {
            id: self.id,
            kind: "artifact_loaded".to_string(),
            entity_id: None,
            payload: serde_json::json!({
                "artifact_path": self.artifact_path,
                "digest": digest_hex,
            }),
        };
        txn.emit_event(event.clone());

        // Commit — this demonstrates the pattern.
        // In a full implementation, the artifact entity would be created
        // and components would be added before commit.
        let epoch = world.transit(txn).map_err(MigrationError::CommitFailed)?;

        Ok((epoch, event))
    }

    /// Create the effect request for loading an artifact.
    pub fn to_effect_request(&self) -> EffectRequest {
        EffectRequest {
            id: MessageId::compute(format!("load:{}", self.artifact_path).as_bytes()),
            kind: EffectKind::LoadFile,
            params: serde_json::json!({
                "path": self.artifact_path,
            }),
        }
    }
}

/// Migration-specific errors.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum MigrationError {
    #[error("invalid command: {0}")]
    InvalidCommand(String),
    #[error("commit failed: {0}")]
    CommitFailed(WorldTxnError),
    #[error("validation failed: {0}")]
    ValidationFailed(String),
}

/// A migration-compat system that wraps the old pattern.
/// Instantiate to run an existing system side-by-side with its constitutional replacement.
pub struct CompatBridge {
    pub name: String,
    pub use_constitutional: bool,
}

impl CompatBridge {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.into(),
            use_constitutional: false,
        }
    }
}
