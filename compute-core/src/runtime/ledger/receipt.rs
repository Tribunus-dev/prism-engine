//! SemanticReceipt trait — projects StampedCommand into SemanticCommandPayload.
//!
//! Each command variant maps to a typed semantic payload for the ledger.

use crate::runtime::ledger::entry::SemanticStampedCommand;
use crate::runtime::ledger::entry::SemanticCommandPayload;
use crate::runtime::ledger::error::LedgerProjectionError;
use crate::runtime::ledger::registry::ComponentTypeRegistry;
use crate::runtime::scheduling::command::{Command, StampedCommand};

/// Project a scheduler-stamped command into a semantic receipt entry.
///
/// The receipt representation is stable, separately serializable, and
/// independent of Rust memory layout or type-erased payload bytes.
pub trait SemanticReceipt {
    fn semantic_receipt(&self, registry: &ComponentTypeRegistry) -> Result<SemanticStampedCommand, LedgerProjectionError>;
}

impl SemanticReceipt for StampedCommand {
    fn semantic_receipt(&self, registry: &ComponentTypeRegistry) -> Result<SemanticStampedCommand, LedgerProjectionError> {
        let cmd = match &self.command {
            Command::Spawn => SemanticCommandPayload::EntitySpawned {
                entity_kind: "worker_request".to_string(),
            },
            Command::Despawn(entity) => SemanticCommandPayload::EntityDespawned {
                reason: format!("entity {} despawned", entity.0),
            },
            Command::Insert { entity: _, type_id, payload } => {
                registry.project(type_id, payload)?
            }
            Command::Remove { entity, .. } => {
                SemanticCommandPayload::EntityDespawned {
                    reason: format!("component removed from entity {}", entity.0),
                }
            }
        };

        Ok(SemanticStampedCommand {
            stage: self.stage,
            system_id: self.system_id,
            entity: self.entity.unwrap_or(crate::runtime::world::Entity(0)),
            entity_generation: None,
            sequence: self.sequence,
            command: cmd,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::ledger::registry::ComponentTypeRegistry;
    use crate::runtime::scheduling::command::CommandWriter;
    use crate::runtime::scheduling::metadata::{Stage, SystemId};
    use crate::runtime::world::World;

    #[test]
    fn spawn_projects_entity_spawned() {
        let mut _world = World::default();
        let mut buffer = Vec::new();
        let stage = Stage::Intake;
        let sys_id = SystemId(100);
        {
            let mut writer = CommandWriter::new(&mut buffer, stage, sys_id);
            writer.spawn().unwrap();
        }
        let stamped = &buffer[0];
        let registry = ComponentTypeRegistry::new();
        let receipt = stamped.semantic_receipt(&registry).unwrap();
        assert_eq!(receipt.stage, stage);
        assert_eq!(receipt.system_id, sys_id);
        assert_eq!(receipt.sequence, 0);
        match receipt.command {
            SemanticCommandPayload::EntitySpawned { .. } => {}
            other => panic!("expected EntitySpawned, got {other:?}"),
        }
    }

    #[test]
    fn despawn_projects_entity_despawned() {
        let mut world = World::default();
        let mut buffer = Vec::new();
        let stage = Stage::Maintenance;
        let sys_id = SystemId(101);
        let entity;
        {
            let mut writer = CommandWriter::new(&mut buffer, stage, sys_id);
            entity = world.spawn().unwrap();
            writer.despawn(entity).unwrap();
        }
        let stamped = &buffer[0];
        let registry = ComponentTypeRegistry::new();
        let receipt = stamped.semantic_receipt(&registry).unwrap();
        match receipt.command {
            SemanticCommandPayload::EntityDespawned { .. } => {}
            other => panic!("expected EntityDespawned, got {other:?}"),
        }
    }
}
