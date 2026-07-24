//! Command buffer for structural World mutations.
//!
//! Systems emit structural commands (spawn, despawn, insert, remove) through
//! a `CommandWriter` that automatically stamps provenance (stage, system_id,
//! sequence number).  The scheduler drains and applies commands at stage
//! boundaries, ensuring deterministic ordering and an authoritative mutation
//! seam for the future append-only receipt ledger.

use crate::entity::Entity;
use crate::scheduling::error::CommandError;
use crate::scheduling::metadata::{Stage, SystemId};
use crate::Component;
use serde::Serialize;

// ---------------------------------------------------------------------------
// Command
// ---------------------------------------------------------------------------

/// A structural World mutation emitted by a system.
#[derive(Debug, Clone)]
pub enum Command {
    /// Spawn a new entity.
    Spawn,
    /// Despawn an existing entity, removing all its components.
    Despawn(Entity),
    /// Insert or replace a component on an entity.
    Insert {
        entity: Entity,
        component_type_name: &'static str,
        /// Type-erased payload bytes for ledger recording.
        payload_bytes: Vec<u8>,
    },
    /// Remove a component from an entity (leave the entity alive).
    Remove {
        entity: Entity,
        component_type_name: &'static str,
    },
}

// ---------------------------------------------------------------------------
// StampedCommand
// ---------------------------------------------------------------------------

/// A `Command` with scheduler-provenance metadata stamped automatically.
#[derive(Debug, Clone)]
pub struct StampedCommand {
    pub command: Command,
    pub stage: Stage,
    pub system_id: SystemId,
    pub sequence: u64,
}

// ---------------------------------------------------------------------------
// CommandWriter
// ---------------------------------------------------------------------------

/// Per-system writer that auto-stamps provenance onto structural commands.
///
/// Constructed by the scheduler at stage dispatch.  The system never supplies
/// `stage` or `system_id` directly — that would allow spoofing the mutation
/// ledger.
pub struct CommandWriter<'a> {
    buffer: &'a mut Vec<StampedCommand>,
    stage: Stage,
    system_id: SystemId,
    sequence: u64,
}

impl<'a> CommandWriter<'a> {
    /// Create a new writer bound to a buffer for the given stage and system.
    pub fn new(buffer: &'a mut Vec<StampedCommand>, stage: Stage, system_id: SystemId) -> Self {
        CommandWriter {
            buffer,
            stage,
            system_id,
            sequence: 0,
        }
    }

    /// Emit a `Command::Spawn`.
    pub fn spawn(&mut self) -> Result<(), CommandError> {
        self.buffer.push(StampedCommand {
            command: Command::Spawn,
            stage: self.stage,
            system_id: self.system_id,
            sequence: self.sequence,
        });
        self.sequence += 1;
        Ok(())
    }

    /// Emit a `Command::Despawn`.
    pub fn despawn(&mut self, entity: Entity) -> Result<(), CommandError> {
        self.buffer.push(StampedCommand {
            command: Command::Despawn(entity),
            stage: self.stage,
            system_id: self.system_id,
            sequence: self.sequence,
        });
        self.sequence += 1;
        Ok(())
    }

    /// Emit a `Command::Insert` with type-erased payload bytes.
    pub fn insert<T: Component + Serialize>(
        &mut self,
        entity: Entity,
        value: &T,
    ) -> Result<(), CommandError> {
        let payload_bytes =
            bincode::serialize(value).map_err(|e| CommandError::InvalidMutation {
                detail: format!("failed to serialize component: {e}"),
            })?;
        self.buffer.push(StampedCommand {
            command: Command::Insert {
                entity,
                component_type_name: std::any::type_name::<T>(),
                payload_bytes,
            },
            stage: self.stage,
            system_id: self.system_id,
            sequence: self.sequence,
        });
        self.sequence += 1;
        Ok(())
    }

    /// Emit a `Command::Remove`.
    pub fn remove<T: Component>(&mut self, entity: Entity) -> Result<(), CommandError> {
        self.buffer.push(StampedCommand {
            command: Command::Remove {
                entity,
                component_type_name: std::any::type_name::<T>(),
            },
            stage: self.stage,
            system_id: self.system_id,
            sequence: self.sequence,
        });
        self.sequence += 1;
        Ok(())
    }
}
