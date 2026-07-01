//! Command buffer for structural World mutations.
//!
//! Systems emit structural commands (spawn, despawn, insert, remove) through
//! a `CommandWriter` that automatically stamps provenance (stage, system_id,
//! sequence number).  The scheduler drains and applies commands at stage
//! boundaries, ensuring deterministic ordering and an authoritative mutation
//! seam for the future append-only receipt ledger.

use crate::runtime::scheduling::metadata::{Stage, SystemId};
use crate::runtime::scheduling::error::CommandError;
use crate::runtime::world::{Entity, Component};
use serde::Serialize;

// ---------------------------------------------------------------------------
// Command
// ---------------------------------------------------------------------------

/// A structural World mutation emitted by a system.
#[derive(Debug, Clone)]
pub enum Command {
    /// Spawn a new entity.  Returns the assigned Entity via the execution result.
    Spawn,
    /// Despawn an existing entity, removing all its components.
    Despawn(Entity),
    /// Insert or replace a component on an entity.
    Insert {
        /// Target entity.
        entity: Entity,
        /// Type-erased component value.
        /// The scheduler stores and applies it using the component's TypeId.
        type_id: std::any::TypeId,
        /// Opaque bytes of the component value.
        /// The scheduler knows how to deserialize based on type_id.
        payload: Vec<u8>,
    },
    /// Remove a component from an entity (leave the entity alive).
    Remove {
        /// Target entity.
        entity: Entity,
        /// TypeId of the component to remove.
        type_id: std::any::TypeId,
    },
}

// ---------------------------------------------------------------------------
// StampedCommand
// ---------------------------------------------------------------------------

/// A `Command` with scheduler-provenance metadata stamped automatically.
#[derive(Debug, Clone)]
pub struct StampedCommand {
    /// Stage in which this command was emitted.
    pub stage: Stage,
    /// Originating system.
    pub system_id: SystemId,
    /// Target entity (None for Spawn, which produces an entity).
    pub entity: Option<Entity>,
    /// Monotonically increasing sequence number within the stage.
    pub sequence: u64,
    /// The structural mutation.
    pub command: Command,
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
    pub fn new(
        buffer: &'a mut Vec<StampedCommand>,
        stage: Stage,
        system_id: SystemId,
    ) -> Self {
        Self {
            buffer,
            stage,
            system_id,
            sequence: 0,
        }
    }

    /// Emit a spawn command.
    ///
    /// The scheduler assigns the entity ID when applying the buffer at the
    /// stage barrier.  Returns `Ok(())` when the command fits.
    pub fn spawn(&mut self) -> Result<(), CommandError> {
        self.buffer.push(StampedCommand {
            stage: self.stage,
            system_id: self.system_id,
            entity: None,
            sequence: self.sequence,
            command: Command::Spawn,
        });
        self.sequence += 1;
        Ok(())
    }

    /// Emit a despawn command for an existing entity.
    pub fn despawn(&mut self, entity: Entity) -> Result<(), CommandError> {
        self.buffer.push(StampedCommand {
            stage: self.stage,
            system_id: self.system_id,
            entity: Some(entity),
            sequence: self.sequence,
            command: Command::Despawn(entity),
        });
        self.sequence += 1;
        Ok(())
    }

    /// Insert or replace a component on an entity.
    pub fn insert<T: Component + Serialize>(
        &mut self,
        entity: Entity,
        component: T,
    ) -> Result<(), CommandError> {
        let payload = serde_json::to_vec(&component)
            .map_err(|e| CommandError::SerializationError(e.to_string()))?;

        self.buffer.push(StampedCommand {
            stage: self.stage,
            system_id: self.system_id,
            entity: Some(entity),
            sequence: self.sequence,
            command: Command::Insert {
                entity,
                type_id: std::any::TypeId::of::<T>(),
                payload,
            },
        });
        self.sequence += 1;
        Ok(())
    }

    /// Remove a component from an entity.
    pub fn remove<T: Component>(&mut self, entity: Entity) -> Result<(), CommandError> {
        self.buffer.push(StampedCommand {
            stage: self.stage,
            system_id: self.system_id,
            entity: Some(entity),
            sequence: self.sequence,
            command: Command::Remove {
                entity,
                type_id: std::any::TypeId::of::<T>(),
            },
        });
        self.sequence += 1;
        Ok(())
    }
}
