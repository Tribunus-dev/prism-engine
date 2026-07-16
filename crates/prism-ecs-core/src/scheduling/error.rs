//! Error types for the scheduling subsystem.
//!
//! Domain separation:
//! - `MaskError` — out-of-range component/resource IDs caught during mask construction.
//! - `RegistryError` — ID collisions caught during component/resource registration.
//! - `ScheduleError` — invalid schedule configurations caught during compilation.
//! - `CommandError` — command buffer capacity or structural mutation failures.

use std::fmt;

use crate::scheduling::metadata::SystemId;

// ---------------------------------------------------------------------------
// MaskError
// ---------------------------------------------------------------------------

/// An out-of-range component or resource ID was used in mask construction.
#[derive(Debug, Clone)]
pub enum MaskError {
    /// Component ID exceeds the maximum of 255.
    ComponentIdOutOfRange { id: u16 },
    /// Resource ID exceeds the maximum of 255.
    ResourceIdOutOfRange { id: u16 },
}

impl fmt::Display for MaskError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MaskError::ComponentIdOutOfRange { id } => {
                write!(f, "component ID {id} exceeds maximum 255")
            }
            MaskError::ResourceIdOutOfRange { id } => {
                write!(f, "resource ID {id} exceeds maximum 255")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// RegistryError
// ---------------------------------------------------------------------------

/// An ID collision or out-of-range ID was detected during registration.
#[derive(Debug, Clone)]
pub enum RegistryError {
    /// Two different component types registered with the same ID.
    ComponentIdCollision {
        id: u16,
        existing: &'static str,
        incoming: &'static str,
    },
    /// Two different resource types registered with the same ID.
    ResourceIdCollision {
        id: u16,
        existing: &'static str,
        incoming: &'static str,
    },
    /// Component array is full (≥256 registered).
    ComponentRegistryFull,
    /// Resource array is full (≥256 registered).
    ResourceRegistryFull,
}

impl fmt::Display for RegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RegistryError::ComponentIdCollision {
                id,
                existing,
                incoming,
            } => {
                write!(
                    f,
                    "component ID {id} collision between {existing} and {incoming}"
                )
            }
            RegistryError::ResourceIdCollision {
                id,
                existing,
                incoming,
            } => {
                write!(
                    f,
                    "resource ID {id} collision between {existing} and {incoming}"
                )
            }
            RegistryError::ComponentRegistryFull => {
                write!(f, "component registry full (max 256)")
            }
            RegistryError::ResourceRegistryFull => {
                write!(f, "resource registry full (max 256)")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ScheduleError
// ---------------------------------------------------------------------------

/// An invalid schedule configuration was detected during compilation.
#[derive(Debug, Clone)]
pub enum ScheduleError {
    /// Two systems share the same SystemId.
    SystemIdCollision(SystemId),
    /// Two systems share the same name.
    SystemNameCollision(&'static str),
    /// An explicit edge targets an unregistered system.
    TargetNotRegistered { from: SystemId, target: SystemId },
    /// An explicit edge would invert the stage ordering.
    StageInversion {
        from: SystemId,
        target: SystemId,
        from_stage: usize,
        target_stage: usize,
    },
    /// A write/write hazard was detected and neither system permits it.
    IllegalHazard {
        system_a: SystemId,
        system_b: SystemId,
        reason: &'static str,
    },
    /// A cycle was detected in the dependency graph.
    CycleDetected(Vec<SystemId>),
}

impl fmt::Display for ScheduleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScheduleError::SystemIdCollision(id) => {
                write!(f, "duplicate system ID {id:?}")
            }
            ScheduleError::SystemNameCollision(name) => {
                write!(f, "duplicate system name '{name}'")
            }
            ScheduleError::TargetNotRegistered { from, target } => {
                write!(
                    f,
                    "system {from:?} edge targets unregistered system {target:?}"
                )
            }
            ScheduleError::StageInversion {
                from,
                target,
                from_stage,
                target_stage,
            } => {
                write!(
                    f,
                    "stage inversion: system {from:?} (stage {from_stage}) depends on \
                     system {target:?} (stage {target_stage}) which is in a later stage"
                )
            }
            ScheduleError::IllegalHazard {
                system_a,
                system_b,
                reason,
            } => {
                write!(
                    f,
                    "illegal write/write hazard between {system_a:?} and {system_b:?}: {reason}"
                )
            }
            ScheduleError::CycleDetected(nodes) => {
                write!(f, "cycle detected in dependency graph: {nodes:?}")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CommandError
// ---------------------------------------------------------------------------

/// A command buffer operation failed.
#[derive(Debug, Clone)]
pub enum CommandError {
    /// The command buffer is full (exceeded configured capacity).
    BufferFull { capacity: usize, attempted: usize },
    /// A structural mutation is invalid for the current schedule state.
    InvalidMutation { detail: String },
}

impl fmt::Display for CommandError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CommandError::BufferFull {
                capacity,
                attempted,
            } => {
                write!(
                    f,
                    "command buffer full: capacity={capacity}, attempted={attempted}"
                )
            }
            CommandError::InvalidMutation { detail } => {
                write!(f, "invalid mutation: {detail}")
            }
        }
    }
}
