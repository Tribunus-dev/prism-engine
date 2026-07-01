//! Error types for the scheduling subsystem.
//!
//! Domain separation:
//! - `MaskError` — out-of-range component/resource IDs caught during mask construction.
//! - `RegistryError` — ID collisions caught during component/resource registration.
//! - `ScheduleError` — invalid schedule configurations caught during compilation.
//! - `CommandError` — command buffer capacity or structural mutation failures.

use std::fmt;

use crate::runtime::scheduling::component_id::ComponentId;
use crate::runtime::scheduling::metadata::SystemId;

// ---------------------------------------------------------------------------
// MaskError
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum MaskError {
    #[allow(missing_docs)]
    OutOfRange { id: u16, max: usize },
    #[allow(missing_docs)]
    ResourceOutOfRange { id: u16, max: usize },
}

impl fmt::Display for MaskError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MaskError::OutOfRange { id, max } => {
                write!(f, "component ID {id} exceeds maximum capacity of {max}")
            }
            MaskError::ResourceOutOfRange { id, max } => {
                write!(f, "resource ID {id} exceeds maximum capacity of {max}")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// RegistryError
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum RegistryError {
    #[allow(missing_docs)]
    ComponentIdCollision(ComponentId, &'static str, &'static str),
    #[allow(missing_docs)]
    ResourceIdCollision(u16, &'static str, &'static str),
}

impl fmt::Display for RegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RegistryError::ComponentIdCollision(id, new, existing) => {
                write!(
                    f,
                    "component ID {id} collision: `{new}` conflicts with `{existing}`"
                )
            }
            RegistryError::ResourceIdCollision(id, new, existing) => {
                write!(
                    f,
                    "resource ID {id} collision: `{new}` conflicts with `{existing}`"
                )
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ScheduleError
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum ScheduleError {
    /// A component mask could not be constructed.
    InvalidComponent(MaskError),
    /// A resource mask could not be constructed.
    InvalidResource(MaskError),
    /// Two systems share the same SystemId.
    SystemIdCollision(SystemId),
    /// Two systems share the same name.
    SystemNameCollision(&'static str),
    /// A cycle was detected in the dependency graph.
    /// The vector contains the ordered edge-path of SystemIds.
    CycleDetected(Vec<SystemId>),
    /// Two systems have an undeclared write/write overlap on a component.
    IllegalHazard {
        /// First system.
        system_a: SystemId,
        /// Second system.
        system_b: SystemId,
        /// Human-readable reason.
        reason: &'static str,
    },
    /// An explicit `before` or `after` edge crosses stage boundaries in
    /// the wrong direction.
    StageInversion {
        /// The system declaring the edge.
        system: SystemId,
        /// The target system.
        target: SystemId,
    },
    /// A system references a system ID that does not exist in the set.
    UnknownTarget {
        /// The system declaring the edge.
        system: SystemId,
        /// The target that is not registered.
        target: SystemId,
    },
}

impl fmt::Display for ScheduleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScheduleError::InvalidComponent(e) => write!(f, "invalid component mask: {e}"),
            ScheduleError::InvalidResource(e) => write!(f, "invalid resource mask: {e}"),
            ScheduleError::SystemIdCollision(id) => {
                write!(f, "duplicate system ID: {id:?}")
            }
            ScheduleError::SystemNameCollision(name) => {
                write!(f, "duplicate system name: `{name}`")
            }
            ScheduleError::CycleDetected(path) => {
                write!(
                    f,
                    "cycle detected: {}",
                    path.iter()
                        .map(|id| format!("{id:?}"))
                        .collect::<Vec<_>>()
                        .join(" -> ")
                )
            }
            ScheduleError::IllegalHazard {
                system_a,
                system_b,
                reason,
            } => {
                write!(
                    f,
                    "write/write hazard between {system_a:?} and {system_b:?}: {reason}"
                )
            }
            ScheduleError::StageInversion { system, target } => {
                write!(
                    f,
                    "stage inversion: {system:?} declares edge to {target:?} in a prior stage"
                )
            }
            ScheduleError::UnknownTarget { system, target } => {
                write!(
                    f,
                    "{system:?} references unknown target system {target:?}"
                )
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CommandError
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum CommandError {
    /// Entity reference does not exist in the World.
    InvalidEntity,
    /// Command buffer capacity reached.
    CapacityExhausted { max: usize },
    /// The requested structural mutation is not allowed in the current stage.
    ProhibitedInStage,
    /// Serialization of a component failed.
    SerializationError(String),
}

impl fmt::Display for CommandError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CommandError::InvalidEntity => write!(f, "invalid entity reference"),
            CommandError::CapacityExhausted { max } => {
                write!(f, "command buffer capacity ({max}) exhausted")
            }
            CommandError::ProhibitedInStage => {
                write!(f, "structural mutation not allowed in current stage")
            }
            CommandError::SerializationError(msg) => {
                write!(f, "component serialization failed: {msg}")
            }
        }
    }
}
