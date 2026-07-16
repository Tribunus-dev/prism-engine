//! System metadata, static contract, and object-safe runtime boundary.
//!
//! The two-level contract:
//! - `SystemSpec` — compile-time trait that declares reads, writes, identity,
//!   stage, ordering constraints, and serialization policy.
//! - The object-safe `run` method is separate; `SystemMetadata` bridges
//!   the two, producing an immutable struct consumed by the schedule compiler.

use serde::{Deserialize, Serialize};

use crate::scheduling::access::{ComponentSet, ResourceSet};
use crate::scheduling::component_id::{ComponentMask, ResourceMask};

// ---------------------------------------------------------------------------
// Stage
// ---------------------------------------------------------------------------

/// Temporal execution band imposed by the schedule compiler.
///
/// All systems in one stage complete (including command-buffer drain) before
/// any system in the next stage runs.  Stages are ordered by discriminant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum Stage {
    /// Initial intake — new request deserialization and entity creation.
    Intake = 0,
    /// Prefill computation — prompt processing.
    Prefill = 1,
    /// Token-by-token decode loop.
    Decode = 2,
    /// Post-decode tool execution or multi-modal processing.
    PostDecode = 3,
    /// Tool execution — function calls, retrieval, external actions.
    ToolExecution = 4,
    /// Receipt recording and telemetry emission.
    Receipt = 5,
    /// Maintenance — compaction, cache eviction, lifecycle transitions.
    Maintenance = 6,
}

impl Stage {
    /// All stages in canonical order.
    pub const ALL: [Stage; 7] = [
        Stage::Intake,
        Stage::Prefill,
        Stage::Decode,
        Stage::PostDecode,
        Stage::ToolExecution,
        Stage::Receipt,
        Stage::Maintenance,
    ];

    /// Number of stage variants.
    pub const COUNT: usize = 7;
}

// ---------------------------------------------------------------------------
// ExecutionClass
// ---------------------------------------------------------------------------

/// How a system is scheduled relative to other systems.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionClass {
    /// Runs as part of the main schedule tick.
    Normal,
    /// Runs only when its trigger condition is met.
    Conditional,
}

// ---------------------------------------------------------------------------
// SerializationPolicy
// ---------------------------------------------------------------------------

/// Policy for resolving write/write conflicts between two systems.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SerializationPolicy {
    /// Any write/write overlap is a compilation error.
    Reject,
    /// Overlap requires an explicit `before` or `after` edge.
    ExplicitOnly,
    /// Overlap is resolved deterministically by (stage, order, system_id).
    StableOrder,
    /// Systems are commutative — no edge is created.
    Commutative,
}

// ---------------------------------------------------------------------------
// SystemMetadata
// ---------------------------------------------------------------------------

/// Immutable declaration of a system's identity, access, and ordering.
///
/// Produced by `SystemSpec::metadata()` and consumed by the schedule
/// compiler.  Never mutated at runtime.
#[derive(Debug, Clone)]
pub struct SystemMetadata {
    pub id: SystemId,
    pub name: &'static str,
    pub stage: Stage,
    pub order: u16,
    pub execution_class: ExecutionClass,
    pub reads: ComponentMask,
    pub writes: ComponentMask,
    pub reads_resources: ResourceMask,
    pub writes_resources: ResourceMask,
    pub after: &'static [SystemId],
    pub before: &'static [SystemId],
    pub serialization: SerializationPolicy,
}

// ---------------------------------------------------------------------------
// SystemId
// ---------------------------------------------------------------------------

/// Stable numeric identity for a system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SystemId(pub u32);

// ---------------------------------------------------------------------------
// SystemResult
// ---------------------------------------------------------------------------

/// Typed result returned by every system execution.
#[derive(Debug, Clone)]
pub enum SystemResult {
    /// System completed normally.
    Ok,
    /// System encountered a recoverable error.
    Err {
        /// Human-readable error description.
        message: String,
    },
}

impl SystemResult {
    /// Convenience constructor for success.
    pub fn ok() -> Self {
        SystemResult::Ok
    }

    /// Convenience constructor for error.
    pub fn error(msg: impl Into<String>) -> Self {
        SystemResult::Err { message: msg.into() }
    }
}

// ---------------------------------------------------------------------------
// SystemSpec — compile-time declaration trait
// ---------------------------------------------------------------------------

/// Static contract every system implements.
pub trait SystemSpec: Send {
    type Reads: ComponentSet;
    type Writes: ComponentSet;
    type ReadsResources: ResourceSet;
    type WritesResources: ResourceSet;

    fn system_id() -> SystemId;
    fn system_name() -> &'static str;
    fn stage() -> Stage;
    fn order() -> u16;
    fn execution_class() -> ExecutionClass {
        ExecutionClass::Normal
    }
    fn after() -> &'static [SystemId] {
        &[]
    }
    fn before() -> &'static [SystemId] {
        &[]
    }
    fn serialization() -> SerializationPolicy {
        SerializationPolicy::StableOrder
    }

    /// Produce the immutable `SystemMetadata` for this system.
    fn metadata() -> SystemMetadata {
        SystemMetadata {
            id: Self::system_id(),
            name: Self::system_name(),
            stage: Self::stage(),
            order: Self::order(),
            execution_class: Self::execution_class(),
            reads: Self::Reads::mask().unwrap_or_else(|e| {
                panic!(
                    "System {}: invalid reads mask: {e}",
                    Self::system_name()
                )
            }),
            writes: Self::Writes::mask().unwrap_or_else(|e| {
                panic!(
                    "System {}: invalid writes mask: {e}",
                    Self::system_name()
                )
            }),
            reads_resources: Self::ReadsResources::mask().unwrap_or_else(|e| {
                panic!(
                    "System {}: invalid reads_resources mask: {e}",
                    Self::system_name()
                )
            }),
            writes_resources: Self::WritesResources::mask().unwrap_or_else(|e| {
                panic!(
                    "System {}: invalid writes_resources mask: {e}",
                    Self::system_name()
                )
            }),
            after: Self::after(),
            before: Self::before(),
            serialization: Self::serialization(),
        }
    }
}
