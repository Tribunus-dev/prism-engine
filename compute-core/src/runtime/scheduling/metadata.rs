//! System metadata, static contract, and object-safe runtime boundary.
//!
//! The two-level contract:
//! - `SystemSpec` — compile-time trait that declares reads, writes, identity,
//!   stage, ordering constraints, and serialization policy.
//! - `ErasedSystem` — object-safe runtime interface stored in `Vec<Box<dyn ErasedSystem>>`.
//!
//! `SystemSpec::metadata()` bridges the two, producing an immutable
//! `SystemMetadata` struct consumed by the schedule compiler.

use crate::runtime::scheduling::access::{ComponentSet, ResourceSet};
use crate::runtime::scheduling::command::CommandWriter;
use serde::{Deserialize, Serialize};
use crate::runtime::scheduling::component_id::{ComponentMask, ResourceMask};
use crate::runtime::scheduling::error::ScheduleError;
use crate::runtime::world::World;

// ---------------------------------------------------------------------------
// Stage
// ---------------------------------------------------------------------------

/// Temporal execution band imposed by the schedule compiler.
///
/// All systems in one stage complete (including command-buffer drain) before
/// any system in the next stage runs.  Stages are ordered by discriminant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Stage {
    /// Request intake – enqueue commands, validate inputs.
    Intake = 0,
    /// Admission control – check budgets, policy, concurrency limits.
    Admission = 1,
    /// Weight and cache residency – migrate data to compute hardware.
    Residency = 2,
    /// ANE-assisted prefill – compute full KV cache for a prompt.
    Prefill = 3,
    /// GPU decode loop – autoregressive token generation.
    Decode = 4,
    /// Post-decode processing – MTP speculation, grammar masking, tool calls.
    PostDecode = 5,
    /// Tool execution – run external functions, collect results.
    ToolExecution = 6,
    /// Periodic maintenance – watchdog, budget reaper, migration tick.
    Maintenance = 7,
    /// Terminal receipt emission – finalize and publish state.
    Receipt = 8,
}

impl Stage {
    /// All stages in declaration order.
    pub const ALL: &'static [Stage] = &[
        Stage::Intake,
        Stage::Admission,
        Stage::Residency,
        Stage::Prefill,
        Stage::Decode,
        Stage::PostDecode,
        Stage::ToolExecution,
        Stage::Maintenance,
        Stage::Receipt,
    ];
}

// ---------------------------------------------------------------------------
// ExecutionClass
// ---------------------------------------------------------------------------

/// How a system is scheduled relative to other systems.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionClass {
    /// Run on the main scheduler thread, in order with all other systems.
    Serial,
    /// May be co-scheduled with other parallel systems on separate threads.
    /// Reserved for a future concurrent scheduler lane implementation.
    Parallel,
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
    /// Systems are commutative — no edge is created; overlap is recorded
    /// in the manifest as a legally permitted hazard but no fallback edge
    /// is generated.
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
    /// Stable numeric identity.
    pub id: SystemId,
    /// Human-readable name (for diagnostics, manifests, receipts).
    pub name: &'static str,
    /// Temporal stage.
    pub stage: Stage,
    /// Components this system reads.
    pub reads: ComponentMask,
    /// Components this system writes.
    pub writes: ComponentMask,
    /// Resources this system reads.
    pub reads_resources: ResourceMask,
    /// Resources this system writes.
    pub writes_resources: ResourceMask,
    /// Must run after these systems.
    pub after: &'static [SystemId],
    /// Must run before these systems.
    pub before: &'static [SystemId],
    /// Relative ordering hint within the same stage.
    pub order: i32,
    /// Execution class.
    pub execution_class: ExecutionClass,
    /// Write/write hazard resolution policy.
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

    /// Convenience constructor for failure.
    pub fn err(message: impl Into<String>) -> Self {
        SystemResult::Err {
            message: message.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// Resources — thin wrapper around World for resource access
// ---------------------------------------------------------------------------

/// Capability-scoped access to singleton resources.
///
/// Wraps `&mut World` but only exposes resource read/write, not component
/// or entity operations.  This enforces the architectural boundary:
/// systems mutate components via `CommandWriter` and read resources
/// through this handle.
pub struct Resources<'a>(&'a mut World);

impl<'a> Resources<'a> {
    /// Create a new resource access handle.
    pub fn new(world: &'a mut World) -> Self {
        Self(world)
    }

    /// Read a resource from the World.
    pub fn get<T: crate::runtime::world::Resource>(&self) -> Option<&T> {
        self.0.get_resource::<T>()
    }

    /// Mutate a resource in the World.
    pub fn get_mut<T: crate::runtime::world::Resource>(&mut self) -> Option<&mut T> {
        self.0.get_resource_mut::<T>()
    }

    /// Insert or replace a resource.
    pub fn insert<T: crate::runtime::world::Resource>(&mut self, resource: T) {
        self.0.insert_resource(resource);
    }
}

// ---------------------------------------------------------------------------
// SystemSpec — compile-time declaration trait
// ---------------------------------------------------------------------------

/// Static contract every system implements.
///
/// The associated types (`Reads`, `Writes`, etc.) are resolved at compile time
/// and compiled into a `SystemMetadata` by `metadata()`.
pub trait SystemSpec: Send {
    /// Component set this system reads.
    type Reads: ComponentSet;
    /// Component set this system writes.
    type Writes: ComponentSet;
    /// Resource set this system reads.
    type ReadResources: ResourceSet;
    /// Resource set this system writes.
    type WriteResources: ResourceSet;

    /// Human-readable name.
    const NAME: &'static str;
    /// Stable numeric identity.
    const ID: SystemId;
    /// Temporal stage.
    const STAGE: Stage;
    /// Relative ordering hint within stage.
    const ORDER: i32;
    /// Must run after these systems.
    const AFTER: &'static [SystemId] = &[];
    /// Must run before these systems.
    const BEFORE: &'static [SystemId] = &[];
    /// Execution class.
    const EXECUTION_CLASS: ExecutionClass = ExecutionClass::Serial;
    /// Write/write hazard policy.
    const SERIALIZATION: SerializationPolicy = SerializationPolicy::Reject;

    /// Compile the static declarations into an immutable metadata struct.
    fn metadata() -> Result<SystemMetadata, ScheduleError> {
        Ok(SystemMetadata {
            id: Self::ID,
            name: Self::NAME,
            stage: Self::STAGE,
            reads: Self::Reads::mask().map_err(ScheduleError::InvalidComponent)?,
            writes: Self::Writes::mask().map_err(ScheduleError::InvalidComponent)?,
            reads_resources: Self::ReadResources::mask()
                .map_err(ScheduleError::InvalidResource)?,
            writes_resources: Self::WriteResources::mask()
                .map_err(ScheduleError::InvalidResource)?,
            after: Self::AFTER,
            before: Self::BEFORE,
            order: Self::ORDER,
            execution_class: Self::EXECUTION_CLASS,
            serialization: Self::SERIALIZATION,
        })
    }
}

// ---------------------------------------------------------------------------
// ErasedSystem — object-safe runtime boundary
// ---------------------------------------------------------------------------

/// Object-safe execution trait stored by the scheduler.
///
/// Does NOT require `Sync` — the scheduler owns each system mutably during
/// execution and never shares references across threads without explicit
/// parallel lane support.
pub trait ErasedSystem: Send {
    /// Return the immutable metadata for this system.
    fn metadata(&self) -> &SystemMetadata;

    /// Execute this system.
    ///
    /// `world` provides mutable World access for the initial Slice 1.
    /// `resources` provides capability-scoped resource access.
    /// `commands` is the structural mutation channel.
    fn run(
        &mut self,
        world: &mut World,
        commands: &mut CommandWriter,
    ) -> SystemResult;
}
