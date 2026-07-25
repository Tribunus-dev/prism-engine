use serde::{Deserialize, Serialize};

/// Legacy entity identifier (no generation tracking).
/// Generational entity handle.
///
/// Each entity carries a generation counter that is incremented on despawn,
/// catching stale references. The generation is opaque — callers construct
/// entities through `World::spawn()` or `Commands::spawn()`, never by
/// fabricating the tuple.
///
/// # Invalid-handle contract
///
/// - `Entity(0, _)` (zero ID) is always invalid — handle it as a null
///   sentinel. Every API returns `None`/`false`, never panics.
/// - A handle whose generation does not match the current generation in the
///   world slot is stale — `is_alive()` returns `false`, queries skip it.
/// - Fabricated handles (created outside `World::spawn()`) are treated as
///   stale unless the slot exists AND generation matches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct Entity(pub u64, pub u32);

impl Entity {
    /// Construct a new entity handle with the given ID and generation.
    ///
    /// Prefer receiving entities from `World::spawn()` over constructing them
    /// directly. Use generation `0` for entities created outside the ECS
    /// lifecycle (e.g., replay, test fixtures).
    pub fn new(id: u64, generation: u32) -> Self {
        Self(id, generation)
    }
    pub fn id(&self) -> u64 {
        self.0
    }
    pub fn generation(&self) -> u32 {
        self.1
    }
}

/// A reserved but not-yet-committed entity, returned by `Commands::spawn()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PendingEntity(pub u64);

/// Result of spawning an entity, with allocation detail.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpawnedEntity {
    pub entity: Entity,
    pub allocation: EntityAllocation,
}

/// Describes how an entity was allocated during spawn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntityAllocation {
    /// Fresh entity slot — no previous occupant.
    NewSlot,
    /// Reused slot from a despawned entity.
    ReusedSlot { previous_generation: u32 },
}

impl From<SpawnedEntity> for Entity {
    fn from(s: SpawnedEntity) -> Self {
        s.entity
    }
}

/// Entity kind classification — used for diagnostic and replay classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntityKind {
    Node,
    Pipeline,
    Model,
    Tensor,
    Layer,
    Expert,
    Dispatch,
    Kernel,
    KernelVariant,
    Buffer,
    CommandBuffer,
    Executable,
    Fence,
    Session,
    Artifact,
    Device,
    Residency,
    Agent,
    /// A work group for scheduling SPU-style work units across execution lanes.
    WorkGroup,
    /// A single work unit within a work group.
    WorkUnit,
}
