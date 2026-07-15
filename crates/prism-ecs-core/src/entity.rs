use serde::{Deserialize, Serialize};

/// Legacy entity identifier (no generation tracking).
#[deprecated(note = "use Entity(u64, u32) for generation safety")]
pub type EntityId = u64;

/// Legacy ID-only entity handle.
#[deprecated(note = "use Entity(u64, u32) for generation safety")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CompEntity(pub EntityId);

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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Entity(pub u64, pub u32);

impl Entity {
    pub fn id(&self) -> u64 {
        self.0
    }
    pub fn generation(&self) -> u32 {
        self.1
    }
}

impl From<CompEntity> for Entity {
    fn from(ce: CompEntity) -> Self {
        Entity(ce.0, 0)
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
}
