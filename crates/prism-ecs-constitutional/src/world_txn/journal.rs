//! Authority: this module owns the canonical journal vocabulary
//! ([`ComponentChange`] / [`ChangeType`]) emitted by every successful
//! [`crate::world_txn::txn::WorldTxn`] commit. The journal is the
//! authority-bearing record of what changed and at which epoch; replay
//! and projection rebuilds consume it as their only input.

use crate::types::SchemaKey;
use crate::types::WorldEpoch;
use prism_ecs_core::Entity;
use serde::Serialize;

/// A single component change recorded in the mutation journal.
///
/// One `ComponentChange` is emitted per durable insert or remove during
/// commit. Transient mutations are not journaled. `before_hash` and
/// `after_hash` are populated by the catalogue encoder once the schema
/// catalogue is wired for encoding (B6 follow-up); both are `None` in
/// the current implementation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ComponentChange {
    /// Entity whose component changed.
    pub entity: Entity,
    /// Stable schema key identifying the component type.
    pub schema_key: SchemaKey,
    /// Insert, update, or remove.
    pub change_type: ChangeType,
    /// Hash of the pre-mutation value, if known. `None` for inserts.
    pub before_hash: Option<[u8; 32]>,
    /// Hash of the post-mutation value, if known. `None` for removes.
    pub after_hash: Option<[u8; 32]>,
    /// World epoch at which the change was committed. The journal entry
    /// is associated with a specific epoch so that replay can advance
    /// the world deterministically.
    pub world_epoch: WorldEpoch,
}

/// The kind of change recorded in a [`ComponentChange`].
///
/// The three variants are mutually exclusive and exhaustive — every
/// commit-time mutation maps to exactly one variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, serde::Deserialize)]
pub enum ChangeType {
    /// A new component was attached to the entity.
    Insert,
    /// An existing component's value was replaced.
    Update,
    /// A component was detached from the entity.
    Remove,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SchemaKey;
    use crate::types::WorldEpoch;
    use prism_ecs_core::Entity;

    /// `ChangeType` is the only enum used to drive replay
    /// branching; its three variants must be exhaustive and
    /// distinguishable.
    #[test]
    fn change_type_three_variants_are_distinct() {
        let a = ChangeType::Insert;
        let b = ChangeType::Update;
        let c = ChangeType::Remove;
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert_ne!(a, c);
    }

    /// `ComponentChange` must carry its `world_epoch` so replay
    /// can advance the world deterministically. Two entries that
    /// differ only by epoch are distinct — the journal entry is
    /// the authority-bearing record of "what changed at this epoch".
    #[test]
    fn component_change_distinguishes_by_epoch() {
        let entity = Entity::new(0, 0);
        let key = SchemaKey {
            namespace: "test",
            id: 1,
            version: 1,
        };
        let mut lhs = ComponentChange {
            entity,
            schema_key: key,
            change_type: ChangeType::Insert,
            before_hash: None,
            after_hash: None,
            world_epoch: WorldEpoch(1),
        };
        let mut rhs = lhs.clone();
        rhs.world_epoch = WorldEpoch(2);
        assert_ne!(lhs, rhs);
        // `Eq + Clone` means replay can deduplicate by structural
        // equality — two entries with identical fields and the
        // same epoch must compare equal.
        lhs.world_epoch = WorldEpoch(2);
        assert_eq!(lhs, rhs);
    }
}
