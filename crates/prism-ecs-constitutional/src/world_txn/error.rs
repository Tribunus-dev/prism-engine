//! Authority: this module owns the canonical error vocabulary for
//! [`WorldTxn`] commit and prepare operations. The variants are
//! classified as `Rejected` (preflight), `Failed` (apply), or `Stale`
//! (fencing mismatch); the error enum is the constitutional surface, not
//! a `String`. No `anyhow::Error` is used here — callers must pattern
//! match on these variants to distinguish rejection from stale state.

use crate::types::{ComponentSchemaId, SchemaKey};
use prism_ecs_core::Entity;
use prism_ecs_core::WorldEpoch;

/// Errors that can occur during transaction commit, classified per AGENTS.md
/// "no `anyhow::Error` in constitutional crates":
///
/// - `Rejected` (preflight): the transaction was rejected before any
///   mutation. `StaleEpoch`, `StaleRead`, `UnregisteredSchema`,
///   `SchemaMismatch`, `InvalidEntity`, `ComponentNotFound`, `Conflict`.
/// - `Failed` (effect): the underlying [`prism_ecs_core::World`] apply
///   operation failed. `WorldApply`.
/// - `Stale` (fencing mismatch): a fencing check failed, indicating the
///   caller's view of the world is out of date. `StaleEpoch`, `StaleRead`.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum WorldTxnError {
    /// The preflight catalogue did not contain a registration for a
    /// durable schema key that a staged insert referenced.
    #[error("durable schema {schema_key:?} not registered in catalogue")]
    UnregisteredSchema { schema_key: SchemaKey },
    /// The world's current epoch did not match the transaction's
    /// expected epoch. This is a fencing violation — the caller's view
    /// is stale and must be refreshed before retrying.
    #[error("stale epoch: expected {expected:?}, current {current:?}")]
    StaleEpoch {
        expected: WorldEpoch,
        current: WorldEpoch,
    },
    /// A read dependency recorded on the transaction observed a component
    /// version that no longer matches the world's current version. This
    /// is optimistic-concurrency control (OCC) failure.
    #[error(
        "stale read: entity {entity:?} schema {schema_id:?} version {observed} != current {current}"
    )]
    StaleRead {
        entity: Entity,
        schema_id: ComponentSchemaId,
        observed: u64,
        current: u64,
    },
    /// The transaction targeted an entity handle that does not exist in
    /// the world (and was not introduced by a same-txn spawn).
    #[error("invalid entity handle: {0:?}")]
    InvalidEntity(Entity),
    /// A staged durable insert used a `SchemaKey` whose `type_id` did
    /// not match the type registered against that key in the catalogue.
    #[error("schema mismatch for {schema_id:?}: expected {expected}")]
    SchemaMismatch {
        schema_id: ComponentSchemaId,
        expected: String,
    },
    /// The transaction referenced a component that is not present on the
    /// targeted entity.
    #[error("component not found: entity {entity:?} schema {schema_id:?}")]
    ComponentNotFound {
        entity: Entity,
        schema_id: ComponentSchemaId,
    },
    /// The transaction staged two operations on the same (entity,
    /// schema) pair, which is forbidden. Rejected at preflight.
    #[error("conflicting operations for entity {entity:?} schema {schema_id:?}")]
    Conflict {
        entity: Entity,
        schema_id: ComponentSchemaId,
    },
    /// The `World::transit` apply phase failed. The transaction has
    /// already been validated; this is an effect-time failure.
    #[error("world apply failed: {0}")]
    WorldApply(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ComponentSchemaId;
    use crate::types::SchemaKey;
    use crate::types::WorldEpoch;
    use prism_ecs_core::Entity;

    /// The error surface is a `thiserror`-derived enum with
    /// `Clone + PartialEq + Eq`, so callers can pattern-match without
    /// string parsing. `StaleEpoch` and `StaleRead` are the two
    /// fencing variants; they must be distinguishable so that
    /// retry policy can differ.
    #[test]
    fn stale_fencing_variants_are_distinguishable() {
        let entity = Entity::new(0, 0);
        let stale_epoch = WorldTxnError::StaleEpoch {
            expected: WorldEpoch(1),
            current: WorldEpoch(2),
        };
        let stale_read = WorldTxnError::StaleRead {
            entity,
            schema_id: ComponentSchemaId(7),
            observed: 3,
            current: 4,
        };
        assert_ne!(stale_epoch, stale_read);
        // Display strings are stable enough for downstream log
        // signatures; the message must mention "stale".
        assert!(stale_epoch.to_string().contains("stale"));
        assert!(stale_read.to_string().contains("stale"));
        // `UnregisteredSchema` carries the schema key verbatim so
        // log scrapers can join against the schema catalogue.
        let unregistered = WorldTxnError::UnregisteredSchema {
            schema_key: SchemaKey {
                namespace: "test",
                id: 9,
                version: 1,
            },
        };
        assert!(unregistered.to_string().contains("test"));
    }
}
