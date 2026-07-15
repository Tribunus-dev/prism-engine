use thiserror::Error;

use crate::Entity;

/// Typed error for World operations.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum WorldError {
    #[error("stale entity handle: {entity:?} — entity has been despawned or never existed")]
    StaleHandle { entity: Entity },

    #[error("direct mutation disallowed for operation: {operation} — use WorldTxn")]
    DirectMutationDisallowed { operation: &'static str },

    #[error("entity capacity exceeded: capacity={capacity}, attempted={attempted}")]
    EntityCapacityExceeded { capacity: u64, attempted: u64 },

    #[error("entity {entity:?} not found")]
    EntityNotFound { entity: Entity },

    #[error("component of type {type_name} already exists on entity {entity:?}")]
    DuplicateComponent {
        entity: Entity,
        type_name: &'static str,
    },

    #[error("component of type {type_name} not found on entity {entity:?}")]
    MissingComponent {
        entity: Entity,
        type_name: &'static str,
    },

    #[error("resource of type {type_name} not found")]
    MissingResource { type_name: &'static str },

    #[error("resource of type {type_name} already exists")]
    DuplicateResource { type_name: &'static str },

    #[error("type mismatch: expected {expected}, got {actual}")]
    TypeMismatch {
        expected: &'static str,
        actual: &'static str,
    },

    #[error("capacity exhausted: {detail}")]
    CapacityExhausted { detail: String },

    #[error("mutation policy violation: {detail}")]
    MutationPolicyViolation { detail: String },

    #[error("borrow conflict: {detail}")]
    BorrowConflict { detail: String },

    #[error("invalid pending entity: {detail}")]
    InvalidPendingEntity { detail: String },

    #[error("invariant violation: {detail}")]
    InvariantViolation { detail: String },
}
