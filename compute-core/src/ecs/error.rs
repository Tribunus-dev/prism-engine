//! Typed errors for World operations.
//!
//! Operational failures (stale handles, missing entities, duplicate components,
//! borrow conflicts, transaction errors) return typed `WorldError` values instead
//! of panicking. Panics remain appropriate only for internal invariant violations
//! that indicate memory corruption or transactional correctness bugs.

use thiserror::Error;

/// Typed error for World operations.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum WorldError {
    #[error("stale entity handle: {entity:?} — entity has been despawned or never existed")]
    StaleHandle { entity: super::Entity },

    #[error("entity {entity:?} not found")]
    EntityNotFound { entity: super::Entity },

    #[error("component of type {type_name} already exists on entity {entity:?}")]
    DuplicateComponent {
        entity: super::Entity,
        type_name: &'static str,
    },

    #[error("component of type {type_name} not found on entity {entity:?}")]
    MissingComponent {
        entity: super::Entity,
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

    #[error("epoch conflict: expected {expected:?}, current {current:?}")]
    EpochConflict {
        expected: super::constitutional::types::WorldEpoch,
        current: super::constitutional::types::WorldEpoch,
    },

    #[error("invalid pending entity: {detail}")]
    InvalidPendingEntity { detail: String },

    #[error("invariant violation: {detail}")]
    InvariantViolation { detail: String },

    #[error(transparent)]
    TransactionError(#[from] super::constitutional::world_txn::WorldTxnError),
}
