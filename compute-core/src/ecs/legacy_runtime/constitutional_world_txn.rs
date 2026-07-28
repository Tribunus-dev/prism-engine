//! Engine re-export shim for the constitutional `WorldTxn` surface.
//!
//! Authority: this file owns nothing. It exists solely to keep the
//! `crate::ecs::runtime::constitutional_world_txn::ConstitutionalWorldTxn`
//! import path resolvable for engine system files that have not yet
//! been migrated to the constitutional `WorldTxn` API.
//!
//! The full consolidation (replacing `ConstitutionalWorldTxn` with the
//! constitutional `WorldTxn` directly, including the API migration
//! from `stage_insert(entity, component)` to
//! `put_durable::<T: DurableComponent>(entity, component)`) is a
//! separate work item — see the godfile-engine-mapping changelog
//! "1. world_txn.rs". The 44+ engine system files that use
//! `ConstitutionalWorldTxn` are out of scope for the decomposition
//! itself; they will be migrated in the follow-up change.
//!
//! The constitutional crate is the source of truth for the types.
pub use prism_ecs_constitutional::world_txn::CommitReceipt;
pub use prism_ecs_constitutional::world_txn::CommittedEpoch;
pub use prism_ecs_constitutional::world_txn::ComponentChange;
pub use prism_ecs_constitutional::world_txn::PreparedWorldTxn;
pub use prism_ecs_constitutional::world_txn::WorldTxn;
pub use prism_ecs_constitutional::world_txn::WorldTxnError;
pub use prism_ecs_constitutional::world_txn::WorldTransitExt;

/// Type alias mapping the engine's historical `ConstitutionalWorldTxn`
/// name to the constitutional `WorldTxn`. Engine system files that
/// have not been migrated to the constitutional `put_durable` /
/// `put_transient` API still reference this name; the alias keeps the
/// symbol resolvable so that compile errors are type-shape errors
/// (e.g. wrong number of arguments) rather than unresolved-import
/// errors.
pub type ConstitutionalWorldTxn = prism_ecs_constitutional::world_txn::WorldTxn;
