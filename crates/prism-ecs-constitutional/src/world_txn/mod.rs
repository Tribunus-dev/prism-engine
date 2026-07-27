//! Authority: this module re-exports the canonical `WorldTxn` surface,
//! decomposed into single-authority sub-modules:
//!
//! - [`access`] — access-kind vocabulary and read/write declarations
//! - [`journal`] — committed mutation journal entries
//! - [`durable`] — durable / transient component classification
//! - [`txn`] — the staged transaction itself, prepare/apply protocol,
//!   staged and prepared operation records, and the [`WorldTransitExt`]
//!   extension trait that binds the transaction to the world
//! - [`epoch`] — post-commit epoch token
//! - [`error`] — error variants classified as `Rejected`, `Failed`, or
//!   `Stale`
//!
//! Every public name lives in exactly one sub-module. The `mod.rs`
//! re-exports the sub-modules' public items so the historical
//! `prism_ecs_constitutional::world_txn::WorldTxn` (etc.) import paths
//! continue to resolve.

pub mod access;
pub mod durable;
pub mod epoch;
pub mod error;
pub mod journal;
pub mod txn;

pub use access::{AccessDeclaration, AccessKind};
pub use durable::{
    ClassifiedComponent, ComponentClass, DurableClass, DurableComponent, TransientClass,
    TransientComponent,
};
pub use epoch::CommittedEpoch;
pub use error::WorldTxnError;
pub use journal::{ChangeType, ComponentChange};
pub use txn::{
    CommitReceipt, PreparedWorldTxn, WorldTransitExt, WorldTxn,
};
