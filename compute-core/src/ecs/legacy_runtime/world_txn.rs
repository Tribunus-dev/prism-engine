//! Engine re-export shim for the constitutional `WorldTxn` surface.
//!
//! Authority: this file owns nothing. It re-exports the canonical
//! `prism_ecs_constitutional::world_txn` types so that engine code
//! that previously imported the engine-local `WorldTxn` continues to
//! resolve at the same path. The constitutional crate is the source
//! of truth; the engine is the projection surface.
pub use prism_ecs_constitutional::world_txn::*;
