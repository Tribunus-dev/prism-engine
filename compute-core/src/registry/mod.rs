//! Runtime Capability Registry — governed model deployment and execution.
//!
//! Implements the six-object lifecycle:
//!   ComputeImageManifest → CapabilityPolicy → RuntimeContract
//!   → LiveAdmissionEnvelope → ExecutionLease → ExecutionReceipt
//!
//! Every privileged path must check the contract before acting.
//! No subsystem independently reads env vars or model metadata for authority.

pub mod compiler;
pub mod disclosure;
pub mod governor;
pub mod ledger;
pub mod overlay;
pub mod trust_store;
pub mod types;

pub use compiler::*;
pub use disclosure::*;
pub use governor::*;
pub use ledger::*;
pub use overlay::*;
pub use trust_store::*;
pub use types::*;
