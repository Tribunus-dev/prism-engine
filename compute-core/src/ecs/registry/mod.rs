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
pub mod overlay;
#[cfg(feature = "legacy_mutations")]
pub mod trust_store;
pub mod types;

pub use compiler::*;
pub use disclosure::*;
pub use governor::*;
pub use overlay::*;
#[cfg(feature = "legacy_mutations")]
pub use trust_store::*;
pub use types::*;
