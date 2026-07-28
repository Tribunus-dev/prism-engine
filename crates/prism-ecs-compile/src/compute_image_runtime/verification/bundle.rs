//! Bundle of all verification receipts — re-exports each individual
//! receipt type so consumers can import the full bundle from one place.

pub use super::phase_graph::PhaseGraphVerificationReceipt;
pub use super::residency::ResidencyVerificationReceipt;
pub use super::resource_fit::ResourceFitReceipt;
