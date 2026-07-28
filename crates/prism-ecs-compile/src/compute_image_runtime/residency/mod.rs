//! Compiled residency plan types — pure data types and pure algorithms
//! for memory residency scheduling.

pub mod admission;
pub mod arena;
pub mod plan;
pub mod prefetch;
pub mod receipts;
pub mod weights;

pub use plan::{
    ActivationArenaRequirements, CompiledResidencyPlan, EvictableWeightObject, EvictionPolicy,
    KvCacheRequirements, MemoryAdmissionContract, PeakMemoryAnalyzer, PeakMemoryEstimate,
    PrefetchAction, PrefetchPriority, RequiredWeightObject, RequiredWeightObjectId,
    ResidencyClass, ResidencyPlanId,
};
