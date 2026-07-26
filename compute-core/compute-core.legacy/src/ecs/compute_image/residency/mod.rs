//! Compiled residency plan types for the SealedComputeImageExecutable.
//!
//! This module contains the compiler-computed memory residency schedule
//! that the runtime executes.  Every type is derived with `Debug`,
//! `Clone`, `Serialize`, and `Deserialize` for inspection, caching,
//! and serialization across process boundaries.
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub(crate) mod arena;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub mod prefetch;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub(crate) mod weights;

#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub use self::weights::{ResidencyClassifier, WeightObject};
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub mod admission;
pub mod receipts;

#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub use self::plan::PeakMemoryAnalyzer;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub use self::prefetch::PrefetchScheduleBuilder;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub use admission::{ResidencyAdmission, ResidencyAdmissionResult, ResidencyRefusalReason};
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub use plan::{
    ActivationArenaRequirements, CompiledResidencyPlan, EvictableWeightObject, EvictionPolicy,
    KvCacheRequirements, MemoryAdmissionContract, PeakMemoryEstimate, PrefetchAction,
    PrefetchPriority, RequiredWeightObject, RequiredWeightObjectId, ResidencyClass,
    ResidencyPlanId,
};
pub use receipts::{ResidencyAdmissionReceipt, ResidencyExecutionReceipt};

#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub mod plan;
