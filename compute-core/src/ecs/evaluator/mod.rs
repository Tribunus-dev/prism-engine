//! Heterogeneous evaluator — backend-neutral evaluation, measurement,
//! and admission of codec-correct candidates across Metal, ANE,
//! Accelerate, and future NPU backends.
//!
//! Every evaluation produces durable, replayable evidence that can directly
//! govern evolutionary mixed-precision admission and cimage promotion.

pub mod admission;
pub mod artifact;
pub mod backend_trait;
pub mod binding_plan;
pub mod fixture;
pub mod generated_executable;
pub mod receipts;
pub mod role;
pub mod system;

pub use admission::AdmissionDecision;
pub use artifact::BackendArtifact;
pub use backend_trait::{BackendEvaluator, EvaluationConfig, EvaluationError, TemperaturePolicy};
pub use binding_plan::{BindingPlan, BindingSlot, ConstantSlot};
pub use fixture::EvaluationFixture;
pub use generated_executable::GeneratedExecutable;
pub use receipts::{
    CodecReceipt, CompilationReceipt, EvaluationReceiptBundle, NumericalReceipt,
    PerformanceReceipt, ProvenanceReceipt, RejectionReceipt, RepeatabilityReceipt,
    StaticValidationReceipt,
};
pub use role::EvaluationRole;
pub use system::{AdmissionPolicy, HeterogeneousEvaluatorSystem};
