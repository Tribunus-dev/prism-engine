//! Backend-neutral evaluation of codec-correct candidates.
//!
//! This module owns the canonical authority for the heterogeneous
//! evaluation surface: the backend-neutral contract each backend
//! satisfies, the codec-correct fixtures that drive a run, the
//! immutable evidence each run emits, the admission decision an
//! admission gate emits, and the system that coordinates evaluation
//! lanes. Every evaluation produces durable, replayable evidence
//! that directly governs evolutionary mixed-precision admission and
//! CImage promotion.
//!
//! All types here are backend-neutral. Backend implementations
//! (Metal, ANE, Accelerate, future NPU) live in their respective
//! runtime crates and implement [`backend_trait::BackendEvaluator`]
//! against the contracts defined here.

pub mod admission;
pub mod artifact;
pub mod backend_trait;
pub mod binding_plan;
pub mod fixture;
pub mod generated_executable;
pub mod kernel_abi;
pub mod receipts;
pub mod role;
pub mod system;

pub use admission::AdmissionDecision;
pub use artifact::BackendArtifact;
pub use backend_trait::{BackendEvaluator, EvaluationConfig, EvaluationError, TemperaturePolicy};
pub use binding_plan::{BindingPlan, BindingSlot, ConstantSlot};
pub use fixture::EvaluationFixture;
pub use generated_executable::GeneratedExecutable;
pub use kernel_abi::KernelAbi;
pub use receipts::{
    CodecReceipt, CompilationReceipt, EvaluationReceiptBundle, NumericalReceipt,
    PerformanceReceipt, ProvenanceReceipt, RejectionReceipt, RepeatabilityReceipt,
    StaticValidationReceipt,
};
pub use role::EvaluationRole;
pub use system::{AdmissionPolicy, HeterogeneousEvaluatorSystem};
