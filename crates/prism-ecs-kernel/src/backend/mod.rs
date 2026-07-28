//! Backend executors and adapters — the kernel-side hardware FFI surface.
//!
//! This module is the constitutional home for the per-backend
//! executors (Metal, ANE, Accelerate, CPU, legacy) and the
//! per-backend state records (memory pools, weight residency,
//! activation arenas, etc.). The runtime scheduling systems
//! produce typed, non-authoritative completion values; the runtime
//! reconciliation system stages the resulting state transition
//! through `ConstitutionalWorldTxn`.
//!
//! # Migration status
//!
//! Per the inventory v2.1 steps 36-50, the engine's adapter files
//! move here. As of 2026-07-27, the per-backend submodules are
//! scaffolded with placeholder executors; the full implementations
//! arrive when the engine's heterogeneous_executor splits in step 36.

pub mod accelerate;
pub mod ane;
pub mod authority;
pub mod completion;
pub mod cpu;
pub mod dispatcher;
pub mod evaluation;
pub mod graph;
pub mod intel_usm;
pub mod legacy;
pub mod metal;
pub mod placement;
pub mod routing;
pub mod shared_event;
pub mod tensor_registry;
pub mod unified_arena;
pub mod lane_executor_registry;

// Re-export trait types from prism-ecs-backend so the kernel is the
// canonical home for the trait surface (BackendCapabilities,
// TensorBackend, MatmulOp, RmsNormOp, RoPEOp, DType, TensorHandle,
// QuantizedWeightHandle, EvaluationReceipt, ReadbackReceipt).
// Engine callers that previously imported these from
// `crate::ecs::backend::*` can now use `prism_ecs_kernel::backend::*`.
pub use prism_ecs_backend::{
    BackendCapabilities, DType, EvaluationReceipt, MatmulOp, QuantizedMatmulOp,
    QuantizedWeightHandle, ReadbackReceipt, RmsNormOp, RoPEOp, TensorBackend, TensorHandle,
};
