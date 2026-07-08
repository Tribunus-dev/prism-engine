//! CPU-first fusion backend — Accelerate + Rayon as a first-class candidate.
//!
//! This module defines the types and lowering pipeline for executing fused
//! operation groups on the CPU using Accelerate (vDSP / BLAS / BNNS) for
//! vectorized compute and Rayon for work-stealing parallelism.
//!
//! The backend is a first-class fusion candidate alongside Metal and ANE:
//! it participates in capability registration, fusion evaluation, and
/// lowering pipelines.

pub mod capabilities;
pub mod lowering;
pub mod rayon_strategy;
pub mod receipts;
