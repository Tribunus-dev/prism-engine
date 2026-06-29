//! PRISM-CIMAGE-HETEROGENEOUS-COMPILATION-0001
//!
//! Heterogeneous execution image section — the compiler-emitted primary
//! artifact for tri-lane (Metal GPU / Core ML ANE / Accelerate CPU) execution.
//!
//! This module defines [`HeterogeneousExecutionImage`] and all supporting
//! types for the compiler-owned executable plan. Every cimage intended for
//! Prism Engine serving must contain a `HeterogeneousExecutionImage`.
//!
//! The runtime consumes this image directly via the heterogeneous runtime
//! — it does not reconstruct backend placement, resource ownership, or
//! concurrency semantics from disconnected manifests.
//!
//! # Structure
//!
//! * [`types`] — All type definitions
//! * [`builder`] — Construction helpers for building the image

pub mod builder;
pub mod types;

pub use builder::*;
pub use types::*;
