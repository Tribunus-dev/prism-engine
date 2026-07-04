//! Level 1 of the distill-compiler — Metal + Accelerate only.
//!
//! Level 1 is the baseline that must work even if Core ML is unavailable, fails
//! model compilation, or cannot represent a teacher region. The teacher path
//! runs with dense Metal kernels or a compatible existing dense backend.
//! The candidate student path runs through the actual ternary page640 Metal
//! kernels intended for runtime.
//!
//! Accelerate owns all control-plane numerical work: moment accumulation, Gram
//! or Hessian-diagonal estimates, threshold selection, per-page and per-channel
//! scale solves, sidecar ranking, deterministic reductions, and receipt hashing.

pub mod checkpoint;
pub mod gates;
pub mod reducer;
pub mod scheduler;
pub mod student;
pub mod teacher;
