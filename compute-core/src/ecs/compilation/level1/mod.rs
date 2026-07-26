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

// Std-only — compiles and unit-tests everywhere (Linux CI included). Its
// Metal-driving producer fns are cfg-stubbed internally.
pub mod kd_gate;

// Metal/Accelerate-coupled Level 1 pipeline — same reachability as when the
// whole `level1` module was `prism-backend`-gated.
#[cfg(feature = "prism-backend")]
pub mod checkpoint;
#[cfg(all(target_os = "macos", feature = "prism-backend"))]
pub mod gates;
#[cfg(feature = "prism-backend")]
pub mod reducer;
#[cfg(all(target_os = "macos", feature = "prism-backend"))]
pub mod scheduler;
#[cfg(feature = "prism-backend")]
pub mod student;
#[cfg(all(target_os = "macos", feature = "prism-backend"))]
pub mod teacher;
