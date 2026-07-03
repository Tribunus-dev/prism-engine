//! Level 2 — stateless Core ML teacher regions.
//!
//! The student remains Metal. Accelerate remains the control plane.
//! The key property is that the Core ML route is optional and replaceable:
//! the compiler must be able to retry any failed or unsupported teacher
//! region through the Level 1 dense Metal fallback without changing the
//! logical computation or receipt semantics.
//!
//! Each Core ML teacher region is stateless (no MLState in Level 2).
//! Stateless regions are restartable, independently hashable, and easy
//! to substitute with Level 1 fallback.

pub mod bridge;
pub mod scheduler;
pub mod gates;
