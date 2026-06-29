//! Accelerate CPU backend — vDSP, BLAS, BNNS, vForce.
//!
//! F32 matmul wired via cblas_sgemm.  All other primitives return
//! "not yet implemented" until native bindings are added.

pub mod ffi;
mod ops;
pub use ops::*;
