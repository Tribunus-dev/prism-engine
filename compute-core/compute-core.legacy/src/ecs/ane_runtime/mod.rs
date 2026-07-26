//! ANE runtime — planar engine program types and lowering from FusedGroup.
//!
//! Planar lowering translates the fusion-scheduler's `FusedGroup` into a
//! `PlanarProgramDescriptor` that the ANE planar engine can execute directly.
//! This is Phase 5 of the fusion compiler IR pipeline.
//!
//! Supported patterns:
//! - FP16 matmul
//! - INT8 bridge projection
//! - matmul+add / matmul+elementwise fusion
//! - gate/up → SiLU fusion (MLP)
//!
//! Rejected patterns (fail closed with a typed reason):
//! - NF4, SymInt4, Ternary codecs
//! - Cross-lane IOSurface dependencies inside a single group

pub mod planar_lowering;
