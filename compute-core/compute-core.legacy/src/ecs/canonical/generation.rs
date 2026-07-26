//! Generation types — what a compiler run produces.
//!
//! A `CimageGeneration` is the canonical output of one compiler invocation:
//! a resolved execution image with fully-specified tensor, kernel, and engram
//! bindings. Every binding carries its own receipt chain.
//!
//! Re-exported from `prism-ecs-ir` (phase 2 of compute-core dependency removal).

pub use prism_ecs_ir::cimage_types::{
    CimageGeneration, EngramBinding, RegionId, RepresentationBinding,
};
