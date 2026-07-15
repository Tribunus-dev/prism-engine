//! Prism Metal runtime — region encoding, PSO caching, and fusion lowering
//! for Metal GPU dispatch.

pub mod fusion_lowering;
pub mod pso_cache;
pub mod region_encoder;

pub use fusion_lowering::{
    derive_function_constants, metal_lower_fused_group, metal_lower_to_kernel_group,
    validate_lowered, FusionPatternId, MetalLoweringError,
};
pub use pso_cache::{PsoCache, PsoEntry, PsoKey};
pub use region_encoder::{MetalPipelineState, MetalRegionEncoder};
