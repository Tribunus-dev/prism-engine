//! Execution plan types — local copy for crate-internal use.
//!
//! These types live in `tribunus-compute-core` and will be migrated to
//! `prism-ecs-core` once the extraction dependency chain is resolved.

use serde::{Deserialize, Serialize};

/// Codec family — quantization representation format.
///
/// NOTE: This is a local copy of the enum from `tribunus_compute_core::execution_plan::CodecFamily`.
/// Keep in sync until the type is migrated to `prism-ecs-core`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[allow(non_camel_case_types)]
#[derive(Default)]
pub enum CodecFamily {
    Nf4,
    Int8,
    Fp16,
    #[default]
    RawF32,
    SymInt4,
    Ternary,
    Ternary1_58,
    Mixed,
    Q8_0,
    #[allow(non_camel_case_types)]
    Q4_K,
    #[allow(non_camel_case_types)]
    Q2_K,
    #[allow(non_camel_case_types)]
    IQ2_XXS,
}
