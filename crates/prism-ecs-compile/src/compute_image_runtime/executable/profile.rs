//! Executable target profile — binds a specific hardware/runtime
//! configuration to a shape-specialized program and residency plan.

use serde::{Deserialize, Serialize};

use crate::compute_image_runtime::residency::plan::CompiledResidencyPlan;
use crate::compute_image_runtime::executable::variant::{ShapeSpecializedProgram, ShapeSpecializedVariantId};
use crate::compute_image_runtime::ContentHash;

/// Opaque identifier for a target profile.
pub type TargetProfileId = String;

/// Executable target profile — binds a specific hardware/runtime
/// configuration to a shape-specialized program and residency plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutableTargetProfile {
    /// Profile identifier.
    pub profile_id: TargetProfileId,
    /// Content hash of the profile.
    pub profile_hash: ContentHash,
    /// Hardware contract (GPU/ANE/unified memory).
    pub hardware_contract: HardwareTargetContract,
    /// Runtime contract (min OS, feature flags).
    pub runtime_contract: RuntimeTargetContract,
    /// Shape-specialized program variants for this profile.
    pub shape_variants: Vec<ShapeSpecializedProgram>,
    /// Residency plans for each variant.
    pub residency_plans: Vec<CompiledResidencyPlan>,
    /// Default variant selection for decode / prefill.
    pub default_variant_selection: DefaultVariantSelection,
}

/// Hardware target contract — describes the physical device the
/// executable is built for.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareTargetContract {
    /// Hardware family identifier (e.g. "apple-silicon-m4").
    pub hardware_family: String,
    /// Number of GPU cores.
    pub gpu_core_count: u32,
    /// Number of ANE cores.
    pub ane_count: u32,
    /// Whether the device has unified CPU/GPU memory.
    pub has_unified_memory: bool,
    /// Maximum threadgroup size supported.
    pub max_threadgroup_size: u32,
}

/// Runtime target contract — describes the minimum runtime environment
/// the executable requires.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeTargetContract {
    /// Minimum OS version string (e.g. "14.0").
    pub min_os_version: String,
    /// Required runtime feature flags.
    pub feature_flags: Vec<String>,
}

/// Default variant selection — which variant to use for decode and
/// prefill by default.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefaultVariantSelection {
    /// Variant id used for single-token decode.
    pub decode_variant_id: ShapeSpecializedVariantId,
    /// Variant id used for prefill.
    pub prefill_variant_id: ShapeSpecializedVariantId,
}
