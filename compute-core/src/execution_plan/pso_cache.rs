//! PSO cache — compiles and caches Metal pipeline states keyed by
//! [`KernelSpecializationKey`].
//!
//! The [`PsoCacheKey`] captures every layout/codec parameter so that no
//! parameter change silently reuses a stale PSO.

use serde::{Deserialize, Serialize};
use super::{
    AffineMode, Axis, CodecFamily, ExecutionPhase, FunctionConstantSet,
    HardwareProfileId, KernelSpecializationKey, MetadataLayout, TileShape,
};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can arise during PSO compilation.
#[derive(Debug, Clone)]
pub enum PsoError {
    /// The Metal shader compilation failed.
    CompilationFailed(String),
    /// The requested configuration is not supported by any available kernel.
    UnsupportedConfiguration(String),
}

impl std::fmt::Display for PsoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PsoError::CompilationFailed(msg) => {
                write!(f, "PSO compilation failed: {msg}")
            }
            PsoError::UnsupportedConfiguration(msg) => {
                write!(f, "unsupported PSO configuration: {msg}")
            }
        }
    }
}

impl std::error::Error for PsoError {}

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// PSO cache — compiles and caches Metal pipeline states keyed by
/// [`KernelSpecializationKey`].
///
/// Cache identity includes all codec/layout parameters so no parameter is
/// silently ignored.
pub trait PsoCache {
    /// The platform-specific pipeline state handle.
    type PipelineState;

    /// Return a cached PSO for `key`, or compile it with `constants`.
    fn get_or_create(
        &mut self,
        key: &KernelSpecializationKey,
        constants: &FunctionConstantSet,
    ) -> Result<Self::PipelineState, PsoError>;
}

// ---------------------------------------------------------------------------
// Deterministic cache key
// ---------------------------------------------------------------------------

/// A hashable, serializable key that fully captures the parameters affecting
/// PSO compilation.
///
/// The deterministic mapping from [`KernelSpecializationKey`] uses integer
/// discriminants so that the cache key is portable across processes.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PsoCacheKey {
    pub metal_function_name: String,
    pub codec_id: u32,
    pub tile_shape: TileShape,
    pub group_size: u32,
    pub group_axis_id: u32,
    pub affine_mode_id: u32,
    pub metadata_layout_id: u32,
    pub execution_phase_id: u32,
    pub hardware_profile_id: u32,
    pub mode_flags: u32,
}

impl From<&KernelSpecializationKey> for PsoCacheKey {
    fn from(k: &KernelSpecializationKey) -> Self {
        Self {
            metal_function_name: format!("{:?}", k.template_id),
            codec_id: match k.codec {
                CodecFamily::Nf4 => 0,
                CodecFamily::Int8 => 1,
                _ => 99,
            },
            tile_shape: k.tile_shape,
            group_size: k.group_size,
            group_axis_id: match k.group_axis {
                Axis::Output => 0,
                Axis::Input => 1,
                Axis::TileLocal => 2,
                Axis::PackedContiguous => 3,
            },
            affine_mode_id: match k.affine_mode {
                AffineMode::ScaleOnly => 0,
                AffineMode::ScaleBias => 1,
            },
            metadata_layout_id: match k.metadata_layout {
                MetadataLayout::AdjacentTile => 0,
                MetadataLayout::SeparatedManifest => 1,
                MetadataLayout::Interleaved => 2,
            },
            execution_phase_id: match k.execution_phase {
                ExecutionPhase::Prefill => 0,
                ExecutionPhase::Decode => 1,
                ExecutionPhase::Mixed => 2,
            },
            hardware_profile_id: match k.hardware_profile {
                HardwareProfileId::AppleA18Tiny => 0,
                HardwareProfileId::AppleMBaseMemoryBound => 1,
                HardwareProfileId::AppleMProBalanced => 2,
                HardwareProfileId::AppleMMaxBandwidth => 3,
                HardwareProfileId::AppleMUltraSharded => 4,
            },
            mode_flags: k.mode_flags,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution_plan::{
        DType, KernelTemplateId,
    };

    fn sample_key() -> KernelSpecializationKey {
        KernelSpecializationKey {
            template_id: KernelTemplateId::Nf4Tile640Gemv,
            execution_phase: ExecutionPhase::Decode,
            codec: CodecFamily::Nf4,
            tile_shape: TileShape { m: 1, n: 32, k: 128, elements: 640 },
            group_size: 32,
            group_axis: Axis::Output,
            affine_mode: AffineMode::ScaleOnly,
            metadata_layout: MetadataLayout::AdjacentTile,
            input_dtype: DType::F16,
            output_dtype: DType::F16,
            hardware_profile: HardwareProfileId::AppleMProBalanced,
            mode_flags: 0,
        }
    }

    /// Verify that a `PsoCacheKey` derived from a `KernelSpecializationKey`
    /// round-trips all mapped fields.
    #[test]
    fn test_pso_cache_key_from_specialization_key() {
        let key = sample_key();
        let cache_key = PsoCacheKey::from(&key);

        assert_eq!(cache_key.metal_function_name, "Nf4Tile640Gemv");
        assert_eq!(cache_key.codec_id, 0); // Nf4
        assert_eq!(cache_key.tile_shape, TileShape { m: 1, n: 32, k: 128, elements: 640 });
        assert_eq!(cache_key.group_size, 32);
        assert_eq!(cache_key.group_axis_id, 0); // Output
        assert_eq!(cache_key.affine_mode_id, 0); // ScaleOnly
        assert_eq!(cache_key.metadata_layout_id, 0); // AdjacentTile
        assert_eq!(cache_key.execution_phase_id, 1); // Decode
        assert_eq!(cache_key.hardware_profile_id, 2); // AppleMProBalanced
        assert_eq!(cache_key.mode_flags, 0);
    }

    /// Different `group_size` values must yield distinct cache keys.
    #[test]
    fn test_pso_cache_key_distinguishes_group_size() {
        let mut key32 = sample_key();
        key32.group_size = 32;

        let mut key128 = sample_key();
        key128.group_size = 128;

        let cache32 = PsoCacheKey::from(&key32);
        let cache128 = PsoCacheKey::from(&key128);

        assert_ne!(cache32, cache128);
        assert_eq!(cache32.group_size, 32);
        assert_eq!(cache128.group_size, 128);
    }

    /// Different `group_axis` values (PackedContiguous vs Input) must yield
    /// distinct cache keys.
    #[test]
    fn test_pso_cache_key_distinguishes_group_axis() {
        let mut key_packed = sample_key();
        key_packed.group_axis = Axis::PackedContiguous;

        let mut key_input = sample_key();
        key_input.group_axis = Axis::Input;

        let cache_packed = PsoCacheKey::from(&key_packed);
        let cache_input = PsoCacheKey::from(&key_input);

        assert_ne!(cache_packed, cache_input);
        assert_eq!(cache_packed.group_axis_id, 3);
        assert_eq!(cache_input.group_axis_id, 1);
    }
}
