//! Kernel variant generator — iterates over a target matrix of hardware profiles
//! and kernel families to produce `KernelParameters` sets for AOT compilation.

use serde::{Deserialize, Serialize};

use super::parameters::{DType, KernelFamily, KernelParameters};
use super::profile_db::AppleSiliconProfileDb;
use super::profile_id::AppleSiliconProfileId;

/// A target matrix for AOT kernel generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AotTargetMatrix {
    /// Which hardware profiles to target.
    pub targets: Vec<AppleSiliconProfileId>,
    /// Which kernel families to generate.
    pub kernel_families: Vec<KernelFamily>,
    /// Per-codec group sizes to try (default: [128, 256]).
    pub group_sizes: Option<Vec<u32>>,
    /// Tile widths to consider.
    pub tile_widths: Option<Vec<u32>>,
}

impl Default for AotTargetMatrix {
    fn default() -> Self {
        Self {
            targets: vec![
                AppleSiliconProfileId::M1,
                AppleSiliconProfileId::M1Pro,
                AppleSiliconProfileId::M1Max,
                AppleSiliconProfileId::M1Ultra,
                AppleSiliconProfileId::M2,
                AppleSiliconProfileId::M2Pro,
                AppleSiliconProfileId::M2Max,
                AppleSiliconProfileId::M2Ultra,
                AppleSiliconProfileId::M3,
                AppleSiliconProfileId::M3Pro,
                AppleSiliconProfileId::M3Max,
                AppleSiliconProfileId::M3Ultra,
                AppleSiliconProfileId::M4,
                AppleSiliconProfileId::M4Pro,
                AppleSiliconProfileId::M4Max,
                AppleSiliconProfileId::M4Ultra,
                AppleSiliconProfileId::M5,
                AppleSiliconProfileId::M5Pro,
                AppleSiliconProfileId::M5Max,
                AppleSiliconProfileId::M5Ultra,
            ],
            kernel_families: vec![
                KernelFamily::GemvNf4Tile,
                KernelFamily::GemvInt8Tile,
                KernelFamily::GemvTernaryTile,
                KernelFamily::GemvQ4_K,
                KernelFamily::GemvQ2_K,
                KernelFamily::GemvIQ2_XXS,
            ],
            group_sizes: Some(vec![128, 256]),
            tile_widths: Some(vec![512, 640, 768, 1024]),
        }
    }
}

/// Generator that produces concrete `KernelParameters` sets from a target matrix.
pub struct KernelVariantGenerator;

impl KernelVariantGenerator {
    /// Generate all `KernelParameters` sets for the given target matrix.
    ///
    /// For each target profile × kernel family combination, produces the
    /// optimal parameter set using heuristic estimation.
    pub fn generate_all(
        matrix: &AotTargetMatrix,
        db: &AppleSiliconProfileDb,
    ) -> Vec<KernelParameters> {
        let mut results = Vec::new();

        for target in &matrix.targets {
            let profile = match db.by_id(*target) {
                Some(p) => p,
                None => continue,
            };

            // Skip profiles that aren't actionable for AOT.
            if !profile.evidence_status.is_actionable_for_aot() {
                continue;
            }

            for family in &matrix.kernel_families {
                let params = estimate_parameters(profile, *family, matrix);
                results.push(params);
            }
        }

        results
    }
}

fn family_to_codec(family: KernelFamily) -> crate::execution_plan::CodecFamily {
    use crate::execution_plan::CodecFamily;
    match family {
        KernelFamily::GemvInt8Tile => CodecFamily::Int8,
        KernelFamily::GemvNf4Tile => CodecFamily::Nf4,
        KernelFamily::GemvTernaryTile => CodecFamily::Ternary1_58,
        KernelFamily::GemvQ8_0 => CodecFamily::Q8_0,
        KernelFamily::GemvQ4_K => CodecFamily::Q4_K,
        KernelFamily::GemvQ2_K => CodecFamily::Q2_K,
        KernelFamily::GemvIQ2_XXS => CodecFamily::IQ2_XXS,
        KernelFamily::CompressedAttention => CodecFamily::RawF32,
        _ => CodecFamily::Nf4,
    }
}

fn estimate_parameters(
    profile: &super::profile_db::AppleSiliconProfile,
    family: KernelFamily,
    matrix: &AotTargetMatrix,
) -> KernelParameters {
    let cu = profile.gpu.compute_units;
    let tgm = profile.gpu.max_threadgroup_memory_bytes;

    // Group size: prefer larger for ternary, default for others.
    let group_size = match family {
        KernelFamily::GemvTernaryTile => matrix
            .group_sizes
            .as_ref()
            .and_then(|gs| gs.iter().max().copied())
            .unwrap_or(256),
        _ => matrix
            .group_sizes
            .as_ref()
            .and_then(|gs| gs.iter().find(|&&g| g == 128).copied())
            .unwrap_or(128),
    };

    // Tile width: scale with CU count, clamp to available options.
    let target_tile = 256u32.max(cu * 16).min(1024);
    let tile_width = matrix
        .tile_widths
        .as_ref()
        .and_then(|tw| {
            tw.iter()
                .filter(|&&t| t <= target_tile)
                .max()
                .or_else(|| tw.iter().min())
                .copied()
        })
        .unwrap_or(640);

    let groups_per_tile = (tile_width + group_size - 1) / group_size;

    // Threadgroup size: prefer 64 on powerful GPUs, 32 otherwise.
    let threadgroup_size = if cu >= 24 { 64 } else { 32 };

    // Lane values: more on GPUs with large threadgroup memory.
    let lane_values = if tgm >= 64 * 1024 { 4 } else { 2 };

    // Unroll factor: prefer 4 on modern GPUs.
    let unroll_factor = if cu >= 16 { 4 } else { 2 };

    // Prefetch distance: more on wide memory bus.
    let prefetch_distance = if profile.memory.memory_bus_width_bits >= 512 {
        4
    } else {
        2
    };

    KernelParameters {
        kernel_family: family,
        codec_family: family_to_codec(family),
        tile_width,
        group_size,
        threadgroup_size,
        simdgroup_width: 32,
        groups_per_tile,
        lane_values,
        unroll_factor,
        use_threadgroup_memory: tgm >= 48 * 1024,
        prefetch_distance,
        accumulation_dtype: DType::Fp32,
        output_dtype: DType::Fp16,
    }
}
