//! Scores and selects the best variant per parent kernel group.
//!
//! Port of `KernelVariantSelector::select_variant()` — groups `KernelVariant`
//! entities by parent kernel, scores each variant using a heuristic that
//! prefers exact profile matches and same-generation matches over generic
//! fallbacks. Attaches a `SelectedVariant` component to each parent kernel.
//! Runs in Phase E (`KernelGeneration`).

use crate::ecs::component::aot::{KernelVariantEntityData, SelectedVariant};
use crate::ecs::component::backend::GPUArch;
#[allow(unused_imports)]
use crate::ecs::Entity;
use crate::ecs::{CompWorld, CompilerSystem, EntityKind, SchedulePhase};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Stable profile identifier for Apple Silicon hardware.
///
/// Coarse enough for kernel variant selection, but not so coarse that
/// M1 and M4 Max collide on the same entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum AppleSiliconProfileId {
    M1,
    M1Pro,
    M1Max,
    M1Ultra,
    M2,
    M2Pro,
    M2Max,
    M2Ultra,
    M3,
    M3Pro,
    M3Max,
    M3Ultra,
    M4,
    M4Pro,
    M4Max,
    M4Ultra,
    M5,
    M5Pro,
    M5Max,
    M5Ultra,
    /// Fallback for unrecognized Apple Silicon or non-Apple GPUs.
    UnknownAppleSilicon,
}

impl AppleSiliconProfileId {
    /// SOC generation: M1-family, M2-family, etc.
    pub fn soc_generation(self) -> u32 {
        match self {
            Self::M1 | Self::M1Pro | Self::M1Max | Self::M1Ultra => 1,
            Self::M2 | Self::M2Pro | Self::M2Max | Self::M2Ultra => 2,
            Self::M3 | Self::M3Pro | Self::M3Max | Self::M3Ultra => 3,
            Self::M4 | Self::M4Pro | Self::M4Max | Self::M4Ultra => 4,
            Self::M5 | Self::M5Pro | Self::M5Max | Self::M5Ultra => 5,
            Self::UnknownAppleSilicon => 0,
        }
    }

    /// Human-readable marketing name.
    pub fn marketing_name(self) -> &'static str {
        match self {
            Self::M1 => "Apple M1",
            Self::M1Pro => "Apple M1 Pro",
            Self::M1Max => "Apple M1 Max",
            Self::M1Ultra => "Apple M1 Ultra",
            Self::M2 => "Apple M2",
            Self::M2Pro => "Apple M2 Pro",
            Self::M2Max => "Apple M2 Max",
            Self::M2Ultra => "Apple M2 Ultra",
            Self::M3 => "Apple M3",
            Self::M3Pro => "Apple M3 Pro",
            Self::M3Max => "Apple M3 Max",
            Self::M3Ultra => "Apple M3 Ultra",
            Self::M4 => "Apple M4",
            Self::M4Pro => "Apple M4 Pro",
            Self::M4Max => "Apple M4 Max",
            Self::M4Ultra => "Apple M4 Ultra",
            Self::M5 => "Apple M5",
            Self::M5Pro => "Apple M5 Pro",
            Self::M5Max => "Apple M5 Max",
            Self::M5Ultra => "Apple M5 Ultra",
            Self::UnknownAppleSilicon => "Unknown Apple Silicon",
        }
    }
}

impl fmt::Display for AppleSiliconProfileId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            self.marketing_name().to_lowercase().replace(' ', "_")
        )
    }
}

/// Selects the best variant per parent kernel via heuristic scoring.
pub struct VariantSelectionSystem;

/// Parse a Display-formatted `AppleSiliconProfileId` string back into the enum.
fn profile_from_str(s: &str) -> Option<AppleSiliconProfileId> {
    match s {
        "apple_m1" => Some(AppleSiliconProfileId::M1),
        "apple_m1_pro" => Some(AppleSiliconProfileId::M1Pro),
        "apple_m1_max" => Some(AppleSiliconProfileId::M1Max),
        "apple_m1_ultra" => Some(AppleSiliconProfileId::M1Ultra),
        "apple_m2" => Some(AppleSiliconProfileId::M2),
        "apple_m2_pro" => Some(AppleSiliconProfileId::M2Pro),
        "apple_m2_max" => Some(AppleSiliconProfileId::M2Max),
        "apple_m2_ultra" => Some(AppleSiliconProfileId::M2Ultra),
        "apple_m3" => Some(AppleSiliconProfileId::M3),
        "apple_m3_pro" => Some(AppleSiliconProfileId::M3Pro),
        "apple_m3_max" => Some(AppleSiliconProfileId::M3Max),
        "apple_m3_ultra" => Some(AppleSiliconProfileId::M3Ultra),
        "apple_m4" => Some(AppleSiliconProfileId::M4),
        "apple_m4_pro" => Some(AppleSiliconProfileId::M4Pro),
        "apple_m4_max" => Some(AppleSiliconProfileId::M4Max),
        "apple_m4_ultra" => Some(AppleSiliconProfileId::M4Ultra),
        "apple_m5" => Some(AppleSiliconProfileId::M5),
        "apple_m5_pro" => Some(AppleSiliconProfileId::M5Pro),
        "apple_m5_max" => Some(AppleSiliconProfileId::M5Max),
        "apple_m5_ultra" => Some(AppleSiliconProfileId::M5Ultra),
        "unknown_apple_silicon" => Some(AppleSiliconProfileId::UnknownAppleSilicon),
        _ => None,
    }
}

/// Determine the target device profile from GPU arch info on kernel entities.
fn target_device_profile(world: &CompWorld) -> Option<AppleSiliconProfileId> {
    let kernels = world.entities_of_kind(EntityKind::Kernel);
    for &k in &kernels {
        if let Some(arch) = world.get_component::<GPUArch>(k) {
            return match arch.compute_units {
                0..=10 => Some(AppleSiliconProfileId::M1),
                11..=20 => Some(AppleSiliconProfileId::M2),
                21..=30 => Some(AppleSiliconProfileId::M3),
                31..=50 => Some(AppleSiliconProfileId::M4),
                _ => Some(AppleSiliconProfileId::M4Max),
            };
        }
    }
    None
}

/// Score a variant relative to the detected device profile.
///
/// Exact profile match → 1.0
/// Same SoC generation  → 0.5
/// All other            → 0.25
/// No device info       → 0.1
fn score_variant(
    variant: &KernelVariantEntityData,
    device_profile: &Option<AppleSiliconProfileId>,
) -> f64 {
    let Some(device) = device_profile else {
        return 0.1;
    };

    let Some(variant_profile) = profile_from_str(&variant.profile_id) else {
        return 0.1;
    };

    if variant_profile == *device {
        1.0
    } else if variant_profile.soc_generation() == device.soc_generation() {
        0.5
    } else {
        0.25
    }
}

impl CompilerSystem for VariantSelectionSystem {
    fn name(&self) -> &str {
        "VariantSelectionSystem"
    }

    fn phase(&self) -> SchedulePhase {
        SchedulePhase::KernelGeneration
    }

    fn run(&self, world: &mut CompWorld) -> anyhow::Result<()> {
        let variant_entities: Vec<Entity> = world.entities_of_kind(EntityKind::KernelVariant);

        if variant_entities.is_empty() {
            return Ok(());
        }

        // Group variants by parent kernel.
        let mut groups: HashMap<Entity, Vec<(Entity, KernelVariantEntityData)>> = HashMap::new();
        for &entity in &variant_entities {
            if let Some(data) = world
                .get_component::<KernelVariantEntityData>(entity)
                .cloned()
            {
                let parent = Entity(data.parent_kernel.0, 0);
                groups.entry(parent).or_default().push((entity, data));
            }
        }

        if groups.is_empty() {
            return Ok(());
        }

        // Detect target device profile once.
        let device_profile = target_device_profile(world);

        // For each group, score and select the best variant.
        for (parent_kernel, variants) in &groups {
            let scored: Vec<f64> = variants
                .iter()
                .map(|(_, data)| score_variant(data, &device_profile))
                .collect();

            // Find the index of the best score.
            let best_idx = scored
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx);

            if let Some(idx) = best_idx {
                let best_data = &variants[idx].1;
                let score = scored[idx];
                world.add_component(
                    *parent_kernel,
                    SelectedVariant {
                        profile_id: best_data.profile_id.clone(),
                        score,
                    },
                );
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {

    #[cfg(feature = "legacy_mutations")]
    #[test]
    fn test_selects_variant_by_group() {
        let mut world = CompWorld::new();
        let parent = world.spawn(EntityKind::Kernel, None);
        let v1 = world.spawn(EntityKind::KernelVariant, None);
        world.add_component(
            v1,
            KernelVariantEntityData {
                profile_id: "apple_m1".into(),
                template_id: KernelTemplateId::Nf4Tile640Gemv,
                parent_kernel: CompEntityRef(parent.0),
            },
        );
        let v2 = world.spawn(EntityKind::KernelVariant, None);
        world.add_component(
            v2,
            KernelVariantEntityData {
                profile_id: "apple_m4_max".into(),
                template_id: KernelTemplateId::Fp16Matmul,
                parent_kernel: CompEntityRef(parent.0),
            },
        );

        let system = VariantSelectionSystem;
        system.run(&mut world).unwrap();

        let selected = world.get_component::<SelectedVariant>(parent);
        assert!(
            selected.is_some(),
            "expected a SelectedVariant on the parent"
        );
    }

    #[cfg(feature = "legacy_mutations")]
    #[test]
    fn test_empty_variants_noop() {
        let mut world = CompWorld::new();
        let system = VariantSelectionSystem;
        system.run(&mut world).unwrap();
        // no panic
    }

    #[cfg(feature = "legacy_mutations")]
    #[test]
    fn test_profile_from_str_roundtrip() {
        let profiles = [
            AppleSiliconProfileId::M1,
            AppleSiliconProfileId::M1Pro,
            AppleSiliconProfileId::M4Max,
            AppleSiliconProfileId::M5Ultra,
            AppleSiliconProfileId::UnknownAppleSilicon,
        ];
        for p in profiles {
            let s = p.to_string();
            let back = profile_from_str(&s)
                .unwrap_or_else(|| panic!("failed to round-trip profile {p:?} (string: {s:?})"));
            assert_eq!(back, p);
        }
    }

    #[cfg(feature = "legacy_mutations")]
    #[test]
    fn test_score_variant_exact_match() {
        let data = KernelVariantEntityData {
            profile_id: "apple_m4_max".into(),
            template_id: KernelTemplateId::Nf4Tile640Gemv,
            parent_kernel: CompEntityRef(1),
        };
        let device = Some(AppleSiliconProfileId::M4Max);
        let score = score_variant(&data, &device);
        assert!(
            (score - 1.0).abs() < 1e-9,
            "expected 1.0 for exact match, got {score}"
        );
    }

    #[cfg(feature = "legacy_mutations")]
    #[test]
    fn test_score_variant_same_gen() {
        let data = KernelVariantEntityData {
            profile_id: "apple_m4".into(),
            template_id: KernelTemplateId::Nf4Tile640Gemv,
            parent_kernel: CompEntityRef(1),
        };
        let device = Some(AppleSiliconProfileId::M4Max);
        let score = score_variant(&data, &device);
        assert!(
            (score - 0.5).abs() < 1e-9,
            "expected 0.5 for same gen, got {score}"
        );
    }

    #[cfg(feature = "legacy_mutations")]
    #[test]
    fn test_score_variant_diff_gen() {
        let data = KernelVariantEntityData {
            profile_id: "apple_m1".into(),
            template_id: KernelTemplateId::Nf4Tile640Gemv,
            parent_kernel: CompEntityRef(1),
        };
        let device = Some(AppleSiliconProfileId::M4Max);
        let score = score_variant(&data, &device);
        assert!(
            (score - 0.25).abs() < 1e-9,
            "expected 0.25 for diff gen, got {score}"
        );
    }
}
