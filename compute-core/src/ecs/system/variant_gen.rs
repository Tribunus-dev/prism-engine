//! Generates KernelVariant entities for every (profile × template) combination.
//!
//! Port of `KernelVariantGenerator::generate_all()` — enumerates all target
//! profiles from the default AOT matrix and all known templates, spawning a
//! `KernelVariant` entity for each combination and linking it to the parent
//! kernel. Runs in Phase E (`KernelGeneration`).

use crate::ecs::component::aot::{CompEntityRef, KernelVariantEntityData};
use crate::ecs::component::fusion::FusionGroup;
use crate::ecs::plan::KernelTemplateId;

use crate::ecs::Entity;
use crate::ecs::{World, CompilerSystem, EntityKind, SchedulePhase};
use serde::{Deserialize, Serialize};
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

/// Generates a KernelVariant entity per (template, profile) combination.
pub struct VariantGenerationSystem;

/// All known kernel template identifiers.
const ALL_TEMPLATES: &[KernelTemplateId] = &[
    KernelTemplateId::Nf4Tile640Gemv,
    KernelTemplateId::Int8Tile640Gemv,
    KernelTemplateId::FusedGateUp,
    KernelTemplateId::FusedGateUpActivation,
    KernelTemplateId::FusedDownProjResidual,
    KernelTemplateId::FusedOProjResidual,
    KernelTemplateId::FusedRmsNormQkv,
    KernelTemplateId::FusedAttentionScoreProbe,
    KernelTemplateId::Gemma4FullInt4,
    KernelTemplateId::RawF32Matmul,
    KernelTemplateId::Fp16Matmul,
];

/// Returns the default set of target profiles from the AOT matrix.
fn target_profiles() -> Vec<AppleSiliconProfileId> {
    vec![
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
        AppleSiliconProfileId::UnknownAppleSilicon,
    ]
}

impl CompilerSystem for VariantGenerationSystem {
    fn name(&self) -> &str {
        "VariantGenerationSystem"
    }

    fn phase(&self) -> SchedulePhase {
        SchedulePhase::KernelGeneration
    }

    fn run(&self, world: &mut World) -> anyhow::Result<()> {
        // Identify dispatches that have FusionGroup info.
        let dispatch_entities: Vec<Entity> = world
            .entities_of_kind(EntityKind::Dispatch)
            .into_iter()
            .filter(|e| world.get_component::<FusionGroup>(*e).is_some())
            .collect();

        if dispatch_entities.is_empty() {
            return Ok(());
        }

        // Collect existing kernel entities to use as parents.
        let kernel_entities: Vec<Entity> = world.entities_of_kind(EntityKind::Kernel);

        // Fallback: if no kernel entities exist yet, create one.
        let fallback_kernel = if kernel_entities.is_empty() {
            Some(world.spawn(EntityKind::Kernel, Some("variant_parent".into())))
        } else {
            None
        };

        let profiles = target_profiles();

        for &dispatch in &dispatch_entities {
            let parent_kernel = if kernel_entities.is_empty() {
                // Use the fallback we created above.
                fallback_kernel.unwrap()
            } else {
                // Simple assignment: pick a kernel from the list (round-robin proxy).
                let idx = dispatch.0 as usize % kernel_entities.len();
                kernel_entities[idx]
            };

            for &template_id in ALL_TEMPLATES {
                for profile in &profiles {
                    let profile_str = profile.to_string();
                    let variant = world.spawn(
                        EntityKind::KernelVariant,
                        Some(format!("variant_{template_id:?}_{profile_str}")),
                    );
                    world.add_component(
                        variant,
                        KernelVariantEntityData {
                            profile_id: profile_str,
                            template_id,
                            parent_kernel: CompEntityRef(parent_kernel.0),
                        },
                    );
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {

    #[cfg(feature = "legacy_mutations")]
    #[test]
    fn test_generates_variants_for_dispatches() {
        let mut world = World::new();
        let dispatch = world.spawn(EntityKind::Dispatch, None);
        world.add_component(
            dispatch,
            FusionGroup {
                root_op_kind: "matmul".into(),
                fused_op_kinds: vec![],
                binding_slots: 4,
                accepted: true,
                reject_reason: None,
            },
        );
        let kernel = world.spawn(EntityKind::Kernel, None);

        let system = VariantGenerationSystem;
        system.run(&mut world).unwrap();

        let variants = world.entities_of_kind(EntityKind::KernelVariant);
        let expected_count = ALL_TEMPLATES.len() * 21; // 21 profiles
        assert_eq!(
            variants.len(),
            expected_count,
            "expected {expected_count} variants ({templates} templates × 21 profiles)",
            templates = ALL_TEMPLATES.len()
        );
    }

    #[cfg(feature = "legacy_mutations")]
    #[test]
    fn test_no_dispatches_no_variants() {
        let mut world = World::new();

        let system = VariantGenerationSystem;
        system.run(&mut world).unwrap();

        let variants = world.entities_of_kind(EntityKind::KernelVariant);
        assert!(variants.is_empty());
    }
}
