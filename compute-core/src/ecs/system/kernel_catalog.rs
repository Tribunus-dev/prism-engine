//! Validates each Kernel entity's `CompiledBinary` against the catalog schema.
//!
//! Port of `CatalogValidator::validate_catalog()` coherency checks applied
//! at the entity level rather than the whole-catalog level. Runs in Phase E
//! (`KernelGeneration`) and attaches a `CatalogEntry` component.

use crate::ecs::component::aot::CatalogEntry;
use crate::ecs::component::backend::{BinaryFormat, CompiledBinary};
use crate::ecs::{CompEntity, CompWorld, CompilerSystem, EntityKind, SchedulePhase};

/// Validates each kernel binary against the catalog schema.
///
/// Checks applied per entity:
/// 1. Binary data is non-empty.
/// 2. Fingerprint is present.
/// 3. Format is a recognized binary format.
pub struct KernelCatalogSystem;

impl CompilerSystem for KernelCatalogSystem {
    fn name(&self) -> &str {
        "KernelCatalogSystem"
    }

    fn phase(&self) -> SchedulePhase {
        SchedulePhase::KernelGeneration
    }

    fn run(&self, world: &mut CompWorld) -> anyhow::Result<()> {
        let kernels: Vec<CompEntity> = world.entities_of_kind(EntityKind::Kernel);

        for &kernel in &kernels {
            let mut errors: Vec<String> = Vec::new();

            let valid = if let Some(binary) = world.get_component::<CompiledBinary>(kernel) {
                // Check 1: binary data is non-empty
                if binary.data.is_empty() {
                    errors.push("compiled binary data is empty".into());
                }
                // Check 2: fingerprint is present
                if binary.fingerprint.is_empty() {
                    errors.push("compiled binary fingerprint is missing".into());
                }
                // Check 3: format is recognized (all variants are valid here)
                match binary.format {
                    BinaryFormat::Metallib
                    | BinaryFormat::HSACO
                    | BinaryFormat::SPIRV
                    | BinaryFormat::LLVMBitcode => {}
                }

                errors.is_empty()
            } else {
                // No compiled binary yet — not an error at this phase; the
                // binary is produced later in Phase F (Compilation).
                true
            };

            world.add_component(kernel, CatalogEntry { valid, errors });
        }

        Ok(())
    }
}

#[cfg(test)]
#[allow(unused_imports)]
mod tests {
    use super::*;
    use crate::ecs::component::backend::{BinaryFormat, CompiledBinary};
    use crate::ecs::{CompWorld, EntityKind};

    #[cfg(feature = "legacy_mutations")]
    #[test]
    fn test_valid_binary_passes_catalog_check() {
        let mut world = CompWorld::new();
        let kernel = world.spawn(EntityKind::Kernel, None);
        world.add_component(
            kernel,
            CompiledBinary {
                format: BinaryFormat::Metallib,
                data: vec![0xde, 0xad],
                fingerprint: "abc123".into(),
            },
        );

        let system = KernelCatalogSystem;
        system.run(&mut world).unwrap();

        let entry = world.get_component::<CatalogEntry>(kernel).unwrap();
        assert!(entry.valid, "expected valid catalog entry");
        assert!(entry.errors.is_empty());
    }

    #[cfg(feature = "legacy_mutations")]
    #[test]
    fn test_empty_binary_fails() {
        let mut world = CompWorld::new();
        let kernel = world.spawn(EntityKind::Kernel, None);
        world.add_component(
            kernel,
            CompiledBinary {
                format: BinaryFormat::Metallib,
                data: vec![],
                fingerprint: "abc123".into(),
            },
        );

        let system = KernelCatalogSystem;
        system.run(&mut world).unwrap();

        let entry = world.get_component::<CatalogEntry>(kernel).unwrap();
        assert!(!entry.valid, "expected invalid catalog entry");
        assert!(!entry.errors.is_empty());
    }

    #[cfg(feature = "legacy_mutations")]
    #[test]
    fn test_no_binary_is_ok() {
        let mut world = CompWorld::new();
        world.spawn(EntityKind::Kernel, None);

        let system = KernelCatalogSystem;
        system.run(&mut world).unwrap();

        // Kernels without a CompiledBinary are considered valid at generation time.
    }

    #[cfg(feature = "legacy_mutations")]
    #[test]
    fn test_missing_fingerprint_fails() {
        let mut world = CompWorld::new();
        let kernel = world.spawn(EntityKind::Kernel, None);
        world.add_component(
            kernel,
            CompiledBinary {
                format: BinaryFormat::HSACO,
                data: vec![0x01, 0x02],
                fingerprint: String::new(),
            },
        );

        let system = KernelCatalogSystem;
        system.run(&mut world).unwrap();

        let entry = world.get_component::<CatalogEntry>(kernel).unwrap();
        assert!(!entry.valid);
        assert!(entry.errors.iter().any(|e| e.contains("fingerprint")));
    }
}
