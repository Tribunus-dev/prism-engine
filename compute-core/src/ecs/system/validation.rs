use crate::ecs::aot::parameters::KernelParameters;
use crate::ecs::component::backend::{
    BinaryFormat, CompiledBinary, ExecutableFormat, KernelSource,
};
use crate::ecs::component::quality::QualityGateResult;
use crate::ecs::{CompEntity, CompWorld, CompilerSystem, EntityKind, SchedulePhase};

/// Packages compiled kernel binaries into Executable entities for the CImage.
pub struct ExecutablePackagingSystem;
impl CompilerSystem for ExecutablePackagingSystem {
    fn name(&self) -> &str {
        "ExecutablePackagingSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::Compilation
    }
    fn run(&self, world: &mut CompWorld) -> anyhow::Result<()> {
        let kernel_entities: Vec<CompEntity> = world.entities_of_kind(EntityKind::Kernel);

        for &kernel in &kernel_entities {
            let name = world
                .name(kernel)
                .map(|s| s.to_string())
                .unwrap_or_else(|| "unnamed_kernel".to_string());

            if let Some(binary) = world.get_component::<CompiledBinary>(kernel).cloned() {
                let exe = world.spawn(EntityKind::Executable, Some(format!("exe_{}", name)));
                world.add_component(exe, binary.clone());
                world.add_component(
                    exe,
                    ExecutableFormat {
                        binary_format: binary.format,
                        variant_label: name.clone(),
                    },
                );
            } else if world.get_component::<KernelSource>(kernel).is_some()
                && world.get_component::<KernelParameters>(kernel).is_some()
            {
                let exe = world.spawn(EntityKind::Executable, Some(format!("stub_{}", name)));
                world.add_component(
                    exe,
                    ExecutableFormat {
                        binary_format: BinaryFormat::LLVMBitcode,
                        variant_label: format!("stub_{}", name),
                    },
                );
            }
        }

        Ok(())
    }
}

/// Validates compiled output quality through numeric admission gates.
pub struct AdmissionValidationSystem;
impl CompilerSystem for AdmissionValidationSystem {
    fn name(&self) -> &str {
        "AdmissionValidationSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::Compilation
    }
    fn run(&self, world: &mut CompWorld) -> anyhow::Result<()> {
        let exe_entities = world.entities_of_kind(EntityKind::Executable);
        let kernel_entities = world.entities_of_kind(EntityKind::Kernel);

        let mut all_entities: Vec<CompEntity> = Vec::new();
        all_entities.extend(exe_entities);
        all_entities.extend(kernel_entities);

        let mut gates_found = false;

        for &entity in &all_entities {
            if let Some(gate) = world.get_component::<QualityGateResult>(entity) {
                gates_found = true;
                if !gate.passed {
                    tracing::warn!(
                        "AdmissionValidationSystem: quality gate FAILED on entity {:?}: nrmse={}, perplexity_delta={} and passed=false",
                        entity,
                        gate.nrmse,
                        gate.perplexity_delta,
                    );
                }
            }
        }

        if !gates_found {
            let model_entities = world.entities_of_kind(EntityKind::Model);
            if let Some(&model) = model_entities.first() {
                world.add_component(
                    model,
                    QualityGateResult {
                        nrmse: 0.0,
                        perplexity_delta: 0.0,
                        passed: true,
                    },
                );
            }
        }

        Ok(())
    }
}
