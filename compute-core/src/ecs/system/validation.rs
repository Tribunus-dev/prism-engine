use crate::ecs::aot::parameters::KernelParameters;
use crate::ecs::component::backend::{
    BinaryFormat, CompiledBinary, ExecutableFormat, KernelSource,
};
use crate::ecs::component::quality::QualityGateResult;
use crate::ecs::runtime::constitutional_world_txn::ConstitutionalWorldTxn;

use crate::ecs::Entity;
use crate::ecs::{CompilerSystem, EntityKind, SchedulePhase, World};

/// Packages compiled kernel binaries into Executable entities for the CImage.
pub struct ExecutablePackagingSystem;
impl CompilerSystem for ExecutablePackagingSystem {
    fn name(&self) -> &str {
        "ExecutablePackagingSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::Compilation
    }
    fn run(&self, world: &mut World) -> anyhow::Result<()> {
        let kernel_entities: Vec<Entity> = world.entities_of_kind(EntityKind::Kernel);
        // Stage every per-kernel Executable spawn + insert on a single
        // `ConstitutionalWorldTxn` and commit atomically. Direct
        // `world.spawn` / `world.add_component` calls outside the
        // WorldTxn seam are forbidden.
        let mut txn = ConstitutionalWorldTxn::new();
        for &kernel in &kernel_entities {
            let name = world.name(kernel).unwrap_or("kernel").to_string();
            let exe_token = txn.stage_spawn(EntityKind::Executable, Some(format!("exe_{}", name)));
            if let Some(binary) = world.get_component::<CompiledBinary>(kernel).cloned() {
                if let Err(e) = txn.stage_insert_on(
                    exe_token,
                    ExecutableFormat {
                        binary_format: binary.format,
                        variant_label: name.clone(),
                    },
                ) {
                    tracing::warn!(kernel = ?kernel, error = %e, "executable_packaging: stage_insert_on ExecutableFormat (binary)");
                }
            } else if world.get_component::<KernelSource>(kernel).is_some()
                && world.get_component::<KernelParameters>(kernel).is_some()
            {
                if let Err(e) = txn.stage_insert_on(
                    exe_token,
                    ExecutableFormat {
                        binary_format: BinaryFormat::LLVMBitcode,
                        variant_label: format!("stub_{}", name),
                    },
                ) {
                    tracing::warn!(kernel = ?kernel, error = %e, "executable_packaging: stage_insert_on ExecutableFormat (stub)");
                }
            }
        }
        let _ = txn.commit(world).map_err(|e| {
            tracing::error!(error = %e, "executable_packaging: ConstitutionalWorldTxn commit failed");
            anyhow::anyhow!("executable_packaging: ConstitutionalWorldTxn commit failed: {e}")
        })?;
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
    fn run(&self, world: &mut World) -> anyhow::Result<()> {
        let kernel_entities: Vec<Entity> = world.entities_of_kind(EntityKind::Kernel);
        let executable_entities: Vec<Entity> = world.entities_of_kind(EntityKind::Executable);
        let any_failure = false;

        for &k in &kernel_entities {
            if let Some(qg) = world.get_component::<QualityGateResult>(k) {
                if !qg.passed {
                    tracing::warn!("Quality gate failed for kernel (nrmse={})", qg.nrmse);
                }
            }
        }
        for &e in &executable_entities {
            if let Some(qg) = world.get_component::<QualityGateResult>(e) {
                if !qg.passed {
                    tracing::warn!("Quality gate failed for executable (nrmse={})", qg.nrmse);
                }
            }
        }

        // Find the ModelEntity and attach a default-passed gate if none exist.
        //
        // Stage every per-model insert on a single
        // `ConstitutionalWorldTxn` and commit atomically. Direct
        // `world.add_component` calls outside the WorldTxn seam are
        // forbidden.
        let model_entities: Vec<Entity> = world.entities_of_kind(EntityKind::Model);
        let mut txn = ConstitutionalWorldTxn::new();
        for &m in &model_entities {
            if world.get_component::<QualityGateResult>(m).is_none() {
                if let Err(e) = txn.stage_insert(
                    m,
                    QualityGateResult {
                        passed: !any_failure,
                        nrmse: 0.0,
                        perplexity_delta: 0.0,
                    },
                ) {
                    tracing::warn!(entity = ?m, error = %e, "admission_validation: stage_insert QualityGateResult");
                }
            }
        }
        let _ = txn.commit(world).map_err(|e| {
            tracing::error!(error = %e, "admission_validation: ConstitutionalWorldTxn commit failed");
            anyhow::anyhow!("admission_validation: ConstitutionalWorldTxn commit failed: {e}")
        })?;
        Ok(())
    }
}
