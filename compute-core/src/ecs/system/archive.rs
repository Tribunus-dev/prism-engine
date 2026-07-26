//! ECS-native ANE archive handling — wraps `compute_image::compile::archive`.
//!
//! Archives .mlmodelc directories into .ane.tar files for ANE deployment.

use std::path::{Path, PathBuf};

use crate::ecs::component::model_source::{AneArchiveComp, AneArchiveResultComp};
use crate::ecs::runtime::constitutional_world_txn::ConstitutionalWorldTxn;

use crate::ecs::{CompilerSystem, EntityKind, SchedulePhase, World};

/// Tar-archive a .mlmodelc directory into a single `.ane.tar` file.
pub fn archive_ane_modelc(src: &Path, dst: &Path) -> std::io::Result<()> {
    let file = std::fs::File::create(dst)?;
    let mut builder = tar::Builder::new(std::io::BufWriter::new(file));
    builder.append_dir_all(".", src)?;
    builder.finish()?;
    Ok(())
}

/// Scan a directory for pre-compiled .mlmodelc bundles, tar-archive each,
/// and write the archives to `output_dir`.
pub fn copy_precompiled_ane_models(src: &Path, output_dir: &Path) -> crate::Result<()> {
    let mut found = 0u32;
    for entry in std::fs::read_dir(src).map_err(|e| {
        crate::Error::from_reason(format!("read ane_models_dir {}: {e}", src.display()))
    })? {
        let entry =
            entry.map_err(|e| crate::Error::from_reason(format!("ane_models_dir entry: {e}")))?;
        let path = entry.path();
        if path.is_dir()
            && path
                .extension()
                .map(|ext| ext == "mlmodelc")
                .unwrap_or(false)
        {
            let stem = path
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned();
            let tar_path = output_dir.join(format!("{stem}.ane.tar"));
            archive_ane_modelc(&path, &tar_path).map_err(|e| {
                crate::Error::from_reason(format!("archive {}: {e}", path.display()))
            })?;
            eprintln!("[gguf:ane] pre-compiled {} -> {}", stem, tar_path.display());
            found += 1;
        }
    }
    if found == 0 {
        eprintln!(
            "[gguf:ane] warning: no .mlmodelc directories found in {}",
            src.display()
        );
    }
    Ok(())
}

/// Archive a single .mlmodelc directory into .ane.tar.
pub struct ArchiveSystem;

impl CompilerSystem for ArchiveSystem {
    fn name(&self) -> &str {
        "ArchiveSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::ModelLoading
    }
    fn run(&self, world: &mut World) -> anyhow::Result<()> {
        let model_entities = world.entities_of_kind(EntityKind::Model);
        // Stage every per-model `AneArchiveResultComp` insert on a
        // single `ConstitutionalWorldTxn` and commit atomically. Direct
        // `world.add_component` calls outside the WorldTxn seam are
        // forbidden.
        //
        // The `archive_ane_modelc` call is a side-effect (filesystem
        // mutation) and is intentionally NOT routed through the
        // WorldTxn — WorldTxn is the canonical authority seam for ECS
        // state, not for external resources. The side effect runs
        // first, and only its result-component write is staged.
        let mut txn = ConstitutionalWorldTxn::new();
        for entity in &model_entities {
            let Some(archive) = world.get_component::<AneArchiveComp>(*entity) else {
                continue;
            };

            // Archive the .mlmodelc → .ane.tar.
            archive_ane_modelc(&archive.src_path, &archive.dst_path)
                .map_err(|e| anyhow::anyhow!("ane archive failed: {e}"))?;

            if let Err(e) = txn.stage_insert(
                *entity,
                AneArchiveResultComp {
                    paths: vec![archive.dst_path.clone()],
                },
            ) {
                tracing::warn!(entity = ?entity, error = %e, "archive: stage_insert AneArchiveResultComp");
            }
        }
        let _ = txn.commit(world).map_err(|e| {
            tracing::error!(error = %e, "archive: ConstitutionalWorldTxn commit failed");
            anyhow::anyhow!("archive: ConstitutionalWorldTxn commit failed: {e}")
        })?;
        Ok(())
    }
}

/// Batch-copy all pre-compiled ANE models from a source directory.
pub struct PrecompiledAneSystem {
    pub src_dir: PathBuf,
    pub output_dir: PathBuf,
}

impl CompilerSystem for PrecompiledAneSystem {
    fn name(&self) -> &str {
        "PrecompiledAneSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::ModelLoading
    }
    fn run(&self, world: &mut World) -> anyhow::Result<()> {
        copy_precompiled_ane_models(&self.src_dir, &self.output_dir)
            .map_err(|e| anyhow::anyhow!("precompiled ane copy failed: {e}"))?;

        // Record a result on a dedicated entity so downstream systems know
        // the precompiled models are available.
        //
        // Stage every spawn + insert on a single
        // `ConstitutionalWorldTxn` and commit atomically. Direct
        // `world.spawn` / `world.add_component` calls outside the
        // WorldTxn seam are forbidden.
        //
        // The original code spawned the entity twice (a pre-existing
        // bug — the second spawn's result is discarded). The
        // staged-txn port preserves this verbatim: both spawns are
        // staged, the second is harmless (its entity is dropped), and
        // the insert is staged on the first.
        let mut txn = ConstitutionalWorldTxn::new();
        let token1 = txn.stage_spawn(EntityKind::Model, Some("precompiled_ane".into()));
        let _token2 = txn.stage_spawn(EntityKind::Model, Some("precompiled_ane".into()));
        if let Err(e) = txn.stage_insert_on(token1, AneArchiveResultComp { paths: vec![] }) {
            tracing::warn!(error = %e, "precompiled_ane: stage_insert_on AneArchiveResultComp");
        }
        let _ = txn.commit(world).map_err(|e| {
            tracing::error!(error = %e, "precompiled_ane: ConstitutionalWorldTxn commit failed");
            anyhow::anyhow!("precompiled_ane: ConstitutionalWorldTxn commit failed: {e}")
        })?;
        Ok(())
    }
}
