//! Archive utilities for cimage compilation — tar packing and model copying.
//! Extracted from pipeline.rs for hermetic access under prism-backend.

use std::path::Path;

/// Tar-archive a .mlmodelc directory into a single `.ane.tar` file.
/// The resulting archive can be extracted to a temp dir at runtime and loaded
/// via CoreAiModel::load (which expects a .mlmodelc directory on disk).
pub fn archive_ane_modelc(src: &Path, dst: &Path) -> std::io::Result<()> {
    let file = std::fs::File::create(dst)?;
    let mut builder = tar::Builder::new(std::io::BufWriter::new(file));
    builder.append_dir_all(".", src)?;
    builder.finish()?;
    Ok(())
}

/// Scan a directory for pre-compiled .mlmodelc bundles, tar-archive each,
/// and write the archives to `output_dir`.  Skips items that are not
/// directories ending in `.mlmodelc`.
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
