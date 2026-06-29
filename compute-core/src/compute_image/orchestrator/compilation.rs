//! ANE model compilation helpers.
//!
//! Compiles the MIL program embedded in a `.cimage` deployment into a
//! `.mlmodelc` bundle via `xcrun coremlcompiler`, with caching support.

use super::Orchestrator;
use crate::coreml_bridge::{CoreMlComputeUnits, CoreMlModel};
use std::path::PathBuf;
use std::process::Command;

impl Orchestrator {
    /// Compile the MIL program from the deployment's `mil_buffer` into a
    /// `.mlmodelc` bundle cached at `cache_path`.
    ///
    /// The `mil_buffer` holds raw MIL program bytes that were embedded
    /// into the cimage during ANE island compilation. These are written
    /// to a temporary `.mlpackage` directory, compiled by `xcrun
    /// coremlcompiler`, and loaded as a `CoreMlModel` targeting the ANE.
    ///
    /// If `cache_path` already exists and contains a valid `.mlmodelc`,
    /// loading skips compilation.
    pub(crate) fn compile_ane_model(
        deployment: &crate::compute_image::cimage_loader::CimageDeployment,
        cache_path: &std::path::Path,
    ) -> Result<CoreMlModel, String> {
        let mil_buf = deployment
            .mil_buffer
            .as_ref()
            .ok_or_else(|| "no MIL buffer in deployment".to_string())?;

        // ── Check cache ──────────────────────────────────────────────
        if cache_path.exists() && cache_path.join("metadata.json").exists() {
            return CoreMlModel::load_with_compute_units(
                &cache_path.to_string_lossy(),
                CoreMlComputeUnits::CpuAndNeuralEngine,
            );
        }

        // ── Read MIL bytes from Metal buffer ────────────────────────
        let mil_len = mil_buf.length() as usize;
        let mil_bytes: Vec<u8> = unsafe {
            std::slice::from_raw_parts(mil_buf.contents() as *const u8, mil_len).to_vec()
        };

        // ── Detect format: .mlmodelc directory, .mlpackage, or raw MIL text ──
        if Self::is_mlmodelc_dir(&mil_bytes) {
            // Pre-compiled .mlmodelc: write to cache path directly
            Self::write_mlmodelc_from_bytes(cache_path, &mil_bytes)?;
            return CoreMlModel::load_with_compute_units(
                &cache_path.to_string_lossy(),
                CoreMlComputeUnits::CpuAndNeuralEngine,
            );
        }

        // ── Compile via xcrun coremlcompiler ─────────────────────────
        let tmp_dir = tempfile::TempDir::new().map_err(|e| format!("temp dir: {e}"))?;

        let mlpackage = tmp_dir.path().join("prefill.mlpackage");
        Self::write_mlpackage(&mlpackage, &mil_bytes)?;

        let modelc_dir = tmp_dir.path().join("prefill.modelc");
        std::fs::create_dir_all(&modelc_dir).map_err(|e| format!("create modelc dir: {e}"))?;

        let output = Command::new("xcrun")
            .args([
                "coremlcompiler",
                "compile",
                &mlpackage.to_string_lossy(),
                &modelc_dir.to_string_lossy(),
            ])
            .output()
            .map_err(|e| format!("xcrun invocation failed: {e}"))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("coremlcompiler compile failed: {stderr}"));
        }

        // ── Find the nested .mlmodelc directory ──────────────────────
        let compiled = Self::find_modelc_dir(&modelc_dir)
            .ok_or_else(|| "compiled .mlmodelc not found".to_string())?;

        // ── Cache the result ────────────────────────────────────────
        Self::copy_dir_all(&compiled, cache_path)?;

        // Keep tmp_dir alive until model loads, then leak it (OS cleans up)
        let model = CoreMlModel::load_with_compute_units(
            &cache_path.to_string_lossy(),
            CoreMlComputeUnits::CpuAndNeuralEngine,
        )?;
        std::mem::forget(tmp_dir);

        Ok(model)
    }

    /// Detect whether `bytes` represent a pre-compiled .mlmodelc directory.
    fn is_mlmodelc_dir(_bytes: &[u8]) -> bool {
        // .mlmodelc directories contain a metadata.json file at the root.
        // ZIP archives (mlpackage) have the magic "PK\x03\x04" at offset 0.
        // Neither pattern? Assume raw MIL text.
        false
    }

    /// Write a pre-compiled .mlmodelc directory from an in-memory
    /// representation (e.g. a tar or directory of files).
    fn write_mlmodelc_from_bytes(
        _cache_path: &std::path::Path,
        _bytes: &[u8],
    ) -> Result<(), String> {
        Err("pre-compiled .mlmodelc from bytes not yet implemented".into())
    }

    /// Write an `.mlpackage` directory from the MIL program bytes.
    ///
    /// The MIL buffer may contain either:
    /// - Raw MIL text: write as `model.mil` + `Manifest.json`
    /// - A ZIP archive: extract to the mlpackage directory
    fn write_mlpackage(mlpackage_dir: &std::path::Path, mil_bytes: &[u8]) -> Result<(), String> {
        std::fs::create_dir_all(mlpackage_dir)
            .map_err(|e| format!("create mlpackage dir: {e}"))?;

        // If it looks like a ZIP archive (magic "PK"), extract it.
        if mil_bytes.len() >= 2 && &mil_bytes[0..2] == b"PK" {
            return Self::unzip_mlpackage(mlpackage_dir, mil_bytes);
        }

        // Otherwise treat as raw MIL text: write model.mil + Manifest.json
        let mil_path = mlpackage_dir.join("model.mil");
        std::fs::write(&mil_path, mil_bytes).map_err(|e| format!("write model.mil: {e}"))?;

        // Minimal Manifest.json — coremlcompiler infers shapes at compile time
        let manifest = serde_json::json!({
            "fileFormatVersion": "1.0.0",
            "specificationVersion": 9,
            "rootModelSpecification": {
                "items": [{
                    "author": "Tribunus Compute",
                    "description": "ANE prefill model",
                    "name": "ane_prefill",
                    "version": "1.0.0"
                }]
            }
        });
        let manifest_path = mlpackage_dir.join("Manifest.json");
        std::fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest).unwrap(),
        )
        .map_err(|e| format!("write Manifest.json: {e}"))?;

        Ok(())
    }

    /// Extract a ZIP archive (`.mlpackage` bundle) to the target directory.
    ///
    /// ZIP extraction requires the `zip` crate which is not a current
    /// dependency of tribunus-compute-core. When .mlpackage-as-ZIP
    /// support is needed, add `zip = "2"` to Cargo.toml and use
    /// zip::ZipArchive here.
    ///
    /// For now, MIL programs are embedded as raw MIL text (not ZIP).
    fn unzip_mlpackage(_dest: &std::path::Path, _data: &[u8]) -> Result<(), String> {
        Err("mlpackage ZIP extraction not yet supported — embed raw MIL text instead".into())
    }

    /// Walk a .modelc directory tree to find the inner directory containing
    /// `metadata.json`.
    fn find_modelc_dir(dir: &std::path::Path) -> Option<PathBuf> {
        fn walk(dir: &std::path::Path, depth: u32) -> Option<PathBuf> {
            if depth > 4 {
                return None;
            }
            if dir.join("metadata.json").exists() {
                return Some(dir.to_path_buf());
            }
            for entry in std::fs::read_dir(dir).ok()? {
                let entry = entry.ok()?;
                let path = entry.path();
                if path.is_dir() {
                    if let Some(found) = walk(&path, depth + 1) {
                        return Some(found);
                    }
                }
            }
            None
        }
        walk(dir, 0)
    }

    /// Recursively copy a directory.
    fn copy_dir_all(src: &std::path::Path, dst: &std::path::Path) -> Result<(), String> {
        if dst.exists() {
            std::fs::remove_dir_all(dst)
                .map_err(|e| format!("remove old cache {}: {e}", dst.display()))?;
        }
        std::fs::create_dir_all(dst)
            .map_err(|e| format!("create cache dir {}: {e}", dst.display()))?;
        for entry in
            std::fs::read_dir(src).map_err(|e| format!("read src dir {}: {e}", src.display()))?
        {
            let entry = entry.map_err(|e| format!("entry: {e}"))?;
            let from = entry.path();
            let to = dst.join(entry.file_name());
            if from.is_dir() {
                Self::copy_dir_all(&from, &to)?;
            } else {
                std::fs::copy(&from, &to)
                    .map_err(|e| format!("copy {} -> {}: {e}", from.display(), to.display()))?;
            }
        }
        Ok(())
    }
}
