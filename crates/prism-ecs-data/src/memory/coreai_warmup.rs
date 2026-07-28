//! Core ML ANE warmup helper.
//!
//! This module owns the canonical authority for the engine-independent
//! warmup flow: assemble a minimal `.mlpackage` bundle containing
//! the [`ane_warmup_mil`](super::ane_warmup_mil) program, compile it
//! via `coremlc`, and clean up. It does not depend on the engine's
//! `Arena` or `worker_memory` — it is the constitutional surface for
//! the ANE-firmware warmup side effect.

use std::path::{Path, PathBuf};
use std::process::Command;

use super::ane_warmup_mil;

/// Subdirectory names inside a .mlpackage bundle.
const MLPACKAGE_MANIFEST: &str = "Manifest.json";
const MLPACKAGE_DATA_DIR: &str = "Data";
const MLPACKAGE_TYPE_DIR: &str = "Type";
const DEFAULT_MIL_FILE: &str = "default.mil";
const MODEL_METADATA_FILE: &str = "metadata.json";

fn write_mlpackage_manifest(path: &Path) -> Result<(), String> {
    let content = r#"{
  "modelVersion": { "major": 1, "minor": 0 },
  "authorName": "Tribunus Compute",
  "description": "ANE firmware warmup — x * x element-wise multiply",
  "license": "MIT",
  "specificationVersion": 7,
  "source": "tribunus",
  "mlModelStructure": "com.apple.CoreML.MLModel"
}
"#;
    std::fs::write(path, content).map_err(|e| format!("write Manifest.json: {e}"))
}

fn write_mlpackage_type_metadata(path: &Path) -> Result<(), String> {
    let content = r#"{
  "com.apple.CoreML.modelMetadata": {
    "author": "Tribunus Compute",
    "description": "ANE firmware warmup",
    "license": "MIT",
    "shortDescription": "ANE firmware warmup — x * x element-wise multiply",
    "version": "1.0"
  },
  "com.apple.CoreML.mlModel": {
    "inputDescriptions": [
      {
        "name": "x",
        "shortDescription": "scalar input tensor",
        "type": { "multiArrayType": { "shape": [1, 1, 1, 1], "dataType": "float16" } }
      }
    ],
    "outputDescriptions": [
      {
        "name": "y",
        "shortDescription": "scalar output tensor (x * x)",
        "type": { "multiArrayType": { "shape": [1, 1, 1, 1], "dataType": "float16" } }
      }
    ],
    "predictedFeatureName": "y"
  }
}
"#;
    std::fs::write(path, content).map_err(|e| format!("write Type/metadata.json: {e}"))
}

/// Build a minimal .mlpackage bundle at `output_path` containing the
/// canonical ANE warmup MIL program. The bundle can then be compiled
/// via `coremlc compile`.
pub fn build_warmup_mlpackage(output_path: &Path) -> Result<(), String> {
    // Directory tree: output.mlpackage/{Manifest.json, Data/default.mil, Type/metadata.json}
    let data_dir = output_path.join(MLPACKAGE_DATA_DIR);
    let type_dir = output_path.join(MLPACKAGE_TYPE_DIR);

    std::fs::create_dir_all(&data_dir).map_err(|e| format!("create Data dir: {e}"))?;
    std::fs::create_dir_all(&type_dir).map_err(|e| format!("create Type dir: {e}"))?;

    write_mlpackage_manifest(&output_path.join(MLPACKAGE_MANIFEST))?;

    // Write the canonical MIL program from the constitutional
    // ane_warmup_mil module.
    std::fs::write(data_dir.join(DEFAULT_MIL_FILE), ane_warmup_mil::ane_warmup_mil())
        .map_err(|e| format!("write Data/default.mil: {e}"))?;

    write_mlpackage_type_metadata(&type_dir.join(MODEL_METADATA_FILE))?;

    Ok(())
}

/// Compile a .mlpackage into a .mlmodelc using `coremlc`.
///
/// Returns the path to the compiled .mlmodelc directory.
pub fn compile_mlpackage(mlpackage_path: &Path, output_dir: &Path) -> Result<PathBuf, String> {
    // Find coremlc via xcrun
    let coremlc = Command::new("xcrun")
        .args(["--find", "coremlc"])
        .output()
        .map_err(|e| format!("xcrun --find coremlc: {e}"))?;
    if !coremlc.status.success() {
        return Err("coremlc not found — Xcode command line tools required".into());
    }
    let coremlc_path = String::from_utf8_lossy(&coremlc.stdout).trim().to_string();

    // Run: coremlc compile <mlpackage> <output_dir>
    let status = Command::new(&coremlc_path)
        .arg("compile")
        .arg(mlpackage_path)
        .arg(output_dir)
        .status()
        .map_err(|e| format!("coremlc compile execution: {e}"))?;

    if !status.success() {
        return Err("coremlc compile failed — see stderr for details".into());
    }

    // The compiled model is named <mlpackage_name>.mlmodelc in output_dir
    let stem = mlpackage_path
        .file_stem()
        .ok_or_else(|| "invalid mlpackage path".to_string())?;
    let mut compiled_path = output_dir.to_path_buf();
    compiled_path.push(format!("{}.mlmodelc", stem.to_string_lossy()));
    if compiled_path.exists() {
        Ok(compiled_path)
    } else {
        Err(format!("mlmodelc not found at {:?}", compiled_path))
    }
}

/// Attempt to warm the ANE through Core ML compilation.
///
/// Strategy:
///   1.  Create a minimal .mlpackage in a temp dir.
///   2.  Compile it via `coremlc` (runs through Core ML framework,
///       which has the ANE compile entitlement).
///   3.  Clean up temp files.
///
/// Returns true if the ANE was successfully warmed.
/// Returns false if Core ML compilation is unavailable (no Xcode tools,
/// no ANE on this machine, etc.) — the caller should fall back gracefully.
pub fn prewarm_ane_via_coreml() -> bool {
    let tmp_dir =
        match std::env::temp_dir().join(format!("tribunus_ane_warmup_{}", std::process::id())) {
            p => p,
        };
    let _ = std::fs::remove_dir_all(&tmp_dir);
    if std::fs::create_dir_all(&tmp_dir).is_err() {
        return false;
    }

    let mlpackage_path = tmp_dir.join("warmup.mlpackage");
    if build_warmup_mlpackage(&mlpackage_path).is_err() {
        let _ = std::fs::remove_dir_all(&tmp_dir);
        return false;
    }

    let _compiled_path = match compile_mlpackage(&mlpackage_path, &tmp_dir) {
        Ok(p) => p,
        Err(_) => {
            let _ = std::fs::remove_dir_all(&tmp_dir);
            return false;
        }
    };

    // Cleanup — the ANE compiler daemon was contacted via Core ML's entitlement.
    // This warms the ANE compiler infrastructure.
    let _ = std::fs::remove_dir_all(&tmp_dir);
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ane_warmup_mil_bytes_non_empty() {
        let bytes = ane_warmup_mil::ane_warmup_mil();
        assert!(!bytes.is_empty(), "ANE warmup MIL must embed bytes");
        let s = std::str::from_utf8(bytes).expect("MIL bytes are UTF-8");
        assert!(s.contains("program(1.3)"), "MIL must declare program(1.3)");
        assert!(s.contains("mul("), "MIL must contain a multiply op");
    }

    #[test]
    fn build_mlpackage_writes_all_three_files() {
        let tmp = std::env::temp_dir().join("prism_ecs_data_test_mlpackage");
        let _ = std::fs::remove_dir_all(&tmp);
        let pkg_path = tmp.join("test.mlpackage");
        build_warmup_mlpackage(&pkg_path).expect("build mlpackage");
        assert!(
            pkg_path.join("Manifest.json").exists(),
            "Manifest.json exists"
        );
        assert!(
            pkg_path.join("Data/default.mil").exists(),
            "Data/default.mil exists"
        );
        assert!(
            pkg_path.join("Type/metadata.json").exists(),
            "Type/metadata.json exists"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
