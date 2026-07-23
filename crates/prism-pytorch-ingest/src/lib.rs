pub mod graph_walker;
/// PyTorch model ingestion: imports torch.export.export() serialized programs.
pub mod importer;
pub mod op_map;

use prism_ecs_core::identity::{GraphProvider, TensorProvider};
use std::path::Path;

/// Import a PyTorch ExportedProgram from a JSON file.
pub fn import_pytorch_export(
    path: &Path,
) -> Result<(Box<dyn GraphProvider>, Box<dyn TensorProvider>), String> {
    let contents =
        std::fs::read_to_string(path).map_err(|e| format!("read pytorch export: {e}"))?;
    let _export: serde_json::Value =
        serde_json::from_str(&contents).map_err(|e| format!("parse pytorch export: {e}"))?;

    let graph_provider = importer::PyTorchGraphProvider::new(path)?;
    let tensor_provider = importer::PyTorchTensorProvider::new(path)?;
    Ok((Box::new(graph_provider), Box::new(tensor_provider)))
}

/// Format detection: check if a file looks like a PyTorch export.
pub fn detect_pytorch_export(path: &Path) -> bool {
    match path.extension().and_then(|e| e.to_str()) {
        Some("pt" | "pth") => true,
        _ => false,
    }
}
