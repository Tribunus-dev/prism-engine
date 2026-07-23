/// ONNX model ingestion: imports .onnx protobuf files.
pub mod importer;
pub mod op_map;
pub mod tensor_map;

use prism_ecs_core::identity::{GraphProvider, TensorProvider};
use std::path::Path;

/// Import an ONNX model from a .onnx file.
pub fn import_onnx(
    path: &Path,
) -> Result<(Box<dyn GraphProvider>, Box<dyn TensorProvider>), String> {
    let contents = std::fs::read(path).map_err(|e| format!("read onnx: {e}"))?;

    let graph_provider = importer::OnnxGraphProvider::new(&contents)?;
    let tensor_provider = tensor_map::OnnxTensorProvider::new(&contents)?;
    Ok((Box::new(graph_provider), Box::new(tensor_provider)))
}

/// Format detection: check if a file looks like an ONNX model.
pub fn detect_onnx(path: &Path) -> bool {
    path.extension().and_then(|e| e.to_str()) == Some("onnx")
}
