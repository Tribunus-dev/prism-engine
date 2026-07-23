use prism_ecs_core::identity::{TensorInfo, TensorProvider, TensorReader};

/// Maps ONNX initializer entries to TensorProvider entries.
pub struct OnnxTensorProvider {
    tensors: Vec<TensorInfo>,
}

impl OnnxTensorProvider {
    pub fn new(_data: &[u8]) -> Result<Self, String> {
        Ok(Self {
            tensors: Vec::new(),
        })
    }
}

impl TensorProvider for OnnxTensorProvider {
    fn list_tensors(&self) -> Result<Vec<TensorInfo>, String> {
        Ok(self.tensors.clone())
    }

    fn open_tensor(&self, _name: &str) -> Result<Box<dyn TensorReader>, String> {
        Err("ONNX tensor reading not hooked up yet".into())
    }
}
