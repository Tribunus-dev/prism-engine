use prism_ecs_core::identity::{GraphProvider, TensorInfo, TensorProvider, TensorReader};
use std::path::Path;

pub struct PyTorchGraphProvider {
    _path: std::path::PathBuf,
}

impl PyTorchGraphProvider {
    pub fn new(path: &Path) -> Result<Self, String> {
        Ok(Self {
            _path: path.to_path_buf(),
        })
    }
}

impl GraphProvider for PyTorchGraphProvider {
    fn import_graph(&self, _world: &mut prism_ecs_core::world::World) -> Result<(), String> {
        Err(
            "PyTorch graph importing not yet implemented — FX graph JSON parser is a future wave"
                .into(),
        )
    }
}

pub struct PyTorchTensorProvider {
    _path: std::path::PathBuf,
    tensors: Vec<TensorInfo>,
}

impl PyTorchTensorProvider {
    pub fn new(path: &Path) -> Result<Self, String> {
        Ok(Self {
            _path: path.to_path_buf(),
            tensors: Vec::new(),
        })
    }
}

impl TensorProvider for PyTorchTensorProvider {
    fn list_tensors(&self) -> Result<Vec<TensorInfo>, String> {
        Ok(self.tensors.clone())
    }

    fn open_tensor(&self, _name: &str) -> Result<Box<dyn TensorReader>, String> {
        Err("PyTorch tensor reading not yet implemented".into())
    }
}
