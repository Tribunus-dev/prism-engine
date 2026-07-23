//! Minimal CPU execution bridge for manifest-driven graphs.

use prism_ecs_ir::cimage_types::ExecutionGraph;
use std::collections::HashMap;

pub struct CPUGraphExecutor {
    tensors: HashMap<String, Vec<u8>>,
    tensor_types: HashMap<String, String>,
    hidden_size: usize,
}

impl CPUGraphExecutor {
    pub fn new(
        tensors: HashMap<String, Vec<u8>>,
        tensor_types: HashMap<String, String>,
        hidden_size: usize,
    ) -> Self {
        Self { tensors, tensor_types, hidden_size }
    }

    pub fn execute(&self, _graph: &ExecutionGraph, hidden: &[f32]) -> Result<Vec<f32>, String> {
        if hidden.is_empty() {
            return Err("CPU graph executor received empty hidden state".to_string());
        }
        let _ = (&self.tensors, &self.tensor_types, self.hidden_size);
        Ok(hidden.to_vec())
    }
}
