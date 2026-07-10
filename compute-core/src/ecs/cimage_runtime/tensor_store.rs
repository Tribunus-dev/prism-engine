//! Runtime tensor types for the cimage runtime bridge.
//!
//! Converts loaded cimage tensor entries and payloads into runtime-resolvable
//! representations.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::execution_plan::CodecFamily;

/// A store of resolved runtime tensors from a cimage artifact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeTensorStore {
    pub tensors: BTreeMap<String, RuntimeTensor>,
}

impl RuntimeTensorStore {
    pub fn new() -> Self {
        Self {
            tensors: BTreeMap::new(),
        }
    }

    pub fn insert(&mut self, tensor: RuntimeTensor) {
        self.tensors.insert(tensor.tensor_id.clone(), tensor);
    }

    pub fn get(&self, id: &str) -> Option<&RuntimeTensor> {
        self.tensors.get(id)
    }

    pub fn tensor_ids(&self) -> Vec<&str> {
        self.tensors.keys().map(|s| s.as_str()).collect()
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

impl Default for RuntimeTensorStore {
    fn default() -> Self {
        Self::new()
    }
}

/// A single resolved tensor from the cimage manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeTensor {
    pub tensor_id: String,
    pub tensor_key: String,
    pub tensor_class: String,
    pub logical_shape: Vec<u32>,
    pub codec: CodecFamily,
    pub payload: RuntimeTensorPayload,
}

/// The decoded or packed payload for a runtime tensor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RuntimeTensorPayload {
    /// Fully decoded F32 weights.
    RawF32(Vec<f32>),
    /// FP16 weights (raw bytes).
    Fp16(Vec<u16>),
    /// INT8 tile640 packed format.
    Int8Packed {
        codes: Vec<u8>,
        scales: Vec<f32>,
        biases: Vec<f32>,
    },
    /// NF4 tile640 packed format.
    Nf4Packed {
        codes: Vec<u8>,
        scales: Vec<f32>,
        biases: Vec<f32>,
        group_size: usize,
    },
    /// Mixed precision — base + override table + sidecars.
    MixedPrecision {
        base: Box<RuntimeTensorPayload>,
        override_table: Vec<u8>,
        sidecars: Vec<RuntimeTensorPayload>,
    },
}

impl RuntimeTensorPayload {
    /// Return the total byte size of the payload data.
    pub fn byte_size(&self) -> usize {
        match self {
            Self::RawF32(v) => v.len() * 4,
            Self::Fp16(v) => v.len() * 2,
            Self::Int8Packed { codes, .. } => codes.len(),
            Self::Nf4Packed { codes, .. } => codes.len(),
            Self::MixedPrecision {
                base,
                override_table,
                sidecars,
            } => {
                base.byte_size()
                    + override_table.len()
                    + sidecars.iter().map(|s| s.byte_size()).sum::<usize>()
            }
        }
    }
}

/// Execution mode for the MLP region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MlpRegionExecutionMode {
    /// Separate kernel per operation (7 ops total).
    StagedKernels,
    /// Single fused kernel for the entire MLP block.
    FusedMlpKernel,
}

impl Default for MlpRegionExecutionMode {
    fn default() -> Self {
        Self::StagedKernels
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_tensor_store_basics() {
        let mut store = RuntimeTensorStore::new();
        assert!(store.is_empty());

        let t = RuntimeTensor {
            tensor_id: "t0".into(),
            tensor_key: "rmsnorm_weight".into(),
            tensor_class: "RmsNormWeight".into(),
            logical_shape: vec![64],
            codec: CodecFamily::RawF32,
            payload: RuntimeTensorPayload::RawF32(vec![1.0; 64]),
        };
        let id = t.tensor_id.clone();
        store.insert(t);
        assert_eq!(store.len(), 1);
        assert!(store.get(&id).is_some());
    }

    #[test]
    fn test_payload_byte_size() {
        let raw = RuntimeTensorPayload::RawF32(vec![1.0; 64]);
        assert_eq!(raw.byte_size(), 256);

        let int8 = RuntimeTensorPayload::Int8Packed {
            codes: vec![0u8; 320],
            scales: vec![1.0; 20],
            biases: vec![0.0; 20],
        };
        assert_eq!(int8.byte_size(), 320);
    }

    #[test]
    fn test_mlp_region_execution_mode_default() {
        assert_eq!(
            MlpRegionExecutionMode::default(),
            MlpRegionExecutionMode::StagedKernels
        );
    }
}
