//! Model-specific XDNA artifact metadata.

use crate::xdna::{XdnaGeneration, XdnaMemory, XdnaProgram};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationKind {
    Int8,
    Int4,
    /// Progressive ternary storage with per-group scale and optional dense
    /// residual lanes retained for tensors that fail the loss gate.
    TernaryMixed,
    F16,
    BF16,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuantizationSpec {
    pub kind: QuantizationKind,
    pub group_size: u32,
    pub scales_buffer: Option<String>,
    pub zero_points_buffer: Option<String>,
    /// Optional dense/residual lane retained for groups that fail the
    /// progressive loss gate. Its presence makes mixed-precision fallback
    /// explicit to the runtime rather than an implicit tensor convention.
    #[serde(default)]
    pub residual_buffer: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResidencyPolicy {
    Host,
    SharedPersistent,
    TilePersistent,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct XdnaTensorManifest {
    pub name: String,
    pub bytes: u64,
    pub quantization: Option<QuantizationSpec>,
    pub residency: ResidencyPolicy,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct XdnaModelManifest {
    pub model_id: String,
    pub compiler_abi: String,
    pub supported_generations: Vec<XdnaGeneration>,
    pub tensors: Vec<XdnaTensorManifest>,
    pub kv_cache_bytes_per_token: u64,
    pub prefill_chunk_tokens: u32,
    /// Number of XDNA array columns requested by the workload resource
    /// solver. This is derived from the compiled worker placement.
    #[serde(default)]
    pub required_columns: u16,
}

impl XdnaModelManifest {
    /// Derive a manifest from Prism's lowered program while preserving the
    /// program's persistent residency decisions.
    pub fn from_program(model_id: impl Into<String>, program: &XdnaProgram) -> Self {
        Self {
            model_id: model_id.into(),
            compiler_abi: "prism-xdna-v1".into(),
            supported_generations: vec![program.topology.generation],
            tensors: program
                .buffers
                .iter()
                .map(|buffer| XdnaTensorManifest {
                    name: buffer.id.clone(),
                    bytes: buffer.bytes as u64,
                    quantization: None,
                    residency: if buffer.persistent {
                        match buffer.memory {
                            XdnaMemory::TileLocal(_) | XdnaMemory::MemoryTile(_) => {
                                ResidencyPolicy::TilePersistent
                            }
                            _ => ResidencyPolicy::SharedPersistent,
                        }
                    } else {
                        ResidencyPolicy::Host
                    },
                })
                .collect(),
            kv_cache_bytes_per_token: 0,
            prefill_chunk_tokens: 1,
            required_columns: program
                .workers
                .iter()
                .map(|worker| worker.tile.col.saturating_add(1))
                .max()
                .unwrap_or(1),
        }
    }

    /// Attach a compiler-selected precision contract to one persistent
    /// tensor. The references are checked by `validate`, so a caller cannot
    /// seal an artifact that mentions missing scale or residual metadata.
    pub fn with_quantization(
        mut self,
        tensor_name: impl AsRef<str>,
        quantization: QuantizationSpec,
    ) -> Result<Self, String> {
        let tensor_name = tensor_name.as_ref();
        let tensor = self
            .tensors
            .iter_mut()
            .find(|tensor| tensor.name == tensor_name)
            .ok_or_else(|| format!("manifest has no tensor {tensor_name}"))?;
        tensor.quantization = Some(quantization);
        Ok(self)
    }

    pub fn validate(&self, generation: XdnaGeneration) -> Result<(), String> {
        if !self.supported_generations.contains(&generation) {
            return Err(format!(
                "model {} does not support {:?}",
                self.model_id, generation
            ));
        }
        if self.compiler_abi.is_empty() {
            return Err("compiler ABI is empty".into());
        }
        if self.prefill_chunk_tokens == 0 {
            return Err("prefill chunk size must be nonzero".into());
        }
        if self.required_columns == 0 {
            return Err("XDNA workload must request at least one column".into());
        }
        if self.kv_cache_bytes_per_token > 0
            && !self.tensors.iter().any(|tensor| {
                tensor.name.to_ascii_lowercase().contains("kv")
                    && matches!(
                        tensor.residency,
                        ResidencyPolicy::SharedPersistent | ResidencyPolicy::TilePersistent
                    )
            })
        {
            return Err("KV cache capacity is declared without a persistent KV tensor".into());
        }
        if self.tensors.iter().any(|tensor| tensor.bytes == 0) {
            return Err("manifest contains zero-sized tensor".into());
        }
        let mut tensor_names = std::collections::HashSet::new();
        for tensor in &self.tensors {
            if !tensor_names.insert(&tensor.name) {
                return Err(format!(
                    "manifest contains duplicate tensor {}",
                    tensor.name
                ));
            }
        }
        if self
            .tensors
            .iter()
            .filter_map(|tensor| tensor.quantization.as_ref())
            .any(|quant| {
                matches!(
                    quant.kind,
                    QuantizationKind::Int8
                        | QuantizationKind::Int4
                        | QuantizationKind::TernaryMixed
                ) && quant.group_size == 0
            })
        {
            return Err("quantized tensor has zero group size".into());
        }
        for tensor in &self.tensors {
            if let Some(quantization) = &tensor.quantization {
                for reference in [
                    quantization.scales_buffer.as_ref(),
                    quantization.zero_points_buffer.as_ref(),
                    quantization.residual_buffer.as_ref(),
                ]
                .into_iter()
                .flatten()
                {
                    if !self
                        .tensors
                        .iter()
                        .any(|candidate| &candidate.name == reference)
                    {
                        return Err(format!(
                            "quantized tensor {} references missing metadata buffer {}",
                            tensor.name, reference
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixed_precision_residual_reference_is_validated() {
        let program = XdnaProgram {
            topology: crate::xdna::XdnaTopology::xdna2(),
            buffers: vec![crate::xdna::XdnaBuffer {
                id: "weights".into(),
                bytes: 4,
                element_type: crate::xdna::XdnaElementType::Int8,
                shape: vec![4],
                memory: XdnaMemory::Shared,
                persistent: true,
            }],
            fifos: vec![],
            transfers: vec![],
            workers: vec![],
            barriers: vec![],
            sequence: vec![],
        };
        let mut manifest = XdnaModelManifest::from_program("model", &program);
        manifest.tensors.push(XdnaTensorManifest {
            name: "residual".into(),
            bytes: 4,
            quantization: None,
            residency: ResidencyPolicy::Host,
        });
        manifest.tensors[0].quantization = Some(QuantizationSpec {
            kind: QuantizationKind::TernaryMixed,
            group_size: 32,
            scales_buffer: None,
            zero_points_buffer: None,
            residual_buffer: Some("residual".into()),
        });
        assert!(manifest.validate(XdnaGeneration::Aie2p).is_ok());
        manifest.tensors[0]
            .quantization
            .as_mut()
            .unwrap()
            .residual_buffer = Some("missing".into());
        assert!(manifest.validate(XdnaGeneration::Aie2p).is_err());
    }
}
