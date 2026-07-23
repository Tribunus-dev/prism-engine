//! Native Prism XDNA artifact container.

use crate::command::XdnaCommandBuffer;
use prism_spatial_ir::xdna::XdnaProgram;
use prism_spatial_ir::xdna_manifest::QuantizationKind;
use prism_spatial_ir::xdna_manifest::XdnaModelManifest;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XdnaArtifact {
    pub program: XdnaProgram,
    pub manifest: XdnaModelManifest,
    /// Firmware-facing XDNA array overlay, when produced by a target encoder.
    #[serde(default)]
    pub overlay: Option<Vec<u8>>,
    /// ERT ctrlcode that orchestrates the overlay and DMA operations.
    #[serde(default)]
    pub ctrlcode: Option<Vec<u8>>,
}

impl XdnaArtifact {
    pub fn command_buffer(&self) -> Result<XdnaCommandBuffer, String> {
        self.validate()?;
        XdnaCommandBuffer::from_program(&self.program)
    }

    pub fn validate(&self) -> Result<(), String> {
        match (&self.overlay, &self.ctrlcode) {
            (Some(overlay), Some(ctrlcode)) => {
                validate_firmware_frame(overlay, b"PXOV", "overlay")?;
                validate_firmware_frame(ctrlcode, b"PXCC", "ctrlcode")?;
            }
            (None, None) => {}
            _ => return Err("XDNA overlay and ctrlcode must be supplied together".into()),
        }
        self.program
            .validate()
            .map_err(|errors| errors.join("; "))?;
        self.manifest.validate(self.program.topology.generation)?;
        if self.manifest.required_columns > self.program.topology.columns {
            return Err(format!(
                "XDNA workload requests {} columns but target provides {}",
                self.manifest.required_columns, self.program.topology.columns
            ));
        }
        for buffer in &self.program.buffers {
            // The program contains compiler-owned transient staging/FIFO
            // buffers in addition to model tensors. Only persistent buffers
            // need a model-manifest entry and residency contract.
            if !buffer.persistent {
                continue;
            }
            let tensor = self
                .manifest
                .tensors
                .iter()
                .find(|tensor| tensor.name == buffer.id)
                .ok_or_else(|| format!("manifest is missing persistent tensor {}", buffer.id))?;
            if tensor.bytes < buffer.bytes as u64 {
                return Err(format!(
                    "manifest tensor {} is smaller than program buffer",
                    buffer.id
                ));
            }
            if let Some(quantization) = &tensor.quantization {
                let compatible = match quantization.kind {
                    QuantizationKind::Int8 => matches!(
                        buffer.element_type,
                        prism_spatial_ir::xdna::XdnaElementType::Int8
                            | prism_spatial_ir::xdna::XdnaElementType::UInt8
                    ),
                    QuantizationKind::Int4 => matches!(
                        buffer.element_type,
                        prism_spatial_ir::xdna::XdnaElementType::Int8
                            | prism_spatial_ir::xdna::XdnaElementType::UInt8
                    ),
                    QuantizationKind::TernaryMixed => matches!(
                        buffer.element_type,
                        prism_spatial_ir::xdna::XdnaElementType::Int8
                            | prism_spatial_ir::xdna::XdnaElementType::UInt8
                    ),
                    QuantizationKind::F16 => matches!(
                        buffer.element_type,
                        prism_spatial_ir::xdna::XdnaElementType::F16
                    ),
                    QuantizationKind::BF16 => matches!(
                        buffer.element_type,
                        prism_spatial_ir::xdna::XdnaElementType::BF16
                    ),
                };
                if !compatible {
                    return Err(format!(
                        "quantization {:?} is incompatible with buffer {} element type {:?}",
                        quantization.kind, buffer.id, buffer.element_type
                    ));
                }
                for reference in [
                    quantization.scales_buffer.as_ref(),
                    quantization.zero_points_buffer.as_ref(),
                    quantization.residual_buffer.as_ref(),
                ]
                .into_iter()
                .flatten()
                {
                    let metadata = self
                        .program
                        .buffers
                        .iter()
                        .find(|buffer| buffer.id == *reference)
                        .ok_or_else(|| {
                            format!(
                                "quantized tensor {} references metadata buffer {} absent from program",
                                tensor.name, reference
                            )
                        })?;
                    if !metadata.persistent {
                        return Err(format!(
                            "quantization metadata buffer {} must be persistent",
                            reference
                        ));
                    }
                }
            }
            let persistent = matches!(
                tensor.residency,
                prism_spatial_ir::xdna_manifest::ResidencyPolicy::SharedPersistent
                    | prism_spatial_ir::xdna_manifest::ResidencyPolicy::TilePersistent
            );
            if persistent != buffer.persistent {
                return Err(format!(
                    "manifest residency for {} disagrees with program",
                    buffer.id
                ));
            }
        }
        for tensor in &self.manifest.tensors {
            if !matches!(
                tensor.residency,
                prism_spatial_ir::xdna_manifest::ResidencyPolicy::Host
            ) && !self
                .program
                .buffers
                .iter()
                .any(|buffer| buffer.id == tensor.name && buffer.persistent)
            {
                return Err(format!(
                    "persistent manifest tensor {} has no program buffer",
                    tensor.name
                ));
            }
        }
        Ok(())
    }
}

fn validate_firmware_frame(bytes: &[u8], magic: &[u8; 4], label: &str) -> Result<(), String> {
    if bytes.len() < 12 || &bytes[..4] != magic {
        return Err(format!("XDNA firmware {label} has an invalid frame"));
    }
    let version = u16::from_le_bytes([bytes[4], bytes[5]]);
    if version != 1 {
        return Err(format!(
            "XDNA firmware {label} has unsupported version {version}"
        ));
    }
    let payload_len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
    if bytes.len() != 12 + payload_len {
        return Err(format!("XDNA firmware {label} frame length mismatch"));
    }
    const XDNA_INSTRUCTION_BUFFER_BYTES: usize = 64 * 1024 * 1024;
    if label == "ctrlcode" && bytes.len() > XDNA_INSTRUCTION_BUFFER_BYTES {
        return Err("XDNA ctrlcode exceeds the 64 MiB instruction buffer".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_spatial_ir::xdna::*;
    use prism_spatial_ir::xdna_manifest::{ResidencyPolicy, XdnaTensorManifest};

    fn artifact() -> XdnaArtifact {
        let program = XdnaProgram {
            topology: XdnaTopology::xdna2(),
            buffers: vec![XdnaBuffer {
                id: "weights".into(),
                bytes: 4,
                element_type: XdnaElementType::Int8,
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
        XdnaArtifact {
            program,
            manifest: XdnaModelManifest {
                model_id: "test".into(),
                compiler_abi: "prism-xdna-v1".into(),
                supported_generations: vec![XdnaGeneration::Aie2p],
                tensors: vec![XdnaTensorManifest {
                    name: "weights".into(),
                    bytes: 4,
                    quantization: None,
                    residency: ResidencyPolicy::SharedPersistent,
                }],
                kv_cache_bytes_per_token: 0,
                prefill_chunk_tokens: 4,
                required_columns: 1,
            },
            overlay: None,
            ctrlcode: None,
        }
    }

    #[test]
    fn binary_artifact_round_trip_preserves_validation() {
        let original = artifact();
        let encoded = original.encode().unwrap();
        let decoded = XdnaArtifact::decode(&encoded).unwrap();
        assert_eq!(decoded.manifest.model_id, "test");
        assert_eq!(decoded.program.buffers[0].id, "weights");
    }

    #[test]
    fn artifact_rejects_manifest_residency_mismatch() {
        let mut value = artifact();
        value.manifest.tensors[0].residency = ResidencyPolicy::Host;
        assert!(value.validate().is_err());
    }

    #[test]
    fn firmware_artifacts_require_a_coherent_overlay_and_ctrlcode_pair() {
        let mut value = artifact();
        value.overlay = Some(vec![b'P', b'X', b'O', b'V', 1, 0, 0, 0, 0, 0, 0, 0]);
        assert!(value.validate().is_err());
        value.ctrlcode = Some(vec![b'P', b'X', b'C', b'C', 1, 0, 0, 0, 0, 0, 0, 0]);
        assert!(value.validate().is_ok());
        value.overlay = Some(Vec::new());
        assert!(value.validate().is_err());
    }

    #[test]
    fn rejects_workload_partition_larger_than_target() {
        let mut value = artifact();
        value.manifest.required_columns = value.program.topology.columns + 1;
        assert!(value.validate().is_err());
    }

    #[test]
    fn transient_program_buffers_are_derived_as_host_manifest_entries() {
        let value = artifact();
        let derived = XdnaModelManifest::from_program("derived", &value.program);
        assert_eq!(
            derived.tensors[0].residency,
            ResidencyPolicy::SharedPersistent
        );
        assert_eq!(derived.tensors[0].bytes, 4);
    }

    #[test]
    fn incompatible_quantization_is_rejected() {
        let mut value = artifact();
        value.manifest.tensors[0].quantization =
            Some(prism_spatial_ir::xdna_manifest::QuantizationSpec {
                kind: prism_spatial_ir::xdna_manifest::QuantizationKind::F16,
                group_size: 0,
                scales_buffer: None,
                zero_points_buffer: None,
                residual_buffer: None,
            });
        assert!(value.validate().is_err());
    }

    #[test]
    fn quantization_metadata_must_be_persistent_program_storage() {
        let mut value = artifact();
        value.program.buffers.push(XdnaBuffer {
            id: "scales".into(),
            bytes: 2,
            element_type: XdnaElementType::F16,
            shape: vec![1],
            memory: XdnaMemory::Shared,
            persistent: true,
        });
        value.manifest.tensors.push(XdnaTensorManifest {
            name: "scales".into(),
            bytes: 2,
            quantization: None,
            residency: ResidencyPolicy::SharedPersistent,
        });
        value.manifest.tensors[0].quantization =
            Some(prism_spatial_ir::xdna_manifest::QuantizationSpec {
                kind: prism_spatial_ir::xdna_manifest::QuantizationKind::TernaryMixed,
                group_size: 32,
                scales_buffer: Some("scales".into()),
                zero_points_buffer: None,
                residual_buffer: None,
            });
        assert!(value.validate().is_ok());
        value.program.buffers.retain(|buffer| buffer.id != "scales");
        assert!(value.validate().is_err());
    }
}
