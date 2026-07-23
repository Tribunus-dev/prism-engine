//! Deterministic Prism-native XDNA command-buffer encoding.
//!
//! This is the compiler-owned command representation placed into an XDNA
//! command BO. Firmware-specific mailbox fields remain outside this format;
//! the boundary is explicit in the header and can be lowered per driver ABI.

use prism_spatial_ir::xdna::{RuntimeCommand, XdnaProgram};
use serde::{Deserialize, Serialize};

const MAGIC: &[u8; 4] = b"PXDC";
const VERSION: u16 = 3;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeviceAddressBinding {
    pub resource: String,
    pub device_address: u64,
    pub bytes: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct XdnaCommandBuffer {
    pub generation: u8,
    pub program: XdnaProgram,
    pub commands: Vec<RuntimeCommand>,
    pub addresses: Vec<DeviceAddressBinding>,
}

/// Firmware-specific lowering boundary for native XDNA submission.
///
/// Prism owns the portable [`XdnaCommandBuffer`] contract; a target adapter
/// supplies the packet layout required by the installed amdxdna firmware.
/// Keeping this trait in Prism avoids making the compiler depend on an
/// external graph compiler while making the ABI boundary explicit.
pub trait XdnaFirmwareEncoder {
    type Error: std::fmt::Display;

    fn encode_firmware(&self, command: &XdnaCommandBuffer) -> Result<Vec<u8>, Self::Error>;
}

/// Deterministic Prism-owned firmware image encoder.
///
/// The two outputs mirror the amdxdna submission model: an array description
/// and an executable command stream. The image is deliberately versioned and
/// self-describing so a future kernel-specific adapter can translate it
/// without changing the compiler IR.
#[derive(Debug, Clone, Copy, Default)]
pub struct XdnaFirmwareImageEncoder;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XdnaFirmwareImage {
    pub overlay: Vec<u8>,
    pub ctrlcode: Vec<u8>,
}

impl XdnaFirmwareImageEncoder {
    pub fn encode_image(command: &XdnaCommandBuffer) -> Result<XdnaFirmwareImage, String> {
        let overlay_payload = bincode::serialize(&(
            &command.program.topology,
            &command.program.buffers,
            &command.program.fifos,
            &command.program.workers,
            &command.program.barriers,
        ))
        .map_err(|error| format!("encode XDNA overlay payload: {error}"))?;
        let ctrlcode_payload =
            bincode::serialize(&(command.generation, &command.commands, &command.addresses))
                .map_err(|error| format!("encode XDNA ctrlcode payload: {error}"))?;
        Ok(XdnaFirmwareImage {
            overlay: framed_firmware_payload(b"PXOV", &overlay_payload)?,
            ctrlcode: framed_firmware_payload(b"PXCC", &ctrlcode_payload)?,
        })
    }
}

impl XdnaFirmwareEncoder for XdnaFirmwareImageEncoder {
    type Error = String;

    fn encode_firmware(&self, command: &XdnaCommandBuffer) -> Result<Vec<u8>, Self::Error> {
        // This trait returns the executable half for submission APIs that
        // accept one buffer; callers needing both images should use encode_image.
        Ok(Self::encode_image(command)?.ctrlcode)
    }
}

fn framed_firmware_payload(magic: &[u8; 4], payload: &[u8]) -> Result<Vec<u8>, String> {
    let payload_len = u32::try_from(payload.len())
        .map_err(|_| "XDNA firmware payload exceeds 4 GiB frame limit".to_string())?;
    let mut output = Vec::with_capacity(12 + payload.len());
    output.extend_from_slice(magic);
    output.extend_from_slice(&1_u16.to_le_bytes());
    output.extend_from_slice(&0_u16.to_le_bytes());
    output.extend_from_slice(&payload_len.to_le_bytes());
    output.extend_from_slice(payload);
    Ok(output)
}

impl XdnaCommandBuffer {
    /// Validate the compiler-owned kernel ABI before producing a device
    /// artifact. Kernel names are part of the native Prism/XDNA contract;
    /// accepting an arbitrary string would create an artifact that can pass
    /// serialization but cannot be dispatched by the runtime.
    pub fn validate_native_kernels(program: &XdnaProgram) -> Result<(), String> {
        for worker in &program.workers {
            match worker.kernel.as_str() {
                "prism.xdna.matmul"
                | "prism.xdna.matmul_accumulate"
                | "prism.xdna.elementwise"
                | "prism.xdna.normalization"
                | "prism.xdna.softmax"
                | "prism.xdna.attention" => {}
                other => {
                    return Err(format!(
                        "unsupported native XDNA kernel '{}' in worker {}",
                        other, worker.id
                    ));
                }
            }
        }
        Ok(())
    }

    pub fn from_program(program: &XdnaProgram) -> Result<Self, String> {
        program.validate().map_err(|errors| errors.join("; "))?;
        Self::validate_native_kernels(program)?;
        Ok(Self {
            generation: match program.topology.generation {
                prism_spatial_ir::xdna::XdnaGeneration::Aie2 => 1,
                prism_spatial_ir::xdna::XdnaGeneration::Aie2p => 2,
            },
            program: program.clone(),
            commands: program.sequence.clone(),
            addresses: Vec::new(),
        })
    }

    pub fn with_addresses(mut self, addresses: Vec<DeviceAddressBinding>) -> Result<Self, String> {
        for binding in &addresses {
            if binding.device_address == 0 || binding.bytes == 0 {
                return Err(format!(
                    "invalid device address binding for {}",
                    binding.resource
                ));
            }
            let buffer = self
                .program
                .buffers
                .iter()
                .find(|buffer| buffer.id == binding.resource)
                .ok_or_else(|| {
                    format!(
                        "address binding references unknown XDNA resource {}",
                        binding.resource
                    )
                })?;
            if u64::from(binding.bytes) > u64::from(buffer.bytes) {
                return Err(format!(
                    "address binding for {} exceeds resource capacity",
                    binding.resource
                ));
            }
        }
        let mut resources: Vec<&str> = addresses
            .iter()
            .map(|binding| binding.resource.as_str())
            .collect();
        resources.sort_unstable();
        resources.dedup();
        if resources.len() != addresses.len() {
            return Err("duplicate XDNA device address binding".into());
        }
        self.addresses = addresses;
        Ok(self)
    }

    pub fn encode(&self) -> Result<Vec<u8>, String> {
        if self.commands.len() > u16::MAX as usize {
            return Err("XDNA command buffer contains too many commands".into());
        }
        let payload = bincode::serialize(&(&self.program, &self.commands, &self.addresses))
            .map_err(|error| format!("encode XDNA command payload: {error}"))?;
        let payload_len = u32::try_from(payload.len())
            .map_err(|_| "XDNA command payload is too large".to_string())?;
        let mut bytes = Vec::with_capacity(16 + payload.len());
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&VERSION.to_le_bytes());
        bytes.push(self.generation);
        bytes.push(0);
        bytes.extend_from_slice(&(self.commands.len() as u16).to_le_bytes());
        bytes.extend_from_slice(&payload_len.to_le_bytes());
        bytes.extend_from_slice(&payload);
        Ok(bytes)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 14 || &bytes[..4] != MAGIC {
            return Err("invalid Prism XDNA command-buffer magic".into());
        }
        let version = u16::from_le_bytes([bytes[4], bytes[5]]);
        if version != VERSION {
            return Err(format!(
                "unsupported Prism XDNA command-buffer version {version}"
            ));
        }
        let generation = bytes[6];
        if generation != 1 && generation != 2 {
            return Err(format!("unsupported XDNA generation tag {generation}"));
        }
        let count = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        let payload_len = u32::from_le_bytes([bytes[10], bytes[11], bytes[12], bytes[13]]) as usize;
        if bytes.len() != 14 + payload_len {
            return Err("XDNA command-buffer payload length mismatch".into());
        }
        let (program, commands, addresses): (
            XdnaProgram,
            Vec<RuntimeCommand>,
            Vec<DeviceAddressBinding>,
        ) = bincode::deserialize(&bytes[14..])
            .map_err(|error| format!("decode XDNA command payload: {error}"))?;
        if commands.len() != count {
            return Err("XDNA command count mismatch".into());
        }
        if program.sequence != commands {
            return Err("XDNA command payload sequence disagrees with program".into());
        }
        program.validate().map_err(|errors| errors.join("; "))?;
        Self::validate_native_kernels(&program)?;
        let addresses = Self {
            generation,
            program,
            commands,
            addresses: Vec::new(),
        }
        .with_addresses(addresses)?;
        Ok(addresses)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_spatial_ir::xdna::XdnaTopology;

    struct TestFirmwareEncoder;
    impl XdnaFirmwareEncoder for TestFirmwareEncoder {
        type Error = String;

        fn encode_firmware(&self, command: &XdnaCommandBuffer) -> Result<Vec<u8>, Self::Error> {
            if command.commands.is_empty() {
                Err("test encoder requires one command".into())
            } else {
                Ok(b"firmware-packet".to_vec())
            }
        }
    }

    #[test]
    fn command_buffer_round_trips_versioned_sequence() {
        let program = XdnaProgram {
            topology: XdnaTopology::xdna2(),
            buffers: vec![],
            fifos: vec![],
            transfers: vec![],
            workers: vec![],
            barriers: vec![],
            sequence: vec![RuntimeCommand::Signal {
                event_id: "ready".into(),
            }],
        };
        let command = XdnaCommandBuffer::from_program(&program).unwrap();
        assert_eq!(command.generation, 2);
        let decoded = XdnaCommandBuffer::decode(&command.encode().unwrap()).unwrap();
        assert_eq!(
            decoded,
            XdnaCommandBuffer {
                generation: 2,
                program,
                commands: vec![RuntimeCommand::Signal {
                    event_id: "ready".into()
                }],
                addresses: Vec::new(),
            }
        );
    }

    #[test]
    fn rejects_corrupt_payload_length() {
        let mut bytes = b"PXDC".to_vec();
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&[1, 0]);
        bytes.extend_from_slice(&0_u16.to_le_bytes());
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        assert!(XdnaCommandBuffer::decode(&bytes).is_err());
    }

    #[test]
    fn validates_and_round_trips_device_addresses() {
        let program = XdnaProgram {
            topology: XdnaTopology::xdna2(),
            buffers: vec![prism_spatial_ir::xdna::XdnaBuffer {
                id: "weights".into(),
                bytes: 4096,
                element_type: prism_spatial_ir::xdna::XdnaElementType::Int8,
                shape: vec![4096],
                memory: prism_spatial_ir::xdna::XdnaMemory::Shared,
                persistent: true,
            }],
            fifos: vec![],
            transfers: vec![],
            workers: vec![],
            barriers: vec![],
            sequence: vec![],
        };
        let command = XdnaCommandBuffer::from_program(&program)
            .unwrap()
            .with_addresses(vec![DeviceAddressBinding {
                resource: "weights".into(),
                device_address: 0x1000,
                bytes: 4096,
            }])
            .unwrap();
        let decoded = XdnaCommandBuffer::decode(&command.encode().unwrap()).unwrap();
        assert_eq!(decoded.addresses, command.addresses);
    }

    #[test]
    fn firmware_encoder_is_separate_from_portable_command_encoding() {
        let program = XdnaProgram {
            topology: XdnaTopology::xdna2(),
            buffers: vec![],
            fifos: vec![],
            transfers: vec![],
            workers: vec![],
            barriers: vec![],
            sequence: vec![RuntimeCommand::Signal {
                event_id: "ready".into(),
            }],
        };
        let command = XdnaCommandBuffer::from_program(&program).unwrap();
        let packet = TestFirmwareEncoder
            .encode_firmware(&command)
            .expect("firmware encoding should succeed");
        assert_eq!(packet, b"firmware-packet");
        assert_ne!(packet, command.encode().unwrap());

        let empty = XdnaCommandBuffer::from_program(&XdnaProgram {
            sequence: vec![],
            ..program
        })
        .unwrap();
        assert!(TestFirmwareEncoder.encode_firmware(&empty).is_err());
    }

    #[test]
    fn native_firmware_image_contains_distinct_overlay_and_ctrlcode() {
        let program = XdnaProgram {
            topology: XdnaTopology::xdna2(),
            buffers: vec![],
            fifos: vec![],
            transfers: vec![],
            workers: vec![],
            barriers: vec![],
            sequence: vec![RuntimeCommand::Signal {
                event_id: "ready".into(),
            }],
        };
        let command = XdnaCommandBuffer::from_program(&program).unwrap();
        let first = XdnaFirmwareImageEncoder::encode_image(&command).unwrap();
        let second = XdnaFirmwareImageEncoder::encode_image(&command).unwrap();
        assert_eq!(first, second);
        assert_eq!(&first.overlay[..4], b"PXOV");
        assert_eq!(&first.ctrlcode[..4], b"PXCC");
        assert_ne!(first.overlay, first.ctrlcode);
    }

    #[test]
    fn rejects_unknown_native_kernel_before_artifact_encoding() {
        let program = XdnaProgram {
            topology: XdnaTopology::xdna2(),
            buffers: vec![],
            fifos: vec![],
            transfers: vec![],
            workers: vec![prism_spatial_ir::xdna::XdnaWorker {
                id: "worker_1".into(),
                tile: prism_spatial_ir::xdna::TileCoord { col: 0, row: 0 },
                kernel: "prism.xdna.future_kernel".into(),
                inputs: vec![],
                outputs: vec![],
                waits_on: vec![],
                input_offsets: vec![],
                output_offsets: vec![],
            }],
            barriers: vec![],
            sequence: vec![RuntimeCommand::Run {
                worker_id: "worker_1".into(),
            }],
        };
        let error = XdnaCommandBuffer::from_program(&program).unwrap_err();
        assert!(error.contains("unsupported native XDNA kernel"));
    }
}
