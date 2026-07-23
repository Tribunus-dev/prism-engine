//! Prism-owned XDNA runtime execution contract.
//!
//! The planner produces an [`XdnaProgram`]; this module owns validation,
//! persistent residency, and command submission. A platform implementation
//! can implement [`XdnaDevice`] for amdxdna/XRT without changing compiler IR.

use crate::artifact::XdnaArtifact;
use crate::command::XdnaCommandBuffer;
use prism_spatial_ir::xdna::{RuntimeCommand, XdnaProgram};
use std::collections::{HashMap, HashSet};

pub trait XdnaDevice {
    type Error: std::fmt::Display;
    fn upload(&mut self, buffer_id: &str, bytes: u32) -> Result<(), Self::Error>;
    /// Upload real resident contents when the caller owns the payload. The
    /// default preserves metadata-only devices that bind memory elsewhere.
    fn upload_payload(&mut self, buffer_id: &str, payload: &[u8]) -> Result<(), Self::Error> {
        self.upload(buffer_id, payload.len() as u32)
    }
    /// Optionally retrieve a completed device buffer. Metadata-only devices
    /// may return `None`; concrete Linux/XDNA implementations can provide
    /// bytes for scheduler output commit.
    fn download_payload(
        &mut self,
        _buffer_id: &str,
        _bytes: u32,
    ) -> Result<Option<Vec<u8>>, Self::Error> {
        Ok(None)
    }
    fn execute(&mut self, command: &RuntimeCommand) -> Result<(), Self::Error>;
}

pub trait XdnaTransport {
    type Error: std::fmt::Display;
    fn upload_buffer(&mut self, buffer_id: &str, bytes: u32) -> Result<(), Self::Error>;
    fn submit_command(&mut self, packet: &[u8]) -> Result<(), Self::Error>;

    fn submit_firmware_artifact(
        &mut self,
        _program: &XdnaProgram,
        _overlay: &[u8],
        _ctrlcode: &[u8],
    ) -> Result<(), String> {
        Err("XDNA transport does not implement firmware artifact submission".into())
    }
}

pub trait XdnaCommandSubmitter: XdnaDevice {
    fn submit_command_buffer(&mut self, packet: &[u8]) -> Result<(), Self::Error>;

    /// Submit the two firmware-facing artifacts produced by the compiler.
    /// Implementations must translate these payloads into the installed
    /// amdxdna command/overlay ABI; the portable `PXDC` envelope is not valid
    /// firmware input.
    fn submit_firmware_artifact(
        &mut self,
        _program: &XdnaProgram,
        _overlay: &[u8],
        _ctrlcode: &[u8],
    ) -> Result<(), String> {
        Err("device does not implement XDNA firmware artifact submission".into())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XdnaAvailability {
    UnsupportedPlatform,
    DeviceUnavailable,
    DevicePresent,
}

pub fn detect_xdna_availability() -> XdnaAvailability {
    #[cfg(target_os = "linux")]
    {
        if std::path::Path::new("/dev/amdxdna").exists()
            || std::path::Path::new("/dev/accel/accel0").exists()
        {
            XdnaAvailability::DevicePresent
        } else {
            XdnaAvailability::DeviceUnavailable
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        XdnaAvailability::UnsupportedPlatform
    }
}

pub struct TransportXdnaDevice<T> {
    pub transport: T,
}

impl<T> TransportXdnaDevice<T> {
    pub fn new(transport: T) -> Self {
        Self { transport }
    }
}

impl<T: XdnaTransport> XdnaDevice for TransportXdnaDevice<T> {
    type Error = TransportError;
    fn upload(&mut self, buffer_id: &str, bytes: u32) -> Result<(), Self::Error> {
        self.transport
            .upload_buffer(buffer_id, bytes)
            .map_err(|error| TransportError(error.to_string()))
    }
    fn execute(&mut self, command: &RuntimeCommand) -> Result<(), Self::Error> {
        let packet =
            bincode::serialize(command).map_err(|error| TransportError(error.to_string()))?;
        self.transport
            .submit_command(&packet)
            .map_err(|error| TransportError(error.to_string()))
    }
}

impl<T: XdnaTransport> XdnaCommandSubmitter for TransportXdnaDevice<T> {
    fn submit_command_buffer(&mut self, packet: &[u8]) -> Result<(), Self::Error> {
        self.transport
            .submit_command(packet)
            .map_err(|error| TransportError(error.to_string()))
    }

    fn submit_firmware_artifact(
        &mut self,
        program: &XdnaProgram,
        overlay: &[u8],
        ctrlcode: &[u8],
    ) -> Result<(), String> {
        self.transport
            .submit_firmware_artifact(program, overlay, ctrlcode)
    }
}

#[derive(Debug)]
pub struct TransportError(pub String);
impl std::fmt::Display for TransportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XdnaExecutionPhase {
    Prefill { tokens: u32 },
    Decode,
}

#[derive(Debug, Default)]
pub struct XdnaRuntime {
    resident_buffers: HashSet<String>,
    allocated_buffers: HashSet<String>,
    payload_resident_buffers: HashSet<String>,
    kv_tokens: HashMap<String, u64>,
}

impl XdnaRuntime {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn resident_buffers(&self) -> impl Iterator<Item = &str> {
        self.resident_buffers.iter().map(String::as_str)
    }

    pub fn kv_tokens(&self, model_id: &str) -> u64 {
        self.kv_tokens.get(model_id).copied().unwrap_or(0)
    }

    pub fn submit<D: XdnaDevice>(
        &mut self,
        program: &XdnaProgram,
        device: &mut D,
    ) -> Result<(), String> {
        self.submit_scoped("__anonymous__", program, device)
    }

    /// Submit a program while uploading compiler-owned persistent contents.
    /// Payloads are optional per buffer so activations and externally-bound
    /// resources remain metadata-only.
    pub fn submit_with_payloads<D: XdnaDevice>(
        &mut self,
        program: &XdnaProgram,
        payloads: &HashMap<String, Vec<u8>>,
        device: &mut D,
    ) -> Result<(), String> {
        self.submit_scoped_with_payloads("__anonymous__", program, payloads, device)
    }

    pub fn submit_artifact<D: XdnaDevice>(
        &mut self,
        artifact: &XdnaArtifact,
        device: &mut D,
    ) -> Result<(), String> {
        artifact.validate()?;
        // Construct and validate the command-buffer payload before any
        // residency mutation or device submission occurs.
        artifact.command_buffer()?.encode()?;
        self.submit_scoped(&artifact.manifest.model_id, &artifact.program, device)
    }

    /// Submit an artifact while binding payloads for both persistent tensors
    /// (weights/KV, uploaded once) and transient tensors (activations,
    /// uploaded for this dispatch). Payload names use the native XDNA buffer
    /// identifiers from the compiled program.
    pub fn submit_artifact_with_payloads<D: XdnaDevice>(
        &mut self,
        artifact: &XdnaArtifact,
        payloads: &HashMap<String, Vec<u8>>,
        device: &mut D,
    ) -> Result<(), String> {
        artifact.validate()?;
        artifact.command_buffer()?.encode()?;
        self.submit_scoped_with_payloads(
            &artifact.manifest.model_id,
            &artifact.program,
            payloads,
            device,
        )
    }

    pub fn download_buffer<D: XdnaDevice>(
        &mut self,
        program: &XdnaProgram,
        buffer_id: &str,
        device: &mut D,
    ) -> Result<Option<Vec<u8>>, String> {
        let buffer = program
            .buffers
            .iter()
            .find(|buffer| buffer.id == buffer_id)
            .ok_or_else(|| format!("XDNA output buffer not found: {buffer_id}"))?;
        device
            .download_payload(&buffer.id, buffer.bytes)
            .map_err(|error| format!("download XDNA buffer {}: {error}", buffer.id))
    }

    pub fn submit_bound_artifact<D: XdnaCommandSubmitter>(
        &mut self,
        artifact: &XdnaArtifact,
        command: &XdnaCommandBuffer,
        device: &mut D,
    ) -> Result<(), String> {
        self.submit_bound_artifact_with_payloads(artifact, command, &HashMap::new(), device)
    }

    /// Submit a pre-bound native command buffer after staging real tensor
    /// contents. Persistent weights/KV payloads are uploaded only on the
    /// first submission for this model; transient activations are uploaded on
    /// every dispatch. The command sequence is owned entirely by `command`.
    pub fn submit_bound_artifact_with_payloads<D: XdnaCommandSubmitter>(
        &mut self,
        artifact: &XdnaArtifact,
        command: &XdnaCommandBuffer,
        payloads: &HashMap<String, Vec<u8>>,
        device: &mut D,
    ) -> Result<(), String> {
        artifact.validate()?;
        if command.program != artifact.program {
            return Err("bound XDNA command buffer does not match artifact program".into());
        }
        self.stage_scoped_with_payloads(
            &artifact.manifest.model_id,
            &artifact.program,
            payloads,
            device,
        )?;
        let overlay = artifact
            .overlay
            .as_deref()
            .ok_or_else(|| "XDNA artifact has no firmware overlay".to_string())?;
        let ctrlcode = artifact
            .ctrlcode
            .as_deref()
            .ok_or_else(|| "XDNA artifact has no firmware ctrlcode".to_string())?;
        device
            .submit_firmware_artifact(&artifact.program, overlay, ctrlcode)
            .map_err(|error| format!("submit XDNA firmware artifact: {error}"))
    }

    /// Phase-aware variant of native command-buffer submission. This keeps
    /// KV-cache capacity and token accounting identical for interpreted and
    /// firmware-bound execution paths.
    pub fn submit_bound_artifact_phase_with_payloads<D: XdnaCommandSubmitter>(
        &mut self,
        artifact: &XdnaArtifact,
        command: &XdnaCommandBuffer,
        phase: XdnaExecutionPhase,
        payloads: &HashMap<String, Vec<u8>>,
        device: &mut D,
    ) -> Result<(), String> {
        artifact.validate()?;
        if let XdnaExecutionPhase::Prefill { tokens } = phase {
            if tokens == 0 || tokens > artifact.manifest.prefill_chunk_tokens {
                return Err(format!(
                    "prefill token count {} exceeds compiled chunk {}",
                    tokens, artifact.manifest.prefill_chunk_tokens
                ));
            }
        }
        let delta = match phase {
            XdnaExecutionPhase::Prefill { tokens } => u64::from(tokens),
            XdnaExecutionPhase::Decode => 1,
        };
        if artifact.manifest.kv_cache_bytes_per_token > 0 {
            let next_tokens = self
                .kv_tokens(&artifact.manifest.model_id)
                .saturating_add(delta);
            let kv_capacity = artifact
                .manifest
                .tensors
                .iter()
                .filter(|tensor| tensor.name.to_ascii_lowercase().contains("kv"))
                .map(|tensor| tensor.bytes)
                .sum::<u64>();
            let required = next_tokens.saturating_mul(artifact.manifest.kv_cache_bytes_per_token);
            if required > kv_capacity {
                return Err(format!(
                    "KV cache capacity exceeded: requires {} bytes for {} tokens, manifest provides {}",
                    required, next_tokens, kv_capacity
                ));
            }
        }
        self.submit_bound_artifact_with_payloads(artifact, command, payloads, device)?;
        *self
            .kv_tokens
            .entry(artifact.manifest.model_id.clone())
            .or_default() += delta;
        Ok(())
    }

    fn submit_scoped<D: XdnaDevice>(
        &mut self,
        model_id: &str,
        program: &XdnaProgram,
        device: &mut D,
    ) -> Result<(), String> {
        self.submit_scoped_with_payloads(model_id, program, &HashMap::new(), device)
    }

    fn submit_scoped_with_payloads<D: XdnaDevice>(
        &mut self,
        model_id: &str,
        program: &XdnaProgram,
        payloads: &HashMap<String, Vec<u8>>,
        device: &mut D,
    ) -> Result<(), String> {
        self.stage_scoped_with_payloads(model_id, program, payloads, device)?;
        for command in &program.sequence {
            device
                .execute(command)
                .map_err(|e| format!("execute {:?}: {e}", command))?;
        }
        Ok(())
    }

    /// Stage resources without interpreting the sequence. A native command
    /// buffer already contains the complete execution schedule, so replaying
    /// the sequence before submitting it would execute the graph twice.
    fn stage_scoped_with_payloads<D: XdnaDevice>(
        &mut self,
        model_id: &str,
        program: &XdnaProgram,
        payloads: &HashMap<String, Vec<u8>>,
        device: &mut D,
    ) -> Result<(), String> {
        program.validate().map_err(|errors| errors.join("; "))?;
        for buffer in &program.buffers {
            let key = format!("{model_id}::{}", buffer.id);
            if let Some(payload) = payloads.get(&buffer.id) {
                if payload.len() > buffer.bytes as usize {
                    return Err(format!(
                        "payload {} is {} bytes but buffer capacity is {}",
                        buffer.id,
                        payload.len(),
                        buffer.bytes
                    ));
                }
                if !buffer.persistent || !self.payload_resident_buffers.contains(&key) {
                    device
                        .upload_payload(&buffer.id, payload)
                        .map_err(|e| format!("upload payload {}: {e}", buffer.id))?;
                }
                if buffer.persistent {
                    self.resident_buffers.insert(key);
                    self.payload_resident_buffers
                        .insert(format!("{model_id}::{}", buffer.id));
                }
                self.allocated_buffers
                    .insert(format!("{model_id}::{}", buffer.id));
            } else if buffer.persistent
                && !self.resident_buffers.contains(&key)
                && !self.allocated_buffers.contains(&key)
            {
                device
                    .upload(&buffer.id, buffer.bytes)
                    .map_err(|e| format!("upload {}: {e}", buffer.id))?;
                // Allocation is tracked separately from authoritative
                // residency so a later real payload can replace it.
                self.resident_buffers.insert(key.clone());
                self.allocated_buffers.insert(key);
            }
        }
        Ok(())
    }

    pub fn submit_phase<D: XdnaDevice>(
        &mut self,
        artifact: &XdnaArtifact,
        phase: XdnaExecutionPhase,
        device: &mut D,
    ) -> Result<(), String> {
        self.submit_phase_with_payloads(artifact, phase, &HashMap::new(), device)
    }

    /// Phase-aware submission variant that carries real persistent tensor
    /// contents into the first prefill/decode residency operation.
    pub fn submit_phase_with_payloads<D: XdnaDevice>(
        &mut self,
        artifact: &XdnaArtifact,
        phase: XdnaExecutionPhase,
        payloads: &HashMap<String, Vec<u8>>,
        device: &mut D,
    ) -> Result<(), String> {
        artifact.validate()?;
        if let XdnaExecutionPhase::Prefill { tokens } = phase {
            if tokens == 0 || tokens > artifact.manifest.prefill_chunk_tokens {
                return Err(format!(
                    "prefill token count {} exceeds compiled chunk {}",
                    tokens, artifact.manifest.prefill_chunk_tokens
                ));
            }
        }
        let delta = match phase {
            XdnaExecutionPhase::Prefill { tokens } => u64::from(tokens),
            XdnaExecutionPhase::Decode => 1,
        };
        if artifact.manifest.kv_cache_bytes_per_token > 0 {
            let next_tokens = self
                .kv_tokens(&artifact.manifest.model_id)
                .saturating_add(delta);
            let kv_capacity = artifact
                .manifest
                .tensors
                .iter()
                .filter(|tensor| tensor.name.to_ascii_lowercase().contains("kv"))
                .map(|tensor| tensor.bytes)
                .sum::<u64>();
            let required = next_tokens.saturating_mul(artifact.manifest.kv_cache_bytes_per_token);
            if required > kv_capacity {
                return Err(format!("KV cache capacity exceeded: requires {} bytes for {} tokens, manifest provides {}", required, next_tokens, kv_capacity));
            }
        }
        self.submit_scoped_with_payloads(
            &artifact.manifest.model_id,
            &artifact.program,
            payloads,
            device,
        )?;
        *self
            .kv_tokens
            .entry(artifact.manifest.model_id.clone())
            .or_default() += delta;
        Ok(())
    }

    pub fn invalidate_residency(&mut self) {
        self.resident_buffers.clear();
        self.allocated_buffers.clear();
        self.payload_resident_buffers.clear();
        self.kv_tokens.clear();
    }

    /// Evict one model's persistent buffers and reset its KV token accounting
    /// without disturbing other models sharing the device.
    pub fn invalidate_model_residency(&mut self, model_id: &str) {
        let prefix = format!("{model_id}::");
        self.resident_buffers
            .retain(|key| !key.starts_with(&prefix));
        self.allocated_buffers
            .retain(|key| !key.starts_with(&prefix));
        self.payload_resident_buffers
            .retain(|key| !key.starts_with(&prefix));
        self.kv_tokens.remove(model_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_spatial_ir::xdna::*;

    #[derive(Default)]
    struct Fake {
        uploads: usize,
        uploaded_bytes: Vec<u32>,
        payloads: Vec<Vec<u8>>,
        commands: usize,
        bound_submissions: usize,
    }
    impl XdnaDevice for Fake {
        type Error = String;
        fn upload(&mut self, _: &str, bytes: u32) -> Result<(), Self::Error> {
            self.uploads += 1;
            self.uploaded_bytes.push(bytes);
            Ok(())
        }
        fn upload_payload(&mut self, _: &str, payload: &[u8]) -> Result<(), Self::Error> {
            self.payloads.push(payload.to_vec());
            Ok(())
        }
        fn download_payload(
            &mut self,
            _: &str,
            bytes: u32,
        ) -> Result<Option<Vec<u8>>, Self::Error> {
            Ok(Some(vec![9; bytes as usize]))
        }
        fn execute(&mut self, _: &RuntimeCommand) -> Result<(), Self::Error> {
            self.commands += 1;
            Ok(())
        }
    }

    impl XdnaCommandSubmitter for Fake {
        fn submit_command_buffer(&mut self, _: &[u8]) -> Result<(), Self::Error> {
            self.bound_submissions += 1;
            Ok(())
        }

        fn submit_firmware_artifact(
            &mut self,
            _: &XdnaProgram,
            overlay: &[u8],
            ctrlcode: &[u8],
        ) -> Result<(), String> {
            if overlay.is_empty() || ctrlcode.is_empty() {
                return Err("empty firmware artifact".into());
            }
            self.bound_submissions += 1;
            Ok(())
        }
    }

    #[derive(Default)]
    struct RecordingTransport {
        overlay: Vec<u8>,
        ctrlcode: Vec<u8>,
    }

    impl XdnaTransport for RecordingTransport {
        type Error = String;

        fn upload_buffer(&mut self, _: &str, _: u32) -> Result<(), Self::Error> {
            Ok(())
        }

        fn submit_command(&mut self, _: &[u8]) -> Result<(), Self::Error> {
            Ok(())
        }

        fn submit_firmware_artifact(
            &mut self,
            _: &XdnaProgram,
            overlay: &[u8],
            ctrlcode: &[u8],
        ) -> Result<(), String> {
            self.overlay = overlay.to_vec();
            self.ctrlcode = ctrlcode.to_vec();
            Ok(())
        }
    }

    #[test]
    fn transport_adapter_forwards_firmware_pair() {
        let program = XdnaProgram {
            topology: XdnaTopology::xdna2(),
            buffers: vec![],
            fifos: vec![],
            transfers: vec![],
            workers: vec![],
            barriers: vec![],
            sequence: vec![],
        };
        let mut device = TransportXdnaDevice::new(RecordingTransport::default());
        device
            .submit_firmware_artifact(&program, b"overlay", b"ctrlcode")
            .unwrap();
        assert_eq!(device.transport.overlay, b"overlay");
        assert_eq!(device.transport.ctrlcode, b"ctrlcode");
    }

    #[test]
    fn bound_submission_does_not_replay_interpreted_sequence() {
        let program = XdnaProgram {
            topology: XdnaTopology::xdna2(),
            buffers: vec![XdnaBuffer {
                id: "weights".into(),
                bytes: 2,
                element_type: XdnaElementType::Int8,
                shape: vec![2],
                memory: XdnaMemory::Shared,
                persistent: true,
            }],
            fifos: vec![],
            transfers: vec![],
            workers: vec![],
            barriers: vec![],
            sequence: vec![RuntimeCommand::Signal {
                event_id: "complete".into(),
            }],
        };
        let artifact = XdnaArtifact {
            program: program.clone(),
            manifest: prism_spatial_ir::xdna_manifest::XdnaModelManifest {
                model_id: "bound-model".into(),
                compiler_abi: "prism-xdna-v1".into(),
                supported_generations: vec![XdnaGeneration::Aie2p],
                tensors: vec![prism_spatial_ir::xdna_manifest::XdnaTensorManifest {
                    name: "weights".into(),
                    bytes: 2,
                    quantization: None,
                    residency: prism_spatial_ir::xdna_manifest::ResidencyPolicy::SharedPersistent,
                }],
                kv_cache_bytes_per_token: 0,
                prefill_chunk_tokens: 1,
                required_columns: 1,
            },
            overlay: Some(vec![0x50, 0x58, 0x4f, 0x56, 1, 0, 0, 0, 0, 0, 0, 0]),
            ctrlcode: Some(vec![0x50, 0x58, 0x43, 0x43, 1, 0, 0, 0, 0, 0, 0, 0]),
        };
        let command = XdnaCommandBuffer::from_program(&program).unwrap();
        let mut runtime = XdnaRuntime::new();
        let mut device = Fake::default();
        let mut payloads = HashMap::new();
        payloads.insert("weights".into(), vec![1, 2]);
        runtime
            .submit_bound_artifact_with_payloads(&artifact, &command, &payloads, &mut device)
            .unwrap();
        runtime
            .submit_bound_artifact_with_payloads(&artifact, &command, &payloads, &mut device)
            .unwrap();
        assert_eq!(device.uploads, 0);
        assert_eq!(device.payloads, vec![vec![1, 2]]);
        assert_eq!(device.commands, 0);
        assert_eq!(device.bound_submissions, 2);
    }

    #[test]
    fn bound_phase_submission_enforces_kv_capacity() {
        let program = XdnaProgram {
            topology: XdnaTopology::xdna2(),
            buffers: vec![XdnaBuffer {
                id: "kv".into(),
                bytes: 2,
                element_type: XdnaElementType::Int8,
                shape: vec![2],
                memory: XdnaMemory::Shared,
                persistent: true,
            }],
            fifos: vec![],
            transfers: vec![],
            workers: vec![],
            barriers: vec![],
            sequence: vec![],
        };
        let artifact = XdnaArtifact {
            program: program.clone(),
            manifest: prism_spatial_ir::xdna_manifest::XdnaModelManifest {
                model_id: "bound-kv".into(),
                compiler_abi: "prism-xdna-v1".into(),
                supported_generations: vec![XdnaGeneration::Aie2p],
                tensors: vec![prism_spatial_ir::xdna_manifest::XdnaTensorManifest {
                    name: "kv".into(),
                    bytes: 2,
                    quantization: None,
                    residency: prism_spatial_ir::xdna_manifest::ResidencyPolicy::SharedPersistent,
                }],
                kv_cache_bytes_per_token: 1,
                prefill_chunk_tokens: 2,
                required_columns: 1,
            },
            overlay: Some(vec![0x50, 0x58, 0x4f, 0x56, 1, 0, 0, 0, 0, 0, 0, 0]),
            ctrlcode: Some(vec![0x50, 0x58, 0x43, 0x43, 1, 0, 0, 0, 0, 0, 0, 0]),
        };
        let command = XdnaCommandBuffer::from_program(&program).unwrap();
        let mut runtime = XdnaRuntime::new();
        let mut device = Fake::default();
        runtime
            .submit_bound_artifact_phase_with_payloads(
                &artifact,
                &command,
                XdnaExecutionPhase::Prefill { tokens: 2 },
                &HashMap::new(),
                &mut device,
            )
            .unwrap();
        assert_eq!(runtime.kv_tokens("bound-kv"), 2);
        assert!(runtime
            .submit_bound_artifact_phase_with_payloads(
                &artifact,
                &command,
                XdnaExecutionPhase::Decode,
                &HashMap::new(),
                &mut device,
            )
            .is_err());
        assert_eq!(device.bound_submissions, 1);
    }

    #[test]
    fn persistent_buffers_upload_once_across_decode_steps() {
        let program = XdnaProgram {
            topology: XdnaTopology::xdna2(),
            buffers: vec![XdnaBuffer {
                id: "weights".into(),
                bytes: 2,
                element_type: XdnaElementType::Int8,
                shape: vec![2],
                memory: XdnaMemory::Shared,
                persistent: true,
            }],
            fifos: vec![],
            transfers: vec![],
            workers: vec![],
            barriers: vec![],
            sequence: vec![],
        };
        let mut runtime = XdnaRuntime::new();
        let mut device = Fake::default();
        runtime.submit(&program, &mut device).unwrap();
        runtime.submit(&program, &mut device).unwrap();
        assert_eq!(device.uploads, 1);
        assert_eq!(device.uploaded_bytes, vec![2]);
    }

    #[test]
    fn late_persistent_payload_replaces_metadata_only_allocation() {
        let program = XdnaProgram {
            topology: XdnaTopology::xdna2(),
            buffers: vec![XdnaBuffer {
                id: "weights".into(),
                bytes: 2,
                element_type: XdnaElementType::Int8,
                shape: vec![2],
                memory: XdnaMemory::Shared,
                persistent: true,
            }],
            fifos: vec![],
            transfers: vec![],
            workers: vec![],
            barriers: vec![],
            sequence: vec![],
        };
        let mut runtime = XdnaRuntime::new();
        let mut device = Fake::default();
        runtime.submit(&program, &mut device).unwrap();
        let mut payloads = HashMap::new();
        payloads.insert("weights".into(), vec![7, 9]);
        runtime
            .submit_with_payloads(&program, &payloads, &mut device)
            .unwrap();
        assert_eq!(device.uploads, 1);
        assert_eq!(device.payloads, vec![vec![7, 9]]);
    }

    #[test]
    fn model_scoped_invalidation_preserves_other_residency() {
        let program = XdnaProgram {
            topology: XdnaTopology::xdna2(),
            buffers: vec![XdnaBuffer {
                id: "weights".into(),
                bytes: 2,
                element_type: XdnaElementType::Int8,
                shape: vec![2],
                memory: XdnaMemory::Shared,
                persistent: true,
            }],
            fifos: vec![],
            transfers: vec![],
            workers: vec![],
            barriers: vec![],
            sequence: vec![],
        };
        let mut runtime = XdnaRuntime::new();
        let mut device = Fake::default();
        runtime
            .submit_scoped("model-a", &program, &mut device)
            .unwrap();
        runtime
            .submit_scoped("model-b", &program, &mut device)
            .unwrap();
        runtime.invalidate_model_residency("model-a");
        assert!(runtime
            .resident_buffers()
            .any(|buffer| buffer == "model-b::weights"));
        assert!(!runtime
            .resident_buffers()
            .any(|buffer| buffer == "model-a::weights"));
    }

    #[test]
    fn payload_submission_uploads_contents_once() {
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
        let mut payloads = HashMap::new();
        payloads.insert("weights".into(), vec![1, 2, 3, 4]);
        let mut runtime = XdnaRuntime::new();
        let mut device = Fake::default();
        runtime
            .submit_with_payloads(&program, &payloads, &mut device)
            .unwrap();
        runtime
            .submit_with_payloads(&program, &payloads, &mut device)
            .unwrap();
        assert_eq!(device.payloads, vec![vec![1, 2, 3, 4]]);
        assert_eq!(device.uploads, 0);
    }

    #[test]
    fn transient_payloads_upload_on_every_dispatch() {
        let program = XdnaProgram {
            topology: XdnaTopology::xdna2(),
            buffers: vec![XdnaBuffer {
                id: "activation".into(),
                bytes: 4,
                element_type: XdnaElementType::Int8,
                shape: vec![4],
                memory: XdnaMemory::Host,
                persistent: false,
            }],
            fifos: vec![],
            transfers: vec![],
            workers: vec![],
            barriers: vec![],
            sequence: vec![],
        };
        let mut payloads = HashMap::new();
        payloads.insert("activation".into(), vec![5, 6, 7, 8]);
        let mut runtime = XdnaRuntime::new();
        let mut device = Fake::default();
        runtime
            .submit_with_payloads(&program, &payloads, &mut device)
            .unwrap();
        runtime
            .submit_with_payloads(&program, &payloads, &mut device)
            .unwrap();
        assert_eq!(device.payloads, vec![vec![5, 6, 7, 8], vec![5, 6, 7, 8]]);
    }

    #[test]
    fn output_buffer_download_is_optional_but_native_when_available() {
        let program = XdnaProgram {
            topology: XdnaTopology::xdna2(),
            buffers: vec![XdnaBuffer {
                id: "C".into(),
                bytes: 4,
                element_type: XdnaElementType::Int8,
                shape: vec![4],
                memory: XdnaMemory::Host,
                persistent: false,
            }],
            fifos: vec![],
            transfers: vec![],
            workers: vec![],
            barriers: vec![],
            sequence: vec![],
        };
        let mut runtime = XdnaRuntime::new();
        let mut device = Fake::default();
        assert_eq!(
            runtime.download_buffer(&program, "C", &mut device).unwrap(),
            Some(vec![9; 4])
        );
    }

    #[test]
    fn phase_submission_tracks_prefill_and_decode_tokens() {
        let program = XdnaProgram {
            topology: XdnaTopology::xdna2(),
            buffers: vec![XdnaBuffer {
                id: "kv".into(),
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
        let artifact = XdnaArtifact {
            program,
            manifest: prism_spatial_ir::xdna_manifest::XdnaModelManifest {
                model_id: "model".into(),
                compiler_abi: "prism-xdna-v1".into(),
                supported_generations: vec![XdnaGeneration::Aie2p],
                tensors: vec![prism_spatial_ir::xdna_manifest::XdnaTensorManifest {
                    name: "kv".into(),
                    bytes: 4,
                    quantization: None,
                    residency: prism_spatial_ir::xdna_manifest::ResidencyPolicy::SharedPersistent,
                }],
                kv_cache_bytes_per_token: 1,
                prefill_chunk_tokens: 4,
                required_columns: 1,
            },
            overlay: Some(vec![0x50, 0x58, 0x4f, 0x56, 1, 0, 0, 0, 0, 0, 0, 0]),
            ctrlcode: Some(vec![0x50, 0x58, 0x43, 0x43, 1, 0, 0, 0, 0, 0, 0, 0]),
        };
        let mut runtime = XdnaRuntime::new();
        let mut device = Fake::default();
        runtime
            .submit_phase(
                &artifact,
                XdnaExecutionPhase::Prefill { tokens: 3 },
                &mut device,
            )
            .unwrap();
        runtime
            .submit_phase(&artifact, XdnaExecutionPhase::Decode, &mut device)
            .unwrap();
        assert_eq!(runtime.kv_tokens("model"), 4);
    }

    #[test]
    fn residency_is_scoped_by_model_id() {
        let program = XdnaProgram {
            topology: XdnaTopology::xdna2(),
            buffers: vec![XdnaBuffer {
                id: "weights".into(),
                bytes: 2,
                element_type: XdnaElementType::Int8,
                shape: vec![2],
                memory: XdnaMemory::Shared,
                persistent: true,
            }],
            fifos: vec![],
            transfers: vec![],
            workers: vec![],
            barriers: vec![],
            sequence: vec![],
        };
        let manifest = |model: &str| prism_spatial_ir::xdna_manifest::XdnaModelManifest {
            model_id: model.into(),
            compiler_abi: "prism-xdna-v1".into(),
            supported_generations: vec![XdnaGeneration::Aie2p],
            tensors: vec![prism_spatial_ir::xdna_manifest::XdnaTensorManifest {
                name: "weights".into(),
                bytes: 2,
                quantization: None,
                residency: prism_spatial_ir::xdna_manifest::ResidencyPolicy::SharedPersistent,
            }],
            kv_cache_bytes_per_token: 0,
            prefill_chunk_tokens: 1,
            required_columns: 1,
        };
        let mut runtime = XdnaRuntime::new();
        let mut device = Fake::default();
        runtime
            .submit_artifact(
                &XdnaArtifact {
                    program: program.clone(),
                    manifest: manifest("a"),
                    overlay: None,
                    ctrlcode: None,
                },
                &mut device,
            )
            .unwrap();
        runtime
            .submit_artifact(
                &XdnaArtifact {
                    program,
                    manifest: manifest("b"),
                    overlay: None,
                    ctrlcode: None,
                },
                &mut device,
            )
            .unwrap();
        assert_eq!(device.uploads, 2);
    }

    #[test]
    fn rejects_kv_manifest_without_persistent_storage() {
        let program = XdnaProgram {
            topology: XdnaTopology::xdna2(),
            buffers: vec![],
            fifos: vec![],
            transfers: vec![],
            workers: vec![],
            barriers: vec![],
            sequence: vec![],
        };
        let artifact = XdnaArtifact {
            program,
            manifest: prism_spatial_ir::xdna_manifest::XdnaModelManifest {
                model_id: "bad-kv".into(),
                compiler_abi: "prism-xdna-v1".into(),
                supported_generations: vec![XdnaGeneration::Aie2p],
                tensors: vec![],
                kv_cache_bytes_per_token: 1,
                prefill_chunk_tokens: 1,
                required_columns: 1,
            },
            overlay: None,
            ctrlcode: None,
        };
        assert!(artifact.validate().is_err());
    }
}
