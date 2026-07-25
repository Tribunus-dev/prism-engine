//! Runtime model — per-kernel binary, descriptor, artifact, and ANE accessors.
//!
//! This module owns the canonical authority for reading the
//! per-kernel payloads, the native XDNA artifacts, the embedded ANE
//! programs, the zero-copy mapped CImage buffer resolver, and the
//! typed kernel-artifact reconstructor. The dispatch layer in
//! [`super::super::unified::dispatch`] consumes
//! [`RuntimeModel::kernel_artifact`] when packing inputs for a kernel
//! variant; the rest are read-side accessors for tooling and
//! admission.

use prism_ecs_kernel::{KernelArtifact, KernelManifest, KernelPayload};
use prism_spatial_ir::execution_plan::FusedScheduleStep;
use prism_spatial_ir::ResolvedBuffer;

use super::super::RuntimeError;
use super::RuntimeModel;

impl RuntimeModel {
    /// Get a kernel's compiled binary by name.
    pub fn get_kernel(&self, name: &str) -> Option<&[u8]> {
        self.kernels.get(name).map(|v| v.as_slice())
    }

    /// Return a decoded, validated native XDNA artifact by name.
    pub fn xdna_artifact(&self, name: &str) -> Option<&prism_amd_npu_runtime::XdnaArtifact> {
        self.xdna_artifacts.get(name)
    }

    /// Resolve a CImage-backed binding to a bounds-checked read-only mapped
    /// slice for zero-copy backend binding.
    pub fn mapped_buffer<'a>(&'a self, buffer: &ResolvedBuffer) -> Result<&'a [u8], RuntimeError> {
        if !buffer.zero_copy {
            return Err(RuntimeError::UnsupportedMode(format!(
                "buffer '{}' is not marked zero-copy",
                buffer.name
            )));
        }
        let offset = buffer.file_offset.ok_or_else(|| {
            RuntimeError::InvalidCImage(format!("buffer '{}' has no file offset", buffer.name))
        })? as usize;
        let end = offset.checked_add(buffer.byte_length).ok_or_else(|| {
            RuntimeError::InvalidCImage(format!("buffer '{}' range overflow", buffer.name))
        })?;
        let mapped = self
            .mapped_cimage
            .as_ref()
            .ok_or_else(|| RuntimeError::InvalidCImage("CImage is not memory-mapped".into()))?;
        mapped.get(offset..end).ok_or_else(|| {
            RuntimeError::InvalidCImage(format!("buffer '{}' exceeds CImage mapping", buffer.name))
        })
    }

    /// Return an embedded ANE program and its declared multi-input contract.
    pub fn get_ane_program(&self, name: &str) -> Option<(&crate::cimage::AneProgramRecord, &[u8])> {
        self.ane_programs
            .get(name)
            .map(|(record, payload)| (record, payload.as_slice()))
    }

    /// Select the embedded ANE program whose declared tensor contract matches
    /// an AOT step. This keeps program selection tied to compiler-emitted
    /// bindings rather than positional or name-prefix guesses.
    pub fn ane_program_for_step(
        &self,
        step: &FusedScheduleStep,
    ) -> Option<(&crate::cimage::AneProgramRecord, &[u8])> {
        let input_names: std::collections::HashSet<&str> = step
            .input_tensors
            .iter()
            .map(|binding| binding.name.as_str())
            .collect();
        let output_names: std::collections::HashSet<&str> = step
            .output_tensors
            .iter()
            .map(|binding| binding.name.as_str())
            .collect();
        self.ane_programs.values().find_map(|(record, payload)| {
            (input_names.contains(record.activation_input.as_str())
                && input_names.contains(record.weights_input.as_str())
                && output_names.contains(record.output.as_str()))
            .then_some((record, payload.as_slice()))
        })
    }

    /// Reconstruct a typed artifact suitable for backend dispatch.
    pub fn kernel_artifact(&self, name: &str) -> Result<KernelArtifact, RuntimeError> {
        let binary = self
            .kernels
            .get(name)
            .ok_or_else(|| RuntimeError::KernelNotFound(name.into()))?;
        let descriptor = self
            .kernel_descriptors
            .get(name)
            .ok_or_else(|| {
                RuntimeError::InvalidCImage(format!("kernel '{name}' has no descriptor"))
            })?
            .clone();
        Ok(KernelArtifact {
            payloads: vec![KernelPayload {
                binary: binary.clone(),
                descriptor: descriptor.clone(),
            }],
            manifest: KernelManifest {
                kernels: vec![descriptor],
                fusion_plan: None,
                manifest_digest: String::new(),
            },
            artifact_digest: String::new(),
        })
    }
}
