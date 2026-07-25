//! Binding resolver — translates AOT tensor bindings into resolved buffers.
//!
//! This module owns the canonical authority for resolving the tensor
//! bindings emitted by the spatial-IR execution plan into the
//! [`ResolvedBuffer`] values that backends consume. The resolver is
//! stateless with respect to dispatch — it only looks up the model tensor
//! table and allocates output buffers when the binding names a new
//! intermediate result. The `BindingResolver` trait is the same trait
//! implemented by other backends; the CImage-specific implementation lives
//! here because every other resolver needs the same shape / dtype / size
//! arithmetic.
//!
//! Per-step `runtime_outputs` caching makes intermediate activations
//! available to subsequent steps without re-reading the CImage.

use std::collections::HashMap;

use prism_spatial_ir::BindingResolver;
use prism_spatial_ir::BufferStorage;
use prism_spatial_ir::ResolvedBuffer;
use prism_spatial_ir::execution_plan::{FusedScheduleStep, TensorBinding};

use super::model::RuntimeModel;

/// Resolves AOT tensor bindings against the loaded CImage tensor table.
pub struct CImageBindingResolver<'a> {
    pub model: &'a RuntimeModel,
    pub runtime_outputs: HashMap<String, ResolvedBuffer>,
}

impl BindingResolver for CImageBindingResolver<'_> {
    fn resolve_inputs(&mut self, step: &FusedScheduleStep) -> Result<Vec<ResolvedBuffer>, String> {
        step.input_tensors
            .iter()
            .map(|binding| {
                self.runtime_outputs
                    .get(&binding.name)
                    .cloned()
                    .map(Ok)
                    .unwrap_or_else(|| self.resolve(binding, &step.input_region, step.zero_copy))
            })
            .collect()
    }

    fn resolve_outputs(&mut self, step: &FusedScheduleStep) -> Result<Vec<ResolvedBuffer>, String> {
        step.output_tensors
            .iter()
            .map(|binding| {
                if self.model.tensors.contains_key(&binding.name) {
                    self.resolve(binding, &step.output_region, step.zero_copy)
                } else {
                    Ok(ResolvedBuffer {
                        name: binding.name.clone(),
                        element_type: binding.element_type.clone(),
                        region: step.output_region.clone(),
                        byte_length: binding
                            .shape
                            .iter()
                            .copied()
                            .fold(1u64, u64::saturating_mul)
                            .saturating_mul(match binding.element_type.as_str() {
                                "int8" => 1,
                                "int32" => 4,
                                "fp16" => 2,
                                "fp32" => 4,
                                _ => 1,
                            }) as usize,
                        zero_copy: false,
                        file_offset: None,
                        storage: BufferStorage::RuntimeOwned,
                        shape: binding.shape.clone(),
                        payload: Some(vec![
                            0;
                            binding
                                .shape
                                .iter()
                                .copied()
                                .fold(1u64, u64::saturating_mul)
                                .saturating_mul(match binding.element_type.as_str() {
                                    "int8" => 1,
                                    "int32" | "fp32" => 4,
                                    "fp16" => 2,
                                    _ => 1,
                                }) as usize
                        ]),
                    })
                }
            })
            .collect()
    }

    fn commit_outputs(
        &mut self,
        _step: &FusedScheduleStep,
        outputs: &[ResolvedBuffer],
    ) -> Result<(), String> {
        for output in outputs {
            self.runtime_outputs
                .insert(output.name.clone(), output.clone());
        }
        Ok(())
    }
}

impl CImageBindingResolver<'_> {
    fn resolve(
        &self,
        binding: &TensorBinding,
        region: &str,
        _zero_copy: bool,
    ) -> Result<ResolvedBuffer, String> {
        let payload = self
            .model
            .tensors
            .get(&binding.name)
            .ok_or_else(|| format!("CImage tensor binding '{}' is missing", binding.name))?;
        let record = self
            .model
            .tensor_records
            .get(&binding.name)
            .ok_or_else(|| format!("CImage tensor record '{}' is missing", binding.name))?;
        let expected_bytes = match binding.element_type.as_str() {
            "int8" => (record.dim_m as usize).saturating_mul(record.dim_n as usize),
            "int32" => (record.dim_m as usize)
                .saturating_mul(record.dim_n as usize)
                .saturating_mul(4),
            "fp16" => (record.dim_m as usize)
                .saturating_mul(record.dim_n as usize)
                .saturating_mul(2),
            _ => payload.len(),
        };
        if payload.len() < expected_bytes {
            return Err(format!(
                "CImage tensor '{}' is {} bytes, expected at least {}",
                binding.name,
                payload.len(),
                expected_bytes
            ));
        }
        Ok(ResolvedBuffer {
            name: binding.name.clone(),
            element_type: binding.element_type.clone(),
            region: region.into(),
            byte_length: payload.len(),
            // RuntimeModel currently owns copied Vec<u8> payloads. Do not
            // advertise zero-copy without also exposing the mapped file
            // offset for a backend to bind.
            zero_copy: self.model.mapped_cimage.is_some(),
            file_offset: Some(record.offset),
            storage: BufferStorage::MappedCImage,
            shape: if record.dim_m == 0 || record.dim_n == 0 {
                vec![]
            } else {
                vec![record.dim_m as u64, record.dim_n as u64]
            },
            payload: Some(payload.clone()),
        })
    }
}
