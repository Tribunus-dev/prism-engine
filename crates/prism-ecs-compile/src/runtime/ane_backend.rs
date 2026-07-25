//! Embedded ANE route backend — Core ML dispatch for stateless int8 programs.
//!
//! This module owns the canonical authority for dispatching embedded
//! stateless int8 ANE programs through the runtime's [`super::UnifiedRuntime`]
//! and the [`AneRouteBackend`] trait. It is a thin adapter: it locates the
//! matching program for an AOT step, looks up the resolved tensor payloads,
//! delegates the actual device call to the runtime's ANE dispatch helpers,
//! and copies the int8 output into the caller's output binding layer.
//!
//! The dispatch is gated on `cfg(all(feature = "ane", target_os = "macos"))`
//! in the helpers that touch ANE; the dispatch methods themselves return a
//! clear "ANE route is unavailable on this target" error on other targets
//! so the trait still compiles but cannot succeed.

use std::collections::HashMap;

use prism_spatial_ir::ResolvedBuffer;
use prism_spatial_ir::execution_plan::FusedScheduleStep;

use super::unified::UnifiedRuntime;

/// Concrete ANE entry points used by the runtime route table. The ANE
/// implementation owns Core ML model loading and IOSurface arena binding;
/// the scheduler supplies the already-resolved tensor contract.
pub trait AneRouteBackend {
    fn dispatch_planar(
        &mut self,
        step: &FusedScheduleStep,
        inputs: &[ResolvedBuffer],
        outputs: &mut [ResolvedBuffer],
    ) -> Result<(), String>;
    fn dispatch_matrix(
        &mut self,
        step: &FusedScheduleStep,
        inputs: &[ResolvedBuffer],
        outputs: &mut [ResolvedBuffer],
    ) -> Result<(), String>;
}

/// ANE route implementation backed by the runtime's embedded stateless Core
/// ML programs. The produced output is retained by the caller's output
/// binding layer; this adapter owns only the device invocation.
pub struct EmbeddedAneRouteBackend<'a> {
    pub runtime: &'a UnifiedRuntime,
    pub outputs: HashMap<String, Vec<i8>>,
}

impl EmbeddedAneRouteBackend<'_> {
    #[cfg(all(feature = "ane", target_os = "macos"))]
    fn dispatch_int8(
        &mut self,
        step: &FusedScheduleStep,
        inputs: &[ResolvedBuffer],
        outputs: &mut [ResolvedBuffer],
    ) -> Result<(), String> {
        let program = self
            .runtime
            .model
            .ane_program_for_step(step)
            .ok_or_else(|| format!("no embedded ANE program matches step {}", step.step_id))?
            .0;
        let activation = inputs
            .iter()
            .find(|buffer| buffer.name == program.activation_input)
            .ok_or_else(|| "ANE activation binding is unresolved".to_string())?;
        let weights = inputs
            .iter()
            .find(|buffer| buffer.name == program.weights_input)
            .ok_or_else(|| "ANE weight binding is unresolved".to_string())?;
        let activation_owned;
        let activation_bytes = if let Some(payload) = activation.payload.as_deref() {
            payload
        } else {
            activation_owned = self
                .runtime
                .model
                .tensors
                .get(&activation.name)
                .cloned()
                .ok_or_else(|| {
                    format!("ANE activation payload '{}' is missing", activation.name)
                })?;
            &activation_owned
        };
        let weights_owned;
        let weight_bytes = if let Some(payload) = weights.payload.as_deref() {
            payload
        } else {
            weights_owned = self
                .runtime
                .model
                .tensors
                .get(&weights.name)
                .cloned()
                .ok_or_else(|| format!("ANE weight payload '{}' is missing", weights.name))?;
            &weights_owned
        };
        if activation_bytes.len() % std::mem::size_of::<i8>() != 0
            || weight_bytes.len() % std::mem::size_of::<i8>() != 0
        {
            return Err("ANE int8 binding payload is not byte aligned".into());
        }
        let activation_shape = shape_2d(activation)?;
        let weight_shape = shape_2d(weights)?;
        let activation_values = activation_bytes
            .iter()
            .map(|&value| value as i8)
            .collect::<Vec<_>>();
        let weight_values = weight_bytes
            .iter()
            .map(|&value| value as i8)
            .collect::<Vec<_>>();
        let output = self
            .runtime
            .dispatch_ane_int8(
                &program.name,
                &activation_values,
                activation_shape,
                &weight_values,
                weight_shape,
            )
            .map_err(|error| error.to_string())?;
        if let Some(binding) = step.output_tensors.first() {
            if let Some(buffer) = outputs.first_mut() {
                buffer.byte_length = output.len();
                buffer.payload = Some(output.iter().map(|&value| value as u8).collect());
            }
            self.outputs.insert(binding.name.clone(), output);
        }
        Ok(())
    }

    #[cfg(all(feature = "ane", target_os = "macos"))]
    fn dispatch_fp16(
        &mut self,
        step: &FusedScheduleStep,
        inputs: &[ResolvedBuffer],
        outputs: &mut [ResolvedBuffer],
    ) -> Result<(), String> {
        let program = self
            .runtime
            .model
            .ane_program_for_step(step)
            .ok_or_else(|| {
                format!(
                    "no embedded ANE planar program matches step {}",
                    step.step_id
                )
            })?
            .0;
        let activation = inputs
            .iter()
            .find(|buffer| buffer.name == program.activation_input)
            .ok_or_else(|| "ANE planar activation binding is unresolved".to_string())?;
        let bias = inputs
            .iter()
            .find(|buffer| buffer.name == program.weights_input)
            .ok_or_else(|| "ANE planar bias binding is unresolved".to_string())?;
        let activation_owned;
        let activation_bytes = if let Some(payload) = activation.payload.as_deref() {
            payload
        } else {
            activation_owned = self
                .runtime
                .model
                .tensors
                .get(&activation.name)
                .cloned()
                .ok_or_else(|| format!("ANE planar activation '{}' is missing", activation.name))?;
            &activation_owned
        };
        let bias_owned;
        let bias_bytes = if let Some(payload) = bias.payload.as_deref() {
            payload
        } else {
            bias_owned = self
                .runtime
                .model
                .tensors
                .get(&bias.name)
                .cloned()
                .ok_or_else(|| format!("ANE planar bias '{}' is missing", bias.name))?;
            &bias_owned
        };
        let result = self
            .runtime
            .dispatch_ane_int8_planar(
                &program.name,
                activation_bytes,
                shape_2d(activation)?,
                bias_bytes,
                shape_2d(bias)?,
            )
            .map_err(|error| error.to_string())?;
        if let Some(binding) = outputs.first_mut() {
            binding.byte_length = result.len();
            binding.payload = Some(result.clone());
        }
        if let Some(binding) = step.output_tensors.first() {
            self.outputs.insert(
                binding.name.clone(),
                result.iter().map(|&v| v as i8).collect(),
            );
        }
        Ok(())
    }

    #[cfg(not(all(feature = "ane", target_os = "macos")))]
    fn dispatch_int8(
        &mut self,
        _step: &FusedScheduleStep,
        _inputs: &[ResolvedBuffer],
        _outputs: &mut [ResolvedBuffer],
    ) -> Result<(), String> {
        Err("ANE route is unavailable on this target or feature set".into())
    }
}

#[cfg(all(feature = "ane", target_os = "macos"))]
fn shape_2d(buffer: &ResolvedBuffer) -> Result<(u32, u32), String> {
    let dims = buffer.shape.as_slice();
    if dims.len() != 2 || dims.iter().any(|&dim| dim == 0 || dim > u32::MAX as u64) {
        return Err(format!(
            "ANE binding '{}' requires a non-zero 2D shape",
            buffer.name
        ));
    }
    Ok((dims[0] as u32, dims[1] as u32))
}

#[cfg(all(feature = "ane", target_os = "macos"))]
pub(crate) fn copy_int8_to_arena(arena: &prism_ane::Arena, values: &[i8]) -> Result<(), String> {
    if values.len() * std::mem::size_of::<i8>() > arena.info.byte_size as usize {
        return Err("int8 input exceeds IOSurface arena".into());
    }
    arena.lock()?;
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr() as *const u8,
            arena.info.base_address as *mut u8,
            values.len(),
        );
    }
    arena.unlock()
}

#[cfg(all(feature = "ane", target_os = "macos"))]
pub(crate) fn read_int32_from_arena(
    arena: &prism_ane::Arena,
    len: usize,
) -> Result<Vec<i32>, String> {
    if len * std::mem::size_of::<i32>() > arena.info.byte_size as usize {
        return Err("int32 output exceeds IOSurface arena".into());
    }
    arena.lock()?;
    let values =
        unsafe { std::slice::from_raw_parts(arena.info.base_address as *const i32, len).to_vec() };
    arena.unlock()?;
    Ok(values)
}

impl AneRouteBackend for EmbeddedAneRouteBackend<'_> {
    fn dispatch_planar(
        &mut self,
        step: &FusedScheduleStep,
        inputs: &[ResolvedBuffer],
        _outputs: &mut [ResolvedBuffer],
    ) -> Result<(), String> {
        #[cfg(all(feature = "ane", target_os = "macos"))]
        {
            self.dispatch_fp16(step, inputs, _outputs)
        }
        #[cfg(not(all(feature = "ane", target_os = "macos")))]
        {
            self.dispatch_int8(step, inputs, _outputs)
        }
    }

    fn dispatch_matrix(
        &mut self,
        step: &FusedScheduleStep,
        inputs: &[ResolvedBuffer],
        _outputs: &mut [ResolvedBuffer],
    ) -> Result<(), String> {
        self.dispatch_int8(step, inputs, _outputs)
    }
}
