//! ANE dispatch methods on [`UnifiedRuntime`].
//!
//! This module owns the canonical authority for dispatching embedded
//! stateless int8 ANE programs through the orchestrator. The actual device
//! invocation lives in [`super::super::ane_backend`]; the methods here
//! accept activation / weight buffers in host memory, validate the int8
//! shape contract, and delegate to the per-tile device helpers.
//!
//! All entry points are gated on `cfg(all(feature = "ane", target_os =
//! "macos"))` so this module is empty on other targets.

use super::super::RuntimeError;
use super::UnifiedRuntime;

#[cfg(all(feature = "ane", target_os = "macos"))]
impl UnifiedRuntime {
    /// Dispatch a registered stateless int8 ANE program. The model payload is
    /// unpacked only at program-load time; all activation, weight, and output
    /// tensors use IOSurface-backed arenas for the actual prediction.
    pub fn dispatch_ane_int8(
        &self,
        program_name: &str,
        activation: &[i8],
        activation_shape: (u32, u32),
        weights: &[i8],
        weight_shape: (u32, u32),
    ) -> Result<Vec<i8>, RuntimeError> {
        self.dispatch_ane_int8_i32(
            program_name,
            activation,
            activation_shape,
            weights,
            weight_shape,
        )
        .map(|output| {
            output
                .into_iter()
                .map(|value| value.clamp(i8::MIN as i32, i8::MAX as i32) as i8)
                .collect()
        })
    }

    pub fn dispatch_ane_int8_i32(
        &self,
        program_name: &str,
        activation: &[i8],
        activation_shape: (u32, u32),
        weights: &[i8],
        weight_shape: (u32, u32),
    ) -> Result<Vec<i32>, RuntimeError> {
        use super::super::ane_backend::{copy_int8_to_arena, read_int32_from_arena};
        let (record, packed_model) = self.model.get_ane_program(program_name).ok_or_else(|| {
            RuntimeError::ExecutionFailed(format!("ANE program not found: {program_name}"))
        })?;
        if record.input_dtype != "int8" || record.output_dtype != "int8" {
            return Err(RuntimeError::UnsupportedMode(
                "ANE program is not int8".into(),
            ));
        }
        let activation_len = (activation_shape.0 as usize)
            .checked_mul(activation_shape.1 as usize)
            .ok_or_else(|| RuntimeError::ExecutionFailed("activation shape overflows".into()))?;
        let weight_len = (weight_shape.0 as usize)
            .checked_mul(weight_shape.1 as usize)
            .ok_or_else(|| RuntimeError::ExecutionFailed("weight shape overflows".into()))?;
        if activation.len() != activation_len || weights.len() != weight_len {
            return Err(RuntimeError::ExecutionFailed(
                "int8 inputs must exactly match their declared IOSurface shapes".into(),
            ));
        }
        let base = std::env::temp_dir().join(format!("prism-ane-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&base).map_err(|e| RuntimeError::ExecutionFailed(e.to_string()))?;
        let result = (|| {
            prism_ane::unpack_mlmodelc(packed_model, &base)
                .map_err(RuntimeError::ExecutionFailed)?;
            let model = prism_ane::coreml_bridge::CoreMlModel::load(&base)
                .map_err(RuntimeError::ExecutionFailed)?;
            let activation_arena = prism_ane::Arena::new(
                activation_shape.0,
                activation_shape.1,
                prism_ane::arena::Dtype::Int8,
            )
            .map_err(RuntimeError::ExecutionFailed)?;
            let weights_arena = prism_ane::Arena::new(
                weight_shape.0,
                weight_shape.1,
                prism_ane::arena::Dtype::Int8,
            )
            .map_err(RuntimeError::ExecutionFailed)?;
            let output_shape = (activation_shape.0, weight_shape.1);
            let mut output_arena = prism_ane::Arena::new(
                output_shape.0,
                output_shape.1,
                prism_ane::arena::Dtype::Int32,
            )
            .map_err(RuntimeError::ExecutionFailed)?;
            copy_int8_to_arena(&activation_arena, activation)
                .map_err(RuntimeError::ExecutionFailed)?;
            copy_int8_to_arena(&weights_arena, weights).map_err(RuntimeError::ExecutionFailed)?;
            model
                .predict_two_int8(
                    &record.activation_input,
                    &activation_arena,
                    &record.weights_input,
                    &weights_arena,
                    &record.output,
                    &mut output_arena,
                )
                .map_err(RuntimeError::ExecutionFailed)?;
            read_int32_from_arena(
                &output_arena,
                output_shape.0 as usize * output_shape.1 as usize,
            )
            .map_err(RuntimeError::ExecutionFailed)
        })();
        let _ = std::fs::remove_dir_all(&base);
        result
    }

    /// Dispatch a complete matrix through stateless ANE tiles, preserving
    /// int32 accumulators while K-slices are combined on the host.
    pub fn dispatch_ane_int8_tiled(
        &self,
        program_name: &str,
        plan: &prism_ecs_quantization::ane_orchestration::AneTiledDispatchPlan,
        activation: &[i8],
        weights: &[i8],
    ) -> Result<Vec<i32>, RuntimeError> {
        let first_shape = plan
            .dispatches
            .first()
            .map(|tile| (tile.rows, tile.cols, tile.depth));
        if plan
            .dispatches
            .iter()
            .any(|tile| Some((tile.rows, tile.cols, tile.depth)) != first_shape)
        {
            return Err(RuntimeError::UnsupportedMode(
                "fixed ANE program cannot execute heterogeneous edge-tile shapes; use dispatch_ane_int8_tiled_with_programs".into(),
            ));
        }
        self.dispatch_ane_int8_tiled_with_programs(plan, activation, weights, |_| {
            Ok(program_name.to_string())
        })
    }

    /// Shape-aware variant for plans with edge tiles. The resolver selects a
    /// separately compiled stateless Core ML program for `(rows, cols, depth)`.
    pub fn dispatch_ane_int8_tiled_with_programs<F>(
        &self,
        plan: &prism_ecs_quantization::ane_orchestration::AneTiledDispatchPlan,
        activation: &[i8],
        weights: &[i8],
        mut program_for_shape: F,
    ) -> Result<Vec<i32>, RuntimeError>
    where
        F: FnMut((usize, usize, usize)) -> Result<String, String>,
    {
        prism_ecs_quantization::ane_orchestration::execute_tiled_int8(
            plan,
            activation,
            weights,
            |(rows, cols), tile_activation, tile_weights| {
                let depth = tile_activation.len() / rows;
                let program_name =
                    program_for_shape((rows, cols, depth)).map_err(|error| error.to_string())?;
                self.dispatch_ane_int8_i32(
                    &program_name,
                    tile_activation,
                    (rows as u32, depth as u32),
                    tile_weights,
                    (depth as u32, cols as u32),
                )
                .map_err(|error| error.to_string())
            },
        )
        .map_err(RuntimeError::ExecutionFailed)
    }

    pub fn dispatch_ane_int8_planar(
        &self,
        program_name: &str,
        activation: &[u8],
        activation_shape: (u32, u32),
        bias: &[u8],
        bias_shape: (u32, u32),
    ) -> Result<Vec<u8>, RuntimeError> {
        let (record, packed_model) = self.model.get_ane_program(program_name).ok_or_else(|| {
            RuntimeError::ExecutionFailed(format!("ANE program not found: {program_name}"))
        })?;
        if record.input_dtype != "int8" || record.output_dtype != "int8" {
            return Err(RuntimeError::UnsupportedMode(
                "ANE program is not int8 planar".into(),
            ));
        }
        let activation_len = (activation_shape.0 as usize)
            .checked_mul(activation_shape.1 as usize)
            .ok_or_else(|| {
                RuntimeError::ExecutionFailed("planar activation shape overflows".into())
            })?;
        let bias_len = (bias_shape.0 as usize)
            .checked_mul(bias_shape.1 as usize)
            .ok_or_else(|| RuntimeError::ExecutionFailed("planar bias shape overflows".into()))?;
        if activation.len() != activation_len || bias.len() != bias_len {
            return Err(RuntimeError::ExecutionFailed(
                "planar int8 inputs must exactly match their declared IOSurface shapes".into(),
            ));
        }
        let base = std::env::temp_dir().join(format!("prism-ane-planar-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&base).map_err(|e| RuntimeError::ExecutionFailed(e.to_string()))?;
        let result = (|| {
            prism_ane::unpack_mlmodelc(packed_model, &base)
                .map_err(RuntimeError::ExecutionFailed)?;
            let model = prism_ane::coreml_bridge::CoreMlModel::load(&base)
                .map_err(RuntimeError::ExecutionFailed)?;
            let activation_arena = prism_ane::Arena::new(
                activation_shape.0,
                activation_shape.1,
                prism_ane::arena::Dtype::Int8,
            )
            .map_err(RuntimeError::ExecutionFailed)?;
            let bias_arena =
                prism_ane::Arena::new(bias_shape.0, bias_shape.1, prism_ane::arena::Dtype::Int8)
                    .map_err(RuntimeError::ExecutionFailed)?;
            let mut output_arena = prism_ane::Arena::new(
                activation_shape.0,
                activation_shape.1,
                prism_ane::arena::Dtype::Int8,
            )
            .map_err(RuntimeError::ExecutionFailed)?;
            for (arena, bytes) in [(&activation_arena, activation), (&bias_arena, bias)] {
                if bytes.len() > arena.info.byte_size as usize {
                    return Err(RuntimeError::ExecutionFailed(
                        "int8 planar input exceeds IOSurface arena".into(),
                    ));
                }
                arena.lock().map_err(RuntimeError::ExecutionFailed)?;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        bytes.as_ptr(),
                        arena.info.base_address as *mut u8,
                        bytes.len(),
                    );
                }
                arena.unlock().map_err(RuntimeError::ExecutionFailed)?;
            }
            model
                .predict_two_int8_planar(
                    &record.activation_input,
                    &activation_arena,
                    &record.weights_input,
                    &bias_arena,
                    &record.output,
                    &mut output_arena,
                )
                .map_err(RuntimeError::ExecutionFailed)?;
            output_arena.lock().map_err(RuntimeError::ExecutionFailed)?;
            let output = unsafe {
                std::slice::from_raw_parts(
                    output_arena.info.base_address as *const u8,
                    output_arena.info.byte_size as usize,
                )
                .to_vec()
            };
            output_arena
                .unlock()
                .map_err(RuntimeError::ExecutionFailed)?;
            Ok(output)
        })();
        let _ = std::fs::remove_dir_all(&base);
        result
    }
}
