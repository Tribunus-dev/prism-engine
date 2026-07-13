//! Engram application — CPU and Metal runtime for applying trained engrams.
//!
//! An engram's payload is applied to activations at a specific region of the
//! execution graph according to the `EngramApplication` mode (additive,
//! multiplicative, low-rank projection, latent prefix, or adapter activation).

use crate::ecs::training_target::spec::EngramApplication;

/// Apply an engram payload to activations on the CPU.
///
/// The `payload` byte slice is interpreted according to the `application` mode:
/// - `AdditiveResidual` — payload is `f32` residuals added element-wise.
/// - `MultiplicativeModulation` — payload is `f32` scales multiplied
///   element-wise.
/// - `LowRankProjection` — placeholder for LoRA-style A/B matrix application.
/// - `LatentPrefix` — placeholder.
/// - `AdapterActivation` — placeholder.
pub fn apply_cpu(
    application: &EngramApplication,
    activations: &mut [f32],
    payload: &[u8],
) -> Result<(), String> {
    if payload.len() % std::mem::size_of::<f32>() != 0 {
        return Err("engram payload is not f32 aligned".into());
    }
    let values: Vec<f32> = payload
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().expect("chunk is four bytes")))
        .collect();
    match application {
        EngramApplication::AdditiveResidual => {
            if values.len() != activations.len() {
                return Err(format!(
                    "additive engram width {} does not match activations {}",
                    values.len(),
                    activations.len()
                ));
            }
            for (a, r) in activations.iter_mut().zip(&values) {
                *a += r;
            }
            Ok(())
        }
        EngramApplication::MultiplicativeModulation => {
            if values.len() != activations.len() {
                return Err(format!(
                    "multiplicative engram width {} does not match activations {}",
                    values.len(),
                    activations.len()
                ));
            }
            for (a, s) in activations.iter_mut().zip(&values) {
                *a *= s;
            }
            Ok(())
        }
        EngramApplication::LowRankProjection => {
            // Simple low-rank adaptation (LoRA-style)
            // payload = [A_matrix_bytes, B_matrix_bytes]
            // TODO: actual matrix-multiply when the decomposition format is settled.
            Ok(())
        }
        EngramApplication::LatentPrefix => {
            // TODO: prepend or splice latent prefix tokens.
            Ok(())
        }
        EngramApplication::AdapterActivation => {
            // TODO: run a small adapter MLP.
            Ok(())
        }
    }
}

/// Apply an engram payload to activations on Metal GPU.
///
/// Compiles an inline Metal shader and dispatches it on the system-default
/// GPU device. Supports `AdditiveResidual` (element-wise add) and
/// `MultiplicativeModulation` (element-wise multiply). Other modes return
/// an error.
#[cfg(feature = "metal-dispatch")]
pub fn apply_metal(
    application: &EngramApplication,
    activations: &mut [f32],
    payload: &[u8],
) -> Result<(), String> {
    // Get the system default Metal device
    let device =
        metal::Device::system_default().ok_or_else(|| "no Metal device available".to_string())?;

    // Build the kernel source inline (simple additive/multiplicative)
    // The kernel does element-wise add or multiply of activations by engram values
    let (src, kernel_name) = match application {
        EngramApplication::AdditiveResidual => (
            "#include <metal_stdlib>\nusing namespace metal;\n\
            kernel void engram_add(device float* activations [[buffer(0)]],\n\
                                   constant float* values [[buffer(1)]],\n\
                                   uint gid [[thread_position_in_grid]]) {\n\
                activations[gid] += values[gid];\n\
            }",
            "engram_add",
        ),
        EngramApplication::MultiplicativeModulation => (
            "#include <metal_stdlib>\nusing namespace metal;\n\
            kernel void engram_mul(device float* activations [[buffer(0)]],\n\
                                   constant float* values [[buffer(1)]],\n\
                                   uint gid [[thread_position_in_grid]]) {\n\
                activations[gid] *= values[gid];\n\
            }",
            "engram_mul",
        ),
        _ => {
            return Err(format!(
                "Metal engram {:?} not yet implemented",
                application
            ))
        }
    };

    // Compile the kernel
    let opts = metal::CompileOptions::new();
    let lib = device
        .new_library_with_source(src, &opts)
        .map_err(|e| format!("Metal compile error: {:?}", e))?;

    let function = lib
        .get_function(kernel_name, None::<metal::FunctionConstantValues>)
        .map_err(|e| format!("Metal function '{}' not found: {:?}", kernel_name, e))?;

    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| format!("Metal pipeline failed for '{}': {:?}", kernel_name, e))?;

    // Convert payload bytes to f32 values
    if payload.len() % std::mem::size_of::<f32>() != 0 {
        return Err("engram payload is not f32 aligned".into());
    }
    let values: Vec<f32> = payload
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().expect("chunk is four bytes")))
        .collect();

    if values.len() != activations.len() {
        return Err(format!(
            "engram width {} does not match activations {}",
            values.len(),
            activations.len()
        ));
    }

    // Create Metal buffers (StorageModeShared for CPU readback)
    let activation_buf = device.new_buffer_with_data(
        activations.as_ptr() as *const std::ffi::c_void,
        (activations.len() * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let values_buf = device.new_buffer_with_data(
        values.as_ptr() as *const std::ffi::c_void,
        (values.len() * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let cmd_queue = device.new_command_queue();
    let cmd_buf = cmd_queue.new_command_buffer();
    let encoder = cmd_buf.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&activation_buf), 0);
    encoder.set_buffer(1, Some(&values_buf), 0);

    let grid = metal::MTLSize::new(activations.len() as u64, 1, 1);
    let threadgroup = metal::MTLSize::new(256, 1, 1);
    encoder.dispatch_threads(grid, threadgroup);

    encoder.end_encoding();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    // Read back results
    let ptr = activation_buf.contents() as *const f32;
    let result = unsafe { std::slice::from_raw_parts(ptr, activations.len()) };
    activations.copy_from_slice(result);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn payload(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn additive_application_requires_matching_width() {
        let mut activations = vec![1.0, 2.0];
        apply_cpu(
            &EngramApplication::AdditiveResidual,
            &mut activations,
            &payload(&[0.5, -1.0]),
        )
        .unwrap();
        assert_eq!(activations, vec![1.5, 1.0]);
        assert!(apply_cpu(
            &EngramApplication::AdditiveResidual,
            &mut activations,
            &payload(&[1.0]),
        )
        .is_err());
    }
}

/// Non-Metal stub when `metal-dispatch` is disabled.
#[cfg(not(feature = "metal-dispatch"))]
pub fn apply_metal(
    _application: &EngramApplication,
    _activations: &mut [f32],
    _payload: &[u8],
) -> Result<(), String> {
    Err("Metal engram application not available — use CPU path".into())
}
