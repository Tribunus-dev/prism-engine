//! Engram application — CPU and Metal runtime for applying trained engrams.
//!
//! An engram's payload is applied to activations at a specific region of the
//! execution graph according to the `EngramApplication` mode (additive,
//! multiplicative, low-rank projection, latent prefix, or adapter activation).

use crate::ecs::training_target::spec::EngramApplication;
use sha2::{Digest, Sha256};

#[cfg(feature = "metal-dispatch")]
use crate::ecs::canonical::kernel_abi::{DispatchGeometryPolicy, KernelAbi};
#[cfg(feature = "metal-dispatch")]
use crate::ecs::metal_backend::compiler::MetalBackendCompiler;

/// Dispatch engram application to the authoritative runtime path.
///
/// Uses `apply_metal` when the `metal-dispatch` feature is enabled,
/// falling back to `apply_cpu` otherwise. Modes that are not yet
/// implemented for production (LowRankProjection, LatentPrefix,
/// AdapterActivation) return an error instead of silently no-opping.
pub fn apply(
    application: &EngramApplication,
    activations: &mut [f32],
    payload: &[u8],
) -> Result<(), String> {
    match application {
        EngramApplication::AdditiveResidual | EngramApplication::MultiplicativeModulation => {
            #[cfg(feature = "metal-dispatch")]
            {
                apply_metal(application, activations, payload)
            }
            #[cfg(not(feature = "metal-dispatch"))]
            {
                apply_cpu(application, activations, payload)
            }
        }
        mode @ (EngramApplication::LowRankProjection
        | EngramApplication::LatentPrefix
        | EngramApplication::AdapterActivation) => Err(format!(
            "{mode:?} is not yet implemented — removed from production admission"
        )),
    }
}

/// Apply an engram with payload digest validation.
///
/// First verifies the payload byte length matches `activations.len() * 4`
/// (f32 alignment), then checks the computed SHA-256 digest against
/// `expected_digest` (when `Some`), and finally dispatches to `apply`.
pub fn apply_with_digest(
    application: &EngramApplication,
    activations: &mut [f32],
    payload: &[u8],
    expected_digest: Option<&str>,
) -> Result<(), String> {
    let expected_byte_len = activations
        .len()
        .checked_mul(4)
        .ok_or_else(|| "activation count overflow".to_string())?;
    if payload.len() != expected_byte_len {
        return Err(format!(
            "engram payload byte length {} does not match activation length {} * 4 = {}",
            payload.len(),
            activations.len(),
            expected_byte_len,
        ));
    }
    if let Some(digest_hex) = expected_digest {
        let mut hasher = Sha256::new();
        hasher.update(payload);
        let computed: String = hasher
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        if computed != digest_hex {
            return Err(format!(
                "engram payload digest mismatch: computed {computed}, expected {digest_hex}"
            ));
        }
    }
    apply(application, activations, payload)
}

/// Apply an engram payload to activations on the CPU.
///
/// The `payload` byte slice is interpreted according to the `application` mode:
/// - `AdditiveResidual` — payload is `f32` residuals added element-wise.
/// - `MultiplicativeModulation` — payload is `f32` scales multiplied
///   element-wise.
/// - `LowRankProjection`, `LatentPrefix`, `AdapterActivation` —
///   return an error (removed from production admission).
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
        EngramApplication::LowRankProjection
        | EngramApplication::LatentPrefix
        | EngramApplication::AdapterActivation => {
            return Err(format!(
                "{:?} is not yet implemented — removed from production admission",
                application
            ))
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
                "{:?} is not yet implemented — removed from production admission",
                application
            ))
        }
    };

    // Compile the kernel
    let compiler = MetalBackendCompiler::new();
    let artifact = compiler
        .compile_source(
            &format!("engram_{}", kernel_name),
            src,
            kernel_name,
            &format!("prism.engram.{}.v1", kernel_name),
            KernelAbi {
                version: 1,
                buffers: Vec::new(),
                constants: Vec::new(),
                threadgroup_memory: Vec::new(),
                dispatch_geometry: DispatchGeometryPolicy::FromOutputBuffer,
                threads_per_threadgroup: (1, 1, 1),
            },
        )
        .map_err(|e| format!("Metal backend compile error: {e:?}"))?;
    let lib = device
        .new_library_with_data(&artifact.compiled_bytes)
        .map_err(|e| format!("Metal library load error: {e:?}"))?;

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

    
    
    use sha2::{Digest, Sha256};

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
    #[test]
    fn apply_rejects_unimplemented_modes() {
        let mut activations = vec![1.0, 2.0];
        let p = payload(&[0.5, -1.0]);
        // LowRankProjection
        assert!(apply(&EngramApplication::LowRankProjection, &mut activations, &p).is_err());
        // LatentPrefix
        assert!(apply(&EngramApplication::LatentPrefix, &mut activations, &p).is_err());
        // AdapterActivation
        assert!(apply(&EngramApplication::AdapterActivation, &mut activations, &p).is_err());
    }

    #[test]
    fn apply_with_digest_rejects_wrong_payload_length() {
        let mut activations = vec![1.0, 2.0];
        // Payload has 3 f32 values (12 bytes) but activations has 2 (needs 8 bytes)
        let p = payload(&[1.0, 2.0, 3.0]);
        let result = apply_with_digest(
            &EngramApplication::AdditiveResidual,
            &mut activations,
            &p,
            None,
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("does not match"));
    }

    #[test]
    fn apply_with_digest_verifies_sha256() {
        let mut activations = vec![1.0, 2.0];
        let p = payload(&[0.25, -0.5]);
        let mut hasher = Sha256::new();
        hasher.update(&p);
        let expected_digest: String = hasher
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        // Correct digest should pass
        apply_with_digest(
            &EngramApplication::AdditiveResidual,
            &mut activations,
            &p,
            Some(&expected_digest),
        )
        .unwrap();
        assert_eq!(activations, vec![1.25, 1.5]);
    }

    #[test]
    fn apply_with_digest_rejects_wrong_digest() {
        let mut activations = vec![1.0, 2.0];
        let p = payload(&[0.25, -0.5]);
        let result = apply_with_digest(
            &EngramApplication::AdditiveResidual,
            &mut activations,
            &p,
            Some("deadbeef"),
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("digest mismatch"));
    }

    #[test]
    fn apply_additive_residual_cpu_produces_correct_result() {
        let mut activations = vec![1.0, 2.0, 3.0];
        apply_cpu(
            &EngramApplication::AdditiveResidual,
            &mut activations,
            &payload(&[0.5, -1.0, 0.0]),
        )
        .unwrap();
        assert_eq!(activations, vec![1.5, 1.0, 3.0]);
    }

    #[test]
    fn apply_multiplicative_modulation_cpu_produces_correct_result() {
        let mut activations = vec![1.0, 2.0, 3.0];
        apply_cpu(
            &EngramApplication::MultiplicativeModulation,
            &mut activations,
            &payload(&[2.0, 0.5, 1.0]),
        )
        .unwrap();
        assert_eq!(activations, vec![2.0, 1.0, 3.0]);
    }

    /// Test that Metal and CPU produce matching results for additive/multiplicative.
    ///
    /// When `metal-dispatch` is enabled, this verifies the GPU path is the
    /// production-authoritative path and matches the CPU reference.
    /// When `metal-dispatch` is not available, only the CPU path is tested.
    #[test]
    fn apply_metal_cpu_match_additive_residual() {
        let input = [1.0, 2.0, 3.0, 5.0, 8.0];
        let residual = [0.5, -0.25, 0.0, -2.0, 0.125];
        let p = payload(&residual);
        let mut cpu_result = input;
        apply_cpu(&EngramApplication::AdditiveResidual, &mut cpu_result, &p).unwrap();

        #[cfg(feature = "metal-dispatch")]
        {
            let mut metal_result = input;
            apply_metal(&EngramApplication::AdditiveResidual, &mut metal_result, &p).unwrap();
            let max_diff = cpu_result
                .iter()
                .zip(&metal_result)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_diff < 1e-6,
                "Metal and CPU additive residual differ by {max_diff}"
            );
        }

        #[cfg(not(feature = "metal-dispatch"))]
        {
            let _ = input; // suppress unused warning in non-metal build
                           // Only CPU is tested when Metal is unavailable
        }
    }

    #[test]
    fn apply_metal_cpu_match_multiplicative_modulation() {
        let input = [1.0, 2.0, 3.0, 5.0, 8.0];
        let scales = [2.0, 0.5, 1.0, 0.0, 0.125];
        let p = payload(&scales);
        let mut cpu_result = input;
        apply_cpu(
            &EngramApplication::MultiplicativeModulation,
            &mut cpu_result,
            &p,
        )
        .unwrap();

        #[cfg(feature = "metal-dispatch")]
        {
            let mut metal_result = input;
            apply_metal(
                &EngramApplication::MultiplicativeModulation,
                &mut metal_result,
                &p,
            )
            .unwrap();
            let max_diff = cpu_result
                .iter()
                .zip(&metal_result)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_diff < 1e-6,
                "Metal and CPU multiplicative modulation differ by {max_diff}"
            );
        }

        #[cfg(not(feature = "metal-dispatch"))]
        {
            let _ = input;
        }
    }

    /// Test the top-level `apply` dispatcher routes additive to CPU (or Metal).
    #[test]
    fn apply_dispatcher_additive_residual() {
        let mut activations = vec![1.0, 2.0];
        apply(
            &EngramApplication::AdditiveResidual,
            &mut activations,
            &payload(&[0.5, -1.0]),
        )
        .unwrap();
        assert_eq!(activations, vec![1.5, 1.0]);
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
