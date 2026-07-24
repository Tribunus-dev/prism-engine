//! ANE dispatch — loads a compiled ANE model and runs inference.
//!
//! On macOS (with the ANE available) this module loads a compiled model via the
//! CoreML runtime, performs synchronous inference, and returns wall-clock timing.
//! Without the runtime or on other platforms, all operations return an error.

use crate::compiler::AneBinary;

/// Wall-clock timing evidence for a completed ANE inference.
#[derive(Debug, Clone, Copy)]
pub struct TimingEvidence {
    /// Model entry point name.
    pub kernel_name: &'static str,
    /// Inference duration in microseconds.
    pub duration_us: u64,
}

/// The shape of an ANE model's buffer — dimension vector + element type.
#[derive(Debug, Clone)]
pub struct TensorDescriptor {
    /// Dimension sizes (e.g. `[M, K]` for a weight matrix).
    pub shape: Vec<u64>,
    /// Scalar type name as used in MIL (e.g. `"float16"`, `"float32"`).
    pub dtype: &'static str,
}

/// Dispatch a compiled [`AneBinary`] on the ANE.
///
/// Loads the compiled model, binds input/output tensors, runs synchronously,
/// and returns timing evidence.
///
/// # Errors
///
/// - Returns an error when the ANE runtime is unavailable (no macOS, no ANE).
/// - Returns an error when model loading, binding, or execution fails.
pub fn dispatch(
    binary: &AneBinary,
    inputs: &[(&str, &[u8], TensorDescriptor)],
    outputs: &mut [(&str, &mut [u8], TensorDescriptor)],
) -> Result<TimingEvidence, String> {
    #[cfg(all(target_os = "macos", feature = "coreml"))]
    {
        let root = tempfile::tempdir().map_err(|e| format!("ANE tempdir: {e}"))?;
        let model_dir = root.path().join("model.mlmodelc");
        std::fs::create_dir_all(&model_dir).map_err(|e| format!("ANE model directory: {e}"))?;
        prism_ane::unpack_mlmodelc(&binary.binary, &model_dir)?;
        let model = prism_ane::coreml_bridge::CoreMlModel::load(&model_dir)?;
        let Some((input_name, input_bytes, input_desc)) = inputs.first() else {
            return Err("ANE dispatch requires one input".into());
        };
        let Some((output_name, output_bytes, output_desc)) = outputs.first_mut() else {
            return Err("ANE dispatch requires one output".into());
        };
        let input_arena = prism_ane::arena_info::ArenaInfo {
            width: 0,
            height: 0,
            logical_dim0: input_desc.shape.first().copied().unwrap_or(0) as i32,
            logical_dim1: input_desc.shape.get(1).copied().unwrap_or(0) as i32,
            pixel_format: 0,
            byte_size: input_bytes.len() as i32,
            bytes_per_row: 0,
            base_address: input_bytes.as_ptr() as *mut _,
            cv_buffer: std::ptr::null_mut(),
            io_surface: std::ptr::null_mut(),
        };
        let mut output_arena = prism_ane::arena_info::ArenaInfo {
            width: 0,
            height: 0,
            logical_dim0: output_desc.shape.first().copied().unwrap_or(0) as i32,
            logical_dim1: output_desc.shape.get(1).copied().unwrap_or(0) as i32,
            pixel_format: 0,
            byte_size: output_bytes.len() as i32,
            bytes_per_row: 0,
            base_address: output_bytes.as_mut_ptr() as *mut _,
            cv_buffer: std::ptr::null_mut(),
            io_surface: std::ptr::null_mut(),
        };
        let started = std::time::Instant::now();
        if let Some((input_name_b, input_bytes_b, input_desc_b)) = inputs.get(1) {
            let input_arena_b = prism_ane::arena_info::ArenaInfo {
                width: 0,
                height: 0,
                logical_dim0: input_desc_b.shape.first().copied().unwrap_or(0) as i32,
                logical_dim1: input_desc_b.shape.get(1).copied().unwrap_or(0) as i32,
                pixel_format: 0,
                byte_size: input_bytes_b.len() as i32,
                bytes_per_row: 0,
                base_address: input_bytes_b.as_ptr() as *mut _,
                cv_buffer: std::ptr::null_mut(),
                io_surface: std::ptr::null_mut(),
            };
            model.predict_two(
                input_name,
                &input_arena,
                input_name_b,
                &input_arena_b,
                output_name,
                &mut output_arena,
            )?;
        } else {
            model.predict(input_name, &input_arena, output_name, &mut output_arena)?;
        }
        return Ok(TimingEvidence {
            kernel_name: Box::leak(binary.entry_point.clone().into_boxed_str()),
            duration_us: started.elapsed().as_micros() as u64,
        });
    }
    #[cfg(not(all(target_os = "macos", feature = "coreml")))]
    {
        let _ = (binary, inputs, outputs);
        Err("ANE dispatch requires macOS with the coreml feature".into())
    }
}

/// Probe whether the ANE runtime is available on the current platform.
///
/// Returns `true` on macOS with Apple Silicon (ANE hardware present).
/// Always returns `false` on other platforms.
#[cfg(target_os = "macos")]
pub fn is_ane_available() -> bool {
    // TODO: probe via IOSurface / MTLDevice registry ID for ANE presence.
    // For now assume Apple Silicon.
    true
}

#[cfg(all(test, feature = "coreml", target_os = "macos"))]
mod tests {
    use super::{dispatch, TensorDescriptor};
    #[test]
    fn dispatches_two_input_coreml_matmul() {
        let binary = crate::compile_mil("MIL PROGRAM matmul_2x3x1").expect("compile");
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [1.0f32, 1.0, 1.0];
        let mut output = [0u8; 8];
        let a_bytes = bytemuck::cast_slice(&a);
        let b_bytes = bytemuck::cast_slice(&b);
        let a_desc = TensorDescriptor {
            shape: vec![2, 3],
            dtype: "float32",
        };
        let b_desc = TensorDescriptor {
            shape: vec![3, 1],
            dtype: "float32",
        };
        let out_desc = TensorDescriptor {
            shape: vec![2, 1],
            dtype: "float32",
        };
        dispatch(
            &binary,
            &[("a", a_bytes, a_desc), ("b", b_bytes, b_desc)],
            &mut [("matmul_0", &mut output, out_desc)],
        )
        .expect("dispatch");
        let values: &[f32] = bytemuck::cast_slice(&output);
        assert_eq!(values, &[6.0, 15.0]);
    }
}

#[cfg(not(target_os = "macos"))]
pub fn is_ane_available() -> bool {
    false
}
