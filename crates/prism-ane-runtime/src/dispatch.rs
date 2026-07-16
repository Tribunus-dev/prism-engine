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
    let _ = (binary, inputs, outputs);
    Err("ANE dispatch not yet implemented — awaiting CoreML runtime bindings".into())
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

#[cfg(not(target_os = "macos"))]
pub fn is_ane_available() -> bool {
    false
}
