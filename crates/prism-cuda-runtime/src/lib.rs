//! Prism CUDA runtime — PTX compilation and kernel dispatch for NVIDIA GPUs.
//!
//! This crate provides two capabilities:
//!
//! 1. **Compilation** — calls the `ptxas` assembler (NVIDIA CUDA Toolkit) to compile
//!    PTX source strings into cubin (CUDA binary) blobs.
//! 2. **Dispatch** — loads a cubin via the CUDA driver API, launches a kernel with
//!    configurable grid/block dimensions, synchronizes, and returns timing evidence.
//!
//! Both paths are feature-gated behind `cuda-runtime`, which itself is `cfg`-gated
//! to Linux (via the `#[cfg(target_os = "linux")]` attribute). On other platforms or
//! when the feature is disabled all operations return a clear error string.

pub mod compiler;
pub mod dispatch;

/// Target hardware format — mirrors `prism_ecs_ir::backend_dispatch::HalFormat`
/// locally to avoid a circular dependency through the workspace runtime crates.
///
/// This crate only supports `Ptx`. All other variants return an error at compile
/// time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HalFormat {
    /// NVIDIA PTX (parallel thread execution).
    Ptx,
}

/// A compiled CUDA kernel, ready for dispatch.
#[derive(Debug, Clone)]
pub struct CudaBinary {
    /// The raw cubin bytes (device code).
    pub cubin: Vec<u8>,
    /// Kernel entry point name (must match a `__global__` function in the PTX).
    pub entry_point: String,
    /// Grid dimensions (blocks per grid).
    pub grid_dims: (u32, u32, u32),
    /// Block dimensions (threads per block).
    pub block_dims: (u32, u32, u32),
}

/// Wall-clock timing for a completed kernel dispatch.
#[derive(Debug, Clone, Copy)]
pub struct TimingEvidence {
    /// Kernel name (from the entry point).
    pub kernel_name: &'static str,
    /// GPU time in microseconds (measured via CUDA events,
    /// or `0` when no GPU was needed for the measurement path).
    pub duration_us: u64,
}

/// Compile a source string for the given `HalFormat` into a `CudaBinary`.
///
/// Currently only `HalFormat::Ptx` is supported. Other formats return an error.
///
/// # Errors
///
/// - Returns an error if `ptxas` is not found on `$PATH`.
/// - Returns an error if `ptxas` fails to assemble the PTX source.
/// - Returns an error for unsupported `HalFormat` values.
pub fn compile(source: &str, format: HalFormat) -> Result<CudaBinary, String> {
    match format {
        HalFormat::Ptx => compiler::compile_ptx(source),
    }
}

/// Dispatch a compiled `CudaBinary` on the current CUDA device.
///
/// Loads the cubin via the CUDA driver API (`cuModuleLoadData`), looks up the
/// entry point, launches the kernel at the binary's grid/block dimensions, and
/// synchronizes. Returns wall-clock timing via CUDA events.
///
/// # Errors
///
/// - Returns an error when the CUDA driver API is unavailable (not on Linux,
///   no NVIDIA driver, or the `cuda-runtime` feature is disabled).
/// - Returns an error when module loading, function lookup, or kernel launch fails.
pub fn dispatch(binary: &CudaBinary) -> Result<TimingEvidence, String> {
    dispatch::dispatch_kernel(binary)
}
