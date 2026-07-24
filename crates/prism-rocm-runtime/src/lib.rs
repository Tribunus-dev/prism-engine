//! Prism ROCm runtime — AMDGCN compilation and kernel dispatch for AMD GPUs.
//!
//! This crate provides two capabilities:
//!
//! 1. **Compilation** — calls the `amdllvm` assembler (ROCm LLVM fork) to
//!    compile AMDGCN assembly source strings into HSACO (HSA Code Object)
//!    blobs.
//! 2. **Dispatch** — loads an HSACO via the HSA runtime, launches a kernel
//!    with configurable grid/block dimensions, synchronizes, and returns
//!    timing evidence.
//!
//! Both paths are feature-gated behind `rocm-runtime`, which itself is
//! `cfg`-gated to Linux. On other platforms or when the feature is disabled,
//! all operations return a clear error string.

pub mod calibration;
pub mod compiler;
pub mod dispatch;
pub mod target;
pub mod ternary;

use prism_ecs_ir::backend_dispatch::HalFormat;

/// A compiled AMD GPU kernel, ready for dispatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AmdBinary {
    /// The raw HSACO bytes (device code).
    pub code_object: Vec<u8>,
    /// Kernel entry point name (must match a `.entry` directive in the AMDGCN
    /// assembly).
    pub entry_point: String,
    /// Grid dimensions (work-groups per grid).
    pub grid_dims: (u32, u32, u32),
    /// Block dimensions (work-items per work-group).
    pub block_dims: (u32, u32, u32),
}

/// Host-side buffer binding for a ROCm launch. Device allocation and
/// host/device copies are owned by the HSA implementation; this contract
/// validates the launch before any device state is touched.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AmdBufferBinding {
    pub name: String,
    pub bytes: Vec<u8>,
    pub writable: bool,
}

/// Buffer-aware MI300X launch contract shared by the future `KernelBackend`
/// adapter and the native HSA dispatcher.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AmdLaunchRequest {
    pub binary: AmdBinary,
    pub bindings: Vec<AmdBufferBinding>,
}

impl AmdLaunchRequest {
    pub fn validate(&self) -> Result<(), String> {
        if self.binary.code_object.is_empty() {
            return Err("ROCm launch code object is empty".into());
        }
        if self.binary.entry_point.is_empty() {
            return Err("ROCm launch entry point is empty".into());
        }
        let dimensions = [
            self.binary.grid_dims.0,
            self.binary.grid_dims.1,
            self.binary.grid_dims.2,
            self.binary.block_dims.0,
            self.binary.block_dims.1,
            self.binary.block_dims.2,
        ];
        if dimensions.contains(&0) {
            return Err("ROCm launch dimensions must be nonzero".into());
        }
        let mut names = std::collections::HashSet::new();
        for binding in &self.bindings {
            if binding.name.is_empty() || !names.insert(&binding.name) {
                return Err("ROCm launch bindings must have unique nonempty names".into());
            }
            if binding.bytes.is_empty() {
                return Err(format!(
                    "ROCm launch binding '{}' has an empty buffer",
                    binding.name
                ));
            }
        }
        Ok(())
    }
}

/// Wall-clock timing for a completed kernel dispatch.
#[derive(Debug, Clone, Copy)]
pub struct TimingEvidence {
    /// Kernel name (from the entry point).
    pub kernel_name: &'static str,
    /// GPU time in microseconds (measured via HSA signals,
    /// or `0` when no GPU was needed for the measurement path).
    pub duration_us: u64,
}

/// Compile a source string for the given `HalFormat` into an `AmdBinary`.
///
/// Currently only `HalFormat::AmdGcn` is supported. Other formats return an
/// error.
///
/// # Errors
///
/// - Returns an error if the ROCm toolkit (`amdllvm` / `hipcc`) is not found.
/// - Returns an error if the assembler fails to compile the source.
/// - Returns an error for unsupported `HalFormat` values.
pub fn compile(source: &str, format: HalFormat) -> Result<AmdBinary, String> {
    match format {
        HalFormat::AmdGcn => compiler::compile_amdgcn(source),
        other => Err(format!(
            "prism-rocm-runtime: unsupported format {other:?} — only AmdGcn is accepted"
        )),
    }
}

/// Dispatch a compiled `AmdBinary` on the current AMD GPU.
///
/// Loads the HSACO via the HSA runtime, looks up the entry point, launches the
/// kernel at the binary's grid/block dimensions, and synchronizes. Returns
/// wall-clock timing via HSA signal timestamps.
///
/// # Errors
///
/// - Returns an error when the HSA runtime is unavailable (not on Linux, no AMD
///   GPU driver, or the `rocm-runtime` feature is disabled).
/// - Returns an error when module loading, function lookup, or kernel launch
///   fails.
pub fn dispatch(binary: &AmdBinary) -> Result<TimingEvidence, String> {
    dispatch::dispatch_kernel(binary)
}
