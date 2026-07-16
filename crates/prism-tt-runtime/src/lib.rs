//! Prism Tenstorrent runtime — compile TT-Metalium kernels into RISCV-32 ELF
//! binaries, dispatch to Wormhole/Grayskull compute cores, and collect timing
//! evidence.
//!
//! This crate is the runtime counterpart to the `HalFormat::Tenstorrent` codegen
//! backend. It translates the textual source produced by codegen into executable
//! ELF files and orchestrates hardware dispatch on a Tenstorrent device mesh.
//!
//! # Architecture
//!
//! Tenstorrent accelerators (Wormhole, Grayskull) use a 2D mesh of RISCV-32
//! compute cores connected via a Network-on-Chip (NoC). There is no cache
//! coherence — data moves explicitly via programmed DMA between DRAM and each
//! core's local scratchpad (SRAM).
//!
//! The per-tensor evolution model maps naturally: each core runs one
//! (format, operation) kernel over the DRAM buffers connected to it via the NoC.
//!
//! # Platform Support
//!
//! TT-Metalium only supports Linux. On non-Linux hosts, `compile` and `dispatch`
//! return a graceful "not available" error. The `TTCore` component type is
//! available on all platforms since it is purely a data descriptor.
//!
//! ```rust,ignore (needs TT-Metalium)
//! use prism_tt_runtime::{compile, dispatch, TtBinary, DispatchConfig};
//!
//! let source = "// TT-Metalium kernel code ...".to_string();
//! let binary = compile(&source, HalFormat::Tenstorrent).expect("compilation failed");
//! let config = single_core_dispatch(binary, 0, 0, input_data, output_size);
//! let timing = dispatch(&binary, &config).expect("dispatch failed");
//! println!("Wall time: {} us", timing.wall_us);
//! ```

pub mod compiler;
pub mod dispatch;
pub mod tt_core;

use prism_ecs_ir::backend_dispatch::HalFormat;

pub use dispatch::DispatchConfig;
pub use dispatch::TimingEvidence;
pub use tt_core::{inter_card_bandwidth, EthernetLink, TTCore, TtHardwareConfig};

/// A compiled Tenstorrent kernel binary (RISCV-32 ELF).
#[derive(Debug, Clone)]
pub struct TtBinary {
    /// Kernel name (matches the entry-point function).
    pub kernel_name: String,
    /// Raw ELF binary bytes.
    pub data: Vec<u8>,
    /// Entry point symbol name.
    pub entry_point: String,
    /// Target architecture string (e.g. `"wormhole_b0"`, `"grayskull"`).
    pub architecture: String,
}

/// Compile TT-Metalium kernel source into a RISCV-32 ELF binary.
///
/// The `format` parameter is validated to be `HalFormat::Tenstorrent`.
///
/// # Errors
///
/// Returns `Err(String)` if:
/// - The target format is not `HalFormat::Tenstorrent`.
/// - TT-Metalium is not installed or configured.
/// - Compilation fails (syntax errors, linking errors).
///
/// On non-Linux platforms this always returns a graceful error explaining that
/// TT-Metalium requires Linux.
pub fn compile(source: &str, format: HalFormat) -> Result<TtBinary, String> {
    if format != HalFormat::Tenstorrent {
        return Err(format!("expected HalFormat::Tenstorrent, got {format:?}"));
    }

    #[cfg(target_os = "linux")]
    {
        let kernel_name = "tt_kernel";
        compiler::compile(source, kernel_name)
    }

    #[cfg(not(target_os = "linux"))]
    {
        let _ = source;
        Err("TT-Metalium requires Linux; cannot compile on this platform".into())
    }
}

/// Dispatch a compiled kernel to a Tenstorrent core mesh.
///
/// Sets up DRAM buffers, dispatches the compiled kernel to the configured
/// core range, synchronizes via ethernet links, and returns wall-clock and
/// cycle-level timing evidence.
///
/// # Errors
///
/// Returns `Err(String)` if:
/// - The binary's kernel name does not match the config's kernel name.
/// - The TT-Metalium runtime is unavailable.
/// - Buffer allocation or kernel dispatch fails.
///
/// On non-Linux platforms this always returns a graceful error.
pub fn dispatch(binary: &TtBinary, config: &DispatchConfig) -> Result<TimingEvidence, String> {
    if binary.kernel_name != config.binary.kernel_name {
        return Err(format!(
            "binary kernel '{}' does not match config kernel '{}'",
            binary.kernel_name, config.binary.kernel_name
        ));
    }

    #[cfg(target_os = "linux")]
    {
        dispatch::dispatch(config)
    }

    #[cfg(not(target_os = "linux"))]
    {
        let _ = (binary, config);
        Err("TT-Metalium requires Linux; cannot dispatch on this platform".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compile_wrong_format() {
        let result = compile("source", HalFormat::Metal);
        assert!(result.is_err());
        let msg = result.unwrap_err();
        assert!(msg.contains("expected HalFormat::Tenstorrent"));
    }

    #[test]
    fn binary_creation() {
        let bin = TtBinary {
            kernel_name: "test".into(),
            data: vec![0u8; 32],
            entry_point: "test".into(),
            architecture: "wormhole_b0".into(),
        };
        assert_eq!(bin.kernel_name, "test");
        assert_eq!(bin.data.len(), 32);
    }

    #[test]
    fn dispatch_binary_mismatch() {
        let bin = TtBinary {
            kernel_name: "kernel_a".into(),
            data: vec![],
            entry_point: "kernel_a".into(),
            architecture: "wormhole_b0".into(),
        };
        let cfg = DispatchConfig {
            binary: TtBinary {
                kernel_name: "kernel_b".into(),
                data: vec![],
                entry_point: "kernel_b".into(),
                architecture: "wormhole_b0".into(),
            },
            core_range: crate::dispatch::CoreRange::new((0, 0), (0, 0)),
            inputs: vec![],
            outputs: vec![],
            scratch_buffers: vec![],
            ethernet_links: vec![],
        };
        let result = dispatch(&bin, &cfg);
        assert!(result.is_err());
        let msg = result.unwrap_err();
        assert!(msg.contains("binary kernel"));
        assert!(msg.contains("does not match"));
    }

    #[test]
    fn compile_platform_gate() {
        // On macOS this returns the platform error; on Linux it tries real compilation.
        let result = compile("kernel source void () {}", HalFormat::Tenstorrent);
        match &result {
            Err(msg) => {
                // Acceptable: either platform gate or missing TT-Metalium.
                assert!(
                    msg.contains("Linux")
                        || msg.contains("TT-Metalium")
                        || msg.to_lowercase().contains("not found"),
                    "unexpected error: {msg}"
                );
            }
            Ok(_) => {
                // If we got here, TT-Metalium is installed and compilation succeeded.
            }
        }
    }

    #[test]
    fn dispatch_platform_gate() {
        let bin = TtBinary {
            kernel_name: "tt_kernel".into(),
            data: vec![],
            entry_point: "tt_kernel".into(),
            architecture: "wormhole_b0".into(),
        };
        let cfg = DispatchConfig {
            binary: TtBinary {
                kernel_name: "tt_kernel".into(),
                data: vec![],
                entry_point: "tt_kernel".into(),
                architecture: "wormhole_b0".into(),
            },
            core_range: crate::dispatch::CoreRange::new((0, 0), (0, 0)),
            inputs: vec![],
            outputs: vec![],
            scratch_buffers: vec![],
            ethernet_links: vec![],
        };
        let result = dispatch(&bin, &cfg);
        match &result {
            Err(msg) => {
                assert!(
                    msg.contains("Linux")
                        || msg.contains("TT-Metalium")
                        || msg.to_lowercase().contains("runtime"),
                    "unexpected error: {msg}"
                );
            }
            Ok(_) => {
                // System has TT-Metalium.
            }
        }
    }
}
