//! Prism AMD XDNA NPU runtime — codegen, compilation, and dispatch
//! for AMD Ryzen AI NPUs (XDNA architecture with AIE2/AIE2P engines).
//!
//! Follows the `prism-metal-runtime` pattern: compile source → binary →
//! dispatch → evidence.

pub mod artifact;
pub mod codec;
pub mod codegen;
pub mod command;
pub mod compiler;
pub mod dispatch;
#[cfg(target_os = "linux")]
pub mod linux;
pub mod route;
pub mod runtime;

pub use artifact::XdnaArtifact;
pub use codegen::{
    lower_attention_to_native_xdna, lower_attention_to_native_xdna_with_target,
    lower_matmul_to_amd_npu, lower_matmul_to_native_xdna, lower_matmul_to_native_xdna_with_target,
    lower_operation_to_native_xdna, lower_to_amd_npu, lower_unary_to_native_xdna,
    lower_unary_to_native_xdna_with_target, AmdNpuLowerError,
};
pub use command::{
    XdnaCommandBuffer, XdnaFirmwareEncoder, XdnaFirmwareImage, XdnaFirmwareImageEncoder,
};
pub use compiler::{
    compile_amd_npu, compile_amd_npu_with_manifest, compile_amd_npu_with_target,
    compile_amd_npu_with_target_and_manifest,
};
pub use dispatch::dispatch_amd_npu;
#[cfg(target_os = "linux")]
pub use linux::LinuxXdnaProbe;
pub use route::XdnaRouteExecutor;
pub use runtime::{
    detect_xdna_availability, TransportError, TransportXdnaDevice, XdnaAvailability,
    XdnaCommandSubmitter, XdnaDevice, XdnaExecutionPhase, XdnaRuntime, XdnaTransport,
};
