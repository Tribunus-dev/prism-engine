//! AMD XDNA NPU compiler interface.
//!
//! Provides the public API for compiling an IR operation tree into an AMD XDNA
//! NPU executable. Wraps the codegen module with additional metadata needed for
//! dispatch on NPU hardware (amdxdna kernel driver model).

use prism_ecs_core::{Entity, World};
use prism_ecs_ir::backend_dispatch::{HalExecutable, HalFormat};

use crate::artifact::XdnaArtifact;
use crate::codegen::lower_matmul_to_native_xdna;
use crate::command::{XdnaCommandBuffer, XdnaFirmwareImageEncoder};
use prism_spatial_ir::xdna_manifest::XdnaModelManifest;
use prism_spatial_ir::XdnaTarget;

/// Compile an IR operation tree into an AMD XDNA NPU `HalExecutable`.
///
/// Wraps the codegen output with default dispatch parameters suitable for AMD
/// XDNA NPU execution: a single graph invocation with 1×1×1 grid/block
/// dimensions (the NPU uses a graph-of-tasks model rather than GPU grid dispatch).
pub fn compile_amd_npu(
    world: &World,
    root_op: Entity,
    format: HalFormat,
) -> Result<HalExecutable, String> {
    debug_assert_eq!(format, HalFormat::AmdNpu);

    let program = lower_with_available_target(world, root_op)
        .map_err(|e| format!("AMD XDNA NPU lowering failed: {:?}", e))?;
    let manifest = XdnaModelManifest::from_program(format!("ecs-op-{}", root_op.id()), &program);
    compile_program_with_manifest(format, program, manifest)
}

pub fn compile_amd_npu_with_manifest(
    world: &World,
    root_op: Entity,
    format: HalFormat,
    manifest: XdnaModelManifest,
) -> Result<HalExecutable, String> {
    debug_assert_eq!(format, HalFormat::AmdNpu);
    let program = lower_with_available_target(world, root_op)
        .map_err(|e| format!("AMD XDNA lowering failed: {:?}", e))?;
    compile_program_with_manifest(format, program, manifest)
}

/// Compile against an explicitly supplied XDNA topology. This is the
/// cross-compilation path used when the compiler runs on macOS, in a Docker
/// build container, or on a host without the target NPU device node.
pub fn compile_amd_npu_with_target(
    world: &World,
    root_op: Entity,
    format: HalFormat,
    target: &XdnaTarget,
) -> Result<HalExecutable, String> {
    debug_assert_eq!(format, HalFormat::AmdNpu);
    let program = crate::codegen::lower_matmul_to_native_xdna_with_target(world, root_op, target)
        .map_err(|error| format!("target-specific XDNA lowering failed: {error:?}"))?;
    let manifest = XdnaModelManifest::from_program(format!("ecs-op-{}", root_op.id()), &program);
    compile_program_with_manifest(format, program, manifest)
}

/// Cross-compile against an explicit target while preserving a caller-owned
/// model manifest (quantization, persistent weights, and KV-cache policy).
pub fn compile_amd_npu_with_target_and_manifest(
    world: &World,
    root_op: Entity,
    format: HalFormat,
    target: &XdnaTarget,
    manifest: XdnaModelManifest,
) -> Result<HalExecutable, String> {
    debug_assert_eq!(format, HalFormat::AmdNpu);
    let program = crate::codegen::lower_matmul_to_native_xdna_with_target(world, root_op, target)
        .map_err(|error| format!("target-specific XDNA lowering failed: {error:?}"))?;
    compile_program_with_manifest(format, program, manifest)
}

fn compile_program_with_manifest(
    format: HalFormat,
    program: prism_spatial_ir::xdna::XdnaProgram,
    manifest: XdnaModelManifest,
) -> Result<HalExecutable, String> {
    let command = XdnaCommandBuffer::from_program(&program)?;
    let firmware = XdnaFirmwareImageEncoder::encode_image(&command)?;
    let artifact = XdnaArtifact {
        program,
        manifest,
        overlay: Some(firmware.overlay),
        ctrlcode: Some(firmware.ctrlcode),
    };
    artifact
        .validate()
        .map_err(|e| format!("native XDNA artifact validation failed: {e}"))?;
    let encoded = artifact.encode()?;
    let source = format!("prism-xdna-bincode-v1:{}", hex_encode(&encoded));

    Ok(HalExecutable {
        format,
        source,
        entry_point: "matmul".into(),
        grid_dims: (1, 1, 1),
        block_dims: (1, 1, 1),
    })
}

fn lower_with_available_target(
    world: &World,
    root_op: Entity,
) -> Result<prism_spatial_ir::xdna::XdnaProgram, String> {
    #[cfg(target_os = "linux")]
    if let Ok(probe) = crate::linux::LinuxXdnaProbe::open() {
        if let Ok(target) = probe.target() {
            return crate::codegen::lower_matmul_to_native_xdna_with_target(
                world, root_op, &target,
            )
            .map_err(|error| format!("target-specific XDNA lowering failed: {error:?}"));
        }
    }
    lower_matmul_to_native_xdna(world, root_op)
        .map_err(|error| format!("default XDNA lowering failed: {error:?}"))
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0xf) as usize] as char);
    }
    output
}
