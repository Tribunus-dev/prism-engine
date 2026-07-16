//! AMD XDNA NPU compiler interface.
//!
//! Provides the public API for compiling an IR operation tree into an AMD XDNA
//! NPU executable. Wraps the codegen module with additional metadata needed for
//! dispatch on NPU hardware (amdxdna kernel driver model).

use prism_ecs_core::{Entity, World};
use prism_ecs_ir::backend_dispatch::{HalExecutable, HalFormat};

use crate::codegen::lower_to_amd_npu;

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

    let source = lower_to_amd_npu(world, root_op)
        .map_err(|e| format!("AMD XDNA NPU codegen failed: {:?}", e))?;

    Ok(HalExecutable {
        format,
        source,
        entry_point: "matmul".into(),
        grid_dims: (1, 1, 1),
        block_dims: (1, 1, 1),
    })
}
