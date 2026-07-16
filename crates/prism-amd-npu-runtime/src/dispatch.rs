//! AMD XDNA NPU dispatch — top-level dispatch for the AMD NPU runtime.
//!
//! Routes an IR operation tree to the AMD XDNA NPU codegen backend and produces
//! a `HalExecutable` suitable for the AMD NPU (XDNA AIE2/AIE2P) driver model.

use prism_ecs_core::{Entity, World};
use prism_ecs_ir::backend_dispatch::{HalExecutable, HalFormat};

use crate::compiler::compile_amd_npu;

/// Route an IR operation to the AMD XDNA NPU codegen backend.
///
/// This is the public entry point for the AMD XDNA NPU runtime. It lowers the
/// operation tree rooted at `root_op` to an XDNA graph description and wraps
/// the result as a `HalExecutable`.
pub fn dispatch_amd_npu(
    world: &World,
    root_op: Entity,
    format: HalFormat,
) -> Result<HalExecutable, String> {
    compile_amd_npu(world, root_op, format)
}
