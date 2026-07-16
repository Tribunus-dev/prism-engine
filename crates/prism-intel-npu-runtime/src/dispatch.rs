//! Intel NPU dispatch — top-level dispatch for the Intel NPU runtime.
//!
//! Routes an IR operation tree to the Intel NPU codegen backend and produces
//! a `HalExecutable` suitable for the Intel NPU (Meteor Lake+) Level Zero
//! driver model.

use prism_ecs_core::{Entity, World};
use prism_ecs_ir::backend_dispatch::{HalExecutable, HalFormat};

use crate::compiler::compile_intel_npu;

/// Route an IR operation to the Intel NPU codegen backend.
///
/// This is the public entry point for the Intel NPU runtime. It lowers the
/// operation tree rooted at `root_op` to an Intel NPU graph description and
/// wraps the result as a `HalExecutable`.
pub fn dispatch_intel_npu(
    world: &World,
    root_op: Entity,
    format: HalFormat,
) -> Result<HalExecutable, String> {
    compile_intel_npu(world, root_op, format)
}
