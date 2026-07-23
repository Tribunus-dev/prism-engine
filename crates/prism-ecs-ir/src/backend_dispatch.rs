//! Hardware Abstraction Layer — common output contract for all codegen backends.
//!
//! Defines the `HalFormat` enum, the `HalExecutable` result type, and the
//! top-level `dispatch_codegen` function that routes an IR operation tree to
//! the appropriate backend code generator based on the target format.

use prism_ecs_core::{Entity, World};

/// Target hardware format for code generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HalFormat {
    /// Apple Metal Shading Language.
    Metal,
    /// NVIDIA PTX (parallel thread execution).
    Ptx,
    /// AMD GCN (Graphics Core Next) assembly.
    AmdGcn,
    /// Portable C source.
    CSource,
    /// Apple Neural Engine MIL.
    Ane,
    /// Intel GPU (Arc, Iris Xe) via SPIR-V / IGC.
    IntelGpu,
    /// AMD XDNA NPU (Ryzen AI) graph pseudo-IR.
    AmdNpu,
    /// Intel NPU (Meteor Lake+) graph description.
    IntelNpu,
    /// Tenstorrent Wormhole/Grayskull via TT-Metalium.
    Tenstorrent,
}

/// A compiled executable unit produced by a codegen backend.
///
/// Contains the generated source code together with metadata needed to launch
/// it on the target platform: entry point name and dispatch dimensions.
#[derive(Debug, Clone)]
pub struct HalExecutable {
    /// Target format.
    pub format: HalFormat,
    /// Generated source code.
    pub source: String,
    /// Name of the entry-point function.
    pub entry_point: String,
    /// Grid / launch dimensions (x, y, z).
    pub grid_dims: (u32, u32, u32),
    /// Block / threadgroup dimensions (x, y, z).
    pub block_dims: (u32, u32, u32),
}

/// Dispatch an IR operation tree to the appropriate codegen backend.
///
/// Given a root operation entity and a target `HalFormat`, this function
/// lowers the operation to the corresponding backend and wraps the result
/// in a `HalExecutable` with sensible default dispatch parameters.
///
/// # Errors
///
/// Returns `Err(String)` when the chosen backend hasn't been implemented yet
/// or when the backend's own lowering function fails.
pub fn dispatch_codegen(
    world: &World,
    root_op: Entity,
    format: HalFormat,
) -> Result<HalExecutable, String> {
    match format {
        HalFormat::Metal => {
            let source = crate::backend_apple_gpu::lower_to_metal(world, root_op)
                .map_err(|e| format!("Metal codegen failed: {:?}", e))?;
            Ok(HalExecutable {
                format,
                source,
                entry_point: "matmul_kernel".into(),
                grid_dims: (1, 1, 1),
                block_dims: (16, 16, 1),
            })
        }
        HalFormat::CSource => {
            let source = crate::backend_cpu::lower_to_cpu(world, root_op)
                .map_err(|e| format!("CPU codegen failed: {:?}", e))?;
            Ok(HalExecutable {
                format,
                source,
                entry_point: "matmul".into(),
                grid_dims: (1, 1, 1),
                block_dims: (1, 1, 1),
            })
        }
        HalFormat::Ane => Err("ANE codegen moved to prism-ane-runtime crate — \
                 use prism_ane_runtime::codegen directly"
            .into()),
        HalFormat::IntelGpu => {
            let source = crate::backend_intel_gpu::lower_to_intel_gpu(world, root_op)
                .map_err(|e| format!("Intel GPU codegen failed: {:?}", e))?;
            Ok(HalExecutable {
                format,
                source,
                entry_point: "matmul".into(),
                grid_dims: (1, 1, 1),
                block_dims: (16, 16, 1),
            })
        }
        HalFormat::Ptx => Err("PTX codegen is not yet implemented".into()),
        HalFormat::AmdGcn => Err("AMD GCN codegen is not yet implemented".into()),
        HalFormat::AmdNpu => Err("AMD XDNA NPU codegen is not yet implemented".into()),
        HalFormat::IntelNpu => {
            // Intel NPU dispatch moved to prism-intel-npu-runtime crate.
            // Call `prism_intel_npu_runtime::dispatch_intel_npu(world, root_op, format)`
            // from the workspace-level dispatch layer.
            Err("Intel NPU runtime not wired (call prism_intel_npu_runtime directly)".into())
        }
        HalFormat::Tenstorrent => Err("Tenstorrent codegen is not yet implemented".into()),
    }
}

/// Dispatch with the native AMD XDNA backend supplied by the workspace-level
/// runtime. `prism-ecs-ir` deliberately does not depend on that runtime (the
/// runtime already consumes this crate), so the integration point is an
/// explicit function pointer rather than a wrapper or a cyclic dependency.
pub fn dispatch_codegen_with_amd_npu(
    world: &World,
    root_op: Entity,
    format: HalFormat,
    amd_npu_codegen: impl FnOnce(&World, Entity) -> Result<HalExecutable, String>,
) -> Result<HalExecutable, String> {
    if format == HalFormat::AmdNpu {
        amd_npu_codegen(world, root_op)
    } else {
        dispatch_codegen(world, root_op, format)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir_types::{FloatKind, TensorType, Type};
    use crate::op::{OpMarker, OpName, Operands, Results};
    use crate::value::{Uses, ValueType};
    use prism_ecs_core::{EntityKind, World};

    // ── helpers ───────────────────────────────────────────────────────────

    fn create_value(world: &mut World, name: &str, ty: Type) -> Entity {
        let e: Entity = world
            .spawn(EntityKind::Node, Some(name.into()))
            .unwrap()
            .into();
        world.add_component(e, ValueType(ty)).unwrap();
        world.add_component(e, Uses(vec![])).unwrap();
        e
    }

    fn create_matmul_op(world: &mut World, a_ty: Type, b_ty: Type, c_ty: Type) -> Entity {
        let a = create_value(world, "A", a_ty.clone());
        let b = create_value(world, "B", b_ty.clone());
        let c = create_value(world, "C", c_ty.clone());
        let result = create_value(world, "result", c_ty);

        let op: Entity = world
            .spawn(EntityKind::Node, Some("matmul".into()))
            .unwrap()
            .into();
        world.add_component(op, OpMarker).unwrap();
        world
            .add_component(op, OpName("linalg.matmul".into()))
            .unwrap();
        world.add_component(op, Operands(vec![a, b, c])).unwrap();
        world.add_component(op, Results(vec![result])).unwrap();
        op
    }

    // ── Metal dispatch ────────────────────────────────────────────────────

    #[test]
    fn dispatch_metal_produces_valid_metal_source() {
        let mut world = World::new();

        let f32 = Type::float(FloatKind::F32);
        let a_ty = Type::Tensor(TensorType::new(vec![2, 3], f32.clone()));
        let b_ty = Type::Tensor(TensorType::new(vec![3, 4], f32.clone()));
        let c_ty = Type::Tensor(TensorType::new(vec![2, 4], f32));

        let op = create_matmul_op(&mut world, a_ty, b_ty, c_ty);
        let exec = dispatch_codegen(&world, op, HalFormat::Metal).expect("Metal dispatch failed");

        assert_eq!(exec.format, HalFormat::Metal);
        assert!(
            exec.source.contains("kernel void"),
            "expected Metal kernel source, got:\n{}",
            exec.source
        );
        assert!(
            exec.source.contains("device float*"),
            "expected device float pointers in:\n{}",
            exec.source
        );
        assert_eq!(exec.entry_point, "matmul_kernel");
        assert_eq!(exec.grid_dims, (1, 1, 1));
        assert_eq!(exec.block_dims, (16, 16, 1));
    }

    // ── CPU dispatch ──────────────────────────────────────────────────────

    #[test]
    fn dispatch_cpu_produces_c_source() {
        let mut world = World::new();

        let f32 = Type::float(FloatKind::F32);
        let a_ty = Type::Tensor(TensorType::new(vec![2, 3], f32.clone()));
        let b_ty = Type::Tensor(TensorType::new(vec![3, 4], f32.clone()));
        let c_ty = Type::Tensor(TensorType::new(vec![2, 4], f32));

        let op = create_matmul_op(&mut world, a_ty, b_ty, c_ty);
        let exec = dispatch_codegen(&world, op, HalFormat::CSource).expect("C dispatch failed");

        assert_eq!(exec.format, HalFormat::CSource);
        assert!(
            exec.source.contains("void"),
            "expected C void function in:\n{}",
            exec.source
        );
        assert!(
            exec.source.contains("for ("),
            "expected C for-loop in:\n{}",
            exec.source
        );
        assert_eq!(exec.entry_point, "matmul");
    }

    // ── ANE dispatch ──────────────────────────────────────────────────────

    #[test]
    fn dispatch_ane_moved_to_ane_runtime_crate() {
        let mut world = World::new();
        let f16 = Type::float(FloatKind::F16);
        let a_ty = Type::Tensor(TensorType::new(vec![2, 3], f16.clone()));
        let b_ty = Type::Tensor(TensorType::new(vec![3, 4], f16.clone()));
        let c_ty = Type::Tensor(TensorType::new(vec![2, 4], f16));
        let op = create_matmul_op(&mut world, a_ty, b_ty, c_ty);
        let err =
            dispatch_codegen(&world, op, HalFormat::Ane).expect_err("ANE dispatch should error");
        assert!(
            err.contains("prism-ane-runtime"),
            "expected error mentioning prism-ane-runtime, got: {err}"
        );
    }

    // ── AMD XDNA NPU dispatch ─────────────────────────────────────────────

    #[test]
    fn dispatch_amd_npu_returns_error() {
        let mut world = World::new();
        let op: Entity = world
            .spawn(EntityKind::Node, Some("dummy".into()))
            .unwrap()
            .into();
        world.add_component(op, OpMarker).unwrap();
        world
            .add_component(op, OpName("linalg.matmul".into()))
            .unwrap();

        let err = dispatch_codegen(&world, op, HalFormat::AmdNpu).unwrap_err();
        assert!(
            err.contains("not yet implemented"),
            "expected 'not yet implemented', got: {err}"
        );
    }

    #[test]
    fn dispatch_amd_npu_accepts_native_runtime_callback() {
        let mut world = World::new();
        let op: Entity = world
            .spawn(EntityKind::Node, Some("dummy".into()))
            .unwrap()
            .into();
        let exec = dispatch_codegen_with_amd_npu(&world, op, HalFormat::AmdNpu, |_world, _root| {
            Ok(HalExecutable {
                format: HalFormat::AmdNpu,
                source: "native-xdna".into(),
                entry_point: "matmul".into(),
                grid_dims: (1, 1, 1),
                block_dims: (1, 1, 1),
            })
        })
        .expect("native callback should be used");
        assert_eq!(exec.source, "native-xdna");
    }

    // ── Intel NPU dispatch ────────────────────────────────────────────────

    #[test]
    fn dispatch_ptx_returns_error() {
        let mut world = World::new();
        let op: Entity = world
            .spawn(EntityKind::Node, Some("dummy".into()))
            .unwrap()
            .into();
        world.add_component(op, OpMarker).unwrap();
        world
            .add_component(op, OpName("linalg.matmul".into()))
            .unwrap();

        let err = dispatch_codegen(&world, op, HalFormat::Ptx).unwrap_err();
        assert!(
            err.contains("not yet implemented"),
            "expected 'not yet implemented', got: {err}"
        );
    }

    #[test]
    fn dispatch_amdgcn_returns_error() {
        let mut world = World::new();
        let op: Entity = world
            .spawn(EntityKind::Node, Some("dummy".into()))
            .unwrap()
            .into();
        world.add_component(op, OpMarker).unwrap();
        world
            .add_component(op, OpName("linalg.matmul".into()))
            .unwrap();

        let err = dispatch_codegen(&world, op, HalFormat::AmdGcn).unwrap_err();
        assert!(
            err.contains("not yet implemented"),
            "expected 'not yet implemented', got: {err}"
        );
    }
}
