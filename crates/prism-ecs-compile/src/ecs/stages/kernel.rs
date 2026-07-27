//! Kernel generation stage.
//!
//! Lowers the legalized [`SpatialGraph`] into a per-backend
//! [`KernelArtifact`] collection. Each backend (CPU, Metal, AMD NPU, ANE)
//! has its own emit branch. The function as a whole is canonical pipeline
//! state — it does not own hardware handles; it calls into the per-backend
//! compile API ([`CpuBackend::compile`], [`MetalBackend::new().compile`],
//! [`prism_amd_npu_runtime::compile_amd_npu`], [`prism_ane_runtime::compile_mil`])
//! to produce the artifact bytes.

use sha2::Digest;

use prism_ecs_core::entity::Entity;
use prism_ecs_core::world::World;
use prism_ecs_kernel::{
    BackendKind, BindingSlot, CpuBackend, DispatchGeometry, KernelArtifact, KernelBackend,
    KernelCompileRequest, KernelDescriptor, KernelManifest, KernelPayload, KernelVariant,
    MetalBackend, FP16_GEMV_MSL,
};
use prism_spatial_ir::graph::SpatialNode;
use prism_spatial_ir::LoweringTarget;

use crate::ecs::components::{
    CompilationSession, KernelCollection, LegalizedPlan, SearchStateComponent, SessionStatus,
    SpatialGraphComponent,
};
use crate::ecs::orchestrator::{read_session_config, session_entity};
use crate::CompileError;

/// Run the **kernel generation** stage.
pub fn system_generate_kernels(world: &mut World) -> Result<(), CompileError> {
    let session = session_entity(world)?;
    let config = read_session_config(world, session)?;
    let graph = world
        .component::<SpatialGraphComponent>(session)
        .map_err(|e| CompileError::KernelGenFailed(e.to_string()))?
        .clone();
    let legalized = world
        .component::<LegalizedPlan>(session)
        .map_err(|e| CompileError::KernelGenFailed(format!("legalized plan missing: {e}")))?;
    if !legalized.is_valid {
        return Err(CompileError::KernelGenFailed(
            "cannot generate kernels from an invalid legalized plan".into(),
        ));
    }

    let format_plan = world
        .component::<SearchStateComponent>(session)
        .ok()
        .and_then(|state| state.format_plan.as_deref())
        .and_then(|json| serde_json::from_str(json).ok());

    let legalized = prism_spatial_ir::legalize::legalize(graph.graph.clone(), |_node| {
        Ok::<(), Vec<prism_spatial_ir::legalize::LegalizationError>>(())
    })
    .map_err(|errors| {
        CompileError::KernelGenFailed(format!("SpatialIR lowering failed: {errors:?}"))
    })?;
    let manifest = prism_spatial_ir::execution_plan::lower_to_manifest(
        legalized.graph(),
        prism_spatial_ir::cost::CostEstimate::zero(),
        format_plan.as_ref(),
    )
    .ok_or_else(|| CompileError::KernelGenFailed("cannot lower cyclic SpatialIR graph".into()))?;

    let lowered_manifests = vec![manifest.clone()];
    let backend_kind = config
        .target_backends
        .first()
        .copied()
        .unwrap_or(BackendKind::CPU);
    let artifacts = manifest
        .kernels
        .iter()
        .map(|descriptor| {
            let spatial_node = graph
                .graph
                .nodes()
                .iter()
                .find(|node| node.id() == descriptor.node_id);
            let supports_uop_lowering = matches!(
                spatial_node,
                Some(prism_spatial_ir::graph::SpatialNode::Compute {
                    kind: prism_spatial_ir::graph::ComputeKind::MatMul,
                    ..
                })
            ) || matches!(
                spatial_node,
                Some(prism_spatial_ir::graph::SpatialNode::Compute {
                    kind: prism_spatial_ir::graph::ComputeKind::Elementwise,
                    ..
                }) if graph
                    .graph
                    .get_annotations(descriptor.node_id)
                    .and_then(|meta| meta.elementwise_op.as_ref())
                    .is_some()
            ) || matches!(
                spatial_node,
                Some(prism_spatial_ir::graph::SpatialNode::Compute {
                    kind: prism_spatial_ir::graph::ComputeKind::Attention,
                    ..
                })
            ) || matches!(
                spatial_node,
                Some(prism_spatial_ir::graph::SpatialNode::Compute {
                    kind: prism_spatial_ir::graph::ComputeKind::Convolution
                        | prism_spatial_ir::graph::ComputeKind::Normalization
                        | prism_spatial_ir::graph::ComputeKind::Softmax
                        | prism_spatial_ir::graph::ComputeKind::RoPE
                        | prism_spatial_ir::graph::ComputeKind::Gather
                        | prism_spatial_ir::graph::ComputeKind::SSM,
                    ..
                })
            );
            if matches!(backend_kind, BackendKind::CPU | BackendKind::Metal)
                && supports_uop_lowering
            {
                let target = if backend_kind == BackendKind::Metal {
                    prism_spatial_ir::LoweringTarget::Metal
                } else {
                    prism_spatial_ir::LoweringTarget::Cpu
                };
                if let Ok((_, mut uop_artifacts)) = crate::compile_spatial_node_with_metadata(
                    spatial_node.unwrap(),
                    graph.graph.get_annotations(descriptor.node_id),
                    target,
                )
                {
                    let mut artifact = uop_artifacts.pop().ok_or_else(|| {
                        CompileError::KernelGenFailed("MatMul UOp lowering produced no artifact".into())
                    })?;
                    let stable_name = format!("spatial_node_{}", descriptor.node_id.0);
                    for payload in &mut artifact.payloads {
                        payload.descriptor.name = stable_name.clone();
                    }
                    for kernel in &mut artifact.manifest.kernels {
                        kernel.name = stable_name.clone();
                    }
                    return Ok(artifact);
                }
            }
            let (kernel_name, variant, source) = match backend_kind {
                BackendKind::CPU => (
                    format!("spatial_node_{}", descriptor.node_id.0),
                    KernelVariant::Custom("spatial-ir".into()),
                    serde_json::to_vec(descriptor)
                        .map_err(|e| CompileError::KernelGenFailed(e.to_string()))?,
                ),
                BackendKind::Metal => (
                    "fp16_gemv".into(),
                    KernelVariant::FP16GEMV,
                    FP16_GEMV_MSL.as_bytes().to_vec(),
                ),
                BackendKind::AmdNpu => {
                    return compile_spatial_matmul_to_native_xdna(
                        spatial_node.ok_or_else(|| {
                            CompileError::KernelGenFailed(format!(
                                "XDNA node {} is missing from SpatialGraph",
                                descriptor.node_id
                            ))
                        })?,
                        descriptor.node_id,
                    );
                }
                #[cfg(feature = "ane")]
                BackendKind::ANE => {
                    let node = graph
                        .graph
                        .nodes()
                        .iter()
                        .find(|node| node.id() == descriptor.node_id)
                        .ok_or_else(|| {
                            CompileError::KernelGenFailed(format!(
                                "ANE kernel node {} is missing from SpatialGraph",
                                descriptor.node_id
                            ))
                        })?;
                    let (m, k, n) = matmul_dimensions(node).ok_or_else(|| {
                        CompileError::KernelGenFailed(format!(
                            "ANE kernel node {} is not a statically shaped MatMul",
                            descriptor.node_id
                        ))
                    })?;
                    let mil = format!(
                        "MIL PROGRAM matmul_{m}x{k}x{n} {{\n  layer @0 = matmul(inputs: [A, B], output: C, M: {m}, K: {k}, N: {n}, type: float16)\n}}\n"
                    );
                    let binary = prism_ane_runtime::compile_mil(&mil).map_err(|error| {
                        CompileError::KernelGenFailed(format!(
                            "ANE Core ML compilation for node {} failed: {error}",
                            descriptor.node_id
                        ))
                    })?;
                    let digest = hex::encode(sha2::Sha256::digest(&binary.binary));
                    let ane_descriptor = KernelDescriptor {
                        name: format!("ane_matmul_{}", descriptor.node_id.0),
                        variant: KernelVariant::Custom("ane-coreml-matmul".into()),
                        backend: BackendKind::ANE,
                        source_digest: hex::encode(sha2::Sha256::digest(mil.as_bytes())),
                        binary_digest: digest.clone(),
                        binding_signature: Vec::new(),
                        dispatch_geometry: DispatchGeometry {
                            threads_per_threadgroup: [descriptor.threadgroup_size, 1, 1],
                            threadgroups_per_grid: [1, 1, 1],
                            threads_per_grid: [descriptor.threadgroup_size, 1, 1],
                        },
                    };
                    return Ok(KernelArtifact {
                        payloads: vec![KernelPayload {
                            binary: binary.binary,
                            descriptor: ane_descriptor.clone(),
                        }],
                        manifest: KernelManifest {
                            kernels: vec![ane_descriptor],
                            fusion_plan: None,
                            manifest_digest: digest.clone(),
                        },
                        artifact_digest: digest,
                    });
                }
                unsupported => {
                    return Err(CompileError::KernelGenFailed(format!(
                        "backend {unsupported:?} is not implemented"
                    )))
                }
            };
            let request = KernelCompileRequest {
                source,
                descriptor: KernelDescriptor {
                    name: kernel_name,
                    variant,
                    backend: backend_kind,
                    source_digest: String::new(),
                    binary_digest: String::new(),
                    binding_signature: Vec::<BindingSlot>::new(),
                    dispatch_geometry: DispatchGeometry {
                        threads_per_threadgroup: [descriptor.threadgroup_size, 1, 1],
                        threadgroups_per_grid: [1, 1, 1],
                        threads_per_grid: [descriptor.threadgroup_size, 1, 1],
                    },
                },
                source_path: None,
            };
            match backend_kind {
                BackendKind::CPU => CpuBackend.compile(&request),
                BackendKind::Metal => MetalBackend::new().compile(&request),
                BackendKind::AmdNpu => unreachable!("native XDNA artifacts return above"),
                _ => unreachable!(),
            }
            .map_err(|e| CompileError::KernelGenFailed(e.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let kernel_count = artifacts
        .iter()
        .map(|artifact| artifact.payloads.len())
        .sum();
    let uop_target = match backend_kind {
        BackendKind::Metal => LoweringTarget::Metal,
        _ => LoweringTarget::Cpu,
    };
    let uop_capture_result = crate::compile_spatial_graph(&graph.graph, uop_target);
    let uop_capture = uop_capture_result
        .as_ref()
        .ok()
        .map(|(capture, _)| capture.clone());
    let strategies = [
        prism_spatial_ir::FusionStrategy::StandardFused,
        prism_spatial_ir::FusionStrategy::InterleavedFused { stages: vec![] },
        prism_spatial_ir::FusionStrategy::PerOperation,
        prism_spatial_ir::FusionStrategy::PersistentMegakernel {
            search_generation: world
                .component::<SearchStateComponent>(session)
                .map(|state| state.generations_completed as u32)
                .unwrap_or(0),
        },
    ];
    let uop_strategy_candidates = if uop_capture_result.is_ok() {
        crate::compile_spatial_graph_strategies(&graph.graph, uop_target, &strategies).map_err(
            |error| {
                CompileError::KernelGenFailed(format!("UOp strategy compilation failed: {error}"))
            },
        )?
    } else {
        Vec::new()
    };
    let uop_strategy_captures = uop_strategy_candidates
        .iter()
        .map(|(strategy, capture, _)| (strategy.stable_id().to_string(), capture.clone()))
        .collect();
    let uop_tuning_receipt = if uop_strategy_candidates.is_empty() {
        Some(
            crate::uop::UOpTuningReceipt::explicit_fallback(
                graph.graph_digest.clone(),
                uop_target,
                "no executable UOp strategy candidates were available for reference measurement",
            )
            .map_err(CompileError::KernelGenFailed)?,
        )
    } else {
        let scenario = prism_spatial_ir::WorkloadScenario {
            realtime: false,
            batch_size: 1,
            sequence_length: 1,
        };
        match crate::benchmark_uop_strategy_candidates(&uop_strategy_candidates, 3).and_then(
            |measurements| {
                crate::uop::UOpTuningReceipt::from_candidates(
                    graph.graph_digest.clone(),
                    uop_target,
                    &uop_strategy_candidates,
                    &[crate::uop::UOpWorkloadMeasurement {
                        scenario,
                        measurements,
                    }],
                    crate::uop::UOpMeasurementSource::CpuReference,
                    true,
                )
            },
        ) {
            Ok(receipt) => Some(receipt),
            Err(error) => Some(
                crate::uop::UOpTuningReceipt::explicit_fallback(
                    graph.graph_digest.clone(),
                    uop_target,
                    format!("CPU reference measurement unavailable: {error}"),
                )
                .map_err(CompileError::KernelGenFailed)?,
            ),
        }
    };
    // Update session status
    if let Ok(status) = world.component_mut::<CompilationSession>(session) {
        status.status = SessionStatus::KernelsGenerated;
    }

    world
        .insert_component(
            session,
            KernelCollection {
                artifacts,
                kernel_count,
                lowered_manifests,
                uop_capture,
                uop_strategy_captures,
                uop_tuning_receipt,
            },
        )
        .map_err(|e| CompileError::KernelGenFailed(e.to_string()))?;

    Ok(())
}

/// Lower a MatMul SpatialNode into a native XDNA artifact payload.
///
/// This is the AMD NPU branch of kernel generation. It compiles a synthetic
/// minimal world with the MatMul operands and dispatches the AMD NPU
/// compiler to produce an [`XdnaArtifact`]. The result is wrapped in a
/// [`KernelArtifact`] keyed by the source node id.
fn compile_spatial_matmul_to_native_xdna(
    node: &prism_spatial_ir::graph::SpatialNode,
    node_id: prism_spatial_ir::graph::SpatialNodeId,
) -> Result<KernelArtifact, CompileError> {
    use prism_ecs_core::entity::EntityKind;
    use prism_ecs_core::world::World;
    use prism_ecs_ir::ir_types::{FloatKind, TensorType, Type};
    use prism_ecs_ir::op::{OpMarker, OpName, Operands, Results};
    use prism_ecs_ir::value::{Uses, ValueType};
    use prism_ecs_core::component::Component;

    let SpatialNode::Compute { shape, kind, .. } = node else {
        return Err(CompileError::KernelGenFailed(format!(
            "XDNA node {node_id} is not a compute node"
        )));
    };
    if *kind != prism_spatial_ir::graph::ComputeKind::MatMul
        || shape.in_shapes.len() < 2
        || shape.out_shapes.is_empty()
    {
        return Err(CompileError::KernelGenFailed(format!(
            "XDNA node {node_id} is not a statically shaped MatMul"
        )));
    }
    let to_tensor = |dims: &prism_ecs_ir::cimage_types::TensorShape| {
        Type::Tensor(TensorType::new(
            dims.dims.iter().map(|dim| *dim as u64).collect(),
            Type::float(FloatKind::F16),
        ))
    };
    let mut synthetic = World::new();
    let make_value = |world: &mut World, name: &str, ty: Type| -> Result<Entity, CompileError> {
        let value: Entity = world
            .spawn(EntityKind::Node, Some(name.into()))
            .map_err(|error| CompileError::KernelGenFailed(error.to_string()))?
            .into();
        world
            .add_component(value, ValueType(ty))
            .map_err(|error| CompileError::KernelGenFailed(error.to_string()))?;
        world
            .add_component(value, Uses(vec![]))
            .map_err(|error| CompileError::KernelGenFailed(error.to_string()))?;
        Ok(value)
    };
    let a = make_value(&mut synthetic, "A", to_tensor(&shape.in_shapes[0]))?;
    let b = make_value(&mut synthetic, "B", to_tensor(&shape.in_shapes[1]))?;
    let c = make_value(&mut synthetic, "C", to_tensor(&shape.out_shapes[0]))?;
    let result = make_value(&mut synthetic, "result", to_tensor(&shape.out_shapes[0]))?;
    let op: Entity = synthetic
        .spawn(EntityKind::Node, Some("matmul".into()))
        .map_err(|error| CompileError::KernelGenFailed(error.to_string()))?
        .into();
    synthetic
        .add_component(op, OpMarker)
        .map_err(|error| CompileError::KernelGenFailed(error.to_string()))?;
    synthetic
        .add_component(
            op,
            OpName(if shape.in_shapes[0].dims.len() == 3 {
                "linalg.batch_matmul".into()
            } else {
                "linalg.matmul".into()
            }),
        )
        .map_err(|error| CompileError::KernelGenFailed(error.to_string()))?;
    synthetic
        .add_component(op, Operands(vec![a, b, c]))
        .map_err(|error| CompileError::KernelGenFailed(error.to_string()))?;
    synthetic
        .add_component(op, Results(vec![result]))
        .map_err(|error| CompileError::KernelGenFailed(error.to_string()))?;
    let executable = prism_amd_npu_runtime::compile_amd_npu(
        &synthetic,
        op,
        prism_ecs_ir::backend_dispatch::HalFormat::AmdNpu,
    )
    .map_err(CompileError::KernelGenFailed)?;
    let artifact = prism_amd_npu_runtime::XdnaArtifact::decode_hex_envelope(&executable.source)
        .map_err(CompileError::KernelGenFailed)?;
    let binary = artifact.encode().map_err(CompileError::KernelGenFailed)?;
    let digest = hex::encode(sha2::Sha256::digest(&binary));
    let descriptor = KernelDescriptor {
        name: format!("xdna_node_{node_id}"),
        variant: KernelVariant::Custom("native-xdna-artifact".into()),
        backend: BackendKind::AmdNpu,
        source_digest: digest.clone(),
        binary_digest: digest.clone(),
        binding_signature: Vec::new(),
        dispatch_geometry: DispatchGeometry {
            threads_per_threadgroup: [1, 1, 1],
            threadgroups_per_grid: [1, 1, 1],
            threads_per_grid: [1, 1, 1],
        },
    };
    Ok(KernelArtifact {
        payloads: vec![KernelPayload {
            binary,
            descriptor: descriptor.clone(),
        }],
        manifest: KernelManifest {
            kernels: vec![descriptor],
            fusion_plan: None,
            manifest_digest: digest.clone(),
        },
        artifact_digest: digest,
    })
}

#[cfg(any(feature = "ane", test))]
fn matmul_dimensions(node: &SpatialNode) -> Option<(usize, usize, usize)> {
    let SpatialNode::Compute {
        kind: prism_spatial_ir::graph::ComputeKind::MatMul,
        shape,
        ..
    } = node
    else {
        return None;
    };
    let a = shape.in_shapes.first()?.dims.as_slice();
    let b = shape.in_shapes.get(1)?.dims.as_slice();
    let c = shape.out_shapes.first()?.dims.as_slice();
    if a.len() != 2 || b.len() != 2 || c.len() != 2 || a[1] != b[0] || c != [a[0], b[1]] {
        return None;
    }
    Some((a[0], a[1], b[1]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ane_matmul_dimension_gate_accepts_only_static_compatible_shapes() {
        let node = SpatialNode::Compute {
            id: prism_spatial_ir::graph::SpatialNodeId(7),
            kind: prism_spatial_ir::graph::ComputeKind::MatMul,
            shape: prism_spatial_ir::graph::ShapeContract::new(
                vec![
                    prism_ecs_ir::cimage_types::TensorShape { dims: vec![2, 4] },
                    prism_ecs_ir::cimage_types::TensorShape { dims: vec![4, 3] },
                ],
                vec![prism_ecs_ir::cimage_types::TensorShape { dims: vec![2, 3] }],
            ),
            intensity: prism_spatial_ir::graph::ComputeIntensity::ComputeBound,
        };
        assert_eq!(matmul_dimensions(&node), Some((2, 4, 3)));
    }

    #[test]
    fn amd_npu_spatial_matmul_emits_native_xdna_artifact_payload() {
        let node = SpatialNode::Compute {
            id: prism_spatial_ir::graph::SpatialNodeId(8),
            kind: prism_spatial_ir::graph::ComputeKind::MatMul,
            shape: prism_spatial_ir::graph::ShapeContract::new(
                vec![
                    prism_ecs_ir::cimage_types::TensorShape { dims: vec![4, 8] },
                    prism_ecs_ir::cimage_types::TensorShape { dims: vec![8, 16] },
                ],
                vec![prism_ecs_ir::cimage_types::TensorShape { dims: vec![4, 16] }],
            ),
            intensity: prism_spatial_ir::graph::ComputeIntensity::ComputeBound,
        };
        let artifact =
            compile_spatial_matmul_to_native_xdna(&node, prism_spatial_ir::graph::SpatialNodeId(8))
                .unwrap();
        let payload = &artifact.payloads[0].binary;
        let decoded = prism_amd_npu_runtime::XdnaArtifact::decode(payload).unwrap();
        assert_eq!(
            decoded.program.topology.generation,
            prism_spatial_ir::xdna::XdnaGeneration::Aie2p
        );
        assert_eq!(artifact.payloads[0].descriptor.backend, BackendKind::AmdNpu);
        assert_eq!(
            artifact.payloads[0].descriptor.variant,
            KernelVariant::Custom("native-xdna-artifact".into())
        );
    }

    #[test]
    fn amd_npu_spatial_batched_matmul_preserves_batch_lowering() {
        let node = SpatialNode::Compute {
            id: prism_spatial_ir::graph::SpatialNodeId(9),
            kind: prism_spatial_ir::graph::ComputeKind::MatMul,
            shape: prism_spatial_ir::graph::ShapeContract::new(
                vec![
                    prism_ecs_ir::cimage_types::TensorShape {
                        dims: vec![2, 4, 8],
                    },
                    prism_ecs_ir::cimage_types::TensorShape {
                        dims: vec![2, 8, 16],
                    },
                ],
                vec![prism_ecs_ir::cimage_types::TensorShape {
                    dims: vec![2, 4, 16],
                }],
            ),
            intensity: prism_spatial_ir::graph::ComputeIntensity::ComputeBound,
        };
        let artifact =
            compile_spatial_matmul_to_native_xdna(&node, prism_spatial_ir::graph::SpatialNodeId(9))
                .expect("batched MatMul must lower natively");
        let decoded =
            prism_amd_npu_runtime::XdnaArtifact::decode(&artifact.payloads[0].binary).unwrap();
        assert!(decoded
            .program
            .buffers
            .iter()
            .any(|buffer| buffer.shape == vec![2, 4, 8]));
    }
}
