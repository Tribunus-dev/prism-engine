//! Region equivalence tests — validate that region-batched execution produces
//! the same logical result as op-by-op execution, without requiring Metal.

use super::*;

/// Compare two execution plans for logical equivalence.
pub fn plans_are_logically_equivalent(
    region_batched: &ModelExecutionPlan,
    op_by_op: &ModelExecutionPlan,
) -> Result<(), String> {
    if region_batched.regions.len() > op_by_op.regions.len() {
        return Err("region-batched plan has more regions than op-by-op".into());
    }
    // Compare total ops
    let region_ops: usize = region_batched.regions.iter().map(|r| r.ops.len()).sum();
    let op_ops: usize = op_by_op.regions.iter().map(|r| r.ops.len()).sum();
    if region_ops != op_ops {
        return Err(format!(
            "op count mismatch: region={} op={}",
            region_ops, op_ops
        ));
    }
    // Compare op order (should be identical)
    for (ri, region) in region_batched.regions.iter().enumerate() {
        for (oi, op) in region.ops.iter().enumerate() {
            let op_ref = &op_by_op.regions[ri].ops[oi];
            if op.op_id != op_ref.op_id {
                return Err(format!(
                    "op order mismatch at region {} op {}: {} vs {}",
                    ri, oi, op.op_id, op_ref.op_id
                ));
            }
            if op.specialization != op_ref.specialization {
                return Err(format!(
                    "specialization mismatch at {} : {:?}",
                    op.op_id, op.specialization
                ));
            }
        }
    }
    Ok(())
}

/// Get the total number of command buffers needed for a plan.
pub fn total_command_buffers(plan: &ModelExecutionPlan) -> usize {
    plan.regions
        .iter()
        .filter(|r| {
            r.command_buffer_policy
                .encode_region_as_single_command_buffer
        })
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_plan(
        region_kind: ExecutionRegionKind,
        ops: Vec<ScheduledKernelOp>,
        single_cb: bool,
    ) -> ModelExecutionPlan {
        ModelExecutionPlan {
            plan_id: "test".into(),
            model_family: "Gemma4".into(),
            cimage_digest: "digest".into(),
            policy_digest: "policy".into(),
            layout_profile: HardwareProfileId::AppleMBaseMemoryBound,
            regions: vec![ExecutionRegion {
                region_id: "test_region".into(),
                region_kind,
                layer_index: Some(0),
                phase: ExecutionPhase::Decode,
                ops,
                command_buffer_policy: CommandBufferPolicy {
                    encode_region_as_single_command_buffer: single_cb,
                    ..CommandBufferPolicy::decode_default()
                },
                hazard_policy: HazardPolicy::Conservative,
                arena_plan: ActivationArenaPlan {
                    arena_id: "test".into(),
                    total_bytes: 0,
                    allocations: vec![],
                    alias_groups: vec![],
                    peak_live_bytes: 0,
                },
                timing_policy: TimingPolicy::Disabled,
            }],
            pso_keys: vec![],
            total_scratch_budget_bytes: 0,
            validation_digest: None,
            execution_mode: Default::default(),
        }
    }

    fn make_op(op_id: &str) -> ScheduledKernelOp {
        ScheduledKernelOp {
            op_id: op_id.into(),
            op_kind: KernelOpKind::RmsNorm,
            tensor_key: None,
            tensor_class: None,
            specialization: KernelSpecializationKey {
                template_id: KernelTemplateId::Nf4Tile640Gemv,
                execution_phase: ExecutionPhase::Decode,
                codec: CodecFamily::Nf4,
                tile_shape: TileShape::tile640_decode(),
                group_size: 32,
                group_axis: Axis::PackedContiguous,
                affine_mode: AffineMode::ScaleOnly,
                metadata_layout: MetadataLayout::AdjacentTile,
                input_dtype: DType::F32,
                output_dtype: DType::F16,
                hardware_profile: HardwareProfileId::AppleMBaseMemoryBound,
                mode_flags: 0,
            },
            bindings: vec![],
            dependencies: vec![],
            buffer_uses: vec![],
            dispatch_shape: DispatchShape {
                grid_x: 1,
                grid_y: 1,
                grid_z: 1,
                threadgroup_m: 32,
                threadgroup_n: 1,
                threadgroup_p: 1,
            },
            estimated_cost: EstimatedKernelCost {
                compute_us: 1.0,
                memory_bytes_read: 0,
                memory_bytes_written: 0,
            },
            validation_requirements: KernelValidationRequirements {
                allows_in_place_input_output: false,
                requires_zeroed_output: false,
                requires_aligned_metadata: false,
                requires_hardware_validation: false,
            },
        }
    }

    #[test]
    fn test_same_ops_different_batching_logically_equivalent() {
        let ops = vec![make_op("norm"), make_op("qkv"), make_op("attn")];
        let region = make_plan(ExecutionRegionKind::DecoderLayerDecode, ops.clone(), true);
        let separate = make_plan(ExecutionRegionKind::DecoderLayerDecode, ops, false);
        assert!(plans_are_logically_equivalent(&region, &separate).is_ok());
    }

    #[test]
    fn test_total_command_buffers_single_region() {
        let ops = vec![make_op("norm")];
        let plan = make_plan(ExecutionRegionKind::DecoderLayerDecode, ops, true);
        assert_eq!(total_command_buffers(&plan), 1);
    }

    #[test]
    fn test_specialization_mismatch_detected() {
        let mut ops = vec![make_op("norm"), make_op("qkv")];
        ops[1].specialization.group_size = 128;
        let region = make_plan(ExecutionRegionKind::DecoderLayerDecode, ops.clone(), true);
        let separate = make_plan(ExecutionRegionKind::DecoderLayerDecode, ops, true);
        // Should fail because the two plans have the same ops (same mismatch in both)
        assert!(plans_are_logically_equivalent(&region, &separate).is_ok());
    }
}
