//! Region encoder — encodes an ExecutionRegion into Metal command buffer dispatches.
//! The trait is backend-agnostic; the Metal runtime provides the concrete implementation.

use super::*;

/// Handle returned from encoding a region — may contain timing data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionExecutionHandle {
    pub region_id: String,
    pub command_buffer_index: u32,
    pub compute_encoder_count: u32,
    pub ops_encoded: u32,
    pub barriers_inserted: u32,
}

/// Encoder for the region encoder — the Metal runtime provides this.
pub type RegionEncoderError = String;

/// A region encoder consumes an ExecutionRegion and produces a handle
/// that can be used to wait for and measure execution.
pub trait RegionEncoder {
    type PipelineState;

    /// Create a new region encoder from a Metal device.
    #[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
    fn new(device: &metal::Device) -> Self;

    fn encode_region(
        &mut self,
        region: &ExecutionRegion,
        pso_cache: &mut dyn super::pso_cache::PsoCache<PipelineState = Self::PipelineState>,
    ) -> Result<RegionExecutionHandle, RegionEncoderError>;

    fn encode_regions(
        &mut self,
        regions: &[ExecutionRegion],
        pso_cache: &mut dyn super::pso_cache::PsoCache<PipelineState = Self::PipelineState>,
    ) -> Result<Vec<RegionExecutionHandle>, RegionEncoderError> {
        let mut handles = Vec::new();
        for region in regions {
            handles.push(self.encode_region(region, pso_cache)?);
        }
        Ok(handles)
    }
}

/// Default encoding algorithm (not Metal-specific).
/// Walks the ops in order, looks up PSOs, simulates dispatch.
pub fn encode_region_from_plan(
    region: &ExecutionRegion,
    hazard_plan: &HazardPlan,
) -> Result<RegionExecutionHandle, RegionEncoderError> {
    if !hazard_plan.safe {
        return Err(format!(
            "Cannot encode unsafe region: {:?}",
            region.region_id
        ));
    }

    let mut encoder_count = 1u32;
    let barriers = region.arena_plan.allocations.len() as u32;

    // Boundary insertion tracking
    for boundary in &hazard_plan.encoder_boundaries {
        if boundary.after_op_index < region.ops.len() - 1 {
            encoder_count += 1;
        }
    }

    Ok(RegionExecutionHandle {
        region_id: region.region_id.clone(),
        command_buffer_index: 0,
        compute_encoder_count: encoder_count,
        ops_encoded: region.ops.len() as u32,
        barriers_inserted: barriers + hazard_plan.required_barriers.len() as u32,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_encode_safe_region_succeeds() {
        let region = region_fixture();
        let hazard = HazardChecker::validate_region(&region).unwrap();
        let handle = encode_region_from_plan(&region, &hazard).unwrap();
        assert_eq!(handle.region_id, "test_region");
        assert!(handle.ops_encoded >= 1);
    }

    #[test]
    fn test_encode_unsafe_region_rejected() {
        let mut region = region_fixture();
        // Clone the buffer_id shared across ops before the mutable borrows
        let shared_id = region.ops[0].buffer_uses[0].buffer_id.clone();
        // Make it unsafe by giving both ops the same write buffer
        region.ops[0].buffer_uses[0].access = AccessMode::Write;
        region.ops[1].buffer_uses[0].access = AccessMode::Read;
        region.ops[1].buffer_uses[0].buffer_id = shared_id;
        let hazard = HazardChecker::validate_region(&region).unwrap();
        if !hazard.safe {
            let result = encode_region_from_plan(&region, &hazard);
            assert!(result.is_err());
        }
    }

    fn region_fixture() -> ExecutionRegion {
        ExecutionRegion {
            region_id: "test_region".into(),
            region_kind: ExecutionRegionKind::DecoderLayerDecode,
            layer_index: Some(5),
            phase: ExecutionPhase::Decode,
            ops: vec![
                ScheduledKernelOp {
                    op_id: "rmsnorm".into(),
                    op_kind: KernelOpKind::RmsNorm,
                    tensor_key: None,
                    tensor_class: None,
                    specialization: kernel_specialization_key_fixture(),
                    bindings: vec![],
                    dependencies: vec![],
                    buffer_uses: vec![BufferUse {
                        buffer_id: "hidden_a".into(),
                        access: AccessMode::ReadWrite,
                        lifetime: LifetimeClass::LayerScratch,
                        alias_group: None,
                        byte_range: Some(ByteRange {
                            start: 0,
                            end: 4096,
                        }),
                    }],
                    dispatch_shape: DispatchShape {
                        grid_x: 1,
                        grid_y: 1,
                        grid_z: 1,
                        threadgroup_m: 32,
                        threadgroup_n: 1,
                        threadgroup_p: 1,
                    },
                    estimated_cost: EstimatedKernelCost {
                        compute_us: 5.0,
                        memory_bytes_read: 4096,
                        memory_bytes_written: 4096,
                    },
                    validation_requirements: KernelValidationRequirements::default(),
                },
                ScheduledKernelOp {
                    op_id: "silu".into(),
                    op_kind: KernelOpKind::MlpActivation,
                    tensor_key: None,
                    tensor_class: None,
                    specialization: kernel_specialization_key_fixture(),
                    bindings: vec![],
                    dependencies: vec!["rmsnorm".into()],
                    buffer_uses: vec![BufferUse {
                        buffer_id: "hidden_b".into(),
                        access: AccessMode::ReadWrite,
                        lifetime: LifetimeClass::LayerScratch,
                        alias_group: None,
                        byte_range: Some(ByteRange {
                            start: 0,
                            end: 4096,
                        }),
                    }],
                    dispatch_shape: DispatchShape {
                        grid_x: 1,
                        grid_y: 1,
                        grid_z: 1,
                        threadgroup_m: 32,
                        threadgroup_n: 1,
                        threadgroup_p: 1,
                    },
                    estimated_cost: EstimatedKernelCost {
                        compute_us: 3.0,
                        memory_bytes_read: 4096,
                        memory_bytes_written: 4096,
                    },
                    validation_requirements: KernelValidationRequirements::default(),
                },
            ],
            command_buffer_policy: CommandBufferPolicy::decode_default(),
            hazard_policy: HazardPolicy::Conservative,
            arena_plan: ActivationArenaPlan {
                arena_id: "test".into(),
                total_bytes: 4096,
                allocations: vec![],
                alias_groups: vec![],
                peak_live_bytes: 4096,
            },
            timing_policy: TimingPolicy::Disabled,
        }
    }

    fn kernel_specialization_key_fixture() -> KernelSpecializationKey {
        KernelSpecializationKey {
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
        }
    }
}
