//! Deterministic heterogeneous routing types — re-exported from prism-ecs-backend.

pub mod lanes;
pub mod policy;

pub use prism_ecs_backend::routing::*;

#[cfg(test)]
#[cfg(feature = "legacy_mutations")]
mod tests {
    use super::*;
    use crate::ecs::backend::residency;

    // ── 1. All BACKEND_* constants are distinct ──────────────────────────

    #[test]
    fn backend_constants_are_distinct() {
        let ids = [
            BACKEND_METAL.0,
            BACKEND_ACCELERATE.0,
            BACKEND_ANE.0,
            BACKEND_MLX.0,
        ];
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                assert_ne!(
                    ids[i], ids[j],
                    "BACKEND_* constants must have distinct values; found duplicate {}",
                    ids[i]
                );
            }
        }
    }

    // ── 2. Canonical values ──────────────────────────────────────────────

    #[test]
    fn backend_constant_values() {
        assert_eq!(BACKEND_METAL.0, 0, "BACKEND_METAL must be 0");
        assert_eq!(BACKEND_ACCELERATE.0, 1, "BACKEND_ACCELERATE must be 1");
        assert_eq!(BACKEND_ANE.0, 2, "BACKEND_ANE must be 2");
        assert_eq!(BACKEND_MLX.0, 3, "BACKEND_MLX must be 3");
    }

    // ── 3. Fixtures using canonical constants ────────────────────────────

    fn make_descriptor(operation_id: u64, family: OperationFamily) -> OperationDescriptor {
        OperationDescriptor {
            operation_id: OperationId(operation_id),
            family,
            layer_index: Some(0),
            phase: Phase::Decode,
            logical_shape: LogicalShape {
                dims: vec![1, 32, 128],
            },
            physical_layout: PhysicalLayout::RowMajor,
            input_dtypes: vec![DType::F16],
            output_dtype: DType::F16,
            quantization: None,
            expected_output_shape: TensorShape {
                dims: vec![1, 32, 128],
            },
            correctness_checkpoint: CorrectnessCheckpointPolicy::None,
        }
    }

    fn make_receipt(operation_id: u64, backend_id: BackendId) -> BackendExecutionReceipt {
        BackendExecutionReceipt {
            operation_id: OperationId(operation_id),
            backend_id,
            backend_version: BackendVersion {
                backend_name: "test".into(),
                version: "0.1.0".into(),
                git_commit: None,
            },
            requested_substrate: None,
            observed_substrate: None,
            graph_build_ns: None,
            compile_ns: None,
            queue_wait_ns: None,
            submit_ns: Some(100),
            execution_ns: Some(1000),
            synchronization_ns: None,
            total_wall_ns: 1200,
            bytes_read: None,
            bytes_written: None,
            temporary_bytes: None,
            active_memory_before: None,
            active_memory_after: None,
            cache_memory_before: None,
            cache_memory_after: None,
            transfer_in_ns: None,
            transfer_out_ns: None,
            fallback_occurred: false,
        }
    }

    #[test]
    fn fixtures_use_canonical_backend_ids() {
        // OperationDescriptor fixtures — just ensure they construct without
        // panicking (BackendId is not a field on this type).
        let _metal_op = make_descriptor(1, OperationFamily::Matmul);
        let _accel_op = make_descriptor(2, OperationFamily::QuantizedMatmul);
        let _ane_op = make_descriptor(3, OperationFamily::AttentionBlock);

        // BackendExecutionReceipt fixtures — assert backend_id is canonical.
        let metal_receipt = make_receipt(1, BACKEND_METAL);
        let accel_receipt = make_receipt(2, BACKEND_ACCELERATE);
        let ane_receipt = make_receipt(3, BACKEND_ANE);
        let mlx_receipt = make_receipt(4, BACKEND_MLX);

        assert_eq!(metal_receipt.backend_id, BACKEND_METAL);
        assert_eq!(accel_receipt.backend_id, BACKEND_ACCELERATE);
        assert_eq!(ane_receipt.backend_id, BACKEND_ANE);
        assert_eq!(mlx_receipt.backend_id, BACKEND_MLX);

        // Guard: ensure we aren't using arbitrary raw integers.
        assert_ne!(metal_receipt.backend_id.0, 99);
        assert_ne!(accel_receipt.backend_id.0, 99);
    }

    // ── 4. residency::BackendId → routing::BackendId mapping ─────────────

    #[test]
    fn residency_to_routing_id_mapping() {
        // MlxMetal → BACKEND_MLX
        assert_eq!(
            residency::BackendId::MlxMetal.to_routing_id(),
            Some(BACKEND_MLX)
        );
        // Accelerate → BACKEND_ACCELERATE
        assert_eq!(
            residency::BackendId::Accelerate.to_routing_id(),
            Some(BACKEND_ACCELERATE)
        );
        // CoreAi → BACKEND_ANE
        assert_eq!(
            residency::BackendId::CoreAi.to_routing_id(),
            Some(BACKEND_ANE)
        );
        // Ane → BACKEND_ANE
        assert_eq!(residency::BackendId::Ane.to_routing_id(), Some(BACKEND_ANE));

        // Variants that map to None
        assert_eq!(residency::BackendId::CandleCpu.to_routing_id(), None);
        assert_eq!(residency::BackendId::TensixTensix.to_routing_id(), None);
        assert_eq!(residency::BackendId::IntelLevelZero.to_routing_id(), None);
        assert_eq!(residency::BackendId::IntelOpenCl.to_routing_id(), None);
        assert_eq!(residency::BackendId::HostCpu.to_routing_id(), None);
        assert_eq!(residency::BackendId::Unknown.to_routing_id(), None);
    }
}
