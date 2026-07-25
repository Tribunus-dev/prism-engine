//! Runtime module tests — exercised through the public re-exports in
//! [`super`] (which is `runtime/mod.rs`).
//!
//! The runtime is decomposed by entity kind, so these tests live in a
//! dedicated `tests` submodule rather than `mod.rs` to keep the parent
//! module below the constitutional 200-LOC limit (see
//! `references/module-discipline.md` and the rust-quality rule on
//! `mod.rs` size).

#[cfg(test)]
use super::*;

    /// Verify that every [`RuntimeError`] variant formats correctly and
    /// that the `std::error::Error` trait (from `thiserror`) is satisfied.
    #[test]
    fn test_runtime_error_types() {
        let err = RuntimeError::FileNotFound("missing.cimage".into());
        assert_eq!(format!("{err}"), "File not found: missing.cimage");

        let err = RuntimeError::InvalidCImage("bad magic".into());
        assert_eq!(format!("{err}"), "Invalid CImage: bad magic");

        let err = RuntimeError::IncompatibleSchema("v2 required".into());
        assert_eq!(format!("{err}"), "Incompatible schema: v2 required");

        let err = RuntimeError::TensorNotFound("weights".into());
        assert_eq!(format!("{err}"), "Tensor not found: weights");

        let err = RuntimeError::KernelNotFound("matmul".into());
        assert_eq!(format!("{err}"), "Kernel not found: matmul");

        let err = RuntimeError::ExecutionFailed("OOM".into());
        assert_eq!(format!("{err}"), "Execution failed: OOM");

        let err = RuntimeError::BackendError("GPU hung".into());
        assert_eq!(format!("{err}"), "Backend error: GPU hung");

        let err = RuntimeError::UnsupportedMode("decode".into());
        assert_eq!(format!("{err}"), "Unsupported execution mode: decode");
    }

    /// [`ExecutionMode`] derives `Clone + Copy`, so a copied value must be
    /// equal to the original and independent.
    #[test]
    fn test_execution_mode_copy() {
        let batch = ExecutionMode::Batch;
        let prefill = ExecutionMode::RealtimePrefill;
        let decode = ExecutionMode::RealtimeDecode;

        // Copy semantics — second binding is a bitwise copy.
        let batch2 = batch;
        let prefill2 = prefill;
        let decode2 = decode;

        assert_eq!(batch, batch2);
        assert_eq!(prefill, prefill2);
        assert_eq!(decode, decode2);

        // All variants are distinct.
        assert_ne!(batch, prefill);
        assert_ne!(batch, decode);
        assert_ne!(prefill, decode);
    }

    /// Construct a [`RuntimeModel`] with empty maps and verify the
    /// accessor methods return `None` for unknown names.
    #[test]
    fn test_runtime_model_new() {
        let manifest = CImageManifest::default();
        let model = RuntimeModel {
            cimage_path: PathBuf::from("test.cimage"),
            source_identity: None,
            source_catalog: None,
            search_trace: None,
            legalization_report: None,
            compilation_events: None,
            manifest,
            tensors: HashMap::new(),
            tensor_records: HashMap::new(),
            tensor_scales: HashMap::new(),
            kernels: HashMap::new(),
            kernel_descriptors: HashMap::new(),
            uop_capture: None,
            uop_program: None,
            uop_strategy_programs: HashMap::new(),
            uop_workload_evidence: Vec::new(),
            ane_programs: HashMap::new(),
            xdna_artifacts: HashMap::new(),
            kv_compression_policy: None,
            model_manifest: None,
            native_ternary_promotion: None,
            joint_tiling_evidence: None,
            format_plan: None,
            heterogeneous_workload_evidence: None,
            execution_plan: None,
            realtime_execution_plan: None,
            tensor_offsets: HashMap::new(),
            mapped_cimage: None,
        };

        assert_eq!(model.cimage_path.to_str(), Some("test.cimage"));
        assert!(model.get_tensor("nonexistent").is_none());
        assert!(model.get_kernel("nonexistent").is_none());
        assert_eq!(model.num_layers(), 0);
    }

    #[test]
    fn kernel_selection_is_scoped_to_route_backend() {
        let geometry = prism_ecs_kernel::DispatchGeometry {
            threads_per_threadgroup: [1, 1, 1],
            threadgroups_per_grid: [1, 1, 1],
            threads_per_grid: [1, 1, 1],
        };
        let descriptor = |backend| prism_ecs_kernel::KernelDescriptor {
            name: String::new(),
            variant: prism_ecs_kernel::KernelVariant::Custom("test".into()),
            backend,
            source_digest: String::new(),
            binary_digest: String::new(),
            binding_signature: Vec::new(),
            dispatch_geometry: geometry,
        };
        let mut descriptors = HashMap::new();
        descriptors.insert("cpu_step".into(), descriptor(BackendKind::CPU));
        descriptors.insert("metal_step".into(), descriptor(BackendKind::Metal));

        let cpu_names = kernel_names_for_backend(&descriptors, BackendKind::CPU);
        let metal_names = kernel_names_for_backend(&descriptors, BackendKind::Metal);
        assert_eq!(
            cpu_names
                .iter()
                .map(|name| name.as_str())
                .collect::<Vec<_>>(),
            ["cpu_step"]
        );
        assert_eq!(
            metal_names
                .iter()
                .map(|name| name.as_str())
                .collect::<Vec<_>>(),
            ["metal_step"]
        );
    }

    #[test]
    fn xdna_route_materializes_mapped_tensor_inputs_when_payload_is_absent() {
        let mut tensors = HashMap::new();
        tensors.insert("weights".into(), vec![1, 2, 3, 4]);
        let model = RuntimeModel {
            cimage_path: PathBuf::from("test.cimage"),
            source_identity: None,
            source_catalog: None,
            search_trace: None,
            legalization_report: None,
            compilation_events: None,
            manifest: CImageManifest::default(),
            tensors,
            tensor_records: HashMap::new(),
            tensor_scales: HashMap::new(),
            kernels: HashMap::new(),
            kernel_descriptors: HashMap::new(),
            uop_capture: None,
            uop_program: None,
            uop_strategy_programs: HashMap::new(),
            uop_workload_evidence: Vec::new(),
            ane_programs: HashMap::new(),
            xdna_artifacts: HashMap::new(),
            kv_compression_policy: None,
            model_manifest: None,
            native_ternary_promotion: None,
            joint_tiling_evidence: None,
            format_plan: None,
            heterogeneous_workload_evidence: None,
            execution_plan: None,
            realtime_execution_plan: None,
            tensor_offsets: HashMap::new(),
            mapped_cimage: None,
        };
        let dispatcher = CImageXdnaRouteDispatcher {
            model: &model,
            runtime: XdnaRuntime::new(),
            device: (),
            phase: XdnaExecutionPhase::Decode,
        };
        let inputs = vec![ResolvedBuffer {
            name: "weights".into(),
            element_type: "u8".into(),
            region: "unified-memory".into(),
            byte_length: 4,
            zero_copy: true,
            file_offset: Some(128),
            storage: BufferStorage::MappedCImage,
            shape: vec![4],
            payload: None,
        }];
        assert_eq!(
            dispatcher.payloads_for_inputs(&inputs)["weights"],
            vec![1, 2, 3, 4]
        );
    }

    #[test]
    fn runtime_selects_workload_strategy_program() {
        let mut graph = prism_spatial_ir::TinyGraph::default();
        let input = graph.add(
            prism_spatial_ir::UOpKind::Input { name: "x".into() },
            vec![],
            vec![2],
        );
        let relu = graph.add(prism_spatial_ir::UOpKind::Relu, vec![input], vec![2]);
        let exp = graph.add(prism_spatial_ir::UOpKind::Exp, vec![relu], vec![2]);
        graph.add(
            prism_spatial_ir::UOpKind::Output { name: "y".into() },
            vec![exp],
            vec![2],
        );
        let standard = UOpCompiledProgram::compile(
            graph
                .lower(prism_spatial_ir::LoweringTarget::Portable)
                .unwrap(),
        )
        .unwrap();
        let per_operation = UOpCompiledProgram::compile(
            graph
                .lower_with_fusion_strategy(
                    prism_spatial_ir::LoweringTarget::Portable,
                    &prism_spatial_ir::FusionStrategy::PerOperation,
                )
                .unwrap(),
        )
        .unwrap();
        let scenario = WorkloadScenario {
            realtime: false,
            batch_size: 32,
            sequence_length: 1,
        };
        let evaluation = prism_spatial_ir::FusionStrategyEvaluation {
            candidates: vec![
                prism_spatial_ir::FusionStrategyCandidate {
                    strategy: prism_spatial_ir::FusionStrategy::StandardFused,
                    kernel_count: 1,
                    estimated_latency_ns: 20,
                    estimated_materialized_bytes: 0,
                    score: 20.0,
                    measured: true,
                },
                prism_spatial_ir::FusionStrategyCandidate {
                    strategy: prism_spatial_ir::FusionStrategy::PerOperation,
                    kernel_count: 2,
                    estimated_latency_ns: 10,
                    estimated_materialized_bytes: 0,
                    score: 10.0,
                    measured: true,
                },
            ],
            selected: 1,
        };
        let plan = ExecutionPlan::new(
            prism_spatial_ir::execution_plan::ExecutionMode::Batch,
            vec![],
            32,
            false,
        )
        .with_workload_evaluations(vec![prism_spatial_ir::WorkloadStrategyEvaluation {
            scenario,
            evaluation,
        }]);
        let model = RuntimeModel {
            cimage_path: PathBuf::from("test.cimage"),
            source_identity: None,
            source_catalog: None,
            search_trace: None,
            legalization_report: None,
            compilation_events: None,
            manifest: CImageManifest::default(),
            tensors: HashMap::new(),
            tensor_records: HashMap::new(),
            tensor_scales: HashMap::new(),
            kernels: HashMap::new(),
            kernel_descriptors: HashMap::new(),
            uop_capture: Some(standard.capture.clone()),
            uop_program: Some(standard.clone()),
            uop_strategy_programs: HashMap::from([
                ("standard_fused".into(), standard.clone()),
                ("per_operation".into(), per_operation),
            ]),
            uop_workload_evidence: vec![crate::cimage::UOpWorkloadEvidence {
                scenario,
                strategies: vec!["standard_fused".into(), "per_operation".into()],
                candidate_capture_digests: Vec::new(),
                measurements: vec![
                    prism_spatial_ir::FusionMeasurement {
                        candidate_index: 0,
                        latency_ns: 100,
                        materialized_bytes: 0,
                    },
                    prism_spatial_ir::FusionMeasurement {
                        candidate_index: 1,
                        latency_ns: 1,
                        materialized_bytes: 0,
                    },
                ],
                selected_strategy: "per_operation".into(),
            }],
            ane_programs: HashMap::new(),
            xdna_artifacts: HashMap::new(),
            kv_compression_policy: None,
            model_manifest: None,
            native_ternary_promotion: None,
            joint_tiling_evidence: None,
            format_plan: None,
            heterogeneous_workload_evidence: None,
            execution_plan: Some(plan),
            realtime_execution_plan: None,
            tensor_offsets: HashMap::new(),
            mapped_cimage: None,
        };
        let mut runtime = UnifiedRuntime::new(model);
        assert_eq!(
            runtime.selected_measured_strategy(scenario),
            Some("per_operation")
        );
        assert_eq!(
            selected_uop_program(&runtime, 2).unwrap().capture.digest(),
            runtime.model.uop_strategy_programs["per_operation"]
                .capture
                .digest()
        );
        assert_eq!(
            runtime.selected_measured_strategy(WorkloadScenario {
                realtime: false,
                batch_size: 32,
                sequence_length: 2,
            }),
            Some("per_operation")
        );
        assert!(runtime.run_batch_for_workload(&[1], 0).is_err());
        assert!(runtime.run_batch_for_workload(&[1, 2, 3], 2).is_err());
        let batch_logits = runtime
            .run_batch_for_workload(&[1, 2], 1)
            .expect("valid packed batch should dispatch");
        assert!(!batch_logits.is_empty());
        assert_eq!(
            runtime
                .model
                .uop_workload_evidence_for(scenario)
                .unwrap()
                .selected_strategy,
            "per_operation"
        );
        assert_eq!(
            selected_uop_program(&runtime, 1)
                .unwrap()
                .capture
                .kernels
                .len(),
            2
        );
        let selected = runtime
            .install_measured_strategy_choice(
                scenario,
                &[
                    prism_spatial_ir::FusionStrategy::StandardFused,
                    prism_spatial_ir::FusionStrategy::PerOperation,
                ],
                &[
                    prism_spatial_ir::FusionMeasurement {
                        candidate_index: 0,
                        latency_ns: 1,
                        materialized_bytes: 0,
                    },
                    prism_spatial_ir::FusionMeasurement {
                        candidate_index: 1,
                        latency_ns: 100,
                        materialized_bytes: 0,
                    },
                ],
            )
            .unwrap();
        assert_eq!(selected, "standard_fused");
        assert_eq!(
            selected_uop_program(&runtime, 1)
                .unwrap()
                .capture
                .kernels
                .len(),
            1
        );
    }

    #[test]
    fn runtime_best_throughput_selection_respects_service_class() {
        use crate::workload_search::{
            ExecutionLane, InferenceWorkloadPhase, ServiceClass, WorkloadProfile,
            WorkloadThroughputEvidence,
        };

        let selected_graph = crate::search::HeterogeneousScheduleEvidence {
            steps: 0,
            route_sequence: Vec::new(),
            zero_copy_steps: 0,
            estimated_latency_ns: 0,
            residency_windows: 0,
            supports_realtime_text: true,
            supports_batched_text: true,
            supports_batched_audio: true,
            workload_profiles: Vec::new(),
            selected_execution_graph: crate::workload_search::SelectedExecutionGraph::default(),
            throughput_evidence: vec![
                WorkloadThroughputEvidence {
                    profile: WorkloadProfile {
                        phase: InferenceWorkloadPhase::Decode,
                        service_class: ServiceClass::Batch,
                        batch_size: 4,
                        concurrency: 4,
                        primary_lane: ExecutionLane::Metal,
                        attention_lane: ExecutionLane::Metal,
                        interleaved_metal: true,
                        stateless_shared_arena: false,
                        ane_compute_int8: false,
                        planar_input_conversion: false,
                        planar_output_conversion: false,
                    },
                    representation: "Int8".into(),
                    tiling_digest: "batch".into(),
                    tokens_per_second: 80.0,
                    latency_ms: 1.0,
                    measured: true,
                    evidence_source: "test".into(),
                    execution_fingerprint: "batch".into(),
                    projected: false,
                    projection_basis: "test".into(),
                    mixed_precision_graph: "batch-only".into(),
                    ..WorkloadThroughputEvidence::default()
                },
                WorkloadThroughputEvidence {
                    profile: WorkloadProfile {
                        phase: InferenceWorkloadPhase::Decode,
                        service_class: ServiceClass::Realtime,
                        batch_size: 1,
                        concurrency: 1,
                        primary_lane: ExecutionLane::Ane,
                        attention_lane: ExecutionLane::Ane,
                        interleaved_metal: false,
                        stateless_shared_arena: true,
                        ane_compute_int8: true,
                        planar_input_conversion: true,
                        planar_output_conversion: true,
                    },
                    representation: "Int8".into(),
                    tiling_digest: "realtime".into(),
                    tokens_per_second: 20.0,
                    latency_ms: 1.0,
                    measured: true,
                    evidence_source: "test".into(),
                    execution_fingerprint: "realtime".into(),
                    projected: false,
                    projection_basis: "test".into(),
                    mixed_precision_graph: "realtime-only".into(),
                    ..WorkloadThroughputEvidence::default()
                },
            ],
        };

        let target = RuntimeModel {
            cimage_path: PathBuf::from("test.cimage"),
            source_identity: None,
            source_catalog: None,
            search_trace: None,
            legalization_report: None,
            compilation_events: None,
            manifest: CImageManifest::default(),
            tensors: HashMap::new(),
            tensor_records: HashMap::new(),
            tensor_scales: HashMap::new(),
            kernels: HashMap::new(),
            kernel_descriptors: HashMap::new(),
            uop_capture: None,
            uop_program: None,
            uop_strategy_programs: HashMap::new(),
            uop_workload_evidence: Vec::new(),
            ane_programs: HashMap::new(),
            xdna_artifacts: HashMap::new(),
            kv_compression_policy: None,
            model_manifest: None,
            native_ternary_promotion: None,
            joint_tiling_evidence: None,
            format_plan: None,
            heterogeneous_workload_evidence: Some(selected_graph),
            execution_plan: None,
            realtime_execution_plan: None,
            tensor_offsets: HashMap::new(),
            mapped_cimage: None,
        };

        let runtime = UnifiedRuntime::new(target);
        let profile = WorkloadProfile {
            phase: InferenceWorkloadPhase::Decode,
            service_class: ServiceClass::Realtime,
            batch_size: 1,
            concurrency: 1,
            primary_lane: ExecutionLane::Ane,
            attention_lane: ExecutionLane::Ane,
            interleaved_metal: false,
            stateless_shared_arena: true,
            ane_compute_int8: true,
            planar_input_conversion: true,
            planar_output_conversion: true,
        };
        let evidence = runtime
            .model
            .best_throughput_evidence_for_profile(&profile)
            .expect("realtime evidence should be visible");
        assert_eq!(evidence.mixed_precision_graph, "realtime-only");
    }

    #[test]
    fn runtime_selected_execution_graph_is_round_tripped_from_heterogeneous_evidence() {
        use crate::workload_search::{
            ExecutionLane, InferenceWorkloadPhase, MixedPrecisionExecutionGraph,
            PrecisionSensitiveOp, SelectedExecutionGraph, ServiceClass, WorkloadProfile,
            WorkloadThroughputEvidence,
        };
        use std::collections::BTreeMap;

        let selected_graph = SelectedExecutionGraph {
            profiles: vec![WorkloadThroughputEvidence {
                profile: WorkloadProfile {
                    phase: InferenceWorkloadPhase::Decode,
                    service_class: ServiceClass::Realtime,
                    batch_size: 1,
                    concurrency: 2,
                    primary_lane: ExecutionLane::Ane,
                    attention_lane: ExecutionLane::Metal,
                    interleaved_metal: true,
                    stateless_shared_arena: true,
                    ane_compute_int8: true,
                    planar_input_conversion: true,
                    planar_output_conversion: true,
                },
                representation: "Ternary158".into(),
                tiling_digest: "tile-x".into(),
                tokens_per_second: 42.0,
                latency_ms: 0.75,
                measured: true,
                evidence_source: "runtime-test".into(),
                execution_fingerprint: "test-fingerprint".into(),
                projected: false,
                projection_basis: "test".into(),
                mixed_precision_graph: "ternary-expert-int8-attention".into(),
                ..WorkloadThroughputEvidence::default()
            }],
            route_sequence: vec![
                ExecutionLane::Ane,
                ExecutionLane::Metal,
                ExecutionLane::Accelerate,
            ],
            fused_interleaved_metal: true,
            stateless_ane: true,
            ane_int8_planar_boundaries: true,
            mixed_precision_graphs: vec![MixedPrecisionExecutionGraph {
                graph_id: "ternary-expert-int8-attention".into(),
                assignments: BTreeMap::from([
                    (PrecisionSensitiveOp::ExpertProjection, "Ternary158".into()),
                    (PrecisionSensitiveOp::Attention, "Int8".into()),
                    (PrecisionSensitiveOp::Router, "Int8".into()),
                    (PrecisionSensitiveOp::Norm, "Bf16".into()),
                    (PrecisionSensitiveOp::OutputHead, "Int8".into()),
                ]),
                ternary_native_ops: vec![PrecisionSensitiveOp::ExpertProjection],
                conversion_boundaries: vec![
                    "ane-planar-int8-in".into(),
                    "ane-planar-int8-out".into(),
                ],
            }],
        };

        let selected_graph_evidence = crate::search::HeterogeneousScheduleEvidence {
            steps: 8,
            route_sequence: vec!["ane".into(), "metal".into(), "accelerate".into()],
            zero_copy_steps: 4,
            estimated_latency_ns: 1_234_567,
            residency_windows: 2,
            supports_realtime_text: true,
            supports_batched_text: true,
            supports_batched_audio: false,
            workload_profiles: vec![WorkloadProfile {
                phase: InferenceWorkloadPhase::Decode,
                service_class: ServiceClass::Realtime,
                batch_size: 1,
                concurrency: 2,
                primary_lane: ExecutionLane::Ane,
                attention_lane: ExecutionLane::Metal,
                interleaved_metal: true,
                stateless_shared_arena: true,
                ane_compute_int8: true,
                planar_input_conversion: true,
                planar_output_conversion: true,
            }],
            selected_execution_graph: selected_graph.clone(),
            throughput_evidence: selected_graph.profiles.clone(),
        };

        let model = RuntimeModel {
            cimage_path: PathBuf::from("test.cimage"),
            source_identity: None,
            source_catalog: None,
            search_trace: None,
            legalization_report: None,
            compilation_events: None,
            manifest: CImageManifest::default(),
            tensors: HashMap::new(),
            tensor_records: HashMap::new(),
            tensor_scales: HashMap::new(),
            kernels: HashMap::new(),
            kernel_descriptors: HashMap::new(),
            uop_capture: None,
            uop_program: None,
            uop_strategy_programs: HashMap::new(),
            uop_workload_evidence: Vec::new(),
            ane_programs: HashMap::new(),
            xdna_artifacts: HashMap::new(),
            kv_compression_policy: None,
            model_manifest: None,
            native_ternary_promotion: None,
            joint_tiling_evidence: None,
            format_plan: None,
            heterogeneous_workload_evidence: Some(selected_graph_evidence),
            execution_plan: None,
            realtime_execution_plan: None,
            tensor_offsets: HashMap::new(),
            mapped_cimage: None,
        };

        let runtime = UnifiedRuntime::new(model);
        let graph = runtime
            .selected_execution_graph()
            .expect("selected execution graph should be present");
        assert_eq!(graph.route_sequence, selected_graph.route_sequence);
        assert!(graph.fused_interleaved_metal);
        assert!(graph.stateless_ane);
        assert!(graph.ane_int8_planar_boundaries);
        assert_eq!(graph.profiles.len(), selected_graph.profiles.len());
        assert_eq!(
            graph.mixed_precision_graphs[0].graph_id,
            "ternary-expert-int8-attention"
        );
        let profile = WorkloadProfile {
            phase: InferenceWorkloadPhase::Decode,
            service_class: ServiceClass::Realtime,
            batch_size: 1,
            concurrency: 2,
            primary_lane: ExecutionLane::Ane,
            attention_lane: ExecutionLane::Metal,
            interleaved_metal: true,
            stateless_shared_arena: true,
            ane_compute_int8: true,
            planar_input_conversion: true,
            planar_output_conversion: true,
        };
        let runtime_profile = runtime
            .model
            .best_throughput_evidence_for_profile(&profile)
            .expect("best profile should resolve through evidence");
        assert_eq!(
            runtime_profile.mixed_precision_graph,
            "ternary-expert-int8-attention"
        );
    }

    #[test]
    fn workload_selection_obeys_selected_execution_graph_route() {
        use crate::workload_search::{
            ExecutionLane, InferenceWorkloadPhase, SelectedExecutionGraph, ServiceClass,
            WorkloadProfile, WorkloadThroughputEvidence,
        };

        let selected_graph = SelectedExecutionGraph {
            profiles: Vec::new(),
            route_sequence: vec![ExecutionLane::Ane, ExecutionLane::Metal],
            fused_interleaved_metal: false,
            stateless_ane: true,
            ane_int8_planar_boundaries: false,
            mixed_precision_graphs: Vec::new(),
        };
        let throughput_evidence = vec![
            WorkloadThroughputEvidence {
                profile: WorkloadProfile {
                    phase: InferenceWorkloadPhase::Decode,
                    service_class: ServiceClass::Batch,
                    batch_size: 4,
                    concurrency: 2,
                    primary_lane: ExecutionLane::Accelerate,
                    attention_lane: ExecutionLane::Metal,
                    interleaved_metal: false,
                    stateless_shared_arena: false,
                    ane_compute_int8: false,
                    planar_input_conversion: false,
                    planar_output_conversion: false,
                },
                representation: "Fp16".into(),
                tiling_digest: "t-accel".into(),
                tokens_per_second: 500.0,
                latency_ms: 1.0,
                measured: true,
                evidence_source: "test".into(),
                execution_fingerprint: "accel-only".into(),
                projected: false,
                projection_basis: "test".into(),
                mixed_precision_graph: "fp16-only".into(),
                ..WorkloadThroughputEvidence::default()
            },
            WorkloadThroughputEvidence {
                profile: WorkloadProfile {
                    phase: InferenceWorkloadPhase::Decode,
                    service_class: ServiceClass::Batch,
                    batch_size: 4,
                    concurrency: 2,
                    primary_lane: ExecutionLane::Ane,
                    attention_lane: ExecutionLane::Metal,
                    interleaved_metal: false,
                    stateless_shared_arena: true,
                    ane_compute_int8: true,
                    planar_input_conversion: true,
                    planar_output_conversion: true,
                },
                representation: "Ternary158".into(),
                tiling_digest: "t-ane".into(),
                tokens_per_second: 120.0,
                latency_ms: 4.2,
                measured: true,
                evidence_source: "test".into(),
                execution_fingerprint: "ane-metal".into(),
                projected: false,
                projection_basis: "test".into(),
                mixed_precision_graph: "ternary-attention".into(),
                ..WorkloadThroughputEvidence::default()
            },
        ];

        let model = RuntimeModel {
            cimage_path: PathBuf::from("test.cimage"),
            source_identity: None,
            source_catalog: None,
            search_trace: None,
            legalization_report: None,
            compilation_events: None,
            manifest: CImageManifest::default(),
            tensors: HashMap::new(),
            tensor_records: HashMap::new(),
            tensor_scales: HashMap::new(),
            kernels: HashMap::new(),
            kernel_descriptors: HashMap::new(),
            uop_capture: None,
            uop_program: None,
            uop_strategy_programs: HashMap::new(),
            uop_workload_evidence: Vec::new(),
            ane_programs: HashMap::new(),
            xdna_artifacts: HashMap::new(),
            kv_compression_policy: None,
            model_manifest: None,
            native_ternary_promotion: None,
            joint_tiling_evidence: None,
            format_plan: None,
            heterogeneous_workload_evidence: Some(crate::search::HeterogeneousScheduleEvidence {
                steps: 4,
                route_sequence: vec!["ane".into(), "metal".into()],
                zero_copy_steps: 1,
                estimated_latency_ns: 1_000_000,
                residency_windows: 2,
                supports_realtime_text: true,
                supports_batched_text: true,
                supports_batched_audio: false,
                workload_profiles: Vec::new(),
                throughput_evidence: throughput_evidence.clone(),
                selected_execution_graph: selected_graph,
            }),
            execution_plan: None,
            realtime_execution_plan: None,
            tensor_offsets: HashMap::new(),
            mapped_cimage: None,
        };

        let mut runtime = UnifiedRuntime::new(model);
        runtime.requested_batch_size = Some(4);
        let selected = runtime
            .workload_profile_for_dispatch(4)
            .expect("should select profile through selected graph route");
        assert_eq!(selected.profile.primary_lane, ExecutionLane::Ane);
        assert_eq!(selected.mixed_precision_graph, "ternary-attention");
    }

    #[test]
    fn workload_selection_requires_fused_interleaved_metal_when_graph_requires_it() {
        use crate::workload_search::{
            ExecutionLane, InferenceWorkloadPhase, SelectedExecutionGraph, ServiceClass,
            WorkloadProfile, WorkloadThroughputEvidence,
        };

        let selected_graph = SelectedExecutionGraph {
            profiles: Vec::new(),
            route_sequence: vec![ExecutionLane::Metal],
            fused_interleaved_metal: true,
            stateless_ane: false,
            ane_int8_planar_boundaries: false,
            mixed_precision_graphs: Vec::new(),
        };
        let profile_non_interleaved = WorkloadThroughputEvidence {
            profile: WorkloadProfile {
                phase: InferenceWorkloadPhase::Decode,
                service_class: ServiceClass::Batch,
                batch_size: 4,
                concurrency: 2,
                primary_lane: ExecutionLane::Metal,
                attention_lane: ExecutionLane::Metal,
                interleaved_metal: false,
                stateless_shared_arena: false,
                ane_compute_int8: false,
                planar_input_conversion: false,
                planar_output_conversion: false,
            },
            representation: "Fp16".into(),
            tiling_digest: "metal".into(),
            tokens_per_second: 5.0,
            latency_ms: 1.0,
            measured: true,
            evidence_source: "test".into(),
            execution_fingerprint: "metal-non-interleaved".into(),
            projected: false,
            projection_basis: "test".into(),
            mixed_precision_graph: "fp16-only".into(),
            ..WorkloadThroughputEvidence::default()
        };
        let profile_interleaved = WorkloadThroughputEvidence {
            profile: WorkloadProfile {
                interleaved_metal: true,
                ..profile_non_interleaved.profile
            },
            representation: "Fp16".into(),
            tiling_digest: "metal".into(),
            tokens_per_second: 7.0,
            latency_ms: 1.0,
            measured: true,
            evidence_source: "test".into(),
            execution_fingerprint: "metal-interleaved".into(),
            projected: false,
            projection_basis: "test".into(),
            mixed_precision_graph: "fp16-only".into(),
            ..WorkloadThroughputEvidence::default()
        };

        let model = RuntimeModel {
            cimage_path: PathBuf::from("test.cimage"),
            source_identity: None,
            source_catalog: None,
            search_trace: None,
            legalization_report: None,
            compilation_events: None,
            manifest: CImageManifest::default(),
            tensors: HashMap::new(),
            tensor_records: HashMap::new(),
            tensor_scales: HashMap::new(),
            kernels: HashMap::new(),
            kernel_descriptors: HashMap::new(),
            uop_capture: None,
            uop_program: None,
            uop_strategy_programs: HashMap::new(),
            uop_workload_evidence: Vec::new(),
            ane_programs: HashMap::new(),
            xdna_artifacts: HashMap::new(),
            kv_compression_policy: None,
            model_manifest: None,
            native_ternary_promotion: None,
            joint_tiling_evidence: None,
            format_plan: None,
            heterogeneous_workload_evidence: Some(crate::search::HeterogeneousScheduleEvidence {
                steps: 1,
                route_sequence: vec!["metal".into()],
                zero_copy_steps: 1,
                estimated_latency_ns: 500,
                residency_windows: 1,
                supports_realtime_text: true,
                supports_batched_text: true,
                supports_batched_audio: false,
                workload_profiles: Vec::new(),
                throughput_evidence: vec![profile_non_interleaved, profile_interleaved],
                selected_execution_graph: selected_graph,
            }),
            execution_plan: None,
            realtime_execution_plan: None,
            tensor_offsets: HashMap::new(),
            mapped_cimage: None,
        };

        let runtime = UnifiedRuntime::new(model);
        let selected = runtime
            .workload_profile_for_dispatch(4)
            .expect("interleaved requirement should force valid candidate");
        assert!(selected.profile.interleaved_metal);
        assert_eq!(selected.tokens_per_second, 7.0);
    }

    #[test]
    fn workload_selection_filters_to_shared_arena_and_ane_planar_boundaries() {
        use crate::workload_search::{
            ExecutionLane, InferenceWorkloadPhase, SelectedExecutionGraph, ServiceClass,
            WorkloadProfile, WorkloadThroughputEvidence,
        };

        let selected_graph = SelectedExecutionGraph {
            profiles: Vec::new(),
            route_sequence: vec![ExecutionLane::Ane],
            fused_interleaved_metal: false,
            stateless_ane: true,
            ane_int8_planar_boundaries: true,
            mixed_precision_graphs: Vec::new(),
        };

        let valid_profile = WorkloadThroughputEvidence {
            profile: WorkloadProfile {
                phase: InferenceWorkloadPhase::Decode,
                service_class: ServiceClass::Batch,
                batch_size: 4,
                concurrency: 2,
                primary_lane: ExecutionLane::Ane,
                attention_lane: ExecutionLane::Ane,
                interleaved_metal: false,
                stateless_shared_arena: true,
                ane_compute_int8: true,
                planar_input_conversion: true,
                planar_output_conversion: true,
            },
            representation: "Int8".into(),
            tiling_digest: "ane".into(),
            tokens_per_second: 3.0,
            latency_ms: 1.0,
            measured: true,
            evidence_source: "test".into(),
            execution_fingerprint: "ane-valid".into(),
            projected: false,
            projection_basis: "test".into(),
            mixed_precision_graph: "int8-only".into(),
            ..WorkloadThroughputEvidence::default()
        };
        let missing_planar = WorkloadThroughputEvidence {
            profile: WorkloadProfile {
                interleaved_metal: false,
                stateless_shared_arena: true,
                ane_compute_int8: true,
                planar_input_conversion: false,
                planar_output_conversion: false,
                ..valid_profile.profile
            },
            representation: "Int8".into(),
            tiling_digest: "ane".into(),
            tokens_per_second: 6.0,
            latency_ms: 1.0,
            measured: true,
            evidence_source: "test".into(),
            execution_fingerprint: "ane-missing-planar".into(),
            projected: false,
            projection_basis: "test".into(),
            mixed_precision_graph: "int8-only".into(),
            ..WorkloadThroughputEvidence::default()
        };
        let invalid_profile = WorkloadThroughputEvidence {
            profile: WorkloadProfile {
                stateless_shared_arena: false,
                ane_compute_int8: false,
                ..valid_profile.profile
            },
            representation: "Int8".into(),
            tiling_digest: "ane".into(),
            tokens_per_second: 99.0,
            latency_ms: 1.0,
            measured: true,
            evidence_source: "test".into(),
            execution_fingerprint: "ane-invalid".into(),
            projected: false,
            projection_basis: "test".into(),
            mixed_precision_graph: "int8-only".into(),
            ..WorkloadThroughputEvidence::default()
        };

        let model = RuntimeModel {
            cimage_path: PathBuf::from("test.cimage"),
            source_identity: None,
            source_catalog: None,
            search_trace: None,
            legalization_report: None,
            compilation_events: None,
            manifest: CImageManifest::default(),
            tensors: HashMap::new(),
            tensor_records: HashMap::new(),
            tensor_scales: HashMap::new(),
            kernels: HashMap::new(),
            kernel_descriptors: HashMap::new(),
            uop_capture: None,
            uop_program: None,
            uop_strategy_programs: HashMap::new(),
            uop_workload_evidence: Vec::new(),
            ane_programs: HashMap::new(),
            xdna_artifacts: HashMap::new(),
            kv_compression_policy: None,
            model_manifest: None,
            native_ternary_promotion: None,
            joint_tiling_evidence: None,
            format_plan: None,
            heterogeneous_workload_evidence: Some(crate::search::HeterogeneousScheduleEvidence {
                steps: 1,
                route_sequence: vec!["ane".into()],
                zero_copy_steps: 1,
                estimated_latency_ns: 900,
                residency_windows: 1,
                supports_realtime_text: true,
                supports_batched_text: true,
                supports_batched_audio: false,
                workload_profiles: Vec::new(),
                throughput_evidence: vec![invalid_profile, missing_planar, valid_profile],
                selected_execution_graph: selected_graph,
            }),
            execution_plan: None,
            realtime_execution_plan: None,
            tensor_offsets: HashMap::new(),
            mapped_cimage: None,
        };

        let runtime = UnifiedRuntime::new(model);
        let selected = runtime
            .workload_profile_for_dispatch(4)
            .expect("should select only valid shared-arena + planar boundary profile");
        assert!(selected.profile.stateless_shared_arena);
        assert!(selected.profile.ane_compute_int8);
        assert!(selected.profile.planar_input_conversion);
        assert!(selected.profile.planar_output_conversion);
    }

    #[test]
    fn validation_load_binds_native_ternary_scales() {
        let path = std::env::temp_dir().join(format!(
            "prism_runtime_native_scales_{}.cimage",
            std::process::id()
        ));
        let mut writer = crate::cimage::CImageWriter::new(&path).expect("create CImage");
        writer
            .append_native_ternary_with_scales(
                "weights",
                &[0, 1, 2, 0],
                &[0, 0, 128, 63],
                1,
                4,
                crate::cimage::TensorType::Ternary158,
                crate::cimage::TernaryDescriptor::legacy_for_type(
                    &crate::cimage::TensorType::Ternary158,
                )
                .unwrap(),
            )
            .expect("append native payload");
        writer.finalize().expect("finalize CImage");

        let model = RuntimeModel::load_for_validation(&path).expect("load validation model");
        assert_eq!(model.get_tensor("weights"), Some(&[0, 1, 2, 0][..]));
        assert_eq!(
            model.get_tensor_scales("weights"),
            Some(&[0, 0, 128, 63][..])
        );
        assert!(model.get_tensor_scales("missing").is_none());
        let _ = std::fs::remove_file(path);
    }

    /// Construct a [`UnifiedRuntime`] from a default model and verify
    /// default execution mode and missing backend.
    #[test]
    fn test_unified_runtime_new() {
        let manifest = CImageManifest::default();
        let model = RuntimeModel {
            cimage_path: PathBuf::from("test.cimage"),
            source_identity: None,
            source_catalog: None,
            search_trace: None,
            legalization_report: None,
            compilation_events: None,
            manifest,
            tensors: HashMap::new(),
            tensor_records: HashMap::new(),
            tensor_scales: HashMap::new(),
            kernels: HashMap::new(),
            kernel_descriptors: HashMap::new(),
            uop_capture: None,
            uop_program: None,
            uop_strategy_programs: HashMap::new(),
            uop_workload_evidence: Vec::new(),
            ane_programs: HashMap::new(),
            xdna_artifacts: HashMap::new(),
            kv_compression_policy: None,
            model_manifest: None,
            native_ternary_promotion: None,
            joint_tiling_evidence: None,
            format_plan: None,
            heterogeneous_workload_evidence: None,
            execution_plan: None,
            realtime_execution_plan: None,
            tensor_offsets: HashMap::new(),
            mapped_cimage: None,
        };

        let mut rt = UnifiedRuntime::new(model);

        // Default mode is batch.
        assert_eq!(rt.mode, ExecutionMode::Batch);

        // No backend attached by default.
        assert!(rt.backend.is_none());

        // No KV cache until prefill.
        assert!(rt.kv_cache.is_none());

        // Stub methods should return errors.
        assert!(rt.run_batch(&[0, 1, 2]).is_err());
        assert!(rt.run_prefill(&[0, 1, 2]).is_err());
        assert!(rt.run_decode().is_err());

        // Reset should be a no-op without KV cache.
        rt.reset_kv_cache();
        assert!(rt.kv_cache.is_none());
        assert_eq!(rt.mode, ExecutionMode::Batch);
    }

    /// Verify that [`RuntimeModel::load`] returns an error (stub).
    #[test]
    fn test_runtime_model_load_stub() {
        let result = RuntimeModel::load(Path::new("nonexistent.cimage"));
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, RuntimeError::InvalidCImage(_)));
    }

    #[test]
    fn replay_aot_covers_streamed_text_and_audio_workloads() {
        use prism_spatial_ir::execution_plan::{
            FusedScheduleStep, PlanBackend, ResidencyWindow, ResidencyWorkload,
        };
        use prism_spatial_ir::{BufferStorage, HeterogeneousExecutor, ResolvedStep};

        struct Executor {
            events: Vec<String>,
        }

        impl HeterogeneousExecutor for Executor {
            fn ensure_residency(&mut self, window_id: usize) -> Result<(), String> {
                self.events.push(format!("resident:{window_id}"));
                Ok(())
            }

            fn dispatch(
                &mut self,
                backend: PlanBackend,
                step: &FusedScheduleStep,
            ) -> Result<(), String> {
                self.events
                    .push(format!("route:{backend:?}:{}", step.step_id));
                Ok(())
            }

            fn dispatch_resolved(
                &mut self,
                backend: PlanBackend,
                resolved: &mut ResolvedStep<'_>,
            ) -> Result<(), String> {
                assert!(resolved
                    .inputs
                    .iter()
                    .chain(resolved.outputs.iter())
                    .all(|buffer| matches!(buffer.storage, BufferStorage::RuntimeOwned)));
                self.events
                    .push(format!("resolved:{backend:?}:{}", resolved.step.step_id));
                Ok(())
            }

            fn synchronize(&mut self, step: &FusedScheduleStep) -> Result<(), String> {
                self.events.push(format!("sync:{}", step.step_id));
                Ok(())
            }
        }

        let plan = prism_spatial_ir::execution_plan::ExecutionPlan {
            mode: prism_spatial_ir::execution_plan::ExecutionMode::Batch,
            schedule: vec![],
            batch_size: 32,
            persistent_cache: false,
            dispatch_policy: Default::default(),
            device_island: Default::default(),
            fused_steps: vec![FusedScheduleStep {
                step_id: 0,
                model_id: None,
                node_ids: vec![],
                backend: PlanBackend::AneMatrix,
                depends_on: vec![],
                input_region: "ane-memory".into(),
                output_region: "ane-memory".into(),
                zero_copy: true,
                estimated_latency_ns: 10,
                input_tensors: vec![],
                output_tensors: vec![],
                dispatch_geometry: [1, 1, 1],
                fusion_strategy: None,
            }],
            residency_windows: vec![ResidencyWindow {
                window_id: 7,
                model_bytes: 4096,
                required_workloads: vec![
                    ResidencyWorkload::RealtimeText,
                    ResidencyWorkload::BatchedText,
                    ResidencyWorkload::BatchedAudio,
                ],
                resident_devices: vec!["ane-memory".into(), "unified-memory".into()],
                prefetch_step: Some(0),
                eviction_step: None,
            }],
            fusion_evaluations: vec![],
            workload_evaluations: vec![],
        };
        let model = RuntimeModel {
            cimage_path: PathBuf::from("test.cimage"),
            source_identity: None,
            source_catalog: None,
            search_trace: None,
            legalization_report: None,
            compilation_events: None,
            manifest: CImageManifest::default(),
            tensors: HashMap::new(),
            tensor_records: HashMap::new(),
            tensor_scales: HashMap::new(),
            kernels: HashMap::new(),
            kernel_descriptors: HashMap::new(),
            uop_capture: None,
            uop_program: None,
            uop_strategy_programs: HashMap::new(),
            uop_workload_evidence: Vec::new(),
            ane_programs: HashMap::new(),
            xdna_artifacts: HashMap::new(),
            kv_compression_policy: None,
            model_manifest: None,
            native_ternary_promotion: None,
            joint_tiling_evidence: None,
            format_plan: None,
            heterogeneous_workload_evidence: None,
            execution_plan: Some(plan),
            realtime_execution_plan: None,
            tensor_offsets: HashMap::new(),
            mapped_cimage: None,
        };
        let runtime = UnifiedRuntime::new(model);
        let mut executor = Executor { events: vec![] };
        let receipt = runtime.replay_aot(&mut executor).unwrap();
        assert_eq!(receipt.steps.len(), 1);
        assert_eq!(receipt.model_residency_windows, 1);
        assert_eq!(
            executor.events,
            vec!["resident:7", "resolved:AneMatrix:0", "sync:0"]
        );
    }

    /// Verify that the free-standing stub functions return errors.
    #[test]
    fn test_stub_functions() {
        let manifest = CImageManifest::default();
        let model = RuntimeModel {
            cimage_path: PathBuf::from("test.cimage"),
            source_identity: None,
            source_catalog: None,
            search_trace: None,
            legalization_report: None,
            compilation_events: None,
            manifest,
            tensors: HashMap::new(),
            tensor_records: HashMap::new(),
            tensor_scales: HashMap::new(),
            kernels: HashMap::new(),
            kernel_descriptors: HashMap::new(),
            uop_capture: None,
            uop_program: None,
            uop_strategy_programs: HashMap::new(),
            uop_workload_evidence: Vec::new(),
            ane_programs: HashMap::new(),
            xdna_artifacts: HashMap::new(),
            kv_compression_policy: None,
            model_manifest: None,
            native_ternary_promotion: None,
            joint_tiling_evidence: None,
            format_plan: None,
            heterogeneous_workload_evidence: None,
            execution_plan: None,
            realtime_execution_plan: None,
            tensor_offsets: HashMap::new(),
            mapped_cimage: None,
        };

        let ref_result = cpu_reference_inference(&model, &[0, 1]);
        assert!(ref_result.is_err());

        let cert_result = certify_inference(
            &model,
            &[0, 1],
            // Use a concrete empty backend — we stub it here as a
            // reference, but in Phase 9 a real backend will be wired.
            &MockBackend,
            0.01,
        );
        assert!(cert_result.is_err());
    }

    /// Dummy backend for testing — returns errors for every method.
    struct MockBackend;

    impl KernelBackend for MockBackend {
        fn validate(
            &self,
            _descriptor: &prism_ecs_kernel::KernelDescriptor,
        ) -> Result<(), prism_ecs_kernel::KernelError> {
            Err(prism_ecs_kernel::KernelError::UnsupportedBackend(
                "mock".into(),
            ))
        }

        fn compile(
            &self,
            _request: &prism_ecs_kernel::KernelCompileRequest,
        ) -> Result<prism_ecs_kernel::KernelArtifact, prism_ecs_kernel::KernelError> {
            Err(prism_ecs_kernel::KernelError::UnsupportedBackend(
                "mock".into(),
            ))
        }

        fn dispatch(
            &self,
            _request: &prism_ecs_kernel::KernelDispatchRequest,
        ) -> Result<prism_ecs_kernel::KernelOutput, prism_ecs_kernel::KernelError> {
            Err(prism_ecs_kernel::KernelError::UnsupportedBackend(
                "mock".into(),
            ))
        }

        fn measure(
            &self,
            _request: &prism_ecs_kernel::KernelMeasurementRequest,
        ) -> Result<prism_ecs_kernel::KernelMeasurement, prism_ecs_kernel::KernelError> {
            Err(prism_ecs_kernel::KernelError::UnsupportedBackend(
                "mock".into(),
            ))
        }

        fn name(&self) -> &str {
            "mock"
        }
    }

