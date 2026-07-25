//! CanonicalCompiler — format-independent pipeline orchestrator.
//!
//! Wires source ingestion, graph building, evolutionary search, legalization,
//! CImage emission, and forensic receipt into one sequential pipeline.
//! Every source format follows the same path.

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use prism_ecs_core::identity::CompilerIdentity;
use prism_ecs_core::identity::TensorProvider;
use prism_ecs_core::world::World;
use prism_ecs_core::Entity;
use prism_ecs_ir::evolution::EvolutionRuntime;
#[cfg(test)]
use prism_ecs_kernel::BackendKind;
#[cfg(test)]
use prism_ecs_source::SourceIdentity;
#[cfg(test)]
use prism_ecs_source::TensorDataProvider;
use prism_ecs_source::{CanonicalSource, CanonicalSourceAdapter, SourceError};

use crate::cimage::{
    MoeTensorDescriptor, TensorPayloadEntry, TensorType, UniversalCImageWriter,
    VisionTensorDescriptor,
};
use crate::compilation_entity::CompilationEntity;
use crate::compilation_systems::*;
use crate::ecs::{
    system_build_graph, system_build_receipt, system_certify, system_detect_source,
    system_emit_cimage, system_generate_kernels, system_legalize, system_run_search,
    CompilationReceipt, CompilationSession, CurrentSource, SessionHandle, SessionStatus,
    SourceAdapterList,
};
use crate::forensic::FileEventSink;
use crate::graph::CanonicalGraphBuilder;
use crate::legalize::CompilerLegalizer;
use crate::observability::{EcsCorrelation, EcsStateStream};
use crate::search::SearchCoordinator;
use crate::{
    CompilationEvent, CompilationEventSink, CompilationStage, CompileConfig, CompileError,
    CompileReceipt, CompileResult, CompileStatus, EventKind, SearchConfig, StageResult,
    VecEventSink,
};

/// Compile one ECS operation directly into a native XDNA CImage artifact.
/// This entry point intentionally bypasses generic GPU kernel wrappers: the
/// AMD runtime owns lowering, artifact validation, and native envelope codec.
pub fn compile_ecs_op_to_xdna_cimage(
    world: &World,
    root_op: Entity,
    output_path: &Path,
) -> Result<(), String> {
    let executable = prism_amd_npu_runtime::compile_amd_npu(
        world,
        root_op,
        prism_ecs_ir::backend_dispatch::HalFormat::AmdNpu,
    )?;
    let artifact = prism_amd_npu_runtime::XdnaArtifact::decode_hex_envelope(&executable.source)?;
    let generation = format!("{:?}", artifact.program.topology.generation);
    let payload = artifact.encode()?;
    let mut writer = UniversalCImageWriter::new(output_path);
    writer.set_model_capabilities(["native-xdna", "persistent-npu"]);
    writer.add_xdna_artifact("main", &payload, "prism-xdna-v1", generation)?;
    writer
        .finalize()
        .map_err(|error| format!("finalize XDNA CImage: {error}"))
}

/// Compile one admitted Tile640 candidate into a stateless int8 ANE program
/// and emit its packed `.mlmodelc` plus ABI record into a CImage artifact.
#[cfg(all(feature = "ane", target_os = "macos"))]
pub fn compile_int8_ane_tile_to_cimage(
    output_path: &Path,
    program_name: &str,
    input_width: usize,
    output_width: usize,
) -> Result<(), String> {
    if input_width == 0 || output_width == 0 {
        return Err("ANE tile dimensions must be non-zero".into());
    }
    let work_dir = std::env::temp_dir().join(format!("prism-ane-compile-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&work_dir).map_err(|e| format!("create ANE work directory: {e}"))?;
    let result = (|| {
        let modelc = prism_ane::ternary_tile::compile_stateless_ternary_tile(
            prism_ane::ternary_tile::StatelessTernaryTileSpec {
                input_width,
                output_width,
            },
            &work_dir,
        )?;
        let payload = prism_ane::pack_mlmodelc(&modelc)?;
        crate::cimage::emit_int8_ane_program(
            output_path,
            program_name,
            &payload,
            "activation",
            "ternary_weights",
            "matmul_int8_2",
        )
    })();
    let _ = std::fs::remove_dir_all(&work_dir);
    result
}

#[cfg(not(all(feature = "ane", target_os = "macos")))]
pub fn compile_int8_ane_tile_to_cimage(
    _output_path: &Path,
    _program_name: &str,
    _input_width: usize,
    _output_width: usize,
) -> Result<(), String> {
    Err("int8 ANE tile compilation requires macOS and the `ane` feature".into())
}

/// Auto-detect the source format and select the correct adapter.
pub fn detect_source(path: &Path) -> Result<Box<dyn CanonicalSourceAdapter>, SourceError> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    if ext == "gguf" || ext == "GGUF" {
        return Ok(Box::new(prism_ecs_source::gguf_adapter::GgufAdapter));
    }
    if ext == "onnx" || ext == "ONNX" {
        return Ok(Box::new(prism_ecs_source::onnx_adapter::OnnxAdapter));
    }

    // Directory-based formats
    if path.is_dir() {
        let has_safetensors = std::fs::read_dir(path)
            .ok()
            .map(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .any(|e| e.path().extension() == Some(std::ffi::OsStr::new("safetensors")))
            })
            .unwrap_or(false);

        if has_safetensors {
            let config_path = path.join("config.json");
            if config_path.exists() {
                if let Ok(config_str) = std::fs::read_to_string(&config_path) {
                    if let Ok(config) = serde_json::from_str::<serde_json::Value>(&config_str) {
                        if config.get("model_type").and_then(|v| v.as_str()).is_some() {
                            return Ok(Box::new(prism_ecs_source::mlx_adapter::MlxAdapter));
                        }
                    }
                }
            }
            return Ok(Box::new(
                prism_ecs_source::safetensors_adapter::SafetensorsAdapter,
            ));
        }
    }

    Err(SourceError::UnsupportedFormat(format!(
        "Unrecognized model format: {}",
        path.display()
    )))
}

pub(crate) fn annotate_model_tensor(
    writer: &mut UniversalCImageWriter,
    role: crate::TensorRole,
    expert_count: Option<u32>,
    tensor_name: &str,
) -> Result<(), String> {
    match role {
        crate::TensorRole::Router { layer } => writer.set_moe_tensor(
            tensor_name,
            MoeTensorDescriptor {
                layer: layer as u32,
                expert: None,
                expert_count: None,
                role: "router".into(),
                component: Some("router".into()),
            },
        ),
        crate::TensorRole::RoutedExpert {
            layer,
            expert,
            component,
        } => writer.set_moe_tensor(
            tensor_name,
            MoeTensorDescriptor {
                layer: layer as u32,
                expert: Some(expert as u32),
                expert_count: None,
                role: "routed_expert".into(),
                component: Some(component),
            },
        ),
        crate::TensorRole::RoutedExpertBank { layer, component } => writer.set_moe_tensor(
            tensor_name,
            MoeTensorDescriptor {
                layer: layer as u32,
                expert: None,
                expert_count,
                role: "routed_expert_bank".into(),
                component: Some(component),
            },
        ),
        crate::TensorRole::SharedExpert { layer, component } => writer.set_moe_tensor(
            tensor_name,
            MoeTensorDescriptor {
                layer: layer as u32,
                expert: None,
                expert_count: None,
                role: "shared_expert".into(),
                component: Some(component),
            },
        ),
        crate::TensorRole::Vision { component } => {
            writer.set_vision_tensor(tensor_name, VisionTensorDescriptor { component })
        }
        _ => Ok(()),
    }
}

fn annotate_qwen_tensor(
    writer: &mut UniversalCImageWriter,
    config: &crate::qwen3_6_moe::Qwen36Config,
    tensor_name: &str,
) -> Result<(), String> {
    let descriptor = crate::qwen3_6_moe::classify_qwen36_tensor(tensor_name);
    annotate_model_tensor(
        writer,
        descriptor.role,
        Some(config.num_experts as u32),
        tensor_name,
    )
}

fn annotate_tensor_for_model(
    writer: &mut UniversalCImageWriter,
    config: Option<&crate::qwen3_6_moe::Qwen36Config>,
    tensor_name: &str,
) -> Result<(), String> {
    if let Some(config) = config {
        annotate_qwen_tensor(writer, config, tensor_name)
    } else {
        annotate_model_tensor(
            writer,
            crate::classify_tensor(tensor_name).role,
            None,
            tensor_name,
        )
    }
}

/// Run the full compilation pipeline on a given source path.
fn target_backend_from_env() -> prism_ecs_kernel::BackendKind {
    target_backend_from_name(std::env::var("PRISM_TARGET_BACKEND").ok().as_deref())
}

fn target_backend_from_name(name: Option<&str>) -> prism_ecs_kernel::BackendKind {
    match name {
        Some("amd-npu") => prism_ecs_kernel::BackendKind::AmdNpu,
        _ => prism_ecs_kernel::BackendKind::Metal,
    }
}

#[cfg(test)]
mod backend_selection_tests {
    #[test]
    fn explicit_amd_npu_backend_is_representable() {
        assert_eq!(
            crate::compiler::target_backend_from_name(Some("amd-npu")),
            prism_ecs_kernel::BackendKind::AmdNpu
        );
    }
}

pub fn compile_path(
    source_path: &Path,
    output_path: &Path,
    production_mode: bool,
) -> Result<CompileResult, CompileError> {
    compile_path_with_backend(
        source_path,
        output_path,
        production_mode,
        target_backend_from_env(),
    )
}

/// Source-compatible names retained at the canonical compiler boundary.
pub fn compile_gguf_compat(source: &Path, output: &Path) -> Result<(), String> {
    compile_path_with_backend(source, output, true, prism_ecs_kernel::BackendKind::Metal)
        .map(|_| ())
        .map_err(|error| error.to_string())
}

pub fn compile_to_cimage_compat(source: &Path, output: &Path) -> Result<(), String> {
    compile_gguf_compat(source, output)
}

pub fn compile_with_autodetect(source: &Path, output: &Path) -> Result<(), String> {
    compile_gguf_compat(source, output)
}

/// Compile a model while explicitly selecting the native target backend.
/// This is the programmatic entry point for AMD XDNA deployment; the legacy
/// `compile_path` wrapper retains Metal as its default.
pub fn compile_path_with_backend(
    source_path: &Path,
    output_path: &Path,
    production_mode: bool,
    backend: prism_ecs_kernel::BackendKind,
) -> Result<CompileResult, CompileError> {
    let correlation = EcsCorrelation::new();
    let stream = EcsStateStream::with_jsonl(
        &output_path.with_extension("cimage.events.jsonl"),
        &correlation,
    )
    .map_err(CompileError::SourceIngestionFailed)?;
    stream.stage(
        &correlation,
        CompilationStage::SourceDetection,
        EventKind::StageStarted,
        "compilation started",
        std::collections::BTreeMap::new(),
    );
    let mut compiler = crate::CanonicalCompiler::new(CompileConfig {
        production_mode,
        max_candidates: 100,
        max_generations: 5,
        max_search_time_ms: 300000,
        target_backends: vec![backend],
        calibration_policy: crate::CalibrationPolicy::None,
        validation_policy: if production_mode {
            crate::ValidationPolicy::Production
        } else {
            crate::ValidationPolicy::Structural
        },
        enable_search: true,
        enable_legalization: true,
        enable_kernel_gen: production_mode,
    });

    let event_sink = FileEventSink::new(None);
    compiler.event_sink = Some(Box::new(event_sink));
    compiler.output_path = Some(output_path.to_path_buf());
    compiler.source_path = Some(source_path.to_path_buf());
    if source_path.is_dir() && source_path.join("config.json").is_file() {
        let config_path = source_path.join("config.json");
        let raw = std::fs::read_to_string(&config_path)
            .map_err(|e| CompileError::SourceIngestionFailed(e.to_string()))?;
        // Model-family detection is centralized behind ModelAdapter.  The
        // legacy Qwen config is retained only as the emission compatibility
        // payload; search and tensor-role consumers can use the adapter API.
        let model_adapter = crate::adapter_for_model_dir(source_path).ok();
        let qwen36_config = serde_json::from_str::<serde_json::Value>(&raw)
            .ok()
            .is_some_and(|value| {
                value
                    .get("model_type")
                    .and_then(|model_type| model_type.as_str())
                    .is_some_and(|model_type| {
                        model_type == "qwen3_5_moe" || model_type == "qwen3_6_moe"
                    })
            })
            .then(|| crate::qwen3_6_moe::Qwen36Config::from_json_str(&raw))
            .transpose()
            .map_err(CompileError::SourceIngestionFailed)?;
        compiler.qwen36_config = qwen36_config;
        let shard_cache = source_path.join(".prism-shard-cache");
        let ternary_cache_complete = shard_cache.join("ternary-manifest.json").is_file();
        if ternary_cache_complete {
            eprintln!(
                "[prism:compile] progressive ternary cache complete: reusing existing payloads"
            );
        } else if let Ok(provider) =
            prism_ecs_quantization::safetensors_provider::SafeTensorProvider::new(source_path)
        {
            eprintln!("[prism:compile] validating safetensors inventory...");
            if let Some(config) = compiler.qwen36_config.as_ref() {
                let summaries = provider
                    .shard_summaries()
                    .map_err(CompileError::SourceIngestionFailed)?;
                let names = summaries
                    .iter()
                    .flat_map(|summary| summary.tensor_names.iter().map(String::as_str));
                config
                    .validate_tensor_inventory(names)
                    .map_err(CompileError::SourceIngestionFailed)?;
            } else if let Some(adapter) = model_adapter.as_ref() {
                let summaries = provider
                    .shard_summaries()
                    .map_err(CompileError::SourceIngestionFailed)?;
                let names: Vec<String> = summaries
                    .iter()
                    .flat_map(|summary| summary.tensor_names.iter().cloned())
                    .collect();
                adapter
                    .validate_inventory(&names)
                    .map_err(CompileError::SourceIngestionFailed)?;
            }
            let manifest_path = shard_cache.join("ternary-manifest.json");
            let cached_records = std::fs::read(&manifest_path)
                .ok()
                .and_then(|bytes| {
                    serde_json::from_slice::<
                        Vec<prism_ecs_quantization::safetensors_provider::TernaryPreprocessRecord>,
                    >(&bytes)
                    .ok()
                })
                .filter(|records| !records.is_empty());
            if let Some(records) = cached_records {
                eprintln!(
                    "[prism:compile] progressive ternary cache complete: {} records (reusing existing payloads)",
                    records.len()
                );
            } else {
                let shard_records = provider
                    .write_preprocess_cache(&shard_cache)
                    .map_err(CompileError::SourceIngestionFailed)?;
                eprintln!(
                    "[prism:compile] shard preprocessing complete: {} records",
                    shard_records.len()
                );
                // This is resumable: rerunning it fills missing tensor records and
                // skips payloads that already have valid cache files.
                provider
                    .write_ternary_preprocess_cache_from_records(&shard_cache, shard_records)
                    .map_err(CompileError::SourceIngestionFailed)?;
                eprintln!("[prism:compile] progressive ternary cache complete");
            }
        }
        compiler.model_adapter = model_adapter.map(std::sync::Arc::from);
    }

    let adapter = detect_source(source_path)
        .map_err(|e| CompileError::SourceDetectionFailed(e.to_string()))?;
    eprintln!("[prism:compile] opening canonical source...");
    let source = adapter
        .open(source_path)
        .map_err(|e| CompileError::SourceIngestionFailed(e.to_string()))?;
    eprintln!(
        "[prism:compile] source opened: {} tensors",
        source.catalog.len()
    );
    if let Some(config) = compiler.qwen36_config.as_ref() {
        let names = source.catalog.iter().map(|tensor| tensor.name.as_str());
        config
            .validate_tensor_inventory(names)
            .map_err(CompileError::SourceIngestionFailed)?;
    }

    eprintln!("[prism:compile] entering ECS compilation stages...");
    stream.stage(
        &correlation,
        CompilationStage::SourceIngestion,
        EventKind::StageCompleted,
        "source opened",
        std::collections::BTreeMap::from([(
            String::from("tensor_count"),
            serde_json::json!(source.catalog.len()),
        )]),
    );
    let result = compile_source(&mut compiler, source);
    let mut state = std::collections::BTreeMap::new();
    state.insert(
        "status".into(),
        serde_json::json!(if result.is_ok() {
            "completed"
        } else {
            "failed"
        }),
    );
    stream.stage(
        &correlation,
        CompilationStage::ReceiptBuild,
        if result.is_ok() {
            EventKind::StageCompleted
        } else {
            EventKind::Error
        },
        "compilation terminal state",
        state,
    );
    result
}

/// Run the full compilation pipeline on an already-opened CanonicalSource.
pub fn compile_source(
    compiler: &mut crate::CanonicalCompiler,
    source: CanonicalSource,
) -> Result<CompileResult, CompileError> {
    let request_id = uuid::Uuid::new_v4();
    let started_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let mut stages: Vec<StageResult> = Vec::new();
    let mut events: Vec<CompilationEvent> = Vec::new();
    let output_path = compiler
        .output_path
        .clone()
        .unwrap_or_else(|| std::path::PathBuf::from("output.cimage"));

    let emit_event = |events: &mut Vec<CompilationEvent>,
                      sequence: u64,
                      phase: CompilationStage,
                      event_type: EventKind,
                      detail: &str,
                      duration_ms: u64| {
        events.push(CompilationEvent {
            sequence,
            timestamp: chrono::Utc::now(),
            phase,
            event_type,
            entity_id: None,
            duration_ms,
            detail: detail.to_string(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            digests: Vec::new(),
            status: "completed".to_string(),
        });
    };

    // Stage 1: Graph construction
    let graph_t0 = std::time::Instant::now();
    let graph_result = match compiler.qwen36_config.as_ref() {
        Some(config) => CanonicalGraphBuilder::build_qwen36(&source, config),
        None => CanonicalGraphBuilder::build(&source),
    }
    .map_err(|e| CompileError::GraphBuildFailed(e.to_string()))?;
    let graph_duration = graph_t0.elapsed().as_millis() as u64;
    emit_event(
        &mut events,
        0,
        CompilationStage::GraphConstruction,
        EventKind::StageCompleted,
        &format!(
            "graph built: {} nodes, digest={}",
            graph_result.graph.node_count(),
            graph_result.graph_digest
        ),
        graph_duration,
    );
    stages.push(StageResult {
        stage: CompilationStage::GraphConstruction,
        success: true,
        duration_ms: graph_duration,
        error: None,
    });

    // Stage 2: Search (if enabled)
    let mut selected_format_plan: Option<prism_ecs_ir::evolution::compile_plan::FormatPlan> = None;
    let mut selected_heterogeneous_workload_evidence: Option<
        crate::search::HeterogeneousScheduleEvidence,
    > = None;
    let mut search_trace = None;
    let mut selection_receipt = None;
    let (candidate_count, generations_count) = if compiler.config.enable_search {
        let search_t0 = std::time::Instant::now();
        let search_config = SearchConfig {
            max_generations: compiler.config.max_generations,
            population_size: (source.catalog.len() * 3).max(20).min(100) as u32,
            mutation_rate: 0.3,
            crossover_rate: 0.7,
            tournament_size: 3,
            elite_count: 4,
            early_stop_generations: 10,
            production_mode: compiler.config.production_mode,
            surrogate_measurement_fraction: 0.2,
            min_quality: None,
            max_p99_latency_ms: None,
            max_peak_memory_bytes: None,
        };
        let mut coordinator =
            SearchCoordinator::new(search_config).with_runtime(EvolutionRuntime::global());
        #[cfg(feature = "phase4_evaluation")]
        let mapped_evaluator;
        #[cfg(feature = "phase4_evaluation")]
        let evaluator_ref = if let Some(evaluator) = compiler.evaluator.as_deref() {
            Some(evaluator)
        } else if compiler.model_adapter.is_some() {
            mapped_evaluator = crate::evaluator::MappedTensorEvaluationStrategy::new(
                compiler.source_path.clone().ok_or_else(|| {
                    CompileError::SearchFailed("model behavioral search requires model path".into())
                })?,
            );
            Some(&mapped_evaluator as &dyn crate::search::EvaluationStrategy)
        } else {
            None
        };
        let search_result = coordinator
            .run_search(
                &source,
                &graph_result.graph,
                #[cfg(feature = "phase4_evaluation")]
                evaluator_ref,
                #[cfg(not(feature = "phase4_evaluation"))]
                None::<&dyn crate::search::EvaluationStrategy>,
                compiler.config.production_mode,
            )
            .map_err(|e| CompileError::SearchFailed(e.to_string()))?;
        search_trace = Some(search_result.trace.clone());
        selected_heterogeneous_workload_evidence = search_result.heterogeneous_schedule.clone();
        selection_receipt = Some(search_result.selection_receipt.clone());
        selected_format_plan = search_result
            .format_plan
            .as_deref()
            .and_then(|json| serde_json::from_str(json).ok());
        if let (Some(model_dir), Some(plan)) =
            (compiler.source_path.as_ref(), selected_format_plan.as_mut())
        {
            if let Err(error) = crate::representation_cache::record_promoted_profiles(
                model_dir,
                plan,
                &source.catalog,
            ) {
                eprintln!("[prism:compile] representation reuse index unavailable: {error}");
            }
            let mut family_policies =
                crate::representation_cache::cluster_tensor_families(&source.catalog);
            for policy in family_policies.values_mut() {
                if let Some(format) = policy
                    .members
                    .iter()
                    .find_map(|name| plan.per_tensor.get(name))
                {
                    policy.format = format!("{format:?}");
                }
            }
            if let Ok(probe) =
                crate::evaluator::MappedTensorBehavioralProbe::try_new_real(model_dir.clone())
            {
                if let Err(error) = probe.evaluate_family_canaries(
                    &mut family_policies,
                    &prism_ecs_ir::evolution::progressive::TernaryAdmissionLimits::default(),
                ) {
                    eprintln!("[prism:compile] family canary verification unavailable: {error}");
                }
            }
            crate::representation_cache::apply_family_formats(
                plan,
                &source.catalog,
                &family_policies,
            );
            if let Err(error) =
                crate::representation_cache::persist_family_policy_map(model_dir, &family_policies)
            {
                eprintln!("[prism:compile] tensor family policy unavailable: {error}");
            }
        }
        let search_duration = search_t0.elapsed().as_millis() as u64;
        emit_event(
            &mut events,
            1,
            CompilationStage::EvolutionarySearch,
            EventKind::StageCompleted,
            &format!(
                "search complete: {} candidates, {} generations, trace_digest={}, route={}, fused_schedule={:?}",
                search_result.candidates_evaluated,
                search_result.generations_completed,
                search_result.trace.trace_digest,
                search_result.evaluation_route,
                search_result.heterogeneous_schedule,
            ),
            search_duration,
        );
        stages.push(StageResult {
            stage: CompilationStage::EvolutionarySearch,
            success: true,
            duration_ms: search_duration,
            error: None,
        });
        (
            search_result.candidates_evaluated,
            search_result.generations_completed,
        )
    } else {
        (0, 0)
    };

    // Stage 3: Legalization
    let legalize_t0 = std::time::Instant::now();
    let is_amd_npu = compiler
        .config
        .target_backends
        .first()
        .is_some_and(|backend| *backend == prism_ecs_kernel::BackendKind::AmdNpu);
    let target = prism_spatial_ir::target::TargetCapabilities {
        sequential_schedules: !is_amd_npu,
        cross_domain_concurrency: is_amd_npu,
        gpu_ane_overlap: !is_amd_npu,
        pipeline_overlap: is_amd_npu,
        max_concurrent_regions: if is_amd_npu { 4 } else { 1 },
        max_weight_memory_bytes: if is_amd_npu {
            192 * 1024 * 1024 * 1024
        } else {
            8 * 1024 * 1024 * 1024
        },
        max_scratch_memory_bytes: if is_amd_npu {
            32 * 1024 * 1024 * 1024
        } else {
            2 * 1024 * 1024 * 1024
        },
        supports_compressed_kv_cache: true,
        supports_multi_gpu: false,
    };
    let legalize_result = CompilerLegalizer::legalize(
        &source,
        &graph_result.graph,
        &target,
        prism_spatial_ir::execution_plan::ExecutionMode::Batch,
    )
    .map_err(|e| CompileError::LegalizationFailed(e.to_string()))?;
    let legalize_duration = legalize_t0.elapsed().as_millis() as u64;
    let legalization_mode = if legalize_result.is_valid() {
        "passed"
    } else {
        "warnings"
    };
    emit_event(
        &mut events,
        2,
        CompilationStage::Legalization,
        EventKind::StageCompleted,
        &format!(
            "legalization {}: {} checks",
            legalization_mode,
            legalize_result.tensor_layout_valid.len()
        ),
        legalize_duration,
    );
    stages.push(StageResult {
        stage: CompilationStage::Legalization,
        success: legalize_result.is_valid(),
        duration_ms: legalize_duration,
        error: None,
    });

    // Stage 4: CImage emission
    let emit_t0 = std::time::Instant::now();
    let mut writer = UniversalCImageWriter::new(&output_path);
    writer.set_source(&source);
    // KV compression is promoted only from measured reference-cache evidence.
    // The evaluator may run on the MI300X during preprocessing and leaves a
    // resumable sidecar beside the tensor cache. This keeps the CImage's
    // default policy evolutionary without inventing a lossless claim when a
    // model has not been evaluated yet.
    if let Some(model_dir) = compiler.source_path.as_ref() {
        let cache_dir = model_dir.join(".prism-shard-cache");
        let evidence_path = cache_dir.join("kv-compression-evidence.json");
        let reference_keys = cache_dir.join("kv-reference-keys.f32");
        let reference_values = cache_dir.join("kv-reference-values.f32");
        if !evidence_path.is_file() && reference_keys.is_file() && reference_values.is_file() {
            let keys =
                read_f32_reference(&reference_keys).map_err(CompileError::CImageEmitFailed)?;
            let values =
                read_f32_reference(&reference_values).map_err(CompileError::CImageEmitFailed)?;
            crate::evaluator::evaluate_kv_reference_cache(&keys, &values, &evidence_path)
                .map_err(CompileError::CImageEmitFailed)?;
        }
        if evidence_path.is_file() {
            let search = prism_ecs_quantization::kv_search::KvCompressionSearch::default();
            let evidence = prism_ecs_quantization::kv_search::KvCompressionSearch::load_evidence(
                &evidence_path,
            )
            .map_err(CompileError::CImageEmitFailed)?;
            let winner = search.select_from_evidence(evidence).ok_or_else(|| {
                CompileError::CImageEmitFailed(
                    "KV compression evidence contains no lossless candidate".into(),
                )
            })?;
            writer.set_model_capabilities(["persistent-kv"]);
            writer
                .set_kv_compression_policy(&winner.candidate, search.max_error)
                .map_err(CompileError::CImageEmitFailed)?;
        }
    }
    if let Some(adapter) = compiler.model_adapter.as_ref() {
        writer
            .set_model_identity(adapter.family(), &serde_json::json!({}))
            .map_err(CompileError::CImageEmitFailed)?;
    }
    if let Some(config) = compiler.qwen36_config.clone() {
        writer
            .set_model_identity("qwen3-moe", &config)
            .map_err(CompileError::CImageEmitFailed)?;
        writer
            .set_qwen36_config(config)
            .map_err(CompileError::CImageEmitFailed)?;
    }
    if let Some(heterogeneous_manifest) = prism_spatial_ir::execution_plan::lower_to_manifest(
        &graph_result.graph,
        prism_spatial_ir::cost::CostEstimate::zero(),
        selected_format_plan.as_ref(),
    ) {
        let plan_json = serde_json::to_string(&heterogeneous_manifest).map_err(|e| {
            CompileError::CImageEmitFailed(format!("serialize execution plan: {e}"))
        })?;
        writer.set_execution_plan(plan_json);
    }
    if let Some(evidence) = selected_heterogeneous_workload_evidence {
        writer.set_heterogeneous_workload_evidence(evidence);
    }
    if let Some(format_plan) = selected_format_plan.as_ref() {
        let plan_json = serde_json::to_string(format_plan)
            .map_err(|e| CompileError::CImageEmitFailed(format!("serialize format plan: {e}")))?;
        writer
            .set_format_plan(plan_json)
            .map_err(CompileError::CImageEmitFailed)?;
    }
    let ternary_manifest: Vec<
        prism_ecs_quantization::safetensors_provider::TernaryPreprocessRecord,
    > = if let Some(model_dir) = compiler.source_path.as_ref() {
        let manifest_path = model_dir.join(".prism-shard-cache/ternary-manifest.json");
        std::fs::read(&manifest_path)
            .ok()
            .and_then(|bytes| {
                serde_json::from_slice::<
                    Vec<prism_ecs_quantization::safetensors_provider::TernaryPreprocessRecord>,
                >(&bytes)
                .ok()
            })
            .unwrap_or_default()
    } else {
        Vec::new()
    };
    for tensor in source.catalog.iter() {
        // For Qwen3.6, the completed ternary manifest is the promotion
        // authority. The format plan may still select backend lanes, but it
        // must not silently demote searched tensors back to source BF16.
        let native_record = ternary_manifest.iter().find(|record| {
            record.tensor_name == tensor.name
                && matches!(record.status.as_str(), "packed" | "packed_blockwise")
        });
        if let Some(record) = native_record {
            let shard_dir = compiler
                .source_path
                .as_ref()
                .unwrap()
                .join(".prism-shard-cache")
                .join(&record.shard_digest);
            let (tensor_type, payload_name) = if !record.metal_packed_file.is_empty() {
                (
                    prism_ecs_quantization::cimage::TensorType::TernaryTile640,
                    &record.metal_packed_file,
                )
            } else {
                (
                    prism_ecs_quantization::cimage::TensorType::Ternary158,
                    &record.packed_file,
                )
            };
            let payload_path = shard_dir.join(payload_name);
            if let Ok(payload) = std::fs::read(&payload_path) {
                let scales = if record.scales_file.is_empty() {
                    Vec::new()
                } else {
                    std::fs::read(shard_dir.join(&record.scales_file)).unwrap_or_default()
                };
                if scales.is_empty() {
                    return Err(CompileError::CImageEmitFailed(format!(
                        "native ternary tensor '{}' has no readable scale payload",
                        tensor.name
                    )));
                }
                {
                    let (rows, cols) = match tensor.shape.as_slice() {
                        [] => (1, 1),
                        [cols] => (1, *cols as u32),
                        shape => (shape[shape.len() - 2] as u32, shape[shape.len() - 1] as u32),
                    };
                    let descriptor = match tensor_type {
                        prism_ecs_quantization::cimage::TensorType::Ternary158 => {
                            crate::cimage::TernaryDescriptor {
                                version: 1,
                                codec: "Ternary158".into(),
                                group_size: record.group_size as u32,
                                scale_encoding: "F32".into(),
                                layout: if record.physical_tile_width > 0 {
                                    format!("Tiled:{}", record.physical_tile_width)
                                } else {
                                    "RowMajor".into()
                                },
                                packing: "TwoBitLE".into(),
                                kernel_variant: if record.physical_tile_width > 0 {
                                    format!("ternary_tiled_{}_gemv", record.physical_tile_width)
                                } else {
                                    "ternary158_gemv".into()
                                },
                                residual: "None".into(),
                            }
                        }
                        prism_ecs_quantization::cimage::TensorType::TernaryTile640 => {
                            crate::cimage::TernaryDescriptor {
                                version: 1,
                                codec: "TernaryTile640".into(),
                                group_size: 640,
                                scale_encoding: "BF16".into(),
                                layout: "Tile640:640".into(),
                                packing: "Base3U32LE".into(),
                                kernel_variant: "ternary_tile640_gemv".into(),
                                residual: "None".into(),
                            }
                        }
                        _ => unreachable!(),
                    };
                    if descriptor.validate().is_ok() {
                        let compile_type = match tensor_type {
                            prism_ecs_quantization::cimage::TensorType::Ternary158 => {
                                crate::cimage::TensorType::Ternary158
                            }
                            prism_ecs_quantization::cimage::TensorType::TernaryTile640 => {
                                crate::cimage::TensorType::TernaryTile640
                            }
                            _ => unreachable!(),
                        };
                        writer
                            .add_native_ternary_payload_with_scales(
                                &tensor.name,
                                &payload,
                                &scales,
                                rows,
                                cols,
                                compile_type,
                                descriptor,
                            )
                            .map_err(CompileError::CImageEmitFailed)?;
                        annotate_tensor_for_model(
                            &mut writer,
                            compiler.qwen36_config.as_ref(),
                            &tensor.name,
                        )
                        .map_err(CompileError::CImageEmitFailed)?;
                        continue;
                    }
                }
            }
        }
        if let Some(provider) = source.provider.as_ref() {
            if let Ok(payload) = provider.read_tensor(tensor) {
                writer
                    .add_tensor_payload(TensorPayloadEntry {
                        name: tensor.name.clone(),
                        payload,
                        representation: tensor.original_dtype.clone(),
                        effective_bpp: 16.0,
                        dim_m: tensor.shape.first().copied().unwrap_or(0) as u32,
                        dim_n: tensor.shape.get(1).copied().unwrap_or(0) as u32,
                        tensor_type: TensorType::Blob,
                    })
                    .map_err(CompileError::CImageEmitFailed)?;
                annotate_tensor_for_model(
                    &mut writer,
                    compiler.qwen36_config.as_ref(),
                    &tensor.name,
                )
                .map_err(CompileError::CImageEmitFailed)?;
            }
        }
    }

    // Materialize stateless ANE programs for admitted Tile640 candidates in
    // the same CImage as their tensor payloads. Search has already selected
    // the representation by this stage, so the artifact is the durable
    // hand-off between evolutionary search and runtime dispatch.
    #[cfg(all(feature = "ane", target_os = "macos"))]
    for tensor in source.catalog.iter().filter(|tensor| {
        let dtype = tensor.original_dtype.to_ascii_lowercase();
        dtype.contains("ternarytile640") || dtype.contains("ternary_tile640")
    }) {
        if tensor.shape.len() < 2 {
            return Err(CompileError::CImageEmitFailed(format!(
                "Tile640 tensor '{}' has no matrix shape",
                tensor.name
            )));
        }
        let output_width = tensor.shape[tensor.shape.len() - 2] as usize;
        let input_width = *tensor.shape.last().unwrap() as usize;
        let work_dir =
            std::env::temp_dir().join(format!("prism-ane-pipeline-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&work_dir).map_err(|e| {
            CompileError::CImageEmitFailed(format!("create ANE work directory: {e}"))
        })?;
        let result = (|| {
            let modelc = prism_ane::ternary_tile::compile_stateless_ternary_tile(
                prism_ane::ternary_tile::StatelessTernaryTileSpec {
                    input_width,
                    output_width,
                },
                &work_dir,
            )
            .map_err(CompileError::CImageEmitFailed)?;
            let payload =
                prism_ane::pack_mlmodelc(&modelc).map_err(CompileError::CImageEmitFailed)?;
            writer
                .add_ane_program(
                    &format!("ane_{}", tensor.name),
                    &payload,
                    "activation",
                    "ternary_weights",
                    "matmul_int8_2",
                )
                .map_err(CompileError::CImageEmitFailed)
        })();
        let _ = std::fs::remove_dir_all(&work_dir);
        result?;
    }

    // Materialize explicitly admitted planar FP16 candidates. The suffix
    // convention is emitted by graph legalization/search so arbitrary model
    // tensors are never promoted to an ANE program accidentally.
    #[cfg(all(feature = "ane", target_os = "macos"))]
    for activation in source.catalog.iter().filter(|tensor| {
        tensor
            .original_dtype
            .to_ascii_lowercase()
            .contains("planar_activation")
    }) {
        let bias_name = activation.name.replace("planar_activation", "planar_bias");
        let Some(bias) = source
            .catalog
            .iter()
            .find(|tensor| tensor.name == bias_name)
        else {
            continue;
        };
        if activation.shape.len() < 2 || bias.shape != activation.shape {
            return Err(CompileError::CImageEmitFailed(format!(
                "planar candidate '{}' and '{}' have incompatible shapes",
                activation.name, bias.name
            )));
        }
        let rows = activation.shape[activation.shape.len() - 2] as usize;
        let columns = activation.shape[activation.shape.len() - 1] as usize;
        let work_dir = std::env::temp_dir().join(format!(
            "prism-ane-planar-pipeline-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&work_dir).map_err(|e| {
            CompileError::CImageEmitFailed(format!("create ANE planar work directory: {e}"))
        })?;
        let result = (|| {
            let modelc = prism_ane::planar::compile_stateless_planar_add(rows, columns, &work_dir)
                .map_err(CompileError::CImageEmitFailed)?;
            let payload =
                prism_ane::pack_mlmodelc(&modelc).map_err(CompileError::CImageEmitFailed)?;
            writer
                .add_ane_program_typed(
                    &format!("ane_planar_{}", activation.name),
                    &payload,
                    &activation.name,
                    &bias.name,
                    "cast_3",
                    "int8",
                    "int8",
                )
                .map_err(CompileError::CImageEmitFailed)
        })();
        let _ = std::fs::remove_dir_all(&work_dir);
        result?;
    }
    writer.set_legalization_report(legalize_result);
    writer.set_events(events.clone());
    if let Some(trace) = search_trace {
        writer.set_search_trace(trace);
    }
    if let Some(receipt) = selection_receipt.clone() {
        writer.set_selection_receipt(receipt);
    }
    let promotion_receipt_path =
        std::env::var_os("PRISM_NATIVE_PROMOTION_EVIDENCE_PATH").map(std::path::PathBuf::from);
    if let Some(receipt_path) = promotion_receipt_path {
        let bytes = std::fs::read(&receipt_path).map_err(|e| {
            CompileError::CImageEmitFailed(format!(
                "read native promotion evidence {}: {e}",
                receipt_path.display()
            ))
        })?;
        let evidence: prism_ecs_quantization::ternarization::promotion::NativeTernaryPromotionEvidence =
            serde_json::from_slice(&bytes).map_err(|e| {
                CompileError::CImageEmitFailed(format!(
                    "parse native promotion evidence {}: {e}",
                    receipt_path.display()
                ))
            })?;
        let mut behavioral_evidence = None;
        // Re-run behavioral admission against an actual mapped router tensor;
        // a receipt cannot manufacture a passing reference result.
        if let Some(model_dir) = compiler.source_path.as_ref() {
            if let Ok(provider) =
                prism_ecs_quantization::safetensors_provider::SafeTensorProvider::new(model_dir)
            {
                if let Some(router_name) = provider
                    .list_tensors()
                    .ok()
                    .into_iter()
                    .flatten()
                    .map(|info| info.name)
                    .find(|name| {
                        let lower = name.to_ascii_lowercase();
                        lower.contains("router") || lower.contains("mlp.gate.weight")
                    })
                {
                    use crate::evaluator::BehavioralProbe;
                    let probe = crate::evaluator::MappedTensorBehavioralProbe::new(model_dir);
                    let context = crate::evaluator::MappedTensorProbeContext {
                        model_dir: model_dir.clone(),
                        tensor_name: router_name,
                    }
                    .to_bytes()
                    .map_err(|e| CompileError::CImageEmitFailed(e.to_string()))?;
                    let mut genome = prism_ecs_ir::evolution::CandidateGenome::new();
                    genome.representation =
                        prism_ecs_ir::evolution::RepresentationAxis::TernaryTile640;
                    let behavioral = probe
                        .evaluate(&genome, &context)
                        .map_err(|e| CompileError::CImageEmitFailed(e.to_string()))?;
                    behavioral_evidence = Some(behavioral);
                }
            }
        }
        writer
            .finalize_unpromoted()
            .map_err(CompileError::CImageEmitFailed)?;
        if let Some(behavioral) = behavioral_evidence {
            crate::cimage::promote_cimage_with_behavioral_evidence(
                &output_path,
                evidence,
                behavioral,
                &prism_ecs_ir::evolution::TernaryAdmissionLimits::default(),
            )
            .map_err(CompileError::CImageEmitFailed)?;
        } else {
            crate::cimage::promote_cimage_after_replay(&output_path, evidence)
                .map_err(CompileError::CImageEmitFailed)?;
        }
    } else {
        if std::env::var_os("PRISM_EMIT_UNPROMOTED_NATIVE").is_some() {
            writer
                .finalize_unpromoted()
                .map_err(CompileError::CImageEmitFailed)?;
        } else {
            writer
                .finalize()
                .map_err(|e| CompileError::CImageEmitFailed(e.to_string()))?;
        }
    }
    let emit_duration = emit_t0.elapsed().as_millis() as u64;
    emit_event(
        &mut events,
        3,
        CompilationStage::CImageEmission,
        EventKind::StageCompleted,
        &format!("cimage emitted: {}", output_path.display()),
        emit_duration,
    );
    stages.push(StageResult {
        stage: CompilationStage::CImageEmission,
        success: true,
        duration_ms: emit_duration,
        error: None,
    });

    // Compute output digest
    let output_digest = compute_file_digest_sha256(&output_path);

    // Build receipt
    let finished_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let duration_ms = finished_at.saturating_sub(started_at);

    let receipt_id = {
        use sha2::{Digest, Sha256};
        hex::encode(Sha256::digest(
            serde_json::to_string(&events)
                .unwrap_or_default()
                .as_bytes(),
        ))
    };

    let receipt = CompileReceipt {
        receipt_id,
        request_id,
        compiler_identity: CompilerIdentity {
            name: "canonical-compiler".into(),
            version: env!("CARGO_PKG_VERSION").into(),
            build_hash: option_env!("PRISM_BUILD_HASH").map(|s| s.to_string()),
            build_timestamp: option_env!("PRISM_BUILD_TIMESTAMP").map(|s| s.to_string()),
        },
        source_identity: source.identity.clone(),
        started_at: chrono::DateTime::<chrono::Utc>::from_timestamp(started_at as i64, 0)
            .unwrap_or_else(|| chrono::Utc::now()),
        completed_at: chrono::DateTime::<chrono::Utc>::from_timestamp(finished_at as i64, 0)
            .unwrap_or_else(|| chrono::Utc::now()),
        duration_ms,
        stages,
        candidate_count: candidate_count as u32,
        generations: generations_count as u32,
        output_digest: output_digest.clone(),
        output_path: output_path.clone(),
        schema_version: "1.0".into(),
        status: CompileStatus::Completed,
        error: None,
        finished_at: chrono::DateTime::<chrono::Utc>::from_timestamp(finished_at as i64, 0)
            .unwrap_or_else(|| chrono::Utc::now()),
        source_digest: None,
        graph_digest: None,
        search_trace_digest: None,
        kernel_manifest_digest: None,
        events_digest: None,
        legalization_mode: None,
        selection_receipt,
        uop_tuning_receipt: None,
    };

    Ok(CompileResult {
        receipt: receipt.clone(),
        status: CompileStatus::Completed,
        request_id,
        events,
        output_digest,
        output_path,
    })
}

fn read_f32_reference(path: &std::path::Path) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path)
        .map_err(|error| format!("read KV reference {}: {error}", path.display()))?;
    if bytes.is_empty() || bytes.len() % std::mem::size_of::<f32>() != 0 {
        return Err(format!(
            "KV reference {} is empty or not a whole number of f32 values",
            path.display()
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

/// Run the full compilation pipeline using ECS systems.
///
/// This function has moved to [`crate::plan_apply::compile_source_ecs`] as
/// part of the constitutional decomposition (2026-07-25). It is the only
/// entry point in `prism_ecs_compile` that directly mutates the canonical
/// world during the compile path; the world-mutating setup is now isolated
/// in a typed constitutional system rather than mixed with the orchestrator.
pub use crate::plan_apply::compile_source_ecs;

fn compute_file_digest_sha256(path: &Path) -> String {
    use sha2::{Digest, Sha256};
    let mut file = std::fs::File::open(path).expect("Failed to open file for digest");
    // The native Qwen artifact is multi-gigabyte. Avoid turning a post-build
    // integrity scan into a second unified-memory workload on macOS.
    #[cfg(target_os = "macos")]
    unsafe {
        use std::os::fd::AsRawFd;
        let _ = libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1);
    }
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let read =
            std::io::Read::read(&mut file, &mut buffer).expect("Failed to read file for digest");
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::RuntimeModel;
    use crate::CanonicalSource;
    use prism_ecs_core::identity::SourceFormat;
    use prism_ecs_core::{EntityKind, World};
    use prism_ecs_ir::ir_types::{FloatKind, TensorType, Type};
    use prism_ecs_ir::op::{OpMarker, OpName, Operands, Results};
    use prism_ecs_ir::value::{Uses, ValueType};
    use prism_ecs_source::{SourceCapabilities, SourceError, TensorCatalog, TensorDescriptor};
    use prism_spatial_ir::execution_plan::{FusedScheduleStep, PlanBackend};
    use prism_spatial_ir::{BufferStorage, ResolvedBuffer, RouteDispatch};

    struct EmptyProvider;

    impl TensorDataProvider for EmptyProvider {
        fn read_tensor(&self, _tensor: &TensorDescriptor) -> Result<Vec<u8>, SourceError> {
            Ok(Vec::new())
        }
    }

    #[test]
    fn reads_little_endian_kv_reference_samples() {
        let path =
            std::env::temp_dir().join(format!("prism-kv-reference-{}", uuid::Uuid::new_v4()));
        std::fs::write(
            &path,
            [1.25f32.to_le_bytes(), (-2.5f32).to_le_bytes()].concat(),
        )
        .unwrap();
        let values = read_f32_reference(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(values, vec![1.25, -2.5]);
    }

    fn make_test_source() -> CanonicalSource {
        let tensors = vec![
            TensorDescriptor {
                name: "model.embed_tokens.weight".into(),
                shape: vec![32000, 4096],
                dtype: "f16".into(),
                byte_offset: 0,
                byte_length: 32000 * 4096 * 2,
                element_size: 2,
                original_dtype: "F16".into(),
                data_offset: None,
                data_size_bytes: 32000 * 4096 * 2,
                layout: "row-major".into(),
            },
            TensorDescriptor {
                name: "lm_head.weight".into(),
                shape: vec![32000, 4096],
                dtype: "f16".into(),
                byte_offset: 0,
                byte_length: 32000 * 4096 * 2,
                element_size: 2,
                original_dtype: "F16".into(),
                data_offset: None,
                data_size_bytes: 32000 * 4096 * 2,
                layout: "row-major".into(),
            },
            TensorDescriptor {
                name: "model.layers.0.self_attn.q_proj.weight".into(),
                shape: vec![4096, 4096],
                dtype: "f16".into(),
                byte_offset: 0,
                byte_length: 4096 * 4096 * 2,
                element_size: 2,
                original_dtype: "F16".into(),
                data_offset: None,
                data_size_bytes: 4096 * 4096 * 2,
                layout: "row-major".into(),
            },
            TensorDescriptor {
                name: "model.layers.0.self_attn.k_proj.weight".into(),
                shape: vec![1024, 4096],
                dtype: "f16".into(),
                byte_offset: 0,
                byte_length: 1024 * 4096 * 2,
                element_size: 2,
                original_dtype: "F16".into(),
                data_offset: None,
                data_size_bytes: 1024 * 4096 * 2,
                layout: "row-major".into(),
            },
            TensorDescriptor {
                name: "model.layers.0.self_attn.v_proj.weight".into(),
                shape: vec![1024, 4096],
                dtype: "f16".into(),
                byte_offset: 0,
                byte_length: 1024 * 4096 * 2,
                element_size: 2,
                original_dtype: "F16".into(),
                data_offset: None,
                data_size_bytes: 1024 * 4096 * 2,
                layout: "row-major".into(),
            },
            TensorDescriptor {
                name: "model.layers.0.self_attn.o_proj.weight".into(),
                shape: vec![4096, 4096],
                dtype: "f16".into(),
                byte_offset: 0,
                byte_length: 4096 * 4096 * 2,
                element_size: 2,
                original_dtype: "F16".into(),
                data_offset: None,
                data_size_bytes: 4096 * 4096 * 2,
                layout: "row-major".into(),
            },
            TensorDescriptor {
                name: "model.layers.0.mlp.gate_proj.weight".into(),
                shape: vec![11008, 4096],
                dtype: "f16".into(),
                byte_offset: 0,
                byte_length: 11008 * 4096 * 2,
                element_size: 2,
                original_dtype: "F16".into(),
                data_offset: None,
                data_size_bytes: 11008 * 4096 * 2,
                layout: "row-major".into(),
            },
            TensorDescriptor {
                name: "model.layers.0.mlp.up_proj.weight".into(),
                shape: vec![11008, 4096],
                dtype: "f16".into(),
                byte_offset: 0,
                byte_length: 11008 * 4096 * 2,
                element_size: 2,
                original_dtype: "F16".into(),
                data_offset: None,
                data_size_bytes: 11008 * 4096 * 2,
                layout: "row-major".into(),
            },
            TensorDescriptor {
                name: "model.layers.0.mlp.down_proj.weight".into(),
                shape: vec![4096, 11008],
                dtype: "f16".into(),
                byte_offset: 0,
                byte_length: 4096 * 11008 * 2,
                element_size: 2,
                original_dtype: "F16".into(),
                data_offset: None,
                data_size_bytes: 4096 * 11008 * 2,
                layout: "row-major".into(),
            },
        ];
        CanonicalSource {
            identity: SourceIdentity {
                format: SourceFormat::Gguf,
                source_digest: "test_digest".into(),
                architecture: "llama".into(),
                model_family: "llama".into(),
            },
            catalog: TensorCatalog::new(tensors),
            provider: Some(std::sync::Arc::new(EmptyProvider)),
            capabilities: SourceCapabilities {
                supports_streaming: false,
                supports_random_access: false,
                supports_dequantize: false,
                random_access: false,
                mmap: false,
                writable: false,
            },
        }
    }

    #[test]
    fn test_compile_source_basic() {
        let source = make_test_source();
        let mut compiler = crate::CanonicalCompiler::new(CompileConfig {
            production_mode: false,
            max_candidates: 10,
            max_generations: 2,
            max_search_time_ms: 1000,
            target_backends: vec![BackendKind::CPU],
            calibration_policy: crate::CalibrationPolicy::None,
            validation_policy: crate::ValidationPolicy::Structural,
            enable_search: true,
            enable_legalization: true,
            enable_kernel_gen: false,
        });

        let result = compile_source(&mut compiler, source);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.status, CompileStatus::Completed);
        assert!(result.receipt.output_digest.len() > 0);
    }

    #[test]
    fn test_compile_source_ecs_basic() {
        use prism_ecs_core::world::World;

        let source = make_test_source();
        let config = CompileConfig {
            production_mode: false,
            max_candidates: 10,
            max_generations: 2,
            max_search_time_ms: 1000,
            target_backends: vec![BackendKind::CPU],
            calibration_policy: crate::CalibrationPolicy::None,
            validation_policy: crate::ValidationPolicy::Structural,
            enable_search: true,
            enable_legalization: true,
            enable_kernel_gen: false,
        };

        let mut world = World::new();
        let result = compile_source_ecs(&mut world, source, config);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.status, CompileStatus::Completed);
        assert!(result.receipt.selection_receipt.is_some());
        assert!(result.receipt.uop_tuning_receipt.is_some());
        let session = *world.get_resource::<crate::ecs::SessionHandle>().unwrap();
        let search = world
            .component::<crate::ecs::SearchStateComponent>(session.0)
            .unwrap();
        assert!(search.generations_completed > 0);
        let kernels = world
            .component::<crate::ecs::KernelCollection>(session.0)
            .unwrap();
        assert!(!kernels.lowered_manifests.is_empty());
        assert!(kernels.kernel_count > 0);
        assert!(!kernels.artifacts.is_empty());
        assert!(kernels.artifacts.iter().all(|artifact| {
            artifact
                .payloads
                .iter()
                .all(|payload| !payload.binary.is_empty())
        }));
    }

    #[test]
    fn compile_ecs_op_to_xdna_cimage_round_trips_native_artifact() {
        let mut world = World::new();
        let f16 = Type::float(FloatKind::F16);
        let a_ty = Type::Tensor(TensorType::new(vec![4, 8], f16.clone()));
        let b_ty = Type::Tensor(TensorType::new(vec![8, 16], f16.clone()));
        let c_ty = Type::Tensor(TensorType::new(vec![4, 16], f16));

        let make_value = |world: &mut World, name: &str, ty: Type| {
            let value: Entity = world
                .spawn(EntityKind::Node, Some(name.into()))
                .unwrap()
                .into();
            world.add_component(value, ValueType(ty)).unwrap();
            world.add_component(value, Uses(vec![])).unwrap();
            value
        };
        let a = make_value(&mut world, "A", a_ty);
        let b = make_value(&mut world, "B", b_ty);
        let c = make_value(&mut world, "C", c_ty.clone());
        let result = make_value(&mut world, "result", c_ty);
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

        let dir = std::env::temp_dir().join(format!("prism-xdna-cimage-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("matmul.cimage");
        compile_ecs_op_to_xdna_cimage(&world, op, &path).unwrap();

        let reader = crate::cimage::CImageReader::open(&path).unwrap();
        reader.validate_payload_ranges_for_validation().unwrap();
        let payload = reader.xdna_artifact("main").unwrap();
        let artifact = prism_amd_npu_runtime::XdnaArtifact::decode(&payload).unwrap();
        assert_eq!(
            artifact.program.topology.generation,
            prism_spatial_ir::xdna::XdnaGeneration::Aie2p
        );
        assert!(artifact
            .overlay
            .as_ref()
            .is_some_and(|bytes| !bytes.is_empty()));
        assert!(artifact
            .ctrlcode
            .as_ref()
            .is_some_and(|bytes| !bytes.is_empty()));
        assert!(reader
            .header
            .model_capabilities
            .iter()
            .any(|capability| capability == "native-xdna"));
        let runtime_model = RuntimeModel::load_for_validation(&path).unwrap();
        assert!(runtime_model.xdna_artifact("main").is_some());
        let inspection = RuntimeModel::inspect_for_validation(&path).unwrap();
        assert!(inspection.has_native_xdna);
        assert_eq!(inspection.xdna_artifact_count, 1);
        struct NoopXdna;
        impl prism_amd_npu_runtime::XdnaDevice for NoopXdna {
            type Error = String;
            fn upload(&mut self, _: &str, _: u32) -> Result<(), Self::Error> {
                Ok(())
            }
            fn execute(
                &mut self,
                _: &prism_spatial_ir::xdna::RuntimeCommand,
            ) -> Result<(), Self::Error> {
                Ok(())
            }
        }
        impl prism_amd_npu_runtime::XdnaCommandSubmitter for NoopXdna {
            fn submit_command_buffer(&mut self, _: &[u8]) -> Result<(), Self::Error> {
                Ok(())
            }

            fn submit_firmware_artifact(
                &mut self,
                _: &prism_spatial_ir::xdna::XdnaProgram,
                _: &[u8],
                _: &[u8],
            ) -> Result<(), String> {
                Ok(())
            }
        }
        let mut route = crate::CImageXdnaRouteDispatcher::new(&runtime_model, NoopXdna).unwrap();
        let step = FusedScheduleStep {
            step_id: 0,
            model_id: None,
            node_ids: vec![],
            backend: PlanBackend::Xdna,
            depends_on: vec![],
            input_region: "host".into(),
            output_region: "host".into(),
            zero_copy: false,
            estimated_latency_ns: 0,
            input_tensors: vec![],
            output_tensors: vec![],
            dispatch_geometry: [1, 1, 1],
            fusion_strategy: None,
        };
        let inputs = [64usize, 256, 128]
            .into_iter()
            .enumerate()
            .map(|(index, byte_length)| ResolvedBuffer {
                name: format!("model_operand_{index}"),
                element_type: "fp16".into(),
                region: "host".into(),
                byte_length,
                zero_copy: false,
                file_offset: None,
                storage: BufferStorage::RuntimeOwned,
                shape: vec![1],
                payload: Some(vec![0; byte_length]),
            })
            .collect::<Vec<_>>();
        let mut outputs = vec![];
        route.dispatch_xdna(&step, &inputs, &mut outputs).unwrap();
        assert!(route
            .runtime
            .resident_buffers()
            .any(|id| id.ends_with("::B")));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[cfg(all(feature = "ane", target_os = "macos"))]
    #[test]
    fn test_compile_int8_ane_tile_to_cimage_records_abi() {
        let dir = std::env::temp_dir().join(format!("prism-compile-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ane_tile.cimage");
        compile_int8_ane_tile_to_cimage(&path, "tile0", 4, 2).unwrap();
        let reader = crate::cimage::CImageReader::open(&path).unwrap();
        let record = reader.header.ane_programs.get("tile0").unwrap();
        assert_eq!(record.activation_input, "activation");
        assert_eq!(record.weights_input, "ternary_weights");
        assert_eq!(record.output, "matmul_int8_2");
        assert_eq!(record.input_dtype, "int8");
        assert_eq!(record.output_dtype, "int8");
        let _ = std::fs::remove_dir_all(dir);
    }
}
