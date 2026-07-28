//! PrismCompiler — the single public compilation entry point.
//!
//! All binary entry points, server endpoints, tests, and constitutional
//! commands call through this API. It is the ownership root for the
//! unified compilation pipeline.
//!
//! PR F — Collapse entry points. Old overlapping entry points are
//! deleted once their callers route through PrismCompiler.

use prism_ecs_constitutional::canonical::compile_plan::{
    compile_timestamp, CimageBuildInput, CompileEvent, CompileEventStream, CompileOutcome,
    CompilePlan, CompileRequest, CompilerStage, InspectRequest, ModelInspection,
};
use prism_ecs_constitutional::canonical::execution_graph::ExecutionGraph;
use prism_ecs_constitutional::canonical::kernel_abi::KernelPlan;
use prism_ecs_constitutional::canonical::model_ir::ModelIr;
use prism_ecs_constitutional::canonical::representation::RepresentationPlan;

/// Frontend trait — accepts a model source and produces canonical ModelIr.
pub trait ModelFrontend: Send + Sync {
    fn inspect(&self, source: &InspectRequest) -> Result<ModelInspection, String>;
    fn import(&self, source: &InspectRequest) -> Result<ModelIr, String>;
}

/// PrismCompiler — the single public compilation entry point.
///
/// Example:
/// ```ignore
/// let compiler = PrismCompiler::default();
/// let outcome = compiler.compile(CompileRequest {
///     source_path: "model.gguf".into(),
///     output_path: Some("out.cimage".into()),
///     ..Default::default()
/// })?;
/// ```
pub struct PrismCompiler {
    pub frontends: Vec<Box<dyn ModelFrontend>>,
    pub metal_backend: Option<crate::ecs::metal_backend::MetalBackendCompiler>,
}

impl Default for PrismCompiler {
    fn default() -> Self {
        let mut pc = Self {
            frontends: Vec::new(),
            metal_backend: None,
        };
        pc.frontends
            .push(Box::new(super::gguf_frontend::GgufFrontend::new()));
        pc
    }
}

impl std::fmt::Debug for PrismCompiler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrismCompiler")
            .field("frontend_count", &self.frontends.len())
            .field("metal_backend", &self.metal_backend.is_some())
            .finish()
    }
}

impl PrismCompiler {
    /// Create a new compiler with no frontends or backends registered.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a model frontend.
    pub fn register_frontend(&mut self, frontend: Box<dyn ModelFrontend>) {
        self.frontends.push(frontend);
    }

    /// Register the Metal backend compiler.
    #[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
    pub fn register_metal_backend(
        &mut self,
        backend: crate::ecs::metal_backend::MetalBackendCompiler,
    ) {
        self.metal_backend = Some(backend);
    }

    /// Inspect a model source without compiling.
    pub fn inspect(&self, request: InspectRequest) -> Result<ModelInspection, String> {
        for frontend in &self.frontends {
            match frontend.inspect(&request) {
                Ok(result) => return Ok(result),
                Err(_) => continue,
            }
        }
        Err(format!(
            "no frontend could inspect source: {}",
            request.source_path
        ))
    }

    /// Produce a CompilePlan without executing it.
    pub fn plan(&self, request: CompileRequest) -> Result<CompilePlan, String> {
        // 1. Import model through first matching frontend
        let inspect_req = InspectRequest {
            source_path: request.source_path.clone(),
            source_type: request.source_type.clone(),
        };
        let model_ir = self
            .frontends
            .iter()
            .find_map(|f| f.import(&inspect_req).ok())
            .ok_or_else(|| format!("no frontend could import source: {}", request.source_path))?;

        // 2. Build empty representation plan
        let rep_plan = RepresentationPlan {
            tensors: std::collections::BTreeMap::new(),
            calibration_receipt: None,
            admission_receipt: None,
            all_raw_f32: true,
        };

        // 3. Build empty execution graph
        let exec_graph = ExecutionGraph {
            regions: vec![],
            edges: vec![],
            state: prism_ecs_constitutional::canonical::execution_graph::RuntimeStatePlan {
                max_context_tokens: model_ir.configuration.max_position_embeddings,
                kv_cache_bytes_per_token: 0,
                total_kv_cache_bytes: 0,
            },
            memory: prism_ecs_constitutional::canonical::execution_graph::MemoryPlan {
                total_activation_bytes: 0,
                total_weight_bytes: 0,
                arena_region_count: 0,
            },
        };

        // 4. Build empty kernel plan
        let kernel_plan = KernelPlan { groups: vec![] };

        Ok(CompilePlan {
            model_ir,
            representation_plan: rep_plan,
            execution_graph: exec_graph,
            kernel_plan,
            estimated_output_size: 0,
        })
    }

    /// Compile a model source end-to-end.
    ///
    /// ## Gate note
    /// Real GGUF compilation (via `compile_gguf_to_canonical`) requires the
    /// `mlx-backend` feature because the pipeline imports from the `emit` module
    /// which depends on `mlx_rs`.  Without `mlx-backend`, the method falls back
    /// to producing structural empty artifacts from `plan()`.
    ///
    /// ## Authority routing
    /// When `request.authority == "SealedComputeImage"`, compilation delegates
    /// to `compile_with_authority`, which enforces validation gates via
    /// `compile_gguf_with_authority`.  Both paths are behind `mlx-backend`.
    pub fn compile(&self, request: CompileRequest) -> Result<CompileOutcome, String> {
        // ── Authority-gated path ─────────────────────────────────────
        #[cfg(feature = "mlx-backend")]
        if request.authority.as_deref() == Some("SealedComputeImage") {
            return self.compile_with_authority(request);
        }

        // Real GGUF compilation delegates to the full pipeline.
        #[cfg(feature = "mlx-backend")]
        if request.source_path.ends_with(".gguf") || request.source_type.as_deref() == Some("gguf")
        {
            let output_dir = request.output_path.clone().unwrap_or_else(|| {
                let stem = std::path::Path::new(&request.source_path)
                    .file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| "output".to_string());
                format!("{}.cimage", stem)
            });

            let quant_mode = request
                .quant_mode
                .as_deref()
                .and_then(crate::ecs::config::CompileQuantMode::from_name);

            let (_compiled_image, mut outcome) =
                crate::ecs::compute_image::legacy_compute_image_compile::compile_gguf_to_canonical(
                    &request.source_path,
                    &output_dir,
                    quant_mode,
                    request.ane_models_dir.as_deref(),
                    request.metallib_path.as_deref(),
                    request.mlx_capture_dir.as_deref(),
                )
                .map_err(|e| e.to_string())?;

            // Build event stream from existing pipeline receipts.
            // As the pipeline is enriched with direct events, this conversion
            // can be replaced with native event propagation.
            let mut event_stream = CompileEventStream::new(&request.source_path);
            for receipt in &outcome.receipts.receipts {
                let timestamp = compile_timestamp();
                event_stream.push(CompileEvent {
                    stage: receipt.stage,
                    success: receipt.success,
                    timestamp,
                    duration_ms: receipt.duration_ms,
                    message: receipt.message.clone(),
                    source_digest: None,
                    policy_digest: None,
                    artifact_digest: None,
                    toolchain_version: None,
                    failure_detail: None,
                });
            }
            event_stream.completed_at = Some(compile_timestamp());
            outcome.event_stream = event_stream;

            return Ok(outcome);
        }

        // Non-GGUF sources (or GGUF without the full pipeline feature):
        // produce a structural CompileOutcome from the plan with empty artifacts.
        let plan = self.plan(request)?;

        // Build a basic event stream for the non-pipeline path.
        let mut event_stream = CompileEventStream::new(&plan.model_ir.identity.name);
        event_stream.push(CompileEvent {
            stage: CompilerStage::SourceResolution,
            success: true,
            timestamp: compile_timestamp(),
            duration_ms: 0.0,
            message: Some(
                "Real GGUF compilation requires the mlx-backend feature — structural empty plan produced"
                    .into(),
            ),
            source_digest: None,
            policy_digest: None,
            artifact_digest: None,
            toolchain_version: None,
            failure_detail: None,
        });
        event_stream.completed_at = Some(compile_timestamp());

        let receipts = event_stream.to_receipt_set();

        Ok(CompileOutcome {
            plan,
            compiled_kernels: vec![],
            build_input: CimageBuildInput {
                model_ir_digest: [0u8; 32],
                representation_plan: RepresentationPlan {
                    tensors: std::collections::BTreeMap::new(),
                    calibration_receipt: None,
                    admission_receipt: None,
                    all_raw_f32: true,
                },
                execution_graph: ExecutionGraph {
                    regions: vec![],
                    edges: vec![],
                    state: prism_ecs_constitutional::canonical::execution_graph::RuntimeStatePlan {
                        max_context_tokens: 0,
                        kv_cache_bytes_per_token: 0,
                        total_kv_cache_bytes: 0,
                    },
                    memory: prism_ecs_constitutional::canonical::execution_graph::MemoryPlan {
                        total_activation_bytes: 0,
                        total_weight_bytes: 0,
                        arena_region_count: 0,
                    },
                },
                compiled_kernels: vec![],
                tensor_payloads: vec![],
                receipts: receipts.clone(),
            },
            receipts,
            output_path: None,
            event_stream,
        })
    }

    /// Compile a target + draft GGUF pair for speculative decoding.
    ///
    /// Delegates to compile_gguf_speculative behind the mlx-backend gate.
    /// Forwards request.target_hardware into the compiled image pipeline.
    #[cfg(feature = "mlx-backend")]
    pub fn compile_speculative(&self, request: CompileRequest) -> Result<CompileOutcome, String> {
        let draft_path = request
            .draft_path
            .as_deref()
            .ok_or_else(|| "compile_speculative requires draft_path to be set".to_string())?;

        let output_dir = request.output_path.clone().unwrap_or_else(|| {
            let stem = std::path::Path::new(&request.source_path)
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "output".to_string());
            format!("{}.cimage", stem)
        });

        let quant_mode = request
            .quant_mode
            .as_deref()
            .and_then(crate::ecs::config::CompileQuantMode::from_name);

        let authority = parse_authority(request.authority.as_deref());

        let target = request
            .target_hardware
            .as_deref()
            .and_then(|s| parse_hardware_target(Some(s)));

        let compiled_image = crate::ecs::compute_image::legacy_compute_image_compile::compile_gguf_speculative(
            &request.source_path,
            draft_path,
            &output_dir,
            authority,
            quant_mode,
            target,
        )
        .map_err(|e| e.to_string())?;

        let mut event_stream = CompileEventStream::new(&request.source_path);
        event_stream.push(CompileEvent {
            stage: CompilerStage::CimageAssembly,
            success: true,
            timestamp: compile_timestamp(),
            duration_ms: 0.0,
            message: Some(format!(
                "Speculative compilation with draft model produced: {}",
                output_dir
            )),
            source_digest: None,
            policy_digest: None,
            artifact_digest: None,
            toolchain_version: None,
            failure_detail: None,
        });
        event_stream.completed_at = Some(compile_timestamp());

        // Build a populated CompileOutcome from the real CompiledImage.
        let outcome = build_outcome_from_image(&compiled_image, &output_dir, &request);

        Ok(CompileOutcome {
            event_stream,
            ..outcome
        })
    }

    /// Authority-gated GGUF compilation.
    ///
    /// Routes through compile_gguf_with_authority when the request
    /// carries `authority: "sealed"`.
    ///
    /// Forwards request fields: ane_models_dir, metallib_path, mlx_capture_dir,
    /// target_hardware into the compiled image pipeline.
    #[cfg(feature = "mlx-backend")]
    fn compile_with_authority(&self, request: CompileRequest) -> Result<CompileOutcome, String> {
        let output_dir = request.output_path.clone().unwrap_or_else(|| {
            let stem = std::path::Path::new(&request.source_path)
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "output".to_string());
            format!("{}.cimage", stem)
        });

        let quant_mode = request
            .quant_mode
            .as_deref()
            .and_then(crate::ecs::config::CompileQuantMode::from_name);

        let authority = parse_authority(request.authority.as_deref());

        let target = request
            .target_hardware
            .as_deref()
            .and_then(|s| parse_hardware_target(Some(s)));
        let ane_dir = request.ane_models_dir.as_deref();
        let metal_path = request.metallib_path.as_deref();
        let mlx_cap = request.mlx_capture_dir.as_deref();

        let compiled_image = crate::ecs::compute_image::legacy_compute_image_compile::compile_gguf_with_authority(
            &request.source_path,
            &output_dir,
            authority,
            quant_mode,
            target,
            ane_dir,
            metal_path,
            mlx_cap,
        )
        .map_err(|e| e.to_string())?;

        let mut event_stream = CompileEventStream::new(&request.source_path);
        event_stream.push(CompileEvent {
            stage: CompilerStage::CimageAssembly,
            success: true,
            timestamp: compile_timestamp(),
            duration_ms: 0.0,
            message: Some(format!(
                "Authority-gated compilation ({}) produced: {}",
                authority, output_dir
            )),
            source_digest: None,
            policy_digest: None,
            artifact_digest: None,
            toolchain_version: None,
            failure_detail: None,
        });
        event_stream.completed_at = Some(compile_timestamp());

        // Build a populated CompileOutcome from the real CompiledImage.
        let outcome = build_outcome_from_image(&compiled_image, &output_dir, &request);

        Ok(CompileOutcome {
            event_stream,
            ..outcome
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use prism_ecs_constitutional::canonical::compile_plan::{CompileRequest, InspectRequest};
    use prism_ecs_constitutional::canonical::model_ir::*;
    use std::collections::HashMap;

    const _GGUF_FRONTEND_ENABLED: bool = cfg!(feature = "prism-backend");

    /// Mock frontend that returns minimal valid ModelIr.
    /// Documents the structural shape needed for compilation pipeline tests.
    #[allow(dead_code)]
    struct MockFrontend;

    impl ModelFrontend for MockFrontend {
        fn inspect(&self, _source: &InspectRequest) -> Result<ModelInspection, String> {
            Ok(ModelInspection {
                identity: ModelIdentity {
                    name: "mock".into(),
                    revision: None,
                },
                architecture: ArchitectureId("mock".into()),
                configuration: ModelConfiguration {
                    hidden_size: 64,
                    intermediate_size: 256,
                    num_attention_heads: 4,
                    num_kv_heads: 2,
                    num_hidden_layers: 1,
                    head_dim: 16,
                    vocab_size: 100,
                    max_position_embeddings: 128,
                    rms_norm_eps: 1e-6,
                    rope_theta: None,
                    partial_rope_dim: None,
                    tie_word_embeddings: false,
                    num_experts: None,
                    num_experts_per_tok: None,
                    moe_intermediate_size: None,
                    num_mtp_heads: None,
                    mtp_hidden_size: None,
                    mtp_intermediate_size: None,
                },
                tensor_count: 0,
                total_weight_bytes: 0,
            })
        }

        fn import(&self, _source: &InspectRequest) -> Result<ModelIr, String> {
            Ok(ModelIr {
                identity: ModelIdentity {
                    name: "mock".into(),
                    revision: None,
                },
                architecture: ArchitectureId("mock".into()),
                configuration: ModelConfiguration {
                    hidden_size: 64,
                    intermediate_size: 256,
                    num_attention_heads: 4,
                    num_kv_heads: 2,
                    num_hidden_layers: 1,
                    head_dim: 16,
                    vocab_size: 100,
                    max_position_embeddings: 128,
                    rms_norm_eps: 1e-6,
                    rope_theta: None,
                    partial_rope_dim: None,
                    tie_word_embeddings: false,
                    num_experts: None,
                    num_experts_per_tok: None,
                    moe_intermediate_size: None,
                    num_mtp_heads: None,
                    mtp_hidden_size: None,
                    mtp_intermediate_size: None,
                },
                tensors: TensorCatalogue {
                    by_id: vec![],
                    by_name: HashMap::new(),
                },
                graph: LogicalGraph {
                    ops: vec![],
                    inputs: vec![],
                    outputs: vec![],
                },
                tokenizer: TokenizerDescriptor {
                    tokenizer_type: "mock".into(),
                    vocab_size: 100,
                    bos_token_id: Some(1),
                    eos_token_id: Some(2),
                    pad_token_id: None,
                },
                source_provenance: SourceProvenance {
                    source_type: SourceType::Gguf,
                    source_path: "mock".into(),
                    file_digests: vec![],
                },
            })
        }
    }

    /// Verifies that a default PrismCompiler has the GGUF frontend registered
    /// when prism-backend is enabled, and none otherwise.
    #[test]
    fn test_default_no_frontends() {
        let compiler = PrismCompiler::default();
        if _GGUF_FRONTEND_ENABLED {
            assert_eq!(
                compiler.frontends.len(),
                1,
                "default PrismCompiler should register one GGUF frontend with prism-backend"
            );
        } else {
            assert!(
                compiler.frontends.is_empty(),
                "default PrismCompiler should have no frontends without prism-backend"
            );
        }
        assert!(
            compiler.metal_backend.is_none(),
            "default PrismCompiler should have no backend"
        );
    }

    /// Verifies that plan() fails when no frontend can import the source.
    /// Documents the structural gap: without registered frontends, compilation
    /// cannot proceed.
    #[test]
    fn test_plan_fails_without_frontend() {
        let compiler = PrismCompiler::default();
        let request = CompileRequest {
            source_path: "nonexistent.gguf".into(),
            source_type: None,
            output_path: None,
            target_lanes: vec![],
            policy_path: None,
            quant_mode: None,
            authority: None,
            draft_path: None,
            ane_models_dir: None,
            metallib_path: None,
            mlx_capture_dir: None,
            target_hardware: None,
        };
        let result = compiler.plan(request);
        assert!(result.is_err(), "plan() should fail without a frontend");
        let err = result.unwrap_err();
        assert!(
            err.contains("no frontend could import"),
            "error should mention missing frontend: {err}"
        );
    }

    /// Verifies that compile() with a mock frontend returns empty artifacts.
    /// Documents that compile produces a structural CompileOutcome with no
    /// compiled_kernels and no output_path when no real backend is wired.
    #[test]
    #[cfg(not(feature = "mlx-backend"))]
    fn test_compile_returns_empty_artifacts() {
        let mut compiler = PrismCompiler::default();
        compiler.register_frontend(Box::new(MockFrontend));

        let outcome = compiler
            .compile(CompileRequest {
                source_path: "mock.gguf".into(),
                source_type: None,
                output_path: None,
                target_lanes: vec![],
                policy_path: None,
                quant_mode: None,
                authority: None,
                draft_path: None,
                ane_models_dir: None,
                metallib_path: None,
                mlx_capture_dir: None,
                target_hardware: None,
            })
            .expect("compile() should succeed with a mock frontend");

        assert!(
            outcome.compiled_kernels.is_empty(),
            "expected empty compiled_kernels, got {}",
            outcome.compiled_kernels.len()
        );
        assert!(
            outcome.output_path.is_none(),
            "expected None output_path, got {:?}",
            outcome.output_path
        );

        // Event stream must be populated for the non-pipeline path.
        assert!(
            !outcome.event_stream.events.is_empty(),
            "event stream should have at least one event"
        );
        assert!(
            outcome.event_stream.all_success(),
            "all events in the stream should be success for the structural path"
        );
    }
}

/// Parse an authority string into a CompilationAuthority.
#[cfg(feature = "mlx-backend")]
fn parse_authority(s: Option<&str>) -> crate::ecs::compute_image::manifest::CompilationAuthority {
    use crate::ecs::compute_image::manifest::CompilationAuthority;
    match s {
        Some("TestFixture") => CompilationAuthority::TestFixture,
        Some("SealedComputeImage") => CompilationAuthority::SealedComputeImage,
        _ => CompilationAuthority::SealedComputeImage,
    }
}
/// Parse a hardware target string into a HardwareTarget.
#[cfg(feature = "mlx-backend")]
fn parse_hardware_target(s: Option<&str>) -> Option<crate::ecs::config::HardwareTarget> {
    use crate::ecs::config::HardwareTarget;
    s.and_then(|t| match t.to_lowercase().as_str() {
        "m1" => Some(HardwareTarget::M1),
        "m1pro" => Some(HardwareTarget::M1Pro),
        "m2" => Some(HardwareTarget::M2),
        "m2ultra" => Some(HardwareTarget::M2Ultra),
        "m3ultra" => Some(HardwareTarget::M3Ultra),
        _ => None,
    })
}

/// Build a populated CompileOutcome from a CompiledImage returned by the
/// GGUF pipeline (compile_gguf_with_authority or compile_gguf_speculative).
///
/// Populates model_ir_digest, execution_graph (weight bytes from segments),
/// compiled kernel entries from metal_kernel_artifacts, and receipts from
/// the compile receipt — avoiding the structural-empty-artifact problem.
///
/// The plan field uses the request model name if available, otherwise a
/// minimal placeholder.
#[cfg(feature = "mlx-backend")]
fn build_outcome_from_image(
    compiled_image: &crate::ecs::compute_image::manifest::CompiledImage,
    output_dir: &str,
    request: &CompileRequest,
) -> CompileOutcome {
    use prism_ecs_constitutional::canonical::kernel_abi::{
        CompiledKernelArtifact, DispatchGeometryPolicy, KernelAbi, KernelImplementationId,
        KernelSemanticId,
    };
    use sha2::{Digest, Sha256};

    // Model IR digest from manifest data
    let mut hasher = Sha256::new();
    hasher.update(compiled_image.manifest.image_hash.as_bytes());
    hasher.update(compiled_image.manifest.source.model_type.as_bytes());
    let model_ir_digest: [u8; 32] = hasher.finalize().into();

    // Execution graph from manifest segments
    let total_weight_bytes: u64 = compiled_image
        .manifest
        .segments
        .iter()
        .map(|s| s.byte_size)
        .sum();

    let execution_graph = ExecutionGraph {
        regions: Vec::new(),
        edges: Vec::new(),
        state: prism_ecs_constitutional::canonical::execution_graph::RuntimeStatePlan {
            max_context_tokens: 0,
            kv_cache_bytes_per_token: 0,
            total_kv_cache_bytes: 0,
        },
        memory: prism_ecs_constitutional::canonical::execution_graph::MemoryPlan {
            total_activation_bytes: 0,
            total_weight_bytes,
            arena_region_count: compiled_image.manifest.segments.len(),
        },
    };

    // Compiled kernel entries from metal kernel artifacts
    let compiled_kernels: Vec<prism_ecs_constitutional::canonical::compile_plan::CompiledKernelEntry> =
        compiled_image
            .manifest
            .metal_kernel_artifacts
            .iter()
            .map(|art| {
                let semi_id = KernelSemanticId(format!("{}:{:?}", art.logical_operation, art.kind));
                prism_ecs_constitutional::canonical::compile_plan::CompiledKernelEntry {
                    artifact: CompiledKernelArtifact {
                        implementation_id: KernelImplementationId(format!(
                            "{}|{}",
                            art.artifact_id, art.gpu_family
                        )),
                        semantic_id: semi_id,
                        compiled_bytes: Vec::new(),
                        sha256: art.checksum.clone(),
                        entry_point: art.artifact_id.clone(),
                        abi: KernelAbi {
                            version: 1,
                            buffers: Vec::new(),
                            constants: Vec::new(),
                            threadgroup_memory: Vec::new(),
                            dispatch_geometry: DispatchGeometryPolicy::Fixed(1, 1, 1),
                            threads_per_threadgroup: (1, 1, 1),
                        },
                    },
                    compile_duration_ms: 0.0,
                    cache_hit: false,
                }
            })
            .collect();

    // Build the kernel plan artifacts for build_input
    let build_input_kernels: Vec<CompiledKernelArtifact> = compiled_kernels
        .iter()
        .map(|e| e.artifact.clone())
        .collect();

    // Receipts from compile receipt
    let cr = &compiled_image.receipt;
    let receipts = vec![
        prism_ecs_constitutional::canonical::compile_plan::CompilerReceipt {
            stage: CompilerStage::SourceResolution,
            success: true,
            duration_ms: 0.0,
            message: Some(format!("Source config hash: {}", cr.source_config_hash)),
        },
        prism_ecs_constitutional::canonical::compile_plan::CompilerReceipt {
            stage: CompilerStage::PayloadPacking,
            success: true,
            duration_ms: 0.0,
            message: Some(format!(
                "{} tensors, {} segments",
                cr.tensor_count,
                cr.segment_hashes.len()
            )),
        },
        prism_ecs_constitutional::canonical::compile_plan::CompilerReceipt {
            stage: CompilerStage::CimageAssembly,
            success: true,
            duration_ms: 0.0,
            message: Some(format!("Complete image hash: {}", cr.complete_image_hash)),
        },
    ];
    let receipt_set = prism_ecs_constitutional::canonical::compile_plan::CompilerReceiptSet { receipts };

    // Model name from request, or fall back to manifest
    let model_name = if !request.source_path.is_empty() {
        std::path::Path::new(&request.source_path)
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "model".to_string())
    } else {
        "model".to_string()
    };

    // Build a minimal plan with the model identity populated
    let plan = CompilePlan {
        model_ir: ModelIr {
            identity: prism_ecs_constitutional::canonical::model_ir::ModelIdentity {
                name: model_name,
                revision: None,
            },
            architecture: prism_ecs_constitutional::canonical::model_ir::ArchitectureId(
                compiled_image.manifest.source.model_type.clone(),
            ),
            configuration: prism_ecs_constitutional::canonical::model_ir::ModelConfiguration {
                hidden_size: 0,
                intermediate_size: 0,
                num_attention_heads: 0,
                num_kv_heads: 0,
                num_hidden_layers: 0,
                head_dim: 0,
                vocab_size: 0,
                max_position_embeddings: 0,
                rms_norm_eps: 0.0,
                rope_theta: None,
                partial_rope_dim: None,
                tie_word_embeddings: false,
                num_experts: None,
                num_experts_per_tok: None,
                moe_intermediate_size: None,
                num_mtp_heads: None,
                mtp_hidden_size: None,
                mtp_intermediate_size: None,
            },
            tensors: prism_ecs_constitutional::canonical::model_ir::TensorCatalogue {
                by_id: vec![],
                by_name: std::collections::HashMap::new(),
            },
            graph: prism_ecs_constitutional::canonical::model_ir::LogicalGraph {
                ops: vec![],
                inputs: vec![],
                outputs: vec![],
            },
            tokenizer: prism_ecs_constitutional::canonical::model_ir::TokenizerDescriptor {
                tokenizer_type: "unknown".into(),
                vocab_size: 0,
                bos_token_id: None,
                eos_token_id: None,
                pad_token_id: None,
            },
            source_provenance: prism_ecs_constitutional::canonical::model_ir::SourceProvenance {
                source_type: prism_ecs_constitutional::canonical::model_ir::SourceType::Gguf,
                source_path: request.source_path.clone(),
                file_digests: vec![],
            },
        },
        representation_plan: RepresentationPlan {
            tensors: std::collections::BTreeMap::new(),
            calibration_receipt: None,
            admission_receipt: None,
            all_raw_f32: true,
        },
        execution_graph: execution_graph.clone(),
        kernel_plan: KernelPlan { groups: vec![] },
        estimated_output_size: total_weight_bytes,
    };

    CompileOutcome {
        plan,
        compiled_kernels,
        build_input: CimageBuildInput {
            model_ir_digest,
            representation_plan: RepresentationPlan {
                tensors: std::collections::BTreeMap::new(),
                calibration_receipt: None,
                admission_receipt: None,
                all_raw_f32: true,
            },
            execution_graph,
            compiled_kernels: build_input_kernels,
            tensor_payloads: Vec::new(),
            receipts: receipt_set.clone(),
        },
        receipts: receipt_set,
        output_path: Some(output_dir.to_string()),
        event_stream: CompileEventStream::new(output_dir),
    }
}
