//! PrismCompiler — the single public compilation entry point.
//!
//! All binary entry points, server endpoints, tests, and constitutional
//! commands call through this API. It is the ownership root for the
//! unified compilation pipeline.
//!
//! PR F — Collapse entry points. Old overlapping entry points are
//! deleted once their callers route through PrismCompiler.

use crate::ecs::canonical::compile_plan::{
    compile_timestamp, CimageBuildInput, CompileEvent, CompileEventStream, CompileOutcome,
    CompilePlan, CompileRequest, CompilerStage, InspectRequest, ModelInspection,
};
use crate::ecs::canonical::execution_graph::ExecutionGraph;
use crate::ecs::canonical::kernel_abi::KernelPlan;
use crate::ecs::canonical::model_ir::ModelIr;
use crate::ecs::canonical::representation::RepresentationPlan;

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
            state: crate::ecs::canonical::execution_graph::RuntimeStatePlan {
                max_context_tokens: model_ir.configuration.max_position_embeddings,
                kv_cache_bytes_per_token: 0,
                total_kv_cache_bytes: 0,
            },
            memory: crate::ecs::canonical::execution_graph::MemoryPlan {
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
    pub fn compile(&self, request: CompileRequest) -> Result<CompileOutcome, String> {
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
                crate::ecs::compute_image::compile::compile_gguf_to_canonical(
                    &request.source_path,
                    &output_dir,
                    quant_mode,
                    None, // ane_models_dir
                    None, // metallib_path
                    None, // mlx_capture_dir
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
            stage: CompilerStage::SourceResolved,
            success: true,
            timestamp: compile_timestamp(),
            duration_ms: 0.0,
            message: Some("Structural compilation plan produced (no backend)".into()),
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
                    state: crate::ecs::canonical::execution_graph::RuntimeStatePlan {
                        max_context_tokens: 0,
                        kv_cache_bytes_per_token: 0,
                        total_kv_cache_bytes: 0,
                    },
                    memory: crate::ecs::canonical::execution_graph::MemoryPlan {
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
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::canonical::compile_plan::{CompileRequest, InspectRequest};
    use crate::ecs::canonical::model_ir::*;
    use std::collections::HashMap;

    const _GGUF_FRONTEND_ENABLED: bool = cfg!(feature = "prism-backend");

    /// Mock frontend that returns minimal valid ModelIr.
    /// Documents the structural shape needed for compilation pipeline tests.
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
