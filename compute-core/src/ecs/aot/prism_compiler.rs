//! PrismCompiler — the single public compilation entry point.
//!
//! All binary entry points, server endpoints, tests, and constitutional
//! commands call through this API. It is the ownership root for the
//! unified compilation pipeline.
//!
//! PR F — Collapse entry points. Old overlapping entry points are
//! deleted once their callers route through PrismCompiler.

use crate::ecs::canonical::compile_plan::{
    CimageBuildInput, CompileOutcome, CompilePlan, CompileRequest, CompilerReceipt,
    CompilerReceiptSet, CompilerStage, InspectRequest, ModelInspection,
};
use crate::ecs::canonical::execution_graph::ExecutionGraph;
use crate::ecs::canonical::kernel_abi::{CompiledKernelArtifact, KernelPlan};
use crate::ecs::canonical::model_ir::{ModelIr, SourceType};
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
        Self {
            frontends: Vec::new(),
            metal_backend: None,
        }
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
        let _plan = self.plan(request)?;
        let mut receipts = CompilerReceiptSet::new();

        receipts.push(CompilerReceipt {
            stage: CompilerStage::SourceResolved,
            success: true,
            duration_ms: 0.0,
            message: None,
        });

        // Stub: actual compilation delegates to the registered backends.
        // The existing compile_unchecked/compile_gguf_unchecked paths are
        // called from here in the full implementation.
        //
        // For now, return a structural CompileOutcome with empty artifacts.

        Ok(CompileOutcome {
            plan: _plan,
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
        })
    }
}
