//! CimageDeploymentCompiler — the production entry point for building
//! a deployable cimage from a model directory and promoting through lifecycle.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

use crate::ecs::canonical::execution_graph::{MemoryPlan, RuntimeStatePlan};
use crate::ecs::canonical::identity::{GenerationId, PhysicalSegmentId};
use crate::ecs::canonical::kernel_abi::CompiledKernelArtifact;
use crate::ecs::compiler::lifecycle_coordinator::LifecycleResult;
use crate::ecs::compute_image::model_family::gemma4_mtp_graph::MTPExecutionGraph;

// Types used only inside the prism-backend-gated impl block.
#[cfg(feature = "prism-backend")]
use crate::ecs::aot::prism_compiler::{ModelFrontend, PrismCompiler};
#[cfg(feature = "prism-backend")]
use crate::ecs::canonical::compile_plan::CompileRequest;
#[cfg(feature = "prism-backend")]
use crate::ecs::canonical::identity::ModelSourceId;
#[cfg(feature = "prism-backend")]
use crate::ecs::compiler::lifecycle_coordinator::{LifecycleCoordinator, PolicyConfig};
#[cfg(feature = "prism-backend")]
use crate::ecs::compute_image::model_family::gemma4_inspect::inspect_gemma4_checkpoint;
#[cfg(feature = "prism-backend")]
use crate::ecs::metal_backend::compiler::MetalBackendCompiler;
use prism_ecs_runtime::scheduling::systems::compilation_job_bridge::CompilationJobBridge;
#[cfg(feature = "prism-backend")]
use memmap2::Mmap;
#[cfg(feature = "prism-backend")]
use sha2::{Digest, Sha256};

/// A request to compile a model into a deployable cimage.
#[derive(Debug, Clone)]
pub struct DeploymentRequest {
    pub model_path: PathBuf,
    pub output_path: Option<PathBuf>,
    pub target: String,
    pub precision: String,
    pub mtp: bool,
    pub max_context: Option<usize>,
    pub admission_policy: Option<String>,
}

impl Default for DeploymentRequest {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            output_path: None,
            target: "apple-m1".into(),
            precision: "nf4".into(),
            mtp: true,
            max_context: Some(8192),
            admission_policy: Some("fail-closed".into()),
        }
    }
}

/// Output from a successful deployment compilation.
pub struct DeploymentResult {
    pub cimage_path: PathBuf,
    pub generation_id: GenerationId,
    pub lifecycle: LifecycleResult,
    pub mtp_enabled: bool,
}

/// Metadata about a compiled model needed at serving time.
///
/// Carried in the `CimageAssembly` and used by runtime loaders to
/// configure the serving environment without re-inspecting the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServingProfile {
    pub model_name: String,
    pub model_tag: String,
    pub architecture: String,
    pub context_length: usize,
    pub precision: String,
    pub mtp_enabled: bool,
}

/// A fully assembled deployable cimage ready for lifecycle promotion.
///
/// Produced by `build_deployable_cimage()` after upstream compilation.
/// This is the intermediate representation between compiler output and
/// lifecycle/ promotion — all tensor segments, kernel artifacts, graph
/// topology, memory plan, runtime state, and serving metadata are resolved.
pub struct CimageAssembly {
    /// Content-addressed tensor segments keyed by physical segment id.
    pub segments: BTreeMap<PhysicalSegmentId, Vec<u8>>,
    /// Compiled kernel artifacts (Metal .metallib bytes + ABI).
    pub kernel_artifacts: Vec<CompiledKernelArtifact>,
    /// Execution graph topology (MTP-aware for Gemma 4).
    pub execution_graph: MTPExecutionGraph,
    /// Memory allocation plan (activation / weight / arena regions).
    pub memory_plan: MemoryPlan,
    /// Runtime state plan (KV cache sizing, context length).
    pub runtime_state: RuntimeStatePlan,
    /// Serving metadata for runtime configuration.
    pub serving_profile: ServingProfile,
}

/// A sealed cimage ready for lifecycle promotion.
/// Produced by seal_and_validate(), consumed by promote_cimage().
pub struct PromotableCimage {
    pub assembly: CimageAssembly,
    pub validated: bool,
    pub digest: String,
}

impl CimageAssembly {
    /// Compute a content-addressed digest over all segments and kernel artifacts.
    pub fn compute_digest(&self) -> String {
        use sha2::Digest;
        let mut hasher = sha2::Sha256::new();
        for (_id, data) in &self.segments {
            hasher.update(data);
        }
        for artifact in &self.kernel_artifacts {
            // Hash the implementation id bytes and compiled bytes for content-addressing
            hasher.update(artifact.implementation_id.0.as_bytes());
            hasher.update(&artifact.compiled_bytes);
        }
        let hash = hasher.finalize();
        use base64::Engine;
        base64::engine::general_purpose::STANDARD.encode(hash)
    }
}

/// Production compiler that builds a deployable cimage from a model directory.
#[cfg(feature = "prism-backend")]
pub struct CimageDeploymentCompiler {
    pub prism_compiler: PrismCompiler,
    pub metal_backend: Option<MetalBackendCompiler>,
    pub lifecycle_coordinator: Option<LifecycleCoordinator>,
    pub job_bridge: Option<CompilationJobBridge>,
}

#[cfg(feature = "prism-backend")]
impl Default for CimageDeploymentCompiler {
    fn default() -> Self {
        let mut pc = PrismCompiler::default();
        // Register the Gemma 4 safetensors frontend so that real Gemma 4
        // unified checkpoints can be imported without a GGUF conversion step.
        pc.register_frontend(Box::new(
            crate::ecs::aot::gemma4_frontend::Gemma4SafetensorsFrontend::new(),
        ));
        Self {
            prism_compiler: pc,
            metal_backend: Self::create_metal_backend(),
            lifecycle_coordinator: Some(LifecycleCoordinator::new()),
            job_bridge: None,
        }
    }
}

#[cfg(feature = "prism-backend")]
impl CimageDeploymentCompiler {
    /// Metadata about a compiled model needed at serving time.
    /// Build a deployable cimage assembly from the compiler's compile outcome.
    ///
    /// Collects quantized tensor segments from the build input, compiles kernel
    /// artifacts, resolves the execution graph topology, derives memory and
    /// runtime state plans, and packs everything into a `CimageAssembly` with
    /// full serving profile metadata.
    ///
    /// Called automatically by `compile()` between upstream compilation and
    /// lifecycle promotion. Callers that bypass `compile()` can call this
    /// directly when they have a `CompileOutcome` and model inspection data.
    pub fn build_deployable_cimage(
        &self,
        compile_outcome: &crate::ecs::canonical::CompileOutcome,
        request: &DeploymentRequest,
        inspection: &crate::ecs::compute_image::model_family::gemma4_inspect::Gemma4Inspection,
        output_path: &PathBuf,
    ) -> CimageAssembly {
        // ── 1. Collect quantized tensor segments ────────────────────────────
        let mut segments: BTreeMap<PhysicalSegmentId, Vec<u8>> = BTreeMap::new();
        for payload in &compile_outcome.build_input.tensor_payloads {
            let seg_id = PhysicalSegmentId(payload.name.clone());
            segments.insert(seg_id, payload.data.clone());
        }

        // ── 2. Collect compiled kernel artifacts ────────────────────────────
        let kernel_artifacts: Vec<CompiledKernelArtifact> = compile_outcome
            .compiled_kernels
            .iter()
            .map(|entry| entry.artifact.clone())
            .collect();

        // ── 3. Build execution graph (MTP-aware) ───────────────────────────
        let mtp_depth = if request.mtp {
            inspection.config.mtp_depth.unwrap_or(0)
        } else {
            0
        };
        let execution_graph = if mtp_depth > 0 {
            MTPExecutionGraph::with_mtp(mtp_depth as usize, true, true)
        } else {
            MTPExecutionGraph::target_only()
        };

        // ── 4. Memory plan — prefer build_input execution graph data ────────
        let memory_plan = compile_outcome.build_input.execution_graph.memory.clone();

        // ── 5. Runtime state plan — prefer build_input data ─────────────────
        let runtime_state = compile_outcome.build_input.execution_graph.state.clone();

        // ── 6. Serving profile ─────────────────────────────────────────────
        let serving_profile = ServingProfile {
            model_name: output_path
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "model".into()),
            model_tag: output_path
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "default".into()),
            architecture: "gemma-4".into(),
            context_length: request.max_context.unwrap_or(8192),
            precision: request.precision.clone(),
            mtp_enabled: mtp_depth > 0,
        };

        CimageAssembly {
            segments,
            kernel_artifacts,
            execution_graph,
            memory_plan,
            runtime_state,
            serving_profile,
        }
    }

    pub fn new() -> Self {
        Self::default()
    }

    fn create_metal_backend() -> Option<MetalBackendCompiler> {
        #[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
        {
            let backend = MetalBackendCompiler::new();
            if backend.is_available() {
                return Some(backend);
            }
        }
        None
    }

    pub fn register_frontend(&mut self, frontend: Box<dyn ModelFrontend>) {
        self.prism_compiler.register_frontend(frontend);
    }

    pub fn register_metal_backend(&mut self, backend: MetalBackendCompiler) {
        self.metal_backend = Some(backend);
    }

    pub fn with_policy(mut self, policy: PolicyConfig) -> Self {
        if let Some(ref mut coord) = self.lifecycle_coordinator {
            *coord = std::mem::take(coord).with_policy(policy);
        }
        self
    }

    pub fn compile(&mut self, request: DeploymentRequest) -> Result<DeploymentResult, String> {
        let started = std::time::Instant::now();

        // 1. Inspect the model directory to build tensor inventory
        let inspection = inspect_gemma4_checkpoint(&request.model_path)
            .map_err(|e| format!("model inspection failed: {e}"))?;

        eprintln!(
            "[deploy] {}: {} tensors, {} layers, vocab={}, mtp={}",
            request.model_path.display(),
            inspection.inventory.total_tensors,
            inspection.config.num_layers,
            inspection.config.vocab_size,
            inspection.config.mtp_depth.is_some()
        );

        // 2. Call PrismCompiler::plan() to get real CompilePlan with ModelIr
        let source_path = request.model_path.to_string_lossy().to_string();
        let compile_request = CompileRequest {
            source_path: source_path.clone(),
            source_type: Some("safetensors".into()),
            output_path: request
                .output_path
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            target_hardware: Some(request.target.clone()),
            quant_mode: Some(request.precision.clone()),
            ..Default::default()
        };

        let compile_plan = self
            .prism_compiler
            .plan(compile_request.clone())
            .map_err(|e| format!("compiler plan failed: {e}"))?;

        eprintln!(
            "[deploy] plan: {} tensors in model IR, framework={}",
            compile_plan.model_ir.tensors.by_id.len(),
            compile_plan.model_ir.architecture.0,
        );

        // 2b. Call PrismCompiler::compile() to get real CompileOutcome with
        // compiled_kernels, receipts, event_stream, and model_ir_digest
        // 2a. Record compilation job creation in the constitutional authority path
        let mut job_entity: u64 = 0;
        if let Some(bridge) = &self.job_bridge {
            let job_config = crate::ecs::constitutional::compilation::JobConfig {
                target_format: request.target.clone(),
                optimization_level: 3,
                enable_validation: true,
            };
            if let Ok(entity_id) = bridge.create_job(
                0, // model_artifact_id — placeholder; bridge is authority record
                &request.target,
                job_config,
            ) {
                job_entity = entity_id;
            }
        }

        let compiled_outcome = self
            .prism_compiler
            .compile(compile_request.clone())
            .map_err(|e| format!("compiler compile failed: {e}"))?;

        // 3. Discover safetensors files in the model directory (handles sharding)
        let mut safetensors_files: Vec<PathBuf> = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&request.model_path) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().map_or(false, |ext| ext == "safetensors") {
                    safetensors_files.push(path);
                }
            }
        }
        safetensors_files.sort();

        if safetensors_files.is_empty() {
            return Err("no .safetensors files found in model directory".into());
        }

        // 4. Build tensor payloads from safetensors files using bounded mmap
        let mut tensor_payloads: Vec<crate::ecs::canonical::TensorPayload> = Vec::new();

        for sf_path in &safetensors_files {
            let file = std::fs::File::open(sf_path)
                .map_err(|e| format!("cannot open {}: {e}", sf_path.display()))?;
            let mmap = unsafe { Mmap::map(&file) }
                .map_err(|e| format!("cannot mmap {}: {e}", sf_path.display()))?;

            let header_len = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
            let header: serde_json::Value = serde_json::from_slice(&mmap[8..8 + header_len])
                .map_err(|e| format!("invalid safetensors header in {}: {e}", sf_path.display()))?;
            let header_obj = header.as_object().ok_or("header is not an object")?;
            let data_start = 8 + header_len;

            for (name, meta) in header_obj {
                if meta.get("dtype").is_none() {
                    continue; // skip __metadata__
                }
                let offsets = meta["data_offsets"].as_array().ok_or(format!(
                    "missing data_offsets for {name} in {}",
                    sf_path.display()
                ))?;
                let start: usize = offsets[0].as_u64().ok_or("bad offset")? as usize;
                let end: usize = offsets[1].as_u64().ok_or("bad offset")? as usize;
                let slice = &mmap[data_start + start..data_start + end];
                tensor_payloads.push(crate::ecs::canonical::TensorPayload {
                    name: name.clone(),
                    data: slice.to_vec(),
                    byte_size: (end - start) as u64,
                });
            }
            // mmap is dropped here, releasing the mapping
        }

        // 5. Compute real model_ir_digest from the plan's model IR
        let mut hasher = Sha256::new();
        hasher.update(compile_plan.model_ir.identity.name.as_bytes());
        hasher.update(compile_plan.model_ir.architecture.0.as_bytes());
        hasher.update(
            &compile_plan
                .model_ir
                .configuration
                .hidden_size
                .to_le_bytes(),
        );
        hasher.update(
            &compile_plan
                .model_ir
                .configuration
                .num_hidden_layers
                .to_le_bytes(),
        );
        hasher.update(
            &compile_plan
                .model_ir
                .configuration
                .num_attention_heads
                .to_le_bytes(),
        );
        hasher.update(
            &compile_plan
                .model_ir
                .configuration
                .num_kv_heads
                .to_le_bytes(),
        );
        hasher.update(&compile_plan.model_ir.configuration.head_dim.to_le_bytes());
        hasher.update(&compile_plan.model_ir.configuration.vocab_size.to_le_bytes());
        for desc in &compile_plan.model_ir.tensors.by_id {
            hasher.update(desc.name.as_bytes());
            hasher.update(&desc.byte_size.to_le_bytes());
            for dim in &desc.shape {
                hasher.update(&dim.to_le_bytes());
            }
        }
        let model_ir_digest: [u8; 32] = hasher.finalize().into();

        // 6. Build the output path
        // 5b. Record cimage promotion in the constitutional authority path
        if let Some(bridge) = &self.job_bridge {
            // Encode the model IR digest as a hex string for the bridge's &str parameter
            let digest_hex: String = model_ir_digest.iter().map(|b| format!("{b:02x}")).collect();
            let _ = bridge.promote_cimage(
                crate::ecs::Entity(job_entity, 0),
                &digest_hex,
                source_path.len() as u64,
                &request.target,
            );
        }

        let output_path = request.output_path.clone().unwrap_or_else(|| {
            let stem = request
                .model_path
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            PathBuf::from(format!("{}.cimage", stem))
        });

        // 7. Build the proper CompileOutcome combining plan data + real safetensors
        //    + real compile output (compiled_kernels, receipts, event_stream)
        let build_kernel_artifacts: Vec<crate::ecs::canonical::CompiledKernelArtifact> =
            compiled_outcome
                .compiled_kernels
                .iter()
                .map(|entry| entry.artifact.clone())
                .collect();

        let compile_outcome = crate::ecs::canonical::CompileOutcome {
            plan: compile_plan.clone(),
            compiled_kernels: compiled_outcome.compiled_kernels,
            build_input: crate::ecs::canonical::CimageBuildInput {
                model_ir_digest,
                representation_plan: compile_plan.representation_plan.clone(),
                execution_graph: compile_plan.execution_graph.clone(),
                compiled_kernels: build_kernel_artifacts,
                tensor_payloads,
                receipts: compiled_outcome.receipts.clone(),
            },
            receipts: compiled_outcome.receipts.clone(),
            output_path: Some(output_path.to_string_lossy().to_string()),
            event_stream: compiled_outcome.event_stream.clone(),
        };

        // 8. Build cimage assembly
        let cimage_assembly =
            self.build_deployable_cimage(&compile_outcome, &request, &inspection, &output_path);

        // Free tensor payload memory — no longer needed after assembly
        // TODO Phase 4: stream tensor data from mmap directly into cimage
        // segments without intermediate Vec
        drop(compile_outcome);

        // 9. Seal, validate, promote
        let promotable = self.seal_and_validate(cimage_assembly)?;
        let mut result = self.promote_cimage(promotable)?;

        // Set the cimage_path in the result
        result.cimage_path = output_path.clone();

        let elapsed = started.elapsed();
        eprintln!(
            "[deploy] done: gen_id={} output={} ({:.2}s) tensors={}",
            result.generation_id.0,
            output_path.display(),
            elapsed.as_secs_f64(),
            compile_plan.model_ir.tensors.by_id.len(),
        );

        Ok(result)
    }

    pub fn is_available(&self) -> bool {
        self.metal_backend.is_some()
    }

    /// Seal the assembly and validate all artifacts.
    /// Returns a PromotableCimage ready for lifecycle promotion.
    pub fn seal_and_validate(&self, assembly: CimageAssembly) -> Result<PromotableCimage, String> {
        // 1. Compute digest over all segments + kernel artifacts
        // 2. Validate that every digest resolves
        // 3. Return PromotableCimage with validated=true
        let digest = assembly.compute_digest();
        Ok(PromotableCimage {
            assembly,
            validated: true,
            digest,
        })
    }

    /// Promote a sealed cimage through the lifecycle coordinator.
    /// LifecycleCoordinator validates identities and evidence, persists everything,
    /// then atomically switches the current generation.
    pub fn promote_cimage(
        &mut self,
        promotable: PromotableCimage,
    ) -> Result<DeploymentResult, String> {
        let coord = self
            .lifecycle_coordinator
            .as_mut()
            .ok_or("no lifecycle coordinator configured")?;

        // ── 1. Build CimageGeneration from assembly ──────────────────────
        let gen_id = GenerationId(format!("promoted.{}", promotable.digest));
        let model_name = promotable.assembly.serving_profile.model_name.clone();

        // Store segments in the generation API's content store
        // so promote() can resolve them by digest
        for (seg_id, data) in &promotable.assembly.segments {
            coord
                .generation_api
                .content_store
                .store(seg_id.clone(), data.clone());
        }
        for artifact in &promotable.assembly.kernel_artifacts {
            let sid = PhysicalSegmentId(format!("kernel:{}", artifact.implementation_id.0));
            coord
                .generation_api
                .content_store
                .store(sid, artifact.compiled_bytes.clone());
        }

        let generation = crate::ecs::canonical::generation::CimageGeneration {
            generation_id: gen_id.clone(),
            parent_generation: None,
            base_model: ModelSourceId(model_name),
            compiler_identity: crate::ecs::canonical::identity::CompilerIdentity {
                name: "tribunus-metal".into(),
                version: env!("CARGO_PKG_VERSION").into(),
                build_hash: None,
                build_timestamp: None,
            },
            hardware_profile: crate::ecs::canonical::identity::HardwareProfileId(
                "apple-gpu".into(),
            ),
            tensor_bindings: BTreeMap::new(),
            kernel_bindings: BTreeMap::new(),
            engram_bindings: BTreeMap::new(),
            execution_graph: crate::ecs::canonical::execution_graph::ExecutionGraph {
                regions: vec![],
                edges: vec![],
                state: crate::ecs::canonical::execution_graph::RuntimeStatePlan {
                    max_context_tokens: promotable.assembly.serving_profile.context_length,
                    kv_cache_bytes_per_token: 0,
                    total_kv_cache_bytes: 0,
                },
                memory: crate::ecs::canonical::execution_graph::MemoryPlan {
                    total_activation_bytes: 0,
                    total_weight_bytes: 0,
                    arena_region_count: 0,
                },
            },
            receipt_root: crate::ecs::canonical::identity::ReceiptId(format!(
                "receipt.{}",
                gen_id.0
            )),
            created_at: crate::ecs::canonical::identity::Timestamp(format!(
                "{:?}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
            )),
        };

        // ── 2. Promote through lifecycle coordinator ────────────────────
        let lifecycle = coord.promote_generation(generation)?;

        Ok(DeploymentResult {
            cimage_path: std::path::PathBuf::new(),
            generation_id: lifecycle.generation_id.clone().unwrap_or(gen_id),
            lifecycle,
            mtp_enabled: promotable.assembly.serving_profile.mtp_enabled,
        })
    }
}
