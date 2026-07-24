//! Adapter that wires [`PrismEngine`] internals into the [`PrefillDecodeRuntime`] trait.
//!
//! Provides the concrete implementation that the SSE streaming code in
//! [`crate::runtime::server::generate_stream`] needs to drive autoregressive
//! generation through the core inference engine, persisting the KV cache
//! across prefill and decode calls.

use crate::engine::cpu_executor::CPUGraphExecutor;
use crate::engine::ecs_engine::PrismEngine;
use crate::engine::inference::{InferenceEngine, KvCache};
use crate::engine::model::{Model, TensorInfo};
use crate::engine::sampling;
use crate::runtime::server::PrefillDecodeRuntime;
use crate::runtime::server_types::SamplingConfig;
use prism_ecs_compile::runtime::UnifiedRuntime;
use prism_ecs_ir::cimage_types::ExecutionGraph;
use prism_ecs_ir::model_graph::ModelGraph;
use prism_spatial_ir::execution::HeterogeneousExecutionReceipt;
use prism_spatial_ir::target::KernelManifest as SpatialKernelManifest;
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

fn encode_pixel_packet_as_png(
    packet: &prism_multimodal::media::MediaPacket,
    pixels: &[u8],
) -> Result<Vec<u8>, String> {
    let width = packet
        .descriptor
        .width
        .ok_or_else(|| "vision packet is missing width".to_string())?;
    let height = packet
        .descriptor
        .height
        .ok_or_else(|| "vision packet is missing height".to_string())?;
    let expected = width as usize * height as usize * 4;
    if pixels.len() != expected {
        return Err(format!(
            "vision packet has {} RGBA bytes, expected {expected}",
            pixels.len()
        ));
    }
    let mut rgba = pixels.to_vec();
    if matches!(
        packet.descriptor.format,
        prism_multimodal::media::PixelFormat::Bgra8
    ) {
        for pixel in rgba.chunks_exact_mut(4) {
            pixel.swap(0, 2);
        }
    }
    let image = image::RgbaImage::from_raw(width, height, rgba)
        .ok_or_else(|| "failed to construct RGBA image from packet".to_string())?;
    let mut encoded = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgba8(image)
        .write_to(&mut encoded, image::ImageFormat::Png)
        .map_err(|error| format!("failed to encode vision packet as PNG: {error}"))?;
    Ok(encoded.into_inner())
}

/// Runtime-relevant subset of a loaded CImage manifest.
///
/// Holds the execution graph, tensor metadata, and file path so that
/// [`WirePrefillDecodeRuntime`] can eventually dispatch through the CImage's
/// execution plan rather than constructing a blanket [`InferenceEngine`].
struct CImageManifest {
    /// Execution graph regions from the CImage, if present.
    execution_graph: Option<ExecutionGraph>,
    /// Canonical SpatialIR manifest emitted by the current compiler.
    canonical_manifest: Option<SpatialKernelManifest>,
    /// Tensor metadata keyed by name (e.g. `"model.layers.0.self_attn.q_proj.weight"`).
    tensor_metadata: HashMap<String, crate::engine::model::TensorInfo>,
    /// Path to the loaded CImage file.
    cimage_path: PathBuf,
}

impl CImageManifest {
    fn validate(&self) -> Result<(), String> {
        if let Some(manifest) = &self.canonical_manifest {
            let mut validated = 0usize;
            for plan in [&manifest.batch_plan, &manifest.realtime_plan]
                .into_iter()
                .flatten()
            {
                plan.validate().map_err(|error| {
                    format!("invalid canonical SpatialIR execution plan: {error}")
                })?;
                validated += 1;
            }
            if validated == 0 {
                return Err(
                    "canonical SpatialIR manifest has no executable batch or realtime plan".into(),
                );
            }
        }
        Ok(())
    }

    fn canonical_plan_counts(&self) -> (usize, usize) {
        self.canonical_manifest
            .as_ref()
            .map(|manifest| {
                (
                    manifest
                        .batch_plan
                        .as_ref()
                        .map(|plan| plan.fused_steps.len())
                        .unwrap_or(0),
                    manifest
                        .realtime_plan
                        .as_ref()
                        .map(|plan| plan.fused_steps.len())
                        .unwrap_or(0),
                )
            })
            .unwrap_or((0, 0))
    }
}

/// Adapter that wraps a [`PrismEngine`] as an autoregressive inference runtime.
///
/// Owns an [`InferenceEngine`] and a [`KvCache`] whose lifetimes span the
/// prefill and decode phases.  Interior mutability (`Mutex`) allows both
/// [`run_prefill`](PrefillDecodeRuntime::run_prefill) and
/// [`run_decode`](PrefillDecodeRuntime::run_decode) to take `&self` as the
/// trait requires.
///
/// Rather than holding a [`PrismEngine`] directly (which is `!Send` due to
/// the `MatmulProvider` closure), this struct extracts and owns the
/// `Model` and `ModelGraph` — both `Send + Sync` — at construction time.
pub struct WirePrefillDecodeRuntime {
    /// Model — used for tokenize/detokenize and [`InferenceEngine`] construction.
    model: Model,
    /// Compute graph — provides `num_layers` for KV-cache sizing.
    graph: ModelGraph,
    /// KV cache created during prefill and consumed during decode.
    kv_cache: Mutex<Option<KvCache>>,
    /// Inference engine created during prefill and reused during decode.
    inference_engine: Mutex<Option<InferenceEngine>>,
    /// CPU graph executor — both prefill and decode use this for the manifest path.
    cpu_executor: Mutex<CPUGraphExecutor>,
    /// Optional CImage execution manifest loaded via [`Self::from_cimage`].
    ///
    /// When `Some`, the runtime *may* dispatch through the CImage's execution
    /// graph instead of a monolithic [`InferenceEngine`].  Currently all
    /// regions fall through to the engine; this field exists for incremental
    /// migration.
    cimage_manifest: Option<CImageManifest>,
    /// Canonical SpatialIR runtime used when the CImage has no legacy graph.
    canonical_runtime: Mutex<Option<UnifiedRuntime>>,
    /// End-of-sequence token ID.
    eos_id: u32,
}

impl WirePrefillDecodeRuntime {
    /// Create a new wire runtime from a shared engine.
    ///
    /// Clones the engine's [`Model`] and [`ModelGraph`] (metadata-only — no
    /// tensor payload copying).  `eos_id` is the end-of-sequence token ID;
    /// use [`Self::detect_eos_id`] to derive it from model metadata, or pass
    /// `0` as a common default.
    pub fn from_engine(engine: &PrismEngine) -> Self {
        let eos_id = Self::detect_eos_id(engine);
        let hidden_size = Self::detect_hidden_size(&engine.model);
        let cpu_executor = Self::build_executor_from_model(&engine.model, hidden_size);
        WirePrefillDecodeRuntime {
            model: engine.model.clone(),
            graph: engine.graph.clone(),
            kv_cache: Mutex::new(None),
            inference_engine: Mutex::new(None),
            cpu_executor: Mutex::new(cpu_executor),
            cimage_manifest: None,
            canonical_runtime: Mutex::new(None),
            eos_id,
        }
    }

    /// Create a new wire runtime with explicit model, graph, and EOS token.
    pub fn new(model: Model, graph: ModelGraph, eos_id: u32) -> Self {
        let hidden_size = Self::detect_hidden_size(&model);
        let cpu_executor = Self::build_executor_from_model(&model, hidden_size);
        WirePrefillDecodeRuntime {
            model,
            graph,
            kv_cache: Mutex::new(None),
            inference_engine: Mutex::new(None),
            cpu_executor: Mutex::new(cpu_executor),
            cimage_manifest: None,
            canonical_runtime: Mutex::new(None),
            eos_id,
        }
    }

    /// Try to detect the EOS token ID from the engine's model metadata.
    ///
    /// Looks for `"eos_token_id"` in `model.metadata`.  Falls back to `0` if
    /// the key is absent or not a number.
    pub fn detect_eos_id(engine: &PrismEngine) -> u32 {
        engine.model.metadata["eos_token_id"]
            .as_u64()
            .map(|id| id as u32)
            .unwrap_or(0)
    }

    /// Construct a wire runtime from a CImage file path.
    ///
    /// Loads the model via [`Model::load`], parses the execution plan from the
    /// CImage header (if present), and stores the [`CImageManifest`] for
    /// future dispatch.  Falls back to direct [`InferenceEngine`] operation
    /// when no execution plan exists in the file.
    pub fn from_cimage(path: &Path, graph: ModelGraph, eos_id: u32) -> Result<Self, String> {
        let model = Model::load(path)?;

        // Parse the execution graph from the header's execution_plan JSON.
        let execution_graph: Option<ExecutionGraph> = model
            .metadata
            .get("execution_plan")
            .and_then(|v| v.as_str())
            .and_then(|json| serde_json::from_str(json).ok());
        let canonical_manifest: Option<SpatialKernelManifest> = model
            .metadata
            .get("execution_plan")
            .and_then(|value| value.as_str())
            .and_then(|json| serde_json::from_str(json).ok());

        let manifest = CImageManifest {
            execution_graph,
            canonical_manifest,
            tensor_metadata: model.tensors.clone(),
            cimage_path: path.to_path_buf(),
        };
        manifest.validate()?;
        let hidden_size = Self::detect_hidden_size(&model);
        let cpu_executor = Self::build_executor_from_model(&model, hidden_size);

        let canonical_runtime =
            if manifest.canonical_manifest.is_some() && manifest.execution_graph.is_none() {
                let runtime = UnifiedRuntime::new(
                    prism_ecs_compile::runtime::RuntimeModel::load(path)
                        .map_err(|error| format!("load canonical runtime: {error}"))?,
                );
                runtime
                    .validate_aot_schedule()
                    .map_err(|error| format!("validate canonical runtime: {error}"))?;
                Some(runtime)
            } else {
                None
            };

        Ok(WirePrefillDecodeRuntime {
            model,
            graph,
            kv_cache: Mutex::new(None),
            inference_engine: Mutex::new(None),
            cpu_executor: Mutex::new(cpu_executor),
            cimage_manifest: Some(manifest),
            canonical_runtime: Mutex::new(canonical_runtime),
            eos_id,
        })
    }

    /// Return a human-readable summary of the loaded execution manifest.
    ///
    /// Returns the number of regions, operations, and tensor count when a
    /// CImage manifest is present, or a "not loaded" message otherwise.
    pub fn execution_plan_summary(&self) -> String {
        match &self.cimage_manifest {
            None => "CImage manifest not loaded — using InferenceEngine directly".to_string(),
            Some(manifest) => {
                let tensor_count = manifest.tensor_metadata.len();
                if manifest.canonical_manifest.is_some() {
                    let (batch_steps, realtime_steps) = manifest.canonical_plan_counts();
                    let mut selected_route = String::new();
                    if let Ok(runtime_guard) = self.canonical_runtime.lock() {
                        if let Some(runtime) = runtime_guard.as_ref() {
                            if let Some(graph) = runtime
                                .selected_execution_graph()
                                .filter(|g| !g.profiles.is_empty())
                            {
                                let route = graph
                                    .route_sequence
                                    .iter()
                                    .map(|lane| format!("{lane:?}"))
                                    .collect::<Vec<_>>()
                                    .join(",");
                                selected_route = format!(
                                    " selected route [{}], {} selected profile(s)",
                                    route,
                                    graph.profiles.len()
                                );
                            }
                        }
                    }
                    let selected_route = if selected_route.is_empty() {
                        String::new()
                    } else {
                        format!(" {selected_route}")
                    };
                    return format!(
                        "CImage loaded from {} ({} tensors) — canonical SpatialIR manifest admitted with {} batch and {} realtime fused steps; server KV runtime uses validated compatibility fallback;{}",
                        manifest.cimage_path.display(),
                        tensor_count,
                        batch_steps,
                        realtime_steps,
                        selected_route,
                    );
                }
                match &manifest.execution_graph {
                    None => {
                        format!(
                            "CImage loaded from {} ({} tensors) — no execution graph in header, \
                             using InferenceEngine",
                            manifest.cimage_path.display(),
                            tensor_count,
                        )
                    }
                    Some(graph) => {
                        let region_count = graph.regions.len();
                        let op_count: usize =
                            graph.regions.iter().map(|r| r.operations.len()).sum();
                        format!(
                            "CImage loaded from {} — {} tensor(s), {} region(s), {} operation(s), \
                             cache: {} ctx tokens, {} kB KV-cache, {} kB weights",
                            manifest.cimage_path.display(),
                            tensor_count,
                            region_count,
                            op_count,
                            graph.state.max_context_tokens,
                            graph.state.total_kv_cache_bytes / 1024,
                            graph.memory.total_weight_bytes / 1024,
                        )
                    }
                }
            }
        }
    }

    /// Replay the canonical CImage schedule through the assembled Apple
    /// backend routes and return measured scheduler evidence. This is kept
    /// separate from token generation because the schedule's output tensor
    /// contract may represent an intermediate fused island rather than final
    /// vocabulary logits.
    pub fn replay_canonical_aot(&self) -> Result<HeterogeneousExecutionReceipt, String> {
        let mut runtime = self
            .canonical_runtime
            .lock()
            .map_err(|error| format!("canonical runtime lock: {error}"))?;
        runtime
            .as_mut()
            .ok_or_else(|| "CImage has no canonical AOT runtime".to_string())?
            .replay_aot_apple()
            .map_err(|error| format!("canonical AOT replay: {error}"))
    }

    /// Replay the canonical CImage schedule with workload-specific fusion
    /// strategy selection (for example realtime batch-1 versus batched
    /// throughput workloads).
    pub fn replay_canonical_aot_for_workload(
        &self,
        scenario: prism_spatial_ir::WorkloadScenario,
    ) -> Result<HeterogeneousExecutionReceipt, String> {
        let mut runtime = self
            .canonical_runtime
            .lock()
            .map_err(|error| format!("canonical runtime lock: {error}"))?;
        runtime
            .as_mut()
            .ok_or_else(|| "CImage has no canonical AOT runtime".to_string())?
            .replay_aot_apple_for_workload(scenario)
            .map_err(|error| format!("canonical workload AOT replay: {error}"))
    }

    /// Install a measured UOp strategy choice into the canonical runtime for
    /// one workload scenario. The sealed CImage remains immutable; only the
    /// in-memory serving policy changes.
    pub fn install_measured_strategy_for_workload(
        &self,
        scenario: prism_spatial_ir::WorkloadScenario,
        strategies: &[prism_spatial_ir::FusionStrategy],
        measurements: &[prism_spatial_ir::FusionMeasurement],
    ) -> Result<String, String> {
        let mut runtime = self
            .canonical_runtime
            .lock()
            .map_err(|error| format!("canonical runtime lock: {error}"))?;
        runtime
            .as_mut()
            .ok_or_else(|| "CImage has no canonical AOT runtime".to_string())?
            .install_measured_strategy_choice(scenario, strategies, measurements)
    }

    /// Execute a batch workload while selecting the measured strategy for
    /// the explicit batch shape.
    pub fn run_batch_for_workload(
        &self,
        input_tokens: &[u32],
        batch_size: u32,
    ) -> Result<Vec<f32>, String> {
        let mut runtime = self
            .canonical_runtime
            .lock()
            .map_err(|error| format!("canonical runtime lock: {error}"))?;
        runtime
            .as_mut()
            .ok_or_else(|| "CImage has no canonical AOT runtime".to_string())?
            .run_batch_for_workload(input_tokens, batch_size)
            .map_err(|error| format!("batch workload execution: {error}"))
    }

    /// Report the effective measured strategy for a workload, including the
    /// deterministic nearest-shape fallback used by the canonical runtime.
    pub fn selected_measured_strategy_for_workload(
        &self,
        scenario: prism_spatial_ir::WorkloadScenario,
    ) -> Result<Option<String>, String> {
        let runtime = self
            .canonical_runtime
            .lock()
            .map_err(|error| format!("canonical runtime lock: {error}"))?;
        Ok(runtime
            .as_ref()
            .ok_or_else(|| "CImage has no canonical AOT runtime".to_string())?
            .selected_measured_strategy(scenario)
            .map(str::to_owned))
    }

    // ── internal helpers ──────────────────────────────────────────────

    /// Build a `CPUGraphExecutor` by reading all tensor data from the model file.
    fn build_executor_from_model(model: &Model, hidden_size: usize) -> CPUGraphExecutor {
        let tensors: HashMap<String, Vec<u8>> = model
            .tensors
            .iter()
            .map(|(name, info)| {
                let data = Self::read_tensor_data_from_path(&model.path, info).unwrap_or_default();
                (name.clone(), data)
            })
            .collect();
        let tensor_types: HashMap<String, String> = model
            .tensors
            .iter()
            .map(|(name, info)| {
                (
                    name.clone(),
                    Self::tensor_type_to_format_str(&info.tensor_type),
                )
            })
            .collect();
        CPUGraphExecutor::new(tensors, tensor_types, hidden_size)
    }

    /// Read raw tensor bytes from the `.cimage` file for a `TensorInfo`.
    fn read_tensor_data_from_path(path: &Path, info: &TensorInfo) -> Result<Vec<u8>, String> {
        let mut file = std::fs::File::open(path)
            .map_err(|e| format!("open model file for tensor read: {e}"))?;
        file.seek(SeekFrom::Start(info.offset))
            .map_err(|e| format!("seek to tensor offset {}: {}", info.offset, e))?;
        let mut data = vec![0u8; info.size as usize];
        file.read_exact(&mut data)
            .map_err(|e| format!("read tensor data ({} bytes): {}", info.size, e))?;
        Ok(data)
    }

    /// Detect hidden size from model tensor metadata.
    fn detect_hidden_size(model: &Model) -> usize {
        for (key, info) in &model.tensors {
            if key.contains("q_proj.weight") {
                return info.dim_n as usize;
            }
        }
        // Fallback: embed_tokens weight's dim_n is also hidden_size
        for (key, info) in &model.tensors {
            if key.ends_with("embed_tokens.weight") {
                return info.dim_n as usize;
            }
        }
        4096
    }

    /// Map a cimage `TensorType` to the format string expected by
    /// `tensor_format_from_str` in the CPU graph executor.
    fn tensor_type_to_format_str(t: &prism_ecs_quantization::cimage::TensorType) -> String {
        use prism_ecs_quantization::cimage::TensorType;
        match t {
            TensorType::StandardFP16 | TensorType::Blob => "FP16",
            TensorType::Palettized4Bit => "PALETTIZED4BIT",
            TensorType::Ternary158 => "TERNARY158",
            TensorType::TernaryTile640 => "TERNARYTILE640",
            TensorType::Binary1 => "BINARY1",
            TensorType::NF4 => "NF4",
            TensorType::Int4 => "INT4",
            TensorType::FP8 => "FP16",
            TensorType::Bf16 => "BF16",
            TensorType::Int8 => "INT8",
            TensorType::Nf8 => "NF8",
        }
        .to_string()
    }

    /// Tokenize text using this struct's model metadata.
    /// Mirrors [`PrismEngine::tokenize`].
    fn tokenize_inner(&self, text: &str) -> Result<Vec<u32>, String> {
        let tokenizer_path = self.model.metadata["tokenizer_path"].as_str().unwrap_or("");
        if tokenizer_path.is_empty() {
            // Fallback: simple whitespace split for testing
            return Ok(text.split_whitespace().map(|_| 1u32).collect());
        }
        let tokenizer = crate::engine::bpe_tokenizer::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| format!("load tokenizer: {e}"))?;
        let encoding = tokenizer
            .encode(text, true)
            .map_err(|e| format!("encode: {e}"))?;
        Ok(encoding.ids)
    }

    /// Detokenize a single token ID using this struct's model metadata.
    /// Mirrors [`PrismEngine::detokenize`].
    fn detokenize_inner(&self, ids: &[u32]) -> Result<String, String> {
        let tokenizer_path = self.model.metadata["tokenizer_path"].as_str().unwrap_or("");
        if tokenizer_path.is_empty() {
            return Ok(format!("<token {}>", ids.first().copied().unwrap_or(0)));
        }
        let tokenizer = crate::engine::bpe_tokenizer::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| format!("load tokenizer: {e}"))?;
        tokenizer
            .decode(ids, true)
            .map_err(|e| format!("decode: {e}"))
    }

    /// Convert a [`SamplingConfig`] (server types, with repetition_penalty) to the
    /// engine's [`crate::engine::inference::SamplingConfig`].
    fn to_engine_sampling(config: &SamplingConfig) -> crate::engine::inference::SamplingConfig {
        crate::engine::inference::SamplingConfig {
            temperature: config.temperature,
            top_k: config.top_k,
            top_p: config.top_p,
            // NOTE: runtime SamplingConfig also carries repetition_penalty;
            // the engine's SamplingConfig does not support it yet.
        }
    }

    /// Prefill text plus already-projected modality rows through the same
    /// transformer and synchronized KV cache used by ordinary generation.
    pub fn run_prefill_conditioned(
        &self,
        prompt_tokens: &[u32],
        modality_rows: &[Vec<f32>],
    ) -> Result<Vec<f32>, String> {
        if prompt_tokens.is_empty() {
            return Err("conditioned prefill requires non-empty text tokens".into());
        }
        let inference_engine = InferenceEngine::new(self.model.clone());
        let hidden_size = Self::detect_hidden_size(&self.model);
        let mut embeddings = inference_engine.embed(prompt_tokens)?;
        for row in modality_rows {
            if row.len() != hidden_size {
                return Err(format!(
                    "conditioned modality row has {}, expected {} values",
                    row.len(),
                    hidden_size
                ));
            }
            embeddings.extend_from_slice(row);
        }
        let max_seq_len = embeddings.len() / hidden_size + 2048;
        let mut kv_cache = KvCache::new(self.graph.num_layers as usize, 32, 128, max_seq_len);
        let logits = inference_engine.forward_embeddings(&embeddings, &mut kv_cache)?;
        *self
            .inference_engine
            .lock()
            .map_err(|e| format!("inference_engine lock: {e}"))? = Some(inference_engine);
        *self
            .kv_cache
            .lock()
            .map_err(|e| format!("kv_cache lock: {e}"))? = Some(kv_cache);
        Ok(logits)
    }

    /// Project native vision/audio rows using the CImage's learned adapter
    /// tensors, then execute conditioned prefill.
    pub fn run_prefill_conditioned_features(
        &self,
        prompt_tokens: &[u32],
        vision_rows: &[Vec<f32>],
        audio_rows: &[Vec<f32>],
    ) -> Result<Vec<f32>, String> {
        let engine = InferenceEngine::new(self.model.clone());
        let hidden_size = Self::detect_hidden_size(&self.model);
        let mut projected = Vec::new();
        for (rows, candidates) in [
            (
                vision_rows,
                [
                    "vision_projector.weight",
                    "multi_modal_projector.weight",
                    "mm_projector.weight",
                ]
                .as_slice(),
            ),
            (
                audio_rows,
                ["audio_projector.weight", "audio_projection.weight"].as_slice(),
            ),
        ] {
            for row in rows {
                let row = match engine.project_modality(candidates, row)? {
                    Some(row) => row,
                    None if row.len() == hidden_size => row.clone(),
                    None => {
                        return Err(format!(
                            "no projector for modality row of {} values",
                            row.len()
                        ))
                    }
                };
                if row.len() != hidden_size {
                    return Err(format!(
                        "projector returned {}, expected {hidden_size}",
                        row.len()
                    ));
                }
                projected.push(row);
            }
        }
        self.run_prefill_conditioned(prompt_tokens, &projected)
    }

    /// Project named cross-model feature rows through the exact adapter tensor
    /// declared by the owning CImage manifest, then append them to the text
    /// sequence. This is the typed runtime boundary for specialist-model
    /// outputs; callers must supply the declared source dimension.
    pub fn run_prefill_conditioned_named_features(
        &self,
        prompt_tokens: &[u32],
        named_rows: &[(&str, &[f32])],
    ) -> Result<Vec<f32>, String> {
        let engine = InferenceEngine::new(self.model.clone());
        let mut projected = Vec::with_capacity(named_rows.len());
        for (tensor_name, row) in named_rows {
            let output = engine
                .project_modality(&[*tensor_name], row)?
                .ok_or_else(|| format!("fusion adapter tensor not found: {tensor_name}"))?;
            projected.push(output);
        }
        self.run_prefill_conditioned(prompt_tokens, &projected)
    }

    /// Consume materialized media packets at the runtime boundary. Audio is
    /// converted through the shared packet feature adapter; image/video
    /// packets require their model-specific encoder and are rejected until
    /// one is declared and loaded.
    pub fn run_prefill_conditioned_packets(
        &self,
        prompt_tokens: &[u32],
        packets: &[prism_multimodal::media::MediaPacket],
    ) -> Result<Vec<f32>, String> {
        self.run_prefill_conditioned_packets_for_model(None, prompt_tokens, packets)
    }

    pub fn run_prefill_conditioned_packets_for_model(
        &self,
        model_id: Option<&str>,
        prompt_tokens: &[u32],
        packets: &[prism_multimodal::media::MediaPacket],
    ) -> Result<Vec<f32>, String> {
        let mut audio_rows = Vec::new();
        for packet in packets {
            if let Some(expected) = model_id {
                if packet.model_id.as_deref() != Some(expected) {
                    return Err(format!(
                        "media packet model namespace {:?} does not match runtime {expected:?}",
                        packet.model_id
                    ));
                }
            }
            match packet.descriptor.kind {
                prism_multimodal::media::MediaKind::Audio => {
                    audio_rows.extend(
                        prism_multimodal::io::audio_packet_features(packet)
                            .map_err(|error| error.to_string())?,
                    );
                }
                kind => {
                    return Err(format!(
                        "no runtime encoder registered for {kind:?} media packet"
                    ));
                }
            }
        }
        self.run_prefill_conditioned_features(prompt_tokens, &[], &audio_rows)
    }

    pub fn run_prefill_conditioned_image(
        &self,
        prompt_tokens: &[u32],
        image_bytes: &[u8],
        config: &prism_multimodal::multimodal::vision_encoder::VisionEncoderConfig,
        weights: &std::collections::HashMap<String, Vec<f32>>,
        matmul: &prism_multimodal::multimodal::vision_encoder::MatmulProvider,
    ) -> Result<Vec<f32>, String> {
        let embedding = prism_multimodal::multimodal::vision_encoder::encode_image(
            image_bytes,
            config,
            weights,
            matmul,
        )?;
        let rows = embedding
            .chunks(config.hidden_dim as usize)
            .map(Vec::from)
            .collect::<Vec<_>>();
        self.run_prefill_conditioned_features(prompt_tokens, &rows, &[])
    }

    /// Extract native modality features without projecting them into this
    /// runtime's language-model hidden space. These rows are the typed output
    /// consumed by another model namespace through a manifest fusion binding.
    pub fn extract_packet_feature_rows(
        &self,
        model_id: Option<&str>,
        packets: &[prism_multimodal::media::MediaPacket],
        vision: Option<(
            &prism_multimodal::multimodal::vision_encoder::VisionEncoderConfig,
            &std::collections::HashMap<String, Vec<f32>>,
            &prism_multimodal::multimodal::vision_encoder::MatmulProvider,
        )>,
    ) -> Result<Vec<Vec<f32>>, String> {
        let mut rows = Vec::new();
        for packet in packets {
            if let Some(expected) = model_id {
                if packet.model_id.as_deref() != Some(expected) {
                    return Err(format!(
                        "media packet namespace does not match runtime {expected:?}"
                    ));
                }
            }
            match packet.descriptor.kind {
                prism_multimodal::media::MediaKind::Audio => rows.extend(
                    prism_multimodal::io::audio_packet_features(packet)
                        .map_err(|error| error.to_string())?,
                ),
                prism_multimodal::media::MediaKind::Image
                | prism_multimodal::media::MediaKind::Video
                    if matches!(
                        packet.descriptor.format,
                        prism_multimodal::media::PixelFormat::Rgba8
                            | prism_multimodal::media::PixelFormat::Bgra8
                    ) =>
                {
                    let (config, weights, matmul) = vision.ok_or_else(|| {
                        "vision feature extraction requires a declared vision encoder".to_string()
                    })?;
                    let native_payload;
                    let pixel_payload;
                    let image_payload = if packet.payload.is_empty() {
                        native_payload = packet
                            .native_video
                            .as_ref()
                            .ok_or_else(|| {
                                "image packet has neither payload nor native buffer".to_string()
                            })?
                            .copy_rgba()?;
                        pixel_payload = encode_pixel_packet_as_png(packet, &native_payload)?;
                        &pixel_payload
                    } else {
                        pixel_payload = encode_pixel_packet_as_png(packet, &packet.payload)?;
                        &pixel_payload
                    };
                    let embedding = prism_multimodal::multimodal::vision_encoder::encode_image(
                        image_payload,
                        config,
                        weights,
                        matmul,
                    )?;
                    rows.extend(embedding.chunks(config.hidden_dim as usize).map(Vec::from));
                }
                kind => {
                    return Err(format!(
                        "no feature extractor registered for {kind:?} packet"
                    ))
                }
            }
        }
        Ok(rows)
    }

    pub fn run_prefill_conditioned_packets_with_vision(
        &self,
        model_id: Option<&str>,
        prompt_tokens: &[u32],
        packets: &[prism_multimodal::media::MediaPacket],
        vision_config: &prism_multimodal::multimodal::vision_encoder::VisionEncoderConfig,
        vision_weights: &std::collections::HashMap<String, Vec<f32>>,
        matmul: &prism_multimodal::multimodal::vision_encoder::MatmulProvider,
    ) -> Result<Vec<f32>, String> {
        let mut vision_rows = Vec::new();
        let mut audio_rows = Vec::new();
        for packet in packets {
            if let Some(expected) = model_id {
                if packet.model_id.as_deref() != Some(expected) {
                    return Err(format!(
                        "media packet namespace does not match runtime {expected:?}"
                    ));
                }
            }
            match packet.descriptor.kind {
                prism_multimodal::media::MediaKind::Audio => audio_rows.extend(
                    prism_multimodal::io::audio_packet_features(packet)
                        .map_err(|error| error.to_string())?,
                ),
                prism_multimodal::media::MediaKind::Image
                | prism_multimodal::media::MediaKind::Video
                    if matches!(
                        packet.descriptor.format,
                        prism_multimodal::media::PixelFormat::Rgba8
                            | prism_multimodal::media::PixelFormat::Bgra8
                    ) =>
                {
                    let native_payload;
                    let image_payload = if packet.payload.is_empty() {
                        native_payload = packet
                            .native_video
                            .as_ref()
                            .ok_or_else(|| {
                                "image packet has neither payload nor native buffer".to_string()
                            })?
                            .copy_rgba()?;
                        &native_payload
                    } else {
                        &packet.payload
                    };
                    let embedding = prism_multimodal::multimodal::vision_encoder::encode_image(
                        image_payload,
                        vision_config,
                        vision_weights,
                        matmul,
                    )?;
                    vision_rows.extend(
                        embedding
                            .chunks(vision_config.hidden_dim as usize)
                            .map(Vec::from),
                    );
                }
                kind => {
                    return Err(format!(
                        "no temporal encoder registered for {kind:?} packet"
                    ))
                }
            }
        }
        self.run_prefill_conditioned_features(prompt_tokens, &vision_rows, &audio_rows)
    }
}

impl PrefillDecodeRuntime for WirePrefillDecodeRuntime {
    fn tokenize(&self, prompt: &str) -> Result<Vec<u32>, String> {
        self.tokenize_inner(prompt)
    }

    fn embed_text(&self, prompt: &str) -> Result<Vec<f32>, String> {
        let tokens = self.tokenize_inner(prompt)?;
        if tokens.is_empty() {
            return Err("embedding input tokenized to an empty sequence".into());
        }
        let engine = InferenceEngine::new(self.model.clone());
        let rows = engine.embed(&tokens)?;
        let hidden = rows.len() / tokens.len();
        if hidden == 0 || rows.len() != tokens.len() * hidden {
            return Err("embedding model returned invalid token embedding shape".into());
        }
        let mut pooled = vec![0.0; hidden];
        for row in rows.chunks_exact(hidden) {
            for (dst, value) in pooled.iter_mut().zip(row) {
                *dst += *value;
            }
        }
        let scale = 1.0 / tokens.len() as f32;
        for value in &mut pooled {
            *value *= scale;
        }
        let norm = pooled.iter().map(|value| value * value).sum::<f32>().sqrt();
        if norm > f32::EPSILON {
            for value in &mut pooled {
                *value /= norm;
            }
        }
        Ok(pooled)
    }

    fn run_prefill(&self, prompt_tokens: &[u32]) -> Result<Vec<f32>, String> {
        if let Some(runtime) = self
            .canonical_runtime
            .lock()
            .map_err(|error| format!("canonical runtime lock: {error}"))?
            .as_mut()
        {
            let logits = runtime
                .run_prefill_logits(prompt_tokens)
                .map_err(|error| format!("canonical prefill: {error}"))?;
            *self
                .kv_cache
                .lock()
                .map_err(|error| format!("kv cache lock: {error}"))? = Some(KvCache::new(
                self.graph.num_layers as usize,
                32,
                128,
                prompt_tokens.len() + 2048,
            ));
            return Ok(logits);
        }

        let inference_engine = InferenceEngine::new(self.model.clone());

        // KV-cache geometry — uses the same defaults as PrismEngine::generate.
        let num_layers = self.graph.num_layers as usize;
        let num_kv_heads: usize = 32;
        let head_dim: usize = 128;
        // Leave generous headroom for decode tokens.
        let max_seq_len = prompt_tokens.len() + 2048;

        let mut kv_cache = KvCache::new(num_layers, num_kv_heads, head_dim, max_seq_len);

        let logits = if let Some(manifest) = &self.cimage_manifest {
            if let Some(graph) = &manifest.execution_graph {
                eprintln!(
                    "[runtime] Dispatching through CImage execution plan: {}",
                    self.execution_plan_summary()
                );
                // Initial hidden state from token embedding.
                let hidden = inference_engine.embed(prompt_tokens)?;
                // Use CPU graph executor — handles both prefill and decode
                // with the same operation dispatch as the batch path.
                self.cpu_executor
                    .lock()
                    .map_err(|e| format!("CPU executor lock: {e}"))?
                    .execute(graph, &hidden)?
            } else {
                // Manifest loaded but no execution graph — fall through.
                inference_engine.forward(prompt_tokens, &mut kv_cache)?
            }
        } else {
            // No manifest — fall through.
            inference_engine.forward(prompt_tokens, &mut kv_cache)?
        };

        // Store for decode phase.
        *self
            .inference_engine
            .lock()
            .map_err(|e| format!("inference_engine lock: {e}"))? = Some(inference_engine);
        *self
            .kv_cache
            .lock()
            .map_err(|e| format!("kv_cache lock: {e}"))? = Some(kv_cache);

        Ok(logits)
    }

    fn run_decode(&self, token: u32) -> Result<Vec<f32>, String> {
        if let Some(runtime) = self
            .canonical_runtime
            .lock()
            .map_err(|error| format!("canonical runtime lock: {error}"))?
            .as_mut()
        {
            return runtime
                .run_decode_logits_for_token(token)
                .map_err(|error| format!("canonical decode: {error}"));
        }

        let inference_engine = self
            .inference_engine
            .lock()
            .map_err(|e| format!("inference_engine lock: {e}"))?;
        let mut kv_cache = self
            .kv_cache
            .lock()
            .map_err(|e| format!("kv_cache lock: {e}"))?;

        let engine = inference_engine
            .as_ref()
            .ok_or_else(|| "run_decode called before run_prefill".to_string())?;
        let cache = kv_cache
            .as_mut()
            .ok_or_else(|| "run_decode called before run_prefill".to_string())?;

        // ── Manifest-driven decode ──────────────────────────────────
        // Start from the token embedding, not from forward() logits.
        // This avoids double-executing the model and mutating KV cache twice.
        if let Some(manifest) = &self.cimage_manifest {
            if let Some(graph) = &manifest.execution_graph {
                let hidden = engine.embed(&[token])?;
                return self
                    .cpu_executor
                    .lock()
                    .map_err(|e| format!("CPU executor lock: {e}"))?
                    .execute(graph, &hidden);
            }
        }

        // No manifest — fall through to direct forward.
        engine.forward(&[token], cache)
    }

    fn sample(&self, logits: &[f32], config: &SamplingConfig) -> Result<u32, String> {
        Ok(sampling::sample(logits, &Self::to_engine_sampling(config)))
    }

    fn detokenize(&self, token: u32) -> Result<String, String> {
        self.detokenize_inner(&[token])
    }

    fn eos_token_id(&self) -> u32 {
        self.eos_id
    }
}
