//! UniFFI bridge between tribunus-compute-core (agent + tools) and the
//! PrismAgent Swift app.  All types here are annotated with `#[derive(uniffi::Enum)]`
//! or `#[derive(uniffi::Record)]` so UniFFI generates native Swift equivalents.
//!
//! # Usage
//!
//! ```bash
//! # Build the dynamic library
//! cargo build --release -p prism-bridge
//!
//! # Generate Swift bindings
//! cargo run --bin uniffi-bindgen generate \
//!   --library target/release/libprism_bridge.dylib \
//!   --language swift \
//!   --out-dir ./swift-bindings
//! ```
//!
//! Drag the generated `.swift` and `.h` files into your Xcode project.
use std::sync::Arc;
use tribunus_compute_core::agent;
#[cfg(target_os = "macos")]
use tribunus_compute_core::backend::create_inference_executor;
#[cfg(target_os = "macos")]
use tribunus_compute_core::backend::heterogeneous_executor::HeterogeneousExecutor;
#[cfg(target_os = "macos")]
use tribunus_compute_core::backend::routing::*;
use tribunus_compute_core::compute_image::cimage_loader::CimageDeployment;
use tribunus_compute_core::config::operation_route::OperationRoute;
use tribunus_compute_core::config::{
    self, CompileQuantMode, GenerationRegime, HardwareTarget, KvCacheMode, ServerConfig,
};
use tribunus_compute_core::device::{
    self, BackendKind, DeviceKind, DeviceMemoryInfo, PcieLinkInfo,
};
use tribunus_compute_core::tts::pipeline::TtsPipeline;
use tribunus_compute_core::tools;

/// Errors that can cross the UniFFI boundary.
#[derive(Debug, Clone, uniffi::Error, thiserror::Error)]
#[uniffi(flat_error)]
pub enum BridgeError {
    #[error("Cimage load failed: {0}")]
    CimageLoadFailed(String),
}

// ── UniFFI scaffold ───────────────────────────────────────────────────
uniffi::setup_scaffolding!();

// ═══════════════════════════════════════════════════════════════════════
// Agent types
// ═══════════════════════════════════════════════════════════════════════

/// Current phase of the agent state machine.
#[derive(uniffi::Enum, Clone)]
pub enum BridgePhase {
    Idle,
    Generating,
    AwaitingTools,
    AwaitingSubagents,
    Done,
}

/// A tool call emitted by the model.
#[derive(uniffi::Record, Clone)]
pub struct BridgeToolCall {
    pub id: String,
    pub name: String,
    pub arguments_json: String,
}

/// A spawned subagent.
#[derive(uniffi::Record, Clone)]
pub struct BridgeSubagentHandle {
    pub id: u64,
    pub goal: String,
    pub sandbox_subpath: String,
    pub tool_allowlist: Vec<String>,
    pub max_revisions: u8,
}

/// Serializable agent state.
#[derive(uniffi::Record, Clone)]
pub struct BridgeAgentState {
    pub phase: BridgePhase,
    pub history_jsonl: String,
    pub current_prompt: String,
}

/// Outcome of one `prism_agent_step` call.
#[derive(uniffi::Enum, Clone)]
pub enum BridgeStepOutcome {
    Generating,
    AwaitingTools {
        tools: Vec<BridgeToolCall>,
    },
    AwaitingSubagents {
        subagents: Vec<BridgeSubagentHandle>,
    },
    Finished {
        result: String,
    },
}

/// Combined result payload returned to Swift.
#[derive(uniffi::Record)]
pub struct BridgeStepResult {
    pub state: BridgeAgentState,
    pub outcome: BridgeStepOutcome,
}

/// A tool definition for the model.
#[derive(uniffi::Record, Clone)]
pub struct BridgeToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters_json: String,
}

// ═══════════════════════════════════════════════════════════════════════
// Exported functions
// ═══════════════════════════════════════════════════════════════════════

/// Drive one step of the agent state machine.
///
/// Takes the current serialised state and the model's output text, runs the
/// pure state transition, and returns the new state + outcome.  The app is
/// responsible for calling inference externally and feeding the output here.
#[uniffi::export]
pub fn prism_agent_step(state_json: String, model_output: String) -> BridgeStepResult {
    // ── Deserialise ─────────────────────────────────────────────────
    let mut inner: agent::AgentState = match serde_json::from_str(&state_json) {
        Ok(s) => s,
        Err(e) => {
            return error_result(&format!("deserialise state: {e}"));
        }
    };

    // ── Step ────────────────────────────────────────────────────────
    let outcome = match agent::step(&mut inner, &model_output) {
        Ok(o) => o,
        Err(e) => {
            return error_result(&format!("step failed: {e}"));
        }
    };

    // ── Serialise outcome ───────────────────────────────────────────
    let bridge_outcome = bridge_outcome_from(&outcome, &inner);

    // ── Build state payload ─────────────────────────────────────────
    let prompt = agent::build_agent_prompt(&inner.messages, &inner.tools);
    let history = serde_json::to_string(&inner.messages).unwrap_or_default();

    let bridge_state = BridgeAgentState {
        phase: bridge_phase_from(&inner.phase),
        history_jsonl: history,
        current_prompt: prompt,
    };

    BridgeStepResult {
        state: bridge_state,
        outcome: bridge_outcome,
    }
}

/// Return the default set of sandbox file tools (read_file, write_file, etc.)
/// as JSON strings that the model can consume.
#[uniffi::export]
pub fn prism_default_tools() -> Vec<BridgeToolDefinition> {
    let mut all = Vec::new();

    // File sandbox tools
    for t in tools::default_sandbox_tools() {
        all.push(BridgeToolDefinition {
            name: t.name,
            description: t.description,
            parameters_json: serde_json::to_string(&t.parameters).unwrap_or_default(),
        });
    }

    // Web browser tools (executed on Swift's WKWebView via the adapter)
    for t in web_tool_defs() {
        all.push(BridgeToolDefinition {
            name: t.name.clone(),
            description: t.description.clone(),
            parameters_json: serde_json::to_string(&t.parameters).unwrap_or_default(),
        });
    }

    all
}

fn web_tool_defs() -> Vec<tools::ToolDefinition> {
    vec![
        tools::ToolDefinition {
            name: "web_navigate".into(),
            description: "Navigate the browser to a URL.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to navigate to"}
                },
                "required": ["url"]
            }),
            required: vec!["url".into()],
        },
        tools::ToolDefinition {
            name: "web_snapshot".into(),
            description: "Take a semantic snapshot of the current page. Returns a JSON tree of content and interactive elements, each with a unique 'id' field.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
            required: vec![],
        },
        tools::ToolDefinition {
            name: "web_interact".into(),
            description: "Interact with a page element by its 'id' from the last web_snapshot.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "id": {"type": "integer", "description": "The target element ID from web_snapshot"},
                    "action": {"type": "string", "enum": ["click", "type", "focus"], "description": "What to do with the element"},
                    "value": {"type": "string", "description": "Text to type if action is 'type'"}
                },
                "required": ["id", "action"]
            }),
            required: vec!["id".into(), "action".into()],
        },
        tools::ToolDefinition {
            name: "web_evaluate_js".into(),
            description: "Execute arbitrary JavaScript in the current page context and return the result.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "script": {"type": "string", "description": "JavaScript code to execute"}
                },
                "required": ["script"]
            }),
            required: vec!["script".into()],
        },
        tools::ToolDefinition {
            name: "web_extract_media".into(),
            description: "Extract raw pixel data from a media element (IMG or VIDEO) by its 'id'.  Returns a file path the agent can read_file.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "id": {"type": "integer", "description": "The target media element ID from web_snapshot"}
                },
                "required": ["id"]
            }),
            required: vec!["id".into()],
        },
        tools::ToolDefinition {
            name: "web_download".into(),
            description: "Download a file from a URL using the browser's authenticated session. Useful for PDFs, CSVs, ZIPs, or any file behind a login portal.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL of the file to download"},
                    "filename": {"type": "string", "description": "Desired filename for the downloaded file"}
                },
                "required": ["url", "filename"]
            }),
            required: vec!["url".into(), "filename".into()],
        },
    ]
}

// ═══════════════════════════════════════════════════════════════════════
// AOT compilation
// ═══════════════════════════════════════════════════════════════════════

/// Errors during GGUF→cimage compilation.
#[derive(Debug, uniffi::Error, thiserror::Error)]
#[uniffi(flat_error)]
pub enum CompilerError {
    #[error("Invalid GGUF format: {message}")]
    InvalidFormat { message: String },
    #[error("I/O error: {message}")]
    IOError { message: String },
    #[error("Quantization failed: {message}")]
    QuantizationFailed { message: String },
}

/// Callback for deterministic compiler progress.
#[uniffi::export(callback_interface)]
pub trait CompilerProgressCallback: Send + Sync {
    fn on_log(&self, message: String);
    fn on_progress(&self, percentage: f32);
}

/// Compile a GGUF model file into a .cimage output directory.
///
/// The output directory will contain the .cimage binary (loaded by
/// `BridgeMultiplexer.load`), manifest.json, segment files, and optional
/// ANE model archives.  Returns the path to the compiled output directory.
#[cfg(target_os = "macos")]
#[uniffi::export]
pub fn prism_compile_gguf(
    gguf_path: String,
    output_dir: String,
    callback: Option<Box<dyn CompilerProgressCallback>>,
) -> Result<String, CompilerError> {
    if let Some(cb) = &callback {
        cb.on_log(format!("Starting AOT compilation for {}", gguf_path));
        cb.on_progress(0.0);
    }

    let gguf = std::path::Path::new(&gguf_path);
    let out = std::path::Path::new(&output_dir);

    if !gguf.exists() {
        return Err(CompilerError::InvalidFormat {
            message: format!("GGUF file not found: {}", gguf_path),
        });
    }

    // Ensure output directory exists
    std::fs::create_dir_all(out).map_err(|e| CompilerError::IOError {
        message: format!("create output dir: {e}"),
    })?;

    // Run the full pipeline
    let result = tribunus_compute_core::compute_image::compile::compile_gguf_unchecked(
        &gguf_path,
        &output_dir,
        None, // quantize_mode — auto-detect from target
        None, // ane_models_dir — optional pre-compiled ANE models
        None, // metallib_path — optional pre-compiled Metal kernels
        None, // mlx_capture_dir — optional MLX capture output
    );

    match result {
        Ok(_compiled) => {
            if let Some(cb) = &callback {
                cb.on_log("Packing ternary page-aligned weights...".to_string());
                cb.on_progress(90.0);
                cb.on_log("Compilation complete.".to_string());
                cb.on_progress(100.0);
            }
            Ok(output_dir)
        }
        Err(e) => {
            if let Some(cb) = &callback {
                cb.on_log(format!("Compilation failed: {e}"));
            }
            Err(CompilerError::QuantizationFailed {
                message: e.to_string(),
            })
        }
    }
}

/// Compile nf4-tile-640 weights into a .cimage using the CLI binary.
///
/// For now this defers to the CLI; the direct bridge path will be wired
/// once the nf4 cimage path is stabilised.
#[uniffi::export]
pub fn prism_compile_nf4(
    safetensors_dir: String,
    output_cimage_path: String,
    tts_repo: Option<String>,
    callback: Option<Box<dyn CompilerProgressCallback>>,
) -> Result<String, CompilerError> {
    let _ = safetensors_dir;
    let _ = output_cimage_path;
    let _ = tts_repo;
    if let Some(cb) = &callback {
        cb.on_log("nf4tile640 compilation not yet wired through bridge".into());
    }
    Err(CompilerError::InvalidFormat { message: "use CLI: gemma4_ingest --nf4".into() })
}

// ═══════════════════════════════════════════════════════════════════════
// Streaming inference
// ═══════════════════════════════════════════════════════════════════════

/// A loaded .cimage model ready for inference.  Passes by reference
/// (Arc) across the FFI boundary — the multiplexer holds the Metal
/// buffers, ECS world, and ANE state.
#[derive(uniffi::Object)]
pub struct BridgeMultiplexer {
    #[allow(dead_code)]
    #[cfg(target_os = "macos")]
    pub(crate) executor: parking_lot::Mutex<Option<HeterogeneousExecutor>>,
    pub(crate) tts: Option<TtsPipeline>,
    pub(crate) tokenizer: Option<tribunus_compute_core::tokenizer::TribunusTokenizer>,
    #[allow(dead_code)]
    pub(crate) cimage_path: Option<std::path::PathBuf>,
}

#[uniffi::export]
impl BridgeMultiplexer {
    /// Load a compiled .cimage and initialise the runtime multiplexer.
    #[uniffi::constructor]
    pub fn load(cimage_path: String, model_dir: String) -> Result<Arc<Self>, BridgeError> {
        let cpath = std::path::Path::new(&cimage_path);
        #[cfg(target_os = "macos")]
        let executor = create_inference_executor(cpath, 1, false)
            .map_err(|e| BridgeError::CimageLoadFailed(e.to_string()))?;
        #[cfg(not(target_os = "macos"))]
        let _executor = ();

        // Try loading TTS if available
        let device = metal::Device::system_default()
            .ok_or_else(|| BridgeError::CimageLoadFailed("no Metal device".into()))?;
        let deployment = CimageDeployment::load(cpath, &device)
            .map_err(|e| BridgeError::CimageLoadFailed(e.to_string()))?;
        let tts = TtsPipeline::from_cimage(&deployment, &device).ok();

        // Load tokenizer
        let tokenizer = tribunus_compute_core::tokenizer::TribunusTokenizer::from_dir(
            std::path::Path::new(&model_dir),
        ).ok();

        Ok(Arc::new(Self {
            #[cfg(target_os = "macos")]
            executor: parking_lot::Mutex::new(Some(executor)),
            tts,
            tokenizer,
            cimage_path: Some(cpath.to_path_buf()),
        }))
    }
}

/// A single event in the streaming inference output stream.
///
/// Text tokens arrive as `Text`.  Raw media (uncompressed pixels, PCM audio,
/// embeddings) arrives as the other variants — Vec<u8> maps to Swift's
/// Foundation.Data and Vec<f32> maps to [Float] with zero manual conversion.
///
/// Each `VideoFrame` carries a monotonic `timestamp_ns` so the Swift export
/// pipeline (AVAssetWriter → ProRes .mov) can reconstruct a deterministic
/// timeline that NLEs like DaVinci Resolve or Final Cut Pro accept without
/// resampling or drift correction.
#[derive(uniffi::Enum)]
pub enum StreamEvent {
    Text {
        token: String,
    },
    /// Raw uncompressed BGRA 8-bit pixel array for direct CVPixelBuffer wrapping.
    ImageFrame {
        pixel_bytes: Vec<u8>,
        width: u32,
        height: u32,
    },
    /// Raw uncompressed BGRA 8-bit pixel array with high-precision timestamp
    /// for ProRes video track assembly.
    VideoFrame {
        pixel_bytes: Vec<u8>,
        width: u32,
        height: u32,
        timestamp_ns: u64,
    },
    /// Raw Linear PCM audio bytes (32-bit float or 16-bit int) for Core Audio
    /// CMSampleBuffer ingestion.  `sample_rate` in Hz, `channels` = 1 (mono)
    /// or 2 (stereo).
    AudioChunk {
        pcm_bytes: Vec<u8>,
        sample_rate: u32,
        channels: u32,
    },
    /// Dense vector embedding from the model.  Used for similarity search,
    /// clustering, or conditioning downstream generators.
    Embedding {
        values: Vec<f32>,
    },
}

/// Callback interface for multimodal streaming inference.
#[uniffi::export(callback_interface)]
pub trait MultimodalStreamCallback: Send + Sync {
    /// Called for each event in the output stream (text token, image, audio, etc.).
    fn on_event(&self, event: StreamEvent);
    /// Generation completed successfully.
    fn on_done(&self);
    /// Generation failed.
    fn on_error(&self, error: String);
}

/// Callback interface that lets the V8 sandbox drive the WKWebView.
/// Implemented in Swift — each method blocks the V8 thread until the
/// WebKit operation completes on the Main Actor.
#[cfg(feature = "deno_core")]
#[uniffi::export(callback_interface)]
pub trait BrowserRuntimeDriver: Send + Sync {
    /// Navigate to a URL. Returns "ok" or an error message starting with "ERROR:".
    fn navigate(&self, url: String) -> String;
    /// Return the semantic DOM snapshot as JSON, or an error starting with "ERROR:".
    fn snapshot(&self) -> String;
    /// Interact with an element. Returns "ok" or an error starting with "ERROR:".
    fn interact(&self, id: u32, action: String, value: Option<String>) -> String;
    /// Evaluate JS in the page. Returns the result, or "ERROR: ...".
    fn evaluate_js(&self, script: String) -> String;
    /// Download a URL using the browser's authenticated session.
    fn download(&self, url: String, filename: String) -> String;
}

#[cfg(feature = "deno_core")]
/// Run JavaScript in the V8 sandbox with a browser driver for web ops.
/// The driver is called synchronously from V8 ops — it must block until
/// the WKWebView operation completes.
#[uniffi::export]
pub fn prism_run_js(
    code: String,
    sandbox_root: String,
    driver: Option<Box<dyn BrowserRuntimeDriver>>,
) -> String {
    use std::sync::Arc;
    use tribunus_compute_core::tools::js_runtime::{self, WebDriver};

    if let Some(d) = driver {
        struct DriverWrapper {
            inner: Box<dyn BrowserRuntimeDriver>,
        }
        impl WebDriver for DriverWrapper {
            fn navigate(&self, url: &str) -> Result<String, String> {
                let r = self.inner.navigate(url.to_string());
                if r.starts_with("ERROR:") {
                    Err(r)
                } else {
                    Ok(r)
                }
            }
            fn snapshot(&self) -> Result<String, String> {
                let r = self.inner.snapshot();
                if r.starts_with("ERROR:") {
                    Err(r)
                } else {
                    Ok(r)
                }
            }
            fn interact(
                &self,
                id: u32,
                action: &str,
                value: Option<&str>,
            ) -> Result<String, String> {
                let r = self
                    .inner
                    .interact(id, action.to_string(), value.map(|s| s.to_string()));
                if r.starts_with("ERROR:") {
                    Err(r)
                } else {
                    Ok(r)
                }
            }
            fn evaluate_js(&self, script: &str) -> Result<String, String> {
                let r = self.inner.evaluate_js(script.to_string());
                if r.starts_with("ERROR:") {
                    Err(r)
                } else {
                    Ok(r)
                }
            }
            fn download(&self, url: &str, filename: &str) -> Result<String, String> {
                let r = self.inner.download(url.to_string(), filename.to_string());
                if r.starts_with("ERROR:") {
                    Err(r)
                } else {
                    Ok(r)
                }
            }
        }
        js_runtime::set_web_driver(Arc::new(DriverWrapper { inner: d }));
    }

    let root = if sandbox_root.is_empty() {
        None
    } else {
        Some(std::path::Path::new(&sandbox_root))
    };
    let result = js_runtime::run_javascript(&code, root, None);
    serde_json::to_string(&result).unwrap_or_default()
}

/// Fetch and X-Ray sanitize a URL through the Rust proxy.
/// Returns the sanitized HTML string with scripts neutered and CSP injected.
#[uniffi::export]
pub async fn prism_xray_navigate(url: String) -> Result<String, BridgeError> {
    match tribunus_compute_core::tools::xray::fetch_and_xray_url(&url).await {
        Ok(html) => Ok(html),
        Err(e) => Err(BridgeError::CimageLoadFailed(format!("X-Ray failure: {e}"))),
    }
}

/// Run inference with streaming output, using the LUT engine path.
///
/// `cimage_path` — compiled .cimage from `prism_compile_gguf`.
/// `model_dir` — directory containing tokenizer.json and config.json.
#[cfg(target_os = "macos")]
#[uniffi::export]
pub fn prism_infer_multimodal_stream(
    cimage_path: String,
    model_dir: String,
    prompt: String,
    callback: Box<dyn MultimodalStreamCallback>,
) {
    let callback = std::sync::Arc::new(callback);

    // Load multiplexer
    let multiplexer = match BridgeMultiplexer::load(cimage_path, model_dir) {
        Ok(m) => m,
        Err(e) => {
            callback.on_error(format!("load: {e}"));
            return;
        }
    };

    std::thread::spawn(move || {
        let mut exec_guard = multiplexer.executor.lock();
        let exec = match &mut *exec_guard {
            Some(e) => e,
            None => {
                callback.on_error("executor not initialized".into());
                return;
            }
        };

        // Tokenize
        let tokenizer = match &multiplexer.tokenizer {
            Some(t) => t,
            None => {
                callback.on_error("no tokenizer".into());
                return;
            }
        };
        let input_ids = match tokenizer.encode(&prompt) {
            Ok(t) => t,
            Err(e) => {
                callback.on_error(format!("tokenize: {e}"));
                return;
            }
        };

        // Build operation descriptor
        let decode_op = OperationDescriptor {
            operation_id: OperationId(0),
            family: OperationFamily::DecoderLayer,
            layer_index: None,
            phase: Phase::Decode,
            logical_shape: LogicalShape { dims: vec![1] },
            physical_layout: PhysicalLayout::RowMajor,
            input_dtypes: vec![],
            output_dtype: DType::F32,
            quantization: None,
            expected_output_shape: TensorShape { dims: vec![] },
            correctness_checkpoint: CorrectnessCheckpointPolicy::None,
        };

        // Prefill
        for (i, &tok) in input_ids[..input_ids.len().saturating_sub(1)].iter().enumerate() {
            let mut op = decode_op.clone();
            op.operation_id = OperationId(tok as u64);
            exec.operation_registry.insert(op.operation_id, op);
            let plan = ExecutionBoundaryPlan {
                group_id: EvaluationGroupId(i as u64),
                backend_id: BACKEND_MEGAKERNEL,
                operations: vec![OperationId(tok as u64)],
                materialized_outputs: vec![],
                policy: EvaluationPolicy::BackendLazy,
                synchronization: SynchronizationPolicy::None,
                release_after: vec![],
                content_digest: None,
            };
            let _ = exec.execute_boundaries(&[plan]);
        }

        // Decode
        let max_tokens = 512;
        let mut last_token = *input_ids.last().unwrap_or(&0) as u64;
        for step in 0..max_tokens {
            let mut op = decode_op.clone();
            op.operation_id = OperationId(last_token);
            exec.operation_registry.insert(op.operation_id, op);
            let plan = ExecutionBoundaryPlan {
                group_id: EvaluationGroupId((input_ids.len() + step) as u64),
                backend_id: BACKEND_MEGAKERNEL,
                operations: vec![OperationId(last_token)],
                materialized_outputs: vec![],
                policy: EvaluationPolicy::BackendLazy,
                synchronization: SynchronizationPolicy::None,
                release_after: vec![],
                content_digest: None,
            };
            let _ = exec.execute_boundaries(&[plan]);

            last_token = match exec.last_decoded_token() {
                Ok(t) => t,
                Err(_) => break,
            };

            if let Ok(text) = tokenizer.decode(&[last_token as u32]) {
                callback.on_event(StreamEvent::Text { token: text });
            }
        }

        callback.on_done();
    });
}

/// Generate streaming audio (TTS) from text.
#[uniffi::export]
pub fn prism_generate_audio(
    cimage_path: String,
    model_dir: String,
    text: String,
    callback: Box<dyn MultimodalStreamCallback>,
) {
    let callback = std::sync::Arc::new(callback);

    let multiplexer = match BridgeMultiplexer::load(cimage_path, model_dir) {
        Ok(m) => m,
        Err(e) => {
            callback.on_error(format!("load: {e}"));
            return;
        }
    };

    std::thread::spawn(move || {
        // TTS pipeline reference lives inside the Arc<BridgeMultiplexer>
        let tts = match &multiplexer.tts {
            Some(t) => t,
            None => {
                callback.on_error("TTS not available in cimage".into());
                return;
            }
        };

        // Tokenize text for TTS
        let tokenizer = match &multiplexer.tokenizer {
            Some(t) => t,
            None => {
                callback.on_error("no tokenizer".into());
                return;
            }
        };
        let tokens = match tokenizer.encode(&text) {
            Ok(t) => t,
            Err(e) => {
                callback.on_error(format!("tokenize: {e}"));
                return;
            }
        };

        // Generate streaming audio
        match tts.generate_streaming(&tokens, 256, 20) {
            Ok(chunks) => {
                for chunk in chunks {
                    // Convert f32 PCM to 16-bit PCM bytes
                    let pcm_bytes: Vec<u8> = chunk.iter()
                        .flat_map(|&s| {
                            let clamped = (s.max(-1.0).min(1.0) * 32767.0) as i16;
                            clamped.to_le_bytes().to_vec()
                        })
                        .collect();

                    callback.on_event(StreamEvent::AudioChunk {
                        pcm_bytes: pcm_bytes.into(),
                        sample_rate: 24000,
                        channels: 1,
                    });
                }
                callback.on_done();
            }
            Err(e) => callback.on_error(format!("TTS: {e}")),
        }
    });
}

// ═══════════════════════════════════════════════════════════════════════
// Device enumeration (DeviceRegistry)
// ═══════════════════════════════════════════════════════════════════════

/// Bridge version of DeviceKind for UniFFI export.
#[derive(uniffi::Enum, Clone, Debug)]
pub enum BridgeDeviceKind {
    Cpu,
    GpuDiscrete,
    GpuIntegrated,
    GpuUnified,
    Npu,
    Accelerator,
}

impl From<DeviceKind> for BridgeDeviceKind {
    fn from(k: DeviceKind) -> Self {
        match k {
            DeviceKind::Cpu => Self::Cpu,
            DeviceKind::GpuDiscrete => Self::GpuDiscrete,
            DeviceKind::GpuIntegrated => Self::GpuIntegrated,
            DeviceKind::GpuUnified => Self::GpuUnified,
            DeviceKind::Npu => Self::Npu,
            DeviceKind::Accelerator => Self::Accelerator,
        }
    }
}

/// Bridge version of BackendKind for UniFFI export.
#[derive(uniffi::Enum, Clone, Debug)]
pub enum BridgeBackendKind {
    Metal,
    Cuda,
    Rocm,
    LevelZero,
    CoreMl,
    Ane,
    Accelerate,
    CandleCpu,
    Cpu,
    Tensix,
}

impl From<BackendKind> for BridgeBackendKind {
    fn from(k: BackendKind) -> Self {
        match k {
            BackendKind::Metal => Self::Metal,
            BackendKind::Cuda => Self::Cuda,
            BackendKind::Rocm => Self::Rocm,
            BackendKind::LevelZero => Self::LevelZero,
            BackendKind::CoreAi => Self::CoreMl,
            BackendKind::Ane => Self::Ane,
            BackendKind::Accelerate => Self::Accelerate,
            BackendKind::CandleCpu => Self::CandleCpu,
            BackendKind::Cpu => Self::Cpu,
            BackendKind::Tensix => Self::Tensix,
        }
    }
}

/// Bridge version of DeviceInfo for UniFFI export.
#[derive(uniffi::Record, Clone, Debug)]
pub struct BridgeDeviceMemoryInfo {
    pub total_bytes: u64,
    pub free_bytes: u64,
    pub bandwidth_gb_per_sec: f64,
    pub unified_with_cpu: bool,
}

impl From<DeviceMemoryInfo> for BridgeDeviceMemoryInfo {
    fn from(m: DeviceMemoryInfo) -> Self {
        Self {
            total_bytes: m.total_bytes,
            free_bytes: m.free_bytes,
            bandwidth_gb_per_sec: m.bandwidth_gb_per_sec,
            unified_with_cpu: m.unified_with_cpu,
        }
    }
}

/// Bridge version of PcieLinkInfo for UniFFI export.
#[derive(uniffi::Record, Clone, Debug)]
pub struct BridgePcieLinkInfo {
    pub generation: u32,
    pub lanes: u32,
    pub max_speed_gb_per_sec: f64,
}

impl From<PcieLinkInfo> for BridgePcieLinkInfo {
    fn from(p: PcieLinkInfo) -> Self {
        Self {
            generation: p.generation,
            lanes: p.lanes,
            max_speed_gb_per_sec: p.max_speed_gb_per_sec,
        }
    }
}

/// Device information for host-app consumption across the FFI bridge.
#[derive(uniffi::Record, Clone, Debug)]
pub struct BridgeDeviceInfo {
    /// Numeric device id.
    pub id: u32,
    /// Broad device category.
    pub kind: BridgeDeviceKind,
    /// Specific compute backend.
    pub backend: BridgeBackendKind,
    /// Human-readable name.
    pub name: String,
    /// Vendor string.
    pub vendor: String,
    /// Driver version.
    pub driver_version: String,
    /// Memory properties.
    pub memory: BridgeDeviceMemoryInfo,
    /// Compute units / cores.
    pub compute_units: u32,
    /// Core clock in MHz.
    pub clock_mhz: u32,
    /// NPU/ANE core count.
    pub ane_cores: u32,
    /// Supported data formats.
    pub supports_f16: bool,
    pub supports_bf16: bool,
    pub supports_int8: bool,
    pub supports_ternary: bool,
    /// PCIe link for discrete devices.
    pub pcie_link: Option<BridgePcieLinkInfo>,
}

/// List all compute devices available on this host.
#[uniffi::export]
pub fn prism_list_devices() -> Vec<BridgeDeviceInfo> {
    device::global_registry()
        .enumerate()
        .iter()
        .map(|d| BridgeDeviceInfo {
            id: d.id.0,
            kind: d.kind.into(),
            backend: d.backend.into(),
            name: d.name.clone(),
            vendor: d.vendor.clone(),
            driver_version: d.driver_version.clone(),
            memory: BridgeDeviceMemoryInfo::from(d.memory.clone()),
            compute_units: d.compute_units,
            clock_mhz: d.clock_mhz,
            ane_cores: d.ane_cores,
            supports_f16: d.supports_f16,
            supports_bf16: d.supports_bf16,
            supports_int8: d.supports_int8,
            supports_ternary: d.supports_ternary,
            pcie_link: d
                .pcie_link
                .as_ref()
                .map(|p| BridgePcieLinkInfo::from(p.clone())),
        })
        .collect()
}

/// Return the number of device slots (for Swift-side table sizing).
#[uniffi::export]
pub fn prism_device_count() -> u32 {
    device::global_registry().count() as u32
}

/// Return a JSON string of all devices (for agent serialization).
#[uniffi::export]
pub fn prism_device_json() -> String {
    device::global_registry().to_json()
}

// ═══════════════════════════════════════════════════════════════════════
// Configuration (ServerConfig + compile-time settings)
// ═══════════════════════════════════════════════════════════════════════

/// Bridge version of HardwareTarget for UniFFI export.
#[derive(uniffi::Enum, Clone, Debug)]
pub enum BridgeHardwareTarget {
    M1,
    M1Pro,
    M2,
    M2Ultra,
    M3Ultra,
}

impl From<HardwareTarget> for BridgeHardwareTarget {
    fn from(h: HardwareTarget) -> Self {
        match h {
            HardwareTarget::M1 => Self::M1,
            HardwareTarget::M1Pro => Self::M1Pro,
            HardwareTarget::M2 => Self::M2,
            HardwareTarget::M2Ultra => Self::M2Ultra,
            HardwareTarget::M3Ultra => Self::M3Ultra,
        }
    }
}

impl From<BridgeHardwareTarget> for HardwareTarget {
    fn from(b: BridgeHardwareTarget) -> Self {
        match b {
            BridgeHardwareTarget::M1 => Self::M1,
            BridgeHardwareTarget::M1Pro => Self::M1Pro,
            BridgeHardwareTarget::M2 => Self::M2,
            BridgeHardwareTarget::M2Ultra => Self::M2Ultra,
            BridgeHardwareTarget::M3Ultra => Self::M3Ultra,
        }
    }
}

/// Bridge version of CompileQuantMode for UniFFI export.
#[derive(uniffi::Enum, Clone, Debug)]
pub enum BridgeCompileQuantMode {
    Nf4 { group_size: u32 },
    Af8 { group_size: u32 },
    Ternary { group_size: u32 },
    TernaryTile640 { group_size: u32 },
    AutoDetect,
}

impl From<CompileQuantMode> for BridgeCompileQuantMode {
    fn from(q: CompileQuantMode) -> Self {
        match q {
            CompileQuantMode::Nf4 { group_size } => Self::Nf4 { group_size },
            CompileQuantMode::Af8 { group_size } => Self::Af8 { group_size },
            CompileQuantMode::Ternary { group_size } => Self::Ternary { group_size },
            CompileQuantMode::TernaryTile640 { group_size } => Self::TernaryTile640 { group_size },
            CompileQuantMode::Nf4Tile640 { group_size } => Self::Nf4 { group_size },
        }
    }
}

impl From<BridgeCompileQuantMode> for CompileQuantMode {
    fn from(b: BridgeCompileQuantMode) -> Self {
        match b {
            BridgeCompileQuantMode::Nf4 { group_size } => Self::Nf4 { group_size },
            BridgeCompileQuantMode::Af8 { group_size } => Self::Af8 { group_size },
            BridgeCompileQuantMode::Ternary { group_size } => Self::Ternary { group_size },
            BridgeCompileQuantMode::TernaryTile640 { group_size } => {
                Self::TernaryTile640 { group_size }
            }
            // AutoDetect has no corresponding core variant; choose reasonable default
            BridgeCompileQuantMode::AutoDetect => Self::Nf4 { group_size: 64 },
        }
    }
}

/// Bridge version of GenerationRegime for UniFFI export.
#[derive(uniffi::Enum, Clone, Debug)]
pub enum BridgeGenerationRegime {
    Autoregressive,
    DiscreteDiffusion,
}

impl From<GenerationRegime> for BridgeGenerationRegime {
    fn from(g: GenerationRegime) -> Self {
        match g {
            GenerationRegime::Autoregressive => Self::Autoregressive,
            GenerationRegime::DiscreteDiffusion => Self::DiscreteDiffusion,
        }
    }
}

/// Bridge version of KvCacheMode for UniFFI export.
#[derive(uniffi::Enum, Clone, Debug)]
pub enum BridgeKvCacheMode {
    AppendOnly,
    FullRecompute,
    BlockCache,
}

impl From<KvCacheMode> for BridgeKvCacheMode {
    fn from(k: KvCacheMode) -> Self {
        match k {
            KvCacheMode::AppendOnly => Self::AppendOnly,
            KvCacheMode::FullRecompute => Self::FullRecompute,
            KvCacheMode::BlockCache => Self::BlockCache,
        }
    }
}

// ── ServerConfig sections ───────────────────────────────────────────

#[derive(uniffi::Record, Clone, Debug)]
pub struct BridgeServerConfigSection {
    pub port: u16,
    pub host: String,
    pub max_concurrent: u32,
    pub rate_limit_per_min: u32,
    pub rate_limit_tokens_per_sec: f64,
    pub rate_limit_burst: u64,
    pub log_level: String,
    pub runtime_mode: String,
}

impl From<config::network::ServerConfigSection> for BridgeServerConfigSection {
    fn from(s: config::network::ServerConfigSection) -> Self {
        Self {
            port: s.port,
            host: s.host,
            max_concurrent: s.max_concurrent,
            rate_limit_per_min: s.rate_limit_per_min,
            rate_limit_tokens_per_sec: s.rate_limit_tokens_per_sec,
            rate_limit_burst: s.rate_limit_burst,
            log_level: s.log_level,
            runtime_mode: s.runtime_mode,
        }
    }
}

impl From<BridgeServerConfigSection> for config::network::ServerConfigSection {
    fn from(b: BridgeServerConfigSection) -> Self {
        Self {
            port: b.port,
            host: b.host,
            max_concurrent: b.max_concurrent,
            rate_limit_per_min: b.rate_limit_per_min,
            rate_limit_tokens_per_sec: b.rate_limit_tokens_per_sec,
            rate_limit_burst: b.rate_limit_burst,
            log_level: b.log_level,
            runtime_mode: b.runtime_mode,
        }
    }
}

#[derive(uniffi::Record, Clone, Debug)]
pub struct BridgeModelConfigSection {
    pub model_path: Option<String>,
    pub auto_download: bool,
    pub max_model_cache_gb: f64,
}

impl From<config::network::ModelConfigSection> for BridgeModelConfigSection {
    fn from(m: config::network::ModelConfigSection) -> Self {
        Self {
            model_path: m.model_path,
            auto_download: m.auto_download,
            max_model_cache_gb: m.max_model_cache_gb,
        }
    }
}

impl From<BridgeModelConfigSection> for config::network::ModelConfigSection {
    fn from(b: BridgeModelConfigSection) -> Self {
        Self {
            model_path: b.model_path,
            auto_download: b.auto_download,
            max_model_cache_gb: b.max_model_cache_gb,
        }
    }
}

#[derive(uniffi::Record, Clone, Debug)]
pub struct BridgeCacheConfigSection {
    pub kv_cache_tiers: u32,
    pub compression_ratio: f64,
    pub evolkv_enabled: bool,
}

impl From<config::network::CacheConfigSection> for BridgeCacheConfigSection {
    fn from(c: config::network::CacheConfigSection) -> Self {
        Self {
            kv_cache_tiers: c.kv_cache_tiers,
            compression_ratio: c.compression_ratio,
            evolkv_enabled: c.evolkv_enabled,
        }
    }
}

impl From<BridgeCacheConfigSection> for config::network::CacheConfigSection {
    fn from(b: BridgeCacheConfigSection) -> Self {
        Self {
            kv_cache_tiers: b.kv_cache_tiers,
            compression_ratio: b.compression_ratio,
            evolkv_enabled: b.evolkv_enabled,
        }
    }
}

#[derive(uniffi::Record, Clone, Debug)]
pub struct BridgeSpecConfigSection {
    pub draft_count: u32,
    pub draft_length: u32,
    pub spechub_enabled: bool,
}

impl From<config::network::SpecConfigSection> for BridgeSpecConfigSection {
    fn from(s: config::network::SpecConfigSection) -> Self {
        Self {
            draft_count: s.draft_count,
            draft_length: s.draft_length,
            spechub_enabled: s.spechub_enabled,
        }
    }
}

impl From<BridgeSpecConfigSection> for config::network::SpecConfigSection {
    fn from(b: BridgeSpecConfigSection) -> Self {
        Self {
            draft_count: b.draft_count,
            draft_length: b.draft_length,
            spechub_enabled: b.spechub_enabled,
        }
    }
}

#[derive(uniffi::Record, Clone, Debug)]
pub struct BridgeClusterConfigSection {
    pub exo_enabled: bool,
    pub exo_port: u16,
    pub autoscale_min: u32,
    pub autoscale_max: u32,
}

impl From<config::network::ClusterConfigSection> for BridgeClusterConfigSection {
    fn from(c: config::network::ClusterConfigSection) -> Self {
        Self {
            exo_enabled: c.exo_enabled,
            exo_port: c.exo_port,
            autoscale_min: c.autoscale_min,
            autoscale_max: c.autoscale_max,
        }
    }
}

impl From<BridgeClusterConfigSection> for config::network::ClusterConfigSection {
    fn from(b: BridgeClusterConfigSection) -> Self {
        Self {
            exo_enabled: b.exo_enabled,
            exo_port: b.exo_port,
            autoscale_min: b.autoscale_min,
            autoscale_max: b.autoscale_max,
        }
    }
}

/// Full server configuration for the Swift host app.
#[derive(uniffi::Record, Clone, Debug)]
pub struct BridgeServerConfig {
    pub server: BridgeServerConfigSection,
    pub model: BridgeModelConfigSection,
    pub cache: BridgeCacheConfigSection,
    pub speculation: BridgeSpecConfigSection,
    pub cluster: BridgeClusterConfigSection,
}

impl From<ServerConfig> for BridgeServerConfig {
    fn from(c: ServerConfig) -> Self {
        Self {
            server: c.server.into(),
            model: c.model.into(),
            cache: c.cache.into(),
            speculation: c.speculation.into(),
            cluster: c.cluster.into(),
        }
    }
}

impl From<BridgeServerConfig> for ServerConfig {
    fn from(b: BridgeServerConfig) -> Self {
        ServerConfig {
            server: b.server.into(),
            model: b.model.into(),
            cache: b.cache.into(),
            speculation: b.speculation.into(),
            cluster: b.cluster.into(),
        }
    }
}

// ── OperationRoute ──────────────────────────────────────────────────

/// Per-operation backend routing for decoder layers.
#[derive(uniffi::Record, Clone, Debug)]
pub struct BridgeOperationRoute {
    pub rms_norm: u32,
    pub silu: u32,
    pub matmul: u32,
    pub attention: u32,
    pub softmax: u32,
    pub rope: u32,
    pub add: u32,
    pub multiply: u32,
    pub transpose: u32,
    pub reshape: u32,
}

impl From<OperationRoute> for BridgeOperationRoute {
    fn from(o: OperationRoute) -> Self {
        Self {
            rms_norm: o.rms_norm,
            silu: o.silu,
            matmul: o.matmul,
            attention: o.attention,
            softmax: o.softmax,
            rope: o.rope,
            add: o.add,
            multiply: o.multiply,
            transpose: o.transpose,
            reshape: o.reshape,
        }
    }
}

// ── Exported functions ──────────────────────────────────────────────

/// Load the server configuration from the default config file path
/// ($HOME/.tribunus/config.toml), environment variables, and defaults.
#[uniffi::export]
pub fn prism_load_config() -> BridgeServerConfig {
    ServerConfig::load().into()
}

/// Load the server configuration from a specific TOML config file path.
/// Falls back to defaults if the file cannot be read.
#[uniffi::export]
pub fn prism_load_config_from(path: String) -> BridgeServerConfig {
    let mut config = ServerConfig::default();
    if let Ok(file_config) = ServerConfig::load_config_toml(&path) {
        config.server = file_config.server;
        config.model = file_config.model;
        config.cache = file_config.cache;
        config.speculation = file_config.speculation;
        config.cluster = file_config.cluster;
    }
    config.load_env_overrides();
    config.into()
}

/// Return the current server configuration as a JSON string.
#[uniffi::export]
pub fn prism_config_json() -> String {
    serde_json::to_string_pretty(&ServerConfig::load()).unwrap_or_else(|_| "{}".to_string())
}

// ═══════════════════════════════════════════════════════════════════════
// Internal conversions
// ═══════════════════════════════════════════════════════════════════════

fn bridge_phase_from(phase: &agent::Phase) -> BridgePhase {
    match phase {
        agent::Phase::Idle => BridgePhase::Idle,
        agent::Phase::Generating => BridgePhase::Generating,
        agent::Phase::AwaitingTools { .. } => BridgePhase::AwaitingTools,
        agent::Phase::AwaitingSubagents => BridgePhase::AwaitingSubagents,
        agent::Phase::Done { .. } => BridgePhase::Done,
    }
}

fn bridge_outcome_from(
    outcome: &agent::StepOutcome,
    _state: &agent::AgentState,
) -> BridgeStepOutcome {
    match outcome {
        agent::StepOutcome::TextChunk(_) => BridgeStepOutcome::Generating,
        agent::StepOutcome::ToolCalls(calls) => BridgeStepOutcome::AwaitingTools {
            tools: calls
                .iter()
                .map(|c| BridgeToolCall {
                    id: c.id.clone(),
                    name: c.name.clone(),
                    arguments_json: serde_json::to_string(&c.arguments).unwrap_or_default(),
                })
                .collect(),
        },
        agent::StepOutcome::SubagentSpawned(handle) => BridgeStepOutcome::AwaitingSubagents {
            subagents: vec![BridgeSubagentHandle {
                id: handle.id,
                goal: handle.goal.clone(),
                sandbox_subpath: handle.sandbox_subpath.clone(),
                tool_allowlist: handle.tool_allowlist.clone(),
                max_revisions: 3,
            }],
        },
        agent::StepOutcome::SubagentResult { .. } => {
            // This is an internal transition — the app feeds subagent
            // results via an explicit call, not through step().
            BridgeStepOutcome::Generating
        }
        agent::StepOutcome::Finished { output } => BridgeStepOutcome::Finished {
            result: output.clone(),
        },
        agent::StepOutcome::Idle => BridgeStepOutcome::Finished {
            result: String::new(),
        },
    }
}

fn error_result(msg: &str) -> BridgeStepResult {
    BridgeStepResult {
        state: BridgeAgentState {
            phase: BridgePhase::Done,
            history_jsonl: String::new(),
            current_prompt: String::new(),
        },
        outcome: BridgeStepOutcome::Finished {
            result: format!("ERROR: {msg}"),
        },
    }
}
