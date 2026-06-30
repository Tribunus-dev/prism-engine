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

use tribunus_compute_core::agent;
use tribunus_compute_core::tools;
use tribunus_compute_core::runtime::agent_slot::MultiplexerState;
use tribunus_compute_core::compute_image::cimage_loader::load_cimage_mmap;
use std::sync::Arc;
use std::path::Path;


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
    AwaitingTools { tools: Vec<BridgeToolCall> },
    AwaitingSubagents { subagents: Vec<BridgeSubagentHandle> },
    Finished { result: String },
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
pub fn prism_agent_step(
    state_json: String,
    model_output: String,
) -> BridgeStepResult {
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
        None,   // quantize_mode — auto-detect from target
        None,   // ane_models_dir — optional pre-compiled ANE models
        None,   // metallib_path — optional pre-compiled Metal kernels
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

// ═══════════════════════════════════════════════════════════════════════
// Streaming inference
// ═══════════════════════════════════════════════════════════════════════

/// A loaded .cimage model ready for inference.  Passes by reference
/// (Arc) across the FFI boundary — the multiplexer holds the Metal
/// buffers, ECS world, and ANE state.
#[derive(uniffi::Object)]
pub struct BridgeMultiplexer {
    pub(crate) inner: Arc<MultiplexerState>,
}

#[uniffi::export]
impl BridgeMultiplexer {
    /// Load a compiled .cimage and initialise the runtime multiplexer.
    #[uniffi::constructor]
    pub fn load(cimage_path: String, _model_dir: String) -> Result<Arc<Self>, BridgeError> {
        let path = Path::new(&cimage_path);
        let (mmap, header) =
            load_cimage_mmap(path).map_err(|e| BridgeError::CimageLoadFailed(format!("load cimage: {e}")))?;
        let mmap_arc = Arc::new(mmap);
        let mut state = MultiplexerState::new();
        state.init_from_cimage(mmap_arc, &header, 3840, 18432);
        Ok(Arc::new(BridgeMultiplexer {
            inner: Arc::new(state),
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

/// Run JavaScript in the V8 sandbox with a browser driver for web ops.
/// The driver is called synchronously from V8 ops — it must block until
/// the WKWebView operation completes.
#[uniffi::export]
pub fn prism_run_js(
    code: String,
    sandbox_root: String,
    driver: Option<Box<dyn BrowserRuntimeDriver>>,
) -> String {
    use tribunus_compute_core::tools::js_runtime::{self, WebDriver};
    use std::sync::Arc;

    if let Some(d) = driver {
        struct DriverWrapper {
            inner: Box<dyn BrowserRuntimeDriver>,
        }
        impl WebDriver for DriverWrapper {
            fn navigate(&self, url: &str) -> Result<String, String> {
                let r = self.inner.navigate(url.to_string());
                if r.starts_with("ERROR:") { Err(r) } else { Ok(r) }
            }
            fn snapshot(&self) -> Result<String, String> {
                let r = self.inner.snapshot();
                if r.starts_with("ERROR:") { Err(r) } else { Ok(r) }
            }
            fn interact(&self, id: u32, action: &str, value: Option<&str>) -> Result<String, String> {
                let r = self.inner.interact(id, action.to_string(), value.map(|s| s.to_string()));
                if r.starts_with("ERROR:") { Err(r) } else { Ok(r) }
            }
            fn evaluate_js(&self, script: &str) -> Result<String, String> {
                let r = self.inner.evaluate_js(script.to_string());
                if r.starts_with("ERROR:") { Err(r) } else { Ok(r) }
            }
            fn download(&self, url: &str, filename: &str) -> Result<String, String> {
                let r = self.inner.download(url.to_string(), filename.to_string());
                if r.starts_with("ERROR:") { Err(r) } else { Ok(r) }
            }
        }
        js_runtime::set_web_driver(Arc::new(DriverWrapper { inner: d }));
    }

    let root = if sandbox_root.is_empty() { None } else { Some(std::path::Path::new(&sandbox_root)) };
    let result = js_runtime::run_javascript(&code, root, None);
    serde_json::to_string(&result).unwrap_or_default()
}

/// Run inference with streaming output, using the LUT engine path.
///
/// `cimage_path` — compiled .cimage from `prism_compile_gguf`.
/// `model_dir` — directory containing tokenizer.json and config.json.
#[uniffi::export]
pub fn prism_infer_multimodal_stream(
    cimage_path: String,
    model_dir: String,
    prompt: String,
    callback: Box<dyn MultimodalStreamCallback>,
) {
    // ── Load model ────────────────────────────────────────────────
    let config_path = std::path::Path::new(&model_dir).join("config.json");
    let config = match tribunus_compute_core::lut::graph::UnifiedConfig::from_file(&config_path) {
        Ok(c) => c,
        Err(e) => {
            callback.on_error(format!("load config: {e}"));
            return;
        }
    };
    let graph = tribunus_compute_core::lut::graph::ModelGraph::build(&config);
    let mut engine = match tribunus_compute_core::lut::engine::PrismEngine::load(
        std::path::Path::new(&cimage_path),
        graph,
    ) {
        Ok(e) => e,
        Err(e) => {
            callback.on_error(format!("load engine: {e}"));
            return;
        }
    };
    let tokenizer = match tribunus_compute_core::tokenizer::TribunusTokenizer::from_dir(
        std::path::Path::new(&model_dir),
    ) {
        Ok(t) => t,
        Err(e) => {
            callback.on_error(format!("load tokenizer: {e}"));
            return;
        }
    };

    // ── Tokenise prompt ──────────────────────────────────────────
    let prompt_tokens = match tokenizer.encode(&prompt) {
        Ok(t) => t,
        Err(e) => {
            callback.on_error(format!("tokenize: {e}"));
            return;
        }
    };

    // ── Run inference on background thread ───────────────────────
    let callback = std::sync::Arc::new(callback);
    std::thread::spawn(move || {
        let max_tokens = 512;
        match engine.generate(&prompt_tokens, max_tokens) {
            Ok(stats) => {
                for &token_id in &stats.generated_tokens {
                    match tokenizer.decode(&[token_id]) {
                        Ok(text) => {
                            callback.on_event(StreamEvent::Text { token: text });
                        }
                        Err(_) => {}
                    }
                }
                callback.on_done();
            }
            Err(e) => {
                callback.on_error(format!("generate: {e}"));
            }
        }
    });
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
        agent::StepOutcome::SubagentSpawned(handle) => {
            BridgeStepOutcome::AwaitingSubagents {
                subagents: vec![BridgeSubagentHandle {
                    id: handle.id,
                    goal: handle.goal.clone(),
                    sandbox_subpath: handle.sandbox_subpath.clone(),
                    max_revisions: 3,
                }],
            }
        }
        agent::StepOutcome::SubagentResult { .. } => {
            // This is an internal transition — the app feeds subagent
            // results via an explicit call, not through step().
            BridgeStepOutcome::Generating
        }
        agent::StepOutcome::Finished { output } => {
            BridgeStepOutcome::Finished {
                result: output.clone(),
            }
        }
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
