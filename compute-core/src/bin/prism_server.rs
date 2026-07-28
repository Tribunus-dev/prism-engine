//! prism-server — OpenAI-compatible local API server for Prism Engine.
//!
//! Loads a compiled `.cimage` model and HuggingFace tokenizer, then serves
//! the OpenAI `/v1/chat/completions` and `/v1/completions` endpoints.
//!
//! Usage:
//!   cargo run --release -p tribunus-compute-core --bin prism-server \
//!       --features metal-dispatch -- \
//!       --cimage /tmp/prism-test/model.cimage \
//!       --model-dir models/qwen2.5-0.5b \
//!       --port 8080

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use axum::{
    body::Body,
    extract::Path,
    extract::Request,
    extract::State,
    http::StatusCode,
    middleware::{self, Next},
    response::sse::{Event, Sse},
    response::{IntoResponse, Json, Response},
    routing::{get, post},
    Router,
};
use base64::Engine;
use clap::Parser;
use futures::stream::Stream;
use futures::StreamExt;
use metal;
use parking_lot::Mutex as ParkingMutex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::convert::Infallible;
use std::pin::Pin;
use std::task::Context;
use std::task::Poll;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use prism_ecs_core::identity::{
    CompilerIdentity, GenerationId, HardwareProfileId, ModelSourceId, ReceiptId, Timestamp,
};
#[cfg(feature = "mlx-backend")]
use tribunus_compute_core::audio_preprocess_accelerate;
use tribunus_compute_core::compilation::cancel::CancelToken;
use tribunus_compute_core::ecs::canonical::{
    CimageGeneration, ExecutionGraph, MemoryPlan, RuntimeStatePlan,
};
use tribunus_compute_core::ecs::legacy_cimage::CIMAGE_MAGIC;
use tribunus_compute_core::ecs::cimage_runtime::context::CimageRuntimeContext;
use tribunus_compute_core::ecs::cimage_runtime::tensor_store::RuntimeTensorStore;
use tribunus_compute_core::ecs::compiler::deployment_compiler::ServingProfile;
use tribunus_compute_core::ecs::compute_image::cimage_loader::CimageDeployment;
use tribunus_compute_core::ecs::runtime::serving::model_instance::ModelRegistry;
use tribunus_compute_core::ecs::runtime::serving::model_instance::{
    CimageModelInstance, InferenceSession, MtpSessionState, SamplerConfig,
};
use tribunus_compute_core::server::distill_worker::{
    DistillationEngine, DistillationJobStatus, DistillationRequest,
};
use tribunus_compute_core::server::state::MemoryAllocationBroker;
use tribunus_compute_core::tokenizer::TribunusTokenizer;
use tribunus_compute_core::tts::pipeline::{pcm_chunk_to_wav, pcm_to_wav, TtsPipeline};

// ── BitNet runtime ─────────────────────────────────────────────────────────

/// Placeholder stub — will be replaced by
/// `prism_ecs_quantization::bitnet::text::BitNetRuntime` when the module
/// is fully wired up.
struct BitNetRuntime;
impl BitNetRuntime {
    fn from_cimage(_path: &std::path::Path) -> Result<Self, String> {
        println!("[prism-server] BitNet runtime initialization (stub)");
        Ok(Self)
    }
}

// ── Dashboard stub handlers (until full indexer is wired) ─────────────────

#[cfg(feature = "server-dashboard")]
mod dashboard_stubs {
    use axum::extract::Path;
    use axum::response::{Html, Json};
    use serde_json::{json, Value};

    pub async fn list_cimages() -> Json<Value> {
        Json(json!({"status": "not yet indexed", "cimages": []}))
    }

    pub async fn get_cimage(Path(digest): Path<String>) -> Json<Value> {
        Json(json!({"status": "not yet indexed", "digest": digest}))
    }

    pub async fn get_cimage_tensors(Path(digest): Path<String>) -> Json<Value> {
        Json(json!({"status": "not yet indexed", "digest": digest, "tensors": []}))
    }

    pub async fn openapi_schema() -> Json<Value> {
        Json(json!({
            "openapi": "3.0.0",
            "info": {"title": "Prism Engine API", "version": "0.1.0"},
            "paths": {}
        }))
    }

    pub async fn dashboard_spa() -> impl axum::response::IntoResponse {
        Html(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/ecs/server/dashboard/page.html"
        )))
    }

    pub async fn dashboard_root() -> axum::response::Redirect {
        axum::response::Redirect::temporary("/dashboard")
    }
}

// ── CLI ─────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "prism-server", about = "OpenAI-compatible local API")]
struct Args {
    /// Path to the compiled .cimage.
    #[arg(long)]
    cimage: PathBuf,

    /// Path to the model directory (for config.json + tokenizer.json).
    #[arg(long)]
    model_dir: PathBuf,

    /// Path to server config file (TOML). Optional — uses defaults if absent.
    #[arg(long)]
    config: Option<PathBuf>,

    /// Server port.
    #[arg(long, default_value = "8080")]
    port: u16,

    /// Cimage format: auto-detect, old (compute_image), or new (cimage).
    #[arg(long, default_value = "auto")]
    cimage_format: String,
}

// ── Config ────────────────────────────────────────────────────────────────

/// Top-level server configuration loaded from a TOML file.
#[derive(Deserialize, Clone)]
#[allow(dead_code)]
struct ServerConfig {
    server: ServerSection,
    auth: AuthSection,
    limits: LimitSection,
    models: Vec<ModelEntry>,
}

#[derive(Deserialize, Clone)]
#[allow(dead_code)]
struct ServerSection {
    #[serde(default = "default_mode")]
    mode: String,
    #[serde(default = "default_host")]
    host: String,
    #[serde(default = "default_port")]
    port: u16,
    #[serde(default)]
    unix_socket: Option<String>,
    #[serde(default)]
    tls: Option<TlsConfig>,
}

#[derive(Deserialize, Clone)]
#[allow(dead_code)]
struct TlsConfig {
    cert_path: PathBuf,
    key_path: PathBuf,
}

#[derive(Deserialize, Clone)]
struct AuthSection {
    #[serde(default)]
    enabled: bool,
    #[serde(default)]
    tokens: Vec<TokenDef>,
    #[serde(default)]
    admin_token: Option<String>,
}

#[derive(Deserialize, Clone)]
#[allow(dead_code)]
struct TokenDef {
    id: String,
    #[serde(rename = "token")]
    raw_token: Option<String>,
    #[serde(default)]
    verifier: Option<String>,
    #[serde(default = "default_scopes")]
    scopes: Vec<String>,
}

#[derive(Deserialize, Clone)]
#[allow(dead_code)]
struct LimitSection {
    #[serde(default = "default_body_bytes")]
    max_body_bytes: u64,
    #[serde(default = "default_prompt_tokens")]
    max_prompt_tokens: u32,
    #[serde(default = "default_output_tokens")]
    max_output_tokens: u32,
    #[serde(default = "default_images")]
    max_images: u32,
    #[serde(default = "default_audio_seconds")]
    max_audio_seconds: u32,
    #[serde(default = "default_stream_timeout")]
    stream_idle_timeout_secs: u32,
    #[serde(default = "default_deadline")]
    total_request_deadline_secs: u32,
}

#[derive(Deserialize, Clone)]
#[allow(dead_code)]
struct ModelEntry {
    name: String,
    cimage_path: PathBuf,
}

// ── Config default values ────────────────────────────────────────────────

fn default_mode() -> String {
    "embedded".into()
}
fn default_host() -> String {
    "127.0.0.1".into()
}
fn default_port() -> u16 {
    8080
}
fn default_body_bytes() -> u64 {
    10 * 1024 * 1024
}
fn default_prompt_tokens() -> u32 {
    128_000
}
fn default_output_tokens() -> u32 {
    4096
}
fn default_images() -> u32 {
    4
}
fn default_audio_seconds() -> u32 {
    30
}
fn default_stream_timeout() -> u32 {
    60
}
fn default_deadline() -> u32 {
    300
}
fn default_scopes() -> Vec<String> {
    vec!["generation.text".into()]
}

// ── Request lifecycle types ─────────────────────────────────────────────

#[derive(Clone, Serialize)]
struct RequestReceipt {
    request_id: String,
    model_digest: Option<String>,
    client_id: Option<String>,
    terminal_state: String,
    prompt_tokens: u32,
    completion_tokens: u32,
    audio_duration_ms: u32,
    queue_time_us: u64,
    execution_time_us: u64,
    error_code: Option<String>,
    error_message: Option<String>,
}

struct StreamState {
    #[allow(dead_code)]
    request_id: String,
    cancel: CancellationToken,
    #[allow(dead_code)]
    start_time: Instant,
    #[allow(dead_code)]
    last_activity: Instant,
    #[allow(dead_code)]
    terminal_sent: AtomicBool,
    receipt: parking_lot::Mutex<Option<RequestReceipt>>,
}

struct StreamTracker {
    streams: HashMap<String, Arc<StreamState>>,
}

impl StreamTracker {
    fn new() -> Self {
        Self {
            streams: HashMap::new(),
        }
    }

    fn register(&mut self, request_id: String, state: Arc<StreamState>) {
        self.streams.insert(request_id, state);
    }

    #[allow(dead_code)]
    fn cancel(&mut self, request_id: &str) {
        if let Some(state) = self.streams.remove(request_id) {
            state.cancel.cancel();
        }
    }

    fn cancel_all(&mut self) {
        for (_, state) in self.streams.drain() {
            state.cancel.cancel();
        }
    }
    fn active_count(&self) -> usize {
        self.streams.len()
    }
}

/// Stream wrapper that cancels the token when the stream is dropped.
struct CancelOnDropStream<S> {
    inner: S,
    token: CancellationToken,
}

impl<S, I> Stream for CancelOnDropStream<S>
where
    S: Stream<Item = I> + Unpin,
{
    type Item = I;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.inner).poll_next(cx)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<S> Drop for CancelOnDropStream<S> {
    fn drop(&mut self) {
        self.token.cancel();
    }
}

// ── State ───────────────────────────────────────────────────────────────

struct AppState {
    model_registry: ParkingMutex<ModelRegistry>,
    tokenizer: TribunusTokenizer,
    #[allow(dead_code)]
    decode_count: std::sync::atomic::AtomicU64,
    #[allow(dead_code)]
    cimage_path: PathBuf,
    /// Server configuration (from TOML or defaults).
    config: ServerConfig,
    /// Token authentication verifier derived from config.
    auth_verifier: AuthVerifier,
    /// Optional TTS pipeline loaded when the cimage contains TTS segments.
    #[allow(dead_code)]
    tts_pipeline: Option<TtsPipeline>,
    /// Whether cimage uses new format (versus legacy compute_image).
    #[allow(dead_code)]
    cimage_format: String,
    /// BitNet runtime for new-format cimages (None when using legacy format).
    bitnet_runtime: Option<ParkingMutex<BitNetRuntime>>,
    /// Model content digest for receipt tracking.
    model_digest: Option<String>,
    stream_tracker: ParkingMutex<StreamTracker>,
    #[allow(dead_code)]
    memory_broker: Arc<MemoryAllocationBroker>,
    distill_engine: Arc<DistillationEngine>,
    #[allow(dead_code)]
    cancel: CancelToken,
}

// ── Auth verifier ─────────────────────────────────────────────────────

/// Token authentication verifier. Stores SHA-256 hashes for constant-time
/// comparison against incoming Bearer tokens.
struct AuthVerifier {
    entries: Vec<([u8; 32], Vec<String>)>,
    admin_token_hash: Option<[u8; 32]>,
}

impl AuthVerifier {
    fn new(config: &AuthSection) -> Self {
        let mut entries = Vec::new();
        for def in &config.tokens {
            let hash: [u8; 32] = if let Some(raw) = &def.raw_token {
                let mut h = Sha256::new();
                h.update(raw.as_bytes());
                h.finalize().into()
            } else if let Some(ver) = &def.verifier {
                let decoded = base64::engine::general_purpose::STANDARD
                    .decode(ver)
                    .unwrap_or_default();
                let mut out = [0u8; 32];
                let len = decoded.len().min(32);
                out[..len].copy_from_slice(&decoded[..len]);
                out
            } else {
                continue;
            };
            entries.push((hash, def.scopes.clone()));
        }
        let admin_token_hash = config.admin_token.as_ref().map(|t| {
            let mut h = Sha256::new();
            h.update(t.as_bytes());
            h.finalize().into()
        });
        Self {
            entries,
            admin_token_hash,
        }
    }

    fn verify(&self, token: &str, required_scope: &str) -> bool {
        let mut h = Sha256::new();
        h.update(token.as_bytes());
        let token_hash: [u8; 32] = h.finalize().into();

        // Admin token has all scopes.
        if let Some(admin) = &self.admin_token_hash {
            if constant_time_eq(&token_hash, admin) {
                return true;
            }
        }
        // Regular tokens must match hash and scope.
        for (stored_hash, scopes) in &self.entries {
            if constant_time_eq(&token_hash, stored_hash)
                && scopes.iter().any(|s| s == required_scope)
            {
                return true;
            }
        }
        false
    }
}

/// Simple constant-time comparison for fixed-size byte arrays.
fn constant_time_eq(a: &[u8; 32], b: &[u8; 32]) -> bool {
    let mut diff = 0u8;
    for i in 0..32 {
        diff |= a[i] ^ b[i];
    }
    diff == 0
}

// ── OpenAI API types ────────────────────────────────────────────────────

#[derive(Deserialize)]
struct ChatRequest {
    model: Option<String>,
    messages: Vec<ChatMessage>,
    max_tokens: Option<u32>,
    #[allow(dead_code)]
    response_format: Option<String>, // "text" or "audio"
    #[allow(dead_code)]
    temperature: Option<f32>,
    stream: Option<bool>,
}

/// OpenAI multimodal content part — text, image, or audio.
#[derive(Deserialize)]
#[serde(tag = "type")]
enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
    #[serde(rename = "input_audio")]
    InputAudio { input_audio: InputAudio },
}

#[derive(Deserialize)]
struct ImageUrl {
    url: String,
}

#[derive(Deserialize)]
struct InputAudio {
    data: String,
    #[allow(dead_code)]
    format: String,
}

/// A chat message content is either a plain string or an array of typed parts.
#[derive(Deserialize)]
#[serde(untagged)]
enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Deserialize)]
struct ChatMessage {
    #[allow(dead_code)]
    role: String,
    #[serde(deserialize_with = "deserialize_content")]
    content: MessageContent,
}

#[derive(Serialize)]
struct ChatResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<Choice>,
    usage: Usage,
}

#[derive(Serialize)]
struct Choice {
    index: u32,
    message: ChatResponseMessage,
    finish_reason: String,
}

#[derive(Serialize)]
struct ChatResponseMessage {
    role: String,
    content: ResponseContent,
}

#[derive(Serialize)]
#[serde(untagged)]
enum ResponseContent {
    Text(String),
    #[allow(dead_code)]
    Parts(Vec<ResponseContentPart>),
}

#[derive(Serialize)]
#[serde(tag = "type")]
enum ResponseContentPart {
    #[serde(rename = "text")]
    #[allow(dead_code)]
    Text { text: String },
    /// Base64-encoded WAV audio data.
    #[serde(rename = "audio_url")]
    #[allow(dead_code)]
    Audio { audio_url: String },
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

#[derive(Serialize)]
struct ModelList {
    object: String,
    data: Vec<ModelInfo>,
}

#[derive(Serialize)]
struct ModelInfo {
    id: String,
    object: String,
    created: u64,
    owned_by: String,
}

// ── /v1/completions types ────────────────────────────────────────────────

#[derive(Deserialize)]
#[allow(dead_code)]
struct CompletionRequest {
    model: Option<String>,
    prompt: String,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    stream: Option<bool>,
    #[allow(dead_code)]
    stop: Option<Vec<String>>,
}

#[derive(Serialize)]
struct CompletionResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<CompletionChoice>,
    usage: Usage,
}

#[derive(Serialize)]
struct CompletionChoice {
    text: String,
    index: u32,
    finish_reason: String,
}

// ── Multimodal types ──────────────────────────────────────────────────

/// A decoded multimodal input part — text, image bytes, or audio bytes.
enum MultimodalPart {
    Text(String),
    #[allow(dead_code)]
    Image(Vec<u8>),
    #[allow(dead_code)]
    Audio(Vec<u8>),
}

// ── Ollama-compatible API types ──────────────────────────────────────────

#[derive(Deserialize)]
#[allow(dead_code)]
struct OllamaGenerateRequest {
    model: String,
    prompt: String,
    #[serde(default = "default_true")]
    stream: bool,
    options: Option<serde_json::Value>,
    format: Option<String>,
}

fn default_true() -> bool {
    true
}

#[derive(Serialize)]
struct OllamaGenerateResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    message: Option<OllamaMessage>,
    done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    total_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    load_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_eval_count: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_eval_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    eval_count: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    eval_duration: Option<u64>,
}

#[derive(Deserialize)]
struct OllamaChatRequest {
    model: String,
    messages: Vec<OllamaChatMessage>,
    #[allow(dead_code)]
    #[serde(default = "default_true")]
    stream: bool,
}

#[derive(Deserialize)]
struct OllamaChatMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct OllamaMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct OllamaTagsResponse {
    models: Vec<OllamaModelInfo>,
}

#[derive(Serialize)]
struct OllamaModelInfo {
    name: String,
    modified_at: String,
    size: u64,
}

#[derive(Deserialize)]
struct OllamaShowRequest {
    model: String,
}

#[derive(Serialize)]
struct OllamaShowResponse {
    model: String,
    modified_at: String,
    size: u64,
    digest: String,
    details: serde_json::Value,
}

#[derive(Serialize)]
struct OllamaPsResponse {
    models: Vec<OllamaPsModelInfo>,
}

#[derive(Serialize)]
struct OllamaPsModelInfo {
    name: String,
    size: u64,
    processor: String,
    memory: u64,
    until: String,
}
// ── Deserialization helpers ───────────────────────────────────────────

/// Deserializes a `MessageContent` from either a JSON string or an array of content parts.
fn deserialize_content<'de, D>(d: D) -> Result<MessageContent, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize as _;
    let value = serde_json::Value::deserialize(d)?;
    match value {
        serde_json::Value::String(s) => Ok(MessageContent::Text(s)),
        serde_json::Value::Array(arr) => {
            let parts: Vec<ContentPart> = serde_json::from_value(serde_json::Value::Array(arr))
                .map_err(serde::de::Error::custom)?;
            Ok(MessageContent::Parts(parts))
        }
        _ => Err(serde::de::Error::custom(
            "expected a string or an array of content parts",
        )),
    }
}

// ── Multimodal helpers ────────────────────────────────────────────────

/// Decode a `data:` URL into raw bytes.
fn decode_data_url(url: &str) -> Result<Vec<u8>, String> {
    let rest = url
        .strip_prefix("data:")
        .ok_or_else(|| "not a data URL".to_string())?;
    let encoded = rest
        .split(',')
        .nth(1)
        .ok_or_else(|| "no data in URL".to_string())?;
    base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .map_err(|e| format!("base64 decode: {e}"))
}

/// Extract multimodal parts from a message content.
fn extract_multimodal_parts(msg: &MessageContent) -> Vec<MultimodalPart> {
    let mut parts = Vec::new();
    match msg {
        MessageContent::Text(t) => {
            parts.push(MultimodalPart::Text(t.clone()));
        }
        MessageContent::Parts(ps) => {
            for p in ps {
                match p {
                    ContentPart::Text { text } => parts.push(MultimodalPart::Text(text.clone())),
                    ContentPart::ImageUrl { image_url } => {
                        if let Ok(bytes) = decode_data_url(&image_url.url) {
                            parts.push(MultimodalPart::Image(bytes));
                        }
                    }
                    ContentPart::InputAudio { input_audio } => {
                        if let Ok(bytes) =
                            base64::engine::general_purpose::STANDARD.decode(&input_audio.data)
                        {
                            parts.push(MultimodalPart::Audio(bytes));
                        }
                    }
                }
            }
        }
    }
    parts
}

// ── API error envelope ────────────────────────────────────────────────────

#[derive(Serialize)]
struct ApiError {
    code: String,
    message: String,
    request_id: String,
    retryable: bool,
}

impl ApiError {
    fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            request_id: Uuid::new_v4().to_string(),
            retryable: false,
        }
    }

    #[allow(dead_code)]
    fn with_request_id(mut self, id: String) -> Self {
        self.request_id = id;
        self
    }

    #[allow(dead_code)]
    fn with_retryable(mut self, val: bool) -> Self {
        self.retryable = val;
        self
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = match self.code.as_str() {
            "unauthorized" => StatusCode::UNAUTHORIZED,
            "forbidden" => StatusCode::FORBIDDEN,
            "rate_limited" => StatusCode::TOO_MANY_REQUESTS,
            "capacity_exhausted" => StatusCode::SERVICE_UNAVAILABLE,
            "invalid_request" => StatusCode::BAD_REQUEST,
            "timeout" => StatusCode::REQUEST_TIMEOUT,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        (status, Json(self)).into_response()
    }
}

impl From<StatusCode> for ApiError {
    fn from(status: StatusCode) -> Self {
        let (code, message) = match status {
            StatusCode::BAD_REQUEST => ("invalid_request", "Bad request"),
            StatusCode::SERVICE_UNAVAILABLE => ("capacity_exhausted", "Service unavailable"),
            StatusCode::INTERNAL_SERVER_ERROR => ("internal_error", "Internal server error"),
            StatusCode::NOT_IMPLEMENTED => ("not_implemented", "Not implemented"),
            StatusCode::REQUEST_TIMEOUT => ("timeout", "Request timed out"),
            _ => ("error", "Unknown error"),
        };
        Self {
            code: code.into(),
            message: message.into(),
            request_id: Uuid::new_v4().to_string(),
            retryable: false,
        }
    }
}

// ── Handlers ────────────────────────────────────────────────────────────

async fn list_models(State(state): State<Arc<AppState>>) -> Json<ModelList> {
    let models = state.model_registry.lock();
    let mut data: Vec<ModelInfo> = models
        .iter()
        .map(|(key, inst)| ModelInfo {
            id: key.to_string(),
            object: "model".to_string(),
            created: inst.loaded_at.elapsed().as_secs(),
            owned_by: "prism".to_string(),
        })
        .collect();
    data.sort_by(|a, b| a.id.cmp(&b.id));
    Json(ModelList {
        object: "list".to_string(),
        data,
    })
}

/// GET /ready — liveness / readiness probe.
async fn readiness() -> Json<serde_json::Value> {
    Json(serde_json::json!({"status": "ready", "version": env!("CARGO_PKG_VERSION")}))
}

/// GET /v1/runtime/receipts/{request_id} — diagnostic receipt lookup.
async fn diagnostic_receipt(
    State(_state): State<Arc<AppState>>,
    Path(request_id): Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let _ = request_id;
    Err(ApiError::new(
        "not_implemented",
        "Receipt lookup not yet wired",
    ))
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatRequest>,
) -> Result<Response, ApiError> {
    if state.bitnet_runtime.is_some() {
        return Ok((
            StatusCode::NOT_IMPLEMENTED,
            "BitNet generation not yet implemented",
        )
            .into_response());
    }
    let request_id = Uuid::new_v4().to_string();
    let start_time = Instant::now();

    // Audio output via SSE streaming when response_format contains "audio"
    if req
        .response_format
        .as_ref()
        .map_or(false, |f| f.contains("audio"))
    {
        let sse = chat_completions_audio_stream(state, req)
            .await
            .map_err(|e| ApiError::new("internal_error", e.to_string()))?;
        return Ok(sse.into_response());
    }

    if req.stream.unwrap_or(false) {
        let stream_response = chat_completions_stream(state, req, request_id).await?;
        return Ok(stream_response.into_response());
    }

    // Stage 1: Validate request properties (no resource reservation)
    if req.max_tokens.unwrap_or(0) > state.config.limits.max_output_tokens {
        return Err(ApiError {
            code: "invalid_request".into(),
            message: format!(
                "max_tokens exceeds limit of {}",
                state.config.limits.max_output_tokens
            ),
            request_id: request_id.clone(),
            retryable: false,
        });
    }

    // Extract prompt from last user message
    let user_msg = req.messages.last().ok_or(StatusCode::BAD_REQUEST)?;
    let multimodal_parts = extract_multimodal_parts(&user_msg.content);

    // Collect text parts for tokenization
    let prompt_text: Vec<&str> = multimodal_parts
        .iter()
        .filter_map(|p| match p {
            MultimodalPart::Text(t) => Some(t.as_str()),
            _ => None,
        })
        .collect();
    let prompt_text = prompt_text.join(" ");

    // Check if multimodal (image or audio present)
    let has_multimodal = multimodal_parts
        .iter()
        .any(|p| matches!(p, MultimodalPart::Image(_) | MultimodalPart::Audio(_)));

    // Multimodal not yet wired through ModelRegistry — skip encoding
    if has_multimodal {
        eprintln!("[prism-server] multimodal input ignored — not yet wired through ModelRegistry");
    }

    // Tokenize (no executor lock needed)
    let input_ids = state
        .tokenizer
        .encode(&prompt_text)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    if input_ids.is_empty() {
        return Err(StatusCode::BAD_REQUEST.into());
    }

    let max_tokens = req.max_tokens.unwrap_or(256) as usize;
    let prompt_len = input_ids.len();

    // Acquire model from registry (removes it temporarily — re-inserted after decode)
    let mut model = {
        let mut registry = state.model_registry.lock();
        registry
            .instances
            .remove("default")
            .ok_or_else(|| ApiError::new("model_not_found", "No model loaded"))?
    };

    // Create inference session metadata (wired in future waves)
    let mut session = InferenceSession {
        session_id: request_id.clone(),
        scheduler_slot: 0,
        kv_epoch: 0,
        sampler: SamplerConfig::default(),
        mtp_state: if model.profile.mtp_enabled {
            Some(MtpSessionState {
                draft_tokens: vec![],
                verified_count: 0,
                accepted_count: 0,
                rejection_position: None,
            })
        } else {
            None
        },
    };

    // Run decode in blocking task (Metal is synchronous)
    let generated_tokens = tokio::task::spawn_blocking(move || {
        let sampling = session.sampler.clone();

        // Prefill
        model
            .prefill(&mut session, &input_ids)
            .map_err(|e| format!("prefill failed: {e}"))?;

        // Autoregressive decode
        let mut generated_tokens = Vec::with_capacity(max_tokens);
        for _ in 0..max_tokens {
            let token_id = model
                .decode(&mut session, &sampling)
                .map_err(|_| "decode failed".to_string())?
                .token_id;
            generated_tokens.push(token_id);
            if token_id == 1 {
                break; // EOS
            }
        }

        Ok::<(Vec<u32>, CimageModelInstance), String>((generated_tokens, model))
    })
    .await;

    // Re-register model in registry
    let (tokens_result, model_opt) = match generated_tokens {
        Ok(Ok((tokens, model))) => (Ok(Ok(tokens)), Some(model)),
        Ok(Err(e)) => (Ok(Err::<Vec<u32>, String>(e)), None),
        Err(e) => (Err::<Result<Vec<u32>, String>, _>(e), None),
    };

    if let Some(model) = model_opt {
        let mut registry = state.model_registry.lock();
        registry.instances.insert("default".to_string(), model);
    }

    let generated_tokens = tokens_result;
    let generated_tokens = match generated_tokens {
        Ok(Ok(tokens)) => tokens,
        Ok(Err(e)) => {
            let receipt = RequestReceipt {
                request_id: request_id.clone(),
                model_digest: state.model_digest.clone(),
                client_id: None,
                terminal_state: "failed".into(),
                prompt_tokens: prompt_len as u32,
                completion_tokens: 0,
                audio_duration_ms: 0,
                queue_time_us: 0,
                execution_time_us: start_time.elapsed().as_micros() as u64,
                error_code: Some("internal_error".into()),
                error_message: Some(e),
            };
            eprintln!(
                "[prism-server] receipt: {}",
                serde_json::to_string(&receipt).unwrap()
            );
            return Err(ApiError::new("internal_error", "Decode failed"));
        }
        Err(e) => {
            let receipt = RequestReceipt {
                request_id: request_id.clone(),
                model_digest: state.model_digest.clone(),
                client_id: None,
                terminal_state: "failed".into(),
                prompt_tokens: prompt_len as u32,
                completion_tokens: 0,
                audio_duration_ms: 0,
                queue_time_us: 0,
                execution_time_us: start_time.elapsed().as_micros() as u64,
                error_code: Some("internal_error".into()),
                error_message: Some(format!("Decode task panicked: {e}")),
            };
            eprintln!(
                "[prism-server] receipt: {}",
                serde_json::to_string(&receipt).unwrap()
            );
            return Err(ApiError::new("internal_error", "Decode task panicked"));
        }
    };

    // Terminal receipt
    let receipt = RequestReceipt {
        request_id: request_id.clone(),
        model_digest: state.model_digest.clone(),
        client_id: None,
        terminal_state: "completed".into(),
        prompt_tokens: prompt_len as u32,
        completion_tokens: generated_tokens.len() as u32,
        audio_duration_ms: 0,
        queue_time_us: 0,
        execution_time_us: start_time.elapsed().as_micros() as u64,
        error_code: None,
        error_message: None,
    };
    eprintln!(
        "[prism-server] receipt: {}",
        serde_json::to_string(&receipt).unwrap()
    );

    let output_text = state
        .tokenizer
        .decode(&generated_tokens)
        .unwrap_or_else(|_| format!("[prism] {} tokens generated", generated_tokens.len()));

    Ok(Json(ChatResponse {
        id: "cmpl-1".to_string(),
        object: "chat.completion".to_string(),
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        model: req.model.unwrap_or_else(|| "prism".to_string()),
        choices: vec![Choice {
            index: 0,
            message: ChatResponseMessage {
                role: "assistant".to_string(),
                content: ResponseContent::Text(output_text),
            },
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens: prompt_len as u32,
            completion_tokens: generated_tokens.len() as u32,
            total_tokens: (prompt_len + generated_tokens.len()) as u32,
        },
    })
    .into_response())
}

/// /v1/completions handler — OpenAI-compatible text completions.
async fn completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>, StatusCode> {
    // Tokenize
    let input_ids = state
        .tokenizer
        .encode(&req.prompt)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    if input_ids.is_empty() {
        return Err(StatusCode::BAD_REQUEST);
    }

    let max_tokens = req.max_tokens.unwrap_or(256) as usize;
    let prompt_len = input_ids.len();

    // Acquire model from registry
    let mut models = state.model_registry.lock();
    let model = models
        .instances
        .get_mut("default")
        .ok_or(StatusCode::SERVICE_UNAVAILABLE)?;

    // Create inference session
    let mut session = InferenceSession {
        session_id: "completions".into(),
        scheduler_slot: 0,
        kv_epoch: 0,
        sampler: SamplerConfig::default(),
        mtp_state: None,
    };

    // Prefill
    model
        .prefill(&mut session, &input_ids)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let sampling = session.sampler.clone();
    let mut generated_text = String::new();

    for _step in 0..max_tokens {
        match model.decode(&mut session, &sampling) {
            Ok(result) => {
                let next = result.token_id;
                let token_text = state.tokenizer.decode(&[next]).unwrap_or_default();
                generated_text.push_str(&token_text);

                if next == 0 || token_text.is_empty() {
                    break;
                }
            }
            Err(_) => break,
        }
    }

    drop(models);

    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let completion_tokens = generated_text.split_whitespace().count() as u32;

    Ok(Json(CompletionResponse {
        id: format!("cmpl-{}", Uuid::new_v4()),
        object: "text_completion".to_string(),
        created,
        model: "prism-model".to_string(),
        choices: vec![CompletionChoice {
            text: generated_text,
            index: 0,
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens: prompt_len as u32,
            completion_tokens,
            total_tokens: (prompt_len + completion_tokens as usize) as u32,
        },
    }))
}

// ── Distillation handlers ────────────────────────────────────────────────

async fn post_distill(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<DistillationRequest>,
) -> Result<Json<DistillationJobStatus>, ApiError> {
    let job_id = state
        .distill_engine
        .submit(payload)
        .await
        .map_err(|e| ApiError::new("internal_error", e))?;
    let status = state
        .distill_engine
        .status(&job_id)
        .await
        .ok_or_else(|| ApiError::new("not_found", format!("job {job_id} not found")))?;
    Ok(Json(status))
}

async fn get_distill_status(
    State(state): State<Arc<AppState>>,
    Path(job_id): Path<String>,
) -> Result<Json<DistillationJobStatus>, ApiError> {
    let status = state
        .distill_engine
        .status(&job_id)
        .await
        .ok_or_else(|| ApiError::new("not_found", format!("job {job_id} not found")))?;
    Ok(Json(status))
}

// ── Model management handlers ────────────────────────────────────────────

#[derive(Deserialize)]
struct LoadModelRequest {
    #[allow(dead_code)]
    cimage_path: String,
}

async fn load_model(
    State(state): State<Arc<AppState>>,
    Json(body): Json<LoadModelRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let _ = state;
    let _ = body;
    Err(ApiError::new(
        "not_implemented",
        "Model loading through registry not yet wired",
    ))
}

#[derive(Deserialize)]
struct DeployModelRequest {
    #[allow(dead_code)]
    cimage_path: String,
    #[serde(default = "default_quality_gate")]
    #[allow(dead_code)]
    quality_gate: bool,
}

fn default_quality_gate() -> bool {
    true
}

async fn deploy_model(
    State(state): State<Arc<AppState>>,
    Json(body): Json<DeployModelRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let _ = state;
    let _ = body;
    Err(ApiError::new(
        "not_implemented",
        "Model deployment through registry not yet wired",
    ))
}

#[derive(Deserialize)]
struct MergeAdapterRequest {
    adapter_name: String,
}

async fn merge_adapter(
    State(state): State<Arc<AppState>>,
    Json(body): Json<MergeAdapterRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let _ = state; // unused for now
    Ok(Json(
        serde_json::json!({"status": "merged", "adapter": body.adapter_name}),
    ))
}

// ── Rollback endpoints ──────────────────────────────────────────────────

/// POST /v1/sessions/{session_id}/rollback — request-scope rollback

async fn session_rollback(
    State(state): State<Arc<AppState>>,
    Path(session_id): Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let mut models = state.model_registry.lock();
    if let Some(model) = models.instances.get_mut("default") {
        let mut session = InferenceSession {
            session_id: session_id.clone(),
            scheduler_slot: 0,
            kv_epoch: 0,
            sampler: SamplerConfig::default(),
            mtp_state: None,
        };
        model.rollback(&mut session);
        Ok(Json(
            serde_json::json!({"status": "rolled_back", "session_id": session_id}),
        ))
    } else {
        Err(ApiError::new("model_not_found", "No model loaded"))
    }
}

/// POST /v1/kv/epoch/{epoch}/rollback — KV epoch rollback
async fn kv_epoch_rollback(
    State(state): State<Arc<AppState>>,
    Path(epoch): Path<u64>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let mut models = state.model_registry.lock();
    if let Some(model) = models.instances.get_mut("default") {
        let mut session = InferenceSession {
            session_id: "kv-rollback".into(),
            scheduler_slot: 0,
            kv_epoch: epoch,
            sampler: SamplerConfig::default(),
            mtp_state: None,
        };
        model.rollback(&mut session);
        Ok(Json(
            serde_json::json!({"status": "rolled_back", "epoch": epoch}),
        ))
    } else {
        Err(ApiError::new("model_not_found", "No model loaded"))
    }
}

/// POST /v1/generations/{gen_id}/rollback — generation rollback
async fn generation_rollback(
    State(_state): State<Arc<AppState>>,
    Path(_gen_id): Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    // Generation rollback restores the parent generation
    // This is handled through the lifecycle coordinator's rollback
    Err(ApiError::new(
        "not_implemented",
        "Generation rollback not yet wired through deployment compiler",
    ))
}

#[allow(dead_code)]
fn decode_image_to_rgb(bytes: &[u8]) -> Result<(u32, u32, Vec<u8>), String> {
    use image::GenericImageView;
    let img = image::load_from_memory(bytes).map_err(|e| format!("image decode failed: {e}"))?;
    let (w, h) = img.dimensions();
    let rgb = img.to_rgb8().into_raw();
    Ok((w, h, rgb))
}

/// Stub: returns empty embeddings until the multimodal projection is wired through model_registry.
#[allow(dead_code)]
fn multimodal_encode_images(_images: &[Vec<u8>]) -> Result<Vec<Vec<f32>>, String> {
    Ok(Vec::new())
}

// ---- TTS helpers --------------------------------------------------------

/// After text generation, if the cimage has TTS segments and the user
/// requested audio output, run TTS on the generated text.
#[allow(dead_code)]
async fn generate_audio_response(state: &AppState, text: &str) -> Result<Vec<u8>, StatusCode> {
    let tts = state
        .tts_pipeline
        .as_ref()
        .ok_or(StatusCode::NOT_IMPLEMENTED)?;

    // 1. Tokenize text for TTS
    let tokens = state
        .tokenizer
        .encode(text)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    // 2. Run TTS pipeline (max 1024 audio tokens)
    let (samples, sample_rate) = tts
        .generate(&tokens, 1024)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    // 3. Convert PCM to WAV bytes
    let wav_bytes =
        pcm_to_wav(&samples, sample_rate).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(wav_bytes)
}

/// Check whether a cimage file contains TTS segments.
fn cimage_has_tts_segments(path: &std::path::Path) -> bool {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(_) => return false,
    };
    use tribunus_compute_core::compute_image::compile::ternary::verify_cimage;
    let (header, _) = match verify_cimage(&bytes) {
        Ok(h) => h,
        Err(_) => return false,
    };
    // Check for Talker and Code Predictor segments (kind values 30, 33)
    // defined by TtsCimageSegments.
    const TTS_TALKER_WEIGHT: u32 = 30;
    const TTS_CP_WEIGHT: u32 = 33;
    let has_talker = header
        .segments
        .iter()
        .any(|s| s.kind == TTS_TALKER_WEIGHT && s.length > 0);
    let has_cp = header
        .segments
        .iter()
        .any(|s| s.kind == TTS_CP_WEIGHT && s.length > 0);
    has_talker && has_cp
}

// ---- SSE streaming helpers -----------------------------------------------

/// Decode a single token ID to its text representation.
#[allow(dead_code)]
fn decode_token_text(tokenizer: &TribunusTokenizer, token: u32) -> Option<String> {
    tokenizer.decode(&[token]).ok()
}

/// SSE streaming handler for /v1/chat/completions.
///
/// Uses channel-based streaming with CancellationToken for:
/// - Client disconnect cancellation (via CancelOnDropStream)
/// - Stream idle timeout
/// - Graceful shutdown (via StreamTracker)
async fn chat_completions_stream(
    state: Arc<AppState>,
    req: ChatRequest,
    request_id: String,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, ApiError> {
    let start_time = Instant::now();

    // Stage 1: Validate request properties
    if req.max_tokens.unwrap_or(0) > state.config.limits.max_output_tokens {
        return Err(ApiError {
            code: "invalid_request".into(),
            message: format!(
                "max_tokens exceeds limit of {}",
                state.config.limits.max_output_tokens
            ),
            request_id: request_id.clone(),
            retryable: false,
        });
    }

    let max_tokens = req.max_tokens.unwrap_or(256) as usize;

    // Tokenize (no executor lock needed — tokenizer is &self)
    let user_msg = req
        .messages
        .last()
        .ok_or_else(|| ApiError::new("invalid_request", "No messages"))?;
    let prompt_text = match &user_msg.content {
        MessageContent::Text(t) => t.clone(),
        MessageContent::Parts(parts) => parts
            .iter()
            .filter_map(|p| match p {
                ContentPart::Text { text } => Some(text.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(" "),
    };
    let input_ids = state
        .tokenizer
        .encode(&prompt_text)
        .map_err(|_| ApiError::new("internal_error", "Tokenization failed"))?;
    let prompt_len = input_ids.len();

    // Acquire model from registry (wrapped in Arc<Mutex> for concurrent spawn_blocking closures)
    let model = {
        let mut registry = state.model_registry.lock();
        Arc::new(ParkingMutex::new(
            registry
                .instances
                .remove("default")
                .ok_or_else(|| ApiError::new("model_not_found", "No model loaded"))?,
        ))
    };

    let cancel = CancellationToken::new();
    let idle_timeout = Duration::from_secs(state.config.limits.stream_idle_timeout_secs as u64);

    // Register with StreamTracker for graceful shutdown
    let stream_state = Arc::new(StreamState {
        request_id: request_id.clone(),
        cancel: cancel.clone(),
        start_time,
        last_activity: Instant::now(),
        terminal_sent: AtomicBool::new(false),
        receipt: parking_lot::Mutex::new(None),
    });
    {
        let mut tracker = state.stream_tracker.lock();
        tracker.register(request_id.clone(), stream_state.clone());
    }

    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(64);
    let state_shared = state.clone();

    // Spawn decode work: runs until max_tokens, EOS, cancel, or idle timeout
    let cancel_spawn = cancel.clone();
    tokio::spawn(async move {
        let model = model;
        let state = state_shared;
        let cancel = cancel_spawn;
        let tx = tx;
        let stream_state = stream_state;
        let idle_timeout = idle_timeout;
        let request_id = request_id;
        let input_ids = input_ids;
        let prompt_len = prompt_len;
        let max_tokens = max_tokens;
        let start_time = start_time;

        // Prefill in blocking task (brief model lock, released after prefill)
        let prefill_ok = tokio::task::spawn_blocking({
            let model = model.clone();
            move || {
                let mut sess = InferenceSession {
                    session_id: "chat-stream-prefill".into(),
                    scheduler_slot: 0,
                    kv_epoch: 0,
                    sampler: SamplerConfig::default(),
                    mtp_state: None,
                };
                let mut guard = model.lock();
                guard.prefill(&mut sess, &input_ids).is_ok()
            }
        })
        .await
        .unwrap_or(false);

        if !prefill_ok {
            // Terminal receipt for failed prefill
            let receipt = RequestReceipt {
                request_id: request_id.clone(),
                model_digest: state.model_digest.clone(),
                client_id: None,
                terminal_state: "failed".into(),
                prompt_tokens: prompt_len as u32,
                completion_tokens: 0,
                audio_duration_ms: 0,
                queue_time_us: 0,
                execution_time_us: start_time.elapsed().as_micros() as u64,
                error_code: Some("internal_error".into()),
                error_message: Some("prefill failed".into()),
            };
            stream_state.receipt.lock().replace(receipt.clone());
            eprintln!(
                "[prism-server] receipt: {}",
                serde_json::to_string(&receipt).unwrap()
            );
            return;
        }

        let mut step = 0;
        let mut last_activity = Instant::now();

        while step < max_tokens && !cancel.is_cancelled() {
            // Check idle timeout
            if last_activity.elapsed() > idle_timeout {
                let _ = tx
                    .send(Ok(Event::default().data("[TIMED_OUT]").event("timed_out")))
                    .await;
                break;
            }

            let step_result = tokio::task::spawn_blocking({
                let model = model.clone();
                move || -> Option<u32> {
                    let mut sess = InferenceSession {
                        session_id: "chat-stream-decode".into(),
                        scheduler_slot: 0,
                        kv_epoch: 0,
                        sampler: SamplerConfig::default(),
                        mtp_state: None,
                    };
                    let sampling = sess.sampler.clone();
                    model
                        .lock()
                        .decode(&mut sess, &sampling)
                        .ok()
                        .map(|r| r.token_id)
                }
            })
            .await;

            let next = match step_result {
                Ok(Some(id)) => id,
                _ => break,
            };

            last_activity = Instant::now();

            let event = Event::default().data(
                serde_json::json!({
                    "choices": [{
                        "delta": {"content": format!("[{}]", next)},
                        "index": 0
                    }]
                })
                .to_string(),
            );

            if tx.send(Ok(event)).await.is_err() {
                // Client disconnected (rx dropped)
                cancel.cancel();
                break;
            }

            step += 1;

            if next == 1 {
                // EOS token
                break;
            }
        }

        // Emit [DONE] event
        let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;

        // Terminal receipt
        let terminal_state = if cancel.is_cancelled() {
            if last_activity.elapsed() > idle_timeout {
                "timed_out"
            } else {
                "cancelled"
            }
        } else {
            "completed"
        };
        let receipt = RequestReceipt {
            request_id,
            model_digest: state.model_digest.clone(),
            client_id: None,
            terminal_state: terminal_state.into(),
            prompt_tokens: prompt_len as u32,
            completion_tokens: step as u32,
            audio_duration_ms: 0,
            queue_time_us: 0,
            execution_time_us: start_time.elapsed().as_micros() as u64,
            error_code: None,
            error_message: None,
        };
        stream_state.receipt.lock().replace(receipt.clone());
        eprintln!(
            "[prism-server] receipt: {}",
            serde_json::to_string(&receipt).unwrap()
        );
    });

    let stream = ReceiverStream::new(rx);
    let stream = CancelOnDropStream {
        inner: stream,
        token: cancel,
    };

    Ok(Sse::new(stream))
}

/// SSE streaming handler for /v1/chat/completions audio output.
///
/// Emits one `event: audio_chunk` SSE event per WAV segment with base64 payload,
/// then a final `event: audio_done` event.
async fn chat_completions_audio_stream(
    state: Arc<AppState>,
    req: ChatRequest,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, StatusCode> {
    use futures::stream;

    let tts = state
        .tts_pipeline
        .as_ref()
        .ok_or(StatusCode::NOT_IMPLEMENTED)?;

    // Extract text from request (same pattern as chat_completions_stream)
    let user_msg = req.messages.last().ok_or(StatusCode::BAD_REQUEST)?;
    let text = match &user_msg.content {
        MessageContent::Text(t) => t.clone(),
        MessageContent::Parts(parts) => parts
            .iter()
            .filter_map(|p| match p {
                ContentPart::Text { text } => Some(text.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(" "),
    };

    // Tokenize for TTS
    let tokens = state
        .tokenizer
        .encode(&text)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let max_audio_tokens = 256; // ~20 seconds at 12.5 Hz
    let chunk_tokens = 20; // ~1.6 seconds per chunk

    // Generate streaming audio chunks
    let pcm_chunks = tts
        .generate_streaming(&tokens, max_audio_tokens, chunk_tokens)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let stream = stream::unfold(
        (pcm_chunks.into_iter(), 0usize),
        move |(mut iter, idx)| async move {
            match iter.next() {
                Some(pcm) => {
                    let wav = pcm_chunk_to_wav(&pcm, 24000);
                    let b64 = base64::engine::general_purpose::STANDARD.encode(&wav);
                    let event = Event::default().event("audio_chunk").data(
                        serde_json::json!({
                            "chunk": b64,
                            "index": idx,
                            "sample_rate": 24000,
                            "format": "audio/wav",
                        })
                        .to_string(),
                    );
                    Some((Ok(event), (iter, idx + 1)))
                }
                None => {
                    let event = Event::default().event("audio_done").data("[DONE]");
                    Some((Ok(event), (iter, idx)))
                }
            }
        },
    );

    Ok(Sse::new(stream))
}

// ── Ollama-compatible API handlers ──────────────────────────────────────

/// POST /api/generate — Ollama-compatible prompt completion.
///
/// Runs real inference through the executor (same engine as /v1/completions).
/// Streaming emits NDJSON; non-streaming aggregates into one response.
async fn ollama_generate(
    State(state): State<Arc<AppState>>,
    Json(req): Json<OllamaGenerateRequest>,
) -> Result<Response, ApiError> {
    let start_time = Instant::now();

    // Tokenize (no executor lock needed)
    let input_ids = state
        .tokenizer
        .encode(&req.prompt)
        .map_err(|_| ApiError::new("internal_error", "Tokenization failed"))?;

    if input_ids.is_empty() {
        return Err(ApiError::new("invalid_request", "Empty prompt"));
    }

    let prompt_len = input_ids.len();
    let max_tokens = 256usize;

    if req.stream {
        ollama_generate_stream(state, req, input_ids, prompt_len, max_tokens, start_time).await
    } else {
        // Non-streaming: run full inference, aggregate into one response
        // Acquire model from registry
        let mut model = {
            let mut registry = state.model_registry.lock();
            registry
                .instances
                .remove("default")
                .ok_or_else(|| ApiError::new("model_not_found", "No model loaded"))?
        };

        let state_clone = state.clone();
        let generated_text = tokio::task::spawn_blocking(move || {
            let mut sess = InferenceSession {
                session_id: "ollama-generate".into(),
                scheduler_slot: 0,
                kv_epoch: 0,
                sampler: SamplerConfig::default(),
                mtp_state: None,
            };
            let sampling = sess.sampler.clone();

            // Prefill
            model
                .prefill(&mut sess, &input_ids)
                .map_err(|e| format!("prefill failed: {e}"))?;

            // Decode loop — generate text
            let mut generated = String::new();
            for _ in 0..max_tokens {
                let token_id = model
                    .decode(&mut sess, &sampling)
                    .map_err(|_| "decode failed".to_string())?
                    .token_id;

                let token_text = state_clone
                    .tokenizer
                    .decode(&[token_id])
                    .unwrap_or_default();
                generated.push_str(&token_text);

                if token_id == 0 || token_id == 1 || token_text.is_empty() {
                    break;
                }
            }
            Ok::<(String, CimageModelInstance), String>((generated, model))
        })
        .await;

        let (generated_text, model) = match generated_text {
            Ok(Ok((text, m))) => (text, m),
            Ok(Err(e)) => {
                eprintln!("[prism-server] ollama generate error: {e}");
                return Err(ApiError::new("internal_error", e));
            }
            Err(e) => {
                eprintln!("[prism-server] ollama generate panicked: {e}");
                return Err(ApiError::new("internal_error", "Inference task panicked"));
            }
        };

        // Re-register model
        {
            let mut registry = state.model_registry.lock();
            registry.instances.insert("default".to_string(), model);
        }

        let elapsed = start_time.elapsed().as_nanos() as u64;
        let eval_count = generated_text.split_whitespace().count() as u64;

        Ok(Json(OllamaGenerateResponse {
            model: Some(req.model),
            response: Some(generated_text),
            message: None,
            done: true,
            total_duration: Some(elapsed),
            load_duration: Some(0),
            prompt_eval_count: Some(prompt_len as u64),
            prompt_eval_duration: Some(0),
            eval_count: Some(eval_count),
            eval_duration: Some(elapsed),
        })
        .into_response())
    }
}

/// Streaming variant of /api/generate — emits NDJSON tokens.
async fn ollama_generate_stream(
    state: Arc<AppState>,
    req: OllamaGenerateRequest,
    input_ids: Vec<u32>,
    prompt_len: usize,
    max_tokens: usize,
    start_time: Instant,
) -> Result<Response, ApiError> {
    // Acquire model from registry (wrapped in Arc<Mutex> for multiple spawn_blocking closures)
    let model = {
        let mut registry = state.model_registry.lock();
        Arc::new(ParkingMutex::new(
            registry
                .instances
                .remove("default")
                .ok_or_else(|| ApiError::new("model_not_found", "No model loaded"))?,
        ))
    };

    let (tx, rx) = mpsc::channel::<String>(64);
    let state_clone = state.clone();
    let model_name = req.model.clone();
    let cancel = CancellationToken::new();

    tokio::spawn(async move {
        let model = model;
        let _inner_state = state_clone;
        let model_name = model_name;
        let cancel = cancel;
        let tx = tx;
        let input_ids = input_ids;
        let prompt_len = prompt_len;
        let max_tokens = max_tokens;
        let start_time = start_time;

        // Prefill in blocking task
        let prefill_ok = tokio::task::spawn_blocking({
            let model = model.clone();
            move || {
                let mut sess = InferenceSession {
                    session_id: "ollama-stream-prefill".into(),
                    scheduler_slot: 0,
                    kv_epoch: 0,
                    sampler: SamplerConfig::default(),
                    mtp_state: None,
                };
                let mut guard = model.lock();
                guard.prefill(&mut sess, &input_ids).is_ok()
            }
        })
        .await
        .unwrap_or(false);

        if !prefill_ok {
            return;
        }

        // Autoregressive decode, emitting each token as NDJSON
        for _step in 0..max_tokens {
            if cancel.is_cancelled() {
                break;
            }

            let step_result = tokio::task::spawn_blocking({
                let model = model.clone();
                move || -> Option<(u32, Option<String>)> {
                    let mut sess = InferenceSession {
                        session_id: "ollama-stream-decode".into(),
                        scheduler_slot: 0,
                        kv_epoch: 0,
                        sampler: SamplerConfig::default(),
                        mtp_state: None,
                    };
                    let sampling = sess.sampler.clone();
                    let token_id = model.lock().decode(&mut sess, &sampling).ok()?.token_id;
                    // Token text not available inside spawned closure without AppState ref
                    Some((token_id, None))
                }
            })
            .await;

            let (next, text) = match step_result {
                Ok(Some(r)) => r,
                _ => break,
            };

            let chunk = serde_json::json!({
                "model": model_name,
                "response": text.unwrap_or_else(|| format!("[{}]", next)),
                "done": false,
            });
            let line = serde_json::to_string(&chunk).unwrap_or_default();
            if tx.send(line + "\n").await.is_err() {
                cancel.cancel();
                break;
            }

            // EOS token or empty token → stop generation
            if next == 0 || next == 1 {
                break;
            }
        }

        // Emit final done chunk with timing
        let elapsed = start_time.elapsed().as_nanos() as u64;
        let done = serde_json::json!({
            "model": model_name,
            "response": "",
            "done": true,
            "total_duration": elapsed,
            "load_duration": 0u64,
            "prompt_eval_count": prompt_len,
            "prompt_eval_duration": 0u64,
            "eval_count": 0u64,
            "eval_duration": elapsed,
        });
        let _ = tx
            .send(serde_json::to_string(&done).unwrap_or_default() + "\n")
            .await;
    });

    let stream =
        tokio_stream::wrappers::ReceiverStream::new(rx).map(|line| Ok::<_, Infallible>(line));

    Ok(Response::builder()
        .header("content-type", "application/x-ndjson")
        .body(Body::from_stream(stream))
        .unwrap())
}

/// POST /api/chat — Ollama-compatible chat completion.
///
/// Builds a prompt from the message list, runs inference, returns Ollama chat format.
async fn ollama_chat(
    State(state): State<Arc<AppState>>,
    Json(req): Json<OllamaChatRequest>,
) -> Result<Response, ApiError> {
    let start_time = Instant::now();

    // Build a simple prompt from messages
    let prompt = req
        .messages
        .iter()
        .map(|m| format!("{}: {}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("\n");

    let input_ids = state
        .tokenizer
        .encode(&prompt)
        .map_err(|_| ApiError::new("internal_error", "Tokenization failed"))?;

    if input_ids.is_empty() {
        return Err(ApiError::new("invalid_request", "Empty messages"));
    }

    let prompt_len = input_ids.len();
    let max_tokens = 256usize;

    // Acquire model from registry and run inference
    let mut models = state.model_registry.lock();
    let model = models
        .instances
        .get_mut("default")
        .ok_or(ApiError::new("capacity_exhausted", "No model loaded"))?;

    let mut session = InferenceSession {
        session_id: "ollama-chat".into(),
        scheduler_slot: 0,
        kv_epoch: 0,
        sampler: SamplerConfig::default(),
        mtp_state: None,
    };

    // Prefill
    model
        .prefill(&mut session, &input_ids)
        .map_err(|_| ApiError::new("internal_error", "Prefill failed"))?;

    // Decode auto-regressively
    let sampling = session.sampler.clone();
    let mut generated = String::new();

    for _step in 0..max_tokens {
        match model.decode(&mut session, &sampling) {
            Ok(result) => {
                let next = result.token_id;
                let token_text = state.tokenizer.decode(&[next]).unwrap_or_default();
                generated.push_str(&token_text);

                if next == 0 || token_text.is_empty() {
                    break;
                }
            }
            Err(_) => break,
        }
    }

    drop(models);

    let elapsed = start_time.elapsed().as_nanos() as u64;
    let eval_count = generated.split_whitespace().count() as u64;

    let resp = OllamaGenerateResponse {
        model: Some(req.model),
        response: None,
        message: Some(OllamaMessage {
            role: "assistant".to_string(),
            content: generated,
        }),
        done: true,
        total_duration: Some(elapsed),
        load_duration: Some(0),
        prompt_eval_count: Some(prompt_len as u64),
        prompt_eval_duration: Some(0),
        eval_count: Some(eval_count),
        eval_duration: Some(elapsed),
    };

    Ok(Json(resp).into_response())
}

/// GET /api/tags — list locally deployed cimage models.
async fn ollama_tags(State(state): State<Arc<AppState>>) -> Json<OllamaTagsResponse> {
    let mut models = Vec::new();

    // Derive model name from the currently loaded cimage path
    if let Some(name) = state
        .cimage_path
        .file_stem()
        .and_then(|s| s.to_str())
        .map(|s| format!("{}:latest", s))
    {
        let size = std::fs::metadata(&*state.cimage_path)
            .map(|m| m.len())
            .unwrap_or(0);
        let modified_at = std::fs::metadata(&*state.cimage_path)
            .and_then(|m| m.modified())
            .map(|t| {
                let dur = t.duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
                let secs = dur.as_secs();
                // ISO 8601 format
                format!(
                    "{}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
                    secs / 31536000 + 1970,          // approximate year
                    (secs % 31536000) / 2592000 + 1, // approximate month
                    (secs % 2592000) / 86400 + 1,    // approximate day
                    (secs % 86400) / 3600,
                    (secs % 3600) / 60,
                    secs % 60,
                )
            })
            .unwrap_or_else(|_| "1970-01-01T00:00:00Z".to_string());

        models.push(OllamaModelInfo {
            name,
            modified_at,
            size,
        });
    }

    // Also list models from config
    for entry in &state.config.models {
        let name = entry.name.clone();
        let size = std::fs::metadata(&entry.cimage_path)
            .map(|m| m.len())
            .unwrap_or(0);
        let modified_at = std::fs::metadata(&entry.cimage_path)
            .and_then(|m| m.modified())
            .map(|t| {
                let dur = t.duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
                let secs = dur.as_secs();
                format!(
                    "{}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
                    secs / 31536000 + 1970,
                    (secs % 31536000) / 2592000 + 1,
                    (secs % 2592000) / 86400 + 1,
                    (secs % 86400) / 3600,
                    (secs % 3600) / 60,
                    secs % 60,
                )
            })
            .unwrap_or_else(|_| "1970-01-01T00:00:00Z".to_string());

        models.push(OllamaModelInfo {
            name,
            modified_at,
            size,
        });
    }

    Json(OllamaTagsResponse { models })
}

/// POST /api/show — return model details.
async fn ollama_show(
    State(state): State<Arc<AppState>>,
    Json(req): Json<OllamaShowRequest>,
) -> Json<OllamaShowResponse> {
    let size = std::fs::metadata(&*state.cimage_path)
        .map(|m| m.len())
        .unwrap_or(0);
    let modified_at = std::fs::metadata(&*state.cimage_path)
        .and_then(|m| m.modified())
        .map(|t| {
            let dur = t.duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
            let secs = dur.as_secs();
            format!(
                "{}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
                secs / 31536000 + 1970,
                (secs % 31536000) / 2592000 + 1,
                (secs % 2592000) / 86400 + 1,
                (secs % 86400) / 3600,
                (secs % 3600) / 60,
                secs % 60,
            )
        })
        .unwrap_or_else(|_| "1970-01-01T00:00:00Z".to_string());

    Json(OllamaShowResponse {
        model: req.model,
        modified_at,
        size,
        digest: state
            .model_digest
            .clone()
            .unwrap_or_else(|| "unknown".to_string()),
        details: serde_json::json!({
            "format": "cimage",
            "family": "prism",
            "parameter_size": "unknown",
            "quantization_level": "unknown"
        }),
    })
}

/// GET /api/ps — report loaded cimage instances and residency.
async fn ollama_ps(State(state): State<Arc<AppState>>) -> Json<OllamaPsResponse> {
    let name = state
        .cimage_path
        .file_stem()
        .and_then(|s| s.to_str())
        .map(|s| format!("{}:latest", s))
        .unwrap_or_else(|| "prism-model:latest".to_string());
    let size = std::fs::metadata(&*state.cimage_path)
        .map(|m| m.len())
        .unwrap_or(0);

    // Estimate residency: extract size is ~10x the cimage for loaded weights
    let estimated_memory = size * 10;

    Json(OllamaPsResponse {
        models: vec![OllamaPsModelInfo {
            name,
            size,
            processor: "metal".to_string(),
            memory: estimated_memory,
            until: "now".to_string(),
        }],
    })
}

// ── Auth middleware ──────────────────────────────────────────────────

async fn auth_middleware(
    State(state): State<Arc<AppState>>,
    req: Request,
    next: Next,
) -> impl IntoResponse {
    if !state.config.auth.enabled {
        return next.run(req).await;
    }

    let path = req.uri().path();

    let required_scope = if path.starts_with("/v1/models") {
        "model.read"
    } else if path.starts_with("/v1/chat/completions") || path.starts_with("/v1/completions") {
        "generation.text"
    } else if path.contains("admin") || path.contains("diagnostics") {
        "admin.manage"
    } else {
        return next.run(req).await;
    };

    let auth_header = req
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));
    let token = match auth_header {
        Some(t) => t,
        None => {
            return ApiError::new("unauthorized", "Missing or invalid authentication token")
                .into_response();
        }
    };

    if !state.auth_verifier.verify(token, required_scope) {
        return ApiError::new("unauthorized", "Token lacks required scope").into_response();
    }

    next.run(req).await
}

// ── Request-size middleware ───────────────────────────────────────────

async fn size_limit_middleware(
    State(state): State<Arc<AppState>>,
    req: Request,
    next: Next,
) -> impl IntoResponse {
    let limit = state.config.limits.max_body_bytes;

    if let Some(cl) = req.headers().get("content-length") {
        if let Ok(len) = cl.to_str().unwrap_or("0").parse::<u64>() {
            if len > limit {
                return ApiError::new(
                    "invalid_request",
                    format!("Request body exceeds {} byte limit", limit),
                )
                .into_response();
            }
        }
    }

    next.run(req).await
}

// ── Config validation ─────────────────────────────────────────────────

fn validate_config(config: &ServerConfig) -> Result<(), String> {
    match config.server.mode.as_str() {
        "network" => {
            if !config.auth.enabled {
                return Err("Network mode requires authentication".into());
            }
            if config.server.tls.is_none() && config.server.host != "127.0.0.1" {
                return Err("Network mode on non-loopback requires TLS".into());
            }
        }
        "local_api" => {
            if !config.auth.enabled {
                return Err("Local API mode requires authentication".into());
            }
        }
        "embedded" => {} // loopback is the auth boundary
        _ => return Err(format!("Unknown mode: {}", config.server.mode)),
    }
    Ok(())
}

// ── Main

/// Detect cimage format by reading the magic bytes.
fn detect_cimage_format(path: &std::path::Path) -> Result<String, String> {
    let mut buf = [0u8; 8];
    use std::io::Read;
    let mut f = std::fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    f.read_exact(&mut buf)
        .map_err(|e| format!("read {}: {e}", path.display()))?;
    if buf == CIMAGE_MAGIC {
        Ok("new".into())
    } else {
        Ok("old".into())
    }
}

#[tokio::main]
async fn main() -> Result<(), String> {
    let args = Args::parse();

    // Load config: if --config given, parse TOML; otherwise use defaults.
    let config: ServerConfig = match &args.config {
        Some(path) => {
            let text = std::fs::read_to_string(path)
                .map_err(|e| format!("Failed to read config {}: {e}", path.display()))?;
            toml::from_str(&text).map_err(|e| format!("Config parse error: {e}"))?
        }
        None => ServerConfig {
            server: ServerSection {
                mode: default_mode(),
                host: default_host(),
                port: default_port(),
                unix_socket: None,
                tls: None,
            },
            auth: AuthSection {
                enabled: false,
                tokens: vec![],
                admin_token: None,
            },
            limits: LimitSection {
                max_body_bytes: default_body_bytes(),
                max_prompt_tokens: default_prompt_tokens(),
                max_output_tokens: default_output_tokens(),
                max_images: default_images(),
                max_audio_seconds: default_audio_seconds(),
                stream_idle_timeout_secs: default_stream_timeout(),
                total_request_deadline_secs: default_deadline(),
            },
            models: vec![],
        },
    };

    validate_config(&config)?;

    let auth_verifier = AuthVerifier::new(&config.auth);

    println!(
        "[prism-server] Loading cimage from {}...",
        args.cimage.display()
    );

    let effective_format = if args.cimage_format == "auto" {
        detect_cimage_format(&args.cimage)?
    } else {
        args.cimage_format.clone()
    };

    println!(
        "[prism-server] Loading tokenizer from {}...",
        args.model_dir.display()
    );
    let tokenizer = TribunusTokenizer::from_dir(&args.model_dir)?;

    let model_registry = ParkingMutex::new(ModelRegistry::new());
    println!("[prism-server] Model registry initialized");
    // Load cimage model into registry
    if let Some(device) = metal::Device::system_default() {
        match CimageDeployment::load(&args.cimage, &device) {
            Ok(_deployment) => {
                let stem = args
                    .cimage
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown")
                    .to_string();

                let profile = ServingProfile {
                    model_name: stem.clone(),
                    model_tag: "latest".into(),
                    architecture: stem.clone(),
                    context_length: 8192,
                    precision: "compiled".into(),
                    mtp_enabled: false,
                };

                let generation = CimageGeneration {
                    generation_id: GenerationId(format!("deploy.{}", stem)),
                    parent_generation: None,
                    base_model: ModelSourceId(args.cimage.to_string_lossy().to_string()),
                    compiler_identity: CompilerIdentity {
                        name: "prism-server".into(),
                        version: env!("CARGO_PKG_VERSION").into(),
                        build_hash: None,
                        build_timestamp: None,
                    },
                    hardware_profile: HardwareProfileId("auto".into()),
                    tensor_bindings: std::collections::BTreeMap::new(),
                    kernel_bindings: std::collections::BTreeMap::new(),
                    engram_bindings: std::collections::BTreeMap::new(),
                    execution_graph: ExecutionGraph {
                        regions: vec![],
                        edges: vec![],
                        state: RuntimeStatePlan {
                            max_context_tokens: 8192,
                            kv_cache_bytes_per_token: 0,
                            total_kv_cache_bytes: 0,
                        },
                        memory: MemoryPlan {
                            total_activation_bytes: 0,
                            total_weight_bytes: 0,
                            arena_region_count: 0,
                        },
                    },
                    receipt_root: ReceiptId("startup".into()),
                    created_at: Timestamp("startup".into()),
                };

                let context = CimageRuntimeContext {
                    generation,
                    tensor_store: RuntimeTensorStore::new(),
                    payloads: std::collections::BTreeMap::new(),
                    kernel_artifacts: std::collections::BTreeMap::new(),
                };

                let instance =
                    CimageModelInstance::new(format!("deploy.{}", stem), context, profile);

                let mut registry = model_registry.lock();
                registry.instances.insert("default".to_string(), instance);
                drop(registry);
                println!("[prism-server] Loaded model into registry");
            }
            Err(e) => {
                eprintln!("[prism-server] Failed to load cimage: {e}");
                eprintln!("[prism-server] Server will start but model registry is empty");
            }
        }
    } else {
        eprintln!("[prism-server] No Metal device — skipping cimage load");
    }
    // Load TTS pipeline if the cimage contains TTS segments
    let tts_pipeline = if cimage_has_tts_segments(&args.cimage) {
        println!("[prism-server] Loading TTS pipeline...");
        let device = metal::Device::system_default()
            .ok_or_else(|| "no Metal device available".to_string())?;
        let deployment = CimageDeployment::load(&args.cimage, &device)
            .map_err(|e| format!("cimage load for TTS failed: {e}"))?;
        match TtsPipeline::from_cimage(&deployment, &device) {
            Ok(pipeline) => {
                println!("[prism-server] TTS pipeline loaded successfully.");
                Some(pipeline)
            }
            Err(e) => {
                eprintln!("[prism-server] TTS pipeline load error: {e}");
                None
            }
        }
    } else {
        eprintln!("[prism-server] No TTS segments found in cimage.");
        None
    };

    let broker = Arc::new(MemoryAllocationBroker::new());
    let state = Arc::new(AppState {
        model_registry,
        tokenizer,
        decode_count: std::sync::atomic::AtomicU64::new(0),
        cimage_path: args.cimage.clone(),
        config,
        auth_verifier,
        tts_pipeline,
        cimage_format: effective_format.clone(),
        bitnet_runtime: if effective_format == "new" {
            match BitNetRuntime::from_cimage(&args.cimage) {
                Ok(rt) => Some(ParkingMutex::new(rt)),
                Err(e) => {
                    eprintln!("[prism-server] BitNet runtime init failed: {e}");
                    None
                }
            }
        } else {
            None
        },
        model_digest: Some(
            blake3::hash(&std::fs::read(&args.cimage).unwrap_or_default()).to_hex()[..16]
                .to_string(),
        ),
        stream_tracker: ParkingMutex::new(StreamTracker::new()),
        memory_broker: broker.clone(),
        distill_engine: Arc::new(DistillationEngine::new(broker)),
        cancel: CancelToken::new(None),
    });

    let host = state.config.server.host.clone();
    let addr = format!("{}:{}", host, args.port);

    let drain_state = state.clone();
    #[allow(unused_mut)]
    let mut app = Router::new()
        .route("/v1/models", get(list_models))
        .route("/ready", get(readiness))
        .route("/v1/runtime/receipts/{request_id}", get(diagnostic_receipt))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/distill", post(post_distill))
        .route("/v1/distill/{job_id}", get(get_distill_status))
        .route("/v1/models/load", post(load_model))
        .route("/v1/models/deploy", post(deploy_model))
        .route("/v1/adapters/merge", post(merge_adapter))
        .route("/v1/sessions/{session_id}/rollback", post(session_rollback))
        .route("/v1/kv/epoch/{epoch}/rollback", post(kv_epoch_rollback))
        .route(
            "/v1/generations/{gen_id}/rollback",
            post(generation_rollback),
        )
        // ── Ollama-compatible endpoints ───────────────────────────
        .route("/api/generate", post(ollama_generate))
        .route("/api/chat", post(ollama_chat))
        .route("/api/tags", get(ollama_tags))
        .route("/api/show", post(ollama_show))
        .route("/api/ps", get(ollama_ps))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            size_limit_middleware,
        ))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ))
        .with_state(state);

    #[allow(unexpected_cfgs)]
    #[allow(unexpected_cfgs)]
    #[cfg(feature = "server-dashboard")]
    {
        app = app
            .route("/v1/openapi.json", get(dashboard_stubs::openapi_schema))
            .route("/v1/cimages", get(dashboard_stubs::list_cimages))
            .route(
                "/v1/cimages/{digest}/tensors",
                get(dashboard_stubs::get_cimage_tensors),
            )
            .route("/v1/cimages/{digest}", get(dashboard_stubs::get_cimage))
            .route("/dashboard", get(dashboard_stubs::dashboard_spa))
            .route("/", get(dashboard_stubs::dashboard_root));
    }

    println!("[prism-server] Listening on http://{}", addr);
    println!("  [prism-server] Ready (press Ctrl+C to stop)");

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(|e| format!("bind: {e}"))?;

    // Graceful shutdown: Ctrl+C triggers drain
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        println!("  ▶ Shutdown requested: draining in-flight requests...");
        let _ = shutdown_tx.send(());
    });

    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            shutdown_rx.await.ok();
            // Cancel all active streams (releases slots + KV) — scoped so MutexGuard drops before await
            let count = {
                let mut tracker = drain_state.stream_tracker.lock();
                let c = tracker.active_count();
                tracker.cancel_all();
                c
            };
            if count > 0 {
                println!("  ▶ Cancelled {} active stream(s)", count);
            }
            // Brief drain deadline: wait for work to settle
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            println!("  ▶ Drain complete, exiting.");
        })
        .await
        .map_err(|e| format!("serve: {e}"))?;

    Ok(())
}
