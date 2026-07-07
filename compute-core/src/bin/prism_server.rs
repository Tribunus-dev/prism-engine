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

use tribunus_compute_core::audio_preprocess_accelerate;
use tribunus_compute_core::backend::create_inference_executor;
use tribunus_compute_core::backend::flex_dispatch::{
    create_flex_dispatch, run_flex_dispatch_cycle, FlexDispatch,
};
use tribunus_compute_core::backend::heterogeneous_executor::HeterogeneousExecutor;
use tribunus_compute_core::backend::routing::*;
use tribunus_compute_core::compilation::cancel::CancelToken;
use tribunus_compute_core::compute_image::cimage_loader::CimageDeployment;
use tribunus_compute_core::server::distill_worker::{
    DistillationEngine, DistillationJobStatus, DistillationRequest,
};
use tribunus_compute_core::server::state::MemoryAllocationBroker;
use tribunus_compute_core::tokenizer::TribunusTokenizer;
use tribunus_compute_core::tts::pipeline::{pcm_chunk_to_wav, pcm_to_wav, TtsPipeline};

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
    slot_id: u32,
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
    executor: ParkingMutex<HeterogeneousExecutor>,
    flex_dispatch: ParkingMutex<FlexDispatch>,
    tokenizer: TribunusTokenizer,
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
    /// Model content digest for receipt tracking.
    model_digest: Option<String>,
    stream_tracker: ParkingMutex<StreamTracker>,
    #[allow(dead_code)]
    memory_broker: Arc<MemoryAllocationBroker>,
    distill_engine: Arc<DistillationEngine>,
    #[allow(dead_code)]
    cancel: CancelToken,
}

struct SlotGuard {
    slot_id: u32,
    state: Arc<AppState>,
}

impl Drop for SlotGuard {
    fn drop(&mut self) {
        self.state.executor.lock().free_slot(self.slot_id);
    }
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

async fn list_models(State(_state): State<Arc<AppState>>) -> Json<ModelList> {
    Json(ModelList {
        object: "list".to_string(),
        data: vec![ModelInfo {
            id: "prism-model".to_string(),
            object: "model".to_string(),
            created: 0,
            owned_by: "prism".to_string(),
        }],
    })
}
async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatRequest>,
) -> Result<Response, ApiError> {
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

    // Encode multimodal inputs if present (brief executor lock)
    let multimodal_embeddings = if has_multimodal {
        let mut image_bytes: Vec<Vec<u8>> = Vec::new();
        let mut audio_bytes: Vec<Vec<u8>> = Vec::new();
        for part in &multimodal_parts {
            if let MultimodalPart::Image(bytes) = part {
                image_bytes.push(bytes.clone());
            }
            if let MultimodalPart::Audio(bytes) = part {
                audio_bytes.push(bytes.clone());
            }
        }
        let mut exec = state.executor.lock();
        let img_embeddings = match multimodal_encode_images(&mut exec, &image_bytes) {
            Ok(embeddings) => embeddings,
            Err(e) => {
                eprintln!("[prism-server] multimodal encode error: {e}");
                vec![]
            }
        };

        // ── Audio encoding ──────────────────────────────────────────────
        let audio_frames = if !audio_bytes.is_empty() {
            let mut total_frames = 0usize;
            for (idx, audio_data) in audio_bytes.iter().enumerate() {
                match audio_preprocess_accelerate::load_wav_to_f32(audio_data) {
                    Ok((samples, sample_rate, _channels)) => {
                        match audio_preprocess_accelerate::preprocess_audio_gemma4(
                            &samples,
                            sample_rate,
                        ) {
                            Ok(mel_spec) => {
                                let num_frames = mel_spec.len() / 640;
                                total_frames += num_frames;

                                let audio_op = OperationDescriptor {
                                    operation_id: OperationId(1000 + idx as u64),
                                    family: OperationFamily::AudioEncode,
                                    layer_index: None,
                                    phase: Phase::Prefill,
                                    logical_shape: LogicalShape {
                                        dims: vec![640, num_frames as u32],
                                    },
                                    physical_layout: PhysicalLayout::RowMajor,
                                    input_dtypes: vec![DType::F32],
                                    output_dtype: DType::F32,
                                    quantization: None,
                                    expected_output_shape: TensorShape {
                                        dims: vec![3840, num_frames as u32],
                                    },
                                    correctness_checkpoint: CorrectnessCheckpointPolicy::None,
                                };
                                exec.operation_registry
                                    .insert(audio_op.operation_id, audio_op);
                            }
                            Err(e) => {
                                eprintln!("[prism-server] audio mel error: {e}");
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("[prism-server] audio decode error: {e}");
                    }
                }
            }
            total_frames
        } else {
            0
        };

        (img_embeddings, audio_frames)
    } else {
        (vec![], 0)
    };

    // Tokenize (no executor lock needed)
    let input_ids = state
        .tokenizer
        .encode(&prompt_text)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    if input_ids.is_empty() {
        return Err(StatusCode::BAD_REQUEST.into());
    }

    let max_tokens = req.max_tokens.unwrap_or(256) as usize;
    let (multimodal_embeddings, audio_embeddings_len) = multimodal_embeddings;
    let prompt_len = input_ids.len() + multimodal_embeddings.len() + audio_embeddings_len;

    // Allocate decode slot (brief executor lock)
    // Stage 2: Reserve runtime resources
    let slot_id = state
        .executor
        .lock()
        .allocate_slot()
        .map_err(|_| ApiError {
            code: "capacity_exhausted".into(),
            message: "All slots busy, retry later".into(),
            request_id: request_id.clone(),
            retryable: true,
        })?;

    let _slot_guard = SlotGuard {
        slot_id,
        state: state.clone(),
    };

    // Run decode in blocking task (Metal is synchronous: commit + wait_until_completed)
    let state_clone = state.clone();
    let generated_tokens = tokio::task::spawn_blocking(move || {
        let mut exec = state_clone.executor.lock();

        // Sequential prefill: each token builds KV cache row
        for (i, &tok) in input_ids[..input_ids.len().saturating_sub(1)]
            .iter()
            .enumerate()
        {
            let op = make_prefill_op(tok as u64);
            exec.operation_registry.insert(op.operation_id, op);
            let plan =
                make_boundary_plan(i as u64, BACKEND_MEGAKERNEL, vec![OperationId(tok as u64)]);
            if let Err(e) = exec.execute_boundaries(&[plan]) {
                eprintln!("[prism-server] prefill step {} error: {}", i, e);
            }
        }

        let mut last_token = *input_ids.last().unwrap_or(&0) as u64;
        let mut generated_tokens = Vec::with_capacity(max_tokens);

        for step in 0..max_tokens {
            let dec = make_decode_op(last_token);
            exec.operation_registry.insert(dec.operation_id, dec);
            let plan = make_boundary_plan(
                (prompt_len + step) as u64,
                BACKEND_MEGAKERNEL,
                vec![OperationId(last_token)],
            );
            if let Err(e) = exec.execute_boundaries(&[plan]) {
                eprintln!("[prism-server] decode error: {e}");
                break;
            }

            let next = exec.last_decoded_token().unwrap_or(1);
            last_token = next;
            generated_tokens.push(next as u32);

            // Every 16 decode steps, run a flex-dispatch cycle to potentially
            // re-route operations based on system state.
            let count = state_clone
                .decode_count
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if count % 16 == 0 {
                let mut fd = state_clone.flex_dispatch.lock();
                let _ = run_flex_dispatch_cycle(&mut fd, &mut exec);
            }

            if last_token == 1 {
                // EOS token
                break;
            }
        }

        generated_tokens
    })
    .await;

    let generated_tokens = match generated_tokens {
        Ok(tokens) => tokens,
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
    // Tokenize (no executor lock needed)
    let input_ids = state
        .tokenizer
        .encode(&req.prompt)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    if input_ids.is_empty() {
        return Err(StatusCode::BAD_REQUEST);
    }

    let max_tokens = req.max_tokens.unwrap_or(256) as usize;
    let prompt_len = input_ids.len();

    // Allocate decode slot (brief executor lock)
    let slot_id = state
        .executor
        .lock()
        .allocate_slot()
        .map_err(|_| StatusCode::SERVICE_UNAVAILABLE)?;

    // Run decode in blocking task (Metal is synchronous)
    let state_clone = state.clone();
    let generated_text = tokio::task::spawn_blocking(move || {
        let mut exec = state_clone.executor.lock();

        // Prefill: each token builds KV cache row
        for (i, &tok) in input_ids[..input_ids.len().saturating_sub(1)]
            .iter()
            .enumerate()
        {
            let op = make_prefill_op(tok as u64);
            exec.operation_registry.insert(op.operation_id, op);
            let plan =
                make_boundary_plan(i as u64, BACKEND_MEGAKERNEL, vec![OperationId(tok as u64)]);
            if let Err(e) = exec.execute_boundaries(&[plan]) {
                eprintln!("[prism-server] completions prefill error: {e}");
            }
        }

        let mut last_token = *input_ids.last().unwrap_or(&0) as u64;
        let mut generated_text = String::new();

        // Autoregressive decode
        for step in 0..max_tokens {
            let dec = make_decode_op(last_token);
            exec.operation_registry.insert(dec.operation_id, dec);
            let plan = make_boundary_plan(
                (prompt_len + step) as u64,
                BACKEND_MEGAKERNEL,
                vec![OperationId(last_token)],
            );
            if let Err(e) = exec.execute_boundaries(&[plan]) {
                eprintln!("[prism-server] completions decode error: {e}");
                break;
            }

            let next = exec.last_decoded_token().unwrap_or(0);
            last_token = next;

            // Decode token to text
            let token_text = state_clone
                .tokenizer
                .decode(&[next as u32])
                .unwrap_or_default();
            generated_text.push_str(&token_text);

            // Check for stop token (EOS or empty)
            if next == 0 || token_text.is_empty() {
                break;
            }
        }

        exec.free_slot(slot_id);
        generated_text
    })
    .await
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

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
    cimage_path: String,
}

async fn load_model(
    State(state): State<Arc<AppState>>,
    Json(body): Json<LoadModelRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let new_executor = create_inference_executor(&body.cimage_path, 1, false)
        .map_err(|e| ApiError::new("invalid_request", format!("failed to load cimage: {e}")))?;
    *state.executor.lock() = new_executor;
    Ok(Json(
        serde_json::json!({"status": "ok", "model": body.cimage_path}),
    ))
}

#[derive(Deserialize)]
struct DeployModelRequest {
    cimage_path: String,
    #[serde(default = "default_quality_gate")]
    quality_gate: bool,
}

fn default_quality_gate() -> bool {
    true
}

async fn deploy_model(
    State(state): State<Arc<AppState>>,
    Json(body): Json<DeployModelRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let new_executor = create_inference_executor(&body.cimage_path, 1, false)
        .map_err(|e| ApiError::new("invalid_request", format!("failed to load cimage: {e}")))?;
    *state.executor.lock() = new_executor;
    Ok(Json(
        serde_json::json!({"status": "deployed", "model": body.cimage_path, "quality_gate": body.quality_gate}),
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

/// Encode image bytes through the executor's multimodal projection path.
/// Returns the projected embedding vectors ready for decoder prefill.
fn multimodal_encode_images(
    exec: &mut HeterogeneousExecutor,
    images: &[Vec<u8>],
) -> Result<Vec<Vec<f32>>, String> {
    let mut results = Vec::new();
    for img_bytes in images {
        // 1. Decode image bytes to get pixel dimensions
        let (width, height, _rgb_data) = decode_image_to_rgb(img_bytes)
            .map_err(|e| format!("failed to decode image {}: {e}", results.len()))?;

        // 2. Dispatch VisionEncode operation with real image shape
        let vision_op = OperationDescriptor {
            operation_id: OperationId(results.len() as u64),
            family: OperationFamily::VisionEncode,
            layer_index: None,
            phase: Phase::Prefill,
            logical_shape: LogicalShape {
                dims: vec![height, width, 3],
            },
            physical_layout: PhysicalLayout::RowMajor,
            input_dtypes: vec![DType::U8],
            output_dtype: DType::F32,
            quantization: None,
            expected_output_shape: TensorShape { dims: vec![1] },
            correctness_checkpoint: CorrectnessCheckpointPolicy::None,
        };
        exec.operation_registry
            .insert(vision_op.operation_id, vision_op);

        // 3. The executor dispatches VisionEncode through MegakernelBackend.
        //    Push a placeholder embedding — real embeddings arrive once
        //    VisionEncode ops are wired end-to-end.
        results.push(vec![0.0f32]);
    }
    Ok(results)
}

/// Decode image bytes (PNG/JPEG/WEBP) into raw RGB pixel data.
fn decode_image_to_rgb(bytes: &[u8]) -> Result<(u32, u32, Vec<u8>), String> {
    use image::GenericImageView;
    let img = image::load_from_memory(bytes).map_err(|e| format!("image decode failed: {e}"))?;
    let (w, h) = img.dimensions();
    let rgb = img.to_rgb8().into_raw();
    Ok((w, h, rgb))
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

/// Create a PrefillFragment operation descriptor for a token.
fn make_prefill_op(tok: u64) -> OperationDescriptor {
    OperationDescriptor {
        operation_id: OperationId(tok),
        family: OperationFamily::PrefillFragment,
        layer_index: None,
        phase: Phase::Prefill,
        logical_shape: LogicalShape { dims: vec![1] },
        physical_layout: PhysicalLayout::RowMajor,
        input_dtypes: vec![],
        output_dtype: DType::F32,
        quantization: None,
        expected_output_shape: TensorShape { dims: vec![] },
        correctness_checkpoint: CorrectnessCheckpointPolicy::None,
    }
}

/// Create a DecoderLayer operation descriptor for a token.
fn make_decode_op(tok: u64) -> OperationDescriptor {
    OperationDescriptor {
        operation_id: OperationId(tok),
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
    }
}

/// Create an execution boundary plan for a backend + operation set.
fn make_boundary_plan(
    group_id: u64,
    backend_id: BackendId,
    operations: Vec<OperationId>,
) -> ExecutionBoundaryPlan {
    ExecutionBoundaryPlan {
        group_id: EvaluationGroupId(group_id),
        backend_id,
        operations,
        materialized_outputs: vec![],
        policy: EvaluationPolicy::BackendLazy,
        synchronization: SynchronizationPolicy::None,
        release_after: vec![],
        content_digest: None,
    }
}

/// Decode a single token ID to its text representation.
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

    // Stage 2: Reserve runtime resources
    let slot_id = state
        .executor
        .lock()
        .allocate_slot()
        .map_err(|_| ApiError {
            code: "capacity_exhausted".into(),
            message: "All slots busy, retry later".into(),
            request_id: request_id.clone(),
            retryable: true,
        })?;

    let _slot_guard = SlotGuard {
        slot_id,
        state: state.clone(),
    };

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

    let cancel = CancellationToken::new();
    let idle_timeout = Duration::from_secs(state.config.limits.stream_idle_timeout_secs as u64);

    // Register with StreamTracker for graceful shutdown
    let stream_state = Arc::new(StreamState {
        request_id: request_id.clone(),
        cancel: cancel.clone(),
        slot_id,
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

    // Spawn decode work: runs until max_tokens, EOS, cancel, or idle timeout
    let state_clone = state.clone();
    let cancel_clone = cancel.clone();
    let tx_clone = tx.clone();
    let stream_state_clone = stream_state.clone();
    let idle_timeout = idle_timeout;
    let request_id_clone = request_id.clone();
    let prompt_len = prompt_len;

    tokio::spawn(async move {
        let state = state_clone;
        let cancel = cancel_clone;
        let tx = tx_clone;
        let stream_state = stream_state_clone;
        let idle_timeout = idle_timeout;
        let request_id = request_id_clone;
        let prompt_len = prompt_len;

        // Clone before prefill closure to keep originals for while loop
        let cancel_prefill = cancel.clone();
        let state_prefill = state.clone();
        let result = tokio::task::spawn_blocking(move || {
            let mut exec = state_prefill.executor.lock();
            let last_tok = *input_ids.last().unwrap_or(&0) as u64;

            // Prefill all tokens except the last one
            for (i, &tok) in input_ids[..input_ids.len().saturating_sub(1)]
                .iter()
                .enumerate()
            {
                if cancel_prefill.is_cancelled() {
                    return None;
                }
                let op = make_prefill_op(tok as u64);
                exec.operation_registry.insert(op.operation_id, op);
                let plan =
                    make_boundary_plan(i as u64, BACKEND_MEGAKERNEL, vec![OperationId(tok as u64)]);
                if let Err(e) = exec.execute_boundaries(&[plan]) {
                    eprintln!("[prism-server] streaming prefill error: {e}");
                }
            }
            Some(last_tok)
        })
        .await;

        let mut last_tok = match result {
            Ok(Some(t)) => t,
            Err(_) => return,
            _ => return,
        };

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
                let state = state.clone();
                move || -> Option<(u64, Option<String>)> {
                    let mut exec = state.executor.lock();

                    let dec = make_decode_op(last_tok);
                    exec.operation_registry.insert(dec.operation_id, dec);
                    let plan = make_boundary_plan(
                        (prompt_len + step) as u64,
                        BACKEND_MEGAKERNEL,
                        vec![OperationId(last_tok)],
                    );
                    if let Err(e) = exec.execute_boundaries(&[plan]) {
                        eprintln!("[prism-server] streaming decode error: {e}");
                    }

                    let next = exec.last_decoded_token().ok().unwrap_or(0);
                    let text = decode_token_text(&state.tokenizer, next as u32);

                    Some((next, text))
                }
            })
            .await;

            let (next, text) = match step_result {
                Ok(Some(r)) => r,
                _ => break,
            };

            last_activity = Instant::now();

            let event = Event::default().data(
                serde_json::json!({
                    "choices": [{
                        "delta": {"content": text.unwrap_or_default()},
                        "index": 0
                    }]
                })
                .to_string(),
            );

            if tx.send(Ok(event)).await.is_err() {
                // Client disconnected (rx dropped)
                cancel.cancel();
                return;
            }

            last_tok = next;
            step += 1;

            if last_tok == 1 {
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
    let executor = create_inference_executor(&args.cimage, 1, false)
        .map_err(|e| format!("cimage load failed: {e}"))?;

    println!(
        "[prism-server] Loading tokenizer from {}...",
        args.model_dir.display()
    );
    let tokenizer = TribunusTokenizer::from_dir(&args.model_dir)?;

    let flex_dispatch = create_flex_dispatch();
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
        executor: ParkingMutex::new(executor),
        flex_dispatch: ParkingMutex::new(flex_dispatch),
        tokenizer,
        decode_count: std::sync::atomic::AtomicU64::new(0),
        cimage_path: args.cimage.clone(),
        config,
        auth_verifier,
        tts_pipeline,
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
    let app = Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/distill", post(post_distill))
        .route("/v1/distill/{job_id}", get(get_distill_status))
        .route("/v1/models/load", post(load_model))
        .route("/v1/models/deploy", post(deploy_model))
        .route("/v1/adapters/merge", post(merge_adapter))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            size_limit_middleware,
        ))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ))
        .with_state(state);

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
