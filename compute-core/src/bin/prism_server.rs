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

use std::path::PathBuf;
use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    response::sse::{Event, Sse},
    response::{IntoResponse, Json, Response},
    routing::{get, post},
    Router,
};
use clap::Parser;
use parking_lot::Mutex as ParkingMutex;
use serde::{Deserialize, Serialize};
use base64::Engine;
use futures::stream::Stream;
use std::convert::Infallible;
use uuid::Uuid;
use metal;

use tribunus_compute_core::backend::create_inference_executor;
use tribunus_compute_core::tts::pipeline::{pcm_chunk_to_wav, pcm_to_wav, TtsPipeline};
use tribunus_compute_core::compute_image::cimage_loader::CimageDeployment;
use tribunus_compute_core::backend::flex_dispatch::{create_flex_dispatch, FlexDispatch, run_flex_dispatch_cycle};
use tribunus_compute_core::backend::heterogeneous_executor::HeterogeneousExecutor;
use tribunus_compute_core::backend::routing::*;
use tribunus_compute_core::tokenizer::TribunusTokenizer;
use tribunus_compute_core::audio_preprocess_accelerate;

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

    /// Server port.
    #[arg(long, default_value = "8080")]
    port: u16,
}

// ── State ───────────────────────────────────────────────────────────────

struct AppState {
    executor: ParkingMutex<HeterogeneousExecutor>,
    flex_dispatch: ParkingMutex<FlexDispatch>,
    tokenizer: TribunusTokenizer,
    decode_count: std::sync::atomic::AtomicU64,
    #[allow(dead_code)]
    cimage_path: PathBuf,
    /// Optional TTS pipeline loaded when the cimage contains TTS segments.
    #[allow(dead_code)]
    tts_pipeline: Option<TtsPipeline>,
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
            let parts: Vec<ContentPart> =
                serde_json::from_value(serde_json::Value::Array(arr))
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
                    ContentPart::Text { text } => {
                        parts.push(MultimodalPart::Text(text.clone()))
                    }
                    ContentPart::ImageUrl { image_url } => {
                        if let Ok(bytes) = decode_data_url(&image_url.url) {
                            parts.push(MultimodalPart::Image(bytes));
                        }
                    }
                    ContentPart::InputAudio { input_audio } => {
                        if let Ok(bytes) =
                            base64::engine::general_purpose::STANDARD
                                .decode(&input_audio.data)
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
) -> Result<Response, StatusCode> {
    // Audio output via SSE streaming when response_format contains "audio"
    if req.response_format.as_ref().map_or(false, |f| f.contains("audio")) {

        let sse = chat_completions_audio_stream(state, req).await?;
        return Ok(sse.into_response());
    }

    if req.stream.unwrap_or(false) {
        let stream_response = chat_completions_stream(state, req).await;
        return Ok(stream_response.into_response());
    }

    // Extract prompt from last user message
    let user_msg = req.messages.last()
        .ok_or(StatusCode::BAD_REQUEST)?;
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
    let has_multimodal = multimodal_parts.iter().any(|p| {
        matches!(p, MultimodalPart::Image(_) | MultimodalPart::Audio(_))
    });

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
                            &samples, sample_rate,
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
        return Err(StatusCode::BAD_REQUEST);
    }

    let max_tokens = req.max_tokens.unwrap_or(256) as usize;
    let (multimodal_embeddings, audio_embeddings_len) = multimodal_embeddings;
    let prompt_len = input_ids.len() + multimodal_embeddings.len() + audio_embeddings_len;

    // Allocate decode slot (brief executor lock)
    let slot_id = state.executor.lock()
        .allocate_slot()
        .map_err(|_| StatusCode::SERVICE_UNAVAILABLE)?;

    // Run decode in blocking task (Metal is synchronous: commit + wait_until_completed)
    let state_clone = state.clone();
    let generated_tokens = tokio::task::spawn_blocking(move || {
        let mut exec = state_clone.executor.lock();

        // Sequential prefill: each token builds KV cache row
        for (i, &tok) in input_ids[..input_ids.len().saturating_sub(1)].iter().enumerate() {
            let op = make_prefill_op(tok as u64);
            exec.operation_registry.insert(op.operation_id, op);
            let plan = make_boundary_plan(i as u64, BACKEND_MEGAKERNEL, vec![OperationId(tok as u64)]);
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
            let count = state_clone.decode_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if count % 16 == 0 {
                let mut fd = state_clone.flex_dispatch.lock();
                let _ = run_flex_dispatch_cycle(&mut fd, &mut exec);
            }

            if last_token == 1 {
                // EOS token
                break;
            }
        }

        exec.free_slot(slot_id);
        generated_tokens
    }).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

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
    }).into_response())
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
    let slot_id = state.executor.lock()
        .allocate_slot()
        .map_err(|_| StatusCode::SERVICE_UNAVAILABLE)?;

    // Run decode in blocking task (Metal is synchronous)
    let state_clone = state.clone();
    let generated_text = tokio::task::spawn_blocking(move || {
        let mut exec = state_clone.executor.lock();

        // Prefill: each token builds KV cache row
        for (i, &tok) in input_ids[..input_ids.len().saturating_sub(1)].iter().enumerate() {
            let op = make_prefill_op(tok as u64);
            exec.operation_registry.insert(op.operation_id, op);
            let plan = make_boundary_plan(i as u64, BACKEND_MEGAKERNEL, vec![OperationId(tok as u64)]);
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
    }).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

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
            logical_shape: LogicalShape { dims: vec![height, width, 3] },
            physical_layout: PhysicalLayout::RowMajor,
            input_dtypes: vec![DType::U8],
            output_dtype: DType::F32,
            quantization: None,
            expected_output_shape: TensorShape { dims: vec![1] },
            correctness_checkpoint: CorrectnessCheckpointPolicy::None,
        };
        exec.operation_registry.insert(vision_op.operation_id, vision_op);

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
    let img = image::load_from_memory(bytes)
        .map_err(|e| format!("image decode failed: {e}"))?;
    let (w, h) = img.dimensions();
    let rgb = img.to_rgb8().into_raw();
    Ok((w, h, rgb))
}

// ---- TTS helpers --------------------------------------------------------

/// After text generation, if the cimage has TTS segments and the user
/// requested audio output, run TTS on the generated text.
#[allow(dead_code)]
async fn generate_audio_response(
    state: &AppState,
    text: &str,
) -> Result<Vec<u8>, StatusCode> {
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
    let wav_bytes = pcm_to_wav(&samples, sample_rate)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

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
    let has_talker = header.segments.iter().any(|s| s.kind == TTS_TALKER_WEIGHT && s.length > 0);
    let has_cp = header.segments.iter().any(|s| s.kind == TTS_CP_WEIGHT && s.length > 0);
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
/// Uses `futures::stream::unfold` to emit one SSE event per decode step,
/// then a final `[DONE]` event.
async fn chat_completions_stream(
    state: Arc<AppState>,
    req: ChatRequest,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    use futures::stream;

    let max_tokens = req.max_tokens.unwrap_or(256) as usize;

    // Tokenize (no executor lock needed — tokenizer is &self)
    let user_msg = req.messages.last().expect("at least one message");
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
    let input_ids = state.tokenizer.encode(&prompt_text).unwrap_or_default();
    let prompt_len = input_ids.len();

    let stream = stream::unfold(
        (state, input_ids, 0usize, 0u64, prompt_len),
        move |(state, ids, step, mut last_tok, prompt_len)| async move {
            if step >= max_tokens {
                // Emit final [DONE] event
                return Some((
                    Ok(Event::default().data("[DONE]")),
                    (state, ids, step, last_tok, prompt_len),
                ));
            }

            // Scope the executor lock so it drops before the state move below.
            let (next, text) = {
                let mut exec = state.executor.lock();

                if step == 0 {
                    // Prefill all tokens except the last one
                    for (i, &tok) in ids[..ids.len().saturating_sub(1)].iter().enumerate() {
                        let op = make_prefill_op(tok as u64);
                        exec.operation_registry.insert(op.operation_id, op);
                        let plan = make_boundary_plan(
                            i as u64,
                            BACKEND_MEGAKERNEL,
                            vec![OperationId(tok as u64)],
                        );
                        if let Err(e) = exec.execute_boundaries(&[plan]) {
                            eprintln!("[prism-server] streaming prefill error: {e}");
                        }
                    }
                    last_tok = *ids.last().unwrap_or(&0) as u64;
                }

                // Decode one step
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
                (next, text)
            }; // exec dropped here — state is no longer borrowed

            let event = Event::default().data(
                serde_json::json!({
                    "choices": [{
                        "delta": {"content": text.unwrap_or_default()},
                        "index": 0
                    }]
                })
                .to_string(),
            );

            Some((Ok(event), (state, ids, step + 1, next, prompt_len)))
        },
    );

    Sse::new(stream)
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
                    let event = Event::default()
                        .event("audio_chunk")
                        .data(serde_json::json!({
                            "chunk": b64,
                            "index": idx,
                            "sample_rate": 24000,
                            "format": "audio/wav",
                        }).to_string());
                    Some((Ok(event), (iter, idx + 1)))
                }
                None => {
                    let event = Event::default()
                        .event("audio_done")
                        .data("[DONE]");
                    Some((Ok(event), (iter, idx)))
                }
            }
        },
    );

    Ok(Sse::new(stream))
}

// ── Main

#[tokio::main]
async fn main() -> Result<(), String> {
    let args = Args::parse();

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

    let state = Arc::new(AppState {
        executor: ParkingMutex::new(executor),
        flex_dispatch: ParkingMutex::new(flex_dispatch),
        tokenizer,
        decode_count: std::sync::atomic::AtomicU64::new(0),
        cimage_path: args.cimage.clone(),
        tts_pipeline,
    });

    let app = Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .with_state(state);

    let addr = format!("0.0.0.0:{}", args.port);
    println!("[prism-server] Listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(|e| format!("bind: {e}"))?;

    axum::serve(listener, app)
        .await
        .map_err(|e| format!("serve: {e}"))?;

    Ok(())
}
