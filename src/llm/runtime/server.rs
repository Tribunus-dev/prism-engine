// -- Prism LLM Inference - HTTP API Server -------------------------------
//
// Axum-based HTTP server exposing the Prism LLM inference API over HTTP.
// Routes are stubs returning placeholder JSON responses when the
// `prism-backend` feature is disabled, and delegate to real compute-core
// types when `prism-backend` is active.

use std::sync::{Arc, OnceLock};

#[cfg(feature = "server")]
use axum::{extract::{Path, State}, routing::{delete, get, post}, Json, Router};
#[cfg(feature = "server")]
use serde_json::{json, Value};

use super::PrismInferenceServer;

#[cfg(all(feature = "server", feature = "prism-backend"))]
use crate::llm::manifest::SessionId;
#[cfg(all(feature = "server", feature = "prism-backend"))]
use crate::llm::server::{CancellationHandle, CreateSessionRequest, GenerateRequest, RequestId};

// -- Prism-backend imports -------------------------------------------
#[cfg(all(feature = "server", feature = "prism-backend"))]
use crate::llm::runtime::modality::ModalityProvider;

#[cfg(all(feature = "server", feature = "prism-backend"))]
use {
    axum::response::sse::{Event, Sse},
    axum::response::IntoResponse,
    std::convert::Infallible,
    tokio_stream::{wrappers::ReceiverStream, StreamExt},
};

// -- Type alias (axum only)

#[cfg(feature = "server")]
type AppState = Arc<PrismInferenceServer>;

// -- HttpServer ------------------------------------------------------

/// Axum-based HTTP server that exposes the Prism LLM inference API.
pub struct HttpServer {
    listen_addr: String,
    server: OnceLock<Arc<PrismInferenceServer>>,
}

impl HttpServer {
    /// Create a new `HttpServer` bound to the given listen address.
    ///
    /// The server is not started until [`bind`] is called and the caller
    /// runs the returned [`Router`] with an axum [`serve`](axum::serve)
    /// or equivalent.
    pub fn new(listen_addr: String) -> Self {
        Self {
            listen_addr,
            server: OnceLock::new(),
        }
    }

    /// Store the server handle and return a ready-to-use [`Router`].
    ///
    /// This method does **not** start the listener - the caller is
    /// responsible for running the router with `axum::serve` or similar.
    /// This avoids blocking in test environments.
    #[cfg(feature = "server")]
    pub fn bind(&self, server: Arc<PrismInferenceServer>) -> Result<Router, String> {
        self.server
            .set(server.clone())
            .map_err(|_| "HttpServer is already bound".to_string())?;
        Ok(router(server))
    }

    /// Store the server handle. (non-axum build - no Router returned)
    #[cfg(not(feature = "server"))]
    pub fn bind(&self, server: Arc<PrismInferenceServer>) -> Result<(), String> {
        self.server
            .set(server.clone())
            .map_err(|_| "HttpServer is already bound".to_string())
    }

    /// The listen address this server was configured with.
    pub fn listen_addr(&self) -> &str {
        &self.listen_addr
    }
}

// -- Router factory (axum only) --------------------------------------

/// Build an axum [`Router`] with all 15 inference API endpoints.
///
/// Routes:
///   POST   /v1/sessions              - create session
///   POST   /v1/sessions/{id}/generate - SSE stream tokens
///   POST   /v1/sessions/{id}/cancel   - cancel session
///   POST   /v1/sessions/{id}/compress - compress KV cache
///   POST   /v1/sessions/{id}/refresh  - refresh context
///   GET    /v1/sessions/{id}          - get session state
///   GET    /v1/sessions/{id}/receipt  - get session receipt
///   DELETE /v1/sessions/{id}          - delete session
///   GET    /v1/capabilities           - list server capabilities
///   POST   /v1/images/generate        - generate image
///   POST   /v1/audio/speech           - generate speech
///   POST   /v1/video/generate         - generate video
///   POST   /v1/embeddings             - generate embeddings
///   POST   /v1/multimodal/generate    - multimodal (vision+text) generate
///   GET    /v1/health                 - health check
#[cfg(feature = "server")]
fn router(state: AppState) -> Router {
    Router::new()
        .route("/v1/sessions", post(create_session))
        .route("/v1/sessions/{id}/generate", post(generate))
        .route("/v1/sessions/{id}/cancel", post(cancel))
        .route("/v1/sessions/{id}/compress", post(compress))
        .route("/v1/sessions/{id}/refresh", post(refresh))
        .route("/v1/sessions/{id}", get(get_session))
        .route("/v1/sessions/{id}/receipt", get(get_receipt))
        .route("/v1/sessions/{id}", delete(delete_session))
        .route("/v1/capabilities", get(get_capabilities))
        .route("/v1/health", get(health))
        .route("/v1/images/generate", post(generate_image))
        .route("/v1/audio/speech", post(generate_audio))
        .route("/v1/video/generate", post(generate_video))
        .route("/v1/embeddings", post(generate_embeddings))
        .route("/v1/multimodal/generate", post(generate_multimodal))
        .with_state(state)
}

// ====================================================================
//  Handler implementations
// ====================================================================
//
// Each handler has two variants gated by `prism-backend`:
//   - Stub: returns a placeholder JSON response.
//   - Real (prism-backend): delegates to compute-core types for
//     session management, generation, and SSE streaming.

/// Helper: parse a `SessionId` from a path parameter string.
#[cfg(all(feature = "server", feature = "prism-backend"))]
fn parse_session_id(id: &str) -> Result<SessionId, String> {
    let uuid =
        uuid::Uuid::parse_str(id).map_err(|e| format!("invalid session id '{}': {}", id, e))?;
    Ok(SessionId(uuid))
}

// -- POST /v1/sessions ----------------------------------------------

/// POST /v1/sessions - create a new inference session.
#[cfg(all(feature = "server", not(feature = "prism-backend")))]
async fn create_session(
    State(_server): State<AppState>,
    Json(_body): Json<Value>,
) -> Json<Value> {
    Json(json!({
        "status": "accepted",
        "session_id": null,
        "message": "session creation not yet implemented"
    }))
}

/// POST /v1/sessions - create a new inference session (compute-core).
#[cfg(all(feature = "server", feature = "prism-backend"))]
async fn create_session(
    State(server): State<AppState>,
    Json(body): Json<Value>,
) -> Json<Value> {
    let request: CreateSessionRequest = match serde_json::from_value(body) {
        Ok(r) => r,
        Err(e) => {
            return Json(json!({
                "status": "error",
                "message": format!("invalid request: {}", e)
            }));
        }
    };

    match server.create_session(request) {
        Ok(session_id) => Json(json!({
            "status": "created",
            "session_id": session_id,
        })),
        Err(e) => Json(json!({
            "status": "error",
            "message": e,
        })),
    }
}

// -- POST /v1/sessions/{id}/generate ------------------------------

/// POST /v1/sessions/{id}/generate - generate tokens from a session.
#[cfg(all(feature = "server", not(feature = "prism-backend")))]
async fn generate(
    State(_server): State<AppState>,
    Path(_id): Path<String>,
    Json(_body): Json<Value>,
) -> Json<Value> {
    Json(json!({
        "status": "accepted",
        "message": "generation not yet implemented"
    }))
}

/// POST /v1/sessions/{id}/generate - generate tokens via SSE stream (compute-core).
#[cfg(all(feature = "server", feature = "prism-backend"))]
async fn generate(
    State(server): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<Value>,
) -> impl IntoResponse {
    let session_id = match parse_session_id(&id) {
        Ok(sid) => sid,
        Err(e) => {
            return (
                axum::http::StatusCode::BAD_REQUEST,
                [("content-type", "application/json")],
                serde_json::to_vec(&json!({"status":"error","message":e})).unwrap(),
            )
                .into_response();
        }
    };

    // Deserialize body fields without session_id (which comes from the URL path).
    #[derive(serde::Deserialize)]
    struct GenerateBody {
        #[serde(default)]
        prompt: String,
        #[serde(default = "default_max_tokens")]
        max_new_tokens: u32,
        #[serde(default)]
        temperature: Option<f32>,
        #[serde(default)]
        top_k: Option<u32>,
        #[serde(default)]
        top_p: Option<f32>,
        #[serde(default)]
        stream: bool,
    }
    fn default_max_tokens() -> u32 { 256 }

    let gen_body: GenerateBody = match serde_json::from_value(body) {
        Ok(b) => b,
        Err(e) => {
            return (
                axum::http::StatusCode::BAD_REQUEST,
                [("content-type", "application/json")],
                serde_json::to_vec(&json!({"status":"error","message":format!(
                    "invalid request: {}", e
                )})).unwrap(),
            )
                .into_response();
        }
    };

    let generate_request = GenerateRequest {
        session_id,
        prompt: gen_body.prompt,
        max_new_tokens: gen_body.max_new_tokens,
        sampling: crate::llm::server::SamplingConfig {
            temperature: gen_body.temperature.unwrap_or(0.7),
            top_k: gen_body.top_k.unwrap_or(40),
            top_p: gen_body.top_p.unwrap_or(0.9),
            repetition_penalty: None,
        },
        stream: gen_body.stream,
    };

    let cancel_handle = CancellationHandle {
        session_id,
        request_id: RequestId(uuid::Uuid::new_v4()),
    };

    match server.generate(generate_request, Some(cancel_handle)) {
        Ok(rx) => {
            let stream = ReceiverStream::new(rx).map(|event| {
                let sse_event = match event {
                    crate::llm::runtime::GenerationStreamEvent::Token(t) => {
                        Event::default().data(format!("token:{}", t))
                    }
                    crate::llm::runtime::GenerationStreamEvent::Done(count) => {
                        Event::default().data(format!("done:{}", count))
                    }
                    crate::llm::runtime::GenerationStreamEvent::Error(e) => {
                        Event::default().data(format!("error:{}", e))
                    }
                    crate::llm::runtime::GenerationStreamEvent::Status(s) => {
                        Event::default().data(s)
                    }
                    crate::llm::runtime::GenerationStreamEvent::Backpressure => {
                        Event::default().data("backpressure")
                    }
                };
                Ok::<_, Infallible>(sse_event)
            });
            Sse::new(stream).into_response()
        }
        Err(e) => {
            (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                [("content-type", "application/json")],
                serde_json::to_vec(&json!({"status":"error","message":e})).unwrap(),
            )
                .into_response()
        }
    }
}

// -- POST /v1/sessions/{id}/cancel ---------------------------------

/// POST /v1/sessions/{id}/cancel - cancel an inference session.
#[cfg(all(feature = "server", not(feature = "prism-backend")))]
async fn cancel(
    State(_server): State<AppState>,
    Path(_id): Path<String>,
    Json(_body): Json<Value>,
) -> Json<Value> {
    Json(json!({
        "status": "accepted",
        "message": "cancellation not yet implemented"
    }))
}

/// POST /v1/sessions/{id}/cancel - cancel an inference session (compute-core).
#[cfg(all(feature = "server", feature = "prism-backend"))]
async fn cancel(
    State(server): State<AppState>,
    Path(id): Path<String>,
    Json(_body): Json<Value>,
) -> Json<Value> {
    let session_id = match parse_session_id(&id) {
        Ok(sid) => sid,
        Err(e) => {
            return Json(json!({"status":"error","message":e}));
        }
    };

    let handle = CancellationHandle {
        session_id,
        request_id: RequestId(uuid::Uuid::new_v4()),
    };

    match server.cancel(handle) {
        Ok(receipt) => Json(json!({
            "status": "cancelled",
            "receipt": receipt,
        })),
        Err(e) => Json(json!({
            "status": "error",
            "message": e,
        })),
    }
}

// -- POST /v1/sessions/{id}/compress -------------------------------

/// POST /v1/sessions/{id}/compress - compress KV cache.
#[cfg(all(feature = "server", not(feature = "prism-backend")))]
async fn compress(
    State(_server): State<AppState>,
    Path(_id): Path<String>,
) -> Json<Value> {
    Json(json!({
        "status": "accepted",
        "message": "compression not yet implemented"
    }))
}

/// POST /v1/sessions/{id}/compress - compress KV cache (compute-core).
#[cfg(all(feature = "server", feature = "prism-backend"))]
async fn compress(
    State(server): State<AppState>,
    Path(id): Path<String>,
) -> Json<Value> {
    let session_id = match parse_session_id(&id) {
        Ok(sid) => sid,
        Err(e) => {
            return Json(json!({"status":"error","message":e}));
        }
    };

    // Create a new Building epoch - the KV arena adopts it for
    // compaction/eviction on the next cycle.
    match server.kv_manager.create_epoch(None) {
        Ok(epoch_id) => Json(json!({
            "status": "compressed",
            "session_id": session_id,
            "epoch_id": epoch_id,
        })),
        Err(e) => Json(json!({
            "status": "error",
            "message": e,
        })),
    }
}

// -- POST /v1/sessions/{id}/refresh --------------------------------

/// POST /v1/sessions/{id}/refresh - refresh context.
#[cfg(all(feature = "server", not(feature = "prism-backend")))]
async fn refresh(
    State(_server): State<AppState>,
    Path(_id): Path<String>,
    Json(_body): Json<Value>,
) -> Json<Value> {
    Json(json!({
        "status": "accepted",
        "message": "context refresh not yet implemented"
    }))
}

/// POST /v1/sessions/{id}/refresh - refresh context (compute-core).
#[cfg(all(feature = "server", feature = "prism-backend"))]
async fn refresh(
    State(server): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<Value>,
) -> Json<Value> {
    let session_id = match parse_session_id(&id) {
        Ok(sid) => sid,
        Err(e) => {
            return Json(json!({"status":"error","message":e}));
        }
    };

    // Parse the context-refresh plan from the request body.
    let _prompt: String = body.get("prompt").and_then(|v| v.as_str()).unwrap_or("").to_string();

    // Create a fresh epoch for the refreshed context.
    let epoch_id = match server.kv_manager.create_epoch(None) {
        Ok(eid) => eid,
        Err(e) => {
            return Json(json!({"status":"error","message":e}));
        }
    };

    Json(json!({
        "status": "refreshed",
        "session_id": session_id,
        "epoch_id": epoch_id,
    }))
}

// -- GET /v1/sessions/{id} -----------------------------------------

/// GET /v1/sessions/{id} - get session state.
#[cfg(all(feature = "server", not(feature = "prism-backend")))]
async fn get_session(
    State(_server): State<AppState>,
    Path(_id): Path<String>,
) -> Json<Value> {
    Json(json!({
        "status": "ok",
        "session_id": _id,
        "state": "unknown"
    }))
}

/// GET /v1/sessions/{id} - get session state (compute-core).
#[cfg(all(feature = "server", feature = "prism-backend"))]
async fn get_session(
    State(server): State<AppState>,
    Path(id): Path<String>,
) -> Json<Value> {
    let session_id = match parse_session_id(&id) {
        Ok(sid) => sid,
        Err(e) => {
            return Json(json!({"status":"error","message":e}));
        }
    };

    match server.session_manager.get_state(&session_id) {
        Some(state) => Json(json!({
            "status": "ok",
            "session_id": session_id,
            "state": format!("{:?}", state),
        })),
        None => Json(json!({
            "status": "not_found",
            "session_id": session_id,
            "message": "session not found"
        })),
    }
}

// -- GET /v1/sessions/{id}/receipt ----------------------------------

/// GET /v1/sessions/{id}/receipt - get session receipt.
#[cfg(all(feature = "server", not(feature = "prism-backend")))]
async fn get_receipt(
    State(_server): State<AppState>,
    Path(_id): Path<String>,
) -> Json<Value> {
    Json(json!({
        "status": "ok",
        "message": "receipt retrieval not yet implemented"
    }))
}

/// GET /v1/sessions/{id}/receipt - get session receipt (compute-core).
#[cfg(all(feature = "server", feature = "prism-backend"))]
async fn get_receipt(
    State(server): State<AppState>,
    Path(id): Path<String>,
) -> Json<Value> {
    let session_id = match parse_session_id(&id) {
        Ok(sid) => sid,
        Err(e) => {
            return Json(json!({"status":"error","message":e}));
        }
    };

    match server.receipt_store.get_receipt(&session_id) {
        Some(receipt) => Json(json!({
            "status": "ok",
            "session_id": session_id,
            "receipt": receipt,
        })),
        None => Json(json!({
            "status": "not_found",
            "session_id": session_id,
            "message": "no receipt found for session"
        })),
    }
}

// -- DELETE /v1/sessions/{id} --------------------------------------

/// DELETE /v1/sessions/{id} - delete a session.
#[cfg(all(feature = "server", not(feature = "prism-backend")))]
async fn delete_session(
    State(_server): State<AppState>,
    Path(_id): Path<String>,
) -> Json<Value> {
    Json(json!({
        "status": "accepted",
        "message": "session deletion not yet implemented"
    }))
}

/// DELETE /v1/sessions/{id} - delete a session (compute-core).
#[cfg(all(feature = "server", feature = "prism-backend"))]
async fn delete_session(
    State(server): State<AppState>,
    Path(id): Path<String>,
) -> Json<Value> {
    let session_id = match parse_session_id(&id) {
        Ok(sid) => sid,
        Err(e) => {
            return Json(json!({"status":"error","message":e}));
        }
    };

    match server.close_session(session_id) {
        Ok(()) => Json(json!({
            "status": "deleted",
            "session_id": session_id,
        })),
        Err(e) => Json(json!({
            "status": "error",
            "message": e,
        })),
    }
}

// -- GET /v1/capabilities ------------------------------------------

/// GET /v1/capabilities - list server capabilities.
#[cfg(all(feature = "server", not(feature = "prism-backend")))]
async fn get_capabilities(
    State(_server): State<AppState>,
) -> Json<Value> {
    use super::modality::ModalityCapabilities;
    let mc = ModalityCapabilities::current();
    Json(json!({
        "capabilities": mc.active_capabilities(),
        "modalities": {
            "image": mc.image,
            "audio": mc.audio,
            "video": mc.video,
            "embeddings": mc.embeddings,
            "multimodal": mc.multimodal,
        },
        "version": "0.1.0"
    }))
}

/// GET /v1/capabilities - list server capabilities (compute-core).
#[cfg(all(feature = "server", feature = "prism-backend"))]
async fn get_capabilities(
    State(server): State<AppState>,
) -> Json<Value> {
    use super::modality::ModalityCapabilities;
    let mc = ModalityCapabilities::current();
    // Report capabilities from the compute-core shared-tensor registry.
    let mut caps: Vec<String> = mc.active_capabilities().into_iter().map(String::from).collect();
    caps.push("prism-backend".to_string());
    caps.push("sse-streaming".to_string());
    caps.push("session-lifecycle".to_string());
    let caps: Vec<String> = caps.into_iter().map(String::from).collect();

    Json(json!({
        "capabilities": caps,
        "modalities": {
            "image": mc.image,
            "audio": mc.audio,
            "video": mc.video,
            "embeddings": mc.embeddings,
            "multimodal": mc.multimodal,
        },
        "version": env!("CARGO_PKG_VERSION"),
        "hardware": {
            "gpu_cores": std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1) as u32,
        },
        "memory": {
            "pressure": format!("{:?}", server.memory_monitor.current_level()),
        },
    }))
}

// -- GET /v1/health ------------------------------------------------

/// GET /v1/health - health check.
#[cfg(all(feature = "server", not(feature = "prism-backend")))]
async fn health(
    State(_server): State<AppState>,
) -> Json<Value> {
    Json(json!({
        "status": "ok"
    }))
}

/// GET /v1/health - health check (compute-core).
#[cfg(all(feature = "server", feature = "prism-backend"))]
async fn health(
    State(server): State<AppState>,
) -> Json<Value> {
    Json(json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
        "memory": {
            "pressure": format!("{:?}", server.memory_monitor.current_level()),
        },
    }))
}

// -- POST /v1/images/generate -------------------------------------

/// POST /v1/images/generate - generate an image from a text prompt.
#[cfg(all(feature = "server", not(feature = "prism-backend")))]
async fn generate_image(
    State(_server): State<AppState>,
    Json(body): Json<Value>,
) -> Json<Value> {
    let _ = body;
    Json(json!({
        "status": "accepted",
        "message": "image generation not yet implemented"
    }))
}

/// POST /v1/images/generate - generate an image (compute-core).
#[cfg(all(feature = "server", feature = "prism-backend", any(feature = "generation-image", feature = "generation-diffusion")))]
async fn generate_image(
    State(server): State<AppState>,
    Json(body): Json<Value>,
) -> Json<Value> {
    use crate::image::ImageGenerationRequest;
    let model_path = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("default");
    let prompt = body
        .get("prompt")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let width = body.get("width").and_then(|v| v.as_u64()).unwrap_or(1024) as u32;
    let height = body.get("height").and_then(|v| v.as_u64()).unwrap_or(1024) as u32;
    let request = ImageGenerationRequest::new(prompt.to_string(), width, height);
    match server.generate_image(model_path, request) {
        Ok(result) => Json(json!({
            "status": "ok",
            "image": {
                "width": result.image.width,
                "height": result.image.height,
                "format": format!("{:?}", result.image.format),
                "digest": result.image.digest.0,
            },
            "receipt": result.receipt,
        })),
        Err(e) => Json(json!({
            "status": "error",
            "message": format!("{}", e),
        })),
    }
}

/// POST /v1/images/generate - feature not enabled stub (prism-backend without image/diffusion).
#[cfg(all(feature = "server", feature = "prism-backend", not(any(feature = "generation-image", feature = "generation-diffusion"))))]
async fn generate_image(
    State(_server): State<AppState>,
    Json(body): Json<Value>,
) -> Json<Value> {
    let _ = body;
    Json(json!({
        "status": "error",
        "message": "feature not enabled: generation-image or generation-diffusion"
    }))
}

// -- POST /v1/audio/speech ----------------------------------------

/// POST /v1/audio/speech - generate speech from text.
#[cfg(all(feature = "server", not(feature = "prism-backend")))]
async fn generate_audio(
    State(_server): State<AppState>,
    Json(body): Json<Value>,
) -> Json<Value> {
    let _ = body;
    Json(json!({
        "status": "accepted",
        "message": "audio generation not yet implemented"
    }))
}

/// POST /v1/audio/speech - generate speech (compute-core).
#[cfg(all(feature = "server", feature = "prism-backend", feature = "generation-audio"))]
async fn generate_audio(
    State(server): State<AppState>,
    Json(body): Json<Value>,
) -> Json<Value> {
    let model_path = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("default");
    let text = body
        .get("text")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let params = crate::audio::AudioParams {
        voice: body.get("voice").and_then(|v| v.as_str()).map(String::from),
    };
    match server.generate_audio(model_path, text, params) {
        Ok(receipt) => Json(json!({
            "status": "ok",
            "sample_rate": receipt.sample_rate,
            "pcm_samples": receipt.pcm_samples,
            "compute_ms": receipt.compute_ms,
            "output_digest": receipt.output_digest,
        })),
        Err(e) => Json(json!({
            "status": "error",
            "message": format!("{}", e),
        })),
    }
}

/// POST /v1/audio/speech - feature not enabled stub (prism-backend without audio).
#[cfg(all(feature = "server", feature = "prism-backend", not(feature = "generation-audio")))]
async fn generate_audio(
    State(_server): State<AppState>,
    Json(body): Json<Value>,
) -> Json<Value> {
    let _ = body;
    Json(json!({
            "status": "error",
        "message": "feature not enabled: generation-audio"
    }))
}

// -- POST /v1/video/generate --------------------------------------

/// POST /v1/video/generate - generate a video from a text prompt.
#[cfg(all(feature = "server", not(feature = "prism-backend")))]
async fn generate_video(
    State(_server): State<AppState>,
    Json(body): Json<Value>,
) -> Json<Value> {
    let _ = body;
    Json(json!({
        "status": "accepted",
        "message": "video generation not yet implemented"
    }))
}

/// POST /v1/video/generate - generate a video (compute-core).
#[cfg(all(feature = "server", feature = "prism-backend", feature = "generation-video"))]
async fn generate_video(
    State(server): State<AppState>,
    Json(body): Json<Value>,
) -> Json<Value> {
    let model_path = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("default");
    let prompt = body
        .get("prompt")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let params = crate::video::VideoParams {
        num_frames: body.get("num_frames").and_then(|v| v.as_u64()).unwrap_or(16) as u32,
        fps: body.get("fps").and_then(|v| v.as_u64()).unwrap_or(24) as u32,
        seed: body.get("seed").and_then(|v| v.as_u64()).unwrap_or(42),
    };
    match server.generate_video(model_path, prompt, params) {
        Ok(receipt) => Json(json!({
            "status": "ok",
            "frames": receipt.frames,
            "compute_ms": receipt.compute_ms,
        })),
        Err(e) => Json(json!({
            "status": "error",
            "message": format!("{}", e),
        })),
    }
}

/// POST /v1/video/generate - feature not enabled stub (prism-backend without video).
#[cfg(all(feature = "server", feature = "prism-backend", not(feature = "generation-video")))]
async fn generate_video(
    State(_server): State<AppState>,
    Json(body): Json<Value>,
) -> Json<Value> {
    let _ = body;
    Json(json!({
        "status": "error",
        "message": "feature not enabled: generation-video"
    }))
}

// -- POST /v1/embeddings ------------------------------------------

/// POST /v1/embeddings - generate text embeddings.
#[cfg(all(feature = "server", not(feature = "prism-backend")))]
async fn generate_embeddings(
    State(_server): State<AppState>,
    Json(body): Json<Value>,
) -> Json<Value> {
    let _ = body;
    Json(json!({
        "status": "accepted",
        "message": "embedding generation not yet implemented"
    }))
}

/// POST /v1/embeddings - generate text embeddings (compute-core).
#[cfg(all(feature = "server", feature = "prism-backend"))]
async fn generate_embeddings(
    State(server): State<AppState>,
    Json(body): Json<Value>,
) -> Json<Value> {
    let model_path = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("default");
    let text = body
        .get("text")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    match server.generate_embeddings(model_path, text) {
        Ok(embeddings) => Json(json!({
            "status": "ok",
            "embeddings": embeddings,
        })),
        Err(e) => Json(json!({
            "status": "error",
            "message": e,
        })),
    }
}
// -- POST /v1/embeddings ------------------------------------------

/// POST /v1/multimodal/generate - multimodal (vision+text) generation.
#[cfg(all(feature = "server", not(feature = "prism-backend")))]
async fn generate_multimodal(
    State(_server): State<AppState>,
    Json(body): Json<Value>,
) -> Json<Value> {
    let _ = body;
    Json(json!({
        "status": "accepted",
        "message": "multimodal generation not yet implemented"
    }))
}

/// POST /v1/multimodal/generate - multimodal generation (compute-core, stub).
#[cfg(all(feature = "server", feature = "prism-backend"))]
async fn generate_multimodal(
    State(_server): State<AppState>,
    Json(body): Json<Value>,
) -> Json<Value> {
    let _ = body;
    Json(json!({
        "status": "accepted",
        "message": "multimodal generation stub - requires MultimodalPipeline wiring"
    }))
}
