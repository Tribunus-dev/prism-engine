//! Request handling — request → command translation and runtime port surface.
//!
//! **Single authority:** This module owns the canonical request shapes
//! (`GenerateRequest`, `SamplingConfig`, `EngineCapabilities`, `GenerationRequest`),
//! the canonical request-decoding helpers (`parse_session_id`), the
//! `PrefillDecodeRuntime` typed port interface that backends must implement,
//! the `AppState` alias, the `HttpServer` server struct, the `router()` factory,
//! the `generate_stream` SSE generator, and the capability / health /
//! telemetry / generate HTTP handlers. No module-local state; the SSE stream
//! holds only an mpsc sender (one per request), the `HttpServer` holds a
//! `OnceLock<Arc<PrismInferenceServer>>` (one per process), and the runtime
//! port is implemented by the engine via `WirePrefillDecodeRuntime`.
//!
//! **Canonical-vs-execution-boundary:** All types and functions in this file
//! are canonical. The execution boundary lives in the engine's
//! `WirePrefillDecodeRuntime` and `ComputeEngine`, which implement the
//! `PrefillDecodeRuntime` port declared here.
//!
//! **MLX boundary note:** The engine retains the MLX-specific execution
//! paths (MLX-backed `ComputeEngine::run_inference_cycle` and
//! `check_memory_pressure`). This module never imports `mlx_*` symbols; all
//! MLX interaction is hidden behind the `PrefillDecodeRuntime` port.

use std::sync::{Arc, OnceLock};

#[cfg(feature = "server")]
use axum::{
    extract::State,
    routing::{delete, get, post},
    Json, Router,
};
#[cfg(feature = "server")]
use serde_json::{json, Value};

#[cfg(all(feature = "server", not(feature = "prism-backend")))]
use crate::runtime::server_types::CreateSessionRequest;
#[cfg(all(feature = "server", feature = "prism-backend"))]
use crate::runtime::server_types::{
    CancellationHandle, CreateSessionRequest, GenerateRequest, RequestId,
};
#[cfg(all(feature = "server", not(feature = "prism-backend")))]
use crate::runtime::server_types::{CancellationHandle, GenerateRequest, RequestId};

use crate::runtime::server_types::SamplingConfig;
use crate::runtime::PrismInferenceServer;

#[cfg(all(feature = "server", feature = "prism-backend"))]
use {
    axum::response::sse::{Event, Sse},
    axum::response::IntoResponse,
    std::convert::Infallible,
    tokio_stream::{wrappers::ReceiverStream, StreamExt as _},
};

// -- Prefill/Decode Runtime port (execution-boundary boundary) ------------

/// Trait for the autoregressive inference runtime.
///
/// **Typed port interface.** Backends implement this trait to provide
/// tokenization, prefill (prompt evaluation), single-token decode, sampling,
/// detokenization, and EOS detection. The engine's `WirePrefillDecodeRuntime`
/// and (when the `prism-backend` feature is enabled) the legacy
/// `ComputeEngine` implement this trait; this module never touches MLX
/// directly.
pub trait PrefillDecodeRuntime: Send + Sync {
    /// Tokenize a text prompt into token IDs.
    fn tokenize(&self, prompt: &str) -> Result<Vec<u32>, String>;
    /// Produce a normalized mean-pooled text embedding for a prompt.
    fn embed_text(&self, prompt: &str) -> Result<Vec<f32>, String>;
    /// Run the prefill (prompt evaluation) forward pass.
    ///
    /// Returns the logits for the first output token position.
    fn run_prefill(&self, prompt_tokens: &[u32]) -> Result<Vec<f32>, String>;
    /// Run a single decode (token generation) forward pass.
    ///
    /// `token` is the previously-generated token ID.  Returns logits for
    /// the next token position.
    fn run_decode(&self, token: u32) -> Result<Vec<f32>, String>;
    /// Sample a token ID from logits using the given sampling configuration.
    fn sample(&self, logits: &[f32], config: &SamplingConfig) -> Result<u32, String>;
    /// Detokenize a single token ID into its text fragment.
    fn detokenize(&self, token: u32) -> Result<String, String>;
    /// The end-of-sequence token ID.
    fn eos_token_id(&self) -> u32;
}

// -- SessionId parsing -----------------------------------------------------

/// Parse a `SessionId` from a path parameter string.
#[cfg(all(feature = "server", feature = "prism-backend"))]
pub fn parse_session_id(id: &str) -> Result<crate::runtime::manifest::SessionId, String> {
    let uuid = uuid::Uuid::parse_str(id).map_err(|e| format!("invalid session id '{id}': {e}"))?;
    Ok(crate::runtime::manifest::SessionId(uuid))
}

// -- AppState alias --------------------------------------------------------

#[cfg(feature = "server")]
pub type AppState = Arc<PrismInferenceServer>;

// -- Model provenance for /v1/capabilities --------------------------------

#[cfg(feature = "server")]
fn registered_model_provenance(server: &AppState) -> Value {
    let model_ids = server
        .model_registry
        .read()
        .map(|registry| registry.keys().cloned().collect::<Vec<_>>())
        .unwrap_or_default();
    let mut provenance = serde_json::Map::new();
    for model_id in model_ids {
        if let Ok(inspection) = server.model_inspection(&model_id) {
            provenance.insert(
                model_id,
                json!({
                    "native_ternary_promotion": inspection.native_ternary_promotion,
                    "joint_tiling_evidence": inspection.joint_tiling_evidence,
                    "tensor_count": inspection.manifest.tensor_count,
                }),
            );
        }
    }
    Value::Object(provenance)
}

// -- HttpServer ------------------------------------------------------------

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

// -- Router factory (axum only) -------------------------------------------

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
        .route("/v1/sessions", post(super::session_lifecycle::create_session))
        .route(
            "/v1/sessions/{id}/generate",
            post(super::session_lifecycle::generate),
        )
        .route(
            "/v1/sessions/{id}/cancel",
            post(super::cancel_recovery::cancel),
        )
        .route(
            "/v1/sessions/{id}/compress",
            post(super::resource_claims::compress),
        )
        .route(
            "/v1/sessions/{id}/refresh",
            post(super::resource_claims::refresh),
        )
        .route(
            "/v1/sessions/{id}",
            get(super::session_lifecycle::get_session),
        )
        .route(
            "/v1/sessions/{id}/receipt",
            get(super::session_lifecycle::get_receipt),
        )
        .route(
            "/v1/sessions/{id}",
            delete(super::session_lifecycle::delete_session),
        )
        .route("/v1/capabilities", get(get_capabilities))
        .route("/v1/telemetry", get(get_telemetry))
        .route("/v1/health", get(health))
        .route(
            "/v1/images/generate",
            post(super::modality_dispatch::generate_image),
        )
        .route(
            "/v1/audio/speech",
            post(super::modality_dispatch::generate_audio),
        )
        .route(
            "/v1/video/generate",
            post(super::modality_dispatch::generate_video),
        )
        .route(
            "/v1/embeddings",
            post(super::modality_dispatch::generate_embeddings),
        )
        .route(
            "/v1/multimodal/generate",
            post(super::modality_dispatch::generate_multimodal),
        )
        .with_state(state)
}

// -- generate_stream ------------------------------------------------------

/// Stream token generation, producing SSE events.
///
/// 1. Tokenize the prompt.
/// 2. Run prefill (prompt evaluation) to populate the KV-cache.
/// 3. Enter the decode loop:
///    - Sample the next token from logits
///    - SSE-stream it to the client
///    - Check the cancellation handle
///    - Check the deadline
///    - Run decode for the new token
///    - Break on EOS or max tokens
/// 4. Emit a `done` event with the final token count.
///
/// Returns a `ReceiverStream` of SSE [`Event`] values ready for axum's `Sse`.
#[cfg(all(feature = "server", feature = "prism-backend"))]
pub async fn generate_stream(
    server: Arc<PrismInferenceServer>,
    request: GenerateRequest,
    cancel: CancellationHandle,
    runtime: Arc<dyn PrefillDecodeRuntime>,
) -> Result<ReceiverStream<Result<Event, Infallible>>, String> {
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(64);
    let cancel_mgr = Arc::clone(&server.cancellation_manager);
    let session_id = cancel.session_id;
    let max_tokens = request.max_new_tokens;
    let sampling = request.sampling.clone();
    let prompt = request.prompt.clone();
    let deadline_dur = request.deadline_ms.map(std::time::Duration::from_millis);

    tokio::spawn(async move {
        // Register cancellation handle so the session can be cancelled.
        cancel_mgr.register_handle(session_id);

        let start = std::time::Instant::now();

        // Helper: check both cancellation and deadline.
        let check_interrupts = || -> bool {
            if cancel_mgr.is_cancelled(&session_id) {
                return true;
            }
            if let Some(dl) = deadline_dur {
                if start.elapsed() >= dl {
                    return true;
                }
            }
            false
        };

        if check_interrupts() {
            let _ = tx
                .send(Ok(Event::default().data("error:cancelled")))
                .await;
            return;
        }

        // 1. Tokenize prompt.
        let prompt_tokens = match runtime.tokenize(&prompt) {
            Ok(t) => t,
            Err(e) => {
                let _ = tx
                    .send(Ok(Event::default().data(format!("error:tokenize {e}"))))
                    .await;
                return;
            }
        };

        // 2. Run prefill (prompt evaluation).
        let _ = tx
            .send(Ok(
                Event::default().data(format!("status:prefill {} tokens", prompt_tokens.len())),
            ))
            .await;

        let logits = match runtime.run_prefill(&prompt_tokens) {
            Ok(l) => l,
            Err(e) => {
                let _ = tx
                    .send(Ok(Event::default().data(format!("error:prefill {e}"))))
                    .await;
                return;
            }
        };

        let _ = tx
            .send(Ok(Event::default().data("status:prefill complete")))
            .await;

        let eos = runtime.eos_token_id();
        let mut token_count = 0u32;
        let mut current_logits = logits;

        // 3. Decode loop.
        for _ in 0..max_tokens {
            if check_interrupts() {
                let _ = tx
                    .send(Ok(Event::default().data("error:cancelled")))
                    .await;
                return;
            }

            // Sample next token from logits.
            let token = match runtime.sample(&current_logits, &sampling) {
                Ok(t) => t,
                Err(e) => {
                    let _ = tx
                        .send(Ok(Event::default().data(format!("error:sample {e}"))))
                        .await;
                    return;
                }
            };

            // Break on EOS.
            if token == eos {
                break;
            }

            // Detokenize and SSE-stream.
            match runtime.detokenize(token) {
                Ok(text) => {
                    if tx
                        .send(Ok(Event::default().data(format!("token:{text}"))))
                        .await
                        .is_err()
                    {
                        return; // Client disconnected.
                    }
                }
                Err(e) => {
                    let _ = tx
                        .send(Ok(Event::default().data(format!("error:detokenize {e}"))))
                        .await;
                    return;
                }
            }

            token_count += 1;

            if check_interrupts() {
                let _ = tx
                    .send(Ok(Event::default().data("error:cancelled")))
                    .await;
                return;
            }

            // Run decode for the newly-generated token → logits for next position.
            current_logits = match runtime.run_decode(token) {
                Ok(l) => l,
                Err(e) => {
                    let _ = tx
                        .send(Ok(Event::default().data(format!("error:decode {e}"))))
                        .await;
                    return;
                }
            };
        }

        // 4. Signal done.
        let _ = tx
            .send(Ok(Event::default().data(format!("done:{token_count}"))))
            .await;
    });

    Ok(ReceiverStream::new(rx))
}

// ====================================================================
//  Capability, telemetry, health handlers
// ====================================================================
//
// The capability / telemetry / health endpoints are part of request
// handling (canonical) because they only project state and never reach
// out to the MLX execution path directly.

/// GET /v1/capabilities - list server capabilities.
#[cfg(all(feature = "server", not(feature = "prism-backend")))]
async fn get_capabilities(State(server): State<AppState>) -> Json<Value> {
    use crate::runtime::lanes::LaneCapabilities;
    use crate::runtime::modality::ModalityCapabilities;
    let mc = ModalityCapabilities::current();
    let lanes = LaneCapabilities::host();
    let capture_devices = prism_multimodal::capture::enumerate_apple_capture_devices()
        .ok()
        .and_then(|devices| serde_json::to_value(devices).ok())
        .unwrap_or_else(|| json!([]));
    let capture_permissions = prism_multimodal::capture::probe_apple_capture_permissions()
        .map(|permissions| {
            json!({
                "microphone": format!("{:?}", permissions.microphone),
                "camera": format!("{:?}", permissions.camera),
            })
        })
        .unwrap_or_else(|error| json!({"error": error.to_string()}));
    let registered_models = server
        .model_manifests
        .read()
        .ok()
        .and_then(|manifests| serde_json::to_value(&*manifests).ok())
        .unwrap_or_else(|| json!({}));
    let registered_model_provenance = registered_model_provenance(&server);
    Json(json!({
        "capabilities": mc.active_capabilities(),
        "modalities": {
            "image": mc.image,
            "audio": mc.audio,
            "video": mc.video,
            "embeddings": mc.embeddings,
            "multimodal": mc.multimodal,
        },
        "capture_devices": capture_devices,
        "registered_model_manifests": registered_models,
        "registered_model_provenance": registered_model_provenance,
        "live_runtime_count": server.live_runtime_count().unwrap_or(0),
        "hardware": {
            "metal": lanes.metal,
            "accelerate": lanes.accelerate,
            "coreml_ane": lanes.coreml_ane,
            "video_toolbox": cfg!(target_os = "macos"),
            "av_foundation": cfg!(target_os = "macos"),
        },
        "capture_permissions": capture_permissions,
        "version": "0.1.0"
    }))
}

/// GET /v1/capabilities - list server capabilities (compute-core).
#[cfg(all(feature = "server", feature = "prism-backend"))]
async fn get_capabilities(State(server): State<AppState>) -> Json<Value> {
    use crate::runtime::lanes::LaneCapabilities;
    use crate::runtime::modality::ModalityCapabilities;
    let mc = ModalityCapabilities::current();
    let lanes = LaneCapabilities::host();
    let mut caps: Vec<String> = mc
        .active_capabilities()
        .into_iter()
        .map(String::from)
        .collect();
    caps.push("prism-backend".to_string());
    caps.push("sse-streaming".to_string());
    caps.push("session-lifecycle".to_string());
    let capture_devices = prism_multimodal::capture::enumerate_apple_capture_devices()
        .ok()
        .and_then(|devices| serde_json::to_value(devices).ok())
        .unwrap_or_else(|| json!([]));
    let capture_permissions = prism_multimodal::capture::probe_apple_capture_permissions()
        .map(|permissions| {
            json!({
                "microphone": format!("{:?}", permissions.microphone),
                "camera": format!("{:?}", permissions.camera),
            })
        })
        .unwrap_or_else(|error| json!({"error": error.to_string()}));
    let registered_models = server
        .model_manifests
        .read()
        .ok()
        .and_then(|manifests| serde_json::to_value(&*manifests).ok())
        .unwrap_or_else(|| json!({}));
    let registered_model_provenance = registered_model_provenance(&server);

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
            "metal": lanes.metal,
            "accelerate": lanes.accelerate,
            "coreml_ane": lanes.coreml_ane,
            "video_toolbox": cfg!(target_os = "macos"),
            "av_foundation": cfg!(target_os = "macos"),
        },
        "capture_devices": capture_devices,
        "capture_permissions": capture_permissions,
        "registered_model_manifests": registered_models,
        "registered_model_provenance": registered_model_provenance,
        "live_runtime_count": server.live_runtime_count().unwrap_or(0),
        "memory": {
            "pressure": format!("{:?}", server.memory_monitor.current_level()),
        },
    }))
}

/// GET /v1/health - health check.
#[cfg(all(feature = "server", not(feature = "prism-backend")))]
async fn health(State(_server): State<AppState>) -> Json<Value> {
    Json(json!({
        "status": "ok"
    }))
}

/// GET /v1/health - health check (compute-core).
#[cfg(all(feature = "server", feature = "prism-backend"))]
async fn health(State(server): State<AppState>) -> Json<Value> {
    Json(json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
        "memory": {
            "pressure": format!("{:?}", server.memory_monitor.current_level()),
        },
    }))
}

/// GET /v1/telemetry - structured inference counters and latency samples.
#[cfg(feature = "server")]
async fn get_telemetry(State(server): State<AppState>) -> Json<Value> {
    Json(
        serde_json::to_value(server.telemetry.snapshot())
            .unwrap_or_else(|_| json!({"error":"telemetry serialization failed"})),
    )
}

// ====================================================================
//  Engine-canonical request types (absorbed from compute-core engine.rs)
// ====================================================================
//
// The following types were originally defined in
// `compute-core/src/ecs/core/engine.rs` and have been re-homed here
// because they are pure data shapes with no hardware dependency. The
// engine's `ComputeEngine` keeps the execution-boundary `LoadedModel` /
// `ComputeEngine` / `check_memory_pressure` and re-exports these.

/// Parameters for a text generation request.
///
/// All numeric fields use their MLX-native defaults when left at zero
/// (the JS side maps `undefined` → `0` / `null` → `None` for Option fields).
///
/// The only required field is `prompt`.
#[derive(Debug, Clone)]
pub struct GenerationRequest {
    /// Input text prompt.
    pub prompt: String,
    /// Opaque session identifier for this generation run.
    pub session_id: String,
    /// Maximum number of tokens to generate (0 = bounded qualification mode).
    pub max_tokens: u32,
    /// Token ID that signals end-of-sequence.
    pub eos_token_id: u32,
    /// Pre-tokenized input token IDs for the prompt.
    pub input_ids: Vec<i32>,
    /// Temperature for softmax scaling.  0.0 = greedy.
    pub temperature: f64,
    /// Top-k filter: retain only the k highest-probability tokens.
    pub top_k: u32,
    /// Top-p (nucleus) filter: retain smallest set whose cumulative
    /// probability exceeds p.
    pub top_p: f64,
    /// Optional PRNG seed for deterministic sampling.
    pub seed: Option<u64>,
    /// Token ID sequences at which generation should stop.
    pub stop_sequences: Vec<String>,
}

impl GenerationRequest {
    /// Return the session identifier for this request.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Return the end-of-sequence token ID.
    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }
}

/// Static capability report for this compute engine instance.
#[derive(Debug, Clone)]
pub struct EngineCapabilities {
    /// Whether a Metal-compatible GPU is available.
    pub supports_gpu: bool,
    /// Whether Core ML model execution is available.
    pub supports_coreml: bool,
    /// MLX framework version string (semver).
    pub mlx_version: String,
}

// ====================================================================
//  Engine-canonical sampling config (absorbed from compute-core session.rs)
// ====================================================================

/// Sampling / decoding configuration for one generation run.
///
/// All fields are optional with the following semantics:
/// - `None` → use a sensible default (greedy decoding defaults are shown
///   below).
/// - `Some(value)` → override the default.
///
/// The default configuration is greedy: `top_k = Some(1)` with all other
/// parameters effectively disabled.
#[derive(Clone, Debug, PartialEq)]
pub struct SamplerConfig {
    /// Temperature for softmax scaling. Lower values sharpen the
    /// distribution. `None` / `Some(0.0)` → greedy (always pick top token).
    /// Default: `None`.
    pub temperature: Option<f32>,
    /// Top-k filtering: retain only the `k` highest-probability tokens
    /// before sampling. `Some(1)` → greedy. `None` → no top-k filtering.
    /// Default: `Some(1)`.
    pub top_k: Option<u32>,
    /// Top-p (nucleus) filtering: retain the smallest set of tokens whose
    /// cumulative probability exceeds `p`. `None` → no top-p filtering.
    /// Default: `None`.
    pub top_p: Option<f32>,
    /// Repetition penalty applied to tokens that have already appeared.
    /// Values > 1.0 penalise repetition; < 1.0 encourage it.
    /// `None` → no penalty (equivalent to 1.0).
    /// Default: `None`.
    pub repetition_penalty: Option<f32>,
    /// PRNG seed for deterministic sampling. `None` → non-deterministic.
    /// Default: `None`.
    pub seed: Option<u64>,
    /// Token IDs at which generation should stop (in addition to `eos_token_id`).
    /// Default: empty.
    pub stop_token_ids: Vec<u32>,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: None,
            top_k: Some(1),
            top_p: None,
            repetition_penalty: None,
            seed: None,
            stop_token_ids: Vec::new(),
        }
    }
}

impl SamplerConfig {
    /// Returns `true` when the config selects greedy (argmax) decoding.
    pub fn is_greedy(&self) -> bool {
        self.top_k == Some(1) || self.temperature == Some(0.0) || self.temperature == None
    }
}

// ====================================================================
//  Tests
// ====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- SamplerConfig: greedy / non-greedy classification ----------------

    #[test]
    fn sampler_config_greedy_default() {
        let config = SamplerConfig::default();
        assert!(config.temperature.is_none());
        assert_eq!(config.top_k, Some(1));
        assert!(config.top_p.is_none());
        assert!(config.repetition_penalty.is_none());
        assert!(config.seed.is_none());
        assert!(config.stop_token_ids.is_empty());
        assert!(config.is_greedy());
    }

    #[test]
    fn sampler_config_temperature_zero_is_greedy() {
        let config = SamplerConfig {
            temperature: Some(0.0),
            top_k: None,
            top_p: None,
            repetition_penalty: None,
            seed: None,
            stop_token_ids: Vec::new(),
        };
        assert!(config.is_greedy());
    }

    #[test]
    fn sampler_config_not_greedy() {
        let config = SamplerConfig {
            temperature: Some(0.8),
            top_k: Some(50),
            top_p: Some(0.9),
            repetition_penalty: Some(1.1),
            seed: Some(42),
            stop_token_ids: vec![3, 4],
        };
        assert!(!config.is_greedy());
        assert_eq!(config.stop_token_ids, vec![3, 4]);
    }

    #[test]
    fn sampler_config_partial_override() {
        let config = SamplerConfig {
            temperature: Some(0.9),
            ..Default::default()
        };
        assert_eq!(config.temperature, Some(0.9));
        assert_eq!(config.top_k, Some(1));
        assert!(config.top_p.is_none());
        assert!(config.is_greedy());
    }

    #[test]
    fn sampler_config_not_greedy_with_top_k_none() {
        let config = SamplerConfig {
            temperature: Some(0.9),
            top_k: None,
            ..Default::default()
        };
        assert!(!config.is_greedy());
    }

    // -- GenerationRequest ---------------------------------------------

    #[test]
    fn generation_request_exposes_session_id_and_eos() {
        let req = GenerationRequest {
            prompt: "hello".into(),
            session_id: "s-1".into(),
            max_tokens: 16,
            eos_token_id: 2,
            input_ids: vec![1, 2, 3],
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            seed: Some(7),
            stop_sequences: Vec::new(),
        };
        assert_eq!(req.session_id(), "s-1");
        assert_eq!(req.eos_token_id(), 2);
    }

    // -- HttpServer: OnceLock + listen_addr -----------------------------

    #[test]
    fn http_server_new_stores_listen_addr_without_binding() {
        let s = HttpServer::new("127.0.0.1:0".into());
        assert_eq!(s.listen_addr(), "127.0.0.1:0");
    }
}
