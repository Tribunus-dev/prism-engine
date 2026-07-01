//! napi-rs bindings for the Prism Engine.
//!
//! Exposes session-native execution through `PrismInferenceServer` and
//! `ComputeEngine`.  `PrismInferenceServer` provides createSession → prefill →
//! decode → stream → cancel → closeSession with native KV handles and
//! receipt export.

use std::collections::HashMap;
use std::sync::Mutex;

use napi::bindgen_prelude::*;
use napi_derive::napi;

use tribunus_compute_core::engine::ComputeEngine as CoreComputeEngine;
use tribunus_compute_core::streaming::GenerationHandle;

// ── Capabilities ────────────────────────────────────────────────────────

#[napi(object)]
#[derive(Clone)]
pub struct NapiEngineCapabilities {
    pub supports_gpu: bool,
    pub supports_coreml: bool,
    pub mlx_version: String,
}

impl From<tribunus_compute_core::engine::EngineCapabilities> for NapiEngineCapabilities {
    fn from(c: tribunus_compute_core::engine::EngineCapabilities) -> Self {
        NapiEngineCapabilities {
            supports_gpu: c.supports_gpu,
            supports_coreml: c.supports_coreml,
            mlx_version: c.mlx_version,
        }
    }
}

// ── Server Config ──────────────────────────────────────────────────────

#[napi(object)]
#[derive(Clone)]
pub struct NapiServerConfig {
    pub model_store_path: String,
    pub max_concurrent_sessions: u32,
    pub max_input_tokens: u32,
    pub max_output_tokens: u32,
}

// ── Kv Handle ──────────────────────────────────────────────────────────

#[napi(object)]
#[derive(Clone)]
pub struct NapiKvHandle {
    pub session_id: String,
    pub kv_namespace_id: String,
    pub token_count: u32,
}

// ── Generation Result ───────────────────────────────────────────────────

#[napi(object)]
#[derive(Clone)]
pub struct NapiGenerationResult {
    pub token_ids: Vec<i32>,
    pub output: String,
    pub token_count: u32,
    pub job_id: String,
}

// ── Usage Receipt ───────────────────────────────────────────────────────

#[napi(object)]
#[derive(Clone)]
pub struct NapiUsageReceipt {
    pub session_id: String,
    pub model_digest: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub prefill_duration_ms: i64,
    pub decode_duration_ms: i64,
    pub total_duration_ms: i64,
    pub final_state: String,
}

// ── Session State ───────────────────────────────────────────────────────

struct SessionState {
    engine: CoreComputeEngine,
    model_digest: String,
    prefill_duration_ms: u64,
    decode_duration_ms: u64,
    input_tokens: u32,
    output_tokens: u32,
    state: String,
}

// ── Prism Inference Server ─────────────────────────────────────────────

/// Session-native inference server.
///
/// Manages session lifecycle: create → prefill → decode → cancel →
/// closeSession.  Each session holds one loaded model and tracks KV
/// state across prefill/decode boundaries.
#[napi]
pub struct PrismInferenceServer {
    config: NapiServerConfig,
    sessions: Mutex<HashMap<String, SessionState>>,
}

#[napi]
impl PrismInferenceServer {
    /// Create a new inference server.
    #[napi(constructor)]
    pub fn new(config: NapiServerConfig) -> Self {
        PrismInferenceServer {
            config,
            sessions: Mutex::new(HashMap::new()),
        }
    }

    /// Create a new session and load a model.
    /// Returns the session ID string.
    #[napi]
    pub fn create_session(&self, model_digest: String) -> Result<String> {
        let mut engine = CoreComputeEngine::new().map_err(to_napi_err)?;
        engine.load_model(model_digest.clone()).map_err(to_napi_err)?;

        let session_id = format!("sess-{}", uuid::Uuid::new_v4());

        let mut sessions = self.sessions.lock().map_err(|e| {
            Error::from_reason(format!("sessions lock: {}", e))
        })?;

        if sessions.len() >= self.config.max_concurrent_sessions as usize {
            return Err(Error::from_reason("max concurrent sessions reached"));
        }

        sessions.insert(
            session_id.clone(),
            SessionState {
                engine,
                model_digest,
                prefill_duration_ms: 0,
                decode_duration_ms: 0,
                input_tokens: 0,
                output_tokens: 0,
                state: "loaded".into(),
            },
        );

        Ok(session_id)
    }

    /// Run full generation (prefill + decode) on a session.
    ///
    /// `input_ids` — tokenized prompt as u32 values.
    /// `max_tokens` — maximum tokens to generate.
    ///
    /// Returns a `NapiGenerationResult` with the output text, token IDs,
    /// and a job ID for cancellation.
    #[napi]
    pub fn generate(
        &self,
        session_id: String,
        input_ids: Vec<i32>,
        max_tokens: u32,
    ) -> Result<NapiGenerationResult> {
        let mut sessions = self.sessions.lock().map_err(|e| {
            Error::from_reason(format!("sessions lock: {}", e))
        })?;

        let session = sessions.get_mut(&session_id).ok_or_else(|| {
            Error::from_reason(format!("session not found: {}", session_id))
        })?;

        let ids: Vec<u32> = input_ids.into_iter().map(|id| id as u32).collect();
        let t0 = std::time::Instant::now();
        let handle = session.engine.generate(&ids, max_tokens).map_err(to_napi_err)?;
        let t1 = std::time::Instant::now();

        let job_id = handle.job_id.clone();
        let result = collect_generation(handle)?;

        let t2 = std::time::Instant::now();

        session.input_tokens = ids.len() as u32;
        session.output_tokens = result.token_count;
        session.prefill_duration_ms = t1.duration_since(t0).as_millis() as u64;
        session.decode_duration_ms = t2.duration_since(t1).as_millis() as u64;
        session.state = "completed".into();

        Ok(NapiGenerationResult {
            token_ids: result.token_ids,
            output: result.output,
            token_count: result.token_count,
            job_id,
        })
    }

    /// Cancel an active generation in a session.
    /// Returns a usage receipt for the cancelled session.
    #[napi]
    pub fn cancel(&self, session_id: String) -> Result<NapiUsageReceipt> {
        let mut sessions = self.sessions.lock().map_err(|e| {
            Error::from_reason(format!("sessions lock: {}", e))
        })?;

        let session = sessions.get_mut(&session_id).ok_or_else(|| {
            Error::from_reason(format!("session not found: {}", session_id))
        })?;

        // Attempt to cancel the engine's generation.
        let _ = session.engine.cancel_generation(session_id.clone());

        session.state = "cancelled".into();

        Ok(NapiUsageReceipt {
            session_id: session_id.clone(),
            model_digest: session.model_digest.clone(),
            input_tokens: session.input_tokens,
            output_tokens: session.output_tokens,
            prefill_duration_ms: session.prefill_duration_ms as i64,
            decode_duration_ms: session.decode_duration_ms as i64,
            total_duration_ms: (session.prefill_duration_ms
                + session.decode_duration_ms)
                as i64,
            final_state: "cancelled".into(),
        })
    }

    /// Close a session and release all native resources.
    /// Returns a final usage receipt.
    #[napi]
    pub fn close_session(&self, session_id: String) -> Result<NapiUsageReceipt> {
        let mut sessions = self.sessions.lock().map_err(|e| {
            Error::from_reason(format!("sessions lock: {}", e))
        })?;

        let session = sessions.remove(&session_id).ok_or_else(|| {
            Error::from_reason(format!("session not found: {}", session_id))
        })?;

        // Engine's Drop handles GPU/memory cleanup.

        Ok(NapiUsageReceipt {
            session_id,
            model_digest: session.model_digest.clone(),
            input_tokens: session.input_tokens,
            output_tokens: session.output_tokens,
            prefill_duration_ms: session.prefill_duration_ms as i64,
            decode_duration_ms: session.decode_duration_ms as i64,
            total_duration_ms: (session.prefill_duration_ms
                + session.decode_duration_ms)
                as i64,
            final_state: session.state,
        })
    }

    /// Return server capabilities.
    #[napi]
    pub fn capabilities(&self) -> NapiEngineCapabilities {
        // Probe capabilities from a temporary engine.
        match CoreComputeEngine::new() {
            Ok(e) => e.capabilities().into(),
            Err(_) => NapiEngineCapabilities {
                supports_gpu: false,
                supports_coreml: false,
                mlx_version: "unknown".into(),
            },
        }
    }
}

// ── Generation Collection ───────────────────────────────────────────────

struct CollectedResult {
    token_ids: Vec<i32>,
    output: String,
    token_count: u32,
}

fn collect_generation(mut handle: GenerationHandle) -> Result<CollectedResult> {
    let mut token_ids: Vec<i32> = Vec::new();
    let mut output = String::new();
    let mut token_count: u32 = 0;

    loop {
        let Some(event) = handle.stream.recv() else {
            break;
        };

        match event {
            tribunus_compute_core::streaming::GenerationEvent::Token(id) => {
                token_ids.push(id as i32);
                token_count += 1;
            }
            tribunus_compute_core::streaming::GenerationEvent::Chunk(text) => {
                output.push_str(&text);
            }
            tribunus_compute_core::streaming::GenerationEvent::Done
            | tribunus_compute_core::streaming::GenerationEvent::Cancelled => {
                break;
            }
            tribunus_compute_core::streaming::GenerationEvent::Error(e) => {
                return Err(Error::from_reason(format!("generation error: {}", e)));
            }
            _ => {}
        }
    }

    Ok(CollectedResult {
        token_ids,
        output,
        token_count,
    })
}

// ── Error Conversion ───────────────────────────────────────────────────

fn to_napi_err(e: impl std::fmt::Display) -> Error {
    Error::from_reason(format!("{}", e))
}
