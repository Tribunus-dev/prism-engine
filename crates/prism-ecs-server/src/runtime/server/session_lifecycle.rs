//! Session lifecycle — create, query, transition, and close inference sessions.
//!
//! **Single authority:** This module owns the canonical session lifecycle:
//! host-side `ControlSessionState` state machine, `SessionOutcome` outcome
//! variants, `GenerationControlSession` (no MLX arrays, no KV cache), and the
//! `InferenceSessionState` worker-side state machine. The HTTP handlers for
//! `POST /v1/sessions`, `GET /v1/sessions/{id}`, `GET
//! /v1/sessions/{id}/receipt`, `DELETE /v1/sessions/{id}`, and
//! `POST /v1/sessions/{id}/generate` also live here, because their authority
//! is "admit / read / close / generate-from session". No hardware handles,
//! no `unsafe`, no FFI.
//!
//! **Canonical-vs-execution-boundary:** All types in this file are
//! canonical. The engine's `InferenceSession` (worker-side, owns
//! `Vec<KvCache>` + `AtomicBool`) stays in the engine; the engine
//! re-exports the canonical state machine types declared here.
//!
//! **Worker-side execution:** The actual prefill / decode loop runs through
//! the engine's `WirePrefillDecodeRuntime` (a `PrefillDecodeRuntime`
//! implementation); this module never holds MLX arrays.

#[cfg(feature = "server")]
use axum::{
    extract::{Path, State},
    Json,
};
#[cfg(feature = "server")]
use serde_json::{json, Value};

use crate::runtime::manifest::SessionId;
#[cfg(all(feature = "server", not(feature = "prism-backend")))]
use crate::runtime::server_types::{
    CancellationHandle, CreateSessionRequest, RequestId,
};
#[cfg(all(feature = "server", feature = "prism-backend"))]
use crate::runtime::server_types::{
    CancellationHandle, CreateSessionRequest, RequestId,
};
#[cfg(feature = "server")]
use crate::runtime::PrismInferenceServer;

#[cfg(feature = "server")]
#[cfg(all(feature = "server", feature = "prism-backend"))]
use crate::runtime::server::request_handling::parse_session_id;

// =====================================================================
//  Engine-absorbed canonical session types
// =====================================================================
//
// Absorbed from `compute-core/src/ecs/core/session.rs`. The state-machine
// shapes are pure data — they describe *what* transitions are legal, not
// *how* MLX executes them — so they belong in the constitutional server.

// -- ControlSessionState (host-side) -----------------------------------

/// Host-side session state machine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControlSessionState {
    /// Session created, pending admission.
    Created,
    /// Session admitted by the engine, awaiting worker submission.
    Admitted,
    /// Session submitted to worker, pending prefill execution.
    Submitted,
    /// Prefill input is available and ready to start (legacy path — kept for
    /// compatibility with callers that bypass admission).
    PrefillReady,
    /// Prefill is actively running.
    PrefillRunning,
    /// Autoregressive decoding loop is running.
    Decoding,
    /// Generation completed normally (EOS or max_tokens reached).
    Completed,
    /// Generation was externally cancelled.
    Cancelled,
    /// Generation failed with an error.
    Failed,
}

impl ControlSessionState {
    /// Returns `true` if the session is in a terminal state.
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed | Self::Cancelled | Self::Failed)
    }

    /// Returns `true` if transitioning to `next` is a legal forward move.
    ///
    /// Terminal states reject all transitions (including to `Failed`). Failed
    /// is reachable from any non-terminal state only.
    pub fn can_transition_to(&self, next: Self) -> bool {
        use ControlSessionState::*;

        // Identity (no-op) is always permitted.
        if *self == next {
            return true;
        }

        // Terminal states reject all non-identity transitions.
        if self.is_terminal() {
            return false;
        }

        match (*self, next) {
            // Mainline path.
            (Created, Admitted)
            | (Admitted, Submitted)
            | (Submitted, PrefillRunning)
            | (PrefillRunning, Decoding)
            | (Decoding, Completed) => true,
            // Cancellation paths.
            (Decoding, Cancelled)
            | (PrefillReady, Cancelled)
            | (PrefillRunning, Cancelled)
            | (Admitted, Cancelled)
            | (Submitted, Cancelled) => true,
            // Legacy: PrefillReady can jump into the mainline at PrefillRunning.
            (PrefillReady, PrefillRunning) => true,
            // Forward to PrefillReady from Created / Admitted.
            (Created, PrefillReady) | (Admitted, PrefillReady) => true,
            // Failed from any non-terminal.
            (_, Failed) => true,
            _ => false,
        }
    }

    /// Attempt a state transition. Returns `Ok(())` on success or `Err` with
    /// a description of the invalid transition.
    pub fn transition(&self, next: Self) -> Result<(), String> {
        if self.can_transition_to(next) {
            Ok(())
        } else {
            Err(format!("Invalid state transition: {:?} → {:?}", self, next))
        }
    }
}

// -- SessionOutcome -----------------------------------------------------

/// Outcome of a completed, cancelled, or failed generation session.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SessionOutcome {
    /// Generation completed with the given number of tokens produced.
    Completed {
        /// Total tokens generated (excluding prompt prefix).
        token_count: u32,
    },
    /// Generation was externally cancelled.
    Cancelled {
        /// Human-readable reason for cancellation.
        reason: String,
    },
    /// Generation failed with an error.
    Failed {
        /// Machine-readable error code (e.g. `"OOM"`, `"TIMEOUT"`).
        error_code: String,
        /// Human-readable error message.
        message: String,
    },
}

// -- GenerationControlSession (host-side, canonical) --------------------

/// Host-side control session — owns identity, policy state, lifecycle state,
/// deadline tracking, stream assignment, and terminal outcome.
///
/// Owns **no** MLX arrays and **no** KV cache — those belong to the worker.
#[derive(Debug)]
pub struct GenerationControlSession {
    /// Opaque session identifier.
    pub session_id: String,
    /// Hash of the model image used for this generation.
    pub model_image_hash: Option<String>,
    /// PID of the worker process executing this session.
    pub worker_pid: Option<u32>,
    /// JSON-serialised admission receipt from the engine.
    pub admission_receipt_json: Option<String>,
    /// Terminal outcome, set when the session reaches a terminal state.
    pub terminal_outcome: Option<SessionOutcome>,
    /// Current token position in the sequence (0-indexed).
    pub position: u32,
    /// Token ID that signals end-of-sequence generation.
    pub eos_token_id: u32,
    /// Maximum number of tokens to generate (inclusive of any prompt
    /// prefix length already consumed before this session).
    pub max_tokens: u32,
    /// Current session state.
    state: ControlSessionState,
}

impl GenerationControlSession {
    /// Create a new generation control session.
    pub fn new(session_id: String, eos_token_id: u32, max_tokens: u32) -> Self {
        Self {
            session_id,
            model_image_hash: None,
            worker_pid: None,
            admission_receipt_json: None,
            terminal_outcome: None,
            position: 0,
            eos_token_id,
            max_tokens,
            state: ControlSessionState::Created,
        }
    }

    /// Return the current state.
    pub fn state(&self) -> ControlSessionState {
        self.state
    }

    /// Attempt a state transition. Returns `Ok(())` or `Err` on invalid
    /// transition (the state is unchanged on error).
    pub fn transition(&mut self, next: ControlSessionState) -> Result<(), String> {
        self.state.transition(next).map(|()| self.state = next)
    }

    /// Returns `true` if the session is in a terminal state.
    pub fn is_terminal(&self) -> bool {
        self.state.is_terminal()
    }
}

// -- WorkerInferencePhase (worker-side, canonical) --------------------

/// Worker-side inference session state machine, absorbed from
/// `compute-core/src/ecs/core/session.rs`.
///
/// **Naming note:** Renamed from `InferenceSessionState` to
/// `WorkerInferencePhase` to avoid collision with the server-side
/// `InferenceSessionState` defined in
/// `crate::runtime::server_types::InferenceSessionState`, which is the
/// `SessionManager` lifecycle state used by `crate::runtime::session`.
/// The two state machines are at different layers: this one drives the
/// worker prefill/decode loop; the other drives the server's session
/// admission / ready / closed transitions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkerInferencePhase {
    /// Session created, not yet started prefill.
    Created,
    /// Prefill is actively running.
    PrefillRunning,
    /// Autoregressive decoding loop is running.
    Decoding,
    /// Generation completed normally (EOS or max_tokens reached).
    Completed,
    /// Generation was externally cancelled.
    Cancelled,
    /// Generation failed with an error.
    Failed,
}

impl WorkerInferencePhase {
    /// Returns `true` if the session is in a terminal phase.
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed | Self::Cancelled | Self::Failed)
    }

    /// Returns `true` if transitioning to `next` is a legal forward move.
    ///
    /// Terminal phases reject all transitions (including to `Failed`). Failed
    /// is reachable from any non-terminal phase only.
    pub fn can_transition_to(&self, next: Self) -> bool {
        // Identity (no-op) is always permitted.
        if *self == next {
            return true;
        }

        // Terminal phases reject all non-identity transitions.
        if self.is_terminal() {
            return false;
        }

        match (*self, next) {
            (Self::Created, Self::PrefillRunning)
            | (Self::PrefillRunning, Self::Decoding)
            | (Self::Decoding, Self::Completed)
            | (Self::Decoding, Self::Cancelled)
            | (Self::PrefillRunning, Self::Cancelled)
            | (_, Self::Failed) => true,
            _ => false,
        }
    }

    /// Attempt a phase transition. Returns `Ok(())` on success or `Err`.
    pub fn transition(&self, next: Self) -> Result<(), String> {
        if self.can_transition_to(next) {
            Ok(())
        } else {
            Err(format!(
                "Invalid WorkerInferencePhase transition: {:?} → {:?}",
                self, next
            ))
        }
    }
}

// =====================================================================
//  HTTP handlers — create / read / close / generate
// =====================================================================

/// POST /v1/sessions - create a new inference session.
#[cfg(all(feature = "server", not(feature = "prism-backend")))]
pub async fn create_session(
    State(server): State<super::request_handling::AppState>,
    Json(body): Json<Value>,
) -> Json<Value> {
    let request: CreateSessionRequest = match serde_json::from_value(body) {
        Ok(request) => request,
        Err(error) => {
            return Json(
                json!({"status":"error", "message": format!("invalid request: {error}")}),
            )
        }
    };
    match server.create_session(request) {
        Ok(session_id) => Json(json!({"status":"created", "session_id": session_id})),
        Err(error) => Json(json!({"status":"error", "message": error})),
    }
}

/// POST /v1/sessions - create a new inference session (compute-core).
#[cfg(all(feature = "server", feature = "prism-backend"))]
pub async fn create_session(
    State(server): State<super::request_handling::AppState>,
    Json(body): Json<Value>,
) -> Json<Value> {
    let request: CreateSessionRequest = match serde_json::from_value(body) {
        Ok(r) => r,
        Err(e) => {
            return Json(json!({
                "status": "error",
                "message": format!("invalid request: {e}")
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

/// POST /v1/sessions/{id}/generate - generate tokens from a session.
#[cfg(all(feature = "server", not(feature = "prism-backend")))]
pub async fn generate(
    State(server): State<super::request_handling::AppState>,
    Path(id): Path<String>,
    Json(body): Json<Value>,
) -> Json<Value> {
    let session_id = match uuid::Uuid::parse_str(&id) {
        Ok(uuid) => SessionId(uuid),
        Err(error) => {
            return Json(
                json!({"status":"error","message":format!("invalid session id '{id}': {error}")}),
            )
        }
    };
    let request = super::super::server_types::GenerateRequest {
        session_id,
        prompt: body
            .get("prompt")
            .and_then(Value::as_str)
            .unwrap_or("")
            .into(),
        max_new_tokens: body
            .get("max_new_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(256) as u32,
        sampling: super::super::server_types::SamplingConfig {
            temperature: body
                .get("temperature")
                .and_then(Value::as_f64)
                .unwrap_or(0.7) as f32,
            top_k: body.get("top_k").and_then(Value::as_u64).unwrap_or(40) as u32,
            top_p: body.get("top_p").and_then(Value::as_f64).unwrap_or(0.9) as f32,
            repetition_penalty: None,
        },
        stream: false,
    };
    let cancel = CancellationHandle {
        session_id,
        request_id: RequestId(uuid::Uuid::new_v4()),
    };
    let receiver = match server.generate(request, Some(cancel)) {
        Ok(receiver) => receiver,
        Err(error) => {
            return Json(json!({"status":"error","code":"runtime_unavailable","message":error}))
        }
    };
    let mut text = String::new();
    let mut generated = 0u32;
    let mut receiver = receiver;
    while let Some(event) = receiver.recv().await {
        match event {
            crate::runtime::GenerationStreamEvent::Token(fragment) => {
                text.push_str(&fragment);
                generated += 1;
            }
            crate::runtime::GenerationStreamEvent::Done(count) => generated = count,
            crate::runtime::GenerationStreamEvent::Error(error) => {
                return Json(json!({"status":"error","message":error}));
            }
            _ => {}
        }
    }
    Json(json!({"status":"ok","text":text,"generated_tokens":generated}))
}

/// POST /v1/sessions/{id}/generate - generate tokens via SSE stream (compute-core).
#[cfg(all(feature = "server", feature = "prism-backend"))]
pub async fn generate(
    State(server): State<super::request_handling::AppState>,
    Path(id): Path<String>,
    Json(body): Json<Value>,
) -> impl axum::response::IntoResponse {
    use axum::response::sse::{Event, Sse};
    use axum::response::IntoResponse as _;
    use serde_json::json;
    use std::convert::Infallible;
    use tokio_stream::{wrappers::ReceiverStream, StreamExt as _};

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
        #[serde(default)]
        deadline_ms: Option<u64>,
    }
    fn default_max_tokens() -> u32 {
        256
    }

    let gen_body: GenerateBody = match serde_json::from_value(body) {
        Ok(b) => b,
        Err(e) => {
            return (
                axum::http::StatusCode::BAD_REQUEST,
                [("content-type", "application/json")],
                serde_json::to_vec(&json!({"status":"error","message":format!(
                    "invalid request: {e}"
                )}))
                .unwrap(),
            )
                .into_response();
        }
    };

    let generate_request = super::super::server_types::GenerateRequest {
        session_id,
        prompt: gen_body.prompt,
        max_new_tokens: gen_body.max_new_tokens,
        sampling: super::super::server_types::SamplingConfig {
            temperature: gen_body.temperature.unwrap_or(0.7),
            top_k: gen_body.top_k.unwrap_or(40),
            top_p: gen_body.top_p.unwrap_or(0.9),
            repetition_penalty: None,
        },
        stream: gen_body.stream,
        deadline_ms: gen_body.deadline_ms,
    };

    let cancel_handle = CancellationHandle {
        session_id,
        request_id: RequestId(uuid::Uuid::new_v4()),
    };

    match server.generate(generate_request, Some(cancel_handle)) {
        Ok(rx) => {
            let stream = ReceiverStream::new(rx).map(|event| {
                let sse_event = match event {
                    crate::runtime::GenerationStreamEvent::Token(t) => {
                        Event::default().data(format!("token:{t}"))
                    }
                    crate::runtime::GenerationStreamEvent::Done(count) => {
                        Event::default().data(format!("done:{count}"))
                    }
                    crate::runtime::GenerationStreamEvent::Error(e) => {
                        Event::default().data(format!("error:{e}"))
                    }
                    crate::runtime::GenerationStreamEvent::Status(s) => Event::default().data(s),
                    crate::runtime::GenerationStreamEvent::Backpressure => {
                        Event::default().data("backpressure")
                    }
                };
                Ok::<_, Infallible>(sse_event)
            });
            Sse::new(stream).into_response()
        }
        Err(e) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            [("content-type", "application/json")],
            serde_json::to_vec(&json!({"status":"error","message":e})).unwrap(),
        )
            .into_response(),
    }
}

/// GET /v1/sessions/{id} - get session state.
#[cfg(all(feature = "server", not(feature = "prism-backend")))]
pub async fn get_session(
    State(server): State<super::request_handling::AppState>,
    Path(id): Path<String>,
) -> Json<Value> {
    let session_id = match uuid::Uuid::parse_str(&id) {
        Ok(uuid) => SessionId(uuid),
        Err(error) => {
            return Json(
                json!({"status":"error","message":format!("invalid session id '{id}': {error}")}),
            )
        }
    };
    match server.session_manager.get_state(&session_id) {
        Some(state) => {
            Json(json!({"status":"ok","session_id":session_id,"state":format!("{state:?}")}))
        }
        None => Json(
            json!({"status":"not_found","session_id":session_id,"message":"session not found"}),
        ),
    }
}

/// GET /v1/sessions/{id} - get session state (compute-core).
#[cfg(all(feature = "server", feature = "prism-backend"))]
pub async fn get_session(
    State(server): State<super::request_handling::AppState>,
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
            "state": format!("{state:?}"),
        })),
        None => Json(json!({
            "status": "not_found",
            "session_id": session_id,
            "message": "session not found"
        })),
    }
}

/// GET /v1/sessions/{id}/receipt - get session receipt.
#[cfg(all(feature = "server", not(feature = "prism-backend")))]
pub async fn get_receipt(
    State(server): State<super::request_handling::AppState>,
    Path(id): Path<String>,
) -> Json<Value> {
    let session_id = match uuid::Uuid::parse_str(&id) {
        Ok(uuid) => SessionId(uuid),
        Err(error) => {
            return Json(
                json!({"status":"error","message":format!("invalid session id '{id}': {error}")}),
            )
        }
    };
    match server.session_manager.get_receipt(&session_id) {
        Some(receipt) => Json(json!({"status":"ok","receipt":receipt})),
        None => Json(json!({"status":"not_found","session_id":session_id,"message":"no receipt found for session"})),
    }
}

/// GET /v1/sessions/{id}/receipt - get session receipt (compute-core).
#[cfg(all(feature = "server", feature = "prism-backend"))]
pub async fn get_receipt(
    State(server): State<super::request_handling::AppState>,
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

/// DELETE /v1/sessions/{id} - delete a session.
#[cfg(all(feature = "server", not(feature = "prism-backend")))]
pub async fn delete_session(
    State(server): State<super::request_handling::AppState>,
    Path(id): Path<String>,
) -> Json<Value> {
    let session_id = match uuid::Uuid::parse_str(&id) {
        Ok(uuid) => SessionId(uuid),
        Err(error) => {
            return Json(
                json!({"status":"error","message":format!("invalid session id '{id}': {error}")}),
            )
        }
    };
    match server.close_session(session_id) {
        Ok(()) => Json(json!({"status":"deleted","session_id":session_id})),
        Err(error) => Json(json!({"status":"error","message":error})),
    }
}

/// DELETE /v1/sessions/{id} - delete a session (compute-core).
#[cfg(all(feature = "server", feature = "prism-backend"))]
pub async fn delete_session(
    State(server): State<super::request_handling::AppState>,
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

// =====================================================================
//  Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- ControlSessionState state machine -----------------------------

    #[test]
    fn control_state_initial_is_not_terminal() {
        assert!(!ControlSessionState::Created.is_terminal());
    }

    #[test]
    fn control_state_terminal_set() {
        assert!(ControlSessionState::Completed.is_terminal());
        assert!(ControlSessionState::Cancelled.is_terminal());
        assert!(ControlSessionState::Failed.is_terminal());
        assert!(!ControlSessionState::Decoding.is_terminal());
    }

    #[test]
    fn control_state_valid_transitions() {
        // Classic legacy path
        assert!(ControlSessionState::Created
            .transition(ControlSessionState::PrefillReady)
            .is_ok());
        assert!(ControlSessionState::PrefillReady
            .transition(ControlSessionState::PrefillRunning)
            .is_ok());
        assert!(ControlSessionState::PrefillRunning
            .transition(ControlSessionState::Decoding)
            .is_ok());
        assert!(ControlSessionState::Decoding
            .transition(ControlSessionState::Completed)
            .is_ok());
        // Cancellation paths
        assert!(ControlSessionState::Decoding
            .transition(ControlSessionState::Cancelled)
            .is_ok());
        assert!(ControlSessionState::PrefillReady
            .transition(ControlSessionState::Cancelled)
            .is_ok());
        assert!(ControlSessionState::PrefillRunning
            .transition(ControlSessionState::Cancelled)
            .is_ok());
        // New admission path
        assert!(ControlSessionState::Created
            .transition(ControlSessionState::Admitted)
            .is_ok());
        assert!(ControlSessionState::Admitted
            .transition(ControlSessionState::Submitted)
            .is_ok());
        assert!(ControlSessionState::Submitted
            .transition(ControlSessionState::PrefillRunning)
            .is_ok());
    }

    #[test]
    fn control_state_failed_from_non_terminal() {
        let non_terminal = [
            ControlSessionState::Created,
            ControlSessionState::Admitted,
            ControlSessionState::Submitted,
            ControlSessionState::PrefillReady,
            ControlSessionState::PrefillRunning,
            ControlSessionState::Decoding,
        ];
        for s in non_terminal {
            assert!(
                s.transition(ControlSessionState::Failed).is_ok(),
                "Failed transition should be valid from {:?}",
                s,
            );
        }
    }

    #[test]
    fn control_state_terminal_rejects_failed() {
        assert!(ControlSessionState::Completed
            .transition(ControlSessionState::Failed)
            .is_err());
        assert!(ControlSessionState::Cancelled
            .transition(ControlSessionState::Failed)
            .is_err());
        assert!(ControlSessionState::Failed
            .transition(ControlSessionState::Failed)
            .is_ok());
    }

    #[test]
    fn control_state_invalid_transitions() {
        assert!(ControlSessionState::Created
            .transition(ControlSessionState::Decoding)
            .is_err());
        assert!(ControlSessionState::Created
            .transition(ControlSessionState::Completed)
            .is_err());
        assert!(ControlSessionState::Completed
            .transition(ControlSessionState::PrefillReady)
            .is_err());
        assert!(ControlSessionState::Cancelled
            .transition(ControlSessionState::PrefillReady)
            .is_err());
        assert!(ControlSessionState::Failed
            .transition(ControlSessionState::Created)
            .is_err());
    }

    // -- GenerationControlSession ---------------------------------------

    #[test]
    fn control_session_initial_state() {
        let session = GenerationControlSession::new("test-1".into(), 2, 100);
        assert_eq!(session.session_id, "test-1");
        assert_eq!(session.eos_token_id, 2);
        assert_eq!(session.max_tokens, 100);
        assert_eq!(session.position, 0);
        assert!(session.model_image_hash.is_none());
        assert!(session.worker_pid.is_none());
        assert!(session.admission_receipt_json.is_none());
        assert!(session.terminal_outcome.is_none());
        assert_eq!(session.state(), ControlSessionState::Created);
        assert!(!session.is_terminal());
    }

    #[test]
    fn control_session_happy_path() {
        let mut session = GenerationControlSession::new("s1".into(), 2, 100);
        session
            .transition(ControlSessionState::PrefillReady)
            .unwrap();
        session
            .transition(ControlSessionState::PrefillRunning)
            .unwrap();
        session.transition(ControlSessionState::Decoding).unwrap();
        session.transition(ControlSessionState::Completed).unwrap();
        assert_eq!(session.state(), ControlSessionState::Completed);
        assert!(session.is_terminal());
    }

    #[test]
    fn control_session_invalid_transition_preserves_state() {
        let mut session = GenerationControlSession::new("s6".into(), 2, 100);
        assert_eq!(session.state(), ControlSessionState::Created);
        assert!(session.transition(ControlSessionState::Decoding).is_err());
        assert_eq!(session.state(), ControlSessionState::Created);
    }

    #[test]
    fn control_session_identity_transition_is_noop() {
        let mut session = GenerationControlSession::new("s7".into(), 2, 100);
        session
            .transition(ControlSessionState::PrefillReady)
            .unwrap();
        assert!(session
            .transition(ControlSessionState::PrefillReady)
            .is_ok());
        assert_eq!(session.state(), ControlSessionState::PrefillReady);
    }

    // -- SessionOutcome -------------------------------------------------

    #[test]
    fn session_outcome_completed() {
        let outcome = SessionOutcome::Completed { token_count: 42 };
        assert_eq!(outcome, SessionOutcome::Completed { token_count: 42 });
    }

    #[test]
    fn session_outcome_cancelled() {
        let outcome = SessionOutcome::Cancelled {
            reason: "user request".into(),
        };
        assert_eq!(
            outcome,
            SessionOutcome::Cancelled {
                reason: "user request".into()
            }
        );
    }

    #[test]
    fn session_outcome_failed() {
        let outcome = SessionOutcome::Failed {
            error_code: "OOM".into(),
            message: "out of memory".into(),
        };
        assert_eq!(
            outcome,
            SessionOutcome::Failed {
                error_code: "OOM".into(),
                message: "out of memory".into()
            }
        );
    }

    // -- WorkerInferencePhase -----------------------------------------

    #[test]
    fn inference_state_initial_is_not_terminal() {
        assert!(!WorkerInferencePhase::Created.is_terminal());
    }

    #[test]
    fn inference_state_valid_transitions() {
        assert!(WorkerInferencePhase::Created
            .transition(WorkerInferencePhase::PrefillRunning)
            .is_ok());
        assert!(WorkerInferencePhase::PrefillRunning
            .transition(WorkerInferencePhase::Decoding)
            .is_ok());
        assert!(WorkerInferencePhase::Decoding
            .transition(WorkerInferencePhase::Completed)
            .is_ok());
        assert!(WorkerInferencePhase::Decoding
            .transition(WorkerInferencePhase::Cancelled)
            .is_ok());
    }

    #[test]
    fn inference_state_failed_from_non_terminal() {
        let non_terminal = [
            WorkerInferencePhase::Created,
            WorkerInferencePhase::PrefillRunning,
            WorkerInferencePhase::Decoding,
        ];
        for s in non_terminal {
            assert!(
                s.transition(WorkerInferencePhase::Failed).is_ok(),
                "Failed from {:?} should be valid",
                s,
            );
        }
    }

    #[test]
    fn inference_state_terminal_rejects_failed() {
        assert!(WorkerInferencePhase::Completed
            .transition(WorkerInferencePhase::Failed)
            .is_err());
        assert!(WorkerInferencePhase::Cancelled
            .transition(WorkerInferencePhase::Failed)
            .is_err());
    }
}
