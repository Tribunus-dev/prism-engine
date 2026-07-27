//! Session lifecycle — create, query, transition, and close inference sessions.
//!
//! **Single authority:** This module owns the canonical session lifecycle:
//! host-side `ControlSessionState` state machine, `SessionOutcome` outcome
//! variants, `GenerationControlSession` (no MLX arrays, no KV cache), and the
//! `WorkerInferencePhase` worker-side state machine. The HTTP handlers for
//! `POST /v1/sessions`, `GET /v1/sessions/{id}`, `GET
//! /v1/sessions/{id}/receipt`, `DELETE /v1/sessions/{id}`, and
//! `POST /v1/sessions/{id}/generate` also live here, because their authority
//! is "admit / read / close / generate-from session". No hardware handles,
//! no `unsafe`, no FFI.
//!
//! **Sub-module decomposition:** the canonical state machines and the session
//! handle live in single-authority sub-modules so the module stays under the
//! 600 LOC / 20 public-item threshold:
//!
//! - [`control_state`] — `ControlSessionState` (host-side transitions).
//! - [`inference_state`] — `WorkerInferencePhase` (worker-side transitions).
//! - [`outcome`] — `SessionOutcome` + `GenerationControlSession` (the
//!   outcome envelope and the host-side session record that holds the
//!   lifecycle state).
//!
//! The re-exports below preserve the canonical paths that the engine and
//! downstream callers already use, so the move from one file to a directory
//! is observably a no-op at the import surface.
//!
//! **Canonical-vs-execution-boundary:** All types in this module are
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

pub mod control_state;
pub mod inference_state;
pub mod outcome;

// =====================================================================
//  Re-exports — preserve the canonical import surface.
// =====================================================================
//
// The engine's `compute-core/src/ecs/core/session.rs` documents these
// canonical paths, and downstream callers depend on them. Re-exporting
// here means the move from one file to a directory is a no-op at the
// `use` site.

pub use control_state::ControlSessionState;
pub use inference_state::WorkerInferencePhase;
pub use outcome::{GenerationControlSession, SessionOutcome};

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
