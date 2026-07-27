//! Cancel propagation and recovery reports.
//!
//! **Single authority:** This module owns the canonical cancel-receipt
//! shape and the `POST /v1/sessions/{id}/cancel` HTTP handler. The actual
//! cooperative-cancellation machinery (a `HashSet<SessionId>` behind a
//! `Mutex`, polled from worker decode loops) lives in
//! `crate::runtime::cancel::CancellationManager` and the engine's
//! `InferenceSession::cancellation_flag`. This module is the *server-side
//! projection* of the receipt returned to the client when a cancel
//! succeeds.
//!
//! **Canonical-vs-execution-boundary:** All types and functions in this
//! file are canonical. No hardware handles, no `unsafe`, no FFI.
//!
//! **Recovery reports:** Recovery for an inference session is reported
//! by the receipt — the `InferenceCancelledReceipt` (see
//! `crate::runtime::server_types`) carries the `state_at_cancellation`,
//! `active_epoch`, `completed_decode_tokens`, and `cleanup_completed`
//! fields. The engine's `ComputeEngine::cancel_generation` is the
//! execution-boundary counterpart that flips the worker-side flag.

#[cfg(feature = "server")]
use axum::{
    extract::{Path, State},
    Json,
};
#[cfg(feature = "server")]
use serde_json::{json, Value};

use crate::runtime::manifest::SessionId;
#[cfg(all(feature = "server", not(feature = "prism-backend")))]
use crate::runtime::server_types::{CancellationHandle, RequestId};
#[cfg(all(feature = "server", feature = "prism-backend"))]
use crate::runtime::server_types::{CancellationHandle, RequestId};
#[cfg(feature = "server")]
use crate::runtime::PrismInferenceServer;

#[cfg(feature = "server")]
#[cfg(all(feature = "server", feature = "prism-backend"))]
use crate::runtime::server::request_handling::parse_session_id;

// =====================================================================
//  HTTP handlers — cancel
// =====================================================================

/// POST /v1/sessions/{id}/cancel - cancel an inference session.
#[cfg(all(feature = "server", not(feature = "prism-backend")))]
pub async fn cancel(
    State(server): State<super::request_handling::AppState>,
    Path(id): Path<String>,
    Json(_body): Json<Value>,
) -> Json<Value> {
    let session_id = match uuid::Uuid::parse_str(&id) {
        Ok(uuid) => SessionId(uuid),
        Err(error) => {
            return Json(
                json!({"status":"error","message":format!("invalid session id '{id}': {error}")}),
            )
        }
    };
    let handle = CancellationHandle {
        session_id,
        request_id: RequestId(uuid::Uuid::new_v4()),
    };
    match server.cancel(handle) {
        Ok(receipt) => Json(json!({"status":"cancelled","receipt":receipt})),
        Err(error) => Json(json!({"status":"error","message":error})),
    }
}

/// POST /v1/sessions/{id}/cancel - cancel an inference session (compute-core).
#[cfg(all(feature = "server", feature = "prism-backend"))]
pub async fn cancel(
    State(server): State<super::request_handling::AppState>,
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

// =====================================================================
//  Tests
// =====================================================================

#[cfg(all(test, feature = "server"))]
mod tests {
    use super::*;
    use crate::runtime::PrismInferenceServer;
    use crate::runtime::ServerConfig;
    use crate::runtime::server_types::InferenceExecutionPolicy;

    fn empty_server() -> std::sync::Arc<PrismInferenceServer> {
        std::sync::Arc::new(PrismInferenceServer::new(ServerConfig {
            cimage_path: String::new(),
            context_profiles: Vec::new(),
            execution_policy: InferenceExecutionPolicy::HybridMetalAccelerate,
            max_concurrent_sessions: 1,
            http_listen: None,
            receipt_store_path: std::env::temp_dir()
                .join("prism-runtime-cancel-recovery.receipts")
                .display()
                .to_string(),
            memory_elevated_threshold_bytes: 1,
            memory_critical_threshold_bytes: 2,
        }))
    }

    #[test]
    fn cancellation_handle_carries_session_and_request_ids() {
        let sid = SessionId(uuid::Uuid::new_v4());
        let rid = RequestId(uuid::Uuid::new_v4());
        let handle = CancellationHandle {
            session_id: sid,
            request_id: rid.clone(),
        };
        assert_eq!(handle.session_id, sid);
        assert_eq!(handle.request_id, rid);
    }

    #[test]
    fn empty_server_exposes_cancellation_manager() {
        // The handler above calls server.cancel(handle). The empty server
        // constructs a CancellationManager (we verify construction here
        // to ensure the type path stays wired).
        let _server = empty_server();
    }
}
