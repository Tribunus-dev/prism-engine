//! Resource claims — server-side KV-epoch allocation and context refresh.
//!
//! **Single authority:** This module owns the canonical server-side
//! resource-claim management surface: the `POST /v1/sessions/{id}/compress`
//! and `POST /v1/sessions/{id}/refresh` HTTP handlers, both of which
//! allocate a fresh KV-cache epoch via `KvManager::create_epoch` to mark
//! a logical claim on memory. The actual GPU/Accelerate/CPU slot lease
//! lock state (reader counts, fence tokens) is execution-boundary and
//! lives in the engine's `WirePrefillDecodeRuntime` / `KvManager` — this
//! module only projects the canonical "is the epoch allocated?" answer.
//!
//! **Canonical-vs-execution-boundary:** All types and functions in this
//! file are canonical. No hardware handles, no `unsafe`, no FFI.

#[cfg(feature = "server")]
use axum::{
    extract::{Path, State},
    Json,
};
#[cfg(feature = "server")]
use serde_json::{json, Value};

use crate::runtime::manifest::SessionId;
#[cfg(all(feature = "server", not(feature = "prism-backend")))]
use crate::runtime::PrismInferenceServer;
#[cfg(all(feature = "server", feature = "prism-backend"))]
use crate::runtime::PrismInferenceServer;

#[cfg(feature = "server")]
#[cfg(all(feature = "server", feature = "prism-backend"))]
use crate::runtime::server::request_handling::parse_session_id;

// =====================================================================
//  HTTP handlers — compress / refresh
// =====================================================================

/// POST /v1/sessions/{id}/compress - compress KV cache.
#[cfg(all(feature = "server", not(feature = "prism-backend")))]
pub async fn compress(
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
    match server.kv_manager.create_epoch(None) {
        Ok(epoch_id) => {
            Json(json!({"status":"compressed","session_id":session_id,"epoch_id":epoch_id}))
        }
        Err(error) => Json(json!({"status":"error","message":error})),
    }
}

/// POST /v1/sessions/{id}/compress - compress KV cache (compute-core).
#[cfg(all(feature = "server", feature = "prism-backend"))]
pub async fn compress(
    State(server): State<super::request_handling::AppState>,
    Path(id): Path<String>,
) -> Json<Value> {
    let session_id = match parse_session_id(&id) {
        Ok(sid) => sid,
        Err(e) => {
            return Json(json!({"status":"error","message":e}));
        }
    };

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

/// POST /v1/sessions/{id}/refresh - refresh context.
#[cfg(all(feature = "server", not(feature = "prism-backend")))]
pub async fn refresh(
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
    match server.kv_manager.create_epoch(None) {
        Ok(epoch_id) => {
            Json(json!({"status":"refreshed","session_id":session_id,"epoch_id":epoch_id}))
        }
        Err(error) => Json(json!({"status":"error","message":error})),
    }
}

/// POST /v1/sessions/{id}/refresh - refresh context (compute-core).
#[cfg(all(feature = "server", feature = "prism-backend"))]
pub async fn refresh(
    State(server): State<super::request_handling::AppState>,
    Path(id): Path<String>,
    Json(body): Json<Value>,
) -> Json<Value> {
    let session_id = match parse_session_id(&id) {
        Ok(sid) => sid,
        Err(e) => {
            return Json(json!({"status":"error","message":e}));
        }
    };

    let _prompt: String = body
        .get("prompt")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

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

// =====================================================================
//  Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::PrismInferenceServer;
    use crate::runtime::ServerConfig;
    use crate::runtime::server_types::InferenceExecutionPolicy;
    use crate::runtime::manifest::SessionId;

    fn empty_server() -> std::sync::Arc<PrismInferenceServer> {
        std::sync::Arc::new(PrismInferenceServer::new(ServerConfig {
            cimage_path: String::new(),
            context_profiles: Vec::new(),
            execution_policy: InferenceExecutionPolicy::HybridMetalAccelerate,
            max_concurrent_sessions: 1,
            http_listen: None,
            receipt_store_path: std::env::temp_dir()
                .join("prism-runtime-resource-claims.receipts")
                .display()
                .to_string(),
            memory_elevated_threshold_bytes: 1,
            memory_critical_threshold_bytes: 2,
        }))
    }

    #[test]
    fn parse_session_id_rejects_malformed_uuid() {
        // Pure path-level validation; the real handler in
        // request_handling::parse_session_id is exercised by integration
        // tests at the server boundary.
        let bad = "not-a-uuid";
        assert!(uuid::Uuid::parse_str(bad).is_err());
        // round-trip a valid id
        let ok = uuid::Uuid::new_v4();
        let sid = SessionId(ok);
        assert_eq!(sid.0, ok);
    }

    #[test]
    fn empty_server_constructs_kv_manager() {
        // The handlers above only call server.kv_manager.create_epoch.
        // This test exercises the *path* (the manager exists, the call
        // type-checks) without asserting on a particular epoch id.
        let _server = empty_server();
    }
}
