//! Text embeddings — `POST /v1/embeddings`.
//!
//! **Single authority:** This sub-module owns the canonical HTTP handler
//! for text-embedding generation. When a live runtime is admitted the
//! handler forwards to `PrismInferenceServer::generate_embeddings` (an
//! execution-boundary trait); when no live runtime is registered, the
//! `prism-backend`-disabled branch returns the canonical
//! `runtime_unavailable` error so clients can distinguish "feature not
//! enabled" from "model not registered".
//!
//! **Canonical-vs-execution-boundary:** This module is canonical. The
//! actual embedding-generation work executes through the modality
//! provider in `crate::runtime::modality` (an execution-boundary trait).

#[cfg(feature = "server")]
use axum::{
    extract::State,
    Json,
};
#[cfg(feature = "server")]
use serde_json::{json, Value};

#[cfg(feature = "server")]
use super::AppState;

// =====================================================================
//  Embedding generation
// =====================================================================

/// POST /v1/embeddings - generate text embeddings.
#[cfg(all(feature = "server", not(feature = "prism-backend")))]
pub async fn generate_embeddings(
    State(_server): State<AppState>,
    Json(body): Json<Value>,
) -> Json<Value> {
    let model = body
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("default");
    Json(json!({
        "status": "error",
        "code": "runtime_unavailable",
        "model": model,
        "message": "text embeddings require an admitted live model runtime; enable prism-backend and register a CImage"
    }))
}

/// POST /v1/embeddings - generate text embeddings (compute-core).
#[cfg(all(feature = "server", feature = "prism-backend"))]
pub async fn generate_embeddings(
    State(server): State<AppState>,
    Json(body): Json<Value>,
) -> Json<Value> {
    let model_path = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("default");
    let text = body.get("text").and_then(|v| v.as_str()).unwrap_or("");
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
