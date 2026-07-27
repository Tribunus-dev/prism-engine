//! Text-to-speech — `POST /v1/audio/speech`.
//!
//! **Single authority:** This sub-module owns the canonical HTTP handlers
//! for text-to-speech generation. The `AudioParams` request shape is
//! owned by `crate::runtime::modality`; this module is a thin canonical
//! adapter that parses the request body, calls the modality provider, and
//! formats the receipt into the response JSON.
//!
//! **Canonical-vs-execution-boundary:** All types and functions here are
//! canonical. The actual audio-synthesis work executes through the
//! modality provider in `crate::runtime::modality` (an execution-boundary
//! trait), which forwards to the engine's audio backend.

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
//  Audio generation
// =====================================================================

/// POST /v1/audio/speech - generate speech from text.
#[cfg(all(feature = "server", not(feature = "prism-backend")))]
pub async fn generate_audio(
    State(_server): State<AppState>,
    Json(body): Json<Value>,
) -> Json<Value> {
    #[cfg(feature = "generation-audio")]
    {
        use crate::runtime::modality::ModalityProvider;
        let model = body
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or("default");
        let text = body.get("text").and_then(Value::as_str).unwrap_or("");
        let mut params = crate::runtime::modality::AudioParams::default();
        params.voice = body
            .get("voice")
            .and_then(Value::as_str)
            .map(str::to_string);
        return match _server.generate_audio(model, text, params) {
            Ok(receipt) => Json(json!({
                "status": "ok",
                "sample_rate": receipt.sample_rate,
                "num_samples": receipt.num_samples,
                "compute_ms": receipt.compute_ms,
                "output_digest": receipt.output_digest,
            })),
            Err(error) => Json(json!({"status": "error", "message": error.to_string()})),
        };
    }
    #[cfg(not(feature = "generation-audio"))]
    {
        let _ = body;
        Json(json!({"status": "error", "message": "feature not enabled: generation-audio"}))
    }
}

/// POST /v1/audio/speech - generate speech (compute-core).
#[cfg(all(
    feature = "server",
    feature = "prism-backend",
    feature = "generation-audio"
))]
pub async fn generate_audio(
    State(server): State<AppState>,
    Json(body): Json<Value>,
) -> Json<Value> {
    let model_path = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("default");
    let text = body.get("text").and_then(|v| v.as_str()).unwrap_or("");
    let mut params = crate::runtime::modality::AudioParams::default();
    params.voice = body.get("voice").and_then(|v| v.as_str()).map(String::from);
    match server.generate_audio(model_path, text, params) {
        Ok(receipt) => Json(json!({
            "status": "ok",
            "sample_rate": receipt.sample_rate,
            "num_samples": receipt.num_samples,
            "compute_ms": receipt.compute_ms,
            "output_digest": receipt.output_digest,
        })),
        Err(e) => Json(json!({
            "status": "error",
            "message": format!("{e}"),
        })),
    }
}

/// POST /v1/audio/speech - feature not enabled stub.
#[cfg(all(
    feature = "server",
    feature = "prism-backend",
    not(feature = "generation-audio")
))]
pub async fn generate_audio(
    State(_server): State<AppState>,
    Json(body): Json<Value>,
) -> Json<Value> {
    let _ = body;
    Json(json!({
        "status": "error",
        "message": "feature not enabled: generation-audio"
    }))
}
