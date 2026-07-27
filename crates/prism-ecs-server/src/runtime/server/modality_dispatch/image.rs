//! Image and video generation — `POST /v1/images/generate`,
//! `POST /v1/video/generate`.
//!
//! **Single authority:** This sub-module owns the canonical HTTP handlers
//! for text-to-image and text-to-video generation, the vision-encoder
//! configuration resolver, and the surface glue needed to admit a vision
//! model (`vision_config_for_model`). The matmul port itself lives at the
//! directory level (`super::make_vision_matmul_provider`) so it stays
//! discoverable from `modality_dispatch::*`.
//!
//! **Canonical-vs-execution-boundary:** All types and functions here are
//! canonical. The Metal matmul kernel dispatch (gated by
//! `metal-dispatch` + `target_os = "macos"`) inside the typed port is
//! execution-boundary.

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
//  Image generation
// =====================================================================

/// POST /v1/images/generate - generate an image from a text prompt.
#[cfg(all(feature = "server", not(feature = "prism-backend")))]
pub async fn generate_image(
    State(_server): State<AppState>,
    Json(body): Json<Value>,
) -> Json<Value> {
    #[cfg(feature = "generation-image")]
    {
        use crate::runtime::modality::ModalityProvider;
        let model = body
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or("default");
        let prompt = body.get("prompt").and_then(Value::as_str).unwrap_or("");
        let width = body.get("width").and_then(Value::as_u64).unwrap_or(1024) as u32;
        let height = body.get("height").and_then(Value::as_u64).unwrap_or(1024) as u32;
        let request = crate::runtime::modality::ImageGenerationRequest::new(
            prompt.to_string(),
            width,
            height,
        );
        return match _server.generate_image(model, request) {
            Ok(result) => Json(json!({
                "status": "ok",
                "width": result.image.width,
                "height": result.image.height,
                "format": format!("{:?}", result.image.format),
                "digest": result.image.digest.0,
                "receipt": result.receipt,
            })),
            Err(error) => Json(json!({"status": "error", "message": error.to_string()})),
        };
    }
    #[cfg(not(feature = "generation-image"))]
    {
        let _ = body;
        Json(json!({"status": "error", "message": "feature not enabled: generation-image"}))
    }
}

/// POST /v1/images/generate - generate an image (compute-core).
#[cfg(all(
    feature = "server",
    feature = "prism-backend",
    any(feature = "generation-image", feature = "generation-diffusion")
))]
pub async fn generate_image(
    State(server): State<AppState>,
    Json(body): Json<Value>,
) -> Json<Value> {
    let model_path = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("default");
    let prompt = body.get("prompt").and_then(|v| v.as_str()).unwrap_or("");
    let width = body.get("width").and_then(|v| v.as_u64()).unwrap_or(1024) as u32;
    let height = body.get("height").and_then(|v| v.as_u64()).unwrap_or(1024) as u32;
    let request =
        crate::runtime::modality::ImageGenerationRequest::new(prompt.to_string(), width, height);
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
            "message": format!("{e}"),
        })),
    }
}

/// POST /v1/images/generate - feature not enabled stub.
#[cfg(all(
    feature = "server",
    feature = "prism-backend",
    not(any(feature = "generation-image", feature = "generation-diffusion"))
))]
pub async fn generate_image(
    State(_server): State<AppState>,
    Json(body): Json<Value>,
) -> Json<Value> {
    let _ = body;
    Json(json!({
        "status": "error",
        "message": "feature not enabled: generation-image or generation-diffusion"
    }))
}

// =====================================================================
//  Video generation
// =====================================================================

/// POST /v1/video/generate - generate a video from a text prompt.
#[cfg(all(feature = "server", not(feature = "prism-backend")))]
pub async fn generate_video(
    State(_server): State<AppState>,
    Json(body): Json<Value>,
) -> Json<Value> {
    #[cfg(feature = "generation-video")]
    {
        use crate::runtime::modality::ModalityProvider;
        let model = body
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or("default");
        let prompt = body.get("prompt").and_then(Value::as_str).unwrap_or("");
        let mut params = crate::runtime::modality::VideoParams::default();
        if let Some(frames) = body.get("num_frames").and_then(Value::as_u64) {
            params.num_frames = frames as u32;
        }
        if let Some(fps) = body.get("fps").and_then(Value::as_f64) {
            params.fps = fps as f32;
        }
        return match _server.generate_video(model, prompt, params) {
            Ok(receipt) => Json(json!({
                "status": "ok",
                "num_frames": receipt.num_frames,
                "compute_ms": receipt.compute_ms,
            })),
            Err(error) => Json(json!({"status": "error", "message": error.to_string()})),
        };
    }
    #[cfg(not(feature = "generation-video"))]
    {
        let _ = body;
        Json(json!({"status": "error", "message": "feature not enabled: generation-video"}))
    }
}

/// POST /v1/video/generate - generate a video (compute-core).
#[cfg(all(
    feature = "server",
    feature = "prism-backend",
    feature = "generation-video"
))]
pub async fn generate_video(
    State(server): State<AppState>,
    Json(body): Json<Value>,
) -> Json<Value> {
    let model_path = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("default");
    let prompt = body.get("prompt").and_then(|v| v.as_str()).unwrap_or("");
    let mut params = crate::runtime::modality::VideoParams::default();
    if let Some(v) = body.get("num_frames").and_then(|v| v.as_u64()) {
        params.num_frames = v as u32;
    }
    if let Some(v) = body.get("fps").and_then(|v| v.as_f64()) {
        params.fps = v as f32;
    }
    if let Some(v) = body.get("seed").and_then(|v| v.as_u64()) {
        params.seed = v;
    }
    match server.generate_video(model_path, prompt, params) {
        Ok(receipt) => Json(json!({
            "status": "ok",
            "num_frames": receipt.num_frames,
            "compute_ms": receipt.compute_ms,
        })),
        Err(e) => Json(json!({
            "status": "error",
            "message": format!("{e}"),
        })),
    }
}

/// POST /v1/video/generate - feature not enabled stub.
#[cfg(all(
    feature = "server",
    feature = "prism-backend",
    not(feature = "generation-video")
))]
pub async fn generate_video(
    State(_server): State<AppState>,
    Json(body): Json<Value>,
) -> Json<Value> {
    let _ = body;
    Json(json!({
        "status": "error",
        "message": "feature not enabled: generation-video"
    }))
}

// =====================================================================
//  Vision-encoder support
// =====================================================================

/// Build a `VisionEncoderConfig` for a given model namespace.
///
/// The hidden dimension is recovered from the manifest's `Embedding`
/// output shape (1-D, positive length); on miss we fall back to the
/// CLIP-ViT-L default of 1024. The remaining fields are CLIP-ViT-L
/// constants — when additional architectures are admitted, the lookup
/// must be extended here.
#[cfg(feature = "server")]
pub(crate) fn vision_config_for_model(
    server: &AppState,
    model_id: &str,
) -> Result<prism_multimodal::multimodal::vision_encoder::VisionEncoderConfig, String> {
    let hidden_dim = server
        .manifest_for_namespace(model_id)?
        .and_then(|model| {
            model
                .outputs
                .iter()
                .find(|output| output.kind == prism_ecs_compile::ModelIoKind::Embedding)
                .and_then(|output| {
                    (output.shape.len() == 1 && output.shape[0] > 0).then_some(output.shape[0])
                })
        })
        .unwrap_or(1024);
    let hidden_dim = u32::try_from(hidden_dim)
        .map_err(|_| format!("vision embedding width is too large for model {model_id:?}"))?;
    Ok(
        prism_multimodal::multimodal::vision_encoder::VisionEncoderConfig {
            arch: prism_multimodal::multimodal::vision_encoder::VisionArch::ClipVitL,
            input_size: (224, 224),
            patch_size: 14,
            num_layers: 24,
            hidden_dim,
            num_heads: 16,
        },
    )
}

// =====================================================================
//  Tests
// =====================================================================

#[cfg(all(test, feature = "server"))]
mod image_tests {
    use super::super::make_vision_matmul_provider;

    #[test]
    fn vision_provider_preserves_gemv_contract() {
        let provider = make_vision_matmul_provider();
        let output = (provider.matmul)(&[2.0, 3.0], &[4.0, 5.0, 6.0, 7.0], 2, 2).unwrap();
        assert_eq!(output, vec![23.0, 33.0]);
    }
}
