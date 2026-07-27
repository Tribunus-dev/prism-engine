//! Modality dispatch — image, audio, video, embeddings, and multimodal routes.
//!
//! **Single authority:** This module owns the canonical modality-routing
//! surface: the `POST /v1/images/generate`, `POST /v1/audio/speech`,
//! `POST /v1/video/generate`, `POST /v1/embeddings`, and
//! `POST /v1/multimodal/generate` HTTP handlers, the multimodal-request
//! plan resolver (`resolve_multimodal_request`), the file-kind / manifest
//! validators, the vision matmul provider (canonical; the actual Metal
//! kernel dispatch is a typed port in `crate::engine::metal`), and the
//! `capture_live_media` envelope.
//!
//! **Canonical-vs-execution-boundary:** All types and functions in this
//! file are canonical. The Metal matmul kernel that
//! `make_vision_matmul_provider` dispatches to when the `metal-dispatch`
//! feature is enabled is execution-boundary; the provider itself is the
//! typed port interface that bridges the canonical plan resolver to the
//! engine's `crate::engine::metal::dispatch_fp16_matmul`.

#[cfg(feature = "server")]
use axum::{
    extract::{Path, State},
    Json,
};
#[cfg(feature = "server")]
use serde_json::{json, Value};

#[cfg(feature = "server")]
use prism_multimodal::capture::admit_live_source;
#[cfg(feature = "server")]
use prism_multimodal::media::{resolve_egress, resolve_ingress, MediaDescriptor, MediaSource};

#[cfg(feature = "server")]
use crate::runtime::PrismInferenceServer;

#[cfg(feature = "server")]
type AppState = std::sync::Arc<PrismInferenceServer>;

// =====================================================================
//  Multimodal request validation helpers
// =====================================================================

#[cfg(feature = "server")]
#[derive(Debug, serde::Deserialize)]
struct MultimodalMediaRequest {
    /// Namespaced model entry in the owning CImage manifest.
    #[serde(default)]
    model_id: Option<String>,
    source: MediaSource,
    descriptor: MediaDescriptor,
    /// Optional inline payload for clients that already captured/materialized
    /// the frame. File and live-device sources may omit this field.
    #[serde(default)]
    payload: Vec<u8>,
}

#[cfg(feature = "server")]
#[derive(Debug, serde::Deserialize)]
struct MultimodalHttpRequest {
    #[serde(default)]
    text: String,
    #[serde(default)]
    model_id: Option<String>,
    #[serde(default)]
    media: Vec<MultimodalMediaRequest>,
    #[serde(default)]
    mode: Option<prism_multimodal::media::MediaSessionMode>,
    #[serde(default)]
    output: Option<MediaDescriptor>,
    /// Optional local path for materializing the imported media output.
    #[serde(default)]
    output_path: Option<String>,
}

#[cfg(feature = "server")]
fn resolve_multimodal_request(body: Value) -> Result<Value, String> {
    let request: MultimodalHttpRequest = serde_json::from_value(body.clone())
        .map_err(|error| format!("invalid multimodal request: {error}"))?;
    let mode = request
        .mode
        .unwrap_or(prism_multimodal::media::MediaSessionMode::Realtime);
    if request.model_id.as_deref().is_some_and(str::is_empty) {
        return Err("model_id must not be empty".into());
    }
    let mut routes = Vec::with_capacity(request.media.len());
    for item in request.media {
        if item.model_id.as_deref().is_some_and(str::is_empty) {
            return Err("media model_id must not be empty".into());
        }
        validate_file_media_kind(&item.source, item.descriptor.kind)?;
        admit_live_source(&item.source)?;
        validate_inline_payload(&item.descriptor, &item.payload)?;
        let route =
            resolve_ingress(&item.source, &item.descriptor).map_err(|error| error.to_string())?;
        routes.push(json!({
            "model_id": item.model_id.clone().or_else(|| request.model_id.clone()),
            "source": item.source,
            "kind": item.descriptor.kind,
            "route": route.route,
            "memory": route.memory,
            "accelerators": route.accelerators,
            "zero_copy": route.zero_copy,
            "payload_bytes": item.payload.len(),
            "materialized": !item.payload.is_empty(),
        }));
    }
    let output = request
        .output
        .as_ref()
        .map(|descriptor| {
            resolve_egress(descriptor.kind, descriptor.format)
                .map(|route| {
                    json!({
                        "kind": descriptor.kind,
                        "format": descriptor.format,
                        "route": route.route,
                        "memory": route.memory,
                        "accelerators": route.accelerators,
                        "zero_copy": route.zero_copy,
                    })
                })
                .map_err(|error| error.to_string())
        })
        .transpose()?;
    if request.output_path.is_some() && request.output.is_none() {
        return Err("output descriptor is required with output_path".into());
    }
    if request.output.is_some() && request.output_path.is_none() {
        return Err("output_path is required when output is requested".into());
    }
    Ok(
        json!({ "text": request.text, "model_id": request.model_id, "mode": mode, "routes": routes, "output": output }),
    )
}

#[cfg(feature = "server")]
fn validate_file_media_kind(
    source: &MediaSource,
    kind: prism_multimodal::media::MediaKind,
) -> Result<(), String> {
    let prism_multimodal::media::MediaSource::File { path } = source else {
        return Ok(());
    };
    let extension = std::path::Path::new(path)
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    let valid = match kind {
        prism_multimodal::media::MediaKind::Image => matches!(
            extension.as_str(),
            "png" | "jpg" | "jpeg" | "webp" | "bmp" | "rgba"
        ),
        prism_multimodal::media::MediaKind::Audio => matches!(
            extension.as_str(),
            "wav" | "wave" | "mp3" | "m4a" | "aac" | "flac" | "pcm"
        ),
        prism_multimodal::media::MediaKind::Video => matches!(
            extension.as_str(),
            "mp4" | "mov" | "m4v" | "mkv" | "webm" | "avi"
        ),
    };
    if valid {
        Ok(())
    } else {
        Err(format!(
            "file extension .{extension} is incompatible with {:?} media",
            kind
        ))
    }
}

#[cfg(feature = "server")]
fn manifest_supports_media_kind(
    manifest: &prism_ecs_compile::MultiModelManifest,
    kind: prism_multimodal::media::MediaKind,
) -> bool {
    manifest.models.values().any(|model| {
        matches!(
            (kind, model.modality.clone()),
            (
                prism_multimodal::media::MediaKind::Audio,
                prism_ecs_compile::ModelModality::Audio
                    | prism_ecs_compile::ModelModality::Multimodal
            ) | (
                prism_multimodal::media::MediaKind::Image,
                prism_ecs_compile::ModelModality::Image
                    | prism_ecs_compile::ModelModality::Vision
                    | prism_ecs_compile::ModelModality::Multimodal
            ) | (
                prism_multimodal::media::MediaKind::Video,
                prism_ecs_compile::ModelModality::Video
                    | prism_ecs_compile::ModelModality::Multimodal
            )
        )
    })
}

#[cfg(feature = "server")]
fn manifest_supports_output_kind(
    manifest: &prism_ecs_compile::MultiModelManifest,
    kind: prism_multimodal::media::MediaKind,
) -> bool {
    manifest.models.values().any(|model| {
        model.outputs.iter().any(|output| {
            matches!(
                (kind, output.kind.clone()),
                (
                    prism_multimodal::media::MediaKind::Audio,
                    prism_ecs_compile::ModelIoKind::AudioPcm
                ) | (
                    prism_multimodal::media::MediaKind::Image,
                    prism_ecs_compile::ModelIoKind::ImageRgba
                ) | (
                    prism_multimodal::media::MediaKind::Video,
                    prism_ecs_compile::ModelIoKind::VideoFrame
                )
            )
        })
    })
}

#[cfg(feature = "server")]
fn validate_plan_models(server: &AppState, plan: &Value) -> Result<(), String> {
    if let Some(model_id) = plan.get("model_id").and_then(Value::as_str) {
        server.model_path(model_id)?;
        if !server.has_live_runtime(model_id)? {
            return Err(format!("model_id has no live runtime: {model_id}"));
        }
    }
    if let Some(routes) = plan.get("routes").and_then(Value::as_array) {
        for route in routes {
            if let Some(model_id) = route.get("model_id").and_then(Value::as_str) {
                if !server.has_live_runtime(model_id)? {
                    if server.manifest_for_namespace(model_id)?.is_some() {
                        return Err(format!(
                            "namespaced model_id {model_id} is declared but has no registered live runtime"
                        ));
                    }
                    return Err(format!("model_id is not registered: {model_id}"));
                }
                if let Some(manifest) = server.manifest_containing_namespace(model_id)? {
                    let kind: prism_multimodal::media::MediaKind = serde_json::from_value(
                        route
                            .get("kind")
                            .cloned()
                            .ok_or_else(|| "media route is missing kind".to_string())?,
                    )
                    .map_err(|error| format!("invalid media route kind: {error}"))?;
                    if !manifest_supports_media_kind(&manifest, kind) {
                        return Err(format!(
                            "model_id {model_id} manifest declares no {:?} input capability",
                            kind
                        ));
                    }
                }
            }
        }
    }
    if let (Some(model_id), Some(output)) = (
        plan.get("model_id").and_then(Value::as_str),
        plan.get("output"),
    ) {
        if let Some(manifest) = server.manifest_containing_namespace(model_id)? {
            let kind: prism_multimodal::media::MediaKind = serde_json::from_value(
                output
                    .get("kind")
                    .cloned()
                    .ok_or_else(|| "media output is missing kind".to_string())?,
            )
            .map_err(|error| format!("invalid media output kind: {error}"))?;
            if !manifest_supports_output_kind(&manifest, kind) {
                return Err(format!(
                    "model_id {model_id} manifest declares no {:?} output capability",
                    kind
                ));
            }
        }
    }
    Ok(())
}

#[cfg(feature = "server")]
fn vision_config_for_model(
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

#[cfg(feature = "server")]
fn capture_live_media(
    source: prism_multimodal::media::MediaSource,
    descriptor: prism_multimodal::media::MediaDescriptor,
    model_id: &str,
    mode: prism_multimodal::media::MediaSessionMode,
) -> Result<Vec<prism_multimodal::media::MediaPacket>, String> {
    use prism_multimodal::capture::CaptureCoordinator;
    use prism_multimodal::media::MediaKind;
    let media_kind = descriptor.kind;
    let is_camera = matches!(
        &source,
        prism_multimodal::media::MediaSource::SystemCamera { .. }
            | prism_multimodal::media::MediaSource::ConnectedIphoneCamera { .. }
    );
    let batch_size = descriptor.batch_size.max(1) as usize;
    let mut coordinator = if is_camera {
        CaptureCoordinator::start_zero_copy_camera_for_model(model_id, source, descriptor, mode)?
    } else {
        CaptureCoordinator::start_for_model(model_id, source, descriptor, mode)?
    };
    let mut packets = Vec::with_capacity(batch_size);
    for _ in 0..200 {
        if let Some(mut packet) = coordinator.poll()? {
            packet.model_id = Some(model_id.to_string());
            packets.push(packet);
            if packets.len() >= batch_size {
                break;
            }
        } else {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }
    if packets.is_empty() {
        return Err(format!(
            "live {:?} capture produced no packets before timeout",
            media_kind
        ));
    }
    if matches!(
        media_kind,
        MediaKind::Audio | MediaKind::Image | MediaKind::Video
    ) {
        Ok(packets)
    } else {
        Err("unsupported live media kind".into())
    }
}

/// Canonical vision-encoder matmul provider.
///
/// **Typed port interface** to the Metal matmul kernel. The closure falls
/// back to a CPU implementation; when the `metal-dispatch` feature is
/// enabled it forwards to `crate::engine::metal::dispatch_fp16_matmul`,
/// which is the execution-boundary Metal kernel.
#[cfg(feature = "server")]
fn make_vision_matmul_provider() -> prism_multimodal::multimodal::vision_encoder::MatmulProvider {
    prism_multimodal::multimodal::vision_encoder::MatmulProvider {
        matmul: Box::new(|input, weight, dim_m, dim_n| {
            let m = dim_m as usize;
            let n = dim_n as usize;
            if input.len() != n || weight.len() < n * m {
                return Err("vision matmul dimension mismatch".into());
            }
            #[cfg(all(feature = "metal-dispatch", target_os = "macos"))]
            {
                let fp16_weights = weight
                    .iter()
                    .flat_map(|value| half::f16::from_f32(*value).to_le_bytes())
                    .collect::<Vec<_>>();
                if let Ok(output) = crate::engine::metal::dispatch_fp16_matmul(
                    "vision_encoder",
                    input,
                    &fp16_weights,
                    dim_m,
                    dim_n,
                ) {
                    return Ok(output);
                }
            }
            Ok((0..m)
                .map(|j| (0..n).map(|i| input[i] * weight[j * n + i]).sum())
                .collect())
        }),
    }
}

#[cfg(feature = "server")]
fn execute_multimodal_backend(server: &AppState, body: Value) -> Result<Value, String> {
    let request: MultimodalHttpRequest = serde_json::from_value(body.clone())
        .map_err(|error| format!("invalid multimodal request: {error}"))?;
    let primary_model_id = request.model_id.clone();

    // A CImage may assign different inputs to specialized model entries. Run
    // each namespace through its own live runtime and return named auxiliary
    // results to the final model orchestration layer. Do not silently feed a
    // packet compiled for one model into another model's graph.
    let mut model_groups = std::collections::BTreeMap::<String, Vec<usize>>::new();
    for (index, item) in request.media.iter().enumerate() {
        let model = item
            .model_id
            .clone()
            .or_else(|| primary_model_id.clone())
            .ok_or_else(|| format!("media[{index}] requires model_id when request has none"))?;
        model_groups.entry(model).or_default().push(index);
    }
    if model_groups.len() > 1 {
        return Err(format!(
            "multi-model fusion across {} specialised runtimes is not yet wired",
            model_groups.len()
        ));
    }
    let _ = model_groups
        .into_iter()
        .next()
        .ok_or_else(|| "no media items in multimodal request".to_string())?;
    Ok(json!({ "status": "noop", "note": "compute-core multimodal execute is a noop canonical pass" }))
}

#[cfg(feature = "server")]
fn validate_inline_payload(descriptor: &MediaDescriptor, payload: &[u8]) -> Result<(), String> {
    if payload.is_empty() {
        return Ok(());
    }
    let expected = match descriptor.format {
        prism_multimodal::media::PixelFormat::Rgba8
        | prism_multimodal::media::PixelFormat::Bgra8 => descriptor
            .width
            .zip(descriptor.height)
            .map(|(w, h)| w as usize * h as usize * 4),
        prism_multimodal::media::PixelFormat::Gray8 => descriptor
            .width
            .zip(descriptor.height)
            .map(|(w, h)| w as usize * h as usize),
        prism_multimodal::media::PixelFormat::F32Pcm => Some(payload.len()),
        prism_multimodal::media::PixelFormat::F32 => Some(payload.len()),
        prism_multimodal::media::PixelFormat::S16Pcm => Some(payload.len()),
        prism_multimodal::media::PixelFormat::Nv12 => descriptor
            .width
            .zip(descriptor.height)
            .map(|(w, h)| w as usize * h as usize * 3 / 2),
    };
    if let Some(expected) = expected {
        if matches!(
            descriptor.format,
            prism_multimodal::media::PixelFormat::F32Pcm
        ) && payload.len() % 4 != 0
        {
            return Err("inline F32Pcm payload must be 4-byte aligned".into());
        }
        if matches!(
            descriptor.format,
            prism_multimodal::media::PixelFormat::S16Pcm
        ) && payload.len() % 2 != 0
        {
            return Err("inline S16Pcm payload must be 2-byte aligned".into());
        }
        if !matches!(
            descriptor.format,
            prism_multimodal::media::PixelFormat::F32Pcm
                | prism_multimodal::media::PixelFormat::S16Pcm
        ) && payload.len() != expected
        {
            return Err(format!(
                "inline payload has {} bytes, expected {expected}",
                payload.len()
            ));
        }
    }
    Ok(())
}

// =====================================================================
//  HTTP handlers — image, audio, video, embeddings, multimodal
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

/// POST /v1/multimodal/generate - multimodal (vision+text) generation.
#[cfg(all(feature = "server", not(feature = "prism-backend")))]
pub async fn generate_multimodal(
    State(server): State<AppState>,
    Json(body): Json<Value>,
) -> Json<Value> {
    match resolve_multimodal_request(body.clone()) {
        Ok(plan) => match validate_plan_models(&server, &plan)
            .and_then(|_| execute_multimodal_backend(&server, body))
        {
            Ok(result) => Json(result),
            Err(message) => Json(json!({ "status": "error", "message": message })),
        },
        Err(message) => Json(json!({ "status": "error", "message": message })),
    }
}

/// POST /v1/multimodal/generate - multimodal generation (compute-core, stub).
#[cfg(all(feature = "server", feature = "prism-backend"))]
pub async fn generate_multimodal(
    State(server): State<AppState>,
    Json(body): Json<Value>,
) -> Json<Value> {
    let execution_body = body.clone();
    match resolve_multimodal_request(body) {
        Ok(plan) => match validate_plan_models(&server, &plan) {
            Ok(()) => match execute_multimodal_backend(&server, execution_body) {
                Ok(execution) => Json(
                    json!({ "status": "executed", "media_plan": plan, "execution": execution }),
                ),
                Err(message) => {
                    Json(json!({ "status": "error", "message": message, "media_plan": plan }))
                }
            },
            Err(message) => Json(json!({ "status": "error", "message": message })),
        },
        Err(message) => Json(json!({ "status": "error", "message": message })),
    }
}

// =====================================================================
//  Tests
// =====================================================================

#[cfg(all(test, feature = "server"))]
mod multimodal_plan_tests {
    use super::*;
    use prism_multimodal::media::{MediaKind, PixelFormat};

    #[test]
    fn vision_provider_preserves_gemv_contract() {
        let provider = make_vision_matmul_provider();
        let output = (provider.matmul)(&[2.0, 3.0], &[4.0, 5.0, 6.0, 7.0], 2, 2).unwrap();
        assert_eq!(output, vec![23.0, 33.0]);
    }

    #[test]
    fn plans_file_input_and_video_toolbox_output_together() {
        let body = json!({
            "text": "describe this",
            "media": [{
                "source": {"kind": "file", "path": "clip.mp4"},
                "descriptor": {"kind": "video", "format": "nv12", "width": 1920, "height": 1080, "sample_rate": null, "channels": null, "batch_size": 4}
            }],
            "output": {"kind": "video", "format": "nv12", "width": 1920, "height": 1080, "sample_rate": null, "channels": null, "batch_size": 1}
            ,"output_path": "/tmp/prism-test-output.mp4"
        });
        let plan = resolve_multimodal_request(body).unwrap();
        assert_eq!(plan["routes"][0]["route"], "video_toolbox_decode");
        assert_eq!(plan["output"]["route"], "video_toolbox_encode");
        assert_eq!(
            plan["routes"][0]["kind"],
            serde_json::to_value(MediaKind::Video).unwrap()
        );
        assert_eq!(
            plan["output"]["format"],
            serde_json::to_value(PixelFormat::Nv12).unwrap()
        );
        assert_eq!(
            plan["mode"],
            serde_json::to_value(prism_multimodal::media::MediaSessionMode::Realtime).unwrap()
        );
    }

    #[test]
    fn preserves_explicit_batched_capture_mode() {
        let body = json!({
            "mode": "batched",
            "media": [{
                "source": {"kind": "file", "path": "image.png"},
                "descriptor": {"kind": "image", "format": "rgba8", "width": 1, "height": 1, "batch_size": 4},
                "payload": [0, 0, 0, 255]
            }]
        });
        let plan = resolve_multimodal_request(body).unwrap();
        assert_eq!(
            plan["mode"],
            serde_json::to_value(prism_multimodal::media::MediaSessionMode::Batched).unwrap()
        );
    }

    #[test]
    fn rejects_unsupported_file_kind_before_dispatch() {
        let body = json!({
            "media": [{
                "source": {"kind": "file", "path": "clip.mp4"},
                "descriptor": {"kind": "audio", "format": "f32_pcm", "width": null, "height": null, "sample_rate": 48000, "channels": 2, "batch_size": 1}
            }]
        });
        assert!(resolve_multimodal_request(body).is_err());
    }

    #[test]
    fn rejects_misaligned_inline_audio() {
        let body = json!({
            "media": [{
                "source": {"kind": "file", "path": "audio.wav"},
                "descriptor": {"kind": "audio", "format": "f32_pcm", "sample_rate": 16000, "channels": 1, "batch_size": 1},
                "payload": [1, 2, 3]
            }]
        });
        assert!(resolve_multimodal_request(body).is_err());
    }

    #[test]
    fn rejects_wrong_inline_rgba_size() {
        let body = json!({
            "media": [{
                "source": {"kind": "file", "path": "image.rgba"},
                "descriptor": {"kind": "image", "format": "rgba8", "width": 2, "height": 2, "batch_size": 1},
                "payload": [0, 0, 0]
            }]
        });
        assert!(resolve_multimodal_request(body).is_err());
    }

    #[test]
    fn rejects_output_without_materialization_path() {
        let body = json!({
            "media": [{
                "source": {"kind": "file", "path": "image.png"},
                "descriptor": {"kind": "image", "format": "rgba8", "width": 2, "height": 2, "batch_size": 1},
                "payload": [0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255]
            }],
            "output": {"kind": "image", "format": "rgba8", "width": 2, "height": 2, "batch_size": 1}
        });
        assert!(resolve_multimodal_request(body).is_err());
    }

    #[test]
    fn rejects_output_path_without_descriptor() {
        let body = json!({"output_path": "/tmp/prism-output.rgba"});
        assert!(resolve_multimodal_request(body).is_err());
    }

    #[test]
    fn preserves_specialist_model_namespaces_in_one_plan() {
        let body = json!({
            "model_id": "fusion",
            "media": [
                {
                    "model_id": "vision",
                    "source": {"kind": "file", "path": "vision.png"},
                    "descriptor": {"kind": "image", "format": "rgba8", "width": 1, "height": 1, "batch_size": 1},
                    "payload": [0, 0, 0, 255]
                },
                {
                    "model_id": "audio",
                    "source": {"kind": "file", "path": "audio.wav"},
                    "descriptor": {"kind": "audio", "format": "f32_pcm", "sample_rate": 16000, "channels": 1, "batch_size": 1},
                    "payload": [0, 0, 0, 0]
                }
            ]
        });
        let plan = resolve_multimodal_request(body).unwrap();
        assert_eq!(plan["routes"][0]["model_id"], "vision");
        assert_eq!(plan["routes"][1]["model_id"], "audio");
        assert_eq!(plan["model_id"], "fusion");
    }
}
