// -- Prism LLM Inference - HTTP API Server -------------------------------
//
// Axum-based HTTP server exposing the Prism LLM and multimodal inference API
// over HTTP. CPU/reference and heterogeneous runtime paths are shared across
// feature configurations; modality generators remain explicitly gated by
// their provider features.

use std::sync::{Arc, OnceLock};

#[cfg(feature = "server")]
use axum::{
    extract::{Path, State},
    routing::{delete, get, post},
    Json, Router,
};
#[cfg(feature = "server")]
use prism_multimodal::capture::admit_live_source;
#[cfg(feature = "server")]
use prism_multimodal::media::{resolve_egress, resolve_ingress, MediaDescriptor, MediaSource};
#[cfg(feature = "server")]
use serde_json::{json, Value};

use super::PrismInferenceServer;
use crate::runtime::server_types::SamplingConfig;

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
    use crate::runtime::server::PrefillDecodeRuntime;
    use prism_multimodal::media::MediaMemory;
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
        let primary = primary_model_id.as_deref().ok_or_else(|| {
            "mixed-model multimodal execution requires a primary model_id".to_string()
        })?;
        let primary_manifest = server
            .model_manifest(primary)?
            .ok_or_else(|| format!("primary model has no multimodal manifest: {primary}"))?;
        let source_media = body
            .get("media")
            .and_then(Value::as_array)
            .ok_or_else(|| "multimodal media must be an array".to_string())?;
        let mut auxiliary_outputs = Vec::new();
        for (group_model, indices) in model_groups {
            if group_model != primary
                && !primary_manifest.models.get(primary).is_some_and(|model| {
                    model
                        .fusion_inputs
                        .iter()
                        .any(|binding| binding == &group_model)
                })
            {
                return Err(format!(
                    "primary model {primary} declares no fusion input for specialist {group_model}"
                ));
            }
            let media = indices
                .into_iter()
                .map(|index| source_media[index].clone())
                .collect::<Vec<_>>();
            let group_body = json!({
                "text": request.text.clone(),
                "model_id": group_model,
                "media": media,
                "mode": request.mode,
            });
            let result = execute_multimodal_backend(server, group_body)?;
            auxiliary_outputs.push(result);
        }
        let primary_runtime = server.live_runtime(primary)?;
        let primary_tokens = primary_runtime.tokenize(&request.text)?;
        let mut fusion_rows = Vec::new();
        for output in &auxiliary_outputs {
            let source_model = output
                .get("model_id")
                .and_then(Value::as_str)
                .ok_or_else(|| "specialist result is missing model_id".to_string())?;
            if source_model == primary {
                continue;
            }
            let binding = primary_manifest
                .models
                .get(primary)
                .and_then(|model| {
                    model
                        .fusion_inputs
                        .iter()
                        .find(|binding| *binding == source_model)
                })
                .ok_or_else(|| format!("missing fusion binding for {source_model}"))?;
            let feature_rows = output
                .get("feature_outputs")
                .and_then(|outputs| outputs.get(source_model))
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    format!("specialist {source_model} did not return declared output")
                })?;
            let mut row = Vec::new();
            for feature_row in feature_rows {
                let values = feature_row.as_array().ok_or_else(|| {
                    format!("specialist {source_model} returned invalid feature row")
                })?;
                row.extend(
                    values
                        .iter()
                        .map(|value| {
                            value.as_f64().map(|value| value as f32).ok_or_else(|| {
                                format!("specialist {source_model} returned non-numeric feature")
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                );
            }
            let _ = binding;
            fusion_rows.push((source_model.to_string(), row));
        }
        let named_rows = fusion_rows
            .iter()
            .map(|(tensor, row)| (tensor.as_str(), row.as_slice()))
            .collect::<Vec<_>>();
        let fused_logits =
            primary_runtime.run_prefill_conditioned_named_features(&primary_tokens, &named_rows)?;
        return Ok(json!({
            "status": "executed_fused_inputs",
            "model_id": primary,
            "logit_count": fused_logits.len(),
            "model_groups": auxiliary_outputs,
            "fusion": {
                "kind": "named_model_outputs",
                "primary_model_id": primary,
                "preserves_packet_namespace": true,
            },
        }));
    }
    let model_id = request
        .model_id
        .as_deref()
        .ok_or_else(|| "backend multimodal execution requires model_id".to_string())?;
    let runtime = server.live_runtime(model_id)?;
    let prompt_tokens = runtime.tokenize(&request.text)?;
    let mut packets = Vec::new();
    for item in request.media {
        if item.model_id.as_deref().unwrap_or(model_id) != model_id {
            return Err("media packet model_id does not match its execution group".into());
        }
        let source = item.source;
        let descriptor = item.descriptor;
        if item.payload.is_empty() {
            if matches!(
                &source,
                prism_multimodal::media::MediaSource::SystemMicrophone { .. }
                    | prism_multimodal::media::MediaSource::ConnectedIphoneMicrophone { .. }
                    | prism_multimodal::media::MediaSource::SystemCamera { .. }
                    | prism_multimodal::media::MediaSource::ConnectedIphoneCamera { .. }
            ) {
                packets.extend(capture_live_media(
                    source,
                    descriptor,
                    model_id,
                    request
                        .mode
                        .unwrap_or(prism_multimodal::media::MediaSessionMode::Realtime),
                )?);
                continue;
            }
            if let prism_multimodal::media::MediaSource::File { path } = &source {
                match descriptor.kind {
                    prism_multimodal::media::MediaKind::Image => {
                        let imported = prism_multimodal::io::import_image_rgba(path)
                            .map_err(|error| error.to_string())?;
                        let imported = match (descriptor.width, descriptor.height) {
                            (Some(width), Some(height))
                                if width != imported.width || height != imported.height =>
                            {
                                let rgba = prism_multimodal::capture::hardware_resize_rgba(
                                    &imported.rgba,
                                    imported.width,
                                    imported.height,
                                    width,
                                    height,
                                )?;
                                prism_multimodal::io::ImportedImage {
                                    width,
                                    height,
                                    rgba,
                                }
                            }
                            _ => imported,
                        };
                        packets.push(imported.into_packet(
                            source,
                            descriptor.batch_size,
                            Some(model_id.to_string()),
                        ));
                    }
                    prism_multimodal::media::MediaKind::Video => {
                        let width = descriptor
                            .width
                            .ok_or_else(|| "video file descriptor requires width".to_string())?;
                        let height = descriptor
                            .height
                            .ok_or_else(|| "video file descriptor requires height".to_string())?;
                        for frame in prism_multimodal::io::import_video_frames(path)
                            .map_err(|error| error.to_string())?
                        {
                            let payload_bytes = frame.len() as u64;
                            packets.push(prism_multimodal::media::MediaPacket {
                                model_id: Some(model_id.to_string()),
                                source: source.clone(),
                                descriptor: prism_multimodal::media::MediaDescriptor::rgba(
                                    width,
                                    height,
                                    descriptor.batch_size,
                                ),
                                memory: MediaMemory::Cpu,
                                timestamp_ns: 0,
                                sequence: packets.len() as u64,
                                payload_bytes,
                                payload: frame,
                                native_video: None,
                            });
                        }
                    }
                    prism_multimodal::media::MediaKind::Audio => {
                        packets.push(
                            prism_multimodal::io::import_audio_packet(
                                path,
                                source.clone(),
                                descriptor.batch_size,
                                Some(model_id.to_string()),
                            )
                            .map_err(|error| error.to_string())?,
                        );
                    }
                }
                continue;
            }
        }
        let payload_bytes = item.payload.len() as u64;
        packets.push(prism_multimodal::media::MediaPacket {
            model_id: Some(model_id.to_string()),
            source,
            descriptor,
            memory: MediaMemory::Cpu,
            timestamp_ns: 0,
            sequence: packets.len() as u64,
            payload_bytes,
            payload: item.payload,
            native_video: None,
        });
    }
    let has_visual_input = packets.iter().any(|packet| {
        matches!(
            packet.descriptor.kind,
            prism_multimodal::media::MediaKind::Image | prism_multimodal::media::MediaKind::Video
        )
    });
    let feature_rows = if has_visual_input {
        let vision_weights = server.vision_weights(model_id)?;
        let vision_config = vision_config_for_model(server, model_id)?;
        let matmul = make_vision_matmul_provider();
        runtime.extract_packet_feature_rows(
            Some(model_id),
            &packets,
            Some((&vision_config, vision_weights.as_ref(), &matmul)),
        )?
    } else {
        runtime.extract_packet_feature_rows(Some(model_id), &packets, None)?
    };
    let exported_path = match (request.output.as_ref(), request.output_path.as_deref()) {
        (Some(output), Some(path)) => {
            if let Some(packet) = packets.first() {
                if output.kind == packet.descriptor.kind {
                    if output.kind == prism_multimodal::media::MediaKind::Video {
                        let width = packet
                            .descriptor
                            .width
                            .ok_or_else(|| "video output requires width".to_string())?;
                        let height = packet
                            .descriptor
                            .height
                            .ok_or_else(|| "video output requires height".to_string())?;
                        let frames: Vec<Vec<u8>> = packets
                            .iter()
                            .map(|packet| packet.payload.clone())
                            .collect();
                        prism_multimodal::io::export_video_frames(
                            &frames,
                            width,
                            height,
                            30.0,
                            path,
                            prism_video::types::VideoCodec::H264,
                        )
                        .map_err(|error| error.to_string())?;
                    } else {
                        prism_multimodal::io::export_packet(packet, path)
                            .map_err(|error| error.to_string())?;
                    }
                    Some(path.to_string())
                } else {
                    return Err("output kind must match the imported media kind".into());
                }
            } else {
                return Err("cannot export multimodal output without media packets".into());
            }
        }
        (Some(_), None) => {
            return Err("output_path is required when output is requested".into());
        }
        (None, Some(_)) => return Err("output descriptor is required with output_path".into()),
        (None, None) => None,
    };
    let ingress_route_evidence = packets
        .iter()
        .map(|packet| {
            let route = resolve_ingress(&packet.source, &packet.descriptor)
                .map_err(|error| error.to_string())?;
            Ok(json!({
                "source": packet.source,
                "planned_route": format!("{:?}", route.route),
                "planned_memory": format!("{:?}", route.memory),
                "accelerators": route.accelerators.iter().map(|accelerator| format!("{:?}", accelerator)).collect::<Vec<_>>(),
                "zero_copy_planned": route.zero_copy,
                "materialized_memory": format!("{:?}", packet.memory),
                "zero_copy_realized": route.zero_copy && packet.native_video.is_some(),
            }))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let egress_route_evidence = request.output.as_ref().map(|output| {
        resolve_egress(output.kind, output.format)
            .map(|route| {
                json!({
                    "route": format!("{:?}", route.route),
                    "memory": format!("{:?}", route.memory),
                    "accelerators": route.accelerators.iter().map(|accelerator| format!("{:?}", accelerator)).collect::<Vec<_>>(),
                    "zero_copy_planned": route.zero_copy,
                })
            })
            .map_err(|error| error.to_string())
    }).transpose()?;
    let logits = if has_visual_input {
        let vision_weights = server.vision_weights(model_id)?;
        let vision_config = vision_config_for_model(server, model_id)?;
        let matmul = make_vision_matmul_provider();
        runtime.run_prefill_conditioned_packets_with_vision(
            Some(model_id),
            &prompt_tokens,
            &packets,
            &vision_config,
            &vision_weights,
            &matmul,
        )?
    } else {
        runtime.run_prefill_conditioned_packets_for_model(
            Some(model_id),
            &prompt_tokens,
            &packets,
        )?
    };
    let feature_output_name = server.manifest_for_namespace(model_id)?.and_then(|model| {
        model
            .outputs
            .into_iter()
            .find(|output| output.kind == prism_ecs_compile::model_manifest::ModelIoKind::Embedding)
            .map(|output| output.name)
    });
    let feature_outputs = feature_output_name
        .as_ref()
        .map(|name| json!({ name: feature_rows.clone() }));
    Ok(json!({
        "status": "executed",
        "model_id": model_id,
        "prompt_tokens": prompt_tokens.len(),
        "logit_count": logits.len(),
        "media_packets": packets.len(),
        "logits": logits,
        "feature_rows": feature_rows,
        "feature_outputs": feature_outputs,
        "exported_path": exported_path,
        "hardware_route_evidence": {
            "ingress": ingress_route_evidence,
            "egress": egress_route_evidence,
        },
    }))
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

#[cfg(feature = "server")]
use crate::runtime::manifest::SessionId;
#[cfg(all(feature = "server", not(feature = "prism-backend")))]
use crate::runtime::server_types::CreateSessionRequest;
#[cfg(all(feature = "server", feature = "prism-backend"))]
use crate::runtime::server_types::{
    CancellationHandle, CreateSessionRequest, GenerateRequest, RequestId,
};
#[cfg(all(feature = "server", not(feature = "prism-backend")))]
use crate::runtime::server_types::{CancellationHandle, GenerateRequest, RequestId};

// -- Prism-backend imports -------------------------------------------
#[cfg(all(feature = "server", feature = "prism-backend"))]
use crate::runtime::modality::ModalityProvider;

#[cfg(all(feature = "server", feature = "prism-backend"))]
use {
    axum::response::sse::{Event, Sse},
    axum::response::IntoResponse,
    std::convert::Infallible,
    tokio_stream::{wrappers::ReceiverStream, StreamExt},
};

// -- Prefill/Decode Runtime trait ------------------------------------

/// Trait for the autoregressive inference runtime.
///
/// Implemented by the companion `WirePrefillDecodeRuntime` task.  Provides
/// tokenization, prefill (prompt evaluation), single-token decode, sampling,
/// detokenization, and EOS detection.
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

// -- Type alias (axum only)

#[cfg(feature = "server")]
type AppState = Arc<PrismInferenceServer>;

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
                    "tensor_count": inspection.tensor_count,
                }),
            );
        }
    }
    Value::Object(provenance)
}

// -- HttpServer ------------------------------------------------------

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

// -- Router factory (axum only) --------------------------------------

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
        .route("/v1/sessions", post(create_session))
        .route("/v1/sessions/{id}/generate", post(generate))
        .route("/v1/sessions/{id}/cancel", post(cancel))
        .route("/v1/sessions/{id}/compress", post(compress))
        .route("/v1/sessions/{id}/refresh", post(refresh))
        .route("/v1/sessions/{id}", get(get_session))
        .route("/v1/sessions/{id}/receipt", get(get_receipt))
        .route("/v1/sessions/{id}", delete(delete_session))
        .route("/v1/capabilities", get(get_capabilities))
        .route("/v1/telemetry", get(get_telemetry))
        .route("/v1/health", get(health))
        .route("/v1/images/generate", post(generate_image))
        .route("/v1/audio/speech", post(generate_audio))
        .route("/v1/video/generate", post(generate_video))
        .route("/v1/embeddings", post(generate_embeddings))
        .route("/v1/multimodal/generate", post(generate_multimodal))
        .with_state(state)
}

// ====================================================================
//  Handler implementations
// ====================================================================
//
// Some handlers have feature-specific provider variants. Shared session,
// runtime, media, and lifecycle handlers use the same implementation in both
// configurations.

/// Helper: parse a `SessionId` from a path parameter string.
#[cfg(all(feature = "server", feature = "prism-backend"))]
fn parse_session_id(id: &str) -> Result<SessionId, String> {
    let uuid = uuid::Uuid::parse_str(id).map_err(|e| format!("invalid session id '{id}': {e}"))?;
    Ok(SessionId(uuid))
}

// -- generate_stream -------------------------------------------------

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
            let _ = tx.send(Ok(Event::default().data("error:cancelled"))).await;
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
        let _ =
            tx.send(Ok(
                Event::default().data(format!("status:prefill {} tokens", prompt_tokens.len()))
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
                let _ = tx.send(Ok(Event::default().data("error:cancelled"))).await;
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
                let _ = tx.send(Ok(Event::default().data("error:cancelled"))).await;
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

// -- POST /v1/sessions ----------------------------------------------

/// POST /v1/sessions - create a new inference session.
#[cfg(all(feature = "server", not(feature = "prism-backend")))]
async fn create_session(State(server): State<AppState>, Json(body): Json<Value>) -> Json<Value> {
    let request: CreateSessionRequest = match serde_json::from_value(body) {
        Ok(request) => request,
        Err(error) => {
            return Json(json!({"status":"error", "message": format!("invalid request: {error}")}))
        }
    };
    match server.create_session(request) {
        Ok(session_id) => Json(json!({"status":"created", "session_id": session_id})),
        Err(error) => Json(json!({"status":"error", "message": error})),
    }
}

/// POST /v1/sessions - create a new inference session (compute-core).
#[cfg(all(feature = "server", feature = "prism-backend"))]
async fn create_session(State(server): State<AppState>, Json(body): Json<Value>) -> Json<Value> {
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

// -- POST /v1/sessions/{id}/generate ------------------------------

/// POST /v1/sessions/{id}/generate - generate tokens from a session.
#[cfg(all(feature = "server", not(feature = "prism-backend")))]
async fn generate(
    State(server): State<AppState>,
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
    let request = GenerateRequest {
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
        sampling: SamplingConfig {
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
async fn generate(
    State(server): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<Value>,
) -> impl IntoResponse {
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

    let generate_request = GenerateRequest {
        session_id,
        prompt: gen_body.prompt,
        max_new_tokens: gen_body.max_new_tokens,
        sampling: crate::runtime::server_types::SamplingConfig {
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

// -- POST /v1/sessions/{id}/cancel ---------------------------------

/// POST /v1/sessions/{id}/cancel - cancel an inference session.
#[cfg(all(feature = "server", not(feature = "prism-backend")))]
async fn cancel(
    State(server): State<AppState>,
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
async fn cancel(
    State(server): State<AppState>,
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

// -- POST /v1/sessions/{id}/compress -------------------------------

/// POST /v1/sessions/{id}/compress - compress KV cache.
#[cfg(all(feature = "server", not(feature = "prism-backend")))]
async fn compress(State(server): State<AppState>, Path(id): Path<String>) -> Json<Value> {
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
async fn compress(State(server): State<AppState>, Path(id): Path<String>) -> Json<Value> {
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

// -- POST /v1/sessions/{id}/refresh --------------------------------

/// POST /v1/sessions/{id}/refresh - refresh context.
#[cfg(all(feature = "server", not(feature = "prism-backend")))]
async fn refresh(
    State(server): State<AppState>,
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
async fn refresh(
    State(server): State<AppState>,
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

// -- GET /v1/sessions/{id} -----------------------------------------

/// GET /v1/sessions/{id} - get session state.
#[cfg(all(feature = "server", not(feature = "prism-backend")))]
async fn get_session(State(server): State<AppState>, Path(id): Path<String>) -> Json<Value> {
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
async fn get_session(State(server): State<AppState>, Path(id): Path<String>) -> Json<Value> {
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

// -- GET /v1/sessions/{id}/receipt ----------------------------------

/// GET /v1/sessions/{id}/receipt - get session receipt.
#[cfg(all(feature = "server", not(feature = "prism-backend")))]
async fn get_receipt(State(server): State<AppState>, Path(id): Path<String>) -> Json<Value> {
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
        None => Json(
            json!({"status":"not_found","session_id":session_id,"message":"no receipt found for session"}),
        ),
    }
}

/// GET /v1/sessions/{id}/receipt - get session receipt (compute-core).
#[cfg(all(feature = "server", feature = "prism-backend"))]
async fn get_receipt(State(server): State<AppState>, Path(id): Path<String>) -> Json<Value> {
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

// -- DELETE /v1/sessions/{id} --------------------------------------

/// DELETE /v1/sessions/{id} - delete a session.
#[cfg(all(feature = "server", not(feature = "prism-backend")))]
async fn delete_session(State(server): State<AppState>, Path(id): Path<String>) -> Json<Value> {
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
async fn delete_session(State(server): State<AppState>, Path(id): Path<String>) -> Json<Value> {
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

// -- GET /v1/capabilities ------------------------------------------

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

// -- GET /v1/health ------------------------------------------------

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

// -- POST /v1/images/generate -------------------------------------

/// POST /v1/images/generate - generate an image from a text prompt.
#[cfg(all(feature = "server", not(feature = "prism-backend")))]
async fn generate_image(State(_server): State<AppState>, Json(body): Json<Value>) -> Json<Value> {
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
async fn generate_image(State(server): State<AppState>, Json(body): Json<Value>) -> Json<Value> {
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
async fn generate_image(State(_server): State<AppState>, Json(body): Json<Value>) -> Json<Value> {
    let _ = body;
    Json(json!({
        "status": "error",
        "message": "feature not enabled: generation-image or generation-diffusion"
    }))
}

// -- POST /v1/audio/speech ----------------------------------------

/// POST /v1/audio/speech - generate speech from text.
#[cfg(all(feature = "server", not(feature = "prism-backend")))]
async fn generate_audio(State(_server): State<AppState>, Json(body): Json<Value>) -> Json<Value> {
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
async fn generate_audio(State(server): State<AppState>, Json(body): Json<Value>) -> Json<Value> {
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
async fn generate_audio(State(_server): State<AppState>, Json(body): Json<Value>) -> Json<Value> {
    let _ = body;
    Json(json!({
        "status": "error",
        "message": "feature not enabled: generation-audio"
    }))
}

// -- POST /v1/video/generate --------------------------------------

/// POST /v1/video/generate - generate a video from a text prompt.
#[cfg(all(feature = "server", not(feature = "prism-backend")))]
async fn generate_video(State(_server): State<AppState>, Json(body): Json<Value>) -> Json<Value> {
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
async fn generate_video(State(server): State<AppState>, Json(body): Json<Value>) -> Json<Value> {
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
async fn generate_video(State(_server): State<AppState>, Json(body): Json<Value>) -> Json<Value> {
    let _ = body;
    Json(json!({
        "status": "error",
        "message": "feature not enabled: generation-video"
    }))
}

// -- POST /v1/embeddings ------------------------------------------

/// POST /v1/embeddings - generate text embeddings.
#[cfg(all(feature = "server", not(feature = "prism-backend")))]
async fn generate_embeddings(
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
async fn generate_embeddings(
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

// -- POST /v1/multimodal/generate ---------------------------------

/// POST /v1/multimodal/generate - multimodal (vision+text) generation.
#[cfg(all(feature = "server", not(feature = "prism-backend")))]
async fn generate_multimodal(
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

/// POST /v1/multimodal/generate - multimodal generation (compute-core, stub).
#[cfg(all(feature = "server", feature = "prism-backend"))]
async fn generate_multimodal(
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
