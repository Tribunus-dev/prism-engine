use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::Html,
    routing::{get, post},
    Json, Router,
};
use futures_util::{SinkExt, StreamExt};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::broadcast;

pub struct DashboardState {
    pub registry: Arc<Mutex<prism_ecs_server::inference::ModelRegistry>>,
    /// Broadcast channel for model list change notifications.
    pub model_tx: broadcast::Sender<Vec<String>>,
}

pub fn router(state: DashboardState) -> Router {
    Router::new()
        .route("/", get(index))
        .route("/api/models", get(list_models))
        .route("/api/models/ws", get(models_ws_handler))
        .route("/api/generate", post(generate))
        .route("/api/ws", get(ws_handler))
        .route("/api/pull", get(pull_model))
        .with_state(Arc::new(state))
}

async fn index() -> Html<&'static str> {
    Html(include_str!("dashboard.html"))
}

async fn list_models(State(state): State<Arc<DashboardState>>) -> Json<Vec<String>> {
    let models = state.registry.lock().list_models();
    Json(models)
}

// ── WebSocket: /api/ws (generation stream) ─────────────────────────────

#[derive(Deserialize)]
#[serde(tag = "command")]
enum WsClientMessage {
    #[serde(rename = "generate")]
    Generate {
        model: String,
        prompt: String,
        max_tokens: Option<u64>,
    },
    #[serde(rename = "stop")]
    Stop,
}

#[derive(Serialize)]
struct TokenPayload {
    token: String,
    index: u64,
    metrics: TokenMetrics,
}

#[derive(Serialize)]
struct TokenMetrics {
    tokens_per_sec: f64,
    time_ms: f64,
    layer: u64,
}

#[derive(Serialize)]
struct AggregateMetrics {
    metrics: AggregateMetricsInner,
}

#[derive(Serialize)]
struct AggregateMetricsInner {
    avg_tokens_per_sec: f64,
    total_tokens: u64,
    total_time_ms: f64,
    peak_memory_mb: f64,
}

#[derive(Serialize)]
struct KvCacheSnapshot {
    kv_cache: KvCacheInner,
}

#[derive(Serialize)]
struct KvCacheInner {
    layer: u64,
    head: u64,
    seq_len: u64,
    cache_utilization_pct: f64,
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<DashboardState>>,
) -> impl axum::response::IntoResponse {
    ws.on_upgrade(move |socket| handle_generation_socket(socket, state))
}

async fn handle_generation_socket(socket: WebSocket, state: Arc<DashboardState>) {
    let (mut sender, mut receiver) = socket.split();

    // Wait for a generate command
    let (model, prompt, max_tokens) = loop {
        let Some(Ok(msg)) = receiver.next().await else {
            return;
        };
        match msg {
            Message::Text(text) => {
                if let Ok(cmd) = serde_json::from_str::<WsClientMessage>(&text) {
                    match cmd {
                        WsClientMessage::Generate {
                            model,
                            prompt,
                            max_tokens,
                        } => {
                            break (model, prompt, max_tokens.unwrap_or(256));
                        }
                        WsClientMessage::Stop => continue,
                    }
                }
            }
            Message::Close(_) => return,
            _ => continue,
        }
    };

    // Verify model is loaded
    let valid_model = {
        let reg = state.registry.lock();
        reg.get_model(&model).is_some()
    };
    if !valid_model {
        let _ = sender
            .send(Message::Text(
                serde_json::json!({"error": format!("Model '{}' not loaded", model)})
                    .to_string()
                    .into(),
            ))
            .await;
        let _ = sender.send(Message::Close(None)).await;
        return;
    }

    // Cancellation flag
    let cancel = Arc::new(AtomicBool::new(false));
    let cancel_clone = cancel.clone();

    // Reader task monitors for stop commands
    let reader_handle = tokio::spawn(async move {
        while let Some(Ok(msg)) = receiver.next().await {
            match msg {
                Message::Text(text) => {
                    if text.contains("\"stop\"") || text.contains("\"command\":\"stop\"") {
                        cancel_clone.store(true, std::sync::atomic::Ordering::SeqCst);
                    }
                }
                Message::Close(_) => {
                    cancel_clone.store(true, std::sync::atomic::Ordering::SeqCst);
                    break;
                }
                _ => {}
            }
        }
    });

    // ── Stub generation loop ───────────────────────────────────────────
    let start_time = Instant::now();
    let mock_tokens: Vec<&str> = prompt
        .split_whitespace()
        .chain([
            "the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "and", "then",
            "it", "runs", "away", "into", "the", "forest", "where", "it", "finds", "a", "stream",
            "and", "drinks", "deeply", ".", "The", "moon", "rises", "high", "overhead", ",",
            "casting", "a", "silver", "glow", "across", "the", "treetops", ".",
        ])
        .collect();
    let total_steps = max_tokens.min(mock_tokens.len() as u64) as usize;
    let mut cancelled = false;

    for i in 0..total_steps {
        if cancel.load(std::sync::atomic::Ordering::SeqCst) {
            cancelled = true;
            break;
        }

        // Simulate per-token latency (~20-80ms)
        let token_delay_ms = 20.0 + (i as f64 * 3.0).fract() * 60.0;
        tokio::time::sleep(tokio::time::Duration::from_secs_f64(
            token_delay_ms / 1000.0,
        ))
        .await;

        if cancel.load(std::sync::atomic::Ordering::SeqCst) {
            cancelled = true;
            break;
        }

        let elapsed = start_time.elapsed().as_secs_f64() * 1000.0;
        let tokens_per_sec = if elapsed > 0.0 {
            (i + 1) as f64 / (elapsed / 1000.0)
        } else {
            0.0
        };

        let token_str = mock_tokens[i];
        let tp = TokenPayload {
            token: token_str.to_string(),
            index: i as u64,
            metrics: TokenMetrics {
                tokens_per_sec,
                time_ms: token_delay_ms,
                layer: (i as u64 % 8) + 1,
            },
        };
        if sender
            .send(Message::Text(serde_json::to_string(&tp).unwrap().into()))
            .await
            .is_err()
        {
            break;
        }

        // Send KV cache update every 3 tokens
        if i % 3 == 0 {
            let num_layers = 8;
            for layer in 0..num_layers {
                let utilization = 10.0 + (i as f64 * 7.0 + layer as f64 * 13.0).fract() * 85.0;
                let kv = KvCacheSnapshot {
                    kv_cache: KvCacheInner {
                        layer,
                        head: 0,
                        seq_len: (i + 1) as u64,
                        cache_utilization_pct: utilization.min(100.0),
                    },
                };
                if sender
                    .send(Message::Text(serde_json::to_string(&kv).unwrap().into()))
                    .await
                    .is_err()
                {
                    break;
                }
            }
        }
    }

    // Send final metrics
    let total_time = start_time.elapsed().as_secs_f64() * 1000.0;
    let tokens_sent = if cancelled {
        let elapsed_check = start_time.elapsed().as_secs_f64();
        (elapsed_check * 10.0) as u64
    } else {
        total_steps as u64
    };

    let agg = AggregateMetrics {
        metrics: AggregateMetricsInner {
            avg_tokens_per_sec: if total_time > 0.0 {
                tokens_sent as f64 / (total_time / 1000.0)
            } else {
                0.0
            },
            total_tokens: tokens_sent,
            total_time_ms: total_time,
            peak_memory_mb: 128.0,
        },
    };

    let finished = if cancelled {
        serde_json::json!({"Stopped": ""})
    } else {
        serde_json::json!({"Finished": ""})
    };

    let _ = sender
        .send(Message::Text(serde_json::to_string(&agg).unwrap().into()))
        .await;

    let _ = sender
        .send(Message::Text(finished.to_string().into()))
        .await;

    let _ = sender.send(Message::Close(None)).await;
    reader_handle.abort();
}

// ── WebSocket: /api/models/ws (model list push) ────────────────────────

async fn models_ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<DashboardState>>,
) -> impl axum::response::IntoResponse {
    ws.on_upgrade(move |socket| handle_models_socket(socket, state))
}

async fn handle_models_socket(socket: WebSocket, state: Arc<DashboardState>) {
    let (mut sender, _receiver) = socket.split();

    let mut rx = state.model_tx.subscribe();

    // Send initial model list
    let models = state.registry.lock().list_models();
    let _ = sender
        .send(Message::Text(
            serde_json::json!({"models": models}).to_string().into(),
        ))
        .await;

    // Listen for broadcast updates, fallback to periodic polling
    loop {
        tokio::select! {
            Ok(models) = rx.recv() => {
                if sender
                    .send(Message::Text(
                        serde_json::json!({"models": models}).to_string().into(),
                    ))
                    .await
                    .is_err()
                {
                    break;
                }
            }
            _ = tokio::time::sleep(tokio::time::Duration::from_secs(5)) => {
                let models = state.registry.lock().list_models();
                if sender
                    .send(Message::Text(
                        serde_json::json!({"models": models}).to_string().into(),
                    ))
                    .await
                    .is_err()
                {
                    break;
                }
            }
        }
    }
}

// ── REST: POST /api/generate ──────────────────────────────────────────

#[derive(Deserialize)]
struct GenerateRequest {
    model: String,
    prompt: String,
    max_tokens: Option<u64>,
}

#[derive(Serialize)]
struct GenerateResponse {
    text: String,
}

async fn generate(
    State(state): State<Arc<DashboardState>>,
    Json(req): Json<GenerateRequest>,
) -> Json<GenerateResponse> {
    let reg = state.registry.lock();
    if reg.get_model(&req.model).is_some() {
        Json(GenerateResponse {
            text: format!(
                "[prism-runtime] inference engine not yet implemented.\n\nPrompt: {}\nModel: {}\nMax tokens: {:?}",
                req.prompt, req.model, req.max_tokens.unwrap_or(256)
            ),
        })
    } else {
        Json(GenerateResponse {
            text: format!("Model '{}' not loaded. Use load_model first.", req.model),
        })
    }
}

// ── WebSocket: GET /api/pull (HF model download + compile) ────────────

#[derive(Deserialize)]
struct PullRequest {
    repo: String,
}

async fn pull_model(
    ws: WebSocketUpgrade,
    State(state): State<Arc<DashboardState>>,
) -> impl axum::response::IntoResponse {
    ws.on_upgrade(move |socket| handle_pull_socket(socket, state))
}

async fn handle_pull_socket(mut socket: WebSocket, state: Arc<DashboardState>) {
    // Wait for the pull request message
    let repo = loop {
        let Some(Ok(msg)) = socket.recv().await else {
            return;
        };
        match msg {
            Message::Text(text) => {
                if let Ok(req) = serde_json::from_str::<PullRequest>(&text) {
                    break req.repo;
                }
            }
            Message::Close(_) => return,
            _ => continue,
        }
    };

    let model_name = repo.rsplit('/').next().unwrap_or(&repo).to_string();
    let base_dir = dirs::data_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("/tmp"))
        .join("prism")
        .join("models")
        .join(&model_name);

    // Helper to send progress messages
    async fn send_progress(sock: &mut WebSocket, msg: &str) {
        let _ = sock
            .send(Message::Text(
                serde_json::json!({"progress": msg}).to_string().into(),
            ))
            .await;
    }

    send_progress(&mut socket, "Connecting to HuggingFace...").await;

    let download_result = tokio::task::spawn_blocking({
        let repo = repo.clone();
        let base_dir = base_dir.clone();
        move || -> Result<std::path::PathBuf, String> {
            let api = hf_hub::api::sync::Api::new().map_err(|e| format!("HF API init: {e}"))?;
            let repo_api = api.model(repo);
            let info = repo_api.info().map_err(|e| format!("HF info: {e}"))?;

            // Download config.json first
            repo_api
                .get("config.json")
                .map_err(|e| format!("Download config.json: {e}"))?;

            // Download all safetensors files
            let mut safetensors_dir: Option<std::path::PathBuf> = None;
            for sibling in &info.siblings {
                if sibling.rfilename.ends_with(".safetensors") {
                    let path = repo_api
                        .get(&sibling.rfilename)
                        .map_err(|e| format!("Download {}: {e}", sibling.rfilename))?;
                    if safetensors_dir.is_none() {
                        safetensors_dir =
                            Some(path.parent().expect("safetensors has parent").to_path_buf());
                    }
                }
            }

            let safetensors_dir =
                safetensors_dir.ok_or_else(|| "No .safetensors files found in repo".to_string())?;

            // Also download tokenizer files
            let _ = repo_api.get("tokenizer.json");
            let _ = repo_api.get("tokenizer_config.json");

            // Create model directory
            std::fs::create_dir_all(&base_dir).map_err(|e| format!("Create model dir: {e}"))?;

            Ok(safetensors_dir)
        }
    })
    .await
    .map_err(|e| format!("Download task join: {e}"))
    .and_then(|r| r);

    let safetensors_dir = match download_result {
        Ok(dir) => dir,
        Err(e) => {
            let _ = socket
                .send(Message::Text(
                    serde_json::json!({"error": format!("Download failed: {e}")})
                        .to_string()
                        .into(),
                ))
                .await;
            let _ = socket.send(Message::Close(None)).await;
            return;
        }
    };

    // Phase 2: Build ModelGraph
    send_progress(&mut socket, "Building model graph...").await;

    let config_path = safetensors_dir.join("config.json");
    let graph_result = tokio::task::spawn_blocking(move || {
        let config = prism_ecs_ir::model_graph::UnifiedConfig::from_file(&config_path)
            .map_err(|e| format!("Parse config: {e}"))?;
        let graph = prism_ecs_ir::model_graph::ModelGraph::build(&config);
        Ok::<_, String>(graph)
    })
    .await
    .map_err(|e| format!("Graph task join: {e}"))
    .and_then(|r| r);

    let graph = match graph_result {
        Ok(g) => g,
        Err(e) => {
            let _ = socket
                .send(Message::Text(
                    serde_json::json!({"error": format!("Graph build failed: {e}")})
                        .to_string()
                        .into(),
                ))
                .await;
            let _ = socket.send(Message::Close(None)).await;
            return;
        }
    };

    // Phase 3: Compile to cimage with per-tensor progress streaming
    send_progress(&mut socket, "Compiling model (palettizing tensors)...").await;

    let output_path = base_dir.join("model.cimage");
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();

    let compile_result = tokio::task::spawn_blocking({
        let output_path = output_path.clone();
        let safetensors_dir = safetensors_dir.clone();
        move || {
            prism_ecs_quantization::compiler::compile_to_cimage(
                &graph,
                &safetensors_dir,
                &output_path,
                true,
                |key: &str, dim_m: u32, dim_n: u32, bpp: f64, elapsed_sec: f64| {
                    let _ = tx.send(
                        serde_json::json!({
                            "tensor": key,
                            "dim_m": dim_m,
                            "dim_n": dim_n,
                            "format": "Palettized4Bit",
                            "bpp": bpp,
                            "elapsed_sec": elapsed_sec,
                        })
                        .to_string(),
                    );
                },
            )
        }
    })
    .await;

    // Forward progress messages from the compile task to the WebSocket
    while let Some(msg) = rx.recv().await {
        let _ = socket.send(Message::Text(msg.into())).await;
    }
    let compile_result = compile_result
        .map_err(|e| format!("Compile task join: {e}"))
        .and_then(|r| r);

    if let Err(e) = compile_result {
        let _ = socket
            .send(Message::Text(
                serde_json::json!({"error": format!("Compilation failed: {e}")})
                    .to_string()
                    .into(),
            ))
            .await;
        let _ = socket.send(Message::Close(None)).await;
        return;
    }

    // Phase 4: Load into registry
    send_progress(&mut socket, "Loading model into registry...").await;

    let load_result;
    {
        let reg = state.registry.lock();
        load_result = reg.load_model(&output_path);
        let models = reg.list_models();
        let _ = state.model_tx.send(models);
    }
    // MutexGuard dropped here before any await
    if let Err(e) = load_result {
        let _ = socket
            .send(Message::Text(
                serde_json::json!({"error": format!("Load model failed: {e}")})
                    .to_string()
                    .into(),
            ))
            .await;
        let _ = socket.send(Message::Close(None)).await;
        return;
    }

    // Success
    let _ = socket
        .send(Message::Text(
            serde_json::json!({
                "status": "done",
                "name": model_name,
                "path": output_path.to_string_lossy(),
            })
            .to_string()
            .into(),
        ))
        .await;
    let _ = socket.send(Message::Close(None)).await;
}

/// Notify model WebSocket subscribers that the model list changed.
#[allow(dead_code)]
pub fn notify_models_changed(state: &DashboardState) {
    let models = state.registry.lock().list_models();
    let _ = state.model_tx.send(models);
}
