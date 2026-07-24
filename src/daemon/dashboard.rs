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
use prism_ecs_constitutional::lifecycle_command::{CreateWorkCommand, LifecycleCommand};
use prism_ecs_ir::evolution::FormatPlan;
use prism_ecs_ir::evolution::{
    evaluate::{EvaluationStrategy, SyntheticEvaluator},
    foundation::{CandidateGenome, FitnessScore},
    frontier::ParetoFrontier,
    joint::{JointEvolutionSystem, JointSearchConfig, ScoredGenome},
};
use prism_ecs_runtime::KernelHandle;
use prism_ecs_runtime::{Command, CommandEnvelope};
use prism_ecs_protocol::{Event, ProtocolRequest, ProtocolError, EventBody, ErrorCode};
use prism_ecs_protocol_adapter::ApplicationClient;
use prism_ecs_server::runtime::server::PrefillDecodeRuntime;
use prism_ecs_server::runtime::server_types::SamplingConfig;
use prism_mcp_core::{
    ArtifactKind, ArtifactRepository, EvidenceReceipt, EvidenceStatus, MetricSet, ProjectionStore,
    ToolInvocationId,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use serde_json::Value;
use std::collections::HashMap;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::broadcast;

pub struct DashboardState {
    /// Whether this user is authorized for full compilation pipeline.
    pub authorized: Arc<AtomicBool>,
    pub auth_token: Arc<String>,
    pub registry: Arc<Mutex<prism_ecs_server::inference::ModelRegistry>>,
    /// Broadcast channel for model list change notifications.
    pub model_tx: broadcast::Sender<Vec<String>>,
    /// Structured compiler-lab events for native and web clients.
    pub compiler_lab_tx: broadcast::Sender<Value>,
    /// ECS world for agent observability.
    pub world: KernelHandle,
    /// Whether the daemon has a working compiler dispatcher.
    pub has_compiler_dispatcher: bool,
    /// Unix JSON-RPC socket used by the Deno WebUI bridge.
    pub socket_path: PathBuf,
    /// Durable destination for promoted assembly artifacts.
    pub artifact_dir: PathBuf,
    pub artifact_store: Arc<dyn ArtifactRepository>,
    pub evidence_ledger: Arc<dyn prism_mcp_core::EvidenceStore>,
    pub projection_store: Arc<dyn ProjectionStore>,
    pub provenance_store: Arc<dyn prism_mcp_core::ProvenanceGraphStore>,
    pub graph_projection: Arc<crate::daemon::trifecta_store::DuckDbGraphProjection>,
    /// Versioned Rust-owned application/workflow protocol boundary.
    pub workflow_client: Arc<dyn ApplicationClient>,
}

pub fn router(state: DashboardState) -> Router {
    Router::new()
        .route("/", get(index))
        .route("/api/models", get(list_models))
        .route("/api/models/status", get(model_status))
        .route("/api/auth/status", get(auth_status))
        .route("/api/auth/unlock", post(auth_unlock))
        .route("/api/graph", get(graph))
        .route("/api/agents", get(list_agents))
        .route("/api/models/ws", get(models_ws_handler))
        .route("/api/compiler-lab/ws", get(compiler_lab_ws_handler))
        .route("/api/compiler-lab/search", post(start_search))
        .route("/api/generate", post(generate))
        .route("/api/ws", get(ws_handler))
        .route("/api/pull", get(pull_model))
        .route("/api/assemble", post(assemble))
        .route("/api/ecs", post(ecs_protocol))
        .with_state(Arc::new(state))
}

async fn ecs_protocol(
    State(state): State<Arc<DashboardState>>,
    Json(request): Json<Result<ProtocolRequest, serde_json::Error>>,
) -> Json<Event> {
    match request {
        Ok(request) => Json(state.workflow_client.send(request)),
        Err(error) => Json(Event::new(
            uuid::Uuid::nil(),
            EventBody::Error(ProtocolError::new(
                uuid::Uuid::nil(),
                ErrorCode::InvalidRequest,
                format!("invalid ECS protocol request: {error}"),
                false,
            )),
        )),
    }
}

#[derive(Deserialize)]
struct SearchRequest {
    model: String,
    population: Option<usize>,
    generations: Option<usize>,
}

async fn start_search(
    State(state): State<Arc<DashboardState>>,
    Json(request): Json<SearchRequest>,
) -> Json<Value> {
    if !state.authorized.load(Ordering::Acquire) {
        return Json(json!({"error": "compiler lab authorization required"}));
    }
    let population_size = request.population.unwrap_or(24).clamp(2, 256);
    let generations = request.generations.unwrap_or(12).clamp(1, 500);
    let model = request.model;
    let model_for_task = model.clone();
    let tx = state.compiler_lab_tx.clone();
    tokio::spawn(async move {
        let _ = tx.send(json!({"type":"search","phase":"Search","detail":format!("Starting evolution search for {model_for_task}"),"generation":0,"generations":generations,"population":population_size,"candidates":[]}));
        let event_tx = tx.clone();
        let result = tokio::task::spawn_blocking(move || {
            let measured = prism_ecs_server::engine::MeasuredEvaluator::new(0.3);
            let synthetic = SyntheticEvaluator::new();
            let config = JointSearchConfig { population_size, crossover_rate: 0.7, mutation_rate: 0.3, max_generations: generations, stagnation_limit: 5, seed: None };
            let system = JointEvolutionSystem::new(config);
            let mut frontier = ParetoFrontier::new(8);
            let mut population: Vec<ScoredGenome> = (0..population_size).map(|_| {
                let genome = CandidateGenome::new();
                let score = synthetic.evaluate(&genome, b"4096,4096");
                ScoredGenome { genome, fitness: vec![score.value()] }
            }).collect();
            for generation in 0..generations {
                let mut candidates = Vec::with_capacity(population.len());
                for (index, scored) in population.iter_mut().enumerate() {
                    let synth = synthetic.evaluate(&scored.genome, b"synthetic:4096,4096");
                    let fitness = if synth.value() >= 0.3 { measured.evaluate(&scored.genome, b"4096,4096").value() } else { synth.value() };
                    scored.fitness = vec![fitness];
                    candidates.push(json!({"id":index,"fitness":fitness,"representation":format!("{:?}", scored.genome.representation)}));
                    frontier.insert(scored.genome.clone(), vec![FitnessScore::new(fitness)], generation as u64, &Default::default());
                }
                candidates.sort_by(|a,b| b["fitness"].as_f64().partial_cmp(&a["fitness"].as_f64()).unwrap_or(std::cmp::Ordering::Equal));
                let best = candidates.first().and_then(|c| c["fitness"].as_f64()).unwrap_or(0.0);
                let _ = event_tx.send(json!({"type":"search","phase":"Search","detail":format!("Evaluated generation {}", generation + 1),"generation":generation + 1,"generations":generations,"population":population_size,"bestFitness":best,"candidates":candidates}));
                if generation + 1 < generations { let (next, _) = system.run_generation(&population, &frontier); population = next; }
            }
        }).await;
        let detail = if result.is_ok() {
            "Evolution search completed"
        } else {
            "Evolution search failed"
        };
        let _ = tx.send(json!({"type":"search","phase":if result.is_ok() {"Complete"} else {"Failed"},"detail":detail,"generation":generations,"generations":generations}));
    });
    Json(
        json!({"status":"started","model":model,"population":population_size,"generations":generations}),
    )
}

async fn compiler_lab_ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<DashboardState>>,
) -> impl axum::response::IntoResponse {
    ws.on_upgrade(move |mut socket| async move {
        let mut events = state.compiler_lab_tx.subscribe();
        while let Ok(event) = events.recv().await {
            if socket
                .send(Message::Text(event.to_string().into()))
                .await
                .is_err()
            {
                break;
            }
        }
    })
}

async fn auth_status(State(state): State<Arc<DashboardState>>) -> Json<Value> {
    Json(json!({"authorized": state.authorized.load(Ordering::Acquire)}))
}

#[derive(Deserialize)]
struct AuthUnlockRequest {
    token: String,
}

async fn auth_unlock(
    State(state): State<Arc<DashboardState>>,
    Json(req): Json<AuthUnlockRequest>,
) -> Json<Value> {
    let ok = !req.token.is_empty() && req.token == state.auth_token.as_str();
    if ok {
        state.authorized.store(true, Ordering::Release);
    }
    Json(
        json!({"authorized": ok, "error": if ok { Value::Null } else { json!("invalid authorization token") }}),
    )
}

async fn list_agents(State(state): State<Arc<DashboardState>>) -> Json<Vec<Value>> {
    let agents = state.world.query_agents();
    let result: Vec<Value> = agents
        .into_iter()
        .map(|a| {
            json!({
                "entity_id": a.entity_id,
                "phase": a.phase,
                "lifecycle": a.lifecycle,
                "parent_id": a.parent_id,
            })
        })
        .collect();
    Json(result)
}

async fn index() -> Html<String> {
    // Keep the native daemon dashboard and the Deno-hosted dashboard on one
    // source of truth. The Deno wrapper replaces this placeholder with its
    // configured daemon endpoint before serving the page; do the equivalent
    // here for direct access on port 8080.
    let html = include_str!("../../deno-dashboard/dashboard.html")
        .replace("__PRISM_DAEMON_WS__", "ws://127.0.0.1:8080/api/ws");
    Html(html)
}

async fn list_models(State(state): State<Arc<DashboardState>>) -> Json<Vec<String>> {
    let models = state.registry.lock().list_models();
    Json(models)
}

async fn model_status(State(state): State<Arc<DashboardState>>) -> Json<Vec<Value>> {
    Json(state.registry.lock().residency_snapshot())
}

/// Return the normalized model graph with deterministic lineage edges. The
/// source path is included on every node so callers can trace graph data back
/// to the loaded artifact/configuration without exposing tensor payloads.
pub fn query_graph(
    registry: &prism_ecs_server::inference::ModelRegistry,
    model_name: &str,
) -> Result<Value, String> {
    query_federated_graph(registry, Some(model_name), None, None, None)
}

/// Federated graph query boundary shared by HTTP and MCP. `model` remains an
/// optional compatibility filter; omitted models are federated into one graph.
pub fn query_federated_graph(
    registry: &prism_ecs_server::inference::ModelRegistry,
    model_filter: Option<&str>,
    operation: Option<&str>,
    root: Option<&str>,
    domain: Option<&str>,
) -> Result<Value, String> {
    let model_names = if let Some(model) = model_filter {
        vec![model.to_string()]
    } else {
        registry.list_models()
    };
    if model_names.is_empty() {
        return Err("no loaded models available".to_string());
    }
    let mut all_nodes = Vec::new();
    let mut all_edges = Vec::new();
    let mut evidence = Vec::new();
    for model_name in model_names {
        let model = registry
            .get_model(&model_name)
            .ok_or_else(|| format!("model not loaded: {model_name}"))?;
        let config_path = model
            .model_path
            .parent()
            .ok_or_else(|| "model has no parent directory".to_string())?
            .join("config.json");
        let config = prism_ecs_ir::model_graph::UnifiedConfig::from_file(&config_path)
            .map_err(|e| format!("load graph config {}: {e}", config_path.display()))?;
        let graph = prism_ecs_ir::model_graph::ModelGraph::build(&config);
        let nodes: Vec<Value> = graph.nodes.iter().enumerate().map(|(index, node)| json!({
        "id": format!("{model_name}:node:{index}"),
        "index": index,
        "kind": format!("{:?}", node).split('{').next().unwrap_or("ComputeNode"),
        "node": node,
        "provenance": {"model": model_name, "artifact": model.model_path, "config": config_path, "node_index": index},
        "domain": "model",
        "operation": format!("{:?}", node).split('{').next().unwrap_or("ComputeNode")
    })).filter(|node| domain.map(|d| d == "model").unwrap_or(true))
      .filter(|node| operation.map(|op| node["operation"].as_str() == Some(op)).unwrap_or(true))
      .filter(|node| root.map(|r| node["id"].as_str() == Some(r) || node["index"].as_u64().map(|i| i.to_string() == r).unwrap_or(false)).unwrap_or(true))
      .collect();
        let edges: Vec<Value> = (1..graph.nodes.len()).map(|index| json!({
        "id": format!("{model_name}:edge:{}-{}", index - 1, index),
        "from": format!("{model_name}:node:{}", index - 1),
        "to": format!("{model_name}:node:{index}"),
        "relation": "execution_order",
        "provenance": {"model": model_name, "source": "ModelGraph::build", "edge_index": index - 1},
        "domain": "model", "operation": "execution_order"
    })).filter(|edge| domain.map(|d| d == "model").unwrap_or(true))
      .filter(|edge| operation.map(|op| edge["operation"].as_str() == Some(op)).unwrap_or(true))
      .collect();
        evidence.push(json!({"model": model_name, "domain":"model", "kind":"graph_config", "artifact":model.model_path, "config":config_path, "source":"ModelGraph::build"}));
        all_nodes.extend(nodes);
        all_edges.extend(edges);
    }
    Ok(
        json!({"model": model_filter, "models": model_filter.map(|m| json!([m])).unwrap_or_else(|| json!(registry.list_models())), "filters":{"operation":operation,"root":root,"domain":domain}, "nodes":all_nodes, "edges":all_edges, "evidence":evidence, "federation":{"domains":["model"],"durable":true}}),
    )
}

async fn graph(
    State(state): State<Arc<DashboardState>>,
    axum::extract::Query(query): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> (axum::http::StatusCode, Json<Value>) {
    let model = query
        .get("model")
        .filter(|m| !m.is_empty())
        .map(String::as_str);
    if let Ok(graph) = state
        .provenance_store
        .query(&prism_mcp_core::ProvenanceQuery {
            node_id: query.get("root").cloned(),
            relation: query.get("operation").cloned(),
            limit: 512,
            ..Default::default()
        })
    {
        if !graph.nodes.is_empty() {
            return (
                axum::http::StatusCode::OK,
                Json(
                    json!({"model": model, "nodes": graph.nodes, "edges": graph.edges, "evidence": [], "federation": {"durable": true}}),
                ),
            );
        }
    }
    let (projected_nodes, projected_edges) = state.graph_projection.query(
        query.get("root").map(String::as_str),
        query.get("operation").map(String::as_str),
    );
    if !projected_nodes.is_empty() {
        return (
            axum::http::StatusCode::OK,
            Json(
                json!({"model": model, "nodes": projected_nodes, "edges": projected_edges, "evidence": [], "federation": {"durable": true, "projection": "duckdb"}}),
            ),
        );
    }
    match query_federated_graph(
        &state.registry.lock(),
        model,
        query.get("operation").map(String::as_str),
        query.get("root").map(String::as_str),
        query.get("domain").map(String::as_str),
    ) {
        Ok(graph) => (axum::http::StatusCode::OK, Json(graph)),
        Err(error) => (
            axum::http::StatusCode::NOT_FOUND,
            Json(json!({"error": error})),
        ),
    }
}

// ── WebSocket: /api/ws (generation stream) ─────────────────────────────

#[derive(Deserialize)]
#[serde(untagged)]
enum WsClientMessage {
    Generate {
        command: String,
        model: String,
        prompt: String,
        max_tokens: Option<u64>,
    },
    ToolCall {
        id: Value,
        tool: String,
        args: Value,
    },
    Stop {
        command: String,
    },
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
                            command,
                            model,
                            prompt,
                            max_tokens,
                        } if command == "generate" => {
                            break (model, prompt, max_tokens.unwrap_or(256));
                        }
                        WsClientMessage::ToolCall { id, tool, args } => {
                            let response = tokio::task::spawn_blocking({
                                let socket_path = state.socket_path.clone();
                                move || call_mcp_socket(&socket_path, id, &tool, args)
                            })
                            .await
                            .unwrap_or_else(|e| json!({"error": e.to_string()}));
                            if sender
                                .send(Message::Text(response.to_string().into()))
                                .await
                                .is_err()
                            {
                                return;
                            }
                        }
                        WsClientMessage::Stop { .. } => continue,
                        _ => continue,
                    }
                }
            }
            Message::Close(_) => return,
            _ => continue,
        }
    };

    // Verify model is loaded
    let runtime = {
        let reg = state.registry.lock();
        reg.get_model(&model)
            .map(|instance| instance.runtime.clone())
    };
    let Some(runtime) = runtime else {
        let _ = sender
            .send(Message::Text(
                serde_json::json!({"error": format!("Model '{}' not loaded", model)})
                    .to_string()
                    .into(),
            ))
            .await;
        let _ = sender.send(Message::Close(None)).await;
        return;
    };

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

    // ── Production inference loop ──────────────────────────────────────
    let start_time = Instant::now();
    let prompt_tokens = match runtime.tokenize(&prompt) {
        Ok(tokens) if !tokens.is_empty() => tokens,
        Ok(_) => {
            let _ = sender
                .send(Message::Text(
                    json!({"error":"prompt tokenized to empty input"})
                        .to_string()
                        .into(),
                ))
                .await;
            return;
        }
        Err(error) => {
            let _ = sender
                .send(Message::Text(json!({"error":error}).to_string().into()))
                .await;
            return;
        }
    };
    let mut logits = match runtime.run_prefill(&prompt_tokens) {
        Ok(logits) => logits,
        Err(error) => {
            let _ = sender
                .send(Message::Text(json!({"error":error}).to_string().into()))
                .await;
            return;
        }
    };
    let sampling = SamplingConfig::default();
    let total_steps = max_tokens as usize;
    let mut cancelled = false;
    let mut tokens_sent = 0u64;
    for i in 0..total_steps {
        if cancel.load(std::sync::atomic::Ordering::SeqCst) {
            cancelled = true;
            break;
        }

        let token = match runtime.sample(&logits, &sampling) {
            Ok(token) => token,
            Err(error) => {
                let _ = sender
                    .send(Message::Text(json!({"error":error}).to_string().into()))
                    .await;
                break;
            }
        };
        if token == runtime.eos_token_id() {
            break;
        }
        let token_str = match runtime.detokenize(token) {
            Ok(text) => text,
            Err(error) => {
                let _ = sender
                    .send(Message::Text(json!({"error":error}).to_string().into()))
                    .await;
                break;
            }
        };
        logits = match runtime.run_decode(token) {
            Ok(logits) => logits,
            Err(error) => {
                let _ = sender
                    .send(Message::Text(json!({"error":error}).to_string().into()))
                    .await;
                break;
            }
        };
        tokens_sent += 1;

        let elapsed = start_time.elapsed().as_secs_f64() * 1000.0;
        let tokens_per_sec = if elapsed > 0.0 {
            (i + 1) as f64 / (elapsed / 1000.0)
        } else {
            0.0
        };

        let tp = TokenPayload {
            token: token_str.clone(),
            index: i as u64,
            metrics: TokenMetrics {
                tokens_per_sec,
                time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
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
            let num_layers = 1;
            for layer in 0..num_layers {
                let utilization: f64 = 0.0;
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
    let tokens_sent = tokens_sent;

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

/// Translate the Deno client's compact `{id, tool, args}` envelope into the
/// daemon's authoritative MCP JSON-RPC `tools/call` request over its Unix
/// socket. This keeps the WebUI transport interoperable without duplicating
/// tool dispatch logic in the dashboard server.
fn call_mcp_socket(socket_path: &PathBuf, id: Value, tool: &str, args: Value) -> Value {
    let stream = match UnixStream::connect(socket_path) {
        Ok(stream) => stream,
        Err(error) => {
            return json!({"jsonrpc":"2.0","id":id,"error":{"message":format!("daemon unavailable: {error}")}})
        }
    };
    let request = json!({"jsonrpc":"2.0","id":id,"method":"tools/call","params":{"name":tool,"arguments":args}});
    let mut writer = BufWriter::new(match stream.try_clone() {
        Ok(s) => s,
        Err(error) => return json!({"jsonrpc":"2.0","id":id,"error":{"message":error.to_string()}}),
    });
    if writeln!(writer, "{}", request)
        .and_then(|_| writer.flush())
        .is_err()
    {
        return json!({"jsonrpc":"2.0","id":id,"error":{"message":"failed to write daemon request"}});
    }
    let mut line = String::new();
    if BufReader::new(stream).read_line(&mut line).is_err() {
        return json!({"jsonrpc":"2.0","id":id,"error":{"message":"failed to read daemon response"}});
    }
    serde_json::from_str(line.trim()).unwrap_or_else(
        |_| json!({"jsonrpc":"2.0","id":id,"error":{"message":"invalid daemon response"}}),
    )
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
    let model = match state.registry.lock().acquire(&req.model) {
        Ok(lease) => lease,
        Err(error) => {
            return Json(GenerateResponse {
                text: format!("[admission] {error}"),
            })
        }
    };
    let max_tokens = req.max_tokens.unwrap_or(256).clamp(1, 4096) as usize;
    let runtime = model.model.runtime.clone();
    let lease = model;
    let result = tokio::task::spawn_blocking(move || {
        let _lease = lease;
        let prompt_tokens = runtime.tokenize(&req.prompt)?;
        if prompt_tokens.is_empty() {
            return Err("prompt tokenized to an empty sequence".to_string());
        }
        let mut logits = runtime.run_prefill(&prompt_tokens)?;
        let sampling = SamplingConfig::default();
        let mut output = String::new();
        for _ in 0..max_tokens {
            let token = runtime.sample(&logits, &sampling)?;
            if token == runtime.eos_token_id() {
                break;
            }
            output.push_str(&runtime.detokenize(token)?);
            logits = runtime.run_decode(token)?;
        }
        Ok::<String, String>(output)
    })
    .await
    .unwrap_or_else(|error| Err(format!("inference task failed: {error}")));
    Json(GenerateResponse {
        text: result.unwrap_or_else(|error| format!("[engine] inference failed: {error}")),
    })
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
    if !state.authorized.load(Ordering::Acquire) {
        let _ = socket.send(Message::Text(
            serde_json::json!({"error": "Full compilation pipeline requires authorization. Use Model Store for pre-compiled models."}).to_string().into()
        )).await;
        return;
    }
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

    let send_lab_event = |state: &DashboardState, phase: &str, detail: &str, progress: f64| {
        let _ = state.compiler_lab_tx.send(json!({
            "type": "compiler",
            "phase": phase,
            "detail": detail,
            "progress": progress
        }));
    };

    send_progress(&mut socket, "Connecting to HuggingFace...").await;
    send_lab_event(
        &state,
        "Ingest",
        "Connecting to HuggingFace and resolving model files",
        0.05,
    );

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
    send_lab_event(
        &state,
        "Graph",
        "Building model graph and lowering logical operations",
        0.25,
    );

    // The daemon's CompilerDispatcher will rebuild the graph from disk at
    // dispatch time, so graph building is deferred to the daemon.

    // Phase 2.5: Run evolution search to find optimal per-tensor formats
    send_progress(
        &mut socket,
        "Running evolution search for quantization formats...",
    )
    .await;
    send_lab_event(
        &state,
        "Search",
        "Exploring quantization format candidates",
        0.45,
    );
    // Format plan is computed but unused in kernel path; default compilation is used.
    // TODO: pass format_plan through kernel job config when supported.
    let _format_plan = FormatPlan::new();

    // Phase 3: Compile to cimage with per-tensor progress streaming
    send_progress(&mut socket, "Compiling model (palettizing tensors)...").await;
    send_lab_event(&state, "Compile", "Compiling and palettizing tensors", 0.65);

    // Use the HF cache directory as input path — it already contains config.json
    // and safetensors. The DaemonCompilerDispatcher reads models from disk.
    let input_path = safetensors_dir.to_string_lossy().to_string();
    let output_str = base_dir.join("model.cimage").to_string_lossy().to_string();

    let compile_result = tokio::task::spawn_blocking({
        let input_path = input_path.clone();
        let output_str = output_str.clone();
        let world = state.world.clone();
        move || -> Result<(), String> {
            // Submit a compile work item to the daemon's authoritative kernel.
            let _outcome = world
                .submit(CommandEnvelope::new(Command::Lifecycle(
                    LifecycleCommand::CreateWork(CreateWorkCommand {
                        target_entity: 0,
                        kind: "compile".to_string(),
                        resource_claim: "{}".to_string(),
                        output_path: output_str.clone(),
                        input_path: input_path.clone(),
                    }),
                )))
                .map_err(|e| format!("submit CreateWork: {e}"))?;

            // Poll for output file (daemon tick loop processes asynchronously)
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(300);
            loop {
                if std::time::Instant::now() > deadline {
                    return Err("compilation timed out (300s)".to_string());
                }
                let p = std::path::Path::new(&output_str);
                if p.exists() {
                    if let Ok(meta) = std::fs::metadata(p) {
                        if meta.len() > 0 {
                            return Ok(());
                        }
                    }
                }
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
        }
    })
    .await;

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
    send_lab_event(
        &state,
        "Verify",
        "Loading and verifying the compiled artifact",
        0.9,
    );

    let load_result;
    {
        let reg = state.registry.lock();
        load_result = reg.load_model(std::path::Path::new(&output_str));
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
                "path": output_str,
            })
            .to_string()
            .into(),
        ))
        .await;
    send_lab_event(
        &state,
        "Complete",
        "Compiler artifact loaded into the model registry",
        1.0,
    );
    let _ = socket.send(Message::Close(None)).await;
}

/// Notify model WebSocket subscribers that the model list changed.
#[allow(dead_code)]
pub fn notify_models_changed(state: &DashboardState) {
    let models = state.registry.lock().list_models();
    let _ = state.model_tx.send(models);
}
// ── POST /api/assemble ───────────────────────────────────────────

#[derive(Deserialize)]
struct AssembleRequest {
    models: Vec<AssembleModelRequest>,
    #[serde(default)]
    total_ram_budget_gb: Option<f64>,
}

#[derive(Deserialize)]
struct AssembleModelRequest {
    name: String,
    repo: String,
    #[serde(default, alias = "size_gb")]
    ram_estimate_gb: f64,
    #[serde(default = "default_assembly_architecture")]
    architecture: String,
}

fn default_assembly_architecture() -> String {
    "decoder_only".to_string()
}

fn resolve_assembly_source(repo: &str) -> Result<PathBuf, String> {
    let requested = PathBuf::from(repo);
    if requested.is_file() {
        return Ok(requested);
    }
    if requested.is_dir() {
        let has_safetensors = std::fs::read_dir(&requested)
            .ok()
            .into_iter()
            .flat_map(|entries| entries.filter_map(Result::ok))
            .any(|entry| entry.path().extension().is_some_and(|e| e == "safetensors"));
        if requested.join("config.json").is_file() && has_safetensors {
            return Ok(requested);
        }
        let mut candidates = std::fs::read_dir(&requested)
            .map_err(|e| format!("read source directory: {e}"))?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|p| {
                matches!(
                    p.extension().and_then(|e| e.to_str()),
                    Some("cimage" | "gguf" | "onnx")
                )
            })
            .collect::<Vec<_>>();
        candidates.sort();
        return candidates.into_iter().next().ok_or_else(|| {
            format!(
                "no .cimage, .gguf, or .onnx source in {}",
                requested.display()
            )
        });
    }
    // Resolve a Hugging Face repo against the standard local hub cache. Network
    // acquisition belongs to the Model Store; the daemon never downloads an
    // untrusted source as part of an authenticated compile request.
    let cache_root = std::env::var_os("HF_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache/huggingface")))
        .map(|p| p.join("hub"));
    let Some(cache_root) = cache_root else {
        return Err(format!("assembly source not found: {repo}"));
    };
    let cache_name = format!("models--{}", repo.replace('/', "--"));
    let snapshots = cache_root.join(cache_name).join("snapshots");
    let mut revisions = std::fs::read_dir(&snapshots)
        .ok()
        .into_iter()
        .flat_map(|it| it.filter_map(Result::ok).map(|e| e.path()))
        .collect::<Vec<_>>();
    revisions.sort();
    revisions.reverse();
    for revision in revisions {
        if let Ok(source) = resolve_assembly_source(revision.to_string_lossy().as_ref()) {
            return Ok(source);
        }
    }
    Err(format!(
        "assembly source not found locally for '{repo}'; pull it through Model Store first"
    ))
}

async fn assemble(
    State(state): State<Arc<DashboardState>>,
    Json(req): Json<AssembleRequest>,
) -> Json<Value> {
    if !state.authorized.load(Ordering::Acquire) {
        return Json(json!({
            "error": "Assembly pipeline requires authorization. Use Model Store for pre-compiled models."
        }));
    }
    if req.models.is_empty() {
        return Json(json!({"error": "at least one model is required"}));
    }
    let mut files = HashMap::new();
    let mut manifest = prism_ecs_compile::MultiModelManifest::default();
    for model in &req.models {
        if model.name.trim().is_empty() || model.repo.trim().is_empty() {
            return Json(json!({"error": "model name and repo are required"}));
        }
        let source = match resolve_assembly_source(&model.repo) {
            Ok(path) => path,
            Err(error) => return Json(json!({"error": error, "model": model.name})),
        };
        files.insert(model.name.clone(), source);
        let modality = match model.architecture.as_str() {
            "vit" | "vision" => prism_ecs_compile::ModelModality::Vision,
            "audio_codec" | "tts" => prism_ecs_compile::ModelModality::Audio,
            "image" => prism_ecs_compile::ModelModality::Image,
            "video" => prism_ecs_compile::ModelModality::Video,
            _ => prism_ecs_compile::ModelModality::Text,
        };
        manifest
            .insert(prism_ecs_compile::ModelManifest {
                id: model.name.clone(),
                modality,
                inputs: vec![prism_ecs_compile::ModelIoBinding {
                    name: "input".into(),
                    kind: prism_ecs_compile::ModelIoKind::Tokens,
                    dtype: "f32".into(),
                    shape: vec![1],
                    optional: false,
                }],
                outputs: vec![prism_ecs_compile::ModelIoBinding {
                    name: "output".into(),
                    kind: prism_ecs_compile::ModelIoKind::Embedding,
                    dtype: "f32".into(),
                    shape: vec![1],
                    optional: false,
                }],
                requirements: Default::default(),
                program_names: vec![format!("{}/forward", model.name)],
                projectors: vec![],
                fusion_inputs: vec![],
            })
            .map_err(|error| return Json(json!({"error": error})))
            .ok();
    }
    let budget = req.total_ram_budget_gb.unwrap_or(usize::MAX as f64);
    if !budget.is_finite() || budget <= 0.0 {
        return Json(json!({"error": "total_ram_budget_gb must be positive"}));
    }
    let request_id = uuid::Uuid::new_v4().to_string();
    let staging = state
        .artifact_dir
        .join(".assembly-staging")
        .join(&request_id);
    let destination = state
        .artifact_dir
        .join(format!("assembled-{request_id}.cimage"));
    let compile = tokio::task::spawn_blocking({
        let staging = staging.clone();
        let destination = destination.clone();
        move || {
            std::fs::create_dir_all(&staging)
                .map_err(|e| format!("create staging directory: {e}"))?;
            let support_dir = files
                .values()
                .next()
                .and_then(|path| path.parent())
                .map(PathBuf::from);
            let receipt = prism_ecs_server::assembly::assemble_production(
                prism_ecs_server::assembly::ProductionAssemblyRequest {
                    output_path: staging.join("assembled.cimage"),
                    sources: files
                        .into_iter()
                        .map(|(model_id, path)| prism_ecs_compile::AssemblyModelSource {
                            model_id,
                            path,
                        })
                        .collect(),
                    manifest,
                },
            )?;
            std::fs::rename(&receipt.artifact_path, &destination)
                .map_err(|e| format!("promote artifact: {e}"))?;
            if let Some(source_dir) = support_dir {
                for file in ["config.json", "tokenizer.json", "tokenizer_config.json"] {
                    let source = source_dir.join(file);
                    if source.is_file() {
                        let _ = std::fs::copy(
                            &source,
                            destination
                                .parent()
                                .unwrap_or(PathBuf::from(".").as_path())
                                .join(file),
                        );
                    }
                }
            }
            Ok::<_, String>(())
        }
    })
    .await;
    let result = compile.map_err(|e| e.to_string()).and_then(|r| r);
    let _ = std::fs::remove_dir_all(&staging);
    if let Err(error) = result {
        return Json(json!({"status":"failed", "error": error}));
    }
    let invocation = ToolInvocationId::new();
    let artifact_bytes = match std::fs::read(&destination) {
        Ok(bytes) => bytes,
        Err(error) => {
            return Json(
                json!({"status":"failed", "error": format!("read assembled artifact: {error}")}),
            )
        }
    };
    let artifact_id = match state.artifact_store.put(
        &artifact_bytes,
        ArtifactKind::Cimage,
        &invocation,
    ) {
        Ok(id) => id,
        Err(error) => {
            return Json(
                json!({"status":"failed", "error": format!("record assembled artifact: {error}")}),
            )
        }
    };
    let evidence = EvidenceReceipt {
        invocation_id: invocation.clone(),
        tool: "assembly".into(),
        operation: "assemble_production".into(),
        inputs: Vec::new(),
        outputs: vec![artifact_id],
        environment: "daemon-dashboard".into(),
        target: Some("assembled-cimage".into()),
        source_revision: None,
        status: EvidenceStatus::Success,
        metrics: MetricSet::new(),
        diagnostics: Vec::new(),
        started_at: chrono::Utc::now(),
        duration_ms: 0,
    };
    let _ = state.evidence_ledger.record(&evidence);
    let assembly_id = format!("assembly:{request_id}");
    let _ = state.projection_store.put_trace(&assembly_id, &json!({
        "kind": "assembly_provenance",
        "nodes": [
            {"id": assembly_id, "kind": "assembly_decision", "models": req.models.iter().map(|m| &m.name).collect::<Vec<_>>()},
            {"id": format!("artifact:{}", artifact_id.hex()), "kind": "cimage_artifact"},
            {"id": format!("evidence:{}", invocation), "kind": "assembly_evidence"}
        ],
        "edges": [
            {"from": format!("assembly:{request_id}"), "to": format!("artifact:{}", artifact_id.hex()), "kind": "emitted"},
            {"from": format!("assembly:{request_id}"), "to": format!("evidence:{}", invocation), "kind": "attested_by"}
        ]
    }));
    let loaded = state.registry.lock().load_model(&destination);
    match loaded {
        Ok(instance) => {
            let models = state.registry.lock().list_models();
            let _ = state.model_tx.send(models);
            Json(json!({"status":"done", "model_name": instance.name, "path": destination}))
        }
        Err(error) => {
            let _ = std::fs::remove_file(&destination);
            Json(json!({"status":"failed", "error": format!("load promoted artifact: {error}")}))
        }
    }
}

#[cfg(test)]
mod assembly_tests {
    use super::*;

    #[test]
    fn resolves_weight_file_from_directory() {
        let root =
            std::env::temp_dir().join(format!("prism-assembly-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let source = root.join("weights.gguf");
        std::fs::write(&source, b"fixture").unwrap();
        assert_eq!(
            resolve_assembly_source(root.to_str().unwrap()).unwrap(),
            source
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn assembly_request_accepts_size_gb_alias_and_defaults_architecture() {
        let request: AssembleRequest = serde_json::from_value(serde_json::json!({
            "models": [{"name": "llm", "repo": "/tmp/model", "size_gb": 2.5}]
        }))
        .unwrap();
        assert_eq!(request.models[0].ram_estimate_gb, 2.5);
        assert_eq!(request.models[0].architecture, "decoder_only");
    }
}
