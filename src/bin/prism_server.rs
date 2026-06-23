//! prism-server — OpenAI-compatible local API server for Prism Engine.
//!
//! Loads a compiled `.cimage` model and HuggingFace tokenizer, then serves
//! the OpenAI `/v1/chat/completions` and `/v1/completions` endpoints.
//!
//! Usage:
//!   cargo run --release -p tribunus-compute-core --bin prism-server \
//!       --features metal-dispatch -- \
//!       --cimage /tmp/prism-test/model.cimage \
//!       --model-dir models/qwen2.5-0.5b \
//!       --port 8080

use std::path::PathBuf;
use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use clap::Parser;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use prism_engine::lut::engine::PrismEngine;
use prism_engine::lut::graph::{ModelGraph, UnifiedConfig};
use prism_engine::tokenizer::TribunusTokenizer;

// ── CLI ─────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "prism-server", about = "OpenAI-compatible local API")]
struct Args {
    /// Path to the compiled .cimage.
    #[arg(long)]
    cimage: PathBuf,

    /// Path to the model directory (for config.json + tokenizer.json).
    #[arg(long)]
    model_dir: PathBuf,

    /// Server port.
    #[arg(long, default_value = "8080")]
    port: u16,
}

// ── State ───────────────────────────────────────────────────────────────

struct AppState {
    engine: PrismEngine,
    graph: ModelGraph,
    tokenizer: TribunusTokenizer,
}

// ── OpenAI API types ────────────────────────────────────────────────────

#[derive(Deserialize)]
struct ChatRequest {
    model: Option<String>,
    messages: Vec<ChatMessage>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    stream: Option<bool>,
}

#[derive(Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ChatResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<Choice>,
    usage: Usage,
}

#[derive(Serialize)]
struct Choice {
    index: u32,
    message: ChatResponseMessage,
    finish_reason: String,
}

#[derive(Serialize)]
struct ChatResponseMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

#[derive(Serialize)]
struct ModelList {
    object: String,
    data: Vec<ModelInfo>,
}

#[derive(Serialize)]
struct ModelInfo {
    id: String,
    object: String,
    created: u64,
    owned_by: String,
}

// ── Handlers ────────────────────────────────────────────────────────────

async fn list_models(State(state): State<Arc<Mutex<AppState>>>) -> Json<ModelList> {
    Json(ModelList {
        object: "list".to_string(),
        data: vec![ModelInfo {
            id: "prism-model".to_string(),
            object: "model".to_string(),
            created: 0,
            owned_by: "prism".to_string(),
        }],
    })
}

async fn chat_completions(
    State(state): State<Arc<Mutex<AppState>>>,
    Json(req): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, StatusCode> {
    if req.stream.unwrap_or(false) {
        return Err(StatusCode::NOT_IMPLEMENTED);
    }

    let mut state = state.lock().await;

    // Build prompt from messages (simple concatenation for now)
    let prompt_text: String = req.messages.iter()
        .map(|m| format!("{}: {}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("\n");

    // Tokenize
    let input_ids = state.tokenizer.encode(&prompt_text)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    if input_ids.is_empty() {
        return Err(StatusCode::BAD_REQUEST);
    }

    let max_tokens = req.max_tokens.unwrap_or(256) as usize;
    let prompt_len = input_ids.len();

    let stats = state.engine.generate(&input_ids, max_tokens)
        .map_err(|e| { eprintln!("[prism-server] generate error: {e}"); StatusCode::INTERNAL_SERVER_ERROR })?;

    let output_text = state.tokenizer.decode(&stats.generated_tokens)
        .unwrap_or_else(|_| format!("[prism] {} tokens generated", stats.generated_tokens.len()));

    Ok(Json(ChatResponse {
        id: "cmpl-1".to_string(),
        object: "chat.completion".to_string(),
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        model: req.model.unwrap_or_else(|| "prism".to_string()),
        choices: vec![Choice {
            index: 0,
            message: ChatResponseMessage {
                role: "assistant".to_string(),
                content: output_text,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens: prompt_len as u32,
            completion_tokens: stats.generated_tokens.len() as u32,
            total_tokens: (prompt_len + stats.generated_tokens.len()) as u32,
        },
    }))
}

// ── Main ────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), String> {
    let args = Args::parse();

    println!("[prism-server] Loading config from {}/config.json", args.model_dir.display());
    let config_path = args.model_dir.join("config.json");
    let config = UnifiedConfig::from_file(&config_path)?;

    println!("[prism-server] Building model graph ({} layers)...", config.num_layers);
    let graph = ModelGraph::build(&config);

    println!("[prism-server] Loading .cimage from {}...", args.cimage.display());
    let mut engine = PrismEngine::load(&args.cimage, graph.clone())?;
    #[cfg(feature = "metal-dispatch")]
    {
        if engine.with_metal().is_err() {
            eprintln!("[prism-server] Metal acceleration not available, using CPU");
        }
    }

    println!("[prism-server] Loading tokenizer from {}...", args.model_dir.display());
    let tokenizer = TribunusTokenizer::from_dir(&args.model_dir)?;

    let state = Arc::new(Mutex::new(AppState { engine, graph, tokenizer }));

    let app = Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(chat_completions))
        .with_state(state);

    let addr = format!("0.0.0.0:{}", args.port);
    println!("[prism-server] Listening on http://{}", addr);
    println!("[prism-server] Try: curl http://localhost:{}/v1/models", args.port);

    let listener = tokio::net::TcpListener::bind(&addr).await
        .map_err(|e| format!("bind: {e}"))?;
    axum::serve(listener, app).await
        .map_err(|e| format!("serve: {e}"))?;

    Ok(())
}
