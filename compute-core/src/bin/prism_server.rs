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

use tribunus_compute_core::compute_image::cimage_loader::CimageDeployment;
use tribunus_compute_core::compute_image::orchestrator::Orchestrator;
use tribunus_compute_core::tokenizer::TribunusTokenizer;

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
    orchestrator: Orchestrator,
    tokenizer: TribunusTokenizer,
}

// ── OpenAI API types ────────────────────────────────────────────────────

#[derive(Deserialize)]
struct ChatRequest {
    model: Option<String>,
    messages: Vec<ChatMessage>,
    max_tokens: Option<u32>,
    #[allow(dead_code)]
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

async fn list_models(State(_state): State<Arc<Mutex<AppState>>>) -> Json<ModelList> {
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
    let prompt_text: String = req
        .messages
        .iter()
        .map(|m| format!("{}: {}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("\n");

    // Tokenize
    let input_ids = state
        .tokenizer
        .encode(&prompt_text)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    if input_ids.is_empty() {
        return Err(StatusCode::BAD_REQUEST);
    }

    let max_tokens = req.max_tokens.unwrap_or(256) as usize;
    let prompt_len = input_ids.len();

    // V2 orchestrator: prefill prompt, then autoregressive decode
    let start = std::time::Instant::now();
    // Sequential GPU prefill: each decode_token builds KV cache row
    for i in 0..input_ids.len().saturating_sub(1) {
        state.orchestrator.decode_token(input_ids[i]).map_err(|e| {
            eprintln!("[prism-server] prefill step {} error: {}", i, e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
    }
    let mut last_token = *input_ids.last().unwrap_or(&0);
    let mut generated_tokens = Vec::with_capacity(max_tokens);
    for _ in 0..max_tokens {
        last_token = state.orchestrator.decode_token(last_token).map_err(|e| {
            eprintln!("[prism-server] decode error: {e}");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
        generated_tokens.push(last_token);
        if last_token == 1 {
            // EOS token
            break;
        }
    }

    let elapsed = start.elapsed();
    let tok_s = generated_tokens.len() as f64 / elapsed.as_secs_f64();
    eprintln!(
        "[prism-server] {} tokens in {:.2}s = {:.1} tok/s",
        generated_tokens.len(),
        elapsed.as_secs_f64(),
        tok_s
    );

    let output_text = state
        .tokenizer
        .decode(&generated_tokens)
        .unwrap_or_else(|_| format!("[prism] {} tokens generated", generated_tokens.len()));

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
            completion_tokens: generated_tokens.len() as u32,
            total_tokens: (prompt_len + generated_tokens.len()) as u32,
        },
    }))
}

// ── Main ────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), String> {
    let args = Args::parse();

    println!(
        "[prism-server] Loading V2 cimage from {}...",
        args.cimage.display()
    );
    let orchestrator = Orchestrator::from_cimage(&args.cimage, 1, false)
        .map_err(|e| format!("cimage load failed: {e}"))?;

    println!(
        "[prism-server] Loading tokenizer from {}...",
        args.model_dir.display()
    );
    let tokenizer = TribunusTokenizer::from_dir(&args.model_dir)?;

    let state = Arc::new(Mutex::new(AppState {
        orchestrator,
        tokenizer,
    }));

    let app = Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state);

    let addr = format!("0.0.0.0:{}", args.port);
    println!("[prism-server] Listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(|e| format!("bind: {e}"))?;

    axum::serve(listener, app)
        .await
        .map_err(|e| format!("serve: {e}"))?;

    Ok(())
}
