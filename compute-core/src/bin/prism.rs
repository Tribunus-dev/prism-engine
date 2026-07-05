//! prism — Ollama-like CLI for the Prism Engine.
//!
//! Subcommands:
//!   run <model>       Start OpenAI-compatible server with a compiled model
//!   list              Show available models in ~/.prism/models/
//!   pull <repo>       Download + compile a model from HuggingFace to .cimage
//!   compile <name>    Recompile an existing source model without re-downloading
//!
//! Model directory structure (~/.prism/models/<name>/):
//!   model.cimage       Compiled palettized weights
//!   config.json        HuggingFace model config
//!   tokenizer.json     HuggingFace tokenizer
//!   tokenizer_config.json

use std::path::PathBuf;
use clap::{Parser, Subcommand};

pub const PRISM_HOME: &str = ".prism";
pub const MODELS_DIR: &str = "models";

fn prism_home() -> PathBuf {
    let home = std::env::var("PRISM_HOME")
        .unwrap_or_else(|_| format!("{}/{}", std::env::var("HOME").unwrap_or_else(|_| "/tmp".into()), PRISM_HOME));
    PathBuf::from(home)
}

fn models_dir() -> PathBuf {
    prism_home().join(MODELS_DIR)
}

fn model_dir(name: &str) -> PathBuf {
    models_dir().join(name)
}

fn cimage_path(name: &str) -> PathBuf {
    model_dir(name).join("model.cimage")
}

#[derive(Parser)]
#[command(name = "prism", about = "Prism Engine CLI — local LLM inference")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the OpenAI-compatible server with a compiled model.
    Run {
        /// Model name (must exist in ~/.prism/models/<name>/).
        model: Option<String>,

        /// Server port.
        #[arg(long, default_value = "8080")]
        port: u16,
    },
    /// List available models.
    List,
    /// Download + compile a model from HuggingFace.
    Pull {
        /// HuggingFace repo ID (e.g. "Qwen/Qwen2.5-0.5B").
        repo: String,
    },
    /// Recompile an existing model's safetensors without re-downloading.
    Compile {
        /// Model name (must exist in ~/.prism/models/<name>/ with safetensors).
        model: String,
    },
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Run { model, port } => run(model, port),
        Commands::List => list(),
        Commands::Pull { repo } => pull(&repo),
        Commands::Compile { model } => compile_model(&model),
    }
}

fn find_model(name: &str) -> Option<PathBuf> {
    let path = models_dir().join(name);
    if path.join("model.cimage").exists() { Some(path) } else { None }
}

fn list() {
    let dir = models_dir();
    if !dir.exists() {
        eprintln!("No models found in {}", dir.display());
        eprintln!("  Pull one: prism pull <huggingface-repo>");
        return;
    }
    let mut found = false;
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if !p.is_dir() { continue; }
            let name = p.file_name().unwrap_or_default().to_string_lossy().to_string();
            let cim = p.join("model.cimage");
            if cim.exists() {
                let size = std::fs::metadata(&cim).map(|m| m.len() / (1024 * 1024)).unwrap_or(0);
                println!("  {:<24} {} MB  (compiled)", name, size);
            } else {
                let has_src = p.join("config.json").exists() && p.join("model.safetensors").exists();
                let has_shards = p.join("model.safetensors.index.json").exists();
                if has_src || has_shards {
                    println!("  {:<24}           (source — run `prism compile {}`)", name, name);
                }
            }
            found = true;
        }
    }
    if !found {
        eprintln!("No models found.");
    }
}

fn run(model: Option<String>, port: u16) {
    let model_path = match &model {
        Some(name) => find_model(name).unwrap_or_else(|| {
            eprintln!("Model '{name}' not found. Pull one: prism pull <repo>");
            std::process::exit(1);
        }),
        None => {
            let dir = models_dir();
            let mut candidates: Vec<_> = std::fs::read_dir(&dir).ok()
                .into_iter().flatten()
                .flatten()
                .filter(|e| e.path().join("model.cimage").exists())
                .collect();
            candidates.sort_by_key(|e| e.file_name().to_os_string());
            candidates.into_iter().next().map(|e| e.path())
                .unwrap_or_else(|| {
                    eprintln!("No models found. Pull one: prism pull <repo>");
                    std::process::exit(1);
                })
        }
    };

    let server = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join("prism-server")))
        .unwrap_or_else(|| "prism-server".into());

    let status = std::process::Command::new(&server)
        .arg("--cimage").arg(model_path.join("model.cimage"))
        .arg("--model-dir").arg(&model_path)
        .arg("--port").arg(port.to_string())
        .spawn()
        .and_then(|mut c| c.wait());

    match status {
        Ok(_) => {}
        Err(e) => eprintln!("Failed to start prism-server: {e} (is it on $PATH?)"),
    }
}

/// Download a model from HuggingFace and compile to .cimage.
fn pull(repo: &str) {
    let name = repo.split('/').last().unwrap_or(repo).to_lowercase();
    let out_dir = model_dir(&name);
    std::fs::create_dir_all(&out_dir).expect("create model dir");

    eprintln!("[prism:pull] Downloading {repo}...");

    // 1. Initialize HF hub API (caches to ~/.cache/huggingface/hub/).
    let api = match hf_hub::api::sync::Api::new() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Failed to init HF API: {e}");
            eprintln!("  Set HF_TOKEN env var if this is a gated model.");
            std::process::exit(1);
        }
    };
    let repo_api = api.model(repo.to_string());

    // 2. Download config.json -> parse graph.
    eprint!("  [1/4] config.json... ");
    let cfg_path = match repo_api.get("config.json") {
        Ok(p) => p,
        Err(e) => { eprintln!("failed: {e}"); std::process::exit(1); }
    };
    std::fs::copy(&cfg_path, out_dir.join("config.json")).ok();
    eprintln!("ok");

    let cfg = tribunus_compute_core::lut::graph::UnifiedConfig::from_file(&out_dir.join("config.json"))
        .unwrap_or_else(|e| { eprintln!("config: {e}"); std::process::exit(1); });
    let graph = tribunus_compute_core::lut::graph::ModelGraph::build(&cfg);
    eprintln!("  Graph: {} layers, {} nodes", graph.num_layers, graph.nodes.len());

    // 3. Download tokenizer files.
    eprint!("  [2/4] tokenizer... ");
    for f in &["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"] {
        if let Ok(p) = repo_api.get(f) {
            std::fs::copy(&p, out_dir.join(f)).ok();
        }
    }
    // Fallback: try tokenizer.model (SentencePiece/BPE).
    if !out_dir.join("tokenizer.json").exists() {
        if let Ok(p) = repo_api.get("tokenizer.model") {
            std::fs::copy(&p, out_dir.join("tokenizer.model")).ok();
        }
    }
    eprintln!("ok");

    // 4. Download safetensors shards.
    eprint!("  [3/4] weights... ");
    let safetensors_dir = out_dir.join("weights");
    std::fs::create_dir_all(&safetensors_dir).ok();

    // Check for single-file model.safetensors.
    let mut shard_count = 0usize;
    if let Ok(p) = repo_api.get("model.safetensors") {
        std::fs::copy(&p, safetensors_dir.join("model.safetensors")).ok();
        shard_count += 1;
    }
    // Check for sharded index.
    if let Ok(index_path) = repo_api.get("model.safetensors.index.json") {
        std::fs::copy(&index_path, safetensors_dir.join("model.safetensors.index.json")).ok();
        let index_text = std::fs::read_to_string(&index_path).unwrap_or_default();
        if let Ok(index) = serde_json::from_str::<serde_json::Value>(&index_text) {
            if let Some(weight_map) = index.get("weight_map").and_then(|m| m.as_object()) {
                let mut shard_set: Vec<&str> = weight_map.values()
                    .filter_map(|v| v.as_str())
                    .collect();
                shard_set.sort();
                shard_set.dedup();
                for shard_name in &shard_set {
                    if let Ok(p) = repo_api.get(shard_name) {
                        let dst = safetensors_dir.join(shard_name);
                        std::fs::copy(&p, &dst).ok();
                        shard_count += 1;
                    }
                }
            }
        }
    }
    // If neither found, try numbered shards directly.
    if shard_count == 0 {
        for i in 1..=10 {
            let name = format!("model-{:05}-of-{:05}.safetensors", i, 10);
            if let Ok(p) = repo_api.get(&name) {
                let dst = safetensors_dir.join(&name);
                std::fs::copy(&p, &dst).ok();
                shard_count += 1;
            } else {
                break;
            }
        }
    }
    if shard_count == 0 {
        eprintln!("no safetensors found — is this a supported model?");
        eprintln!("  Attempted: model.safetensors, model.safetensors.index.json, model-00001-of-*.safetensors");
        std::process::exit(1);
    }
    eprintln!("{} shard(s)", shard_count);

    // 5. Compile to .cimage.
    eprintln!("  [4/4] compiling... ");
    let out_cimage = cimage_path(&name);
    if let Err(e) = tribunus_compute_core::lut::compiler::compile_to_cimage(
        &graph, &safetensors_dir, &out_cimage,
    ) {
        eprintln!("Compilation failed: {e}");
        std::process::exit(1);
    }

    let size_mb = std::fs::metadata(&out_cimage).map(|m| m.len() / (1024 * 1024)).unwrap_or(0);
    eprintln!("[prism:pull] Done — {name} ({size_mb} MB)");
    eprintln!("  Run: prism run {name}");
}

/// Recompile an existing model's safetensors (already downloaded) to .cimage.
fn compile_model(name: &str) {
    let dir = model_dir(name);
    if !dir.join("config.json").exists() {
        eprintln!("No config.json in {}. First: prism pull <repo>", dir.display());
        std::process::exit(1);
    }

    let safetensors_dir = if dir.join("weights").exists() {
        dir.join("weights")
    } else {
        dir.clone()
    };

    if !safetensors_dir.read_dir().ok()
        .map(|rd| rd.flatten().any(|e| e.path().extension().map_or(false, |ext| ext == "safetensors")))
        .unwrap_or(false)
    {
        eprintln!("No safetensors found in {}. Re-pull the model.", safetensors_dir.display());
        std::process::exit(1);
    }

    eprint!("[prism:compile] Building graph from {}... ", dir.join("config.json").display());
    let cfg = tribunus_compute_core::lut::graph::UnifiedConfig::from_file(&dir.join("config.json"))
        .unwrap_or_else(|e| { eprintln!("config error: {e}"); std::process::exit(1); });
    let graph = tribunus_compute_core::lut::graph::ModelGraph::build(&cfg);
    eprintln!("{} layers, {} nodes", graph.num_layers, graph.nodes.len());

    let out = cimage_path(name);
    eprintln!("[prism:compile] Compiling to {}...", out.display());
    match tribunus_compute_core::lut::compiler::compile_to_cimage(&graph, &safetensors_dir, &out) {
        Ok(()) => {
            let size = std::fs::metadata(&out).map(|m| m.len() / (1024 * 1024)).unwrap_or(0);
            eprintln!("[prism:compile] Done — {name} ({size} MB)");
            eprintln!("  Run: prism run {name}");
        }
        Err(e) => eprintln!("Compilation failed: {e}"),
    }
}
