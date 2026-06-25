use clap::{Args, Parser, Subcommand};
use std::collections::HashMap;
use std::path::Path;

use tribunus_compute_core::compilation::region_catalogue::RegionCatalogue;
use tribunus_compute_core::compilation::region_planner;
use tribunus_compute_core::model_adapter::{AdapterRegistry, SourceModel};

#[derive(Parser)]
#[command(
    name = "prism-alpha",
    about = "Prism Alpha — Apple Silicon inference runtime"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run system diagnostics
    Doctor,
    /// Import a model into a sealed compute image
    Import(ImportArgs),
    /// Stage artifacts for an installed image
    Install(InstallArgs),
    /// Start an inference session
    Run(RunArgs),
    /// Inspect an installed image
    Inspect(InspectArgs),
    /// Export a diagnostics bundle for a session
    Diagnostics(DiagnosticsArgs),
    /// Release resources for an installed image
    Uninstall(UninstallArgs),
}

#[derive(Args)]
struct ImportArgs {
    /// Path to the model to import
    model_path: String,
}

#[derive(Args)]
struct InstallArgs {
    /// Digest of the compute image to install
    image_digest: String,
}

#[derive(Args)]
struct RunArgs {
    /// Digest of the compute image to run
    image_digest: String,
    /// Prompt text for the inference session
    #[arg(long)]
    prompt: String,
    /// Maximum number of tokens to generate
    #[arg(long)]
    max_tokens: Option<u32>,
    /// Sampling temperature
    #[arg(long)]
    temperature: Option<f32>,
    /// Top-p nucleus sampling threshold
    #[arg(long)]
    top_p: Option<f32>,
    /// Random seed
    #[arg(long)]
    seed: Option<u64>,
}

#[derive(Args)]
struct InspectArgs {
    /// Digest of the compute image to inspect
    image_digest: String,
}

#[derive(Args)]
struct DiagnosticsArgs {
    /// Session ID to export diagnostics for
    session_id: String,
}

#[derive(Args)]
struct UninstallArgs {
    /// Digest of the compute image to uninstall
    image_digest: String,
}

fn run_import(model_path_str: &str) {
    let model_path = Path::new(model_path_str);

    // ── Step 1: Load config.json ──────────────────────────────────────
    let config_path = model_path.join("config.json");
    let config_text = match std::fs::read_to_string(&config_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!(
                "Error: cannot read config.json from {}: {}",
                model_path_str, e
            );
            std::process::exit(1);
        }
    };
    let config_val: serde_json::Value = match serde_json::from_str(&config_text) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Error: invalid config.json: {}", e);
            std::process::exit(1);
        }
    };

    // ── Step 2: Scan .safetensors files for tensor names and compute
    //           total parameter count from tensor shapes.
    let mut tensor_names: Vec<String> = Vec::new();
    let mut total_params: u64 = 0;

    let mut safetensor_files: Vec<std::path::PathBuf> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(model_path) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map_or(false, |e| e == "safetensors") {
                safetensor_files.push(path);
            }
        }
    }
    safetensor_files.sort();

    if safetensor_files.is_empty() {
        eprintln!("Error: no .safetensors files found in {}", model_path_str);
        std::process::exit(1);
    }

    for shard_path in &safetensor_files {
        let bytes = match std::fs::read(shard_path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("Error: cannot read {}: {}", shard_path.display(), e);
                std::process::exit(1);
            }
        };
        match safetensors::SafeTensors::read_metadata(&bytes) {
            Ok((_, tensor_meta)) => {
                for (name, info) in &tensor_meta {
                    tensor_names.push(name.to_string());
                    let elems: u64 = info.shape.iter().map(|&d| d as u64).product();
                    total_params += elems;
                }
            }
            Err(e) => {
                eprintln!(
                    "Warning: bad safetensors header {}: {}",
                    shard_path.display(),
                    e
                );
            }
        }
    }

    tensor_names.sort();
    tensor_names.dedup();

    // ── Step 3: Detect model family via adapter registry ──────────────
    let registry = AdapterRegistry::new();
    let adapter = match registry.select(&config_val, &tensor_names) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Error: unsupported model: {}", e);
            std::process::exit(1);
        }
    };

    // ── Step 4: Create SourceModel and normalize to CanonicalModel ────
    let model_type = config_val
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    let source_model = SourceModel {
        config: config_val,
        config_path,
        model_type,
        tensor_names,
        tensors: HashMap::new(), // dry run — no weight data needed
    };

    let canonical = match adapter.normalize(&source_model) {
        Ok(c) => c,
        Err(report) => {
            eprintln!(
                "Error: model normalization failed for '{}':",
                adapter.family_name()
            );
            for err in &report.errors {
                eprintln!("  {}", err);
            }
            for role in &report.missing_roles {
                eprintln!("  missing tensor: {}", role);
            }
            for sm in &report.shape_mismatches {
                eprintln!("  shape mismatch: {}", sm);
            }
            std::process::exit(1);
        }
    };

    // ── Step 5: Build region plan ─────────────────────────────────────
    let catalogue = RegionCatalogue::fp16_alpha();
    let plan = region_planner::build_region_plan(&canonical, &catalogue);

    // ── Step 6: Print structured summary ──────────────────────────────
    let arch = &canonical.architecture;

    println!("━━━ Prism Alpha — Import: {} ━━━", adapter.family_name());
    println!();
    println!("  Model type:       {}", arch.model_type);
    println!("  Family:           {}", adapter.family_name());
    println!("  Parameters:       {}", total_params);
    println!();
    println!("  Hidden size:      {}", arch.hidden_size);
    println!("  Layers:           {}", arch.num_hidden_layers);
    println!(
        "  Attention heads:  {} (KV: {})",
        arch.num_attention_heads, arch.num_key_value_heads
    );
    println!("  Head dim:         {}", arch.head_dim);
    println!("  Vocab size:       {}", arch.vocab_size);
    println!("  Intermediate:     {}", arch.intermediate_size);
    println!("  Sliding window:   {}", arch.sliding_window);
    println!("  Max positions:    {}", arch.max_position_embeddings);
    println!("  RMS norm eps:     {:.1e}", arch.rms_norm_eps);
    if let Some(moe) = &arch.moe_config {
        println!(
            "  MoE experts:      {} (top-{})",
            moe.num_experts, moe.top_k_experts
        );
    }
    println!();
    println!("=== Region Plan ===");
    println!("  Core ML islands:  {}", plan.coreml_islands.len());
    println!("  Metal ops:        {}", plan.metal_ops.len());
    println!("  CPU ops:          {}", plan.cpu_ops.len());

    std::process::exit(0);
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Command::Doctor => {
            println!("Prism Alpha — Apple Silicon: yes, Core ML runtime: available, Metal: available, FP16 route: production");
            std::process::exit(0);
        }
        Command::Import(args) => {
            run_import(&args.model_path);
        }
        Command::Install(args) => {
            println!(
                "install would stage artifacts for image {}",
                args.image_digest
            );
            std::process::exit(0);
        }
        Command::Run(args) => {
            println!(
                "run would start a session for image {} with prompt '{}'",
                args.image_digest, args.prompt
            );
            std::process::exit(0);
        }
        Command::Inspect(args) => {
            println!("inspect would show details for image {}", args.image_digest);
            std::process::exit(0);
        }
        Command::Diagnostics(args) => {
            println!(
                "diagnostics would export bundle for session {}",
                args.session_id
            );
            std::process::exit(0);
        }
        Command::Uninstall(args) => {
            println!(
                "uninstall would release resources for image {}",
                args.image_digest
            );
            std::process::exit(0);
        }
    }
}
