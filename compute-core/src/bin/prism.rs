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

use clap::{Parser, Subcommand, ValueEnum};
use std::path::{Path, PathBuf};
mod release;

use tribunus_compute_core::compute_image::manifest::CompilationAuthority;
use tribunus_compute_core::config::CompileQuantMode;
use tribunus_compute_core::config::HardwareTarget;

pub const PRISM_HOME: &str = ".prism";
pub const MODELS_DIR: &str = "models";

fn prism_home() -> PathBuf {
    let home = std::env::var("PRISM_HOME").unwrap_or_else(|_| {
        format!(
            "{}/{}",
            std::env::var("HOME").unwrap_or_else(|_| "/tmp".into()),
            PRISM_HOME
        )
    });
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

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliAuthority {
    #[clap(name = "test-fixture")]
    TestFixture,
    #[clap(name = "sealed")]
    Sealed,
}

impl From<CliAuthority> for CompilationAuthority {
    fn from(a: CliAuthority) -> Self {
        match a {
            CliAuthority::TestFixture => CompilationAuthority::TestFixture,
            CliAuthority::Sealed => CompilationAuthority::SealedComputeImage,
        }
    }
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
    /// Compile a GGUF model file to .cimage.
    CompileGGUF {
        /// Path to the GGUF file (e.g. ~/Downloads/model.gguf).
        gguf_path: PathBuf,

        /// Output path for the compiled .cimage.
        #[arg(short, long, default_value = "model.cimage")]
        output: PathBuf,

        /// Optional draft GGUF path for speculative decoding (MTP).
        #[arg(long)]
        draft: Option<PathBuf>,

        /// Compilation authority — controls validation gates.
        #[arg(long, default_value = "test-fixture", value_enum)]
        authority: CliAuthority,

        /// Target hardware (e.g. m1, m1pro, m2, m2ultra, m3ultra). Auto-detected if omitted.
        #[arg(long)]
        target_hardware: Option<String>,

        /// Quantize mode (e.g. nf4, nf4-128, 8bit, ternary, tile640). Uses target default if omitted.
        #[arg(long)]
        quantize_mode: Option<String>,

        /// Skip model validation checks.
        #[arg(long)]
        skip_validation: bool,

        /// Use legacy LUT-based compilation path.
        #[arg(long)]
        legacy_lut: bool,

        /// Directory containing pre-compiled .mlmodelc bundles from the Swift/cross-compilation
        /// host.  When set, the compiler skips MIL generation and uses these instead.
        #[arg(long)]
        ane_models_dir: Option<PathBuf>,

        /// Path to a pre-compiled .metallib file (Metal inference kernels) from the
        /// Swift/cross-compilation host.  When set, the compiler skips xcrun metal
        /// and embeds this library directly.
        #[arg(long)]
        metallib_path: Option<PathBuf>,

        /// Directory containing MLX JIT-captured Metal source (generated.metal)
        /// for AOT compilation.  When set and generated.metal exists, it is
        /// compiled to .metallib via xcrun metal instead of using template
        /// kernels.
        #[arg(long)]
        mlx_capture_dir: Option<PathBuf>,

        /// Directory for MLX CUDA JIT cache (MLX_PTX_CACHE_DIR).  When set,
        /// MLX's CUDA backend writes compiled .ptx and source .cu files here
        /// during the trace, enabling AOT reuse on NVIDIA hardware.
        #[arg(long)]
        cuda_cache_dir: Option<PathBuf>,

        /// Directory for MLX ROCm JIT cache (MLX_HIP_CACHE_DIR).  When set,
        /// MLX's ROCm/hiprtc backend writes compiled .hsaco and source .hip
        /// files here during the trace, enabling AOT reuse on AMD hardware.
        #[arg(long)]
        rocm_cache_dir: Option<PathBuf>,

        /// Directory for MLX Level Zero JIT cache (MLX_L0_CACHE_DIR).  When set,
        /// MLX's Level Zero/ocloc backend writes compiled .spv and source .cl
        /// files here during the trace, enabling AOT reuse on Intel GPUs.
        #[arg(long)]
        l0_cache_dir: Option<PathBuf>,
    },
    /// Compile ANE subgraphs for an already-downloaded model.
    AncCompile {
        /// Model name (must exist in ~/.prism/models/<name>/).
        model: String,
    },
    /// Qualification placeholder for a model.
    Qualify {
        /// Model name to qualify.
        model: String,
        /// Optional output path for the qualification report.
        output: Option<PathBuf>,
    },
    /// Run diagnostic checks on the runtime and environment.
    Doctor {
        /// Verify the provider backend is healthy.
        #[arg(long)]
        verify_provider: bool,
    },
    /// List available capabilities from installed release manifests.
    Capabilities,
    /// Collect a full diagnostic bundle.
    Diagnostics {
        /// Output path for the diagnostic archive.
        output: PathBuf,
        /// Include potentially sensitive information (env vars, paths).
        #[arg(long)]
        include_sensitive: bool,
    },
    /// Roll back to the previous release via versioned install dir.
    Rollback,
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Run { model, port } => run(model, port),
        Commands::List => list(),
        Commands::Pull { repo } => pull(&repo),
        Commands::Compile { model } => compile_model(&model),
        Commands::CompileGGUF {
            gguf_path,
            output,
            draft,
            authority,
            target_hardware,
            quantize_mode,
            skip_validation,
            legacy_lut,
            ane_models_dir,
            metallib_path,
            mlx_capture_dir,
            cuda_cache_dir,
            rocm_cache_dir,
            l0_cache_dir,
        } => compile_gguf(
            &gguf_path,
            &output,
            draft.as_deref(),
            authority.into(),
            target_hardware.as_deref(),
            quantize_mode.as_deref(),
            skip_validation,
            legacy_lut,
            ane_models_dir.as_deref(),
            metallib_path.as_deref(),
            mlx_capture_dir.as_deref(),
            cuda_cache_dir.as_deref(),
            rocm_cache_dir.as_deref(),
            l0_cache_dir.as_deref(),
        ),
        #[cfg(feature = "prism-backend")]
        Commands::AncCompile { model } => ane_compile(&model),
        #[cfg(not(feature = "prism-backend"))]
        Commands::AncCompile { model: _ } => {
            eprintln!("[prism] ANE compilation requires the `prism-backend` feature");
        }
        Commands::Qualify { model, output } => qualify(&model, output),
        Commands::Doctor { verify_provider } => doctor(verify_provider),
        Commands::Capabilities => capabilities(),
        Commands::Diagnostics {
            output,
            include_sensitive,
        } => diagnostics(&output, include_sensitive),
        Commands::Rollback => rollback(),
    }
}

fn find_model(name: &str) -> Option<PathBuf> {
    let path = models_dir().join(name);
    if path.join("model.cimage").exists() {
        Some(path)
    } else {
        None
    }
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
            if !p.is_dir() {
                continue;
            }
            let name = p
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            let cim = p.join("model.cimage");
            if cim.exists() {
                let size = std::fs::metadata(&cim)
                    .map(|m| m.len() / (1024 * 1024))
                    .unwrap_or(0);
                println!("  {:<24} {} MB  (compiled)", name, size);
            } else {
                let has_src =
                    p.join("config.json").exists() && p.join("model.safetensors").exists();
                let has_shards = p.join("model.safetensors.index.json").exists();
                if has_src || has_shards {
                    println!(
                        "  {:<24}           (source — run `prism compile {}`)",
                        name, name
                    );
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
            let mut candidates: Vec<_> = std::fs::read_dir(&dir)
                .ok()
                .into_iter()
                .flatten()
                .flatten()
                .filter(|e| e.path().join("model.cimage").exists())
                .collect();
            candidates.sort_by_key(|e| e.file_name().to_os_string());
            candidates
                .into_iter()
                .next()
                .map(|e| e.path())
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
        .arg("--cimage")
        .arg(model_path.join("model.cimage"))
        .arg("--model-dir")
        .arg(&model_path)
        .arg("--port")
        .arg(port.to_string())
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
        Err(e) => {
            eprintln!("failed: {e}");
            std::process::exit(1);
        }
    };
    std::fs::copy(&cfg_path, out_dir.join("config.json")).ok();
    eprintln!("ok");

    let cfg =
        tribunus_compute_core::lut::graph::UnifiedConfig::from_file(&out_dir.join("config.json"))
            .unwrap_or_else(|e| {
                eprintln!("config: {e}");
                std::process::exit(1);
            });
    let graph = tribunus_compute_core::lut::graph::ModelGraph::build(&cfg);
    eprintln!(
        "  Graph: {} layers, {} nodes",
        graph.num_layers,
        graph.nodes.len()
    );

    // 3. Download tokenizer files.
    eprint!("  [2/4] tokenizer... ");
    for f in &[
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    ] {
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
        std::fs::copy(
            &index_path,
            safetensors_dir.join("model.safetensors.index.json"),
        )
        .ok();
        let index_text = std::fs::read_to_string(&index_path).unwrap_or_default();
        if let Ok(index) = serde_json::from_str::<serde_json::Value>(&index_text) {
            if let Some(weight_map) = index.get("weight_map").and_then(|m| m.as_object()) {
                let mut shard_set: Vec<&str> =
                    weight_map.values().filter_map(|v| v.as_str()).collect();
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
        &graph,
        &safetensors_dir,
        &out_cimage,
        &out_dir.join("config.json"),
    ) {
        eprintln!("Compilation failed: {e}");
        std::process::exit(1);
    }

    let size_mb = std::fs::metadata(&out_cimage)
        .map(|m| m.len() / (1024 * 1024))
        .unwrap_or(0);
    eprintln!("[prism:pull] Done — {name} ({size_mb} MB)");
    eprintln!("  Run: prism run {name}");

    // Optional: compile ANE subgraphs.
    // On Apple Silicon, automatically compile ANE/Core ML subgraphs.
    #[cfg(all(target_os = "macos", feature = "prism-backend"))]
    ane_compile(&name);
}

/// Recompile an existing model's safetensors (already downloaded) to .cimage.
fn compile_model(name: &str) {
    let dir = model_dir(name);
    if !dir.join("config.json").exists() {
        eprintln!(
            "No config.json in {}. First: prism pull <repo>",
            dir.display()
        );
        std::process::exit(1);
    }

    let safetensors_dir = if dir.join("weights").exists() {
        dir.join("weights")
    } else {
        dir.clone()
    };

    if !safetensors_dir
        .read_dir()
        .ok()
        .map(|rd| {
            rd.flatten().any(|e| {
                e.path()
                    .extension()
                    .map_or(false, |ext| ext == "safetensors")
            })
        })
        .unwrap_or(false)
    {
        eprintln!(
            "No safetensors found in {}. Re-pull the model.",
            safetensors_dir.display()
        );
        std::process::exit(1);
    }

    eprint!(
        "[prism:compile] Building graph from {}... ",
        dir.join("config.json").display()
    );
    let cfg = tribunus_compute_core::lut::graph::UnifiedConfig::from_file(&dir.join("config.json"))
        .unwrap_or_else(|e| {
            eprintln!("config error: {e}");
            std::process::exit(1);
        });
    let graph = tribunus_compute_core::lut::graph::ModelGraph::build(&cfg);
    eprintln!("{} layers, {} nodes", graph.num_layers, graph.nodes.len());

    let out = cimage_path(name);
    eprintln!("[prism:compile] Compiling to {}...", out.display());
    match tribunus_compute_core::lut::compiler::compile_to_cimage(
        &graph,
        &safetensors_dir,
        &out,
        &dir.join("config.json"),
    ) {
        Ok(()) => {
            let size = std::fs::metadata(&out)
                .map(|m| m.len() / (1024 * 1024))
                .unwrap_or(0);
            eprintln!("[prism:compile] Done — {name} ({size} MB)");
            eprintln!("  Run: prism run {name}");
        }
        Err(e) => eprintln!("Compilation failed: {e}"),
    }
}
/// Compile a GGUF model file to .cimage.
fn compile_gguf(
    gguf_path: &Path,
    output_path: &Path,
    draft: Option<&Path>,
    authority: CompilationAuthority,
    raw_target: Option<&str>,
    raw_quant: Option<&str>,
    _skip_validation: bool,
    legacy_lut: bool,
    ane_models_dir: Option<&Path>,
    metallib_path: Option<&Path>,
    mlx_capture_dir: Option<&Path>,
    cuda_cache_dir: Option<&Path>,
    rocm_cache_dir: Option<&Path>,
    l0_cache_dir: Option<&Path>,
) {
    if !gguf_path.exists() {
        eprintln!("GGUF file not found: {}", gguf_path.display());
        std::process::exit(1);
    }

    if draft.is_some() && legacy_lut {
        eprintln!("error: --draft and --legacy-lut are mutually exclusive");
        std::process::exit(1);
    }

    // Set MLX CUDA PTX cache dir before any MLX operations.
    // MLX's CUDA backend reads MLX_PTX_CACHE_DIR at Device init and writes
    // compiled .ptx, .cu, and .txt files to this directory.
    if let Some(cuda_dir) = cuda_cache_dir {
        std::env::set_var("MLX_PTX_CACHE_DIR", cuda_dir);
        eprintln!("[prism:cuda] MLX_PTX_CACHE_DIR = {}", cuda_dir.display());
    }

    // Set MLX ROCm HIP cache dir before any MLX operations.
    // MLX's ROCm backend reads MLX_HIP_CACHE_DIR at device init and writes
    // compiled .hsaco, .hip, and .txt files to this directory.
    if let Some(rocm_dir) = rocm_cache_dir {
        std::env::set_var("MLX_HIP_CACHE_DIR", rocm_dir);
        eprintln!("[prism:rocm] MLX_HIP_CACHE_DIR = {}", rocm_dir.display());
    }

    // Set MLX Level Zero cache dir for ocloc JIT artifacts.
    // MLX's Level Zero backend reads MLX_L0_CACHE_DIR and writes
    // compiled .spv and source .cl files to this directory.
    if let Some(l0_dir) = l0_cache_dir {
        std::env::set_var("MLX_L0_CACHE_DIR", l0_dir);
        eprintln!("[prism:l0] MLX_L0_CACHE_DIR = {}", l0_dir.display());
    }

    // Parse optional target and quantize mode into their rich types.
    let target = raw_target.and_then(|t| match t.to_lowercase().as_str() {
        "m1" => Some(HardwareTarget::M1),
        "m1pro" => Some(HardwareTarget::M1Pro),
        "m2" => Some(HardwareTarget::M2),
        "m2ultra" => Some(HardwareTarget::M2Ultra),
        "m3ultra" => Some(HardwareTarget::M3Ultra),
        other => {
            eprintln!("warning: unknown target '{}', using auto-detect", other);
            None
        }
    });

    let quantize_mode = raw_quant.and_then(|q| match CompileQuantMode::from_name(q) {
        Some(qm) => Some(qm),
        None => {
            eprintln!(
                "warning: unknown quantize mode '{}', using target default",
                q
            );
            None
        }
    });

    eprintln!(
        "[prism:compile-gguf] Compiling {} → {}",
        gguf_path.display(),
        output_path.display()
    );

    // ── Legacy LUT path (always available) ─────────────────────────
    if legacy_lut {
        match tribunus_compute_core::lut::compiler::compile_gguf_to_cimage(gguf_path, output_path) {
            Ok(()) => {
                let size = std::fs::metadata(output_path)
                    .map(|m| m.len() / (1024 * 1024))
                    .unwrap_or(0);
                eprintln!("[prism:compile-gguf] Done — {} MB", size);
            }
            Err(e) => {
                eprintln!("Compilation failed: {e}");
                std::process::exit(1);
            }
        }
        return;
    }

    // ── New GGUF pipeline requires prism-backend ───────────────────
    #[cfg(not(feature = "prism-backend"))]
    {
        eprintln!(
            "[prism:compile-gguf] New GGUF pipeline requires --features prism-backend. Use --legacy-lut to fall back."
        );
        std::process::exit(1);
    }

    #[cfg(feature = "prism-backend")]
    {
        let new_pipeline_result = if let Some(draft_path) = draft {
            // Speculative compilation with draft GGUF.
            let gguf_str = gguf_path.to_string_lossy();
            let draft_str = draft_path.to_string_lossy();
            let out_str = output_path.to_string_lossy();
            tribunus_compute_core::compute_image::compile::compile_gguf_speculative(
                &gguf_str,
                &draft_str,
                &out_str,
                authority,
                quantize_mode,
                target,
            )
            .map(|_| ())
        } else if matches!(authority, CompilationAuthority::SealedComputeImage) {
            // Authority-gated compilation.
            let gguf_str = gguf_path.to_string_lossy();
            let out_str = output_path.to_string_lossy();
            let ane_str = ane_models_dir.map(|p| p.to_string_lossy().into_owned());
            let metal_str = metallib_path.map(|p| p.to_string_lossy().into_owned());
            let mlx_str = mlx_capture_dir.map(|p| p.to_string_lossy().into_owned());
            tribunus_compute_core::compute_image::compile::compile_gguf_with_authority(
                &gguf_str,
                &out_str,
                authority,
                quantize_mode,
                target,
                ane_str.as_deref(),
                metal_str.as_deref(),
                mlx_str.as_deref(),
            )
            .map(|_| ())
        } else {
            // Unchecked default path.
            let gguf_str = gguf_path.to_string_lossy();
            let out_str = output_path.to_string_lossy();
            let ane_str = ane_models_dir.map(|p| p.to_string_lossy().into_owned());
            let metal_str = metallib_path.map(|p| p.to_string_lossy().into_owned());
            let mlx_str = mlx_capture_dir.map(|p| p.to_string_lossy().into_owned());
            tribunus_compute_core::compute_image::compile::compile_gguf_unchecked(
                &gguf_str,
                &out_str,
                quantize_mode,
                ane_str.as_deref(),
                metal_str.as_deref(),
                mlx_str.as_deref(),
            )
            .map(|_| ())
        };

        match new_pipeline_result {
            Ok(()) => {
                let size = std::fs::metadata(output_path)
                    .map(|m| m.len() / (1024 * 1024))
                    .unwrap_or(0);
                eprintln!("[prism:compile-gguf] Done — {} MB", size);
            }
            Err(e) => {
                eprintln!("Compilation failed: {e}");
                std::process::exit(1);
            }
        }
    }
}

/// Qualification stub — not yet implemented.
fn qualify(model: &str, output: Option<PathBuf>) {
    eprintln!("[prism] Qualification not yet implemented");
    if let Some(path) = output {
        eprintln!("  (output would be written to {})", path.display());
    }
    eprintln!("  Model: {model}");
}

/// Doctor — runtime diagnostic checks.
fn doctor(verify_provider: bool) {
    eprintln!("Prism Engine Diagnostic Report");
    eprintln!("  Version:              {}", env!("CARGO_PKG_VERSION"));
    eprintln!("  Home:                 {}", prism_home().display());
    eprintln!("  Models dir:           {}", models_dir().display());
    eprintln!(
        "  Releases dir:         {}",
        release::VersionedInstallDir::releases_dir().display()
    );
    eprintln!(
        "  Current link:         {}",
        release::VersionedInstallDir::current_link().display()
    );
    eprintln!(
        "  Previous link:        {}",
        release::VersionedInstallDir::previous_link().display()
    );

    if verify_provider {
        eprintln!("  Provider verification: not yet implemented");
    }
}

/// Capabilities — enumerate what the installed release supports.
fn capabilities() {
    let releases_dir = release::VersionedInstallDir::releases_dir();
    if !releases_dir.exists() {
        eprintln!("No release manifests found at {}", releases_dir.display());
        return;
    }

    match std::fs::read_dir(&releases_dir) {
        Ok(entries) => {
            let mut found = false;
            for entry in entries.flatten() {
                let manifest_path = entry.path().join("release.json");
                if !manifest_path.exists() {
                    continue;
                }
                match std::fs::read_to_string(&manifest_path) {
                    Ok(text) => {
                        match serde_json::from_str::<release::PrismReleaseManifest>(&text) {
                            Ok(m) => {
                                println!(
                                    "  {} [{:?}] — prism {} | compute-core {}",
                                    m.release_version,
                                    m.channel,
                                    m.prism_version,
                                    m.compute_core_version
                                );
                                for plat in &m.supported_platforms {
                                    println!(
                                        "      platform: {}/{} (min_macos: {:?})",
                                        plat.os, plat.arch, plat.min_macos_version
                                    );
                                }
                                println!(
                                    "      schemas: {:?} | checksums: {} | status: {:?}",
                                    m.artifact_schema_versions,
                                    m.checksums.len(),
                                    m.signing_status
                                );
                                if let Some(digest) = &m.compatibility_manifest_digest {
                                    println!("      compat digest: {digest:?}");
                                }
                                found = true;
                            }
                            Err(e) => {
                                println!(
                                    "  (invalid manifest at {}: {e})",
                                    manifest_path.display()
                                );
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("  Error reading {}: {e}", manifest_path.display());
                    }
                }
            }
            if !found {
                eprintln!("No valid release manifests found.");
            }
        }
        Err(e) => {
            eprintln!("Error reading releases dir: {e}");
        }
    }
}

/// Diagnostics — collect a diagnostic bundle.
fn diagnostics(output: &PathBuf, include_sensitive: bool) {
    eprintln!("[prism] Diagnostics collection not yet implemented");
    eprintln!("  Output path: {}", output.display());
    eprintln!("  Include sensitive: {include_sensitive}");
}

/// Compile ANE subgraphs for a downloaded model.
#[allow(dead_code)]
fn ane_compile(name: &str) {
    let dir = model_dir(name);
    if !dir.join("config.json").exists() {
        eprintln!(
            "No config.json in {}. First: prism pull <repo>",
            dir.display()
        );
        std::process::exit(1);
    }

    eprintln!("[prism:ane] Compiling ANE subgraphs for {name}...");

    #[cfg(feature = "prism-backend")]
    {
        use tribunus_compute_core::ane_compile;
        match ane_compile::compile_ane_artifacts(&dir) {
            Ok(paths) => {
                eprintln!("[prism:ane] Generated {} .mlmodelc file(s):", paths.len());
                for p in &paths {
                    eprintln!("    {p}");
                }
            }
            Err(e) => eprintln!("[prism:ane] ANE compilation failed: {e}"),
        }
    }
}

/// Rollback — revert to previous release.
fn rollback() {
    match release::VersionedInstallDir::rollback() {
        Ok(()) => {
            let current = release::VersionedInstallDir::current_link();
            println!(
                "Rolled back. Current release link: {:?}",
                current.read_link().unwrap_or(current)
            );
        }
        Err(e) => {
            eprintln!("[prism] Rollback failed: {e}");
            std::process::exit(1);
        }
    }
}
