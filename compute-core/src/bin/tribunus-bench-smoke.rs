//! Benchmark harness for smoke-testing the worker IPC path end-to-end.
//!
//! Spawns a worker process, loads the model, tokenizes a prompt, runs
//! generation, and reports metrics as CSV to stdout.
//!
//! Usage:
//!   tribunus-bench-smoke --model-path <path> \
//!                        [--prompt <text>] \
//!                        [--max-tokens <N>]
//!
//! The worker binary is resolved from `TRIBUNUS_WORKER_BINARY` if set,
//! otherwise discovered alongside the current executable.

use std::path::Path;
use std::time::Instant;

use tribunus_compute_core::logging::{log_error, log_info};
use tribunus_compute_core::tokenizer::TribunusTokenizer;

fn print_usage() {
    eprintln!(
        "Usage: tribunus-bench-smoke --model-path <path> [--prompt <text>] [--max-tokens <N>]"
    );
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --model-path <dir>   Path to model image directory (required)");
    eprintln!("  --prompt <text>      Prompt text (default: Hello, world!)");
    eprintln!("  --max-tokens <N>     Max output tokens (default: 50)");
    eprintln!("  --help, -h           Show this help");
}

fn main() {
    // macOS workaround: unset MallocStackLogging inherited from Xcode/LLDB
    // to suppress "can't turn off malloc stack logging because it was not enabled"
    // on stderr during process exit, which corrupts terminal output.
    unsafe {
        std::env::set_var("MallocStackLogging", "0");
        std::env::set_var("MallocStackLoggingNoCompact", "0");
    }

    // ── Parse CLI args ───────────────────────────────────────────────
    let args: Vec<String> = std::env::args().collect();
    let mut model_path: Option<String> = None;
    let mut prompt = "Hello, world!".to_string();
    let mut max_tokens: u32 = 50;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model-path" => {
                i += 1;
                if let Some(p) = args.get(i) {
                    model_path = Some(p.clone());
                }
            }
            "--prompt" => {
                i += 1;
                if let Some(p) = args.get(i) {
                    prompt = p.clone();
                }
            }
            "--max-tokens" => {
                i += 1;
                if let Some(s) = args.get(i) {
                    max_tokens = s.parse().unwrap_or(50);
                }
            }
            "--help" | "-h" => {
                print_usage();
                return;
            }
            other => {
                eprintln!("error: unknown argument: {other}");
                print_usage();
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let model_path = match model_path {
        Some(p) => p,
        None => {
            eprintln!("error: --model-path is required");
            print_usage();
            std::process::exit(1);
        }
    };

    // Print CSV header and fixed fields.
    println!("metric,value,unit");
    println!("model_path,{},", &model_path);
    println!("prompt,{},", &prompt);
    println!("max_tokens,{},", max_tokens);

    // ── Validate model path ──────────────────────────────────────────
    let path = Path::new(&model_path);
    if !path.exists() {
        log_error!("Model path not found");
        println!("status,error,");
        println!("error,model path not found: {},", &model_path);
        return;
    }

    // ── Load tokenizer ───────────────────────────────────────────────
    let tokenizer_load_start = Instant::now();
    let tokenizer = match TribunusTokenizer::from_dir(path) {
        Ok(tok) => {
            let elapsed_ms = tokenizer_load_start.elapsed().as_millis() as u64;
            log_info!("Tokenizer loaded in {}ms", elapsed_ms);
            println!("tokenizer_load_ms,{},ms", elapsed_ms);
            println!("tokenizer_available,true,");
            Some(tok)
        }
        Err(e) => {
            log_error!("Failed to load tokenizer: {}", e);
            println!("tokenizer_load_ms,0,ms");
            println!("tokenizer_available,false,");
            None
        }
    };

    eprintln!("ECS-only mode — bench harness not wired");
    return;
}
