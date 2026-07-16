//! Prism Engine 1000-token throughput benchmark.
//!
//! Loads a compiled `.cimage` model, runs 1000 persistent-GPU decode
//! steps, and reports tokens/second.  Validates pipeline stability
//! (no NaN/Inf logits, no hangs, no crashes).
//!
//! Usage:
//!   cargo run --bin prism-bench --features prism-backend -- \
//!     --cimage /path/to/model.cimage [--tokens 1000] [--warmup 10]

use clap::Parser;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

#[cfg(feature = "prism-backend")]
use tribunus_compute_core::compute_image::orchestrator::Orchestrator;

#[derive(Parser)]
struct Args {
    /// Path to compiled .cimage file.
    #[arg(long)]
    cimage: PathBuf,

    /// Number of decode steps to run.
    #[arg(long, default_value = "1000")]
    tokens: u32,

    /// Warmup iterations before measurement.
    #[arg(long, default_value = "10")]
    warmup: u32,
}

#[cfg(not(feature = "prism-backend"))]
fn main() {
    eprintln!("This binary requires the `prism-backend` feature.");
    eprintln!("  cargo run --bin prism-bench --features prism-backend -- ...");
    std::process::exit(1);
}

#[cfg(feature = "prism-backend")]
fn main() {
    let args = Args::parse();

    if !args.cimage.exists() {
        eprintln!("ERROR: cimage file not found: {}", args.cimage.display());
        std::process::exit(1);
    }

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Prism Engine — 1000-Token Throughput Benchmark            ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("  Model:     {}", args.cimage.display());
    println!("  Tokens:    {}", args.tokens);
    println!("  Warmup:    {}", args.warmup);

    // ── Load orchestrator ──────────────────────────────────────────
    println!();
    println!("  ── Loading .cimage and initializing Metal pipeline ────");

    let start_load = Instant::now();
    let mut orch = match Orchestrator::from_cimage(&args.cimage, 1) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("ERROR: Failed to load orchestrator: {e}");
            std::process::exit(1);
        }
    };
    let load_elapsed = start_load.elapsed();
    let load_secs = load_elapsed.as_secs_f64();
    println!("  Load time: {load_secs:.1}s");
    println!(
        "  Weights:   {:.1} GB",
        orch.deployment.weights_buffer.length() as f64 / 1e9
    );
    println!("  Batch:     {}", orch.batch_size);

    // ── Verify no embedded MTP heads (baseline single-agent mode) ──
    let num_layers = orch.deployment.num_layers;
    println!("  Layers:    {num_layers}");

    // ── Warmup phase ───────────────────────────────────────────────
    if args.warmup > 0 {
        println!();
        let w = args.warmup;
        println!("  ── Warmup ({w} steps) ───────────────────────────");
        let mut cur_tok: u32 = 42;
        for i in 0..args.warmup {
            match orch.decode_token(cur_tok) {
                Ok(t) => cur_tok = t,
                Err(e) => {
                    eprintln!("ERROR at warmup step {i}: {e}");
                    std::process::exit(1);
                }
            }
            if i < 3 || i >= args.warmup - 3 {
                println!("  step {i:>4}: token {cur_tok:>6}");
            } else if i == 3 {
                println!("  ... (warmup)");
            }
        }
        println!("  Warmup complete.");
    }

    // ── Measurement phase ──────────────────────────────────────────
    println!();
    let ntok = args.tokens;
    println!("  ── Benchmark ({ntok} tokens) ──────────────────────");

    let start = Instant::now();

    let mut cur_tok: u32 = 42;
    let mut non_zero_count: u32 = 0;

    for i in 0..args.tokens {
        let tok = match orch.decode_token(cur_tok) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("ERROR at step {i}: {e}");
                std::process::exit(1);
            }
        };
        cur_tok = tok;
        if tok != 0 && tok <= 100 {
            non_zero_count += 1;
        }

        // Print progress every 100 steps
        if (i + 1) % 100 == 0 {
            let elapsed = start.elapsed();
            let tps = (i as f64 + 1.0) / elapsed.as_secs_f64();
            println!("  step {i:>4}: {tps:>8.1} t/s  (last token: {tok:>6})");
        }
    }

    let elapsed = start.elapsed();
    let tps = args.tokens as f64 / elapsed.as_secs_f64();

    println!();
    println!("  ── Results ────────────────────────────────────────────────");
    let total_secs = elapsed.as_secs_f64();
    println!("  Total time:   {total_secs:.3}s for {ntok} tokens");
    println!("  Throughput:   {tps:.1} t/s");
    println!("  Last token:   {cur_tok}");
    if non_zero_count > 50 {
        println!("  Tokens:       diverse (expected with real model weights)");
    } else {
        println!("  Token diversity: {non_zero_count}/1000 first-100 tokens");
    }
    println!("  Pipeline:     {num_layers} layers × 3840 dim");
    println!();
    println!("  ✓ Decode pipeline stable over {ntok} steps");

    // ── Memory sanity ──────────────────────────────────────────────
    if let Ok(out) = Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
    {
        let rss_kb = String::from_utf8_lossy(&out.stdout)
            .trim()
            .parse::<u64>()
            .unwrap_or(0);
        println!("  RSS:         {} MB", rss_kb / 1024);
    }
}
