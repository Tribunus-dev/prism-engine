use std::path::Path;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cimage_path = args
        .iter()
        .position(|a| a == "--cimage")
        .and_then(|i| args.get(i + 1))
        .map(Path::new)
        .unwrap_or_else(|| Path::new("gemma4-12b-it.cimage"));

    let decode_tokens: usize = args
        .iter()
        .position(|a| a == "--decode-tokens")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);

    if !cimage_path.exists() {
        eprintln!("cimage not found: {}", cimage_path.display());
        eprintln!("Usage: tribunus-minimal-bench --cimage <path> [--decode-tokens N]");
        std::process::exit(1);
    }

    println!("Loading cimage: {}", cimage_path.display());
    let load_start = Instant::now();
    let deployment = tribunus_compute_core::compute_image::cimage_loader::CimageDeployment::load(cimage_path)
        .expect("Failed to load cimage");
    println!("  Loaded in {:.1}s ({:.1} GB)", load_start.elapsed().as_secs_f64(),
        deployment.total_size() as f64 / 1_073_741_824.0);

    // ── Warmup ──────────────────────────────────────────────────
    let prompt = vec![1u32; 4];
    println!("Warmup: {} prompt tokens", prompt.len());
    let warmup = deployment.prefill(&prompt).expect("warmup prefill");
    let mut next = warmup;
    for _ in 0..2 {
        next = deployment.decode_one(next).expect("warmup decode");
    }

    // ── Benchmark ───────────────────────────────────────────────
    let prompt = vec![1u32; 4];
    println!("Benchmark: {} prompt → {} decode tokens", prompt.len(), decode_tokens);
    let prefill_start = Instant::now();
    let mut next = deployment.prefill(&prompt).expect("bench prefill");
    let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

    let decode_start = Instant::now();
    for _step in 0..decode_tokens {
        next = deployment.decode_one(next).expect("bench decode");
    }
    let decode_elapsed = decode_start.elapsed();
    let tok_s = (decode_tokens as f64) / decode_elapsed.as_secs_f64();

    println!();
    println!("═══════════════════════════════════════════════");
    println!("  Prefill:  {:.1} ms ({} tokens)", prefill_ms, prompt.len());
    println!("  Decode:   {} tokens in {:.2}s = {:.1} tok/s", decode_tokens, decode_elapsed.as_secs_f64(), tok_s);
    println!("  Total:    {:.2}s", (prefill_start.elapsed().as_secs_f64()));
    println!("═══════════════════════════════════════════════");
}
