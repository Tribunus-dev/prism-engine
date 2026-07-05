//! prism-bench-ab — truthful NF4-teacher vs ternary-student benchmark runner.
//!
//! Loads one or two `.cimage` models, runs a FIXED token stream through each
//! with greedy decode, captures prefill/decode timings + per-token logits, and
//! emits the `bench_metrics` (perplexity, throughput) + `distill_core`
//! (top-1 agreement, KL) comparison table. Gated to `prism-backend` (Metal).
//!
//! Fairness: identical token stream, greedy decode, warmup discarded, per-step
//! decode samples → median/p90/p99 (see kernels/BENCHMARK.md).
//!
//! Usage (macOS):
//!   cargo run --release --features prism-backend --bin prism-bench-ab -- \
//!     --teacher ~/models/gemma4-nf4.cimage \
//!     --student ~/models/gemma4-ternary.cimage \
//!     --max-tokens 128 --warmup 8 --eval-len 256
//!
//! Pass real tokenized text via --eval-tokens/--prompt-tokens (comma/space
//! separated u32 IDs) for linguistically-meaningful perplexity; the built-in
//! deterministic stream gives a fair *relative* teacher-vs-student comparison.

use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(about = "NF4-teacher vs ternary-student benchmark (performance + accuracy)")]
struct Args {
    /// Teacher (or single) .cimage.
    #[arg(long)]
    teacher: PathBuf,
    /// Optional student .cimage for a head-to-head comparison.
    #[arg(long)]
    student: Option<PathBuf>,
    /// Generated tokens per decode measurement.
    #[arg(long, default_value = "64")]
    max_tokens: u32,
    /// Warmup decode steps discarded before measuring.
    #[arg(long, default_value = "8")]
    warmup: u32,
    /// Length of the teacher-forcing perplexity stream.
    #[arg(long, default_value = "128")]
    eval_len: usize,
    /// Prompt length for the prefill measurement.
    #[arg(long, default_value = "16")]
    prompt_len: usize,
    /// Optional file of comma/space-separated u32 token IDs for perplexity.
    #[arg(long)]
    eval_tokens: Option<PathBuf>,
    /// Optional file of token IDs for the prefill prompt.
    #[arg(long)]
    prompt_tokens: Option<PathBuf>,
    /// Total parameter count, to report effective bits-per-weight from file size.
    #[arg(long)]
    params: Option<u64>,
    /// Safe upper bound for built-in token IDs (keep below every model's vocab).
    #[arg(long, default_value = "1000")]
    vocab_cap: u32,
    /// KD softmax temperature.
    #[arg(long, default_value = "2.0")]
    temperature: f32,
}

#[cfg(not(feature = "prism-backend"))]
fn main() {
    eprintln!("prism-bench-ab requires the `prism-backend` feature (Metal).");
    eprintln!("  cargo run --release --features prism-backend --bin prism-bench-ab -- ...");
    std::process::exit(1);
}

#[cfg(feature = "prism-backend")]
fn load_tokens(path: &Option<PathBuf>, n: usize, vocab_cap: u32, seed: u64) -> Vec<u32> {
    if let Some(p) = path {
        let text = std::fs::read_to_string(p)
            .unwrap_or_else(|e| panic!("read tokens {}: {e}", p.display()));
        let toks: Vec<u32> = text
            .split(|c: char| c == ',' || c.is_whitespace())
            .filter(|s| !s.is_empty())
            .map(|s| s.parse::<u32>().expect("token id"))
            .collect();
        assert!(!toks.is_empty(), "no tokens parsed from {}", p.display());
        return toks;
    }
    // Deterministic built-in stream (fair relative comparison; not natural text).
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            1 + ((s >> 40) as u32) % vocab_cap.max(2)
        })
        .collect()
}

#[cfg(feature = "prism-backend")]
fn main() {
    use std::time::Instant;
    use tribunus_compute_core::compilation::bench_metrics::{
        compare, effective_bpw, perplexity, throughput_stats, token_nll, ModelRunMetrics,
    };
    use tribunus_compute_core::compilation::distill_core::{kd_divergence_batch, top1_agreement};
    use tribunus_compute_core::compute_image::orchestrator::Orchestrator;

    let args = Args::parse();
    let eval = load_tokens(&args.eval_tokens, args.eval_len, args.vocab_cap, 1);
    let prompt = load_tokens(&args.prompt_tokens, args.prompt_len, args.vocab_cap, 2);

    // Run one model: returns (metrics, per-position logits from the PPL pass, vocab).
    let run = |name: &str, path: &PathBuf| -> (ModelRunMetrics, Vec<Vec<f32>>, usize) {
        // ── Perplexity + logits via teacher forcing (fresh KV) ──
        let mut orch = Orchestrator::from_cimage(path, 1)
            .unwrap_or_else(|e| panic!("load {name} ({}): {e}", path.display()));
        let mut nlls: Vec<f32> = Vec::new();
        let mut logits_seq: Vec<Vec<f32>> = Vec::new();
        let mut vocab = 0usize;
        for i in 0..eval.len().saturating_sub(1) {
            let (_tok, logits) = orch
                .decode_token_logits(eval[i])
                .unwrap_or_else(|e| panic!("{name} decode@{i}: {e}"));
            vocab = logits.len();
            let target = eval[i + 1] as usize;
            if target < vocab {
                nlls.push(token_nll(&logits, target));
            }
            logits_seq.push(logits);
        }
        let ppl = perplexity(&nlls);

        // ── Throughput on a fresh orchestrator (clean KV) ──
        let mut orch = Orchestrator::from_cimage(path, 1)
            .unwrap_or_else(|e| panic!("reload {name}: {e}"));
        let t_prefill = Instant::now();
        orch.prefill_text(&prompt)
            .unwrap_or_else(|e| panic!("{name} prefill: {e}"));
        let prefill_secs = t_prefill.elapsed().as_secs_f64().max(1e-9);
        let prefill_tps = throughput_stats(&[prompt.len() as f64 / prefill_secs]);

        let mut cur = *prompt.last().unwrap_or(&1);
        for _ in 0..args.warmup {
            cur = orch.decode_token(cur).unwrap_or_else(|e| panic!("{name} warmup: {e}"));
        }
        let mut ttft_ms = prefill_secs * 1000.0; // prompt→first token
        let mut per_step: Vec<f64> = Vec::with_capacity(args.max_tokens as usize);
        for step in 0..args.max_tokens {
            let s = Instant::now();
            cur = orch.decode_token(cur).unwrap_or_else(|e| panic!("{name} decode: {e}"));
            let dt = s.elapsed().as_secs_f64().max(1e-9);
            if step == 0 {
                ttft_ms += dt * 1000.0;
            }
            per_step.push(1.0 / dt); // tokens/sec for this step
        }
        let decode_tps = throughput_stats(&per_step);

        let bpw = match args.params {
            Some(p) => effective_bpw(std::fs::metadata(path).map(|m| m.len()).unwrap_or(0), p),
            None => f64::NAN,
        };

        (
            ModelRunMetrics {
                name: name.to_string(),
                prefill_tok_s: prefill_tps,
                decode_tok_s: decode_tps,
                ttft_ms: throughput_stats(&[ttft_ms]),
                perplexity: ppl,
                effective_bpw: bpw,
            },
            logits_seq,
            vocab,
        )
    };

    let print_metrics = |m: &ModelRunMetrics| {
        println!("── {} ──", m.name);
        println!("  prefill  : {:>8.1} tok/s (median)", m.prefill_tok_s.median);
        println!(
            "  decode   : {:>8.1} tok/s  (median; p90 {:.1}, p99 {:.1})",
            m.decode_tok_s.median, m.decode_tok_s.p90, m.decode_tok_s.p99
        );
        println!("  TTFT     : {:>8.1} ms", m.ttft_ms.median);
        println!("  perplexity: {:>7.3}", m.perplexity);
        if m.effective_bpw.is_finite() {
            println!("  bpw      : {:>8.3}", m.effective_bpw);
        }
    };

    println!("═══ Prism teacher/student benchmark ═══");
    println!(
        "eval_len={} prompt_len={} max_tokens={} warmup={}\n",
        eval.len(),
        prompt.len(),
        args.max_tokens,
        args.warmup
    );

    let (t_metrics, t_logits, vocab) = run("teacher", &args.teacher);
    print_metrics(&t_metrics);

    if let Some(student_path) = &args.student {
        let (s_metrics, s_logits, _) = run("student", student_path);
        print_metrics(&s_metrics);

        // Head-to-head. Align the per-position logits captured during PPL.
        let n = t_logits.len().min(s_logits.len());
        let flat_t: Vec<f32> = t_logits[..n].iter().flatten().copied().collect();
        let flat_s: Vec<f32> = s_logits[..n].iter().flatten().copied().collect();
        let top1 = top1_agreement(&flat_t, &flat_s, vocab);
        let kd = kd_divergence_batch(&flat_t, &flat_s, vocab, args.temperature);
        let cmp = compare(&t_metrics, &s_metrics);

        println!("\n── student vs teacher ──");
        println!("  decode speedup     : {:>6.2}×", cmp.decode_speedup);
        println!("  prefill speedup    : {:>6.2}×", cmp.prefill_speedup);
        println!(
            "  perplexity ratio   : {:>6.3}  ({})",
            cmp.perplexity_ratio,
            if cmp.perplexity_ratio > 1.0 { "student worse" } else { "student better" }
        );
        if cmp.bpw_ratio.is_finite() {
            println!("  bpw ratio          : {:>6.3}", cmp.bpw_ratio);
        }
        println!("  top-1 agreement    : {:>6.1}%", top1 * 100.0);
        println!("  logit KL (T={:.1}) : {:>6.4}", args.temperature, kd);
        println!(
            "\n→ student trades {:.1}% perplexity for {:.2}× decode at {}× the density.",
            (cmp.perplexity_ratio - 1.0) * 100.0,
            cmp.decode_speedup,
            if cmp.bpw_ratio.is_finite() {
                format!("{:.2}", 1.0 / cmp.bpw_ratio)
            } else {
                "?".to_string()
            }
        );
    }
}
