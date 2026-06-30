//! Real-weight qualification: baseline vs interleaved decode across context buckets.
//!
//! Loads the production Gemma 4 12B model, runs both the standard persistent
//! decoder (baseline) and the concurrent decode+prefetch (interleaved) through
//! multiple context lengths, and reports p50/p95 token latency for each.
//!
//! Run: cargo test --test real_weight_qualification --features prism-backend -- --nocapture
//! Requires: models/gemma4-12b-it/gemma4_12b_v4.cimage

#![cfg(all(target_os = "macos", feature = "prism-backend"))]

use metal::*;
use std::time::{Duration, Instant};

use tribunus_compute_core::compute_image::cimage_loader::CimageDeployment;
use tribunus_compute_core::compute_image::megakernel::Megakernel;
use tribunus_compute_core::compute_image::megakernel::MAX_CONTEXT;

/// Find the cimage file.
fn find_cimage() -> Option<std::path::PathBuf> {
    let candidates = [
        "models/gemma4-12b-it/gemma4_12b.cimage",
        "../../models/gemma4-12b-it/gemma4_12b.cimage",
        "/Users/user/Developer/GitHub/models/gemma4-12b-it/gemma4_12b.cimage",
    ];
    for c in &candidates {
        let p = std::path::Path::new(c);
        if p.exists() {
            return Some(p.to_path_buf());
        }
    }
    if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
        let p = std::path::Path::new(&manifest).join("../../models/gemma4-12b-it/gemma4_12b_v4.cimage");
        if p.exists() {
            return Some(p);
        }
    }
    None
}

const TOKENS_PER_RUN: usize = 8;
const WARMUP: u32 = 2;
const CONTEXT_BUCKETS: &[u32] = &[1, 64, 256, 1024, 2048];

fn run_tokens(deployment: &CimageDeployment, interleaved: bool, context_len: u32) -> Vec<Duration> {
    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue();

    let mut mk = Megakernel::new(&device, &queue, deployment, false)
        .expect("Megakernel::new");

    // Set interleave mode before launch
    mk.kv_prefetch_enabled = interleaved;
    mk.interleave_plan.enabled = interleaved;

    let buffers = mk.launch(deployment, 1).expect("launch");

    // Set context length via the ring entry's sequence_id.
    // The kernel reads num_cached = min(current_pos + 1, MAX_CTX).
    let seq_pos = context_len.saturating_sub(1);

    // Warmup
    for i in 0..WARMUP {
        mk.submit_work(&buffers, 0, i, seq_pos + i, 0);
        while !mk.poll_work(&buffers, 0) {
            std::hint::spin_loop();
        }
    }

    // Benchmark
    let mut times = Vec::with_capacity(TOKENS_PER_RUN);
    for i in WARMUP..WARMUP + TOKENS_PER_RUN as u32 {
        mk.submit_work(&buffers, 0, i, seq_pos + i, 0);
        let start = Instant::now();
        while !mk.poll_work(&buffers, 0) {
            std::hint::spin_loop();
        }
        times.push(start.elapsed());
    }

    times
}

fn print_stats(label: &str, times: &[Duration]) {
    let mut s = times.to_vec();
    s.sort();
    let n = s.len();
    let total: Duration = s.iter().sum();
    let mean = total / n as u32;
    let p50 = s[n / 2];
    let p95_idx = ((n * 95 + 99) / 100).saturating_sub(1);
    let p95 = s[p95_idx.min(n - 1)];
    let min = s[0];
    let max = s[n - 1];
    println!("  {label:>25}  {mean:>8.1?}  {p50:>8.1?}  {p95:>8.1?}  {min:>8.1?}  {max:>8.1?}",
        mean=mean, p50=p50, p95=p95, min=min, max=max);
}

#[test]
fn real_weight_qualification() {
    let cimage_path = find_cimage().expect("cimage file not found. Set CIMAGE_PATH or place at models/gemma4-12b-it/gemma4_12b_v4.cimage");
    let file_size = std::fs::metadata(&cimage_path).unwrap().len();
    println!("Loading cimage: {} ({})", cimage_path.display(), human_size(file_size));

    let device = Device::system_default().unwrap();
    let deployment = CimageDeployment::load(&cimage_path, &device)
        .expect("load cimage failed");
    println!("Model: {} layers, {}M weights",
        deployment.num_layers, deployment.num_weights / 1_000_000);
    println!();

    println!("── Real-Weight Qualification ──────────────────────────────────────");
    println!("  {:>25}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}", "mode", "mean", "p50", "p95", "min", "max");
    println!("  {}", "-".repeat(77));

    for &ctx in CONTEXT_BUCKETS {
        if ctx > MAX_CONTEXT { continue; }

        let t0 = Instant::now();
        let baseline = run_tokens(&deployment, false, ctx);
        let t1 = Instant::now();
        let interleaved = run_tokens(&deployment, true, ctx);
        let total_elapsed = t0.elapsed();

        print_stats(&format!("ctx={} baseline", ctx), &baseline);
        print_stats(&format!("ctx={} interleaved", ctx), &interleaved);

        let b_mean = baseline.iter().sum::<Duration>() / baseline.len() as u32;
        let i_mean = interleaved.iter().sum::<Duration>() / interleaved.len() as u32;
        let delta = if i_mean > b_mean {
            format!("+{:?}", i_mean - b_mean)
        } else {
            format!("-{:?}", b_mean - i_mean)
        };
        println!("  {:>25}  delta={}  (total {:.1}s)", "", delta, total_elapsed.as_secs_f64());
        println!();
    }
}

fn human_size(bytes: u64) -> String {
    let b = bytes as f64;
    if b > 1_000_000_000.0 { format!("{:.1} GB", b / 1_000_000_000.0) }
    else if b > 1_000_000.0 { format!("{:.1} MB", b / 1_000_000.0) }
    else { format!("{} B", bytes) }
}
