//! ANE fused depth sweep — measures utilization gains from operation fusion.
//!
//! Tests the hypothesis that chaining multiple matmuls in a single MIL program
//! improves ANE utilization by keeping intermediate results in ~32 MB SRAM
//! instead of flushing to IOSurface between ops.
//!
//! For each depth N in [1, 2, 4, 8, 16]:
//!   1. N weight matrices W_i[512, 512] each
//!   2. N sequential matmuls: x[1, 512] @ W_0 -> W_1 @ ... @ W_{N-1} -> [1, 512]
//!   3. Total FLOPs = N × 2 × 512 × 512 = N × 524K
//!   4. If utilization climbs from ~60% (N=1, I/O dominated) toward ~94% (N=16, compute bound),
//!      the ANE successfully keeps intermediate results in SRAM across fused ops.
//!
//! Run: cargo test --test ane_fused_depth --features prism-backend -- --nocapture
//! Requires: macOS 14.0+ on Apple Silicon

#![cfg(all(target_os = "macos", feature = "prism-backend"))]

use std::path::{Path, PathBuf};
use std::time::Instant;

use coreml_proto::proto::mil_spec;
use tribunus_compute_core::arena::Arena;
use tribunus_compute_core::arena::DataType;
use tribunus_compute_core::coreml_bridge::{CoreMlComputeUnits, CoreMlModel};
use tribunus_compute_core::coreml_pipeline::compile_mlpackage;
use tribunus_compute_core::mil_builder::MilBuilder;
use tribunus_compute_core::mlpackage::{write_mlpackage, ModelMeta};

// ── Constants ──────────────────────────────────────────────────────────────

const TEST_DIR: &str = "/tmp/prism_ane_fused_depth";
/// Hidden dimension: small enough for any depth to fit in ANE SRAM.
const H: i64 = 512;
/// True FP16 theoretical peak for M1 ANE: 5.5 TMAC/s = 11 TFLOPS.
const THEORETICAL_PEAK_MACS: f64 = 5_500_000_000_000.0;
const THEORETICAL_PEAK_TMACS: f64 = 5.5; // TMAC/s
const WARMUP: usize = 5;
const SAMPLES: usize = 15;
/// Depths to sweep: each level N chains N sequential matmuls.
const DEPTHS: &[usize] = &[1, 2, 4, 8, 16];

// ── Helpers ────────────────────────────────────────────────────────────────

fn model_dir(name: &str) -> PathBuf {
    let p = Path::new(TEST_DIR).join(name);
    let _ = std::fs::create_dir_all(&p);
    p
}

fn seeded_weights(seed: u64, rows: i64, cols: i64) -> Vec<f32> {
    use std::hash::{Hash, Hasher};
    let mut w = Vec::with_capacity((rows * cols) as usize);
    for i in 0..((rows * cols) as u64) {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (seed + i).hash(&mut h);
        w.push((h.finish() as f32 % 1000.0 - 500.0) / 500.0);
    }
    w
}

/// Build fused chained-matmul MIL: x @ W_0 @ W_1 @ ... @ W_{N-1} -> [1, H].
///
///   x[1, H] -> matmul(x, W_0[H, H]) -> matmul(., W_1[H, H]) -> ... -> [1, H]
///
/// All matmuls are chained sequentially so the ANE keeps intermediate results
/// in SRAM rather than flushing to IOSurface between ops.
fn build_fused_depth_mil(depth: usize) -> Result<(mil_spec::Program, String), String> {
    let mut b = MilBuilder::new("main").input("x", mil_spec::DataType::Float16, &[1, H]);

    // Chain N matmuls: x @ W_0, then result @ W_1, etc.
    let mut prev = "x".to_string();
    for i in 0..depth {
        let w = seeded_weights(i as u64, H, H);
        b = b.const_f16(&format!("w_{}", i), &w, &[H, H]);
        let wn = b
            .last_name()
            .ok_or_else(|| format!("weight_{}", i))?
            .to_string();
        b = b.matmul(&prev, &wn);
        prev = b
            .last_name()
            .ok_or_else(|| format!("matmul_{}", i))?
            .to_string();
    }

    let out_name = prev;
    let b = b.output(&out_name);
    let prog = b.build().map_err(|e| format!("MIL build error: {}", e))?;
    Ok((prog, out_name))
}

fn compile(tag: &str, prog: mil_spec::Program, meta: ModelMeta) -> Result<PathBuf, String> {
    let dir = model_dir(tag);
    let pkg = write_mlpackage(prog, &dir, &meta).map_err(|e| format!("mlpackage: {}", e))?;
    let od = dir.join("compiled");
    let _ = std::fs::create_dir_all(&od);
    let r = compile_mlpackage(&pkg, &od, tag, "cpuAndNeuralEngine", "macOS26")
        .map_err(|e| format!("compile: {}", e))?;
    Ok(PathBuf::from(&r.compiled_modelc_path))
}

fn fill_arena(arena: &Arena, count: usize) {
    arena.lock().unwrap();
    unsafe {
        let ptr = arena.base_ptr() as *mut u16;
        for i in 0..count {
            *ptr.add(i) = ((i as u16).wrapping_mul(265).wrapping_add(1234)) & 0x7FFF;
        }
    }
    arena.unlock().unwrap();
}

fn bench_one(
    path: &str,
    cu: CoreMlComputeUnits,
    in_name: &str,
    in_arena: &Arena,
    out_name: &str,
    out_arena: &Arena,
) -> Result<f64, String> {
    let m = CoreMlModel::load_with_compute_units(path, cu)
        .map_err(|e| format!("load({:?}): {}", cu, e))?;

    for _ in 0..WARMUP {
        m.predict(in_name, &in_arena.info, out_name, &out_arena.info)
            .map_err(|e| format!("warmup: {}", e))?;
    }

    let mut samples = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let t0 = Instant::now();
        m.predict(in_name, &in_arena.info, out_name, &out_arena.info)
            .map_err(|e| format!("run: {}", e))?;
        samples.push(t0.elapsed().as_nanos() as f64);
    }
    samples.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(samples[samples.len() / 2])
}

// ═════════════════════════════════════════════════════════════════════════════
// T E S T
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn ane_fused_depth_sweep() {
    println!("\n=== ANE FUSED DEPTH SWEEP ===");
    println!(
        "Model: N sequential matmuls x[1,{}] @ W_i[{}, {}] -> [1, {}]",
        H, H, H, H
    );
    println!("Total FLOPs scales linearly with N (more compute per I/O round-trip)");
    println!(
        "Key metric: MAC utilization climbs from I/O dominated (N=1) toward compute bound (N=16)"
    );
    println!(
        "Theoretical peak: {:.1} TMAC/s (M1 ANE FP16)",
        THEORETICAL_PEAK_TMACS
    );
    println!("Depths: {:?}", DEPTHS);
    println!("warmup={}, samples={}", WARMUP, SAMPLES);
    println!("{}", "=".repeat(110));

    println!(
        "{:>6} {:>10} {:>12} {:>15} {:>10}",
        "Depth", "FLOPs", "Time(us)", "MACs/s", "%Peak"
    );
    println!("{}", "-".repeat(110));

    for &depth in DEPTHS {
        let tag = format!("fused_depth_{}", depth);

        // ── Build MIL ─────────────────────────────────────────────
        let (prog, out_name) = match build_fused_depth_mil(depth) {
            Ok(v) => v,
            Err(e) => {
                println!(
                    "{:>6} {:>10} {:>12} {:>15} {:>10}",
                    depth, "N/A", "BUILD_FAIL", "N/A", "N/A"
                );
                eprintln!("  {} BUILD: {}", tag, e);
                continue;
            }
        };

        let meta = ModelMeta {
            model_name: tag.clone(),
            function_name: "main".into(),
            short_description: format!("ane_fused_depth_{}", depth),
            version: "1.0".into(),
            author: "prism".into(),
            output_name: out_name.clone(),
            inputs: vec![("x".into(), vec![1, H])],
            outputs: vec![(out_name.clone(), vec![1, H])],

        };

        // ── Compile ───────────────────────────────────────────────
        let model_path = match compile(&tag, prog, meta) {
            Ok(p) => p,
            Err(e) => {
                println!(
                    "{:>6} {:>10} {:>12} {:>15} {:>10}",
                    depth, "N/A", "COMPILE_FAIL", "N/A", "N/A"
                );
                eprintln!("  {} COMPILE: {}", tag, e);
                continue;
            }
        };
        let path_str = model_path.to_str().expect("valid path");

        // ── Allocate arenas ───────────────────────────────────────
        let in_arena = match Arena::new(1, H as u32, DataType::Float16) {
            Ok(a) => a,
            Err(e) => {
                println!(
                    "{:>6} {:>10} {:>12} {:>15} {:>10}",
                    depth, "N/A", "ALLOC_FAIL", "N/A", "N/A"
                );
                eprintln!("  {} arena: {}", tag, e);
                continue;
            }
        };
        let out_arena = match Arena::new(1, H as u32, DataType::Float16) {
            Ok(a) => a,
            Err(e) => {
                println!(
                    "{:>6} {:>10} {:>12} {:>15} {:>10}",
                    depth, "N/A", "ALLOC_FAIL", "N/A", "N/A"
                );
                eprintln!("  {} output: {}", tag, e);
                continue;
            }
        };
        fill_arena(&in_arena, 1 * H as usize);

        // ── ANE benchmark ─────────────────────────────────────────
        let time_ns = match bench_one(
            path_str,
            CoreMlComputeUnits::CpuAndNeuralEngine,
            "x",
            &in_arena,
            &out_name,
            &out_arena,
        ) {
            Ok(t) => t,
            Err(e) => {
                println!(
                    "{:>6} {:>10} {:>12} {:>15} {:>10}",
                    depth, "N/A", "ANE_FAIL", "N/A", "N/A"
                );
                eprintln!("  {} ANE: {}", tag, e);
                continue;
            }
        };

        // ── Compute metrics ───────────────────────────────────────
        // Each matmul: FLOPs = 2 × M × N × K = 2 × 1 × H × H = 2 × H²
        // Total FLOPs = depth × 2 × H × H
        let flops_per_matmul = 2.0 * 1.0 * H as f64 * H as f64;
        let total_flops = depth as f64 * flops_per_matmul;
        // MACs = FLOPs / 2 (one MAC = multiply + accumulate = 2 FLOPs)
        let total_macs = total_flops / 2.0;

        let time_us = time_ns / 1000.0;
        let time_s = time_ns / 1_000_000_000.0;

        let mac_throughput = if time_s > 0.0 {
            total_macs / time_s
        } else {
            0.0
        };

        let pct_peak = if THEORETICAL_PEAK_MACS > 0.0 {
            mac_throughput / THEORETICAL_PEAK_MACS * 100.0
        } else {
            0.0
        };

        println!(
            "{:>6} {:>10.0e} {:>12.1} {:>15.3e} {:>9.1}%",
            depth, total_flops, time_us, mac_throughput, pct_peak
        );
    }

    println!("{}", "=".repeat(110));
    println!("If utilization climbs with depth, ANE keeps intermediates in SRAM across fused ops.");
    println!("Flat utilization across depth: single matmul already saturates ANE bandwidth.");
}
