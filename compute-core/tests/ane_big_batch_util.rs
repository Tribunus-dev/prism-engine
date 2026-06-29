//! ANE matmul utilization — large batch extension.
//!
//! Extends the batch sweep to the ANE's maximum batch capacity to see if
//! utilization approaches 100% at large batch sizes.
//!
//! Model: x[batch, 2048] @ W[2048, 4096] -> [batch, 4096]
//!
//! Run: cargo test --test ane_big_batch_util --features prism-backend -- --nocapture

#![cfg(all(target_os = "macos", feature = "prism-backend"))]

use std::path::{Path, PathBuf};
use std::time::Instant;

use coreml_proto::proto::mil_spec;
use mlx_rs::Dtype;
use tribunus_compute_core::arena::Arena;
use tribunus_compute_core::arena::DataType;
use tribunus_compute_core::coreml_bridge::{CoreMlComputeUnits, CoreMlModel};
use tribunus_compute_core::coreml_pipeline::compile_mlpackage;
use tribunus_compute_core::mil_builder::MilBuilder;
use tribunus_compute_core::mlpackage::{write_mlpackage, ModelMeta};

const TEST_DIR: &str = "/tmp/prism_ane_big_batch";
const H: i64 = 2048;
const FFN: i64 = 4096;
const THEORETICAL_PEAK_GFLOPS: f64 = 11_000.0;
const WARMUP: usize = 3;
const SAMPLES: usize = 10;
const CPU_FALLBACK_RATIO: f64 = 0.8;

// Focused batch sizes: known working range up to ANE limit
const BATCH_SIZES: &[u32] = &[8192, 16384, 32768, 65536, 131072];

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

fn build_mil(batch: u32) -> Result<(mil_spec::Program, String), String> {
    let w = seeded_weights(42, H, FFN);
    let b = MilBuilder::new("main")
        .input("x", mil_spec::DataType::Float16, &[batch as i64, H])
        .const_f16("w", &w, &[H, FFN]);
    let wn = b.last_name().ok_or("weight name")?.to_string();
    let b = b.matmul("x", &wn);
    let out_name = b.last_name().ok_or("matmul name")?.to_string();
    let b = b.output(&out_name);
    let prog = b.build().map_err(|e| format!("MIL build error: {}", e))?;
    Ok((prog, out_name))
}

fn compile_model(tag: &str, prog: mil_spec::Program, meta: ModelMeta) -> Result<PathBuf, String> {
    let dir = model_dir(tag);
    let pkg = write_mlpackage(prog, &dir, &meta).map_err(|e| format!("mlpackage: {}", e))?;
    let od = dir.join("compiled");
    let _ = std::fs::create_dir_all(&od);
    let r = compile_mlpackage(&pkg, &od, tag, "cpuAndNeuralEngine", "macOS26")
        .map_err(|e| format!("compile: {}", e))?;
    Ok(PathBuf::from(&r.compiled_modelc_path))
}

fn fill_arena(arena: &Arena, batch: u32, cols: u32) -> Result<(), String> {
    arena.lock().map_err(|e| format!("arena lock: {}", e))?;
    unsafe {
        let ptr = arena.base_ptr() as *mut u16;
        let count = (batch as usize) * (cols as usize);
        for i in 0..count {
            let val = ((i as u16).wrapping_mul(265).wrapping_add(1234)) & 0x7FFF;
            *ptr.add(i) = val;
        }
    }
    arena.unlock().map_err(|e| format!("arena unlock: {}", e))?;
    Ok(())
}

fn bench_one(
    path: &str,
    cu: CoreMlComputeUnits,
    in_name: &str,
    in_arena: &Arena,
    out_name: &str,
    out_arena: &Arena,
) -> Result<(f64, f64, f64), String> {
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
    let p50 = samples[samples.len() / 2];
    let p95 = samples[(samples.len() as f64 * 0.95) as usize];
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    Ok((p50, p95, mean))
}

#[test]
fn ane_big_batch_util_sweep() {
    println!("\n=== ANE BIG BATCH UTILIZATION SWEEP ===");
    println!(
        "Model: x[batch,{}] @ W[{},{}] -> [batch,{}]",
        H, H, FFN, FFN
    );
    println!(
        "Theoretical peak: {} GFLOPS (M1 ANE FP16)",
        THEORETICAL_PEAK_GFLOPS as u64
    );
    println!("Batch sizes: {:?}", BATCH_SIZES);
    println!("{}", "=".repeat(95));
    println!(
        "{:>8} {:>12} {:>10} {:>10} {:>10} {:>8} {:>12}",
        "Batch", "FLOPs", "Time(us)", "GFLOPS", "%Peak", "Status", "tok/s"
    );
    println!("{}", "-".repeat(95));

    let mut max_utilization: f64 = 0.0;
    let mut max_util_batch: u32 = 0;

    for &batch in BATCH_SIZES {
        let tag = format!("big_batch_{}", batch);
        eprintln!("\n--- Building batch={} ---", batch);

        let (prog, out_name) = match build_mil(batch) {
            Ok(v) => v,
            Err(e) => {
                println!(
                    "{:>8} {:>12} {:>10} {:>10} {:>10} {:>8} {:>12}",
                    batch, "N/A", "BUILD_FAIL", "N/A", "N/A", "ERR", "N/A"
                );
                eprintln!("  BUILD: {}", e);
                continue;
            }
        };

        let meta = ModelMeta {
            model_name: tag.clone(),
            function_name: "main".into(),
            short_description: format!("ane_big_batch_{}", batch),
            version: "1.0".into(),
            author: "prism".into(),
            output_name: out_name.clone(),
            inputs: vec![("x".into(), vec![batch as i64, H])],
            outputs: vec![(out_name.clone(), vec![batch as i64, FFN])],

        };

        eprintln!("  Compiling batch={}...", batch);
        let compile_start = Instant::now();
        let model_path = match compile_model(&tag, prog, meta) {
            Ok(p) => p,
            Err(e) => {
                println!(
                    "{:>8} {:>12} {:>10} {:>10} {:>10} {:>8} {:>12}",
                    batch, "N/A", "COMPILE_FAIL", "N/A", "N/A", "ERR", "N/A"
                );
                eprintln!("  COMPILE: {}", e);
                continue;
            }
        };
        let compile_ms = compile_start.elapsed().as_millis();
        let path_str = model_path.to_str().expect("valid path");
        eprintln!("  Compiled in {}ms", compile_ms);

        eprintln!(
            "  Allocating arenas batch={}, H={}, FFN={}...",
            batch, H, FFN
        );
        let in_arena = match Arena::new(batch, H as u32, DataType::Float16) {
            Ok(a) => a,
            Err(e) => {
                println!(
                    "{:>8} {:>12} {:>10} {:>10} {:>10} {:>8} {:>12}",
                    batch, "N/A", "ALLOC_FAIL", "N/A", "N/A", "ERR", "N/A"
                );
                eprintln!("  arena: {}", e);
                continue;
            }
        };
        let out_arena = match Arena::new(batch, FFN as u32, DataType::Float16) {
            Ok(a) => a,
            Err(e) => {
                println!(
                    "{:>8} {:>12} {:>10} {:>10} {:>10} {:>8} {:>12}",
                    batch, "N/A", "ALLOC_FAIL", "N/A", "N/A", "ERR", "N/A"
                );
                eprintln!("  output: {}", e);
                continue;
            }
        };
        eprintln!("  Arenas allocated");
        if let Err(e) = fill_arena(&in_arena, batch, H as u32) {
            eprintln!("  fill: {}", e);
        }

        eprintln!("  Loading ANE model...");
        let ane_loaded = Instant::now();
        let ane_result = match bench_one(
            path_str,
            CoreMlComputeUnits::CpuAndNeuralEngine,
            "x",
            &in_arena,
            &out_name,
            &out_arena,
        ) {
            Ok(r) => r,
            Err(e) => {
                println!(
                    "{:>8} {:>12} {:>10} {:>10} {:>10} {:>8} {:>12}",
                    batch, "N/A", "ANE_FAIL", "N/A", "N/A", "ERR", "N/A"
                );
                eprintln!("  ANE: {}", e);
                continue;
            }
        };
        let (ane_p50_ns, _ane_p95_ns, ane_mean_ns) = ane_result;
        let ane_ms = ane_loaded.elapsed().as_millis();

        eprintln!("  Loading CPU model...");
        let cpu_result = match bench_one(
            path_str,
            CoreMlComputeUnits::CpuOnly,
            "x",
            &in_arena,
            &out_name,
            &out_arena,
        ) {
            Ok(r) => r,
            Err(_) => (0.0, 0.0, 0.0),
        };
        let (_cpu_p50, _cpu_p95, cpu_mean_ns) = cpu_result;

        let total_flops = 2.0 * batch as f64 * H as f64 * FFN as f64;
        let time_us = ane_p50_ns / 1000.0;
        let time_s = ane_p50_ns / 1_000_000_000.0;
        let gflops = if time_s > 0.0 {
            total_flops / time_s / 1_000_000_000.0
        } else {
            0.0
        };
        let pct_peak = if THEORETICAL_PEAK_GFLOPS > 0.0 {
            gflops / THEORETICAL_PEAK_GFLOPS * 100.0
        } else {
            0.0
        };

        let ratio = if cpu_mean_ns > 0.0 {
            ane_mean_ns / cpu_mean_ns
        } else {
            0.0
        };
        let status = if ratio > CPU_FALLBACK_RATIO {
            "CPU_FB"
        } else {
            "on-ANE"
        };

        let tok_s = if time_us > 0.0 {
            1_000_000.0 / (time_us * 48.0 / batch as f64)
        } else {
            0.0
        };

        println!(
            "{:>8} {:>12.0e} {:>10.1} {:>10.2} {:>9.3}% {:>8} {:>12.1}  (load+bench: {}ms, compile: {}ms)",
            batch, total_flops, time_us, gflops, pct_peak, status, tok_s, ane_ms, compile_ms
        );

        // Track peak utilization
        if pct_peak > max_utilization {
            max_utilization = pct_peak;
            max_util_batch = batch;
        }

        // Check if we're approaching 100%
        let gap = 100.0 - pct_peak;
        println!("  → {}% headroom to theoretical peak", gap as u64);
    }

    println!("{}", "=".repeat(95));
    println!(
        "Peak utilization: {:.2}% of {} GFLOPS at batch={}",
        max_utilization, THEORETICAL_PEAK_GFLOPS as u64, max_util_batch
    );

    if max_utilization > 90.0 {
        println!("Assessment: ANE CAN be saturated to near-100% with large enough batches");
        println!("The ~10% gap is likely Core ML runtime overhead, not ANE compute limit");
    } else if max_utilization > 75.0 {
        println!("Assessment: ANE mostly saturated. Remaining gap is Core ML dispatch + IOSurface overhead");
    } else if max_utilization > 60.0 {
        println!(
            "Assessment: Significant headroom remains. Bottleneck may be ANE memory bandwidth,"
        );
        println!("  not compute. IOSurface read bandwidth caps at ~60% of ANE compute throughput.");
    } else {
        println!(
            "Assessment: ANE is not compute-bound for large matmuls. The bottleneck is likely"
        );
        println!("  IOSurface bandwidth or Core ML runtime dispatch overhead.");
    }
}
