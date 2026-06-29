//! 16-Core Forced Chunking — test #2 from the SRAM geometry plan.
//!
//! Tests whether splitting a large conv2d into N smaller independent conv2d ops
//! along the output channel dimension reduces ANE latency by enabling better
//! core utilization (more independent work items to schedule across 16 cores).
//!
//! Compares monolithic conv2d vs explicitly chunked conv2d for
//! N = 1, 2, 4, 8, 16 chunks.  FLOPs are identical across all N.
//!
//! Run: cargo test --test ane_forced_chunking --features prism-backend -- --nocapture

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

const TEST_DIR: &str = "/tmp/prism_ane_forced_chunking";
const H: i64 = 2048; // hidden dimension
const FFN: i64 = 4096; // FFN dimension
const BATCH: i64 = 256; // sequence length (last axis)
const THEORETICAL_PEAK_GFLOPS: f64 = 11_000.0;
const WARMUP: usize = 5;
const SAMPLES: usize = 15;

/// Number of chunks to test.
const CHUNK_COUNTS: &[i64] = &[1, 2, 4, 8, 16];

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

/// Build a monolithic conv2d model: single input, single conv, single output.
///
///   x[1, H, 1, B] → conv(W[FFN, H, 1, 1]) → y[1, FFN, 1, B]
fn build_monolith() -> Result<(mil_spec::Program, String, String), String> {
    let w = seeded_weights(0, H, FFN);
    let mut b = MilBuilder::new("main")
        .input("x", mil_spec::DataType::Float16, &[1, H, 1, BATCH])
        .const_f16("w", &w, &[FFN, H, 1, 1]);
    let wn = b.last_name().ok_or("weight name")?.to_string();

    b = b.conv("conv", "x", &wn, &[1, 1], "valid");
    let out_name = b.last_name().ok_or("conv name")?.to_string();
    let b = b.output(&out_name);
    let prog = b.build().map_err(|e| format!("MIL: {}", e))?;
    Ok((prog, "x".into(), out_name))
}

/// Build a chunked model: single input, N conv ops (each with weight
/// [FFN/N, H, 1, 1]), then concat along axis=1.
///
/// Each conv sees the full input x[1, H, 1, B] and the i-th weight slice
/// W_i[FFN/N, H, 1, 1], producing y_i[1, FFN/N, 1, B].
///
///   concat([y_0, …, y_{N-1}], axis=1) → y[1, FFN, 1, B]
///
/// Total FLOPs are identical to the monolith for any N:
///   N × ((2 × H × (FFN/N) × B)) = 2 × H × FFN × B
fn build_chunked(chunk_n: i64) -> Result<(mil_spec::Program, String, String), String> {
    let ffn_chunk = FFN / chunk_n;

    let mut b = MilBuilder::new("main").input("x", mil_spec::DataType::Float16, &[1, H, 1, BATCH]);
    let mut conv_outputs = Vec::new();

    for i in 0..chunk_n {
        let w = seeded_weights((i as u64 + 1) * 100, H, ffn_chunk);
        let w_name = format!("w_{}", i);
        b = b.const_f16(&w_name, &w, &[ffn_chunk, H, 1, 1]);
        let wn = b
            .last_name()
            .ok_or(format!("weight name for chunk {}", i))?
            .to_string();

        b = b.conv(&format!("conv_{}", i), "x", &wn, &[1, 1], "valid");
        let conv_out = b
            .last_name()
            .ok_or(format!("conv name for chunk {}", i))?
            .to_string();
        conv_outputs.push(conv_out);
    }

    // Concat all conv outputs along axis=1
    let concat_inputs: Vec<&str> = conv_outputs.iter().map(|s| s.as_str()).collect();
    b = b.concat("concat", &concat_inputs, 1, false);
    let out_name = b.last_name().ok_or("concat name")?.to_string();
    let b = b.output(&out_name);
    let prog = b.build().map_err(|e| format!("MIL: {}", e))?;
    Ok((prog, "x".into(), out_name))
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
    let m = CoreMlModel::load_with_compute_units(path, cu).map_err(|e| format!("load: {}", e))?;

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
    Ok(samples[samples.len() / 2] / 1000.0) // ns → us
}

#[test]
fn ane_forced_chunking() {
    println!("\n=== 16-CORE FORCED CHUNKING ===");
    println!("Tests whether splitting a conv2d's output channels reduces ANE latency");
    println!(
        "Conv: x[1, {}, 1, {}] → W[{}, {}, 1, 1] → y[1, {}, 1, {}]",
        H, BATCH, FFN, H, FFN, BATCH
    );
    println!(
        "Chunked: N convs with W_i[{}/N, {}, 1, 1] on same input, concat axis=1",
        FFN, H
    );
    println!("Total FLOPs identical for all N\n");

    // ── Build all models first ───────────────────────────────────
    println!("[build] Monolithic...");
    let (mono_prog, mono_in_name, mono_out_name) = match build_monolith() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("ERROR: monolithic build failed: {}", e);
            return;
        }
    };
    let mono_meta = ModelMeta {
        model_name: "monolith".into(),
        function_name: "main".into(),
        short_description: "monolith_conv".into(),
        version: "1.0".into(),
        author: "prism".into(),
        output_name: mono_out_name.clone(),
        inputs: vec![("x".into(), vec![1, H, 1, BATCH])],
        outputs: vec![(mono_out_name.clone(), vec![1, FFN, 1, BATCH])],
    };
    println!("[compile] Monolithic...");
    let mono_path = match compile("monolith", mono_prog, mono_meta) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("ERROR: monolithic compile failed: {}", e);
            return;
        }
    };

    let mut chunked_models: Vec<(i64, PathBuf, String, String)> = Vec::new();
    for &n in CHUNK_COUNTS {
        if n == 1 {
            continue; // N=1 is the monolith
        }
        let tag = format!("chunked_n{}", n);
        println!("[build/compile] Chunked N={}...", n);
        let (prog, in_name, out_name) = match build_chunked(n) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("  ERROR: chunk N={} build failed: {}", n, e);
                continue;
            }
        };
        let chunk_meta = ModelMeta {
            model_name: tag.clone(),
            function_name: "main".into(),
            short_description: format!("chunked_N{}", n),
            version: "1.0".into(),
            author: "prism".into(),
            output_name: out_name.clone(),
            inputs: vec![("x".into(), vec![1, H, 1, BATCH])],
            outputs: vec![(out_name.clone(), vec![1, FFN, 1, BATCH])],

        };
        match compile(&tag, prog, chunk_meta) {
            Ok(p) => chunked_models.push((n, p, in_name, out_name)),
            Err(e) => eprintln!("  ERROR: chunk N={} compile failed: {}", n, e),
        }
    }

    // ── Benchmark ─────────────────────────────────────────────────
    let in_arena = Arena::new(1, (H as u32) * (BATCH as u32), DataType::Float16).expect("input arena");
    let out_arena =
        Arena::new(1, (FFN as u32) * (BATCH as u32), DataType::Float16).expect("output arena");
    fill_arena(&in_arena, 1 * H as usize * 1 * BATCH as usize);

    println!();
    println!(
        "{:>6} {:>25} {:>20} {:>14} {:>14}",
        "N", "time(us)", "vs_mono_speedup", "GFLOPS", "%peak"
    );
    println!("{}", "=".repeat(85));

    let total_flops = 2.0 * (H as f64) * (FFN as f64) * (BATCH as f64);

    // Monolithic baseline
    let mono_time_us = match bench_one(
        mono_path.to_str().unwrap(),
        CoreMlComputeUnits::CpuAndNeuralEngine,
        &mono_in_name,
        &in_arena,
        &mono_out_name,
        &out_arena,
    ) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("WARN: monolithic ANE failed: {}", e);
            f64::NAN
        }
    };

    let mono_gflops = if mono_time_us.is_nan() || mono_time_us <= 0.0 {
        f64::NAN
    } else {
        total_flops / (mono_time_us * 1e3) / 1e9
    };
    let mono_pct = if mono_gflops.is_nan() {
        f64::NAN
    } else {
        (mono_gflops / THEORETICAL_PEAK_GFLOPS) * 100.0
    };

    println!(
        "{:>6} {:>25.1} {:>20} {:>14.1} {:>13.1}%",
        1, mono_time_us, "1.000 (ref)", mono_gflops, mono_pct,
    );

    for &(n, ref path, ref in_name, ref out_name) in &chunked_models {
        let time_us = match bench_one(
            path.to_str().unwrap(),
            CoreMlComputeUnits::CpuAndNeuralEngine,
            in_name,
            &in_arena,
            out_name,
            &out_arena,
        ) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("  chunk N={} ANE failed: {}", n, e);
                println!(
                    "{:>6} {:>25} {:>20} {:>14} {:>14}",
                    n, "ANE_FAIL", "N/A", "N/A", "N/A",
                );
                continue;
            }
        };

        let speedup = if time_us > 0.0 {
            mono_time_us / time_us
        } else {
            f64::NAN
        };
        let gflops = if time_us > 0.0 {
            total_flops / (time_us * 1e3) / 1e9
        } else {
            f64::NAN
        };
        let pct = if gflops.is_nan() {
            f64::NAN
        } else {
            (gflops / THEORETICAL_PEAK_GFLOPS) * 100.0
        };

        println!(
            "{:>6} {:>25.1} {:>20.3} {:>14.1} {:>13.1}%",
            n, time_us, speedup, gflops, pct,
        );
    }

    println!("\n--- done ---");
    println!(
        "Total FLOPs: {:.0} = {:.1} GFLOPs",
        total_flops,
        total_flops / 1e9,
    );
    println!("Theoretical peak: {:.0} GFLOPS", THEORETICAL_PEAK_GFLOPS);
}
