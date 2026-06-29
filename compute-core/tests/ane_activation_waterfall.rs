//! Activation Thrashing Waterfall — test #4 from the SRAM geometry plan.
//!
//! Sweeps hidden dimension C to find the internal buffer limit between the
//! MAC arrays and the Planar Engine (Swish/SiLU activation).
//!
//! Model topology:
//!   Input: x[1, C, 1, 64]
//!   Conv2d 1x1: W1[C, C, 1, 1] → y1[1, C, 1, 64]
//!   SiLU activation: silu(y1) → y2[1, C, 1, 64]
//!   Conv2d 1x1: W2[C, C, 1, 1] → y3[1, C, 1, 64]
//!
//! FLOPs = 2 * 1 * C * C * 64 * 2 = 256 * C^2
//!
//! At small C, the silu output fits in the internal buffer between the MAC
//! array and Planar Engine — minimal overhead. At a specific C, the
//! intermediate tensor spills to DRAM — overhead spikes.
//!
//! Each C is tested with two models:
//!   - conv→silu→conv (with activation)
//!   - conv→conv (no activation, isolate the activation overhead)
//!
//! Run: cargo test --test ane_activation_waterfall --features prism-backend -- --nocapture

#![cfg(all(target_os = "macos", feature = "prism-backend"))]

use std::path::{Path, PathBuf};
use std::time::Instant;

use coreml_proto::proto::mil_spec;
use mlx_rs::Dtype;
use tribunus_compute_core::arena::Arena;
use tribunus_compute_core::coreml_bridge::{CoreMlComputeUnits, CoreMlModel};
use tribunus_compute_core::coreml_pipeline::compile_mlpackage;
use tribunus_compute_core::mil_builder::MilBuilder;
use tribunus_compute_core::mlpackage::{write_mlpackage, ModelMeta};

const TEST_DIR: &str = "/tmp/prism_ane_activation_waterfall";
const WARMUP: usize = 5;
const SAMPLES: usize = 15;

/// Hidden dimensions to sweep. C = in_ch = out_ch for both conv layers.
const C_VALUES: &[u32] = &[128, 256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2304, 2560];

fn model_dir(name: &str) -> PathBuf {
    let p = Path::new(TEST_DIR).join(name);
    let _ = std::fs::create_dir_all(&p);
    p
}

fn seeded_weights(seed: u64, count: usize) -> Vec<f32> {
    use std::hash::{Hash, Hasher};
    let mut w = Vec::with_capacity(count);
    for i in 0..count as u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (seed + i).hash(&mut h);
        w.push((h.finish() as f32 % 1000.0 - 500.0) / 500.0);
    }
    w
}

/// Build conv→silu→conv model.
///
/// Input: x[1, C, 1, 64]
/// Conv2d(W1): W1[C, C, 1, 1] → y1[1, C, 1, 64]
/// SiLU: silu(y1) → y2[1, C, 1, 64]
/// Conv2d(W2): W2[C, C, 1, 1] → y3[1, C, 1, 64]
fn build_conv_silu_conv(c: u32) -> Result<(mil_spec::Program, String, String), String> {
    let w1 = seeded_weights(0, (c * c) as usize);
    let w2 = seeded_weights(1, (c * c) as usize);
    let mut b = MilBuilder::new("main")
        .set_opset("CoreML9")
        .input("x", mil_spec::DataType::Float16, &[1, c as i64, 1, 64])
        .const_f16("w1", &w1, &[c as i64, c as i64, 1, 1]);
    let w1n = b.last_name().ok_or("w1 name")?.to_string();

    b = b.conv("conv1", "x", &w1n, &[1, 1], "valid");
    let conv1_out = b.last_name().ok_or("conv1 out")?.to_string();

    b = b.silu("silu", &conv1_out);
    let silu_out = b.last_name().ok_or("silu out")?.to_string();

    b = b.const_f16("w2", &w2, &[c as i64, c as i64, 1, 1]);
    let w2n = b.last_name().ok_or("w2 name")?.to_string();

    b = b.conv("conv2", &silu_out, &w2n, &[1, 1], "valid");
    let out_name = b.last_name().ok_or("conv2 out")?.to_string();
    let b = b.output(&out_name);
    let prog = b.build().map_err(|e| format!("MIL: {}", e))?;
    Ok((prog, "x".into(), out_name))
}

/// Build conv→conv model (no activation).
///
/// Input: x[1, C, 1, 64]
/// Conv2d(W1): W1[C, C, 1, 1] → y1[1, C, 1, 64]
/// Conv2d(W2): W2[C, C, 1, 1] → y2[1, C, 1, 64]
///
/// This isolates the activation overhead by providing a baseline without SiLU.
fn build_conv_conv(c: u32) -> Result<(mil_spec::Program, String, String), String> {
    let w1 = seeded_weights(0, (c * c) as usize);
    let w2 = seeded_weights(1, (c * c) as usize);
    let mut b = MilBuilder::new("main")
        .set_opset("CoreML9")
        .input("x", mil_spec::DataType::Float16, &[1, c as i64, 1, 64])
        .const_f16("w1", &w1, &[c as i64, c as i64, 1, 1]);
    let w1n = b.last_name().ok_or("w1 name")?.to_string();

    b = b.conv("conv1", "x", &w1n, &[1, 1], "valid");
    let conv1_out = b.last_name().ok_or("conv1 out")?.to_string();

    b = b.const_f16("w2", &w2, &[c as i64, c as i64, 1, 1]);
    let w2n = b.last_name().ok_or("w2 name")?.to_string();

    b = b.conv("conv2", &conv1_out, &w2n, &[1, 1], "valid");
    let out_name = b.last_name().ok_or("conv2 out")?.to_string();
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
    arena.lock().map_err(|e| format!("lock: {}", e)).unwrap();
    unsafe {
        let ptr = arena.base_ptr() as *mut u16;
        for i in 0..count {
            *ptr.add(i) = ((i as u16).wrapping_mul(265).wrapping_add(1234)) & 0x7FFF;
        }
    }
    arena
        .unlock()
        .map_err(|e| format!("unlock: {}", e))
        .unwrap();
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
    Ok(samples[samples.len() / 2])
}

#[test]
fn ane_activation_waterfall() {
    println!("\n=== ACTIVATION THRASHING WATERFALL ===");
    println!("Sweeps hidden dim C to find the inter-engine buffer limit between");
    println!("MAC arrays and Planar Engine (SiLU activation).");
    println!("Model: conv1(CxC1x1) → silu → conv2(CxC1x1), input x[1, C, 1, 64]");
    println!("At the spill point, SiLU intermediate no longer fits in internal buffer → DRAM round-trip overhead spikes.");
    println!("{}", "=".repeat(160));
    println!(
        "{:>6} {:>17} {:>17} {:>10} {:>10} {:>10}",
        "C", "with_silu(us)", "no_silu(us)", "diff(us)", "diff_pct", "FLOPs(G)"
    );
    println!("{}", "-".repeat(160));

    let mut prev_diff_pct: Option<f64> = None;

    for &c in C_VALUES {
        let total_flops = 256.0 * (c as f64) * (c as f64);
        let total_gflops = total_flops / 1_000_000_000.0;

        // ── Build conv→silu→conv model ────────────────────────────
        let silu_tag = format!("conv_silu_conv_c{}", c);
        let (silu_prog, silu_in, silu_out) = match build_conv_silu_conv(c) {
            Ok(v) => v,
            Err(e) => {
                println!(
                    "{:>6} {:>17} {:>17} {:>10} {:>10} {:>10.2}",
                    c, "BUILD_FAIL", "N/A", "N/A", "N/A", total_gflops
                );
                eprintln!("  silu build C={}: {}", c, e);
                continue;
            }
        };

        let silu_meta = ModelMeta {
            model_name: silu_tag.clone(),
            function_name: "main".into(),
            short_description: format!("silu_waterfall_{}", c),
            version: "1.0".into(),
            author: "prism".into(),
            output_name: silu_out.clone(),
            inputs: vec![("x".into(), vec![1, c as i64, 1, 64])],
            outputs: vec![(silu_out.clone(), vec![1, c as i64, 1, 64])],
            spec_version: 10,
        };

        let silu_path = match compile(&silu_tag, silu_prog, silu_meta) {
            Ok(p) => p,
            Err(e) => {
                println!(
                    "{:>6} {:>17} {:>17} {:>10} {:>10} {:>10.2}",
                    c, "COMPILE_FAIL", "N/A", "N/A", "N/A", total_gflops
                );
                eprintln!("  silu compile C={}: {}", c, e);
                continue;
            }
        };

        let elem_count = (c as u32) * 64;
        let silu_in_arena = Arena::new(1, elem_count, Dtype::Float16).expect("silu in arena");
        let silu_out_arena = Arena::new(1, elem_count, Dtype::Float16).expect("silu out arena");
        fill_arena(&silu_in_arena, elem_count as usize);

        let silu_time = match bench_one(
            silu_path.to_str().unwrap(),
            CoreMlComputeUnits::CpuAndNeuralEngine,
            &silu_in,
            &silu_in_arena,
            &silu_out,
            &silu_out_arena,
        ) {
            Ok(t) => t / 1000.0, // ns → us
            Err(e) => {
                println!(
                    "{:>6} {:>17} {:>17} {:>10} {:>10} {:>10.2}",
                    c, "ANE_FAIL", "N/A", "N/A", "N/A", total_gflops
                );
                eprintln!("  silu ANE C={}: {}", c, e);
                continue;
            }
        };

        // ── Build conv→conv model (no activation) ─────────────────
        let conv_tag = format!("conv_conv_c{}", c);
        let (conv_prog, conv_in, conv_out) = match build_conv_conv(c) {
            Ok(v) => v,
            Err(e) => {
                println!(
                    "{:>6} {:>17} {:>17} {:>10.1} {:>10} {:>10.2}",
                    c, silu_time, "BUILD_FAIL", "N/A", "N/A", total_gflops
                );
                eprintln!("  conv build C={}: {}", c, e);
                continue;
            }
        };

        let conv_meta = ModelMeta {
            model_name: conv_tag.clone(),
            function_name: "main".into(),
            short_description: format!("conv_waterfall_{}", c),
            version: "1.0".into(),
            author: "prism".into(),
            output_name: conv_out.clone(),
            inputs: vec![("x".into(), vec![1, c as i64, 1, 64])],
            outputs: vec![(conv_out.clone(), vec![1, c as i64, 1, 64])],
            spec_version: 10,
        };

        let conv_path = match compile(&conv_tag, conv_prog, conv_meta) {
            Ok(p) => p,
            Err(e) => {
                println!(
                    "{:>6} {:>17} {:>17} {:>10.1} {:>10} {:>10.2}",
                    c, silu_time, "COMPILE_FAIL", "N/A", "N/A", total_gflops
                );
                eprintln!("  conv compile C={}: {}", c, e);
                continue;
            }
        };

        let conv_in_arena = Arena::new(1, elem_count, Dtype::Float16).expect("conv in arena");
        let conv_out_arena = Arena::new(1, elem_count, Dtype::Float16).expect("conv out arena");
        fill_arena(&conv_in_arena, elem_count as usize);

        let conv_time = match bench_one(
            conv_path.to_str().unwrap(),
            CoreMlComputeUnits::CpuAndNeuralEngine,
            &conv_in,
            &conv_in_arena,
            &conv_out,
            &conv_out_arena,
        ) {
            Ok(t) => t / 1000.0,
            Err(e) => {
                println!(
                    "{:>6} {:>17} {:>17} {:>10.1} {:>10} {:>10.2}",
                    c, silu_time, "ANE_FAIL", "N/A", "N/A", total_gflops
                );
                eprintln!("  conv ANE C={}: {}", c, e);
                continue;
            }
        };

        let diff_us = silu_time - conv_time;
        let diff_pct = if conv_time > 0.0 {
            (silu_time - conv_time) / conv_time * 100.0
        } else {
            0.0
        };

        println!(
            "{:>6} {:>17.1} {:>17.1} {:>10.1} {:>9.2}% {:>10.2}",
            c, silu_time, conv_time, diff_us, diff_pct, total_gflops
        );

        // Detect a sudden spike in diff_pct (≥2× the previous value indicates spill)
        if let Some(prev) = prev_diff_pct {
            if prev > 0.5 && diff_pct > prev * 2.0 {
                println!("  → SPILL DETECTED at C={}: diff_pct jumped from {:.1}% to {:.1}% (>{:.0}% increase)",
                    c, prev, diff_pct, (diff_pct - prev));
            }
        }
        prev_diff_pct = Some(diff_pct);
    }

    println!("{}", "=".repeat(160));
    println!("Expected: at small C, silu fits in internal MAC→PlanarEngine buffer → low overhead.");
    println!("At a specific C, intermediate tensor spills to DRAM → overhead spike (sudden ↑ in diff_pct).");
    println!("The C at which the spike occurs reveals the inter-engine buffer capacity.");
}
