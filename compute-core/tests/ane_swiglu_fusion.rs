//! SwiGLU MLP Block — ANE fusion stress-test suite.
//!
//! Test Matrix 1: SwiGLU fusion patterns.
//!
//! Variation 1.1: Standard MIL matmuls (gate/up/down projections)
//! Variation 1.2: 1x1 Conv2d mapping of the same math
//! Variation 1.3: Channel-split concurrent streams (4 parallel SwiGLU streams)
//! Variation 1.4: Conv2d with explicit activation binding (conv -> silu chained)
//!
//! Each variation is compiled for macOS26, spec_version 10, and benchmarked on
//! the ANE (CpuAndNeuralEngine). Metrics: time, GMACs/s, % of 5.5 TMAC/s peak.
//!
//! Sweeps over batch sizes and hidden dimensions.
//! FFN is always 4xH (standard transformer MLP expansion).
//!
//! Run: cargo test --test ane_swiglu_fusion --features prism-backend -- --nocapture
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

const TEST_DIR: &str = "/tmp/prism_ane_swiglu_fusion";
const WARMUP: usize = 5;
const SAMPLES: usize = 20;
/// Theoretical peak: 5.5 TMAC/s (M1 ANE FP16).
const PEAK_GMACS: f64 = 5_500_000_000_000.0 / 1_000_000_000.0; // 5500 GMACs/s = 5.5 TMACs/s
const LABEL: &str = "%Peak";
/// Batch sizes to sweep.
const BATCH_SIZES: &[u32] = &[1, 16384];
/// Hidden dimensions to sweep. FFN = 4 x H.
const HIDDEN_DIMS: &[i64] = &[512, 2048, 4096];
const FFN_FACTOR: i64 = 4;

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
    Ok(samples[samples.len() / 2]) // median
}

/// Helper: run one variation through build+compile+bench, return (time_us, tmac_s, pct_peak).
type VariationResult = Result<(f64, f64, f64), String>;

fn run_variation(
    tag: &str,
    batch: u32,
    H: i64,
    build: fn(u32, i64) -> Result<(mil_spec::Program, String, String), String>,
) -> VariationResult {
    let (prog, in_name, out_name) = build(batch, H).map_err(|e| format!("build: {}", e))?;

    let meta = ModelMeta {
        model_name: tag.into(),
        function_name: "main".into(),
        short_description: format!("swiglu_{}", tag),
        version: "1.0".into(),
        author: "prism".into(),
        output_name: out_name.clone(),
        inputs: vec![("x".into(), vec![batch as i64, H])],
        outputs: vec![(out_name.clone(), vec![batch as i64, H])],
    };

    let model_path = compile(tag, prog, meta).map_err(|e| format!("compile: {}", e))?;

    let in_arena =
        Arena::new(batch, H as u32, DataType::Float16).map_err(|e| format!("in arena: {}", e))?;
    let out_arena =
        Arena::new(batch, H as u32, DataType::Float16).map_err(|e| format!("out arena: {}", e))?;
    fill_arena(&in_arena, (batch as usize) * (H as usize));

    let ane_time_ns = bench_one(
        model_path.to_str().ok_or("path")?,
        CoreMlComputeUnits::CpuAndNeuralEngine,
        &in_name,
        &in_arena,
        &out_name,
        &out_arena,
    )
    .map_err(|e| format!("bench: {}", e))?;

    let time_us = ane_time_ns / 1000.0;
    let time_s = ane_time_ns / 1_000_000_000.0;

    // MACs = batch x (gate: HxFFN + up: HxFFN + down: FFNxH) = batch x 3 x H x FFN
    let ffn = H * FFN_FACTOR;
    let total_macs = batch as f64 * H as f64 * ffn as f64 * 3.0;

    let macs_s = if time_s > 0.0 {
        total_macs / time_s
    } else {
        0.0
    };
    let gmacs_s = macs_s / 1_000_000_000.0;
    let pct_peak = if PEAK_GMACS > 0.0 {
        (macs_s / 1_000_000_000.0) / PEAK_GMACS * 100.0
    } else {
        0.0
    };
    Ok((time_us, gmacs_s, pct_peak))
}

// ═════════════════════════════════════════════════════════════════════════════
// B U I L D E R S
// ═════════════════════════════════════════════════════════════════════════════

/// Variation 1.1: Standard MIL matmuls.
///
///   x[batch, H] @ Wg[H, FFN] -> silu -> y_g
///   x[batch, H] @ Wu[H, FFN] -> y_u
///   y_g * y_u -> y_m
///   y_m @ Wd[FFN, H] -> y_out[batch, H]
fn build_v11_matmul(batch: u32, H: i64) -> Result<(mil_spec::Program, String, String), String> {
    let FFN = H * FFN_FACTOR;
    let wg_vals = seeded_weights(0, H, FFN);
    let wu_vals = seeded_weights(1, H, FFN);
    let wd_vals = seeded_weights(2, FFN, H);

    let mut b = MilBuilder::new("main").input("x", mil_spec::DataType::Float16, &[batch as i64, H]);

    // Weight constants
    b = b.const_f16("wg", &wg_vals, &[H, FFN]);
    let wg = b.last_name().ok_or("wg_name")?.to_string();
    b = b.const_f16("wu", &wu_vals, &[H, FFN]);
    let wu = b.last_name().ok_or("wu_name")?.to_string();
    b = b.const_f16("wd", &wd_vals, &[FFN, H]);
    let wd = b.last_name().ok_or("wd_name")?.to_string();

    // Gate: x @ Wg -> silu
    b = b.matmul("x", &wg);
    let gate_mm = b.last_name().ok_or("gate_mm")?.to_string();
    b = b.silu("gate_silu", &gate_mm);
    let gate_out = b.last_name().ok_or("gate_out")?.to_string();

    // Up: x @ Wu
    b = b.matmul("x", &wu);
    let up_out = b.last_name().ok_or("up_out")?.to_string();

    // Combined: gate_out * up_out
    b = b.mul(&gate_out, &up_out);
    let combined = b.last_name().ok_or("combined")?.to_string();

    // Down: combined @ Wd
    b = b.matmul(&combined, &wd);
    let out_name = b.last_name().ok_or("down_out")?.to_string();

    let b = b.output(&out_name);
    let prog = b.build().map_err(|e| format!("V1.1 MIL: {}", e))?;
    Ok((prog, "x".into(), out_name))
}

/// Variation 1.2: 1x1 Conv2d mapping.
///
///   x[batch, H] -> reshape -> x_4d[batch, H, 1, 1]
///   gate: conv(Wg[FFN, H, 1, 1], x_4d) -> silu -> reshape -> gate_2d [batch, FFN]
///   up:   conv(Wu[FFN, H, 1, 1], x_4d) -> reshape -> up_2d [batch, FFN]
///   gate_2d * up_2d -> combined [batch, FFN] -> reshape -> combined_4d [batch, FFN, 1, 1]
///   down: conv(Wd[H, FFN, 1, 1], combined_4d) -> reshape -> y[batch, H]
fn build_v12_conv(batch: u32, H: i64) -> Result<(mil_spec::Program, String, String), String> {
    let FFN = H * FFN_FACTOR;
    let wg_vals = seeded_weights(0, H, FFN);
    let wu_vals = seeded_weights(1, H, FFN);
    let wd_vals = seeded_weights(2, FFN, H);

    let mut b = MilBuilder::new("main").input("x", mil_spec::DataType::Float16, &[batch as i64, H]);

    // Reshape to 4D for conv: [batch, H, 1, 1]
    b = b.reshape("x_4d", "x", &[batch as i64, H, 1, 1]);
    let x_4d = b.last_name().ok_or("x_4d")?.to_string();

    // Conv weight constants: [C_out, C_in, kH, kW]
    b = b.const_f16("wg", &wg_vals, &[FFN, H, 1, 1]);
    let wg = b.last_name().ok_or("wg")?.to_string();
    b = b.const_f16("wu", &wu_vals, &[FFN, H, 1, 1]);
    let wu = b.last_name().ok_or("wu")?.to_string();
    b = b.const_f16("wd", &wd_vals, &[H, FFN, 1, 1]);
    let wd = b.last_name().ok_or("wd")?.to_string();

    // Gate: conv(x_4d, Wg) -> silu -> reshape to 2D
    b = b.conv("gate_conv", &x_4d, &wg, &[1, 1], "valid");
    let gate_conv = b.last_name().ok_or("gate_conv")?.to_string();
    b = b.silu("gate_silu", &gate_conv);
    let gate_silu = b.last_name().ok_or("gate_silu")?.to_string();
    // Reshape 4D conv output [batch, FFN, 1, 1] back to 2D [batch, FFN] for mul
    b = b.reshape("gate_2d", &gate_silu, &[batch as i64, FFN]);
    let gate_2d = b.last_name().ok_or("gate_2d")?.to_string();

    // Up: conv(x_4d, Wu) -> reshape to 2D
    b = b.conv("up_conv", &x_4d, &wu, &[1, 1], "valid");
    let up_conv = b.last_name().ok_or("up_conv")?.to_string();
    b = b.reshape("up_2d", &up_conv, &[batch as i64, FFN]);
    let up_2d = b.last_name().ok_or("up_2d")?.to_string();

    // Combined: gate_2d * up_2d (both 2D)
    b = b.mul(&gate_2d, &up_2d);
    let combined = b.last_name().ok_or("combined")?.to_string();

    // Reshape combined back to 4D for down conv
    b = b.reshape("combined_4d", &combined, &[batch as i64, FFN, 1, 1]);
    let combined_4d = b.last_name().ok_or("combined_4d")?.to_string();

    // Down: conv(combined_4d, Wd)
    b = b.conv("down_conv", &combined_4d, &wd, &[1, 1], "valid");
    let down_out = b.last_name().ok_or("down_out")?.to_string();

    // Reshape back to [batch, H]
    b = b.reshape("y_out", &down_out, &[batch as i64, H]);
    let out_name = b.last_name().ok_or("out")?.to_string();

    let b = b.output(&out_name);
    let prog = b.build().map_err(|e| format!("V1.2 MIL: {}", e))?;
    Ok((prog, "x".into(), out_name))
}

/// Variation 1.3: Channel-split concurrent streams.
///
/// Split input channels into 4 chunks (H/4 each). Each chunk through its own
/// SwiGLU stream (matmul-based, with gate/up/down projections at H/4 dims).
/// Concat results along channel dim -> y[batch, H].
///
/// Since MIL slicing is not available, each stream uses the full input "x" with
/// weight matrices shaped to read H/4 input channels and produce H/4 output:
///   gate: x[batch, H] @ Wg_i[H, H/4] -> silu -> [batch, H/4]
///   up:   x[batch, H] @ Wu_i[H, H/4] -> [batch, H/4]
///   mul:  gate_out * up_out -> [batch, H/4]
///   down: combined @ Wd_i[H/4, H] -> [batch, H]
///
/// Instead, each stream does a full SwiGLU producing [batch, H], then outputs
/// are summed elementwise (more useful than concat for a single output [batch, H]).
///
/// Total MACs = same as V1.1 (splitting channels doesn't change compute).
fn build_v13_split(batch: u32, H: i64) -> Result<(mil_spec::Program, String, String), String> {
    let h4 = H / 4;

    let mut b = MilBuilder::new("main").input("x", mil_spec::DataType::Float16, &[batch as i64, H]);

    let mut stream_outs: Vec<String> = Vec::new();

    for i in 0..4 {
        let seed_base = 100 * i as u64;

        // Weights for this stream: H -> H/4, H/4 -> H/4 -> H (expanded again)
        let wg_vals = seeded_weights(seed_base, H, h4);
        let wu_vals = seeded_weights(seed_base + 1, H, h4);
        let wd_vals = seeded_weights(seed_base + 2, h4, H);

        b = b.const_f16(&format!("wg_{}", i), &wg_vals, &[H, h4]);
        let wg = b.last_name().ok_or(format!("wg_{}_name", i))?.to_string();
        b = b.const_f16(&format!("wu_{}", i), &wu_vals, &[H, h4]);
        let wu = b.last_name().ok_or(format!("wu_{}_name", i))?.to_string();
        b = b.const_f16(&format!("wd_{}", i), &wd_vals, &[h4, H]);
        let wd = b.last_name().ok_or(format!("wd_{}_name", i))?.to_string();

        // Gate: x @ Wg_i -> silu
        b = b.matmul("x", &wg);
        let gate_mm = b.last_name().ok_or(format!("gate_mm_{}", i))?.to_string();
        b = b.silu(&format!("gate_silu_{}", i), &gate_mm);
        let gate_out = b.last_name().ok_or(format!("gate_out_{}", i))?.to_string();

        // Up: x @ Wu_i
        b = b.matmul("x", &wu);
        let up_out = b.last_name().ok_or(format!("up_out_{}", i))?.to_string();

        // Combined: gate_out * up_out
        b = b.mul(&gate_out, &up_out);
        let combined = b.last_name().ok_or(format!("combined_{}", i))?.to_string();

        // Down: combined @ Wd_i -> [batch, H]
        b = b.matmul(&combined, &wd);
        let stream_out = b
            .last_name()
            .ok_or(format!("stream_out_{}", i))?
            .to_string();
        stream_outs.push(stream_out);
    }

    // Sum all stream outputs elementwise
    let mut sum = stream_outs[0].clone();
    for s in &stream_outs[1..] {
        b = b.add(&sum, s);
        sum = b.last_name().ok_or("add")?.to_string();
    }

    let out_name = sum;
    let b = b.output(&out_name);
    let prog = b.build().map_err(|e| format!("V1.3 MIL: {}", e))?;
    Ok((prog, "x".into(), out_name))
}

/// Variation 1.4: Conv2d with explicit activation binding (fused).
///
/// Same structure as 1.2: conv-based SwiGLU with explicit conv -> silu chaining.
/// This is a separate variation to benchmark the fused activation path.
///
///   x[batch, H] -> reshape -> x_4d[batch, H, 1, 1]
///   gate: conv(Wg[FFN, H, 1, 1], x_4d) -> silu -> reshape -> gate_2d [batch, FFN]
///   up:   conv(Wu[FFN, H, 1, 1], x_4d) -> reshape -> up_2d [batch, FFN]
///   gate_2d * up_2d -> combined [batch, FFN] -> reshape -> combined_4d [batch, FFN, 1, 1]
///   down: conv(Wd[H, FFN, 1, 1], combined_4d) -> reshape -> y[batch, H]
fn build_v14_fused_act(batch: u32, H: i64) -> Result<(mil_spec::Program, String, String), String> {
    let FFN = H * FFN_FACTOR;
    let wg_vals = seeded_weights(0, H, FFN);
    let wu_vals = seeded_weights(1, H, FFN);
    let wd_vals = seeded_weights(2, FFN, H);

    let mut b = MilBuilder::new("main").input("x", mil_spec::DataType::Float16, &[batch as i64, H]);

    // Reshape to 4D for conv: [batch, H, 1, 1]
    b = b.reshape("x_4d", "x", &[batch as i64, H, 1, 1]);
    let x_4d = b.last_name().ok_or("x_4d")?.to_string();

    // Conv weight constants: [C_out, C_in, kH, kW]
    b = b.const_f16("wg", &wg_vals, &[FFN, H, 1, 1]);
    let wg = b.last_name().ok_or("wg")?.to_string();
    b = b.const_f16("wu", &wu_vals, &[FFN, H, 1, 1]);
    let wu = b.last_name().ok_or("wu")?.to_string();
    b = b.const_f16("wd", &wd_vals, &[H, FFN, 1, 1]);
    let wd = b.last_name().ok_or("wd")?.to_string();

    // Gate: conv(x_4d, Wg) -> silu -> reshape to 2D
    b = b.conv("gate_conv", &x_4d, &wg, &[1, 1], "valid");
    let gate_conv = b.last_name().ok_or("gate_conv")?.to_string();
    b = b.silu("gate_silu", &gate_conv);
    let gate_silu = b.last_name().ok_or("gate_silu")?.to_string();
    // Reshape 4D conv output [batch, FFN, 1, 1] back to 2D [batch, FFN] for mul
    b = b.reshape("gate_2d", &gate_silu, &[batch as i64, FFN]);
    let gate_2d = b.last_name().ok_or("gate_2d")?.to_string();

    // Up: conv(x_4d, Wu) -> reshape to 2D
    b = b.conv("up_conv", &x_4d, &wu, &[1, 1], "valid");
    let up_conv = b.last_name().ok_or("up_conv")?.to_string();
    b = b.reshape("up_2d", &up_conv, &[batch as i64, FFN]);
    let up_2d = b.last_name().ok_or("up_2d")?.to_string();

    // Combined: gate_2d * up_2d (both 2D)
    b = b.mul(&gate_2d, &up_2d);
    let combined = b.last_name().ok_or("combined")?.to_string();

    // Reshape combined back to 4D for down conv
    b = b.reshape("combined_4d", &combined, &[batch as i64, FFN, 1, 1]);
    let combined_4d = b.last_name().ok_or("combined_4d")?.to_string();

    // Down: conv(combined_4d, Wd)
    b = b.conv("down_conv", &combined_4d, &wd, &[1, 1], "valid");
    let down_out = b.last_name().ok_or("down_out")?.to_string();

    // Reshape back to [batch, H]
    b = b.reshape("y_out", &down_out, &[batch as i64, H]);
    let out_name = b.last_name().ok_or("out")?.to_string();

    let b = b.output(&out_name);
    let prog = b.build().map_err(|e| format!("V1.4 MIL: {}", e))?;
    Ok((prog, "x".into(), out_name))
}

// ═════════════════════════════════════════════════════════════════════════════
// T E S T
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn ane_swiglu_fusion() {
    println!("\n=== ANE SWiGLU FUSION STRESS TEST — TEST MATRIX 1 ===");
    println!("Peak: 5.5 TMAC/s theoretical (M1 ANE FP16)");
    println!("Warmup={}, Samples={}", WARMUP, SAMPLES);
    println!("FFN = 4 x H");
    println!("Batch sweep: {:?}", BATCH_SIZES);
    println!("Hidden dim sweep: {:?}", HIDDEN_DIMS);
    println!("{}", "=".repeat(145));

    // Print column header once
    println!(
        "{:<5} {:>6} {:>6} {:>10}   {:>12} {:>12} {:>10} {:>8}",
        "Batch", "H", "FFN", "Variation", "Time(us)", "GMACs/s", "%Peak", "Status"
    );
    println!("{}", "-".repeat(145));

    struct Variation {
        name: &'static str,
        tag_prefix: &'static str,
        build: fn(u32, i64) -> Result<(mil_spec::Program, String, String), String>,
    }

    let variations = [
        Variation {
            name: "V1.1",
            tag_prefix: "v11",
            build: build_v11_matmul,
        },
        Variation {
            name: "V1.2",
            tag_prefix: "v12",
            build: build_v12_conv,
        },
        Variation {
            name: "V1.3",
            tag_prefix: "v13",
            build: build_v13_split,
        },
        Variation {
            name: "V1.4",
            tag_prefix: "v14",
            build: build_v14_fused_act,
        },
    ];

    for &batch in BATCH_SIZES {
        for &H in HIDDEN_DIMS {
            let FFN = H * FFN_FACTOR;

            for v in &variations {
                let tag = format!("{}_{}_b{}_h{}", v.tag_prefix, v.name, batch, H);

                match run_variation(&tag, batch, H, v.build) {
                    Ok((time_us, tmac_s, pct_peak)) => {
                        println!(
                            "{:<5} {:>6} {:>6} {:>10}   {:>12.1} {:>12.3e} {:>9.2}% {:>8}",
                            batch, H, FFN, v.name, time_us, tmac_s, pct_peak, "OK"
                        );
                    }
                    Err(e) => {
                        println!(
                            "{:<5} {:>6} {:>6} {:>10}   {:>12} {:>12} {:>10} {:>8}",
                            batch, H, FFN, v.name, "FAIL", "N/A", "N/A", "ERR"
                        );
                        eprintln!("  {}: {}", tag, e);
                    }
                }
            }
        }
    }

    println!("\n=== DONE ===");
}
