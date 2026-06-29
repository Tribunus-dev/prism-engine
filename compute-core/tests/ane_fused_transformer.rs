//! ANE fused transformer macro-block — IOSurface round-trip reduction test.
//!
//! Compares a single fused MIL graph (SwiGLU + residual) against 5 separate
//! models for the same compute.  The fused graph keeps ALL intermediate
//! results in ANE SRAM (~32 MB), requiring only one IOSurface round-trip.
//! Separate models flush to IOSurface between every op — 5 round-trips.
//!
//! Fused topology:
//!   x[1,H] → matmul(Wg[H,FFN]) → silu → y_gate
//!   x[1,H] → matmul(Wu[H,FFN])         → y_up
//!   mul(y_gate, y_up)                    → y_mlp
//!   matmul(y_mlp, Wd[FFN,H])            → y_down
//!   add(x, y_down)                       → y_out
//!
//! Isolated: 5 separate models, one per op (gate, up, mul, down, add).
//! Total MACs = 3 × batch × H × FFN  (swiglu + residual; elementwise ops ~0).
//!
//! Key metric: speedup = Σ(isolated_times) / fused_time.
//! Expect >2× for small compute (I/O dominated) diminishing to ~1.1× for large.
//!
//! Run: cargo test --test ane_fused_transformer --features prism-backend -- --nocapture
//! Requires: macOS 14.0+ on Apple Silicon

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

// ── Constants ──────────────────────────────────────────────────────────────

const TEST_DIR: &str = "/tmp/prism_ane_fused_transformer";
const WARMUP: usize = 5;
const SAMPLES: usize = 15;
/// Theoretical peak: 5.5 TMAC/s (M1 ANE FP16).
const PEAK_GMACS: f64 = 5500.0;
const LABEL: &str = "%Peak";
/// Batch sizes to sweep.
const BATCH_SIZES: &[u32] = &[1, 16384];
/// Hidden dimensions to sweep. FFN = 4 × H.
const HIDDEN_DIMS: &[i64] = &[512, 2048];
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

// ═════════════════════════════════════════════════════════════════════════════
// B U I L D E R S  —  F U S E D
// ═════════════════════════════════════════════════════════════════════════════

/// Build the fused SwiGLU + residual MIL graph.
///
/// SSA trace (counter starts at 0):
///   .input("x", ...)                            → "x"
///   .const_f16("wg", ...)                       → "wg_0"
///   .const_f16("wu", ...)                       → "wu_1"
///   .const_f16("wd", ...)                       → "wd_2"
///   .matmul("x", "wg_0")                       → "matmul_3"
///   .silu("gate_silu", "matmul_3")             → "gate_silu_4"
///   .matmul("x", "wu_1")                       → "matmul_5"
///   .mul("gate_silu_4", "matmul_5")            → "mul_6"
///   .matmul("mul_6", "wd_2")                   → "matmul_7"
///   .add("x", "matmul_7")                      → "add_8"
///   .output("add_8")
///
/// Input: "x" [batch, H]
/// Output: "add_8" [batch, H]
fn build_fused(batch: u32, H: i64) -> Result<(mil_spec::Program, String, String), String> {
    let FFN = H * FFN_FACTOR;
    let wg_vals = seeded_weights(100, H, FFN);
    let wu_vals = seeded_weights(101, H, FFN);
    let wd_vals = seeded_weights(102, FFN, H);

    let mut b = MilBuilder::new("main").input("x", mil_spec::DataType::Float16, &[batch as i64, H]);

    // Weights
    b = b.const_f16("wg", &wg_vals, &[H, FFN]);
    let wg = b.last_name().ok_or("wg")?.to_string();
    b = b.const_f16("wu", &wu_vals, &[H, FFN]);
    let wu = b.last_name().ok_or("wu")?.to_string();
    b = b.const_f16("wd", &wd_vals, &[FFN, H]);
    let wd = b.last_name().ok_or("wd")?.to_string();

    // Gate: x @ Wg → silu
    b = b.matmul("x", &wg);
    let gate_mm = b.last_name().ok_or("gate_mm")?.to_string();
    b = b.silu("gate_silu", &gate_mm);
    let gate_out = b.last_name().ok_or("gate_out")?.to_string();

    // Up: x @ Wu
    b = b.matmul("x", &wu);
    let up_out = b.last_name().ok_or("up_out")?.to_string();

    // Combine: gate_out * up_out → y_mlp
    b = b.mul(&gate_out, &up_out);
    let combined = b.last_name().ok_or("combined")?.to_string();

    // Down: y_mlp @ Wd → y_down
    b = b.matmul(&combined, &wd);
    let down_out = b.last_name().ok_or("down_out")?.to_string();

    // Residual: add(x, y_down) → y_out
    b = b.add("x", &down_out);
    let out_name = b.last_name().ok_or("out")?.to_string();

    let b = b.output(&out_name);
    let prog = b.build().map_err(|e| format!("fused MIL: {}", e))?;
    Ok((prog, "x".into(), out_name))
}

// ═════════════════════════════════════════════════════════════════════════════
// B U I L D E R S  —  I S O L A T E D
// ═════════════════════════════════════════════════════════════════════════════

/// Isolated gate model: x[1,H] @ Wg[H,FFN] → silu → [1,FFN]
fn build_gate_isolated(batch: u32, H: i64) -> Result<(mil_spec::Program, String, String), String> {
    let FFN = H * FFN_FACTOR;
    let wg_vals = seeded_weights(100, H, FFN);

    let mut b = MilBuilder::new("main").input("x", mil_spec::DataType::Float16, &[batch as i64, H]);

    b = b.const_f16("wg", &wg_vals, &[H, FFN]);
    let wg = b.last_name().ok_or("wg")?.to_string();

    b = b.matmul("x", &wg);
    let mm = b.last_name().ok_or("mm")?.to_string();
    b = b.silu("gate_out", &mm);
    let out_name = b.last_name().ok_or("out")?.to_string();

    let b = b.output(&out_name);
    let prog = b.build().map_err(|e| format!("gate isolated MIL: {}", e))?;
    Ok((prog, "x".into(), out_name))
}

/// Isolated up model: x[1,H] @ Wu[H,FFN] → [1,FFN]
fn build_up_isolated(batch: u32, H: i64) -> Result<(mil_spec::Program, String, String), String> {
    let FFN = H * FFN_FACTOR;
    let wu_vals = seeded_weights(101, H, FFN);

    let mut b = MilBuilder::new("main").input("x", mil_spec::DataType::Float16, &[batch as i64, H]);

    b = b.const_f16("wu", &wu_vals, &[H, FFN]);
    let wu = b.last_name().ok_or("wu")?.to_string();

    b = b.matmul("x", &wu);
    let out_name = b.last_name().ok_or("out")?.to_string();

    let b = b.output(&out_name);
    let prog = b.build().map_err(|e| format!("up isolated MIL: {}", e))?;
    Ok((prog, "x".into(), out_name))
}

/// Isolated elementwise-mul model: x[1,FFN] * const[1,FFN] → [1,FFN]
///
/// Second operand is baked as a constant (MIL `const_f16`).  The compute path
/// (IOSurface → elementwise → IOSurface) is the same; only the data-dependent
/// value changes.  Benchmark overhead is dominated by the IOSurface round-trip,
/// not the tiny numerical difference from a dynamic second source.
fn build_mul_isolated(batch: u32, H: i64) -> Result<(mil_spec::Program, String, String), String> {
    let FFN = H * FFN_FACTOR;
    // Ones — mul(x, 1) ≈ identity, but the elementwise pipeline is the same.
    let const_vals: Vec<f32> = vec![1.0; FFN as usize];

    let mut b =
        MilBuilder::new("main").input("x", mil_spec::DataType::Float16, &[batch as i64, FFN]);

    b = b.const_f16("const_factor", &const_vals, &[1, FFN]);
    let c = b.last_name().ok_or("const")?.to_string();

    b = b.mul("x", &c);
    let out_name = b.last_name().ok_or("out")?.to_string();

    let b = b.output(&out_name);
    let prog = b.build().map_err(|e| format!("mul isolated MIL: {}", e))?;
    Ok((prog, "x".into(), out_name))
}

/// Isolated down-projection model: x[1,FFN] @ Wd[FFN,H] → [1,H]
fn build_down_isolated(batch: u32, H: i64) -> Result<(mil_spec::Program, String, String), String> {
    let FFN = H * FFN_FACTOR;
    let wd_vals = seeded_weights(102, FFN, H);

    let mut b =
        MilBuilder::new("main").input("x", mil_spec::DataType::Float16, &[batch as i64, FFN]);

    b = b.const_f16("wd", &wd_vals, &[FFN, H]);
    let wd = b.last_name().ok_or("wd")?.to_string();

    b = b.matmul("x", &wd);
    let out_name = b.last_name().ok_or("out")?.to_string();

    let b = b.output(&out_name);
    let prog = b.build().map_err(|e| format!("down isolated MIL: {}", e))?;
    Ok((prog, "x".into(), out_name))
}

/// Isolated elementwise-add model: x[1,H] + const[1,H] → [1,H]
///
/// Same rationale as `build_mul_isolated`: the constant second operand means
/// the IOSurface round-trip dominates timing, not the actual data.
fn build_add_isolated(batch: u32, H: i64) -> Result<(mil_spec::Program, String, String), String> {
    // Zeros — add(x, 0) ≈ identity.
    let const_vals: Vec<f32> = vec![0.0; H as usize];

    let mut b = MilBuilder::new("main").input("x", mil_spec::DataType::Float16, &[batch as i64, H]);

    b = b.const_f16("const_offset", &const_vals, &[1, H]);
    let c = b.last_name().ok_or("const")?.to_string();

    b = b.add("x", &c);
    let out_name = b.last_name().ok_or("out")?.to_string();

    let b = b.output(&out_name);
    let prog = b.build().map_err(|e| format!("add isolated MIL: {}", e))?;
    Ok((prog, "x".into(), out_name))
}

// ═════════════════════════════════════════════════════════════════════════════
// B E N C H M A R K  H E L P E R
// ═════════════════════════════════════════════════════════════════════════════

/// Compile a model and return the median ANE latency (ns).
fn bench_model(
    tag: &str,
    batch: u32,
    rows: i64,
    cols: i64,
    in_name: &str,
    out_name: &str,
    prog: mil_spec::Program,
) -> Result<f64, String> {
    let meta = ModelMeta {
        model_name: tag.into(),
        function_name: "main".into(),
        short_description: format!("ane_fused_{}", tag),
        version: "1.0".into(),
        author: "prism".into(),
        output_name: out_name.into(),
        inputs: vec![("x".into(), vec![batch as i64, rows])],
        outputs: vec![(out_name.into(), vec![batch as i64, cols])],
        spec_version: 10,
    };

    let model_path = compile(tag, prog, meta).map_err(|e| format!("compile {}: {}", tag, e))?;
    let path_str = model_path.to_str().ok_or(format!("bad path for {}", tag))?;

    let in_arena = Arena::new(batch, rows as u32, Dtype::Float16)
        .map_err(|e| format!("{} in arena: {}", tag, e))?;
    let out_arena = Arena::new(batch, cols as u32, Dtype::Float16)
        .map_err(|e| format!("{} out arena: {}", tag, e))?;
    fill_arena(&in_arena, (batch as usize) * (rows as usize));

    let time_ns = bench_one(
        path_str,
        CoreMlComputeUnits::CpuAndNeuralEngine,
        in_name,
        &in_arena,
        out_name,
        &out_arena,
    )
    .map_err(|e| format!("{} bench: {}", tag, e))?;

    Ok(time_ns)
}

// ═════════════════════════════════════════════════════════════════════════════
// T E S T
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn ane_fused_transformer() {
    println!("\n=== ANE FUSED TRANSFORMER — MACRO-BLOCK IOSURFACE REDUCTION ===");
    println!("Peak: {:.1} GMAC/s (M1 ANE FP16)", PEAK_GMACS);
    println!("Warmup={}  Samples={}", WARMUP, SAMPLES);
    println!("FFN = {} × H", FFN_FACTOR);
    println!("Batch sweep: {:?}", BATCH_SIZES);
    println!("Hidden dim sweep: {:?}", HIDDEN_DIMS);
    println!("Key metric: speedup = Σ(isolated_times) / fused_time");
    println!("{}", "=".repeat(145));

    println!(
        "{:>6} {:>6} {:>7} {:>14} {:>14} {:>10} {:>10} {:>8}",
        "Batch", "H", "FFN", "Fused(us)", "Isolated(us)", "Speedup", "GMAC/s", "%Peak"
    );
    println!("{}", "-".repeat(145));

    for &batch in BATCH_SIZES {
        for &H in HIDDEN_DIMS {
            let FFN = H * FFN_FACTOR;
            let tag_prefix = format!("b{}_h{}", batch, H);

            // ── Fused: build + compile + bench ──────────────────────
            let fused_time_ns = match (|| -> Result<f64, String> {
                let (prog, in_name, out_name) = build_fused(batch, H)?;
                bench_model(
                    &format!("fused_{}", tag_prefix),
                    batch,
                    H,
                    H,
                    &in_name,
                    &out_name,
                    prog,
                )
            })() {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("  FUSED: {}", e);
                    println!(
                        "{:>6} {:>6} {:>7} {:>14} {:>14} {:>10} {:>10} {:>8}",
                        batch, H, FFN, "FAIL", "N/A", "N/A", "N/A", "ERR"
                    );
                    continue;
                }
            };

            // ── Isolated: 5 separate models, bench each, sum ───────
            let isolated_time_ns = match (|| -> Result<f64, String> {
                let (p_gate, in_gate, out_gate) = build_gate_isolated(batch, H)?;
                let t_gate = bench_model(
                    &format!("gate_{}", tag_prefix),
                    batch,
                    H,
                    FFN,
                    &in_gate,
                    &out_gate,
                    p_gate,
                )?;
                let (p_up, in_up, out_up) = build_up_isolated(batch, H)?;
                let t_up = bench_model(
                    &format!("up_{}", tag_prefix),
                    batch,
                    H,
                    FFN,
                    &in_up,
                    &out_up,
                    p_up,
                )?;
                let (p_mul, in_mul, out_mul) = build_mul_isolated(batch, H)?;
                let t_mul = bench_model(
                    &format!("mul_{}", tag_prefix),
                    batch,
                    1,
                    FFN,
                    &in_mul,
                    &out_mul,
                    p_mul,
                )?;
                let (p_down, in_down, out_down) = build_down_isolated(batch, H)?;
                let t_down = bench_model(
                    &format!("down_{}", tag_prefix),
                    batch,
                    FFN,
                    H,
                    &in_down,
                    &out_down,
                    p_down,
                )?;
                let (p_add, in_add, out_add) = build_add_isolated(batch, H)?;
                let t_add = bench_model(
                    &format!("add_{}", tag_prefix),
                    batch,
                    H,
                    H,
                    &in_add,
                    &out_add,
                    p_add,
                )?;
                Ok(t_gate + t_up + t_mul + t_down + t_add)
            })() {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("  ISOLATED: {}", e);
                    let t_us = fused_time_ns / 1000.0;
                    let total_macs = batch as f64 * H as f64 * FFN as f64 * 3.0;
                    let time_s = fused_time_ns / 1_000_000_000.0;
                    let gmacs = if time_s > 0.0 {
                        total_macs / time_s / 1_000_000_000.0
                    } else {
                        0.0
                    };
                    let pct = if PEAK_GMACS > 0.0 {
                        gmacs / PEAK_GMACS * 100.0
                    } else {
                        0.0
                    };
                    // Can show fused result even if isolated failed
                    println!(
                        "{:>6} {:>6} {:>7} {:>14.1} {:>14} {:>10} {:>10.2} {:>7.1}%",
                        batch, H, FFN, t_us, "ISOLATED_FAIL", "N/A", gmacs, pct
                    );
                    continue;
                }
            };

            // ── Compute metrics ────────────────────────────────────
            let fused_us = fused_time_ns / 1000.0;
            let isolated_us = isolated_time_ns / 1000.0;
            let speedup = if fused_time_ns > 0.0 {
                isolated_time_ns / fused_time_ns
            } else {
                0.0
            };

            // MACs = 3 matmuls × batch × H × FFN
            let total_macs = batch as f64 * H as f64 * FFN as f64 * 3.0;
            let time_s = fused_time_ns / 1_000_000_000.0;
            let gmacs = if time_s > 0.0 {
                total_macs / time_s / 1_000_000_000.0
            } else {
                0.0
            };
            let pct = if PEAK_GMACS > 0.0 {
                gmacs / PEAK_GMACS * 100.0
            } else {
                0.0
            };

            println!(
                "{:>6} {:>6} {:>7} {:>14.1} {:>14.1} {:>9.2}× {:>10.2} {:>7.1}%",
                batch, H, FFN, fused_us, isolated_us, speedup, gmacs, pct
            );
        }
    }

    println!("\n=== DONE ===");
}
