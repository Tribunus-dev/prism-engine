//! ANE vs GPU attention crossover benchmark.
//!
//! Measures per-component latency (QKV projection, score matmul, weighted-V,
//! output projection) across sequence lengths S ∈ {1,8,32,128,512} on Apple
//! Silicon ANE vs GPU Q4 kernel.  Identifies the crossover point where ANE's
//! compute advantage overtakes GPU's bandwidth advantage for the attention
//! phase of transformer inference.
//!
//! Architecture:
//!   ANE path: separate Core ML FP16 matmul models per operation
//!   GPU path: Q4 (4-bit quantized) Metal GEMV kernel
//!   Both paths reuse QKV and output projection models across all S; only
//!   the score (attention) models change with sequence length.
//!
//! Run:  cargo test --test attn_lane_crossover --features prism-backend -- --nocapture
//! Requires: macOS 14.0+ on Apple Silicon

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

// ── Constants ──────────────────────────────────────────────────────────────

/// Hidden dimension (half Llama-3 8B).
const H: usize = 2048;
/// Number of attention heads.
const NH: usize = 8;
/// Head dimension.
const HD: usize = 128;
/// QKV output dimension = 3 × hidden.
const QKV_DIM: usize = 6144;
/// Q4 group size.
const GS: usize = 32;
/// Warmup iterations per model.
const WARMUP: usize = 10;
/// Measured iterations per model.
const ITERS: usize = 100;
/// Scratch directory for compiled models.
const TD: &str = "/tmp/prism_attn_crossover";

/// Sequence lengths to sweep.
const SEQUENCES: &[usize] = &[1, 8, 32, 128, 512];

// ── Arena helpers ─────────────────────────────────────────────────────────

fn md(name: &str) -> PathBuf {
    let p = Path::new(TD).join(name);
    let _ = std::fs::create_dir_all(&p);
    p
}

fn ma(d0: u32, d1: u32) -> Arena {
    Arena::new(d0, d1, DataType::Float16).expect("arena")
}

// ── Deterministic data ────────────────────────────────────────────────────

fn rw(r: i64, c: i64) -> Vec<f32> {
    use std::hash::{Hash, Hasher};
    let mut w = Vec::with_capacity((r * c) as usize);
    for i in 0..(r * c) {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        i.hash(&mut h);
        w.push((h.finish() as f32 % 1000.0 - 500.0) / 500.0);
    }
    w
}

fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let mant = (bits >> 13) & 0x3FF;
    (if exp <= 0 {
        sign | (mant >> 1)
    } else if exp >= 31 {
        sign | 0x7C00 | mant
    } else {
        sign | ((exp as u32) << 10) | mant
    }) as u16
}

#[allow(dead_code)]
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits as u32) >> 15) << 31;
    let exp = ((bits >> 10) & 0x1F) as i32 - 15 + 127;
    let mant = (bits & 0x3FF) as u32;
    if exp <= 0 {
        f32::from_bits(sign | mant << 13)
    } else if exp >= 255 {
        f32::from_bits(sign | 0x7F800000 | (mant << 13))
    } else {
        f32::from_bits(sign | ((exp as u32) << 23) | (mant << 13))
    }
}

// ── MIL program builders ──────────────────────────────────────────────────

fn build_mm(m: i64, k: i64, n: i64) -> mil_spec::Program {
    let b = MilBuilder::new("main").input("x", mil_spec::DataType::Float16, &[1, m]);
    let b = b.const_f16("w", &rw(k, n), &[k, n]);
    let wn = b.last_name().unwrap().to_string();
    let b = b.matmul("x", &wn);
    let on = b.last_name().unwrap().to_string();
    b.output(&on).build().unwrap()
}

/// Write an .mlpackage and compile it for the ANE (cpuAndNeuralEngine).
/// Returns (compiled_modelc_path, output_name) or None on failure.
fn compile_model(prog: mil_spec::Program, tag: &str, m: i64, n: i64) -> Option<(PathBuf, String)> {
    let on = prog.functions["main"].block_specializations["CoreML9"].outputs[0].clone();
    let meta = ModelMeta {
        model_name: tag.into(),
        function_name: "main".into(),
        short_description: tag.into(),
        version: "1.0".into(),
        author: "attn_lane_crossover".into(),
        output_name: on.clone(),
        inputs: vec![("x".into(), vec![1, m])],
        outputs: vec![(on.clone(), vec![1, n])],
    };
    let dir = md(tag);
    let pkg = match write_mlpackage(prog, &dir, &meta) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("  write_mlpackage {}: {}", tag, e);
            return None;
        }
    };
    let od = dir.join("compiled");
    let _ = std::fs::create_dir_all(&od);
    match compile_mlpackage(&pkg, &od, tag, "cpuAndNeuralEngine", "macOS26") {
        Ok(receipt) => Some((PathBuf::from(&receipt.compiled_modelc_path), on)),
        Err(e) => {
            eprintln!("  compile_mlpackage {}: {}", tag, e);
            None
        }
    }
}

// ── Q4 packing ────────────────────────────────────────────────────────────

fn pack_q4(data: &[f32], n: usize, k: usize) -> (Vec<u32>, Vec<u16>) {
    let ng = k / GS;
    let mut packed = vec![0u32; n * (k / 8)];
    let mut scales = vec![0u16; n * ng];
    for row in 0..n {
        for g in 0..ng {
            let mut max_abs = 0.0f32;
            for j in 0..GS {
                let a = data[row * k + g * GS + j].abs();
                if a > max_abs {
                    max_abs = a;
                }
            }
            let scale = if max_abs > 0.0 { max_abs / 7.0 } else { 1.0 };
            scales[row * ng + g] = f32_to_f16_bits(scale);
            let base = row * k + g * GS;
            for j in 0..(GS / 8) {
                let mut word = 0u32;
                for nib in 0..8 {
                    let orig = data[base + j * 8 + nib];
                    let q = (orig / scale).round().clamp(-8.0, 7.0) as i32;
                    word |= ((q & 0x0F) as u32) << (nib * 4);
                }
                packed[row * (k / 8) + g * (GS / 8) + j] = word;
            }
        }
    }
    (packed, scales)
}

// ── Q4 Metal kernel source (sizes embedded at CPU side) ────────────────────

fn q4_source(n: usize, k: usize) -> String {
    let ng = k / GS;
    format!(
        r##"#include <metal_stdlib>
using namespace metal;

kernel void q4_gemv(
    device const half*      input   [[buffer(0)]],
    device const uint*      weights [[buffer(1)]],
    device const half*      scales  [[buffer(2)]],
    device half*            output  [[buffer(3)]],
    uint                    row     [[thread_position_in_grid]])
{{
    if (row >= {n}) return;
    float acc_f = 0.0f;
    uint base = row * ({k} / 8);
    for (uint g = 0; g < {ng}; ++g) {{
        float group_acc = 0.0f;
        half scale = scales[row * {ng} + g];
        for (uint j = 0; j < {gs} / 8; ++j) {{
            uint packed = weights[base + g * ({gs} / 8) + j];
            uchar4 bytes = as_type<uchar4>(packed);
            uint off = g * {gs} + j * 8;
#define NIB(n, i) {{ uint x = (n >> (i*4)) & 0xFu; group_acc += float(int(x ^ 8u) - 8) * float(scale) * float(input[off + i]); }}
            NIB(bytes[0],0) NIB(bytes[0],1) NIB(bytes[1],0) NIB(bytes[1],1)
            NIB(bytes[2],0) NIB(bytes[2],1) NIB(bytes[3],0) NIB(bytes[3],1)
#undef NIB
        }}
        acc_f += group_acc;
    }}
    output[row] = half(acc_f);
}}
"##,
        n = n,
        k = k,
        ng = ng,
        gs = GS
    )
}

// ── ANE benchmark ─────────────────────────────────────────────────────────

/// Load a compiled Core ML model (ANE) and benchmark median predict latency.
fn bench_ane_latency(path: &Path, output_name: &str, ia: &Arena, oa: &Arena) -> Option<f64> {
    let m = CoreMlModel::load_with_compute_units(
        path.to_str()?,
        CoreMlComputeUnits::CpuAndNeuralEngine,
    )
    .ok()?;
    for _ in 0..WARMUP {
        m.predict("x", &ia.info, output_name, &oa.info).ok()?;
    }
    let mut samples = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t0 = Instant::now();
        m.predict("x", &ia.info, output_name, &oa.info).ok()?;
        samples.push(t0.elapsed().as_nanos() as f64);
    }
    samples.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    Some(samples[ITERS / 2]) // median
}

// ── Accelerate (CPU) benchmark ──────────────────────────────────────────────

/// Load a compiled Core ML model (CPU-only, uses Accelerate/vDSP internally)
/// and benchmark median predict latency.
fn bench_acc_latency(path: &Path, output_name: &str, ia: &Arena, oa: &Arena) -> Option<f64> {
    let m =
        CoreMlModel::load_with_compute_units(path.to_str()?, CoreMlComputeUnits::CpuOnly).ok()?;
    for _ in 0..WARMUP {
        m.predict("x", &ia.info, output_name, &oa.info).ok()?;
    }
    let mut samples = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t0 = Instant::now();
        m.predict("x", &ia.info, output_name, &oa.info).ok()?;
        samples.push(t0.elapsed().as_nanos() as f64);
    }
    samples.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    Some(samples[ITERS / 2])
}

// ── GPU Q4 benchmark ──────────────────────────────────────────────────────

#[cfg(feature = "metal-dispatch")]
fn bench_gpu_q4_latency(
    packed: &[u32],
    scales: &[u16],
    k: usize,
    n: usize,
    tag: &str,
) -> Option<f64> {
    use tribunus_compute_core::compute_image::metal_pipeline::compile_metal_source;

    let src = q4_source(n, k);
    let out = compile_metal_source(tag, &src)?;
    let device = metal::Device::system_default()?;
    let lib = device.new_library_with_data(&out.metallib_bytes).ok()?;
    let func = lib.get_function("q4_gemv", None).ok()?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&func)
        .ok()?;
    let queue = device.new_command_queue();

    let sb = metal::MTLResourceOptions::StorageModeShared;
    let buf_in = device.new_buffer((k * 2) as u64, sb);
    let buf_w = device.new_buffer((packed.len() * 4) as u64, sb);
    let buf_s = device.new_buffer((scales.len() * 2) as u64, sb);
    let buf_out = device.new_buffer((n * 2) as u64, sb);

    // Copy Q4 weight data into Metal buffers (packed weights + scales).
    unsafe {
        std::ptr::copy_nonoverlapping(packed.as_ptr(), buf_w.contents() as *mut u32, packed.len());
        std::ptr::copy_nonoverlapping(scales.as_ptr(), buf_s.contents() as *mut u16, scales.len());
    }

    let tg = metal::MTLSize {
        width: 64,
        height: 1,
        depth: 1,
    };
    let gg = metal::MTLSize {
        width: ((n + 63) / 64) as u64,
        height: 1,
        depth: 1,
    };

    for _ in 0..WARMUP {
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&buf_in), 0);
        enc.set_buffer(1, Some(&buf_w), 0);
        enc.set_buffer(2, Some(&buf_s), 0);
        enc.set_buffer(3, Some(&buf_out), 0);
        enc.dispatch_thread_groups(gg, tg);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    let mut samples = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t0 = Instant::now();
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&buf_in), 0);
        enc.set_buffer(1, Some(&buf_w), 0);
        enc.set_buffer(2, Some(&buf_s), 0);
        enc.set_buffer(3, Some(&buf_out), 0);
        enc.dispatch_thread_groups(gg, tg);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        samples.push(t0.elapsed().as_nanos() as f64);
    }
    samples.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    Some(samples[ITERS / 2])
}

#[cfg(not(feature = "metal-dispatch"))]
fn bench_gpu_q4_latency(
    _packed: &[u32],
    _scales: &[u16],
    _k: usize,
    _n: usize,
    _tag: &str,
) -> Option<f64> {
    None
}

// ── Test entry point ──────────────────────────────────────────────────────

#[test]
fn test_attn_crossover() {
    println!();
    println!("=== ATTENTION LANE CROSSOVER: ACCELERATE vs GPU vs ANE ===");
    println!("Config: H={}, NH={}, HD={}, QKV_DIM={}", H, NH, HD, QKV_DIM);
    println!("Hardware: Apple Silicon M1, macOS 26.5");
    println!();
    println!("Building shared models (QKV projection, output projection)...");

    // ── Shared models (independent of S) ────────────────────────────────
    // QKV projection: x[1, H] @ W_qkv[H, 3H] → [1, 3H]
    let qkv_prog = build_mm(H as i64, H as i64, QKV_DIM as i64);
    let (qkv_path, qkv_out) = match compile_model(qkv_prog, "qkv_proj", H as i64, QKV_DIM as i64) {
        Some(p) => p,
        None => {
            eprintln!("FATAL: QKV model compilation failed — aborting.");
            return;
        }
    };
    println!("  QKV model: compiled ✓");

    // Output projection: attn[1, H] @ W_out[H, H] → [1, H]
    let out_prog = build_mm(H as i64, H as i64, H as i64);
    let (out_path, out_out) = match compile_model(out_prog, "out_proj", H as i64, H as i64) {
        Some(p) => p,
        None => {
            eprintln!("FATAL: Output projection model compilation failed — aborting.");
            return;
        }
    };
    println!("  Output proj model: compiled ✓");

    println!("  Accelerate/CPU: uses Core ML CpuOnly (Accelerate/vDSP backend)");

    // ── Shared Q4 weight data (independent of S) ────────────────────────
    let w_qkv = rw(H as i64, QKV_DIM as i64);
    let (qkv_packed, qkv_scales) = pack_q4(&w_qkv, QKV_DIM, H);

    let w_out = rw(H as i64, H as i64);
    let (out_packed, out_scales) = pack_q4(&w_out, H, H);
    // ── Sweep S ─────────────────────────────────────────────────────────
    println!();
    println!(
        "{:<6} {:>58} {:>58} {:>58}",
        "S", "Accelerate", "GPU", "ANE"
    );
    println!("{:-<6} {:->58} {:->58} {:->58}", "", "", "", "");

    for &s in SEQUENCES {
        // --- ANE: compile score model for this S ---
        // Score: q[1, HD] @ K[HD, S] → [1, S]
        let score_tag = format!("score_s{}", s);
        let score_prog = build_mm(HD as i64, HD as i64, s as i64);
        let (score_model, ane_score_ret) =
            match compile_model(score_prog, &score_tag, HD as i64, s as i64) {
                Some(p) => p,
                None => {
                    eprintln!("  Score model S={} compile FAILED — skipping this S", s);
                    println!("{:<6}  (compile failed, skipped)", format!("S={}", s));
                    println!();
                    continue;
                }
            };
        let ane_score_on = ane_score_ret;

        // --- ANE: benchmark ---
        // We need dedicated arenas for each model because predict writes
        // into the output arena.
        let qkv_iane = ma(1, H as u32);
        let qkv_oane = ma(1, QKV_DIM as u32);
        let out_iane = ma(1, H as u32);
        let out_oane = ma(1, H as u32);
        let score_iane = ma(1, HD as u32);
        let score_oane = ma(1, s as u32);

        let ane_qkv_ns = bench_ane_latency(&qkv_path, &qkv_out, &qkv_iane, &qkv_oane);
        let ane_out_ns = bench_ane_latency(&out_path, &out_out, &out_iane, &out_oane);
        let ane_score_ns = bench_ane_latency(&score_model, &ane_score_on, &score_iane, &score_oane);

        // --- GPU: benchmark Q4 ---
        let gpu_qkv_ns = bench_gpu_q4_latency(
            &qkv_packed,
            &qkv_scales,
            H,
            QKV_DIM,
            &format!("gpu_qkv_s{}", s),
        );
        let gpu_out_ns =
            bench_gpu_q4_latency(&out_packed, &out_scales, H, H, &format!("gpu_out_s{}", s));

        // GPU score matmul only for large S (small S too many tiny dispatches)
        let gpu_score_ns = if s >= 128 {
            let w_k = rw(HD as i64, s as i64);
            let (k_packed, k_scales) = pack_q4(&w_k, s, HD);
            bench_gpu_q4_latency(&k_packed, &k_scales, HD, s, &format!("gpu_score_s{}", s))
        } else {
            None
        };

        // --- Compute totals ---
        // ANE total = QKV + Out + NH × Score (one predict per head)
        let ane_total_ns = ane_qkv_ns.unwrap_or(0.0)
            + ane_out_ns.unwrap_or(0.0)
            + NH as f64 * ane_score_ns.unwrap_or(0.0);

        // --- Accelerate: benchmark ---
        let acc_qkv_ns = bench_acc_latency(&qkv_path, &qkv_out, &qkv_iane, &qkv_oane);
        let acc_out_ns = bench_acc_latency(&out_path, &out_out, &out_iane, &out_oane);
        let acc_score_ns = bench_acc_latency(&score_model, &ane_score_on, &score_iane, &score_oane);
        let acc_total_ns = acc_qkv_ns.unwrap_or(0.0)
            + acc_out_ns.unwrap_or(0.0)
            + NH as f64 * acc_score_ns.unwrap_or(0.0);

        // GPU total = QKV + Out
        let mut gpu_total_ns = gpu_qkv_ns.unwrap_or(0.0) + gpu_out_ns.unwrap_or(0.0);
        if s >= 128 {
            // At large S, score matmul is significant: NH × dispatches
            gpu_total_ns += NH as f64 * gpu_score_ns.unwrap_or(0.0);
        }

        // --- Format ---
        let fmt_ane = |ns: f64| format!("{:>7.1}us", ns / 1000.0);
        let fmt_acc = |ns: f64| format!("{:>7.1}us", ns / 1000.0);
        let fmt_gpu = |ns: Option<f64>| {
            if let Some(v) = ns {
                format!("{:>7.1}us", v / 1000.0)
            } else {
                "  FAILED".to_string()
            }
        };

        let winner = {
            let min_val = ane_total_ns.min(gpu_total_ns).min(acc_total_ns);
            if ane_total_ns == min_val {
                "ANE"
            } else if gpu_total_ns == min_val {
                "GPU"
            } else {
                "Acc"
            }
        };

        let _crossover = if s == 1 || s == 8 {
            // At small S, expect GPU to win (bandwidth-bound)
            "  (GPU expected winner at small S)"
        } else if s == 128 || s == 512 {
            // At large S, expect ANE to win (compute-bound)
            "  (ANE expected winner at large S)"
        } else {
            // At S=32, likely the crossover region
            "  *** CROSSOVER region ***"
        };

        println!(
            "S={:<3}:  Acc QKV={} score={} out={} tot={} | GPU QKV={} score={} out={} tot={} | ANE QKV={} score={} out={} tot={} | win={}  {}",
            s,
            fmt_acc(acc_qkv_ns.unwrap_or(0.0)),
            fmt_acc(NH as f64 * acc_score_ns.unwrap_or(0.0)),
            fmt_acc(acc_out_ns.unwrap_or(0.0)),
            fmt_acc(acc_total_ns),
            fmt_gpu(gpu_qkv_ns),
            if s >= 128 {
                fmt_gpu(gpu_score_ns.map(|v| NH as f64 * v))
            } else {
                "  N/A".to_string()
            },
            fmt_gpu(gpu_out_ns),
            fmt_ane(gpu_total_ns),  // GPU total formatted as ANE for consistency — same unit
            fmt_ane(ane_qkv_ns.unwrap_or(0.0)),
            fmt_ane(NH as f64 * ane_score_ns.unwrap_or(0.0)),
            fmt_ane(ane_out_ns.unwrap_or(0.0)),
            fmt_ane(ane_total_ns),
            winner,
            if winner == "Acc" { "Accelerate wins" } else if s <= 8 { "GPU wins (small S)" } else { "ANE wins (large S)" },
        );
    }

    println!();
    println!("=== Notes ===");
    println!("- Accelerate path: Core ML CpuOnly (uses Espresso/Accelerate/vDSP on CPU)");
    println!("- ANE score time = NH × per-head predict (serialized)");
    #[cfg(feature = "metal-dispatch")]
    println!("- GPU path uses Q4 GEMV kernel (symmetric quantization, GS=32)");
    #[cfg(not(feature = "metal-dispatch"))]
    println!("- GPU path not available (metal-dispatch feature disabled)");
    println!("- GPU score matmul measured only for S >= 128 (too small otherwise)");
    println!(
        "- Latency measured as median over {} iterations after {} warmup",
        ITERS, WARMUP
    );
}
