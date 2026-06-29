//! Asymmetric multi-tenant co-processing: ANE prefill (Agent A) does not
//! interfere with GPU decode (Agent B).
//!
//! Agent A: ANE batch prefill — x[N, 2048] @ W[2048, 4096] → [N, 4096]
//! Agent B: GPU single-token Q4 decode — continuous loop
//!
//! Because Agent A and Agent B have independent memory (different IOSurface
//! regions for ANE vs Metal buffers for GPU), ANE compute should not affect
//! GPU decode latency.
//!
//! Run: cargo test --test asymmetric_coprocessing --features prism-backend -- --nocapture

#![cfg(all(target_os = "macos", feature = "prism-backend"))]

use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use coreml_proto::proto::mil_spec;
use parking_lot::Mutex;
use tribunus_compute_core::arena::Arena;
use tribunus_compute_core::arena::DataType;
use tribunus_compute_core::compute_image::metal_pipeline::compile_metal_source;
use tribunus_compute_core::coreml_bridge::{CoreMlComputeUnits, CoreMlModel};
use tribunus_compute_core::coreml_pipeline::compile_mlpackage;
use tribunus_compute_core::mil_builder::MilBuilder;
use tribunus_compute_core::mlpackage::{write_mlpackage, ModelMeta};

// ── Dimensions ─────────────────────────────────────────────────────────────

const BATCH: i64 = 256; // Agent A: 256 tokens per prefill
const H: i64 = 2048; // hidden dimension
const FFN: i64 = 4096; // FFN dimension (Agent A output, Agent B input)
const GS: usize = 128; // Q4 block symmetric group size
const GPU_ITERS: usize = 200;
const NUM_GPU_DECODE_SAMPLES: usize = 10;

const MODEL_DIR: &str = "/tmp/asymmetric_coprocessing_models";

// ── FP16 conversion helpers ───────────────────────────────────────────────

fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let mant = ((bits >> 13) & 0x3FF) as u16;
    if exp <= 0 {
        sign | (mant >> 1)
    } else if exp >= 31 {
        sign | 0x7C00 | mant
    } else {
        sign | ((exp as u16) << 10) | mant
    }
}

fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) >> 15) << 31;
    let exp = ((h >> 10) & 0x1Fu16) as i32 - 15 + 127;
    let mant = (h & 0x3FF) as u32;
    if exp <= 0 {
        f32::from_bits(sign | mant << 13)
    } else if exp >= 255 {
        f32::from_bits(sign | 0x7F800000 | (mant << 13))
    } else {
        f32::from_bits(sign | ((exp as u32) << 23) | (mant << 13))
    }
}

// ── Deterministic data generation ─────────────────────────────────────────

fn seeded_f32(seed: u64) -> f32 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    seed.hash(&mut h);
    (h.finish() as f32 % 1000.0 - 500.0) / 500.0
}

// ── FP32 reference matmul ─────────────────────────────────────────────────
// Weight is [in_dim, out_dim] row-major: W[i][j] = flat[i * out_dim + j]
// Output[j] = sum_i input[i] * W[i][j]

fn ref_matmul(input: &[f32], weight: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; out_dim];
    for j in 0..out_dim {
        let mut sum = 0.0f32;
        for i in 0..in_dim {
            sum += input[i] * weight[i * out_dim + j];
        }
        out[j] = sum;
    }
    out
}

// ── Q4 block-symmetric packing ────────────────────────────────────────────

fn pack_q4_block_sym(data: &[f32], n: usize, k: usize, gs: usize) -> (Vec<u32>, Vec<u16>) {
    let ng = k / gs;
    let mut packed = vec![0u32; n * (k / 8)];
    let mut scales = vec![0u16; n * ng];

    for row in 0..n {
        for g in 0..ng {
            let group_start = row * k + g * gs;
            let group = &data[group_start..group_start + gs];

            let mut max_abs = 0.0f32;
            for &v in group {
                let a = v.abs();
                if a > max_abs {
                    max_abs = a;
                }
            }

            let scale = if max_abs > 0.0 {
                max_abs / 7.0f32
            } else {
                1.0f32
            };
            scales[row * ng + g] = f32_to_f16_bits(scale);

            for j in 0..(gs / 8) {
                let mut word = 0u32;
                for nib in 0..8 {
                    let idx = group_start + j * 8 + nib;
                    let orig = data[idx];
                    let q = (orig / scale).round().clamp(-8.0, 7.0) as i32;
                    let uq = (q & 0x0F) as u32;
                    word |= uq << (nib * 4);
                }
                packed[row * (k / 8) + g * (gs / 8) + j] = word;
            }
        }
    }
    (packed, scales)
}

// ── Q4 Metal kernel source (q4_gemv, same as q4_block_sym_bench) ──────────

const Q4_KERNEL_SRC: &str = r##"#include <metal_stdlib>
using namespace metal;

kernel void q4_gemv(
    device const half*      input   [[buffer(0)]],
    device const uint*      weights [[buffer(1)]],
    device const half*      scales  [[buffer(2)]],
    device half*            output  [[buffer(3)]],
    constant uint&          K       [[buffer(4)]],
    constant uint&          N       [[buffer(5)]],
    constant uint&          gs      [[buffer(6)]],
    constant uint&          ng      [[buffer(7)]],
    uint                    row     [[thread_position_in_grid]])
{
    if (row >= N) return;

    float acc_f = 0.0f;
    uint base = row * (K / 8);

    for (uint g = 0; g < ng; ++g) {
        float group_acc = 0.0f;
        half scale = scales[row * ng + g];

        for (uint j = 0; j < gs / 8; ++j) {
            uint packed = weights[base + g * (gs / 8) + j];
            uchar4 bytes = as_type<uchar4>(packed);
            uint off = g * gs + j * 8;

            { uint n = bytes[0] & 0x0Fu; float v = float(int(n ^ 8u) - 8) * float(scale) * float(input[off + 0]); group_acc += v; }
            { uint n = (bytes[0] >> 4) & 0x0Fu; float v = float(int(n ^ 8u) - 8) * float(scale) * float(input[off + 1]); group_acc += v; }
            { uint n = bytes[1] & 0x0Fu; float v = float(int(n ^ 8u) - 8) * float(scale) * float(input[off + 2]); group_acc += v; }
            { uint n = (bytes[1] >> 4) & 0x0Fu; float v = float(int(n ^ 8u) - 8) * float(scale) * float(input[off + 3]); group_acc += v; }
            { uint n = bytes[2] & 0x0Fu; float v = float(int(n ^ 8u) - 8) * float(scale) * float(input[off + 4]); group_acc += v; }
            { uint n = (bytes[2] >> 4) & 0x0Fu; float v = float(int(n ^ 8u) - 8) * float(scale) * float(input[off + 5]); group_acc += v; }
            { uint n = bytes[3] & 0x0Fu; float v = float(int(n ^ 8u) - 8) * float(scale) * float(input[off + 6]); group_acc += v; }
            { uint n = (bytes[3] >> 4) & 0x0Fu; float v = float(int(n ^ 8u) - 8) * float(scale) * float(input[off + 7]); group_acc += v; }
        }
        acc_f += group_acc;
    }

    output[row] = half(acc_f);
}
"##;

// ── Metal kernel compilation ─────────────────────────────────────────────

fn compile_metal(name: &str, source: &str) -> Vec<u8> {
    let output = compile_metal_source(name, source).expect("Metal compile must succeed");
    output.metallib_bytes
}

// ── ANE model building ──────────────────────────────────────────────────

/// Build a simple matmul model: x[1, H] @ weight[H, FFN] → [1, FFN].
/// Weight is constructed from deterministic data.
fn build_ane_model(model_dir: &Path, h: i64, ffn: i64) -> Result<(PathBuf, String), String> {
    let weight_size = (h * ffn) as usize;
    let weight_f32: Vec<f32> = (0..weight_size).map(|i| seeded_f32(i as u64)).collect();

    let b = MilBuilder::new("main").set_opset("CoreML9").input(
        "x",
        mil_spec::DataType::Float16,
        &[1, h],
    );

    let b = b.const_f16("w", &weight_f32, &[h, ffn]);
    let w_ssa = b.last_name().unwrap_or("w_0").to_string();
    let b = b.matmul("x", &w_ssa);
    let out_name = b.last_name().unwrap_or("matmul_0").to_string();
    let b = b.output(&out_name);

    let prog = b
        .build()
        .map_err(|e| format!("MilBuilder build: {:?}", e))?;

    let meta = ModelMeta {
        model_name: "ane_prefill".into(),
        function_name: "main".into(),
        short_description: "ANE prefill matmul: x[1,H] @ W[H,FFN]".into(),
        version: "1.0.0".into(),
        author: "asymmetric-coprocessing-test".into(),
        output_name: out_name.clone(),
        inputs: vec![("x".into(), vec![1, h])],
        outputs: vec![(out_name.clone(), vec![1, ffn])],
    };

    let mlpackage_path =
        write_mlpackage(prog, model_dir, &meta).map_err(|e| format!("write_mlpackage: {}", e))?;

    // Compile
    let output_dir = model_dir.join("compiled");
    std::fs::create_dir_all(&output_dir).map_err(|e| format!("mkdir compiled: {}", e))?;
    let receipt = compile_mlpackage(
        &mlpackage_path,
        &output_dir,
        "ane_prefill",
        "cpuAndNeuralEngine",
        "macOS26",
    )
    .map_err(|e| format!("compile_mlpackage: {}", e))?;

    let compiled = PathBuf::from(&receipt.compiled_modelc_path);
    if !compiled.exists() {
        let alt = output_dir.join("ane_prefill.mlmodelc");
        if alt.exists() {
            return Ok((alt, out_name));
        }
        return Err(format!(
            "compiled modelc not found at: {}",
            compiled.display()
        ));
    }
    Ok((compiled, out_name))
}

// ── GPU Q4 decode benchmark ──────────────────────────────────────────────

/// Run N iterations of the GPU Q4 GEMV kernel, return per-invocation latency in ns.
fn bench_gpu_q4(
    pl: &metal::ComputePipelineStateRef,
    input_buf: &metal::BufferRef,
    weight_buf: &metal::BufferRef,
    scale_buf: &metal::BufferRef,
    output_buf: &metal::BufferRef,
    const_bufs: &[&metal::BufferRef],
    wg: metal::MTLSize,
    gg: metal::MTLSize,
    iters: usize,
) -> f64 {
    let dev = metal::Device::system_default().unwrap();
    let q = dev.new_command_queue();

    // Warmup
    for _ in 0..5 {
        let cb = q.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pl);
        enc.set_buffer(0, Some(input_buf), 0);
        enc.set_buffer(1, Some(weight_buf), 0);
        enc.set_buffer(2, Some(scale_buf), 0);
        enc.set_buffer(3, Some(output_buf), 0);
        for (i, &eb) in const_bufs.iter().enumerate() {
            enc.set_buffer((4 + i) as u64, Some(eb), 0);
        }
        enc.dispatch_thread_groups(gg, wg);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    let t0 = Instant::now();
    for _ in 0..iters {
        let cb = q.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pl);
        enc.set_buffer(0, Some(input_buf), 0);
        enc.set_buffer(1, Some(weight_buf), 0);
        enc.set_buffer(2, Some(scale_buf), 0);
        enc.set_buffer(3, Some(output_buf), 0);
        for (i, &eb) in const_bufs.iter().enumerate() {
            enc.set_buffer((4 + i) as u64, Some(eb), 0);
        }
        enc.dispatch_thread_groups(gg, wg);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }
    t0.elapsed().as_nanos() as f64 / iters as f64
}

// ── Main test ─────────────────────────────────────────────────────────────

#[test]
fn test_asymmetric_coprocessing() {
    println!("\n=== ASYMMETRIC CO-PROCESSING: ANE vs GPU ===");
    println!("Agent A: ANE prefill x[256, 2048] @ W[2048, 4096] (B=256 tokens)");
    println!(
        "Agent B: GPU Q4 GEMV decode (1 token, 200 iter/sample, {} samples)",
        NUM_GPU_DECODE_SAMPLES
    );
    println!();

    // ── 1. Build and compile ANE model ─────────────────────────────────
    let model_dir = Path::new(MODEL_DIR);
    let _ = std::fs::remove_dir_all(model_dir);
    std::fs::create_dir_all(model_dir).expect("[Setup] model dir");

    println!("[Build] Compiling ANE matmul model...");
    let build_t0 = Instant::now();
    let (modelc_path, out_name) =
        build_ane_model(model_dir, H, FFN).expect("[Build] ANE model build + compile must succeed");
    let build_ms = build_t0.elapsed().as_secs_f64() * 1000.0;
    println!(
        "[Build] Model compiled at: {}  ({:.0} ms)",
        modelc_path.display(),
        build_ms
    );

    // ── 2. Load ANE model ──────────────────────────────────────────────
    println!("[ANE] Loading model...");
    let model = CoreMlModel::load_with_compute_units(
        &modelc_path.to_string_lossy(),
        CoreMlComputeUnits::CpuAndNeuralEngine,
    )
    .expect("[ANE] Model load must succeed");
    let ane_model = Arc::new(model);

    // ── 3. Allocate Agent A arenas (ANE input/output) ─────────────────
    let agent_a_input =
        Arena::new(1, H as u32, mlx_rs::Dtype::Float16).expect("[Arena] Agent A input arena");
    let agent_a_output =
        Arena::new(1, FFN as u32, mlx_rs::Dtype::Float16).expect("[Arena] Agent A output arena");

    // Fill Agent A input with deterministic data
    unsafe {
        let ptr = agent_a_input.base_ptr() as *mut u16;
        for i in 0..H as usize {
            ptr.add(i)
                .write(f32_to_f16_bits(seeded_f32(i as u64 ^ 0xABCD)));
        }
    }

    // ── 4. Allocate Agent B GPU decode buffers (Q4) ──────────────────
    let gpu_n = FFN as usize; // output neurons
    let gpu_k = H as usize; // input dimension
    let gpu_ng = gpu_k / GS; // groups per row

    // Generate deterministic decode weights (FP32) — shape [H, FFN]
    let decode_weight_f32: Vec<f32> = (0..gpu_n * gpu_k)
        .map(|i| seeded_f32(i as u64 ^ 0xFF00))
        .collect();

    // Generate deterministic decode input (FP32)
    let decode_input_f32: Vec<f32> = (0..gpu_k).map(|i| seeded_f32(i as u64 ^ 0x00FF)).collect();

    // Pack weights to Q4
    let (q4_packed, q4_scales) = pack_q4_block_sym(&decode_weight_f32, gpu_n, gpu_k, GS);

    // Create Metal buffers
    let dev = metal::Device::system_default().unwrap();
    let sb = metal::MTLResourceOptions::StorageModeShared;

    let decode_input_buf = dev.new_buffer((gpu_k as u64) * 2, sb);
    unsafe {
        let ptr = decode_input_buf.contents() as *mut u16;
        for i in 0..gpu_k {
            ptr.add(i).write(f32_to_f16_bits(decode_input_f32[i]));
        }
    }

    let decode_weight_buf = dev.new_buffer((q4_packed.len() * 4) as u64, sb);
    unsafe {
        std::ptr::copy_nonoverlapping(
            q4_packed.as_ptr() as *const u8,
            decode_weight_buf.contents() as *mut u8,
            q4_packed.len() * 4,
        );
    }

    let decode_scale_buf = dev.new_buffer((q4_scales.len() * 2) as u64, sb);
    unsafe {
        std::ptr::copy_nonoverlapping(
            q4_scales.as_ptr() as *const u8,
            decode_scale_buf.contents() as *mut u8,
            q4_scales.len() * 2,
        );
    }

    let decode_output_buf = dev.new_buffer((gpu_n as u64) * 2, sb);

    // Constants for GPU kernel
    let const_k = dev.new_buffer(4, sb);
    unsafe {
        *(const_k.contents() as *mut u32) = gpu_k as u32;
    }
    let const_n = dev.new_buffer(4, sb);
    unsafe {
        *(const_n.contents() as *mut u32) = gpu_n as u32;
    }
    let const_gs = dev.new_buffer(4, sb);
    unsafe {
        *(const_gs.contents() as *mut u32) = GS as u32;
    }
    let const_ng = dev.new_buffer(4, sb);
    unsafe {
        *(const_ng.contents() as *mut u32) = gpu_ng as u32;
    }

    // ── 5. Compile Q4 Metal kernel ────────────────────────────────────
    println!("[Metal] Compiling Q4 decode kernel...");
    let metallib = compile_metal("q4_gemv", Q4_KERNEL_SRC);
    let q4_lib = dev.new_library_with_data(&metallib).unwrap();
    let q4_fn = q4_lib.get_function("q4_gemv", None).unwrap();
    let q4_pl = dev
        .new_compute_pipeline_state_with_function(&q4_fn)
        .unwrap();

    const TG: u64 = 256;
    let wg = metal::MTLSize {
        width: TG,
        height: 1,
        depth: 1,
    };
    let gg = metal::MTLSize {
        width: ((gpu_n as u64 + TG - 1) / TG),
        height: 1,
        depth: 1,
    };

    let const_bufs: &[&metal::BufferRef] = &[&const_k, &const_n, &const_gs, &const_ng];

    // ── 6. Baseline: GPU decode without ANE ───────────────────────────
    println!("[Baseline] GPU Q4 decode (no ANE)...");
    let mut baseline_samples: Vec<f64> = Vec::with_capacity(NUM_GPU_DECODE_SAMPLES);
    for _ in 0..NUM_GPU_DECODE_SAMPLES {
        let ns = bench_gpu_q4(
            &q4_pl,
            &decode_input_buf,
            &decode_weight_buf,
            &decode_scale_buf,
            &decode_output_buf,
            const_bufs,
            wg,
            gg,
            GPU_ITERS,
        );
        baseline_samples.push(ns);
        print!(".");
    }
    println!();
    baseline_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let baseline_median = baseline_samples[NUM_GPU_DECODE_SAMPLES / 2];

    // ── 7. Measure standalone ANE time ────────────────────────────────
    println!("[ANE] Warmup + standalone time...");
    for _ in 0..3 {
        ane_model
            .predict("x", &agent_a_input.info, &out_name, &agent_a_output.info)
            .expect("[ANE] Predict warmup");
    }

    let ane_t0 = Instant::now();
    for _ in 0..10 {
        ane_model
            .predict("x", &agent_a_input.info, &out_name, &agent_a_output.info)
            .expect("[ANE] Predict");
    }
    let ane_one_call_ns = ane_t0.elapsed().as_nanos() as f64 / 10.0;

    let ane_total_prefill_ns = ane_one_call_ns * BATCH as f64;
    println!(
        "[ANE] One call: {:.1} us   Total prefill ({} tokens): ~{:.0} us ({:.1} ms)",
        ane_one_call_ns / 1000.0,
        BATCH,
        ane_total_prefill_ns / 1000.0,
        ane_total_prefill_ns / 1_000_000.0,
    );

    // ── 8. Concurrent: ANE prefill runs while GPU decode is benchmarked ──
    println!("[Concurrent] Starting ANE prefill thread + GPU decode...");
    use std::sync::atomic::{AtomicBool, Ordering};

    let ane_done = Arc::new(AtomicBool::new(false));
    let gpu_concurrent_results = Arc::new(Mutex::new(Vec::with_capacity(NUM_GPU_DECODE_SAMPLES)));

    let ane_done_clone = ane_done.clone();
    let gpu_results_clone = gpu_concurrent_results.clone();
    let ane_model_clone = ane_model.clone();
    let out_name_c = out_name.clone();
    let input_info = agent_a_input.info;
    let output_info = agent_a_output.info;

    let ane_handle = std::thread::spawn(move || {
        for _ in 0..BATCH {
            ane_model_clone
                .predict("x", &input_info, &out_name_c, &output_info)
                .expect("[ANE] Concurrent predict");
        }
        ane_done_clone.store(true, Ordering::SeqCst);
    });

    for _ in 0..NUM_GPU_DECODE_SAMPLES {
        let ns = bench_gpu_q4(
            &q4_pl,
            &decode_input_buf,
            &decode_weight_buf,
            &decode_scale_buf,
            &decode_output_buf,
            const_bufs,
            wg,
            gg,
            GPU_ITERS,
        );
        gpu_results_clone.lock().push(ns);
        print!(".");
        let _ = ane_done.load(Ordering::SeqCst);
    }
    println!();

    ane_handle.join().expect("[Thread] ANE thread join");

    // ── 9. Compute results ───────────────────────────────────────────
    let mut concurrent_samples = gpu_concurrent_results.lock().clone();
    concurrent_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let concurrent_median = concurrent_samples[NUM_GPU_DECODE_SAMPLES / 2];

    let ratio = concurrent_median / baseline_median;
    let within_10pct = ratio < 1.10;

    // ── 10. Verify ANE output correctness ────────────────────────────
    let h = H as usize;
    let ffn = FFN as usize;
    let ane_input_0: Vec<f32> = (0..h).map(|i| seeded_f32(i as u64 ^ 0xABCD)).collect();
    // Same weight used in MIL model: seeded_f32(i) for i in 0..h*ffn, shape [h, ffn]
    let weight_f32: Vec<f32> = (0..h * ffn).map(|i| seeded_f32(i as u64)).collect();
    let ref_out_0 = ref_matmul(&ane_input_0, &weight_f32, h, ffn);

    // Read ANE output
    let mut ane_out_0 = vec![0.0f32; ffn];
    unsafe {
        let ptr = agent_a_output.base_ptr() as *mut u16;
        for i in 0..ffn {
            ane_out_0[i] = f16_bits_to_f32(ptr.add(i).read());
        }
    }

    let check_n = 16.min(ffn);
    let mut ane_pass = true;
    for i in 0..check_n {
        let err = (ane_out_0[i] - ref_out_0[i]).abs();
        // FP16 precision: allow up to 10% relative error (FP16 has ~3.3 decimal digits)
        let rel_err = if ref_out_0[i].abs() > 1e-3 {
            err / ref_out_0[i].abs()
        } else {
            err
        };
        if rel_err > 0.10 {
            ane_pass = false;
        }
    }

    // ── 11. Print results table ───────────────────────────────────────
    println!();
    println!("=====================================================");
    println!("  ASYMMETRIC CO-PROCESSING RESULTS");
    println!("=====================================================");
    println!();
    println!("GPU decode latency:");
    println!(
        "  Baseline (no ANE):      {:>8.1} us",
        baseline_median / 1000.0
    );
    println!(
        "  Concurrent (with ANE):  {:>8.1} us",
        concurrent_median / 1000.0
    );
    println!("  Ratio:                  {:>8.2}", ratio);
    println!(
        "  Verdict:                {}",
        if within_10pct {
            "NO INTERFERENCE (within 10%)"
        } else {
            "INTERFERENCE DETECTED"
        }
    );
    println!();
    println!(
        "ANE prefill ({} tokens): {:>8.1} us ({:.1} ms)",
        BATCH,
        ane_total_prefill_ns / 1000.0,
        ane_total_prefill_ns / 1_000_000.0
    );
    println!(
        "  Correctness (first {}):  {}",
        check_n,
        if ane_pass { "PASS" } else { "FAIL" }
    );
    if !ane_pass {
        println!(
            "  First 5 ANE:  {:>10.6} {:>10.6} {:>10.6} {:>10.6} {:>10.6}",
            ane_out_0[0], ane_out_0[1], ane_out_0[2], ane_out_0[3], ane_out_0[4]
        );
        println!(
            "  First 5 Ref:  {:>10.6} {:>10.6} {:>10.6} {:>10.6} {:>10.6}",
            ref_out_0[0], ref_out_0[1], ref_out_0[2], ref_out_0[3], ref_out_0[4]
        );
    }
    println!();

    assert!(
        within_10pct,
        "INTERFERENCE DETECTED: ratio={:.2} (threshold 1.10)",
        ratio
    );
    assert!(ane_pass, "ANE output correctness check FAILED");
}
