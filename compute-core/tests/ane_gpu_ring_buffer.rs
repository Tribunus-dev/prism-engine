//! ANE→Metal ring buffer pipeline with ternary palettized weights.
//!
//! Demonstrates the full pipeline:
//!   1. CPU packs ternary weights as 4-bit UINT8 nibbles into an IOSurface arena
//!   2. ANE runs `constexpr_lut_to_dense` + `matmul` using those weights
//!   3. ANE output written to a ring buffer slot (IOSurface-backed)
//!   4. Metal compute kernel reads the ring buffer via shared IOSurface memory
//!   5. MTLSharedEvent synchronizes GPU completion to CPU (nanosecond-scale poll)
//!   6. Latency breakdown printed: ANE time, Metal+sync time, pipeline total
//!
//! Run: cargo test --test ane_gpu_ring_buffer --features prism-backend -- --nocapture
//!
//! Requires: macOS 14.0+, Apple Silicon (M1 tested), xcrun toolchain

#![cfg(all(target_os = "macos", feature = "prism-backend"))]

use std::path::{Path, PathBuf};
use std::time::Instant;

use coreml_proto::proto::mil_spec;
use metal::*;
use mlx_rs::Dtype;
use tribunus_compute_core::arena::Arena;
use tribunus_compute_core::coreml_bridge::{CoreMlComputeUnits, CoreMlModel};
use tribunus_compute_core::coreml_pipeline::compile_mlpackage;
use tribunus_compute_core::mil_builder::MilBuilder;
use tribunus_compute_core::mlpackage::{write_mlpackage, ModelMeta};

// ── Constants ──────────────────────────────────────────────────────────────

const TEST_DIR: &str = "/tmp/prism_ane_gpu_ring_buffer";
const H: i64 = 2048; // hidden dimension
const FFN: i64 = 4096; // FFN intermediate dimension
const CODEBOOK_SIZE: i64 = 16; // entries per output channel (4-bit indices)
const WARMUP: usize = 5;
const SAMPLES: usize = 15;

/// Index values for ternary weights:
///   0 →  0.0
///   1 → +1.0
///   2 → -1.0
///   3 … 15 → 0.0 (unused)
const TERNARY_VALUES: [f32; 16] = [
    0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
];

/// Metal kernel: reads ring buffer contents and copies to an output buffer.
/// This represents any post-ANE processing step (dequant, activation, etc.).
const RING_BUFFER_KERNEL: &str = r##"#include <metal_stdlib>
using namespace metal;

/// Copy ring buffer contents to an output buffer.
/// Simulates post-ANE processing on the GPU.
kernel void process_ring_buffer(
    device const half* ring_buffer [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& element_count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < element_count) {
        output[gid] = ring_buffer[gid];
    }
}
"##;

// ── Helpers ────────────────────────────────────────────────────────────────

fn model_dir(name: &str) -> PathBuf {
    let p = Path::new(TEST_DIR).join(name);
    let _ = std::fs::create_dir_all(&p);
    p
}

/// Generate pseudorandom ternary weights: each weight is -1.0, 0.0, or +1.0.
fn ternary_weights(seed: u64, rows: i64, cols: i64) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let n = (rows * cols) as usize;
    let mut w = Vec::with_capacity(n);
    for i in 0..n as u64 {
        let mut h = DefaultHasher::new();
        (seed + i).hash(&mut h);
        // Ternary: map hash to -1, 0, or 1 with roughly equal probability
        let v = h.finish() % 3;
        match v {
            0 => w.push(0.0),
            1 => w.push(1.0),
            _ => w.push(-1.0),
        }
    }
    w
}

/// Pack ternary weights into 4-bit LUT indices (2 per byte).
///
/// Each weight is mapped to an index 0, 1, or 2 (corresponding to 0.0, +1.0, -1.0).
/// Two indices are packed per byte: upper nibble = weight[2i], lower nibble = weight[2i+1].
/// Returns (codebook, packed_indices).
fn pack_ternary(weights: &[f32], out_dim: usize, in_dim: usize) -> (Vec<f32>, Vec<u8>) {
    // Codebook: one 16-entry row per output channel, each row = [0,1,-1,0,0,…]
    // Stored as f32; MilBuilder::const_f16 accepts f32 and converts internally to FP16.
    let mut codebook = Vec::with_capacity(out_dim * 16);
    for _ in 0..out_dim {
        codebook.extend_from_slice(&TERNARY_VALUES);
    }

    // Pack indices: 2 per byte
    let packed_len = (out_dim * in_dim + 1) / 2; // ceiling division
    let mut indices = vec![0u8; packed_len];
    for row in 0..out_dim {
        for col in 0..in_dim {
            let weight = weights[row * in_dim + col];
            let idx: u8 = if weight == 0.0 {
                0
            } else if weight > 0.0 {
                1
            } else {
                2
            };
            let linear = row * in_dim + col;
            let byte_pos = linear >> 1; // divide by 2
            let is_upper = (linear & 1) == 0;
            if is_upper {
                indices[byte_pos] = (indices[byte_pos] & 0x0F) | (idx << 4);
            } else {
                indices[byte_pos] = (indices[byte_pos] & 0xF0) | idx;
            }
        }
    }

    (codebook, indices)
}

/// Build the MIL program: x[1, H] → constexpr_lut_to_dense → matmul → y[1, FFN].
fn build_mil(h: i64, ffn: i64) -> (mil_spec::Program, String, String) {
    let w = ternary_weights(42, h, ffn);
    let (codebook, packed_idx) = pack_ternary(&w, ffn as usize, h as usize);

    // Indices shape: [ffn, h/2] — each byte packs 2 nibble indices (vector_axis=1, codebook_size=16)
    let indices_shape = &[ffn, h / 2];

    // Codebook shape: [ffn, 1, 16, 1] — 16-entry LUT per output channel
    let codebook_shape = &[ffn, 1, CODEBOOK_SIZE, 1];

    // Dense weight shape
    let dense_shape = &[h, ffn];

    let b = MilBuilder::new("main")
        .input("x", mil_spec::DataType::Float16, &[1, h])
        .const_uint8("w_idx", &packed_idx, indices_shape);
    let idx_name = b.last_name().expect("idx name").to_string();
    let b = b.const_f16("w_lut", &codebook, codebook_shape);
    let lut_name = b.last_name().expect("lut name").to_string();

    let b = b.constexpr_lut_to_dense("w_dense", &idx_name, &lut_name, dense_shape, 1);
    let wd = b.last_name().unwrap().to_string();

    let b = b.matmul("x", &wd);
    let on = b.last_name().unwrap().to_string();

    let b = b.output(&on);
    let prog = b.build().expect("MilBuilder::build");

    (prog, "x".into(), on)
}

/// Compile MIL program to .modelc with macOS26 target.
fn compile(prog: mil_spec::Program, meta: ModelMeta, tag: &str) -> PathBuf {
    let dir = model_dir(tag);
    let pkg = write_mlpackage(prog, &dir, &meta).expect("write_mlpackage");
    let od = dir.join("compiled");
    let _ = std::fs::create_dir_all(&od);
    let r = compile_mlpackage(&pkg, &od, tag, "cpuAndNeuralEngine", "macOS26")
        .expect("compile_mlpackage");
    PathBuf::from(&r.compiled_modelc_path)
}

/// Fill an FP16 arena with a deterministic input pattern.
fn fill_input(arena: &Arena, count: usize) {
    arena.lock().expect("arena lock");
    unsafe {
        let ptr = arena.base_ptr() as *mut u16;
        for i in 0..count {
            // Deterministic FP16 values in [0.0, 1.0)
            let val = ((i as f32).sin() * 0.5 + 0.5).clamp(0.0, 1.0);
            ptr.add(i).write(f32_to_f16_bits(val));
        }
    }
    arena.unlock().expect("arena unlock");
}

/// Convert f32 to FP16 bit representation.
fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let mant = (bits >> 13) & 0x03FF;
    if exp <= 0 {
        sign
    } else if exp >= 31 {
        sign | 0x7C00u16
            | (if exp == 31 && mant != 0 {
                0x0200u16
            } else {
                0u16
            })
    } else {
        sign | ((exp as u16) << 10) | (mant as u16)
    }
}

/// Run one ANE prediction and return the elapsed time in nanoseconds.
fn bench_ane(
    model: &CoreMlModel,
    in_name: &str,
    in_arena: &Arena,
    out_name: &str,
    out_arena: &Arena,
) -> f64 {
    let t0 = Instant::now();
    model
        .predict(in_name, &in_arena.info, out_name, &out_arena.info)
        .expect("predict");
    t0.elapsed().as_nanos() as f64
}

/// Compute median of sorted f64 samples.
fn median(samples: &mut [f64]) -> f64 {
    samples.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

// ── Main test ──────────────────────────────────────────────────────────────

#[test]
fn test_ane_gpu_ring_buffer() {
    println!();
    println!("═══ ANE→Metal Ring Buffer Pipeline ═══");
    println!("  Model:   x[1,{}] @ W[{},{}] → y[1,{}]", H, H, FFN, FFN);
    println!("  Weights: ternary (-1/0/+1) palettized via constexpr_lut_to_dense");
    println!(
        "  Codebook: {} entries per output channel (4-bit indices)",
        CODEBOOK_SIZE
    );
    println!(
        "  Ring buffer: IOSurface-backed FP16 arena, {} FP16 elements",
        FFN
    );
    println!("  Synchronization: MTLSharedEvent (poll)");
    println!();
    println!("  Iterations: {} timed (+ {} warmup)", SAMPLES, WARMUP);
    println!();

    // ── 1. Build and compile the MIL program ───────────────────────────────

    let (prog, in_name, out_name) = build_mil(H, FFN);
    let meta = ModelMeta {
        model_name: "ane_ring_buffer".into(),
        function_name: "main".into(),
        short_description: "ANE→GPU ring buffer with ternary palettized weights".into(),
        version: "1.0".into(),
        author: "prism".into(),
        output_name: out_name.clone(),
        inputs: vec![("x".into(), vec![1, H])],
        outputs: vec![(out_name.clone(), vec![1, FFN])],
        spec_version: 10,
    };
    let modelc_path = compile(prog, meta, "ane_rb");

    // ── 2. Allocate arenas ─────────────────────────────────────────────────

    let in_arena = Arena::new(1, H as u32, Dtype::Float16).expect("input arena");
    let out_arena = Arena::new(1, FFN as u32, Dtype::Float16).expect("output arena (ring buffer)");
    let out_arena2 =
        Arena::new(1, FFN as u32, Dtype::Float16).expect("output arena (ring buffer 2)");

    fill_input(&in_arena, (1 * H) as usize);
    fill_input(&out_arena, (1 * FFN) as usize);
    fill_input(&out_arena2, (1 * FFN) as usize);

    // ── 3. Set up Metal ────────────────────────────────────────────────────

    let dev = Device::system_default().expect("Metal device");
    let q = dev.new_command_queue();

    // Compile Metal kernel
    let ml_out = tribunus_compute_core::compute_image::metal_pipeline::compile_metal_source(
        "process_ring_buffer",
        RING_BUFFER_KERNEL,
    )
    .expect("Metal kernel compilation failed");
    let lib = dev
        .new_library_with_data(&ml_out.metallib_bytes)
        .expect("new_library_with_data");
    let func = lib
        .get_function("process_ring_buffer", None)
        .expect("get_function");
    let pl = dev
        .new_compute_pipeline_state_with_function(&func)
        .expect("new_compute_pipeline_state");

    // Create MTLSharedEvent for GPU→CPU synchronization
    let shared_event = dev.new_shared_event();

    // Create a Metal output buffer for the GPU result
    let metal_out = dev.new_buffer(
        (FFN as u64) * 2, // FP16 = 2 bytes each
        MTLResourceOptions::StorageModeShared,
    );

    // ── 4. Load the Core ML model ──────────────────────────────────────────

    let model = CoreMlModel::load_with_compute_units(
        modelc_path.to_str().unwrap(),
        CoreMlComputeUnits::CpuAndNeuralEngine,
    )
    .expect("load CoreML model");

    // ── 5. Collect ANE-only timings ────────────────────────────────────────

    for _ in 0..WARMUP {
        bench_ane(&model, &in_name, &in_arena, &out_name, &out_arena);
    }

    let mut ane_samples = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let t = bench_ane(&model, &in_name, &in_arena, &out_name, &out_arena);
        ane_samples.push(t);
    }
    let ane_median_ns = median(&mut ane_samples);
    let ane_mean_ns = ane_samples.iter().sum::<f64>() / ane_samples.len() as f64;

    // ── 6. Collect ANE+Metal pipeline timings ──────────────────────────────

    let mut pipeline_samples = Vec::with_capacity(SAMPLES);

    for _ in 0..WARMUP {
        // ANE predict
        bench_ane(&model, &in_name, &in_arena, &out_name, &out_arena);

        // Map ANE output (IOSurface) into a Metal buffer via pointer sharing.
        // On Apple Silicon with unified memory, the IOSurface-backed arena memory
        // is directly accessible by the GPU. We wrap it with no-copy zero-overhead.
        let ring_buf = unsafe {
            dev.new_buffer_with_bytes_no_copy(
                out_arena.base_ptr() as *const std::ffi::c_void,
                (FFN as u64) * 2,
                MTLResourceOptions::StorageModeShared,
                None,
            )
        };

        // Submit Metal work
        shared_event.set_signaled_value(0);
        std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::Release);

        let cb: &CommandBufferRef = q.new_command_buffer();
        {
            let enc: &ComputeCommandEncoderRef = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pl);
            enc.set_buffer(0, Some(&ring_buf), 0);
            enc.set_buffer(1, Some(&metal_out), 0);
            let count: u32 = FFN as u32;
            enc.set_bytes(
                2,
                std::mem::size_of::<u32>() as u64,
                &count as *const u32 as *const std::ffi::c_void,
            );
            enc.dispatch_thread_groups(
                MTLSize {
                    width: 1 + (FFN as u64 / 256),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            enc.end_encoding();
        }
        cb.encode_signal_event(&*shared_event, 1);
        cb.commit();
        cb.wait_until_completed();
    }

    for _ in 0..SAMPLES {
        // ANE predict
        let t0 = Instant::now();
        model
            .predict(&in_name, &in_arena.info, &out_name, &out_arena.info)
            .expect("predict");

        // Wrap the IOSurface arena memory as a Metal buffer.
        let ring_buf = unsafe {
            dev.new_buffer_with_bytes_no_copy(
                out_arena.base_ptr() as *const std::ffi::c_void,
                (FFN as u64) * 2,
                MTLResourceOptions::StorageModeShared,
                None,
            )
        };

        // Reset event
        shared_event.set_signaled_value(0);
        std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::Release);

        // Submit Metal compute work
        let cb: &CommandBufferRef = q.new_command_buffer();
        {
            let enc: &ComputeCommandEncoderRef = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pl);
            enc.set_buffer(0, Some(&ring_buf), 0);
            enc.set_buffer(1, Some(&metal_out), 0);
            let count: u32 = FFN as u32;
            enc.set_bytes(
                2,
                std::mem::size_of::<u32>() as u64,
                &count as *const u32 as *const std::ffi::c_void,
            );
            enc.dispatch_thread_groups(
                MTLSize {
                    width: 1 + (FFN as u64 / 256),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            enc.end_encoding();
        }
        cb.encode_signal_event(&*shared_event, 1);
        cb.commit();

        std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);

        // Poll the shared event — nanosecond-scale GPU→CPU sync
        loop {
            if shared_event.signaled_value() >= 1 {
                std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::Acquire);
                break;
            }
            std::hint::spin_loop();
        }

        let elapsed = t0.elapsed().as_nanos() as f64;
        pipeline_samples.push(elapsed);

        // Ensure GPU is fully done before next iteration
        cb.wait_until_completed();
    }

    let pipeline_median_ns = median(&mut pipeline_samples);
    let pipeline_mean_ns = pipeline_samples.iter().sum::<f64>() / pipeline_samples.len() as f64;
    let overhead_ns = pipeline_median_ns - ane_median_ns;

    // ── 7. Also test with a second ring buffer slot (alternating) ──────────

    let mut alt_pipeline_samples = Vec::with_capacity(SAMPLES);

    for _ in 0..WARMUP {
        model
            .predict(&in_name, &in_arena.info, &out_name, &out_arena2.info)
            .expect("predict alt");
    }

    for i in 0..SAMPLES {
        // Alternate between two ring buffer slots
        let slot = if i % 2 == 0 { &out_arena } else { &out_arena2 };

        let t0 = Instant::now();
        model
            .predict(&in_name, &in_arena.info, &out_name, &slot.info)
            .expect("predict");

        let ring_buf = unsafe {
            dev.new_buffer_with_bytes_no_copy(
                slot.base_ptr() as *const std::ffi::c_void,
                (FFN as u64) * 2,
                MTLResourceOptions::StorageModeShared,
                None,
            )
        };

        shared_event.set_signaled_value(0);
        std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::Release);

        let cb: &CommandBufferRef = q.new_command_buffer();
        {
            let enc: &ComputeCommandEncoderRef = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pl);
            enc.set_buffer(0, Some(&ring_buf), 0);
            enc.set_buffer(1, Some(&metal_out), 0);
            let count: u32 = FFN as u32;
            enc.set_bytes(
                2,
                std::mem::size_of::<u32>() as u64,
                &count as *const u32 as *const std::ffi::c_void,
            );
            enc.dispatch_thread_groups(
                MTLSize {
                    width: 1 + (FFN as u64 / 256),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            enc.end_encoding();
        }
        cb.encode_signal_event(&*shared_event, 1);
        cb.commit();

        std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);

        loop {
            if shared_event.signaled_value() >= 1 {
                std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::Acquire);
                break;
            }
            std::hint::spin_loop();
        }

        let elapsed = t0.elapsed().as_nanos() as f64;
        alt_pipeline_samples.push(elapsed);
        cb.wait_until_completed();
    }

    let alt_median_ns = median(&mut alt_pipeline_samples);
    let alt_mean_ns = alt_pipeline_samples.iter().sum::<f64>() / alt_pipeline_samples.len() as f64;

    // ── 8. Print results ───────────────────────────────────────────────────

    println!("{:─<85}", "");
    println!(
        "  {:<30}  {:>10}  {:>10}  {:>10}",
        "Phase", "Median(ns)", "Mean(ns)", "Median(us)"
    );
    println!("{:─<85}", "");

    let ane_us = ane_median_ns / 1000.0;
    let pipe_us = pipeline_median_ns / 1000.0;
    let over_us = overhead_ns / 1000.0;
    let alt_us = alt_median_ns / 1000.0;

    println!(
        "  {:<30}  {:>10.0}  {:>10.0}  {:>10.2}",
        "ANE predict only", ane_median_ns, ane_mean_ns, ane_us
    );
    println!(
        "  {:<30}  {:>10.0}  {:>10.0}  {:>10.2}",
        "ANE + Metal pipeline", pipeline_median_ns, pipeline_mean_ns, pipe_us
    );
    println!(
        "  {:<30}  {:>10.0}  {:>10}  {:>10.2}",
        "Overhead (Metal+sync)", overhead_ns, "", over_us
    );
    println!(
        "  {:<30}  {:>10.0}  {:>10.0}  {:>10.2}",
        "Pipeline (2-slot alt)", alt_median_ns, alt_mean_ns, alt_us
    );
    println!("{:─<85}", "");
    println!();

    println!("  Ring buffer slots: 2 (alternating)");
    println!(
        "  Metal kernel: process_ring_buffer (copy FFN={} FP16 values)",
        FFN
    );
    println!("  Sync: MTLSharedEvent poll (nanosecond-scale)");
    println!(
        "  Overhead: {:.2} us ({:.1}% of total)",
        over_us,
        (overhead_ns / pipeline_median_ns) * 100.0
    );
    println!();

    // Sanity checks
    assert!(
        ane_median_ns > 0.0,
        "ANE time should be positive (got {:.0} ns)",
        ane_median_ns
    );
    assert!(
        pipeline_median_ns >= ane_median_ns,
        "Pipeline time ({:.0} ns) should be >= ANE time ({:.0} ns)",
        pipeline_median_ns,
        ane_median_ns
    );
    eprintln!(
        "[ane_gpu_ring_buffer] PASS — ANE={:.2}us, pipeline={:.2}us, overhead={:.2}us",
        ane_us, pipe_us, over_us
    );
}
