//! Disaggregated pipeline: ANE→CPU→GPU.
//!
//! Builds an MLP model, runs it on ANE (FP16), packs the output to Q4_BLOCK_SYM_32
//! on CPU, then dequantizes + GEMVs on GPU. Measures per-stage latency.
//!
//! Run: cargo test --test disaggregated_pipeline --features prism-backend -- --nocapture

#![cfg(all(target_os = "macos", feature = "prism-backend"))]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use coreml_proto::proto::mil_spec::{self, argument, dimension, tensor_value, value, value_type};
use tribunus_compute_core::arena::Arena;
use tribunus_compute_core::coreml_bridge::{CoreMlComputeUnits, CoreMlModel};
use tribunus_compute_core::coreml_pipeline::compile_mlpackage;
use tribunus_compute_core::mil_builder::MilBuilder;
use tribunus_compute_core::mlpackage::{write_mlpackage, ModelMeta};

// ── Dimensions ─────────────────────────────────────────────────────────────
const BATCH: i64 = 1;
const IN_DIM: i64 = 2048; // input feature dim
const HID_DIM: i64 = 4096; // gate/down hidden dim
const OUT_DIM: i64 = 2048; // output feature dim
const GS: usize = 32; // Q4 group size
const WARMUP: usize = 10;
const MEASURED: usize = 100;

const MODEL_DIR: &str = "/tmp/disaggregated_pipeline_models";

// ── FP16 conversion helpers ───────────────────────────────────────────────

fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let mant = (bits >> 13) & 0x3FF;
    if exp <= 0 {
        sign | (mant >> 1) as u16
    } else if exp >= 31 {
        sign | 0x7C00 | (mant as u16)
    } else {
        ((sign as u32) | ((exp as u32) << 10) | (mant as u32)) as u16
    }
}

fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) >> 15) << 31;
    let exp = ((h >> 10) & 0x1F) as i32 - 15 + 127;
    let mant = (h & 0x3FF) as u32;
    if exp <= 0 {
        f32::from_bits(sign | (mant << 13))
    } else if exp >= 255 {
        f32::from_bits(sign | 0x7F800000 | (mant << 13))
    } else {
        f32::from_bits(sign | ((exp as u32) << 23) | (mant << 13))
    }
}

// ── Model building ─────────────────────────────────────────────────────────

/// Build, compile, and return (modelc_path, output_name).
fn build_and_compile_mlp() -> Result<(PathBuf, String), String> {
    let model_dir = Path::new(MODEL_DIR);
    let _ = std::fs::create_dir_all(model_dir);

    let modelc_path = model_dir.join("mlp_ane.mlmodelc");
    if modelc_path.exists() {
        // Find the actual modelc directory
        let compiled = find_modelc_dir("mlp_ane", model_dir);
        if let Some(p) = compiled {
            return Ok((p, "matmul_1".to_string()));
        }
    }

    // ── Generate random weights ──
    let gate_elems = (IN_DIM * HID_DIM) as usize;
    let down_elems = (HID_DIM * OUT_DIM) as usize;
    // Use small random weights so MLP output fits in Q4-friendly range (~±30).
    // With 2048×4096 matmuls, random [-1,1] weights produce output ~±30000.
    // Scale ~0.05 → output ~±8 — well within Q4 GS=32 tolerance.
    let weight_scale: f32 = 0.05;
    let mut rng_state: u64 = 42;
    let mut next_f32 = || -> f32 {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        ((rng_state % 1000) as f32 / 500.0 - 1.0) * weight_scale
    };
    let gate_weight: Vec<f32> = (0..gate_elems).map(|_| next_f32()).collect();
    let down_weight: Vec<f32> = (0..down_elems).map(|_| next_f32()).collect();

    // ── Build MIL program ──
    // x[1,2048] @ gate[2048,4096] → silu → mul → @ down[4096,2048] → output[1,2048]
    let b = MilBuilder::new("main");
    let b = b.input("x", mil_spec::DataType::Float16, &[BATCH, IN_DIM]);
    let b = b.const_f16("gate", &gate_weight, &[IN_DIM, HID_DIM]);
    let gate_ssa = b.last_name().unwrap_or("gate_0").to_string();
    let b = b.matmul("x", &gate_ssa);
    let h_ssa = b.last_name().unwrap_or("matmul_0").to_string();

    // Manual silu operation
    let silu_name = "silu_1".to_string();
    let silu_vt = mil_spec::ValueType {
        r#type: Some(value_type::Type::TensorType(mil_spec::TensorType {
            data_type: mil_spec::DataType::Float16 as i32,
            rank: 2,
            dimensions: vec![
                mil_spec::Dimension {
                    dimension: Some(dimension::Dimension::Constant(
                        dimension::ConstantDimension { size: BATCH as u64 },
                    )),
                },
                mil_spec::Dimension {
                    dimension: Some(dimension::Dimension::Constant(
                        dimension::ConstantDimension {
                            size: HID_DIM as u64,
                        },
                    )),
                },
            ],
            attributes: HashMap::new(),
        })),
    };
    let mut silu_inputs = HashMap::new();
    silu_inputs.insert(
        "x".to_string(),
        mil_spec::Argument {
            arguments: vec![argument::Binding {
                binding: Some(argument::binding::Binding::Name(h_ssa.clone())),
            }],
        },
    );
    let name_attr = mil_spec::Value {
        doc_string: String::new(),
        r#type: Some(mil_spec::ValueType {
            r#type: Some(value_type::Type::TensorType(mil_spec::TensorType {
                data_type: mil_spec::DataType::String as i32,
                rank: 0,
                dimensions: vec![],
                attributes: HashMap::new(),
            })),
        }),
        value: Some(value::Value::ImmediateValue(value::ImmediateValue {
            value: Some(value::immediate_value::Value::Tensor(
                mil_spec::TensorValue {
                    value: Some(tensor_value::Value::Strings(
                        tensor_value::RepeatedStrings {
                            values: vec![silu_name.clone()],
                        },
                    )),
                },
            )),
        })),
    };
    let mut silu_attrs = HashMap::new();
    silu_attrs.insert("name".to_string(), name_attr);
    let silu_op = mil_spec::Operation {
        r#type: "silu".to_string(),
        inputs: silu_inputs,
        outputs: vec![mil_spec::NamedValueType {
            name: silu_name.clone(),
            r#type: Some(silu_vt.clone()),
        }],
        blocks: vec![],
        attributes: silu_attrs,
    };
    let b = b.operation(silu_op, Some((&silu_name, silu_vt)));

    // mul silu_out * h → gated
    let b = b.mul(&silu_name, &h_ssa);
    let gated_ssa = b.last_name().unwrap_or("mul_0").to_string();

    // down projection
    let b = b.const_f16("down", &down_weight, &[HID_DIM, OUT_DIM]);
    let down_ssa = b.last_name().unwrap_or("down_0").to_string();
    let b = b.matmul(&gated_ssa, &down_ssa);
    let out_ssa = b.last_name().unwrap_or("matmul_1").to_string();
    let b = b.output(&out_ssa);

    let prog = b
        .build()
        .map_err(|e| format!("MilBuilder::build: {:?}", e))?;

    // ── Write .mlpackage ──
    let meta = ModelMeta {
        model_name: "mlp_ane".into(),
        function_name: "main".into(),
        short_description: "MLP gate-silu-down".into(),
        version: "1.0.0".into(),
        author: "disaggregated-pipeline-test".into(),
        output_name: out_ssa.clone(),
        inputs: vec![("x".into(), vec![BATCH, IN_DIM])],
        outputs: vec![(out_ssa.clone(), vec![BATCH, OUT_DIM])],
        spec_version: 9,
    };
    let mlpackage_dir =
        write_mlpackage(prog, model_dir, &meta).map_err(|e| format!("write_mlpackage: {}", e))?;

    // ── Compile ──
    let output_dir = model_dir.join("compiled");
    std::fs::create_dir_all(&output_dir).map_err(|e| format!("mkdir compiled: {}", e))?;
    let receipt = compile_mlpackage(
        &mlpackage_dir,
        &output_dir,
        "mlp_ane",
        "cpuAndNeuralEngine",
        "macOS26",
    )
    .map_err(|e| format!("compile_mlpackage: {}", e))?;

    let compiled = PathBuf::from(&receipt.compiled_modelc_path);
    if !compiled.exists() {
        return Err(format!(
            "compiled modelc not found at: {}",
            compiled.display()
        ));
    }
    Ok((compiled, out_ssa))
}

fn find_modelc_dir(name: &str, base: &Path) -> Option<PathBuf> {
    let candidates = [
        base.join("compiled").join(format!("{}.modelc", name)),
        base.join(format!("{}.modelc", name)),
        base.join("compiled").join("mlp_ane.mlmodelc"),
    ];
    candidates.into_iter().find(|p| p.exists())
}

// ── Q4 block-symmetric packing (GS=32) ──────────────────────────────────

fn pack_q4_block_sym_gs32(data: &[f32]) -> (Vec<u32>, Vec<u16>) {
    let n = 1; // single row
    let k = data.len();
    let ng = k / GS;
    let mut packed = vec![0u32; n * (k / 8)];
    let mut scales = vec![0u16; n * ng];

    for row in 0..n {
        for g in 0..ng {
            let group_start = row * k + g * GS;
            let group = &data[group_start..group_start + GS];

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

            for j in 0..(GS / 8) {
                let mut word = 0u32;
                for nib in 0..8 {
                    let idx = group_start + j * 8 + nib;
                    let orig = data[idx];
                    let q = (orig / scale).round().clamp(-8.0, 7.0) as i32;
                    let uq = (q & 0x0F) as u32;
                    word |= uq << (nib * 4);
                }
                packed[row * (k / 8) + g * (GS / 8) + j] = word;
            }
        }
    }
    (packed, scales)
}

fn dequant_q4_gs32(packed: &[u32], scales: &[u16], k: usize) -> Vec<f32> {
    let n = 1;
    let ng = k / GS;
    let mut out = vec![0.0f32; n * k];
    for row in 0..n {
        for g in 0..ng {
            let scale = f16_bits_to_f32(scales[row * ng + g]);
            for j in 0..(GS / 8) {
                let word = packed[row * (k / 8) + g * (GS / 8) + j];
                for nib in 0..8 {
                    let nibble = (word >> (nib * 4)) & 0x0F;
                    let signed_val = (nibble ^ 8) as i32 - 8;
                    let idx = row * k + g * GS + j * 8 + nib;
                    out[idx] = (signed_val as f32) * scale;
                }
            }
        }
    }
    out
}

// ── Metal Q4 kernel (GS=32 variant, same as q4_gemv but inlined) ─────────

// Parallel Q4 dequant kernel: one thread per element.
// Reads Q4-packed data, dequantizes to FP16, writes out.
const Q4_DEQUANT_SRC: &str = r##"#include <metal_stdlib>
using namespace metal;

kernel void q4_dequant(
    device const uint*      packed  [[buffer(0)]],
    device const half*      scales  [[buffer(1)]],
    device half*            output  [[buffer(2)]],
    constant uint&          K       [[buffer(3)]],
    constant uint&          gs      [[buffer(4)]],
    constant uint&          ng      [[buffer(5)]],
    uint                    idx     [[thread_position_in_grid]])
{
    if (idx >= K) return;

    uint g = idx / gs;
    uint sub = idx % gs;
    uint nibble_idx = sub % 8;
    uint word_bias = g * (gs / 8) + sub / 8;

    uint packed_word = packed[word_bias];
    uint nibble = (packed_word >> (nibble_idx * 4)) & 0x0Fu;
    float val = float(int(nibble ^ 8u) - 8) * float(scales[g]);
    output[idx] = half(val);
}"##;

// ── Metal helpers ─────────────────────────────────────────────────────────

fn compile(
    name: &str,
    source: &str,
) -> Option<tribunus_compute_core::compute_image::metal_pipeline::MetalPipelineOutput> {
    tribunus_compute_core::compute_image::metal_pipeline::compile_metal_source(name, source)
}

/// Benchmark a single-invocation kernel, write results into output buffer.
/// Returns per-invocation latency in nanoseconds.
fn bench_dequant_kernel(
    pl: &metal::ComputePipelineStateRef,
    packed_buf: &metal::BufferRef,
    scale_buf: &metal::BufferRef,
    output_buf: &metal::BufferRef,
    const_bufs: &[&metal::BufferRef],
    wg: metal::MTLSize,
    gg: metal::MTLSize,
    warmup: usize,
    measured: usize,
) -> (f64, f64) {
    let dev = metal::Device::system_default().unwrap();
    let q = dev.new_command_queue();

    // Warmup
    for _ in 0..warmup {
        let cb = q.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pl);
        enc.set_buffer(0, Some(packed_buf), 0);
        enc.set_buffer(1, Some(scale_buf), 0);
        enc.set_buffer(2, Some(output_buf), 0);
        for (i, &b) in const_bufs.iter().enumerate() {
            enc.set_buffer((3 + i) as u64, Some(b), 0);
        }
        enc.dispatch_thread_groups(gg, wg);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    let t0 = Instant::now();
    for _ in 0..measured {
        let cb = q.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pl);
        enc.set_buffer(0, Some(packed_buf), 0);
        enc.set_buffer(1, Some(scale_buf), 0);
        enc.set_buffer(2, Some(output_buf), 0);
        for (i, &b) in const_bufs.iter().enumerate() {
            enc.set_buffer((3 + i) as u64, Some(b), 0);
        }
        enc.dispatch_thread_groups(gg, wg);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }
    let total_ns = t0.elapsed().as_nanos() as f64;
    (total_ns / measured as f64, total_ns)
}

fn compute_rmse(a: &[f32], b: &[f32]) -> f64 {
    let k = a.len().min(b.len());
    let mut sum_sq = 0.0f64;
    for i in 0..k {
        let err = (a[i] - b[i]).abs() as f64;
        sum_sq += err * err;
    }
    (sum_sq / k as f64).sqrt()
}

fn compute_max_abs_err(a: &[f32], b: &[f32]) -> f64 {
    let k = a.len().min(b.len());
    let mut max_err = 0.0f64;
    for i in 0..k {
        let err = (a[i] - b[i]).abs() as f64;
        if err > max_err {
            max_err = err;
        }
    }
    max_err
}

// ── Main test ─────────────────────────────────────────────────────────────

#[test]
fn test_disaggregated_pipeline() {
    println!("\n=== DISAGGREGATED PIPELINE: ANE → CPU(Q4) → GPU ===");
    println!("MLP: x[1,2048] @ gate[2048,4096] → silu → mul → @ down[4096,2048]");
    println!("Q4 GS=32  |  Warmup={}  Measured={}\n", WARMUP, MEASURED);

    // ── 1. Build and compile MLP model for ANE ──────────────────────
    let (modelc_path, out_name) =
        build_and_compile_mlp().expect("[Build] MLP model build + compile must succeed");
    println!("[Build] Model compiled at: {}", modelc_path.display());

    // ── 2. Load model on ANE and generate random input ─────────────
    let model = CoreMlModel::load_with_compute_units(
        &modelc_path.to_string_lossy(),
        CoreMlComputeUnits::CpuAndNeuralEngine,
    )
    .expect("[ANE] Model load must succeed");

    let input_arena = Arena::new(BATCH as u32, IN_DIM as u32, mlx_rs::Dtype::Float16)
        .expect("[ANE] Input arena allocation");
    let output_arena = Arena::new(BATCH as u32, OUT_DIM as u32, mlx_rs::Dtype::Float16)
        .expect("[ANE] Output arena allocation");

    // Fill input with random FP16 values
    let mut rng_state: u64 = 12345;
    let input_scale: f32 = 1.0;
    let mut next_f32 = || -> f32 {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        ((rng_state % 10000) as f32 / 5000.0 - 1.0) * input_scale
    };
    let input_elems = (BATCH * IN_DIM) as usize;
    unsafe {
        let ptr = input_arena.base_ptr() as *mut u16;
        for i in 0..input_elems {
            ptr.add(i).write(f32_to_f16_bits(next_f32()));
        }
    }

    // ── 3. ANE predict (latency captured) ────────────────────────────
    // Warmup
    for _ in 0..WARMUP {
        model
            .predict("x", &input_arena.info, &out_name, &output_arena.info)
            .expect("[ANE] Predict warmup");
    }

    let ane_t0 = Instant::now();
    for _ in 0..MEASURED {
        model
            .predict("x", &input_arena.info, &out_name, &output_arena.info)
            .expect("[ANE] Predict measured");
    }
    let ane_total_ns = ane_t0.elapsed().as_nanos() as f64;
    let ane_per_ns = ane_total_ns / MEASURED as f64;
    println!(
        "[ANE] Predict:     {:>8.0} ns/iter  ({:.1} us)",
        ane_per_ns,
        ane_per_ns / 1000.0
    );

    // Read ANE output from arena
    let out_elems = (BATCH * OUT_DIM) as usize;
    let mut ane_output_f32 = vec![0.0f32; out_elems];
    unsafe {
        let ptr = output_arena.base_ptr() as *mut u16;
        for i in 0..out_elems {
            ane_output_f32[i] = f16_bits_to_f32(ptr.add(i).read());
        }
    }

    // Verify ANE produced non-zero output
    let ane_sum: f32 = ane_output_f32.iter().sum();
    println!("[ANE] Output: {} elements, sum={:.4}", out_elems, ane_sum);
    // Debug: print first/last few values and range
    let ane_min = ane_output_f32.iter().cloned().fold(f32::INFINITY, f32::min);
    let ane_max = ane_output_f32
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    println!("[ANE] Output range: [{:.4}, {:.4}]", ane_min, ane_max);
    println!(
        "[ANE] First 5: {:.4} {:.4} {:.4} {:.4} {:.4}",
        ane_output_f32[0],
        ane_output_f32[1],
        ane_output_f32[2],
        ane_output_f32[3],
        ane_output_f32[4]
    );
    assert!(
        ane_sum.abs() > 0.0,
        "ANE output must contain non-zero values"
    );

    // ── 4. CPU: pack FP16 ANE output to Q4_BLOCK_SYM_32 ───────────
    let cpu_t0 = Instant::now();
    for _ in 0..MEASURED {
        let (_packed, _scales) = pack_q4_block_sym_gs32(&ane_output_f32);
        std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
    }
    let cpu_total_ns = cpu_t0.elapsed().as_nanos() as f64;
    let cpu_per_ns = cpu_total_ns / MEASURED as f64;
    println!(
        "[CPU] Q4 Pack:    {:>8.0} ns/iter  ({:.1} us)",
        cpu_per_ns,
        cpu_per_ns / 1000.0
    );

    let (q4_packed, q4_scales) = pack_q4_block_sym_gs32(&ane_output_f32);
    let k = OUT_DIM as usize;
    let ng = k / GS;
    println!(
        "[CPU] Q4 data: {} packed u32 rows, {} scale f16 entries",
        q4_packed.len(),
        q4_scales.len()
    );

    // ── 5. GPU: compile Q4 kernel, set up buffers, benchmark ────────
    let q4_metal = compile("q4_dequant", Q4_DEQUANT_SRC).expect("[GPU] Q4 kernel compile");
    let dev = metal::Device::system_default().unwrap();
    let sb = metal::MTLResourceOptions::StorageModeShared;

    // Q4 packed weight buffer
    let packed_buf = dev.new_buffer((q4_packed.len() * 4) as u64, sb);
    unsafe {
        std::ptr::copy_nonoverlapping(
            q4_packed.as_ptr() as *const u8,
            packed_buf.contents() as *mut u8,
            q4_packed.len() * 4,
        );
    }

    // Q4 scale buffer (output of ANE dequantize per group, FP16)
    let scale_buf = dev.new_buffer((q4_scales.len() * 2) as u64, sb);
    unsafe {
        std::ptr::copy_nonoverlapping(
            q4_scales.as_ptr() as *const u8,
            scale_buf.contents() as *mut u8,
            q4_scales.len() * 2,
        );
    }

    // Output buffer: K dequantized FP16 values
    let output_buf = dev.new_buffer((k as u64) * 2, sb);

    // Constant buffers
    let const_k = dev.new_buffer(4, sb);
    unsafe {
        *(const_k.contents() as *mut u32) = k as u32;
    }
    let const_gs = dev.new_buffer(4, sb);
    unsafe {
        *(const_gs.contents() as *mut u32) = GS as u32;
    }
    let const_ng = dev.new_buffer(4, sb);
    unsafe {
        *(const_ng.contents() as *mut u32) = ng as u32;
    }

    // Build pipeline state
    let q4_lib = dev.new_library_with_data(&q4_metal.metallib_bytes).unwrap();
    let q4_fn = q4_lib.get_function("q4_dequant", None).unwrap();
    let q4_pl = dev
        .new_compute_pipeline_state_with_function(&q4_fn)
        .unwrap();

    // Parallel dispatch: one thread per output element
    const TG: u64 = 256;
    let wg = metal::MTLSize {
        width: TG,
        height: 1,
        depth: 1,
    };
    let gg = metal::MTLSize {
        width: ((k as u64) + TG - 1) / TG,
        height: 1,
        depth: 1,
    };

    let (gpu_per_ns, _gpu_total_ns) = bench_dequant_kernel(
        &q4_pl,
        &packed_buf,
        &scale_buf,
        &output_buf,
        &[&const_k, &const_gs, &const_ng],
        wg,
        gg,
        WARMUP,
        MEASURED,
    );
    println!(
        "[GPU] Q4 dequant: {:>8.0} ns/iter  ({:.1} us)",
        gpu_per_ns,
        gpu_per_ns / 1000.0
    );

    // Read GPU output: K dequantized FP16 values
    let mut gpu_dequant = vec![0.0f32; k];
    unsafe {
        let ptr = output_buf.contents() as *mut u16;
        for i in 0..k {
            gpu_dequant[i] = f16_bits_to_f32(ptr.add(i).read());
        }
    }
    let gpu_dequant_sum: f32 = gpu_dequant.iter().sum();

    // ── 5b. CPU dequant for element-wise accuracy reference ────────
    let q4_reconstructed = dequant_q4_gs32(&q4_packed, &q4_scales, k);

    // Debug: first few reconstructed values
    let q4_min = q4_reconstructed
        .iter()
        .cloned()
        .fold(f32::INFINITY, f32::min);
    let q4_max = q4_reconstructed
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    println!("[Q4] Reconstructed range: [{:.4}, {:.4}]", q4_min, q4_max);
    println!(
        "[Q4] First 5: {:.4} {:.4} {:.4} {:.4} {:.4}",
        q4_reconstructed[0],
        q4_reconstructed[1],
        q4_reconstructed[2],
        q4_reconstructed[3],
        q4_reconstructed[4]
    );

    // GPU dequant vs CPU dequant — Metal kernel parity check
    let gpu_rmse = compute_rmse(&gpu_dequant, &q4_reconstructed);
    let gpu_max_err = compute_max_abs_err(&gpu_dequant, &q4_reconstructed);

    // ANE FP16 vs Q4 dequant — pipeline accuracy
    let pipeline_rmse = compute_rmse(&ane_output_f32, &q4_reconstructed);
    let pipeline_max_err = compute_max_abs_err(&ane_output_f32, &q4_reconstructed);

    println!(
        "[Compare] GPU vs CPU dequant:       RMSE={:.8}  max_err={:.8}",
        gpu_rmse, gpu_max_err
    );
    println!(
        "[Compare] ANE -> CPU Q4 -> dequant: RMSE={:.6}  max_err={:.6}",
        pipeline_rmse, pipeline_max_err
    );
    println!(
        "[Compare] ANE sum={:.4}  GPU dequant sum={:.4}  diff={:.4}",
        ane_sum,
        gpu_dequant_sum,
        (ane_sum - gpu_dequant_sum).abs()
    );

    let dequant_t0 = Instant::now();
    for _ in 0..MEASURED {
        let _ = dequant_q4_gs32(&q4_packed, &q4_scales, k);
        std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
    }
    let dequant_total_ns = dequant_t0.elapsed().as_nanos() as f64;
    let dequant_per_ns = dequant_total_ns / MEASURED as f64;

    // ── 6. Pipeline latency breakdown ───────────────────────────────
    let total_per_ns = ane_per_ns + cpu_per_ns + gpu_per_ns;
    let _total_dequant_ns = ane_per_ns + cpu_per_ns + dequant_per_ns;
    println!("\n--- Pipeline Latency Breakdown ---");
    println!(
        "  ANE predict:     {:>8.0} ns  ({:>5.1}%)",
        ane_per_ns,
        ane_per_ns / total_per_ns * 100.0
    );
    println!(
        "  CPU Q4 pack:     {:>8.0} ns  ({:>5.1}%)",
        cpu_per_ns,
        cpu_per_ns / total_per_ns * 100.0
    );
    println!(
        "  GPU Q4 dequant:  {:>8.0} ns  ({:>5.1}%)",
        gpu_per_ns,
        gpu_per_ns / total_per_ns * 100.0
    );
    println!(
        "  Total (pipeline): {:>8.0} ns  ({:.1} us)",
        total_per_ns,
        total_per_ns / 1000.0
    );
    println!(
        "  Total (FP16->Q4->dequant): {:>8.0} ns  ({:.1} us)",
        _total_dequant_ns,
        _total_dequant_ns / 1000.0
    );
    println!("----------------------------------\n");

    // ── 7. Assertions ──────────────────────────────────────────────
    assert!(
        gpu_rmse < 0.01,
        "GPU vs CPU dequant must match; RMSE={}",
        gpu_rmse
    );
    assert!(
        pipeline_rmse < 1.0,
        "RMSE must be within Q4 quantization tolerance (<1.0); got {}",
        pipeline_rmse
    );
    assert!(
        ((ane_sum - gpu_dequant_sum).abs() as f64) < ((k as f64) * 1.0),
        "GPU GEMV sum must approximately match ANE sum"
    );
    println!("[PASS] All assertions passed.");
}
