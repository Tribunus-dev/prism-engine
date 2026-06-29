//! ANE matmul stream parallelism sweep.
//!
//! Tests whether partitioning a single logical matmul into N independent
//! parallel streams (one per ANE compute engine) improves utilization.
//!
//! For each stream count N in [1, 2, 4, 8, 16]:
//!   1. Split W[2048, 4096] column-wise into N chunks: W_i[2048, 4096/N]
//!   2. N parallel matmuls: x[batch, 2048] @ W_i -> [batch, 4096/N]
//!   3. Concat along axis=1 -> [batch, 4096]
//!   4. Total FLOPs = 2 x batch x 2048 x 4096 (identical across all N)
//!   5. Higher GFLOPS at high N means ANE benefits from more parallel streams
//!
//! Run: cargo test --test ane_matmul_streams --features prism-backend -- --nocapture
//! Requires: macOS 14.0+ on Apple Silicon

#![cfg(all(target_os = "macos", feature = "prism-backend"))]

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::time::Instant;

use coreml_proto::proto::mil_spec::{self, argument, dimension, tensor_value, value};
use mlx_rs::Dtype;
use tribunus_compute_core::arena::Arena;
use tribunus_compute_core::arena::DataType;
use tribunus_compute_core::coreml_bridge::{CoreMlComputeUnits, CoreMlModel};
use tribunus_compute_core::coreml_pipeline::compile_mlpackage;
use tribunus_compute_core::mil_builder::MilBuilder;
use tribunus_compute_core::mlpackage::{write_mlpackage, ModelMeta};

// ── Constants ──────────────────────────────────────────────────────────────

const TEST_DIR: &str = "/tmp/prism_ane_matmul_streams";
const H: i64 = 2048;
const FFN: i64 = 4096;
const THEORETICAL_PEAK_GFLOPS: f64 = 11_000.0;
const WARMUP: usize = 5;
const SAMPLES: usize = 20;
const BATCH: u32 = 512;
const STREAMS: &[usize] = &[1, 2, 4, 8, 16];

// ── Helpers (mirrors mil_builder internals) ────────────────────────────────

fn tensor_type(dtype: mil_spec::DataType, shape: &[i64]) -> mil_spec::TensorType {
    let dims: Vec<mil_spec::Dimension> = shape
        .iter()
        .map(|&s| mil_spec::Dimension {
            dimension: Some(dimension::Dimension::Constant(
                dimension::ConstantDimension { size: s as u64 },
            )),
        })
        .collect();
    mil_spec::TensorType {
        data_type: dtype as i32,
        rank: shape.len() as i64,
        dimensions: dims,
        attributes: HashMap::new(),
    }
}

fn value_type_tensor(tt: mil_spec::TensorType) -> mil_spec::ValueType {
    mil_spec::ValueType {
        r#type: Some(mil_spec::value_type::Type::TensorType(tt)),
    }
}

#[allow(dead_code)]
fn named_arg(name: &str) -> mil_spec::Argument {
    mil_spec::Argument {
        arguments: vec![argument::Binding {
            binding: Some(argument::binding::Binding::Name(name.to_string())),
        }],
    }
}

fn bool_attr(val: bool) -> mil_spec::Value {
    let bool_tensor = mil_spec::TensorValue {
        value: Some(tensor_value::Value::Bools(tensor_value::RepeatedBools {
            values: vec![val],
        })),
    };
    mil_spec::Value {
        doc_string: String::new(),
        r#type: Some(mil_spec::ValueType {
            r#type: Some(mil_spec::value_type::Type::TensorType(
                mil_spec::TensorType {
                    data_type: mil_spec::DataType::Bool as i32,
                    rank: 0,
                    dimensions: vec![],
                    attributes: HashMap::new(),
                },
            )),
        }),
        value: Some(value::Value::ImmediateValue(value::ImmediateValue {
            value: Some(value::immediate_value::Value::Tensor(bool_tensor)),
        })),
    }
}

fn int32_attr(val: i32) -> mil_spec::Value {
    let int_tensor = mil_spec::TensorValue {
        value: Some(tensor_value::Value::Ints(tensor_value::RepeatedInts {
            values: vec![val],
        })),
    };
    mil_spec::Value {
        doc_string: String::new(),
        r#type: Some(mil_spec::ValueType {
            r#type: Some(mil_spec::value_type::Type::TensorType(
                mil_spec::TensorType {
                    data_type: mil_spec::DataType::Int32 as i32,
                    rank: 0,
                    dimensions: vec![],
                    attributes: HashMap::new(),
                },
            )),
        }),
        value: Some(value::Value::ImmediateValue(value::ImmediateValue {
            value: Some(value::immediate_value::Value::Tensor(int_tensor)),
        })),
    }
}

#[allow(dead_code)]
fn int_attr(val: i64) -> mil_spec::Value {
    let int_tensor = mil_spec::TensorValue {
        value: Some(tensor_value::Value::LongInts(
            tensor_value::RepeatedLongInts { values: vec![val] },
        )),
    };
    mil_spec::Value {
        doc_string: String::new(),
        r#type: Some(mil_spec::ValueType {
            r#type: Some(mil_spec::value_type::Type::TensorType(
                mil_spec::TensorType {
                    data_type: mil_spec::DataType::Int64 as i32,
                    rank: 0,
                    dimensions: vec![],
                    attributes: HashMap::new(),
                },
            )),
        }),
        value: Some(value::Value::ImmediateValue(value::ImmediateValue {
            value: Some(value::immediate_value::Value::Tensor(int_tensor)),
        })),
    }
}

fn string_attr(val: &str) -> mil_spec::Value {
    let string_tensor = mil_spec::TensorValue {
        value: Some(tensor_value::Value::Strings(
            tensor_value::RepeatedStrings {
                values: vec![val.to_string()],
            },
        )),
    };
    mil_spec::Value {
        doc_string: String::new(),
        r#type: Some(mil_spec::ValueType {
            r#type: Some(mil_spec::value_type::Type::TensorType(
                mil_spec::TensorType {
                    data_type: mil_spec::DataType::String as i32,
                    rank: 0,
                    dimensions: vec![],
                    attributes: HashMap::new(),
                },
            )),
        }),
        value: Some(value::Value::ImmediateValue(value::ImmediateValue {
            value: Some(value::immediate_value::Value::Tensor(string_tensor)),
        })),
    }
}

// ── Model building ─────────────────────────────────────────────────────────

fn seeded_weights(seed: u64, rows: i64, cols: i64) -> Vec<f32> {
    let mut w = Vec::with_capacity((rows * cols) as usize);
    for i in 0..((rows * cols) as u64) {
        let mut h = DefaultHasher::new();
        (seed + i).hash(&mut h);
        w.push((h.finish() as f32 % 1000.0 - 500.0) / 500.0);
    }
    w
}

/// Single matmul: x[batch, H] @ W[H, FFN] -> [batch, FFN].
fn build_single_matmul(batch: u32) -> Result<(mil_spec::Program, String), String> {
    let w = seeded_weights(0, H, FFN);
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

/// N-stream matmul: N parallel matmuls x @ W_i, concat along axis=1.
/// K = FFN / N. All matmuls share the same input x.
fn build_streamed_matmul(
    batch: u32,
    num_streams: usize,
) -> Result<(mil_spec::Program, String), String> {
    assert!(
        FFN % num_streams as i64 == 0,
        "FFN must be divisible by num_streams"
    );
    let k_per_stream = FFN / num_streams as i64;

    let mut b = MilBuilder::new("main").input("x", mil_spec::DataType::Float16, &[batch as i64, H]);

    // ── N parallel matmuls: x @ W_i, W_i[H, K] ─────────────────────
    let mut matmul_names: Vec<String> = Vec::with_capacity(num_streams);
    for i in 0..num_streams {
        let w = seeded_weights(i as u64, H, k_per_stream);
        b = b.const_f16(&format!("w_{}", i), &w, &[H, k_per_stream]);
        let wn = b
            .last_name()
            .ok_or_else(|| format!("weight_{}", i))?
            .to_string();
        b = b.matmul("x", &wn);
        let mn = b
            .last_name()
            .ok_or_else(|| format!("matmul_{}", i))?
            .to_string();
        matmul_names.push(mn);
    }

    // ── Concat all matmul outputs along axis=1 ────────────────────
    let concat_name = "concat_out";
    let vt = value_type_tensor(tensor_type(
        mil_spec::DataType::Float16,
        &[batch as i64, FFN],
    ));

    let mut inputs = HashMap::new();

    // concat takes a single "values" input with multiple argument bindings
    let values_args: Vec<argument::Binding> = matmul_names
        .iter()
        .map(|mn| argument::Binding {
            binding: Some(argument::binding::Binding::Name(mn.clone())),
        })
        .collect();
    inputs.insert(
        "values".to_string(),
        mil_spec::Argument {
            arguments: values_args,
        },
    );

    let mut attrs = HashMap::new();
    attrs.insert("axis".to_string(), int32_attr(1));
    attrs.insert("interleave".to_string(), bool_attr(false));
    attrs.insert("name".to_string(), string_attr(concat_name));

    let op = mil_spec::Operation {
        r#type: "concat".to_string(),
        inputs,
        outputs: vec![mil_spec::NamedValueType {
            name: concat_name.to_string(),
            r#type: Some(vt.clone()),
        }],
        blocks: vec![],
        attributes: attrs,
    };

    b = b.operation(op, Some((concat_name, vt)));
    let mil_text = b.to_mil_text();
    eprintln!(
        "[STREAM_{}] MIL text for concat model:\n{}\n",
        num_streams, mil_text
    );
    let b = b.output(concat_name);
    let prog = b.build().map_err(|e| format!("MIL build error: {}", e))?;
    Ok((prog, concat_name.to_string()))
}

// ── Compilation & benchmarking ─────────────────────────────────────────────

fn model_dir(name: &str) -> PathBuf {
    let p = Path::new(TEST_DIR).join(name);
    let _ = std::fs::create_dir_all(&p);
    p
}

fn compile_with_target(
    tag: &str,
    prog: mil_spec::Program,
    meta: ModelMeta,
    target: &str,
) -> Result<PathBuf, String> {
    let dir = model_dir(tag);
    let pkg = write_mlpackage(prog, &dir, &meta).map_err(|e| format!("mlpackage: {}", e))?;
    let od = dir.join("compiled");
    let _ = std::fs::create_dir_all(&od);
    let r = compile_mlpackage(&pkg, &od, tag, "cpuAndNeuralEngine", target)
        .map_err(|e| format!("compile: {}", e))?;
    Ok(PathBuf::from(&r.compiled_modelc_path))
}

#[allow(dead_code)]
fn compile(tag: &str, prog: mil_spec::Program, meta: ModelMeta) -> Result<PathBuf, String> {
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

// ═════════════════════════════════════════════════════════════════════════════
// T E S T
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn ane_matmul_stream_sweep() {
    println!("\n=== ANE MATMUL STREAM PARALLELISM SWEEP ===");
    println!(
        "Model: x[{},{}] @ W[{},{}] -> [{},{}]",
        BATCH, H, H, FFN, BATCH, FFN
    );
    println!("Structured as N parallel matmuls + concat (same total FLOPS)");
    println!(
        "Theoretical peak: {} GFLOPS (M1 ANE FP16)",
        THEORETICAL_PEAK_GFLOPS as u64
    );
    println!("Streams: {:?}", STREAMS);
    println!("batch={}, warmup={}, samples={}", BATCH, WARMUP, SAMPLES);
    println!("{}", "=".repeat(140));

    println!(
        "{:>7} {:>12} {:>12} {:>10} {:>10} {:>8} {:>8} {:>12} {:>10}",
        "Streams",
        "FLOPs",
        "Time(us)",
        "GFLOPS",
        "%Peak",
        "Speedup",
        "Status",
        "tok/s",
        "Compile(ms)"
    );
    println!("{}", "-".repeat(140));

    let mut baseline_gflops: Option<f64> = None;

    for &num_streams in STREAMS {
        let tag = format!("streams_{}", num_streams);

        // ── Build MIL ─────────────────────────────────────────────
        let (prog, out_name) = if num_streams == 1 {
            match build_single_matmul(BATCH) {
                Ok(v) => v,
                Err(e) => {
                    println!(
                        "{:>7} {:>12} {:>12} {:>10} {:>10} {:>8} {:>8} {:>12} {:>10}",
                        num_streams, "N/A", "BUILD_FAIL", "N/A", "N/A", "N/A", "ERR", "N/A", "N/A"
                    );
                    eprintln!("  {} BUILD: {}", tag, e);
                    continue;
                }
            }
        } else {
            match build_streamed_matmul(BATCH, num_streams) {
                Ok(v) => v,
                Err(e) => {
                    println!(
                        "{:>7} {:>12} {:>12} {:>10} {:>10} {:>8} {:>8} {:>12} {:>10}",
                        num_streams, "N/A", "BUILD_FAIL", "N/A", "N/A", "N/A", "ERR", "N/A", "N/A"
                    );
                    eprintln!("  {} BUILD: {}", tag, e);
                    continue;
                }
            }
        };

        let meta = ModelMeta {
            model_name: tag.clone(),
            function_name: "main".into(),
            short_description: format!("ane_matmul_stream_{}", num_streams),
            version: "1.0".into(),
            author: "prism".into(),
            output_name: out_name.clone(),
            inputs: vec![("x".into(), vec![BATCH as i64, H])],
            outputs: vec![(out_name.clone(), vec![BATCH as i64, FFN])],

        };

        // ── Compile ───────────────────────────────────────────────
        let compile_start = Instant::now();
        let target = if num_streams == 1 { "macOS26" } else { "iOS18" };
        let model_path = match compile_with_target(&tag, prog, meta, target) {
            Ok(p) => p,
            Err(e) => {
                println!(
                    "{:>7} {:>12} {:>12} {:>10} {:>10} {:>8} {:>8} {:>12} {:>10}",
                    num_streams, "N/A", "COMPILE_FAIL", "N/A", "N/A", "N/A", "ERR", "N/A", "N/A"
                );
                eprintln!("  {} COMPILE: {}", tag, e);
                continue;
            }
        };
        let compile_ms = compile_start.elapsed().as_millis();
        let path_str = model_path.to_str().expect("valid path");

        // ── Allocate arenas ───────────────────────────────────────
        let in_arena = match Arena::new(BATCH, H as u32, DataType::Float16) {
            Ok(a) => a,
            Err(e) => {
                println!(
                    "{:>7} {:>12} {:>12} {:>10} {:>10} {:>8} {:>8} {:>12} {:>10}",
                    num_streams, "N/A", "ALLOC_FAIL", "N/A", "N/A", "N/A", "ERR", "N/A", "N/A"
                );
                eprintln!("  {} arena: {}", tag, e);
                continue;
            }
        };
        let out_arena = match Arena::new(BATCH, FFN as u32, DataType::Float16) {
            Ok(a) => a,
            Err(e) => {
                println!(
                    "{:>7} {:>12} {:>12} {:>10} {:>10} {:>8} {:>8} {:>12} {:>10}",
                    num_streams, "N/A", "ALLOC_FAIL", "N/A", "N/A", "N/A", "ERR", "N/A", "N/A"
                );
                eprintln!("  {} output: {}", tag, e);
                continue;
            }
        };
        if let Err(e) = fill_arena(&in_arena, BATCH, H as u32) {
            eprintln!("  {} fill: {}", tag, e);
        }

        // ── ANE benchmark ─────────────────────────────────────────
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
                    "{:>7} {:>12} {:>12} {:>10} {:>10} {:>8} {:>8} {:>12} {:>10}",
                    num_streams, "N/A", "ANE_FAIL", "N/A", "N/A", "N/A", "ERR", "N/A", "N/A"
                );
                eprintln!("  {} ANE: {}", tag, e);
                continue;
            }
        };
        let (ane_p50_ns, _ane_p95_ns, _ane_mean_ns) = ane_result;

        // ── CPU benchmark (fallback detection) ────────────────────
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

        // ── Compute metrics ───────────────────────────────────────
        let total_flops = 2.0 * BATCH as f64 * H as f64 * FFN as f64;
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

        let speedup = match baseline_gflops {
            Some(b) if b > 0.0 => gflops / b,
            _ => 1.0,
        };

        let ratio = if cpu_mean_ns > 0.0 {
            ane_p50_ns / cpu_mean_ns
        } else {
            0.0
        };
        let status = if ratio > 0.8 { "CPU_FB" } else { "on-ANE" };

        let tok_s = if time_us > 0.0 {
            1_000_000.0 / (time_us * 48.0 / BATCH as f64)
        } else {
            0.0
        };

        println!(
            "{:>7} {:>12.0e} {:>12.1} {:>10.2} {:>10.3}% {:>8.3}x {:>8} {:>12.1} {:>10}",
            num_streams, total_flops, time_us, gflops, pct_peak, speedup, status, tok_s, compile_ms
        );

        if baseline_gflops.is_none() {
            baseline_gflops = Some(gflops);
        }
    }

    println!("{}", "=".repeat(140));
    println!("Speedup > 1.0 means stream parallelism improves ANE utilization");
    println!("Ideal: 16 streams exposes all 16 ANE compute engines simultaneously");
}
