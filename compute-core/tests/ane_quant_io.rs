//! Test whether compressing IOSurface traffic (INT8 I/O) improves ANE utilization.
//!
//! Compares three models at batch=16384:
//!   baseline:  x[FP16] @ W[FP16] -> y[FP16]            (as measured: ~52%)
//!   q_input:   x[INT8] -> dequant -> matmul -> y[FP16]  (2x less input bandwidth)
//!   q_io:      x[INT8] -> dequant -> matmul -> quant -> y[INT8] (2x less I/O bandwidth)
//!
//! Run: cargo test --test ane_quant_io --features prism-backend -- --nocapture

#![cfg(all(target_os = "macos", feature = "prism-backend"))]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use coreml_proto::proto::mil_spec::{self, argument, dimension, tensor_value, value};
use mlx_rs::Dtype;
use tribunus_compute_core::arena::Arena;
use tribunus_compute_core::coreml_bridge::{CoreMlComputeUnits, CoreMlModel};
use tribunus_compute_core::coreml_pipeline::compile_mlpackage;
use tribunus_compute_core::mil_builder::MilBuilder;
use tribunus_compute_core::mlpackage::{write_mlpackage, ModelMeta};

const TEST_DIR: &str = "/tmp/prism_ane_quant_io";
const BATCH: u32 = 16384;
const H: i64 = 2048;
const FFN: i64 = 4096;
const THEORETICAL_PEAK_GFLOPS: f64 = 11_000.0;
const WARMUP: usize = 5;
const SAMPLES: usize = 20;

// ── Proto helpers ──────────────────────────────────────────────────────────

fn tensor_type(dtype: mil_spec::DataType, shape: &[i64]) -> mil_spec::TensorType {
    let dims = shape
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

fn vt(tt: mil_spec::TensorType) -> mil_spec::ValueType {
    mil_spec::ValueType {
        r#type: Some(mil_spec::value_type::Type::TensorType(tt)),
    }
}

fn named_arg(name: &str) -> mil_spec::Argument {
    mil_spec::Argument {
        arguments: vec![argument::Binding {
            binding: Some(argument::binding::Binding::Name(name.to_string())),
        }],
    }
}

#[allow(dead_code)]
fn float_attr(val: f32) -> mil_spec::Value {
    let ft = mil_spec::TensorValue {
        value: Some(tensor_value::Value::Floats(tensor_value::RepeatedFloats {
            values: vec![val],
        })),
    };
    mil_spec::Value {
        doc_string: String::new(),
        r#type: Some(vt(mil_spec::TensorType {
            data_type: mil_spec::DataType::Float32 as i32,
            rank: 0,
            dimensions: vec![],
            attributes: HashMap::new(),
        })),
        value: Some(value::Value::ImmediateValue(value::ImmediateValue {
            value: Some(value::immediate_value::Value::Tensor(ft)),
        })),
    }
}

fn string_attr(val: &str) -> mil_spec::Value {
    let st = mil_spec::TensorValue {
        value: Some(tensor_value::Value::Strings(
            tensor_value::RepeatedStrings {
                values: vec![val.to_string()],
            },
        )),
    };
    mil_spec::Value {
        doc_string: String::new(),
        r#type: Some(vt(mil_spec::TensorType {
            data_type: mil_spec::DataType::String as i32,
            rank: 0,
            dimensions: vec![],
            attributes: HashMap::new(),
        })),
        value: Some(value::Value::ImmediateValue(value::ImmediateValue {
            value: Some(value::immediate_value::Value::Tensor(st)),
        })),
    }
}

fn float16_arg(val: f32) -> mil_spec::Argument {
    // FP16 scalar stored as raw bytes (like const_f16 in MilBuilder)
    let val_f16 = half_f32_to_u16(val);
    let bytes_tensor = mil_spec::TensorValue {
        value: Some(tensor_value::Value::Bytes(tensor_value::RepeatedBytes {
            values: val_f16.to_le_bytes().to_vec(),
        })),
    };
    let v = mil_spec::Value {
        doc_string: String::new(),
        r#type: Some(mil_spec::ValueType {
            r#type: Some(mil_spec::value_type::Type::TensorType(
                mil_spec::TensorType {
                    data_type: mil_spec::DataType::Float16 as i32,
                    rank: 0,
                    dimensions: vec![],
                    attributes: HashMap::new(),
                },
            )),
        }),
        value: Some(value::Value::ImmediateValue(value::ImmediateValue {
            value: Some(value::immediate_value::Value::Tensor(bytes_tensor)),
        })),
    };
    mil_spec::Argument {
        arguments: vec![argument::Binding {
            binding: Some(argument::binding::Binding::Value(v)),
        }],
    }
}

fn half_f32_to_u16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7FFFFF;
    if exp == 0xFF {
        return sign | 0x7C00 | (if mant != 0 { 1 } else { 0 });
    }
    if exp <= 112 {
        return sign;
    }
    let half_exp = exp - 127 + 15;
    if half_exp >= 31 {
        return sign | 0x7C00;
    }
    let half_mant = (mant >> 13) as u16;
    sign | ((half_exp as u16) << 10) | half_mant
}

fn int_attr(val: i64) -> mil_spec::Value {
    let it = mil_spec::TensorValue {
        value: Some(tensor_value::Value::LongInts(
            tensor_value::RepeatedLongInts { values: vec![val] },
        )),
    };
    mil_spec::Value {
        doc_string: String::new(),
        r#type: Some(vt(mil_spec::TensorType {
            data_type: mil_spec::DataType::Int64 as i32,
            rank: 0,
            dimensions: vec![],
            attributes: HashMap::new(),
        })),
        value: Some(value::Value::ImmediateValue(value::ImmediateValue {
            value: Some(value::immediate_value::Value::Tensor(it)),
        })),
    }
}

#[allow(dead_code)]
fn bool_attr(val: bool) -> mil_spec::Value {
    let bt = mil_spec::TensorValue {
        value: Some(tensor_value::Value::Bools(tensor_value::RepeatedBools {
            values: vec![val],
        })),
    };
    mil_spec::Value {
        doc_string: String::new(),
        r#type: Some(vt(mil_spec::TensorType {
            data_type: mil_spec::DataType::Bool as i32,
            rank: 0,
            dimensions: vec![],
            attributes: HashMap::new(),
        })),
        value: Some(value::Value::ImmediateValue(value::ImmediateValue {
            value: Some(value::immediate_value::Value::Tensor(bt)),
        })),
    }
}

fn make_op(
    op_type: &str,
    _op_name: &str,
    inputs: HashMap<String, mil_spec::Argument>,
    outputs: &[(&str, &mil_spec::ValueType)],
    attrs: HashMap<String, mil_spec::Value>,
) -> mil_spec::Operation {
    mil_spec::Operation {
        r#type: op_type.to_string(),
        inputs,
        outputs: outputs
            .iter()
            .map(|(n, vt_)| mil_spec::NamedValueType {
                name: n.to_string(),
                r#type: Some((*vt_).clone()),
            })
            .collect(),
        blocks: vec![],
        attributes: attrs,
    }
}

// ── Model builds ──────────────────────────────────────────────────────────

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

/// Baseline: x[FP16] @ W[FP16] -> y[FP16]
fn build_baseline() -> Result<(mil_spec::Program, String, String), String> {
    let w = seeded_weights(42, H, FFN);
    let b = MilBuilder::new("main")
        .input("x", mil_spec::DataType::Float16, &[BATCH as i64, H])
        .const_f16("w", &w, &[H, FFN]);
    let wn = b.last_name().ok_or("weight name")?.to_string();
    let b = b.matmul("x", &wn);
    let out_name = b.last_name().ok_or("matmul name")?.to_string();
    let b = b.output(&out_name);
    let prog = b.build().map_err(|e| format!("MIL: {}", e))?;
    Ok((prog, "x".into(), out_name))
}

/// Quantized input (iOS18): x[INT8] -> dequant -> matmul -> y[FP16]
/// Manually constructs the dequantize op with iOS18 param names.
fn build_quantized_input() -> Result<(mil_spec::Program, String, String), String> {
    let scale_val = 1.0 / 127.0;
    let w = seeded_weights(42, H, FFN);

    let mut b = MilBuilder::new("main")
        .input("x", mil_spec::DataType::Int8, &[BATCH as i64, H])
        .const_f16("w", &w, &[H, FFN]);
    let wn = b.last_name().ok_or("weight name")?.to_string();

    // Dequantize: input[INT8] -> output[FP16]
    // iOS18: scale is an ATTRIBUTE, also needs output_dtype
    let dq_name = "dq";
    let dq_vt = vt(tensor_type(mil_spec::DataType::Float16, &[BATCH as i64, H]));
    let mut dq_inputs = HashMap::new();
    dq_inputs.insert("input".to_string(), named_arg("x"));
    let mut dq_attrs = HashMap::new();
    dq_attrs.insert("name".to_string(), string_attr(dq_name));
    dq_attrs.insert("axis".to_string(), int_attr(-1));
    dq_attrs.insert("output_dtype".to_string(), string_attr("fp16"));
    // scale must be an input argument, not an attribute
    dq_inputs.insert("scale".to_string(), float16_arg(scale_val));
    let dq_op = make_op(
        "dequantize",
        dq_name,
        dq_inputs,
        &[(dq_name, &dq_vt)],
        dq_attrs,
    );
    b = b.operation(dq_op, Some((dq_name, dq_vt)));

    // Matmul: dq[FP16] @ w[FP16] -> y[FP16]
    b = b.matmul(dq_name, &wn);
    let out_name = b.last_name().ok_or("matmul name")?.to_string();
    let b = b.output(&out_name);
    let prog = b.build().map_err(|e| format!("MIL: {}", e))?;
    Ok((prog, "x".into(), out_name))
}

/// Quantized I/O (iOS18): x[INT8] -> dequant -> matmul -> quant -> y[INT8]
fn build_quantized_io() -> Result<(mil_spec::Program, String, String), String> {
    let scale_val = 1.0 / 127.0;
    let w = seeded_weights(42, H, FFN);

    let mut b = MilBuilder::new("main")
        .input("x", mil_spec::DataType::Int8, &[BATCH as i64, H])
        .const_f16("w", &w, &[H, FFN]);
    let wn = b.last_name().ok_or("weight name")?.to_string();

    // Dequantize
    let dq_name = "dq";
    let dq_vt = vt(tensor_type(mil_spec::DataType::Float16, &[BATCH as i64, H]));
    let mut dq_inputs = HashMap::new();
    dq_inputs.insert("input".to_string(), named_arg("x"));
    // dq_attrs removed — scale is an input
    let mut dq_attrs = HashMap::new();
    dq_attrs.insert("name".to_string(), string_attr(dq_name));
    dq_attrs.insert("axis".to_string(), int_attr(-1));
    dq_attrs.insert("output_dtype".to_string(), string_attr("fp16"));
    dq_inputs.insert("scale".to_string(), float16_arg(scale_val));
    let dq_op = make_op(
        "dequantize",
        dq_name,
        dq_inputs,
        &[(dq_name, &dq_vt)],
        dq_attrs,
    );
    b = b.operation(dq_op, Some((dq_name, dq_vt)));

    // Matmul
    b = b.matmul(dq_name, &wn);
    let matmul_name = b.last_name().ok_or("matmul name")?.to_string();

    // Quantize: matmul_out[FP16] -> y[INT8]
    let q_name = "q";
    let q_vt = vt(tensor_type(mil_spec::DataType::Int8, &[BATCH as i64, FFN]));
    let mut q_inputs = HashMap::new();
    q_inputs.insert("input".to_string(), named_arg(&matmul_name));
    let mut q_attrs = HashMap::new();
    q_attrs.insert("name".to_string(), string_attr(q_name));
    q_attrs.insert("axis".to_string(), int_attr(-1));
    q_inputs.insert("scale".to_string(), float16_arg(scale_val));
    q_inputs.insert(
        "output_dtype".to_string(),
        mil_spec::Argument {
            arguments: vec![mil_spec::argument::Binding {
                binding: Some(mil_spec::argument::binding::Binding::Value(string_attr(
                    "int8",
                ))),
            }],
        },
    );
    let q_op = make_op("quantize", q_name, q_inputs, &[(q_name, &q_vt)], q_attrs);
    b = b.operation(q_op, Some((q_name, q_vt)));

    let b = b.output(q_name);
    let prog = b.build().map_err(|e| format!("MIL: {}", e))?;
    Ok((prog, "x".into(), q_name.to_string()))
}

// ── Compilation & benchmarking ─────────────────────────────────────────────

fn model_dir(name: &str) -> PathBuf {
    let p = Path::new(TEST_DIR).join(name);
    let _ = std::fs::create_dir_all(&p);
    p
}

fn compile(
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

fn fill_arena_fp16(arena: &Arena, batch: u32, cols: u32) -> Result<(), String> {
    arena.lock().map_err(|e| format!("lock: {}", e))?;
    unsafe {
        let ptr = arena.base_ptr() as *mut u16;
        for i in 0..(batch as usize * cols as usize) {
            let val = ((i as u16).wrapping_mul(265).wrapping_add(1234)) & 0x7FFF;
            *ptr.add(i) = val;
        }
    }
    arena.unlock().map_err(|e| format!("unlock: {}", e))?;
    Ok(())
}

fn fill_arena_int8(arena: &Arena, batch: u32, cols: u32) -> Result<(), String> {
    arena.lock().map_err(|e| format!("lock: {}", e))?;
    unsafe {
        let ptr = arena.base_ptr() as *mut i8;
        for i in 0..(batch as usize * cols as usize) {
            let val = ((i as i16).wrapping_mul(13).wrapping_add(42)) as i8;
            *ptr.add(i) = val;
        }
    }
    arena.unlock().map_err(|e| format!("unlock: {}", e))?;
    Ok(())
}

fn bench_one(
    path: &str,
    cu: CoreMlComputeUnits,
    in_name: &str,
    in_arena: &Arena,
    out_name: &str,
    out_arena: &Arena,
) -> Result<(f64, f64), String> {
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
    Ok((
        samples[samples.len() / 2],
        samples.iter().sum::<f64>() / samples.len() as f64,
    ))
}

// ═════════════════════════════════════════════════════════════════════════════
// T E S T
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn ane_quant_io_sweep() {
    println!("\n=== ANE QUANTIZED I/O SWEEP (batch={}) ===", BATCH);
    println!(
        "Theoretical peak: {} GFLOPS",
        THEORETICAL_PEAK_GFLOPS as u64
    );
    println!("{}", "=".repeat(120));
    println!(
        "{:>16} {:>8} {:>10} {:>10} {:>10} {:>8} {:>12} {:>10}",
        "Config", "I/O B/W", "Time(us)", "GFLOPS", "%Peak", "Status", "tok/s", "Compile(ms)"
    );
    println!("{}", "-".repeat(120));

    let configs: &[(
        fn() -> Result<(mil_spec::Program, String, String), String>,
        &str,
        Dtype,
        Dtype,
        &str,
    )] = &[
        (
            build_baseline,
            "FP16 I/O",
            Dtype::Float16,
            Dtype::Float16,
            "macOS26",
        ),
        (
            build_quantized_input,
            "INT8 in",
            Dtype::Int8,
            Dtype::Float16,
            "macOS26",
        ),
        (
            build_quantized_io,
            "INT8 I/O",
            Dtype::Int8,
            Dtype::Int8,
            "macOS26",
        ),
    ];

    for (build_fn, label, in_dtype, out_dtype, target) in configs {
        let tag = format!(
            "qio_{}",
            label.replace(' ', "_").to_lowercase().replace("/", "")
        );
        eprintln!("\n--- {} ({}, target={}) ---", label, tag, target);

        let (prog, in_name, out_name) = match build_fn() {
            Ok(v) => v,
            Err(e) => {
                println!(
                    "{:>16} {:>8} {:>10} {:>10} {:>10} {:>8} {:>12} {:>10}",
                    label, "N/A", "BUILD_FAIL", "N/A", "N/A", "ERR", "N/A", "N/A"
                );
                eprintln!("  BUILD: {}", e);
                continue;
            }
        };

        let meta = ModelMeta {
            model_name: tag.clone(),
            function_name: "main".into(),
            short_description: format!("ane_qio_{}", tag),
            version: "1.0".into(),
            author: "prism".into(),
            output_name: out_name.clone(),
            inputs: vec![("x".into(), vec![BATCH as i64, H])],
            outputs: vec![(out_name.clone(), vec![BATCH as i64, FFN])],
            spec_version: if *in_dtype == Dtype::Int8 { 10 } else { 9 },
        };

        let compile_start = Instant::now();
        let model_path = match compile(&tag, prog, meta, target) {
            Ok(p) => p,
            Err(e) => {
                println!(
                    "{:>16} {:>8} {:>10} {:>10} {:>10} {:>8} {:>12} {:>10}",
                    label, "N/A", "COMPILE_FAIL", "N/A", "N/A", "ERR", "N/A", "N/A"
                );
                eprintln!("  COMPILE: {}", e);
                continue;
            }
        };
        let compile_ms = compile_start.elapsed().as_millis();
        let path_str = model_path.to_str().expect("path");
        let in_arena = Arena::new(BATCH, H as u32, *in_dtype).expect("in arena");
        let out_arena = Arena::new(BATCH, FFN as u32, *out_dtype).expect("out arena");

        match *in_dtype {
            Dtype::Float16 => {
                let _ = fill_arena_fp16(&in_arena, BATCH, H as u32);
            }
            _ => {
                let _ = fill_arena_int8(&in_arena, BATCH, H as u32);
            }
        }

        let ane_result = match bench_one(
            path_str,
            CoreMlComputeUnits::CpuAndNeuralEngine,
            &in_name,
            &in_arena,
            &out_name,
            &out_arena,
        ) {
            Ok(r) => r,
            Err(e) => {
                println!(
                    "{:>16} {:>8} {:>10} {:>10} {:>10} {:>8} {:>12} {:>10}",
                    label, "N/A", "ANE_FAIL", "N/A", "N/A", "ERR", "N/A", "N/A"
                );
                eprintln!("  ANE: {}", e);
                continue;
            }
        };
        let (ane_p50_ns, ane_mean_ns) = ane_result;

        let cpu_result = match bench_one(
            path_str,
            CoreMlComputeUnits::CpuOnly,
            &in_name,
            &in_arena,
            &out_name,
            &out_arena,
        ) {
            Ok(r) => r,
            Err(_) => (0.0, 0.0),
        };
        let (_cpu_p50, cpu_mean_ns) = cpu_result;

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
        let ratio = if cpu_mean_ns > 0.0 {
            ane_mean_ns / cpu_mean_ns
        } else {
            0.0
        };
        let status = if ratio > 0.8 { "CPU_FB" } else { "on-ANE" };
        let tok_s = if time_us > 0.0 {
            1_000_000.0 / (time_us * 48.0 / BATCH as f64)
        } else {
            0.0
        };

        let bw_ratio = if *in_dtype == Dtype::Int8 && *out_dtype == Dtype::Int8 {
            "1/4"
        } else if *in_dtype == Dtype::Int8 {
            "1/2"
        } else {
            "1x"
        };

        println!(
            "{:>16} {:>8} {:>10.1} {:>10.2} {:>9.3}% {:>8} {:>12.1} {:>10}",
            label, bw_ratio, time_us, gflops, pct_peak, status, tok_s, compile_ms
        );
    }
    println!("{}", "=".repeat(120));
}
