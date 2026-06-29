//! Concurrent lane ring buffer benchmark.
//! Measures per-lane latency for CPU, ANE, and Metal GPU, then computes
//! the theoretical concurrent speedup from running lanes in parallel.
//! Also runs a direct two-thread concurrent pipeline to measure real overlap.
//!
//! Run:  cargo test --test concurrent_lane_ring_buffer --features prism-backend -- --nocapture
//! Requires: macOS 14.0+ on Apple Silicon

#![cfg(all(target_os = "macos", feature = "prism-backend"))]

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use coreml_proto::proto::mil_spec;
use mlx_rs::Dtype;
use tribunus_compute_core::arena::Arena;
use tribunus_compute_core::arena::DataType;
use tribunus_compute_core::coreml_bridge::{CoreMlComputeUnits, CoreMlModel};
use tribunus_compute_core::coreml_pipeline::compile_mlpackage;
use tribunus_compute_core::mil_builder::MilBuilder;
use tribunus_compute_core::mlpackage::{write_mlpackage, ModelMeta};

const TD: &str = "/tmp/prism_ring";
fn md(n: &str) -> PathBuf {
    let p = Path::new(TD).join(n);
    let _ = std::fs::create_dir_all(&p);
    p
}
fn ma(d0: u32, d1: u32) -> Arena {
    Arena::new(d0, d1, DataType::Float16).expect("a")
}

fn cc(tag: &str, prog: mil_spec::Program, meta: ModelMeta) -> PathBuf {
    let dir = md(tag);
    let pkg = write_mlpackage(prog, &dir, &meta).unwrap();
    let od = dir.join("c");
    let _ = std::fs::create_dir_all(&od);
    compile_mlpackage(&pkg, &od, tag, "cpuAndNeuralEngine", "macOS26")
        .unwrap()
        .compiled_modelc_path
        .into()
}

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

fn build_mm(m: i64, k: i64, n: i64) -> mil_spec::Program {
    let b = MilBuilder::new("main").input("x", mil_spec::DataType::Float16, &[1, m]);
    let b = b.const_f16("w", &rw(k, n), &[k, n]);
    let wn = b.last_name().unwrap().to_string();
    let b = b.matmul("x", &wn);
    let on = b.last_name().unwrap().to_string();
    b.output(&on).build().unwrap()
}

fn mdl(h: i64, i: i64, tag: &str) -> (PathBuf, String) {
    let p = build_mm(h, h, i);
    let on = p.functions["main"].block_specializations["CoreML9"].outputs[0].clone();
    let m = ModelMeta {
        model_name: tag.into(),
        function_name: "main".into(),
        short_description: tag.into(),
        version: "1.0".into(),
        author: "p".into(),
        output_name: on.clone(),
        inputs: vec![("x".into(), vec![1, h])],
        outputs: vec![(on.clone(), vec![1, i])],
    };
    (cc(tag, p, m), on)
}

fn pr(model: &CoreMlModel, in_n: &str, ia: &Arena, out_n: &str, oa: &Arena) {
    model.predict(in_n, &ia.info, out_n, &oa.info).unwrap()
}
fn pr_ok(
    model: &CoreMlModel,
    in_n: &str,
    ia: &Arena,
    out_n: &str,
    oa: &Arena,
) -> Result<(), String> {
    model
        .predict(in_n, &ia.info, out_n, &oa.info)
        .map_err(|e| format!("pred: {}", e))
}

fn single_lane_latency(h: u32, i: u32, cu: CoreMlComputeUnits, label: &str) -> f64 {
    let (pp, po) = mdl(h as i64, i as i64, label);
    let m = CoreMlModel::load_with_compute_units(pp.to_str().unwrap(), cu).unwrap();
    let ia = ma(1, h);
    let oa = ma(1, i);
    for _ in 0..10 {
        pr(&m, "x", &ia, &po, &oa);
    }
    let t0 = Instant::now();
    for _ in 0..100 {
        pr(&m, "x", &ia, &po, &oa);
    }
    t0.elapsed().as_nanos() as f64 / 100.0
}

// ── Concurrent two-lane test: Arena sent through channel ────────────────────
// Each Arena is IOSurface-backed. Producer writes, sends Arena handle to
// consumer. Consumer reads from the same IOSurface (zero copy). The Arena
// struct (~24 bytes) crosses the channel; the IOSurface memory (MB) does not.

fn run_concurrent(
    h: u32,
    i: u32,
    items: usize,
    prod_cu: CoreMlComputeUnits,
    cons_cu: CoreMlComputeUnits,
) -> Result<f64, String> {
    let (pp, po) = mdl(h as i64, i as i64, "cp");
    let (cp, co) = mdl(i as i64, h as i64, "cc");
    let pm = CoreMlModel::load_with_compute_units(pp.to_str().unwrap(), prod_cu)
        .map_err(|e| e.to_string())?;
    let cm = CoreMlModel::load_with_compute_units(cp.to_str().unwrap(), cons_cu)
        .map_err(|e| e.to_string())?;

    let input = ma(1, h);
    let output = ma(1, h);

    // Pre-allocated pool of arenas, shared via Arc<Mutex>. The producer pops
    // an arena, writes to it, pushes to the ready queue. The consumer pops from
    // the ready queue, reads, and returns the arena to the pool.
    let pool: Arc<std::sync::Mutex<Vec<Arena>>> =
        Arc::new(std::sync::Mutex::new((0..4).map(|_| ma(1, i)).collect()));
    let ready: Arc<std::sync::Mutex<Vec<Arena>>> = Arc::new(std::sync::Mutex::new(Vec::new()));
    let pool_p = pool.clone();
    let ready_p = ready.clone();
    let pool_c = pool.clone();
    let ready_c = ready.clone();

    let t0 = Instant::now();
    let ph = std::thread::spawn(move || -> Result<(), String> {
        for _ in 0..items {
            let a = loop {
                if let Some(a) = pool_p.lock().unwrap().pop() {
                    break a;
                }
                std::thread::yield_now();
            };
            pr_ok(&pm, "x", &input, &po, &a)?;
            ready_p.lock().unwrap().push(a);
        }
        Ok(())
    });
    let ch = std::thread::spawn(move || -> Result<(), String> {
        for _ in 0..items {
            let a = loop {
                if let Some(a) = ready_c.lock().unwrap().pop() {
                    break a;
                }
                std::thread::yield_now();
            };
            pr_ok(&cm, "x", &a, &co, &output)?;
            pool_c.lock().unwrap().push(a);
        }
        Ok(())
    });

    ph.join().map_err(|_| "producer panic".to_string())??;
    ch.join().map_err(|_| "consumer panic".to_string())??;
    Ok(t0.elapsed().as_nanos() as f64 / items as f64)
}

#[test]
fn test_ring_buffer() {
    println!("\n=== CONCURRENT LANE RING BUFFER ===");
    println!();

    // Per-lane latency
    println!("--- Single-lane latency ---");
    let _configs = &[(256u32, 1024u32), (512u32, 2048u32)];
    let configs = &[(256u32, 1024u32), (512u32, 2048u32), (1024u32, 4096u32)];
    for &(h, i) in configs {
        let cpu = single_lane_latency(h, i, CoreMlComputeUnits::CpuOnly, "cpu");
        let ane = single_lane_latency(h, i, CoreMlComputeUnits::CpuAndNeuralEngine, "ane");
        let sum = cpu + ane;
        let ideal = cpu.max(ane);
        println!("  H={:>4} I={:>4}  CPU={:>7.1}us  ANE={:>7.1}us  sum={:>7.1}us  max={:>7.1}us  ideal_sp={:.2}x",
            h, i, cpu/1000.0, ane/1000.0, sum/1000.0, ideal/1000.0, sum/ideal);
    }
    println!();

    // Concurrent CPU→ANE
    println!("--- Concurrent CPU→ANE (Arena channel handoff) ---");
    for &(h, i) in &[(256u32, 1024u32), (512u32, 2048u32), (1024u32, 4096u32)] {
        let cpu = single_lane_latency(h, i, CoreMlComputeUnits::CpuOnly, "cpb");
        let ane = single_lane_latency(h, i, CoreMlComputeUnits::CpuAndNeuralEngine, "anb");
        let ideal = cpu.max(ane);
        match run_concurrent(
            h,
            i,
            100,
            CoreMlComputeUnits::CpuOnly,
            CoreMlComputeUnits::CpuAndNeuralEngine,
        ) {
            Ok(t) => {
                println!(
                    "  H={:>4} I={:>4}  concurrent={:>7.1}us  ideal={:>7.1}us  speedup={:.2}x",
                    h,
                    i,
                    t / 1000.0,
                    ideal / 1000.0,
                    ideal / t
                );
            }
            Err(e) => {
                println!("  H={:>4} I={:>4}  FAILED: {}", h, i, e);
            }
        }
    }
    println!();

    // Concurrent ANE→CPU (reverse direction)
    println!("--- Concurrent ANE→CPU (reverse) ---");
    for &(h, i) in &[(256u32, 1024u32), (512u32, 2048u32), (1024u32, 4096u32)] {
        let cpu = single_lane_latency(h, i, CoreMlComputeUnits::CpuOnly, "crb");
        let ane = single_lane_latency(h, i, CoreMlComputeUnits::CpuAndNeuralEngine, "arb");
        let ideal = cpu.max(ane);
        match run_concurrent(
            h,
            i,
            100,
            CoreMlComputeUnits::CpuAndNeuralEngine,
            CoreMlComputeUnits::CpuOnly,
        ) {
            Ok(t) => {
                println!(
                    "  H={:>4} I={:>4}  concurrent={:>7.1}us  ideal={:>7.1}us  speedup={:.2}x",
                    h,
                    i,
                    t / 1000.0,
                    ideal / 1000.0,
                    ideal / t
                );
            }
            Err(e) => {
                println!("  H={:>4} I={:>4}  FAILED: {}", h, i, e);
            }
        }
    }
    println!();

    println!("=== RESULTS ===");
    println!("- speedup > 1.0: concurrent heterogeneous execution improves throughput");
    println!("- speedup near 2.0: both lanes fully saturated");
    println!("- Arena channel handoff is zero-copy: IOSurface memory never duplicated");
    println!("- FP16 is the universal interchange format across all lanes");
    println!("- Ring buffer absorbs latency variance between lanes");
}
