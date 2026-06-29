//! Tri-lane concurrent pipeline — serialized vs heterogeneous concurrent.
//!
//! Removes all identified bottlenecks from the previous ring-buffer attempt:
//!   1. Mutex spin-yield → lock-free AtomicBool flags (tight spin, no yield)
//!   2. Per-item Arena alloc → pre-allocated slot pool transferred via atomic handoff
//!   3. Single-thread pipeline → producer/consumer threads with atomic ready/consumed handshake
//!
//! Measures: per-item latency for serialized (same thread) vs concurrent (two threads)
//! pipeline, swept over operation sizes to find where concurrency wins.
//!
//! Run:  cargo test --test tri_lane_concurrent --features prism-backend -- --nocapture
//! Requires: macOS 14.0+ on Apple Silicon

#![cfg(all(target_os = "macos", feature = "prism-backend"))]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
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

const TD: &str = "/tmp/prism_tri_lane";
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

fn pred(model: &CoreMlModel, in_n: &str, ia: &Arena, out_n: &str, oa: &Arena) {
    model.predict(in_n, &ia.info, out_n, &oa.info).unwrap()
}

// ── Lock-free ring slot ────────────────────────────────────────────────────
//
// Each slot has an Arena and two atomic flags:
//   ready:    producer → consumer  (data is written, consumer may read)
//   consumed: consumer → producer  (consumer is done, producer may overwrite)
//
// Both sides tight-spin (no yield!) because the wait is bounded by the
// partner's predict latency (microseconds), not by OS scheduling (milliseconds).

struct Slot {
    arena: Arena,
    ready: AtomicBool,
    consumed: AtomicBool,
}

// SAFETY: Slot is Send because Arena is Send, and AtomicBool is Send+Sync.
// The atomic flags protect against concurrent access — producer and consumer
// never touch the same arena simultaneously because the flags enforce ordering.
unsafe impl Send for Slot {}
// SAFETY: atomic flags guarantee exclusive access — producer and consumer
// never access the same slot simultaneously.
unsafe impl Sync for Slot {}

fn spin_wait(flag: &AtomicBool, target: bool) {
    while flag.load(Ordering::Acquire) != target {
        std::hint::spin_loop();
    }
}

fn latency(h: u32, i: u32, cu: CoreMlComputeUnits, label: &str) -> f64 {
    let (pp, po) = mdl(h as i64, i as i64, label);
    let m = CoreMlModel::load_with_compute_units(pp.to_str().unwrap(), cu).unwrap();
    let ia = ma(1, h);
    let oa = ma(1, i);
    for _ in 0..10 {
        pred(&m, "x", &ia, &po, &oa);
    }
    let t0 = Instant::now();
    for _ in 0..200 {
        pred(&m, "x", &ia, &po, &oa);
    }
    t0.elapsed().as_nanos() as f64 / 200.0
}

// ── Serialized baseline: same thread, sequential predict calls ──────────────

fn run_serial(
    h: u32,
    i: u32,
    items: usize,
    prod_cu: CoreMlComputeUnits,
    cons_cu: CoreMlComputeUnits,
) -> f64 {
    let (pp, po) = mdl(h as i64, i as i64, "sp");
    let (cp, co) = mdl(i as i64, h as i64, "sc");
    let pm = CoreMlModel::load_with_compute_units(pp.to_str().unwrap(), prod_cu).unwrap();
    let cm = CoreMlModel::load_with_compute_units(cp.to_str().unwrap(), cons_cu).unwrap();
    let ia = ma(1, h);
    let ring = ma(1, i);
    let oa = ma(1, h);
    for _ in 0..10 {
        pred(&pm, "x", &ia, &po, &ring);
        pred(&cm, "x", &ring, &co, &oa);
    }
    let t0 = Instant::now();
    for _ in 0..items {
        pred(&pm, "x", &ia, &po, &ring);
        pred(&cm, "x", &ring, &co, &oa);
    }
    t0.elapsed().as_nanos() as f64 / items as f64
}

// ── Concurrent: two threads, atomic slot handoff, no mutex, no yield ───────

fn run_concurrent(
    h: u32,
    i: u32,
    items: usize,
    depth: usize,
    prod_cu: CoreMlComputeUnits,
    cons_cu: CoreMlComputeUnits,
) -> f64 {
    let (pp, po) = mdl(h as i64, i as i64, "cp");
    let (cp, co) = mdl(i as i64, h as i64, "cc");
    let pm = CoreMlModel::load_with_compute_units(pp.to_str().unwrap(), prod_cu).unwrap();
    let cm = CoreMlModel::load_with_compute_units(cp.to_str().unwrap(), cons_cu).unwrap();

    // Pre-allocate ring slots
    let ring: Vec<Slot> = (0..depth)
        .map(|_| {
            let a = ma(1, i);
            // Initially: not ready (producer must write), but consumed=true (producer may write)
            Slot {
                arena: a,
                ready: AtomicBool::new(false),
                consumed: AtomicBool::new(true),
            }
        })
        .collect();
    let ring = Arc::new(ring);
    let ring_p = ring.clone();
    let ring_c = ring.clone();

    let input = ma(1, h);
    let output = ma(1, h);

    let t0 = Instant::now();

    // Producer thread (writes to ring)
    let p_handle = std::thread::spawn(move || {
        for idx in 0..items {
            let slot = idx % depth;
            // Wait until consumer has released this slot
            spin_wait(&ring_p[slot].consumed, true);
            // Write
            pred(&pm, "x", &input, &po, &ring_p[slot].arena);
            // Signal consumer
            ring_p[slot].ready.store(true, Ordering::Release);
            ring_p[slot].consumed.store(false, Ordering::Release);
        }
    });

    // Consumer thread (reads from ring)
    let c_handle = std::thread::spawn(move || {
        for idx in 0..items {
            let slot = idx % depth;
            // Wait until producer has written
            spin_wait(&ring_c[slot].ready, true);
            // Read
            pred(&cm, "x", &ring_c[slot].arena, &co, &output);
            // Signal producer: slot is free
            ring_c[slot].ready.store(false, Ordering::Release);
            ring_c[slot].consumed.store(true, Ordering::Release);
        }
    });

    p_handle.join().unwrap();
    c_handle.join().unwrap();
    t0.elapsed().as_nanos() as f64 / items as f64
}

// ── Tri-lane pipeline (CPU→ANE→Metal) ────────────────────────────────────
// Stage 1: CPU produces intermediate
// Stage 2: ANE transforms intermediate
// Stage 3: (future) Metal consumes

#[test]
fn test_tri_lane() {
    println!("\n=== TRI-LANE CONCURRENT: SERIALIZED vs CONCURRENT ===");
    println!("Lock-free atomic slot handoff. Tight spin (no scheduler yield).");
    println!();

    let sizes: &[(u32, u32, &str)] = &[
        (256, 1024, "small"),
        (512, 2048, "medium"),
        (1024, 4096, "large"),
    ];

    for &(h, i, lb) in sizes {
        // Baseline: single-lane latency
        let cpu = latency(h, i, CoreMlComputeUnits::CpuOnly, &format!("cpu_{}", lb));
        let ane = latency(
            h,
            i,
            CoreMlComputeUnits::CpuAndNeuralEngine,
            &format!("ane_{}", lb),
        );

        // Serialized pipeline: same thread, CPU then ANE
        let ser = run_serial(
            h,
            i,
            200,
            CoreMlComputeUnits::CpuOnly,
            CoreMlComputeUnits::CpuAndNeuralEngine,
        );

        // Concurrent pipeline: CPU thread + ANE thread, atomic ring depth 2
        let con = run_concurrent(
            h,
            i,
            200,
            2,
            CoreMlComputeUnits::CpuOnly,
            CoreMlComputeUnits::CpuAndNeuralEngine,
        );

        let speedup = ser / con;

        println!("{} (H={} I={}):", lb, h, i);
        println!("  CPU-only:        {:>7.1}us", cpu / 1000.0);
        println!("  ANE-only:        {:>7.1}us", ane / 1000.0);
        let m = metal_latency(h, i, &format!("mtl_{}", lb));
        if m > 0.0 {
            println!("  Metal-only:      {:>7.1}us", m / 1000.0);
        }
        println!(
            "  Serialized:      {:>7.1}us  (CPU+ANE sum: {:>7.1}us)",
            ser / 1000.0,
            (cpu + ane) / 1000.0
        );
        println!(
            "  Concurrent:      {:>7.1}us  speedup={:.3}x",
            con / 1000.0,
            speedup
        );
        println!(
            "  Ideal (max lane): {:>7.1}us  ideal_sp={:.3}x",
            cpu.max(ane) / 1000.0,
            (cpu + ane) / cpu.max(ane)
        );
        println!();
    }

    println!("=== RESULTS ===");
    println!("- speedup > 1.0 = concurrent heterogeneous execution wins");
    println!("- speedup near ideal_sp = atomic handoff adds negligible overhead");
    println!("- speedup < 1.0 = sync overhead still dominates compute");
    println!("- Lock-free AtomicBool avoids OS scheduler (no yield_now)");
    println!("- ring depth shields producer from consumer latency variance");
}

// ── Metal lane ───────────────────────────────────────────────────────────

#[cfg(feature = "metal-dispatch")]
fn metal_source(_m: u32, n: u32, k: u32, name: &str) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    write!(s, "#include <metal_stdlib>\nusing namespace metal;\n").unwrap();
    write!(
        s,
        "kernel void {}(device const half* input [[buffer(0)]],\n",
        name
    )
    .unwrap();
    write!(
        s,
        "                    device const half* weight [[buffer(1)]],\n"
    )
    .unwrap();
    write!(
        s,
        "                    device half* output [[buffer(2)]],\n"
    )
    .unwrap();
    write!(
        s,
        "                    uint2 gid [[thread_position_in_grid]]) {{\n"
    )
    .unwrap();
    write!(s, "    uint row = gid.y;\n").unwrap();
    write!(s, "    if (row >= {}) return;\n", n).unwrap();
    write!(s, "    half acc = 0;\n").unwrap();
    write!(s, "    for (uint i = 0; i < {}; ++i) {{\n", k).unwrap();
    write!(s, "        acc += input[i] * weight[row * {} + i];\n", k).unwrap();
    write!(s, "    }}\n").unwrap();
    write!(s, "    output[row] = acc;\n}}\n").unwrap();
    s
}

#[cfg(feature = "metal-dispatch")]
fn metal_latency(h: u32, i: u32, label: &str) -> f64 {
    use tribunus_compute_core::compute_image::metal_pipeline::compile_metal_source;
    let src = metal_source(h, i, h, "mm");
    let out = match compile_metal_source(label, &src) {
        Some(o) => o,
        None => {
            eprintln!("  Metal compile failed for {}", label);
            return 0.0;
        }
    };
    let device = match metal::Device::system_default() {
        Some(d) => d,
        None => {
            eprintln!("  No Metal device");
            return 0.0;
        }
    };
    let lib = device.new_library_with_data(&out.metallib_bytes).unwrap();
    let func = lib.get_function("mm", None).unwrap();
    let pipeline = device
        .new_compute_pipeline_state_with_function(&func)
        .unwrap();
    let queue = device.new_command_queue();

    let sb = metal::MTLResourceOptions::StorageModeShared;
    let buf_a = device.new_buffer((h as u64 * 2) as u64, sb);
    let buf_w = device.new_buffer((h as u64 * i as u64 * 2) as u64, sb);
    let buf_c = device.new_buffer((i as u64 * 2) as u64, sb);

    for _ in 0..10 {
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&buf_a), 0);
        enc.set_buffer(1, Some(&buf_w), 0);
        enc.set_buffer(2, Some(&buf_c), 0);
        let tg = metal::MTLSize {
            width: 16,
            height: 1,
            depth: 1,
        };
        let gg = metal::MTLSize {
            width: ((i + 15) / 16) as u64,
            height: 1,
            depth: 1,
        };
        enc.dispatch_thread_groups(gg, tg);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    let t0 = Instant::now();
    for _ in 0..200 {
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&buf_a), 0);
        enc.set_buffer(1, Some(&buf_w), 0);
        enc.set_buffer(2, Some(&buf_c), 0);
        let tg = metal::MTLSize {
            width: 16,
            height: 1,
            depth: 1,
        };
        let gg = metal::MTLSize {
            width: ((i + 15) / 16) as u64,
            height: 1,
            depth: 1,
        };
        enc.dispatch_thread_groups(gg, tg);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }
    t0.elapsed().as_nanos() as f64 / 200.0
}

#[cfg(not(feature = "metal-dispatch"))]
fn metal_latency(_h: u32, _i: u32, _label: &str) -> f64 {
    0.0
}
