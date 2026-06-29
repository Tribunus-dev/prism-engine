//! In-band IOSurface spinlock — embeds synchronization flag directly in shared
//! IOSurface memory. Compares three mechanisms: ideal (max of two lanes),
//! AtomicBool (Rust std), and in-band (volatile load/store in IOSurface).
//!
//! Run: cargo test --test inband_iosurface_spinlock --features prism-backend -- --nocapture

#![cfg(all(target_os = "macos", feature = "prism-backend"))]

use coreml_proto::proto::mil_spec;
use mlx_rs::Dtype;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tribunus_compute_core::arena::Arena;
use tribunus_compute_core::arena::DataType;
use tribunus_compute_core::arena_info::ArenaInfo;
use tribunus_compute_core::coreml_bridge::{CoreMlComputeUnits, CoreMlModel};
use tribunus_compute_core::coreml_pipeline::compile_mlpackage;
use tribunus_compute_core::mil_builder::MilBuilder;
use tribunus_compute_core::mlpackage::{write_mlpackage, ModelMeta};

const TD: &str = "/tmp/prism_spin";
fn md(n: &str) -> PathBuf {
    let p = Path::new(TD).join(n);
    let _ = std::fs::create_dir_all(&p);
    p
}
fn ma(d0: u32, d1: u32) -> Arena {
    Arena::new(d0, d1, DataType::Float16).expect("a")
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
fn pr(model: &CoreMlModel, in_n: &str, ia: &ArenaInfo, out_n: &str, oa: &ArenaInfo) {
    model.predict(in_n, ia, out_n, oa).unwrap()
}

// ── Shared state wrapper (unsafe Sync because atomic flags prevent concurrent access) ──

struct ArenaPair {
    #[allow(dead_code)]
    data_bytes: usize,
    flag_ptr: *mut u64,
    data_info: ArenaInfo,
}
unsafe impl Send for ArenaPair {}
unsafe impl Sync for ArenaPair {}

fn make_pair(i: u32) -> ArenaPair {
    let data_bytes = i as usize * 2; // i FP16 elements = shape [1, i]
    let alignment: usize = 16384; // 16KB page for M1 ANE DMA
                                  // Align data_bytes up to 16KB then add room for the 64-bit flag at the tail
    let padded = ((data_bytes + alignment - 1) / alignment) * alignment + 64;
    let wide = Arena::new_bytes(padded as u32).expect("pair arena");
    // Flag goes AFTER the tensor data (tail of the IOSurface)
    let flag_ptr = unsafe { (wide.info.base_address as *mut u8).add(data_bytes) as *mut u64 };
    let mut data_info = wide.info.clone();
    // Keep base_address at the page-aligned IOSurface start — no shift
    data_info.logical_dim0 = 1;
    data_info.logical_dim1 = i as i32; // matches model shape [1, i]
    data_info.byte_size = data_bytes as i32;
    // Write initial flag value: 0 = producer may write
    unsafe {
        std::ptr::write(flag_ptr, 0u64);
    }
    // Leak the arena so the pointer stays valid across threads.
    // The memory is freed by the OS at process exit.
    let _ = Box::into_raw(Box::new(wide));
    ArenaPair {
        data_bytes,
        flag_ptr,
        data_info,
    }
}

#[allow(dead_code)]
fn bench_ideal(h: u32, i: u32, _items: usize) -> f64 {
    let (pp, po) = mdl(h as i64, i as i64, "idp");
    let (cp, co) = mdl(i as i64, h as i64, "idc");
    let pm =
        CoreMlModel::load_with_compute_units(pp.to_str().unwrap(), CoreMlComputeUnits::CpuOnly)
            .unwrap();
    let cm = CoreMlModel::load_with_compute_units(
        cp.to_str().unwrap(),
        CoreMlComputeUnits::CpuAndNeuralEngine,
    )
    .unwrap();
    let ia = ma(1, h);
    let oa = ma(1, h);
    let ra = ma(1, i);
    for _ in 0..10 {
        pr(&pm, "x", &ia.info, &po, &ra.info);
        pr(&cm, "x", &ra.info, &co, &oa.info);
    }
    let t0 = Instant::now();
    for _ in 0..200 {
        pr(&pm, "x", &ia.info, &po, &ra.info);
        pr(&cm, "x", &ra.info, &co, &oa.info);
    }
    t0.elapsed().as_nanos() as f64 / 200.0
}

fn bench_ideal_concurrent(h: u32, i: u32, items: usize) -> f64 {
    let cpu = {
        let (pp, po) = mdl(h as i64, i as i64, "icp");
        let m =
            CoreMlModel::load_with_compute_units(pp.to_str().unwrap(), CoreMlComputeUnits::CpuOnly)
                .unwrap();
        let ia = ma(1, h);
        let oa = ma(1, i);
        for _ in 0..10 {
            pr(&m, "x", &ia.info, &po, &oa.info);
        }
        let t0 = Instant::now();
        for _ in 0..items {
            pr(&m, "x", &ia.info, &po, &oa.info);
        }
        t0.elapsed().as_nanos() as f64 / items as f64
    };
    let ane = {
        let (cp, co) = mdl(i as i64, h as i64, "icc");
        let m = CoreMlModel::load_with_compute_units(
            cp.to_str().unwrap(),
            CoreMlComputeUnits::CpuAndNeuralEngine,
        )
        .unwrap();
        let ia = ma(1, i);
        let oa = ma(1, h);
        for _ in 0..10 {
            pr(&m, "x", &ia.info, &co, &oa.info);
        }
        let t0 = Instant::now();
        for _ in 0..items {
            pr(&m, "x", &ia.info, &co, &oa.info);
        }
        t0.elapsed().as_nanos() as f64 / items as f64
    };
    cpu.max(ane)
}

fn bench_sync_atomic(h: u32, i: u32, items: usize) -> f64 {
    let (pp, po) = mdl(h as i64, i as i64, "sap");
    let (cp, co) = mdl(i as i64, h as i64, "sac");
    let pm =
        CoreMlModel::load_with_compute_units(pp.to_str().unwrap(), CoreMlComputeUnits::CpuOnly)
            .unwrap();
    let cm = CoreMlModel::load_with_compute_units(
        cp.to_str().unwrap(),
        CoreMlComputeUnits::CpuAndNeuralEngine,
    )
    .unwrap();
    let ia = ma(1, h);
    let oa = ma(1, h);
    // For AtomicBool: share the predict arena via Arc. Producer owns it.
    let ring_arena = Arc::new(ma(1, i));
    let flag = Arc::new(AtomicBool::new(true)); // true = producer may write
    let (ra, f) = (ring_arena.clone(), flag.clone());
    let ph = std::thread::spawn(move || {
        for _ in 0..items {
            while !f.load(Ordering::Acquire) {
                std::hint::spin_loop();
            }
            pr(&pm, "x", &ia.info, &po, &ra.info);
            f.store(false, Ordering::Release);
        }
    });
    let (ra2, f2) = (ring_arena, flag);
    let ch = std::thread::spawn(move || {
        for _ in 0..items {
            while f2.load(Ordering::Acquire) {
                std::hint::spin_loop();
            }
            pr(&cm, "x", &ra2.info, &co, &oa.info);
            f2.store(true, Ordering::Release);
        }
    });
    let t0 = Instant::now();
    ph.join().unwrap();
    ch.join().unwrap();
    t0.elapsed().as_nanos() as f64 / items as f64
}

fn bench_sync_inband(h: u32, i: u32, items: usize) -> f64 {
    let (pp, po) = mdl(h as i64, i as i64, "sip");
    let (cp, co) = mdl(i as i64, h as i64, "sic");
    let pm =
        CoreMlModel::load_with_compute_units(pp.to_str().unwrap(), CoreMlComputeUnits::CpuOnly)
            .unwrap();
    let cm = CoreMlModel::load_with_compute_units(
        cp.to_str().unwrap(),
        CoreMlComputeUnits::CpuAndNeuralEngine,
    )
    .unwrap();
    let ia = ma(1, h);
    let oa = ma(1, h);
    let pair = Arc::new(make_pair(i));

    let (p1, _f1) = (pair.clone(), &pair.flag_ptr);
    let ph = std::thread::spawn(move || {
        for _ in 0..items {
            unsafe {
                while (*p1.flag_ptr) != 0 {
                    std::hint::spin_loop();
                }
            }
            pr(&pm, "x", &ia.info, &po, &p1.data_info);
            unsafe {
                (*p1.flag_ptr) = 1;
            }
        }
    });
    let (p2, _f2) = (pair.clone(), &pair.flag_ptr);
    let ch = std::thread::spawn(move || {
        for _ in 0..items {
            unsafe {
                while (*p2.flag_ptr) != 1 {
                    std::hint::spin_loop();
                }
            }
            pr(&cm, "x", &p2.data_info, &co, &oa.info);
            unsafe {
                (*p2.flag_ptr) = 0;
            }
        }
    });
    let t0 = Instant::now();
    ph.join().unwrap();
    ch.join().unwrap();
    t0.elapsed().as_nanos() as f64 / items as f64
}

#[test]
fn test_spinlock() {
    println!("\n=== IN-BAND IOSURFACE SPINLOCK vs ATOMICBOOL ===");
    let sizes = [(256u32, 1024u32), (512u32, 2048u32)];
    for &(h, i) in &sizes {
        let ideal = bench_ideal_concurrent(h, i, 200);
        let ab = bench_sync_atomic(h, i, 100);
        let ib = bench_sync_inband(h, i, 100);
        println!("  H={} I={}:  ideal={:>7.1}us  AtomicBool={:>7.1}us (+{:>5.1}us)  in-band={:>7.1}us (+{:>5.1}us)",
            h, i, ideal/1000.0, ab/1000.0, (ab-ideal)/1000.0, ib/1000.0, (ib-ideal)/1000.0);
    }
    println!("\n  in-band < AtomicBool = in-IOSurface flag wins (avoids cache-coherency)");
    println!("  in-band == AtomicBool = flag location irrelevant, predict latency dominates");
}
