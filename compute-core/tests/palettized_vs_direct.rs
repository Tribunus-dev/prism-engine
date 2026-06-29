//! Palettized weights vs direct FP16 — measures artifact size, load time,
//! RSS delta, and per-token latency for both weight formats on the same MLP.
//! Tests whether constexpr_lut_to_dense saves runtime memory or only disk space.
//!
//! Run: cargo test --test palettized_vs_direct --features prism-backend -- --nocapture
//! Requires: macOS 14.0+ on Apple Silicon

#![cfg(all(target_os = "macos", feature = "prism-backend"))]

use std::path::{Path, PathBuf};
use std::time::Instant;

use coreml_proto::proto::mil_spec;
use mlx_rs::Dtype;
use tribunus_compute_core::arena::Arena;
use tribunus_compute_core::coreml_bridge::CoreMlModel;
use tribunus_compute_core::coreml_pipeline::compile_mlpackage;
use tribunus_compute_core::mil_builder::MilBuilder;
use tribunus_compute_core::mlpackage::{write_mlpackage, ModelMeta};

const TD: &str = "/tmp/prism_pal_bench";
fn md(n: &str) -> PathBuf {
    let p = Path::new(TD).join(n);
    let _ = std::fs::create_dir_all(&p);
    p
}
fn ma(d0: u32, d1: u32) -> Arena {
    Arena::new(d0, d1, Dtype::Float16).expect("a")
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

fn pr(model: &CoreMlModel, in_n: &str, ia: &Arena, out_n: &str, oa: &Arena) {
    model.predict(in_n, &ia.info, out_n, &oa.info).unwrap()
}

fn rss() -> u64 {
    unsafe {
        let mut info = std::mem::zeroed::<libc::rusage>();
        if libc::getrusage(libc::RUSAGE_SELF, &mut info) == 0 {
            return info.ru_maxrss as u64;
        }
    }
    0
}

fn dir_size(path: &Path) -> u64 {
    let mut t = 0u64;
    if let Ok(e) = std::fs::read_dir(path) {
        for en in e.flatten() {
            let p = en.path();
            if p.is_dir() {
                t += dir_size(&p);
            } else if let Ok(m) = std::fs::metadata(&p) {
                t += m.len();
            }
        }
    }
    t
}

// ── Build MLP with const_f16 (direct FP16 weights) ───────────────────────

fn build_mlp_direct(h: i64, i: i64) -> mil_spec::Program {
    let b = MilBuilder::new("main").input("x", mil_spec::DataType::Float16, &[1, h]);
    let b = b.const_f16("wg", &rw(h, i), &[h, i]);
    let wg = b.last_name().unwrap().to_string();
    let b = b.matmul("x", &wg);
    let gn = b.last_name().unwrap().to_string();
    let b = b.const_f16("wu", &rw(h, i), &[h, i]);
    let wu = b.last_name().unwrap().to_string();
    let b = b.matmul("x", &wu);
    let un = b.last_name().unwrap().to_string();
    let b = b.mul(&gn, &un);
    let mn = b.last_name().unwrap().to_string();
    let b = b.const_f16("wd", &rw(i, h), &[i, h]);
    let wd = b.last_name().unwrap().to_string();
    let b = b.matmul(&mn, &wd);
    let on = b.last_name().unwrap().to_string();
    b.output(&on).build().unwrap()
}

// ── Build MLP with constexpr_lut_to_dense (palettized) ────────────────────
// Uses const_uint8 for indices and const_f16 for codebook, then applies
// constexpr_lut_to_dense to decompress at load time.

fn palettize(weights: &[f32], out_dim: usize, in_dim: usize) -> (Vec<f32>, Vec<u8>) {
    // Per-output-channel 256-entry codebook + 8-bit indices (one per element)
    let k = 256;
    let mut codebook = Vec::with_capacity(out_dim * k);
    let mut indices = vec![0u8; out_dim * in_dim];
    for row in 0..out_dim {
        let start = row * in_dim;
        // Codebook: sample k evenly-spaced values
        let mut cb: Vec<f32> = (0..k)
            .map(|i| weights[start + ((i * in_dim) / k).min(in_dim - 1)])
            .collect();
        cb.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        codebook.extend_from_slice(&cb);
        // Encode: nearest centroid
        for col in 0..in_dim {
            let val = weights[start + col];
            let best = (0..k)
                .min_by(|&i, &j| {
                    (cb[i] - val)
                        .abs()
                        .partial_cmp(&(cb[j] - val).abs())
                        .unwrap()
                })
                .unwrap_or(0);
            indices[row * in_dim + col] = best as u8;
        }
    }
    (codebook, indices)
}

fn build_mlp_palettized(h: i64, i: i64) -> mil_spec::Program {
    let gate_w = rw(h, i);
    let (gate_cb, gate_idx) = palettize(&gate_w, i as usize, h as usize);
    let up_w = rw(h, i);
    let (up_cb, up_idx) = palettize(&up_w, i as usize, h as usize);
    let down_w = rw(i, h);
    let (down_cb, down_idx) = palettize(&down_w, h as usize, i as usize);

    let b = MilBuilder::new("main").input("x", mil_spec::DataType::Float16, &[1, h]);
    // Gate: indices + codebook -> lut -> matmul
    // gate weight: [h, i] so matmul("x", gw) = [1,h] @ [h,i] = [1,i]
    let b = b.const_uint8("gi", &gate_idx, &[i as i64, h as i64]);
    let gi = b.last_name().unwrap().to_string();
    let b = b.const_f16("gc", &gate_cb, &[i as i64, 1, 256, 1]);
    let gc = b.last_name().unwrap().to_string();
    let b = b.constexpr_lut_to_dense("gw", &gi, &gc, &[h as i64, i as i64], 1);
    let gw = b.last_name().unwrap().to_string();
    let b = b.matmul("x", &gw);
    let gn = b.last_name().unwrap().to_string();
    // Up
    // up weight: [h, i]
    let b = b.const_uint8("ui", &up_idx, &[i as i64, h as i64]);
    let ui = b.last_name().unwrap().to_string();
    let b = b.const_f16("uc", &up_cb, &[i as i64, 1, 256, 1]);
    let uc = b.last_name().unwrap().to_string();
    let b = b.constexpr_lut_to_dense("uw", &ui, &uc, &[h as i64, i as i64], 1);
    let uw = b.last_name().unwrap().to_string();
    let b = b.matmul("x", &uw);
    let un = b.last_name().unwrap().to_string();
    let b = b.mul(&gn, &un);
    let mn = b.last_name().unwrap().to_string();
    // Down
    // down weight: [i, h] so matmul(mul, dw) = [1,i] @ [i,h] = [1,h]
    let b = b.const_uint8("di", &down_idx, &[h as i64, i as i64]);
    let di = b.last_name().unwrap().to_string();
    let b = b.const_f16("dc", &down_cb, &[h as i64, 1, 256, 1]);
    let dc = b.last_name().unwrap().to_string();
    let b = b.constexpr_lut_to_dense("dw", &di, &dc, &[i as i64, h as i64], 1);
    let dw = b.last_name().unwrap().to_string();
    let b = b.matmul(&mn, &dw);
    let on = b.last_name().unwrap().to_string();
    b.output(&on).build().unwrap()
}

fn measure(tag: &str, prog: mil_spec::Program, h: i64, _i: i64) {
    let on = prog.functions["main"].block_specializations["CoreML9"].outputs[0].clone();
    let meta = ModelMeta {
        model_name: tag.into(),
        function_name: "main".into(),
        short_description: tag.into(),
        version: "1.0".into(),
        author: "p".into(),
        output_name: on.clone(),
        inputs: vec![("x".into(), vec![1, h])],
        outputs: vec![(on.clone(), vec![1, h])],
        spec_version: 9,
    };
    let mp = cc(tag, prog, meta);
    let art = dir_size(&mp);
    let r0 = rss();
    let t0 = Instant::now();
    let model = CoreMlModel::load(mp.to_str().unwrap()).unwrap();
    let load_ns = t0.elapsed().as_nanos();
    let r1 = rss();
    let fp = r1.saturating_sub(r0);

    let ia = ma(1, h as u32);
    let oa = ma(1, h as u32);
    for _ in 0..10 {
        pr(&model, "x", &ia, &on, &oa);
    }
    let mut lats = Vec::with_capacity(200);
    for _ in 0..200 {
        let t = Instant::now();
        pr(&model, "x", &ia, &on, &oa);
        lats.push(t.elapsed().as_nanos());
    }
    lats.sort_unstable();
    let p50 = lats[lats.len() / 2];
    let mean = lats.iter().sum::<u128>() / lats.len() as u128;
    println!(
        "  {:<10} artifact={:>6.1}MB  load={:>6.1}ms  RSS={:>5.1}MB  p50={:>7.1}us  mean={:>7.1}us",
        tag,
        art as f64 / (1024. * 1024.),
        load_ns as f64 / 1e6,
        fp as f64 / (1024. * 1024.),
        p50 as f64 / 1000.,
        mean as f64 / 1000.
    );
}

#[test]
fn test_palettized_vs_direct() {
    println!("\n=== PALETTIZED vs DIRECT FP16 WEIGHTS ===");
    println!("MLP block (gate+up+silu+down) on cpuAndNeuralEngine");
    println!();
    for &(h, i, lb) in &[
        (512i64, 2048i64, "H512I2048"),
        (1024i64, 4096i64, "H1024I4096"),
    ] {
        println!("--- {} ---", lb);
        let dp = build_mlp_direct(h, i);
        measure(&format!("direct_{}", lb), dp, h, i);
        let pp = build_mlp_palettized(h, i);
        measure(&format!("palet_{}", lb), pp, h, i);
        println!();
    }
    println!("=== INTERPRETATION ===");
    println!(
        "- If palet artifact < direct artifact: palettization saves disk space (expected ~4x)"
    );
    println!("- If palet RSS ~= direct RSS: constexpr_lut_to_dense decompresses at load time,");
    println!("  so runtime memory is THE SAME. Palettization saves disk only.");
    println!("- If palet RSS < direct RSS: ANE keeps palette compressed in SRAM");
    println!("  (would contradict Orion's finding)");
    println!("- If palet p50 < direct p50: decompressed weights compute faster (bandwidth win)");
}
