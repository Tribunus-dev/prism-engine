//! Cross-lane per-op per-token benchmark.
//! Measures every relevant inference operation across CPU (Core ML cpuOnly)
//! and ANE (Core ML cpuAndNeuralEngine). Identifies crossover points.
//!
//! Run:  cargo test --test cross_lane_ops_benchmark --features prism-backend -- --nocapture
//! Requires: macOS 14.0+ on Apple Silicon

#![cfg(all(target_os = "macos", feature = "prism-backend"))]

use coreml_proto::proto::mil_spec;
use mlx_rs::Dtype;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tribunus_compute_core::arena::Arena;
use tribunus_compute_core::arena::DataType;
use tribunus_compute_core::coreml_bridge::{CoreMlComputeUnits, CoreMlModel};
use tribunus_compute_core::coreml_pipeline::compile_mlpackage;
use tribunus_compute_core::mil_builder::MilBuilder;
use tribunus_compute_core::mlpackage::{write_mlpackage, ModelMeta};

const TD: &str = "/tmp/prism_cross_lane_bench";
const WU: usize = 10;
const SS: usize = 100;

fn md(name: &str) -> PathBuf {
    let p = Path::new(TD).join(name);
    let _ = std::fs::create_dir_all(&p);
    p
}

fn cc(tag: &str, prog: mil_spec::Program, meta: ModelMeta) -> Result<PathBuf, String> {
    let dir = md(tag);
    let pkg = write_mlpackage(prog, &dir, &meta).map_err(|e| format!("pkg: {}", e))?;
    let od = dir.join("compiled");
    let _ = std::fs::create_dir_all(&od);
    let r = compile_mlpackage(&pkg, &od, tag, "cpuAndNeuralEngine", "macOS26")
        .map_err(|e| format!("cmp: {}", e))?;
    Ok(PathBuf::from(&r.compiled_modelc_path))
}

fn ma(d0: u32, d1: u32) -> Arena {
    Arena::new(d0, d1, DataType::Float16).expect("arena")
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

#[derive(Clone)]
struct LR {
    lane: &'static str,
    p50: f64,
    #[allow(dead_code)]
    p95: f64,
    mean: f64,
}

fn bc(
    path: &str,
    cu: CoreMlComputeUnits,
    in_n: &str,
    ia: &Arena,
    out_n: &str,
    oa: &Arena,
    lb: &str,
) -> Result<LR, String> {
    let lane = match cu {
        CoreMlComputeUnits::CpuOnly => "CPU",
        CoreMlComputeUnits::CpuAndNeuralEngine => "ANE",
        _ => "?",
    };
    let m = CoreMlModel::load_with_compute_units(path, cu)
        .map_err(|e| format!("load({} {}): {}", lb, lane, e))?;
    for _ in 0..WU {
        m.predict(in_n, &ia.info, out_n, &oa.info)
            .map_err(|e| format!("warm: {}", e))?;
    }
    let mut s = Vec::with_capacity(SS);
    for _ in 0..SS {
        let t = Instant::now();
        m.predict(in_n, &ia.info, out_n, &oa.info)
            .map_err(|e| format!("run: {}", e))?;
        s.push(t.elapsed().as_nanos() as f64);
    }
    s.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(LR {
        lane,
        p50: s[s.len() / 2],
        p95: s[(s.len() as f64 * 0.95) as usize],
        mean: s.iter().sum::<f64>() / s.len() as f64,
    })
}

fn bo(
    tag: &str,
    prog: mil_spec::Program,
    meta: ModelMeta,
    in_n: &str,
    out_n: &str,
    m: u32,
    n: u32,
) -> Vec<LR> {
    let mut r = Vec::new();
    let mp = match cc(tag, prog, meta) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("  {}: COMPILE FAIL: {}", tag, e);
            return r;
        }
    };
    let ps = mp.to_str().unwrap().to_string();
    let ia = ma(1, m);
    let oa = ma(1, n);
    match bc(&ps, CoreMlComputeUnits::CpuOnly, in_n, &ia, out_n, &oa, tag) {
        Ok(x) => r.push(x),
        Err(e) => {
            eprintln!("  {}: CPU FAIL: {}", tag, e);
        }
    }
    match bc(
        &ps,
        CoreMlComputeUnits::CpuAndNeuralEngine,
        in_n,
        &ia,
        out_n,
        &oa,
        tag,
    ) {
        Ok(x) => r.push(x),
        Err(e) => {
            eprintln!("  {}: ANE FAIL: {}", tag, e);
        }
    }
    r
}

fn pt(rows: &[(&str, u64, u64, Vec<LR>)]) {
    let lanes: Vec<&str> = rows
        .iter()
        .flat_map(|(_, _, _, r)| r.iter().map(|x| x.lane))
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    let mut sorted_lanes: Vec<&&str> = lanes.iter().collect();
    sorted_lanes.sort();
    print!("{:<22} {:>5} {:>5}", "op", "M", "N");
    for l in &sorted_lanes {
        print!(
            "  |  {:>9} {:>9}",
            format!("{}_p50", l),
            format!("{}_avg", l)
        );
    }
    println!("  |  {:>8}", "winner");
    for (op, m, n, res) in rows {
        print!("{:<22} {:>5} {:>5}", op, m, n);
        let mut bm = f64::MAX;
        let mut bl = "?";
        for l in &sorted_lanes {
            if let Some(r) = res.iter().find(|x| x.lane == **l) {
                print!("  |  {:>9.1} {:>9.1}", r.p50 / 1000.0, r.mean / 1000.0);
                if r.mean < bm {
                    bm = r.mean;
                    bl = r.lane;
                }
            } else {
                print!("  |  {:>9} {:>9}", "-", "-");
            }
        }
        let dec = if res.len() >= 2 {
            let mut v: Vec<&LR> = res.iter().collect();
            v.sort_by(|a, b| a.mean.partial_cmp(&b.mean).unwrap());
            let r = v[1].mean / v[0].mean.max(1.0);
            if r > 1.5 {
                format!("{}>>{}", v[0].lane, v[1].lane)
            } else {
                format!("{}~{}", v[0].lane, v[1].lane)
            }
        } else {
            bl.to_string()
        };
        println!("  |  {:>8}", dec);
    }
}

fn bm(m: i64, k: i64, n: i64) -> Result<mil_spec::Program, String> {
    let b = MilBuilder::new("main").input("x", mil_spec::DataType::Float16, &[1, m]);
    let b = b.const_f16("w", &rw(k, n), &[k, n]);
    let wn = b.last_name().ok_or("w")?.to_string();
    let b = b.matmul("x", &wn);
    let on = b.last_name().ok_or("out")?.to_string();
    b.output(&on).build().map_err(|e| format!("MIL: {}", e))
}

fn bmlp(h: i64, i: i64) -> Result<mil_spec::Program, String> {
    let b = MilBuilder::new("main").input("x", mil_spec::DataType::Float16, &[1, h]);
    let b = b.const_f16("wg", &rw(h, i), &[h, i]);
    let wg = b.last_name().ok_or("wg")?.to_string();
    let b = b.matmul("x", &wg);
    let gn = b.last_name().ok_or("g")?.to_string();
    let b = b.const_f16("wu", &rw(h, i), &[h, i]);
    let wu = b.last_name().ok_or("wu")?.to_string();
    let b = b.matmul("x", &wu);
    let un = b.last_name().ok_or("u")?.to_string();
    let b = b.mul(&gn, &un);
    let mn = b.last_name().ok_or("m")?.to_string();
    let b = b.const_f16("wd", &rw(i, h), &[i, h]);
    let wd = b.last_name().ok_or("wd")?.to_string();
    let b = b.matmul(&mn, &wd);
    let on = b.last_name().ok_or("o")?.to_string();
    b.output(&on).build().map_err(|e| format!("MIL: {}", e))
}

fn bew(h: i64, is_mul: bool) -> Result<mil_spec::Program, String> {
    let b = MilBuilder::new("main")
        .input("a", mil_spec::DataType::Float16, &[1, h])
        .input("b", mil_spec::DataType::Float16, &[1, h]);
    let b = if is_mul {
        b.mul("a", "b")
    } else {
        b.add("a", "b")
    };
    let on = b.last_name().ok_or("o")?.to_string();
    b.output(&on).build().map_err(|e| format!("MIL: {}", e))
}

#[test]
fn bench_all() {
    println!("\n=== CROSS-LANE PER-OP LATENCY BENCHMARK ===");
    println!("All times in microseconds. winner = decisive lane (>1.5x faster)");
    println!();

    let mut all: Vec<(&str, u64, u64, Vec<LR>)> = Vec::new();

    // Matmul: projects input [1,M] through weight [K,N] to output [1,N]
    for &(m, k, n, lb) in &[
        (64, 64, 64, "mm_tiny"),
        (256, 256, 256, "mm_small"),
        (512, 512, 512, "mm_med"),
        (1024, 1024, 1024, "mm_large"),
        (4096, 4096, 4096, "mm_xl"),
        (512, 512, 2048, "mm_rect_up"),
        (2048, 2048, 512, "mm_rect_down"),
    ] {
        match bm(m, k, n) {
            Ok(p) => {
                let on = p.functions["main"].block_specializations["CoreML9"].outputs[0].clone();
                let meta = ModelMeta {
                    model_name: lb.into(),
                    function_name: "main".into(),
                    short_description: lb.into(),
                    version: "1.0".into(),
                    author: "p".into(),
                    output_name: on.clone(),
                    inputs: vec![("x".into(), vec![1, m])],
                    outputs: vec![(on.clone(), vec![1, n])],
        
                };
                all.push((
                    lb,
                    m as u64,
                    n as u64,
                    bo(lb, p, meta, "x", &on, m as u32, n as u32),
                ));
            }
            Err(e) => eprintln!("  {}: BUILD FAIL {}", lb, e),
        }
    }

    // MLP: gate+up+silu+down
    for &(h, i, lb) in &[
        (128, 512, "mlp_tiny"),
        (256, 1024, "mlp_small"),
        (512, 2048, "mlp_med"),
        (1024, 4096, "mlp_large"),
    ] {
        match bmlp(h, i) {
            Ok(p) => {
                let on = p.functions["main"].block_specializations["CoreML9"].outputs[0].clone();
                let meta = ModelMeta {
                    model_name: lb.into(),
                    function_name: "main".into(),
                    short_description: lb.into(),
                    version: "1.0".into(),
                    author: "p".into(),
                    output_name: on.clone(),
                    inputs: vec![("x".into(), vec![1, h])],
                    outputs: vec![(on.clone(), vec![1, h])],
        
                };
                all.push((
                    lb,
                    h as u64,
                    i as u64,
                    bo(lb, p, meta, "x", &on, h as u32, h as u32),
                ));
            }
            Err(e) => eprintln!("  {}: BUILD FAIL {}", lb, e),
        }
    }

    // Element-wise: mul and add
    for &(h, lb, is_mul) in &[
        (512, "mul_med", true),
        (4096, "mul_large", true),
        (512, "add_med", false),
        (4096, "add_large", false),
    ] {
        match bew(h, is_mul) {
            Ok(p) => {
                let on = p.functions["main"].block_specializations["CoreML9"].outputs[0].clone();
                let meta = ModelMeta {
                    model_name: lb.into(),
                    function_name: "main".into(),
                    short_description: lb.into(),
                    version: "1.0".into(),
                    author: "p".into(),
                    output_name: on.clone(),
                    inputs: vec![("a".into(), vec![1, h]), ("b".into(), vec![1, h])],
                    outputs: vec![(on.clone(), vec![1, h])],
        
                };
                all.push((
                    lb,
                    h as u64,
                    h as u64,
                    bo(lb, p, meta, "a", &on, h as u32, h as u32),
                ));
            }
            Err(e) => eprintln!("  {}: BUILD FAIL {}", lb, e),
        }
    }

    pt(&all);

    // Crossover analysis
    println!("\n=== CROSSOVER ANALYSIS ===");
    for (op, m, n, res) in &all {
        if res.len() < 2 {
            continue;
        }
        let cpu = res.iter().find(|r| r.lane == "CPU");
        let ane = res.iter().find(|r| r.lane == "ANE");
        match (cpu, ane) {
            (Some(c), Some(a)) => {
                let ratio = c.mean / a.mean.max(1.0);
                let dec = if ratio > 1.5 {
                    "ANE wins"
                } else if ratio < 0.67 {
                    "CPU wins"
                } else {
                    "tie"
                };
                println!("  {:<20} M={:>4} N={:>4} | CPU={:>8.1}us  ANE={:>8.1}us  ratio={:>5.2}x  -> {}",
                    op, m, n, c.mean / 1000.0, a.mean / 1000.0, ratio, dec);
            }
            _ => {}
        }
    }
}
