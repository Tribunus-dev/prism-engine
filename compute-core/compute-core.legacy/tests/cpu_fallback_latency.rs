//! CPU fallback latency benchmark — compares three strategies for MoE
//! expert matmul (GEMV: hidden_dim -> FFN_dim) and routing matmul
//! (hidden_dim -> num_experts).
//!
//! Strategies:
//!   1. Pure scalar — naive dot product in a for loop
//!   2. Accelerate cblas_sgemv — system BLAS GEMV via FFI
//!   3. NEON intrinsics — aarch64 SIMD dot product (vld1q_f32 / vaddvq_f32)
//!
//! Workload: H=2048, I=4096 (expert matmul); 8 experts (router).
//! 10 warmup, 100 measured iterations.
//!
//! Run: cargo test --test cpu_fallback_latency --features prism-backend -- --nocapture
//! Requires: macOS 14.0+ on Apple Silicon

#![cfg(all(target_os = "macos", feature = "prism-backend"))]

use std::time::Instant;

// ── Constants ──────────────────────────────────────────────────────────

const H: usize = 2048; // hidden / input dimension
const I: usize = 4096; // FFN intermediate dimension
const N_EXPERTS: usize = 8;
const WARMUP: usize = 10;
const SAMPLES: usize = 100;

// cblas constants (mirrors accelerate_ffi.rs)
const CBLAS_ROW_MAJOR: i32 = 101;
const CBLAS_NO_TRANS: i32 = 111;

// ── FFI: Accelerate cblas_sgemv ────────────────────────────────────────
//
// cblas_sgemv (RowMajor, NoTrans):
//   y[0..M-1] = alpha * A[MxN] * x[0..N-1] + beta * y[0..M-1]

#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn cblas_sgemv(
        order: i32,
        trans: i32,
        m: i32,
        n: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        x: *const f32,
        incx: i32,
        beta: f32,
        y: *mut f32,
        incy: i32,
    );
}

// ── Deterministic data generation ─────────────────────────────────────

fn make_data(n: usize, seed: u64) -> Vec<f32> {
    use std::hash::{Hash, Hasher};
    (0..n)
        .map(|i| {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            (i as u64 ^ seed).hash(&mut h);
            (h.finish() as f32 % 1000.0 - 500.0) / 500.0
        })
        .collect()
}

// ── Strategy 1: Pure scalar GEMV ──────────────────────────────────────

fn scalar_gemv(a: &[f32], x: &[f32], y: &mut [f32], m: usize, n: usize) {
    for i in 0..m {
        let row = &a[i * n..(i + 1) * n];
        let mut dot = 0.0f32;
        for j in 0..n {
            dot += row[j] * x[j];
        }
        y[i] = dot;
    }
}

// ── Strategy 2: Accelerate cblas_sgemv ─────────────────────────────────

fn cblas_gemv(a: &[f32], x: &[f32], y: &mut [f32], m: usize, n: usize) {
    unsafe {
        cblas_sgemv(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            m as i32,
            n as i32,
            1.0, // alpha
            a.as_ptr(),
            n as i32, // lda
            x.as_ptr(),
            1,   // incx
            0.0, // beta (clear y)
            y.as_mut_ptr(),
            1, // incy
        );
    }
}

// ── Strategy 3: NEON intrinsics GEMV ──────────────────────────────────

fn neon_gemv(a: &[f32], x: &[f32], y: &mut [f32], m: usize, n: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        for i in 0..m {
            let row = &a[i * n..(i + 1) * n];
            let mut acc = [0.0f32; 4];

            let mut j = 0;
            while j + 16 <= n {
                unsafe {
                    let a0 = vld1q_f32(row.as_ptr().add(j));
                    let x0 = vld1q_f32(x.as_ptr().add(j));
                    let m0 = vmulq_f32(a0, x0);

                    let a1 = vld1q_f32(row.as_ptr().add(j + 4));
                    let x1 = vld1q_f32(x.as_ptr().add(j + 4));
                    let m1 = vmulq_f32(a1, x1);

                    let a2 = vld1q_f32(row.as_ptr().add(j + 8));
                    let x2 = vld1q_f32(x.as_ptr().add(j + 8));
                    let m2 = vmulq_f32(a2, x2);

                    let a3 = vld1q_f32(row.as_ptr().add(j + 12));
                    let x3 = vld1q_f32(x.as_ptr().add(j + 12));
                    let m3 = vmulq_f32(a3, x3);

                    acc[0] += vaddvq_f32(m0);
                    acc[1] += vaddvq_f32(m1);
                    acc[2] += vaddvq_f32(m2);
                    acc[3] += vaddvq_f32(m3);
                }
                j += 16;
            }
            while j < n {
                unsafe {
                    let a4 = vld1q_f32(if j + 4 <= n {
                        row.as_ptr().add(j)
                    } else {
                        let mut buf = [0.0f32; 4];
                        let rem = n - j;
                        for k in 0..rem {
                            buf[k] = row[j + k];
                        }
                        buf.as_ptr()
                    });
                    let x4 = vld1q_f32(if j + 4 <= n {
                        x.as_ptr().add(j)
                    } else {
                        let mut buf = [0.0f32; 4];
                        let rem = n - j;
                        for k in 0..rem {
                            buf[k] = x[j + k];
                        }
                        buf.as_ptr()
                    });
                    let m4 = vmulq_f32(a4, x4);
                    acc[0] += vaddvq_f32(m4);
                }
                j += 4;
            }

            y[i] = acc[0] + acc[1] + acc[2] + acc[3];
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_gemv(a, x, y, m, n);
    }
}

// ── Benchmark runner ───────────────────────────────────────────────────

struct BenchResult {
    name: &'static str,
    median_ns: f64,
}

fn bench_gemv<F>(
    name: &'static str,
    a: &[f32],
    x: &[f32],
    y: &mut [f32],
    m: usize,
    n: usize,
    f: F,
) -> BenchResult
where
    F: Fn(&[f32], &[f32], &mut [f32], usize, usize),
{
    for _ in 0..WARMUP {
        f(a, x, y, m, n);
    }

    let mut samples = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let t0 = Instant::now();
        f(a, x, y, m, n);
        let dt = t0.elapsed();
        samples.push(dt.as_nanos() as f64);
    }

    samples.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let median = if SAMPLES % 2 == 0 {
        (samples[SAMPLES / 2 - 1] + samples[SAMPLES / 2]) / 2.0
    } else {
        samples[SAMPLES / 2]
    };

    BenchResult {
        name,
        median_ns: median,
    }
}

fn run_benchmark<F>(
    label: &'static str,
    m: usize,
    n: usize,
    a: &[f32],
    x: &[f32],
    y: &mut [f32],
    f: F,
) -> BenchResult
where
    F: Fn(&[f32], &[f32], &mut [f32], usize, usize),
{
    bench_gemv(label, a, x, y, m, n, f)
}

fn print_results(results: &[BenchResult], title: &str, m: usize, n: usize) {
    let flops = m * n * 2;

    println!();
    println!(
        "=== CPU FALLBACK {} ({}x{} = {} x {} -> {}) ===",
        title, m, n, m, n, m
    );
    println!(
        "  FLOPs per call: {} ({:.0} MFLOPS @ 1us)",
        flops,
        flops as f64 / 1_000_000.0
    );
    println!("  Iterations: {} warmup, {} measured", WARMUP, SAMPLES);
    println!();

    let scalar_median = results[0].median_ns;
    println!(
        "  {:20} {:>12} {:>14}",
        "Strategy", "Median (ns)", "Speedup vs scalar"
    );
    println!("  {}", "-".repeat(50));

    for r in results {
        let speedup = scalar_median / r.median_ns;
        println!("  {:20} {:>12.0} {:>13.2}x", r.name, r.median_ns, speedup);
    }
    println!();

    // Verify speedup table consistency
    let best = results
        .iter()
        .map(|r| r.median_ns)
        .fold(f64::INFINITY, f64::min);
    for r in results {
        if r.median_ns > best * 100.0 {
            eprintln!("  WARNING: {} is >100x slower than best", r.name);
        }
    }
}

// ── Test: Expert matmul (H -> I) ───────────────────────────────────────

#[test]
fn bench_expert_matmul_cpu_fallback() {
    let n_elements_a = I * H; // 4096 x 2048 = 8,388,608
    let n_elements_x = H; // 2048
    let n_elements_y = I; // 4096

    let a = make_data(n_elements_a, 42);
    let x = make_data(n_elements_x, 99);
    let mut y_scalar = vec![0.0f32; n_elements_y];
    let mut y_cblas = vec![0.0f32; n_elements_y];
    let mut y_neon = vec![0.0f32; n_elements_y];

    // ── Correctness check ──────────────────────────────────────────
    scalar_gemv(&a, &x, &mut y_scalar, I, H);

    cblas_gemv(&a, &x, &mut y_cblas, I, H);
    let max_diff_cblas: f32 = y_scalar
        .iter()
        .zip(y_cblas.iter())
        .map(|(s, c)| (s - c).abs())
        .fold(0.0f32, f32::max);

    neon_gemv(&a, &x, &mut y_neon, I, H);
    let max_diff_neon: f32 = y_scalar
        .iter()
        .zip(y_neon.iter())
        .map(|(s, c)| (s - c).abs())
        .fold(0.0f32, f32::max);

    println!(
        "  Max diff vs scalar: cblas={:.2e}, neon={:.2e}",
        max_diff_cblas, max_diff_neon
    );

    y_cblas.fill(0.0);
    y_neon.fill(0.0);

    // ── Benchmark ──────────────────────────────────────────────────
    let results = vec![
        run_benchmark("Pure scalar", I, H, &a, &x, &mut y_scalar, scalar_gemv),
        run_benchmark("cblas_sgemv", I, H, &a, &x, &mut y_cblas, cblas_gemv),
        run_benchmark("NEON intrinsics", I, H, &a, &x, &mut y_neon, neon_gemv),
    ];

    print_results(&results, "EXPERT MATMUL", I, H);
}

// ── Test: Router matmul (H -> N_EXPERTS) ───────────────────────────────

#[test]
fn bench_router_matmul_cpu_fallback() {
    let n_elements_a = N_EXPERTS * H; // 8 x 2048 = 16,384
    let n_elements_x = H; // 2048
    let n_elements_y = N_EXPERTS; // 8

    let a = make_data(n_elements_a, 77);
    let x = make_data(n_elements_x, 88);
    let mut y_scalar = vec![0.0f32; n_elements_y];
    let mut y_cblas = vec![0.0f32; n_elements_y];
    let mut y_neon = vec![0.0f32; n_elements_y];

    // ── Correctness ────────────────────────────────────────────────
    scalar_gemv(&a, &x, &mut y_scalar, N_EXPERTS, H);

    cblas_gemv(&a, &x, &mut y_cblas, N_EXPERTS, H);
    let max_diff_cblas: f32 = y_scalar
        .iter()
        .zip(y_cblas.iter())
        .map(|(s, c)| (s - c).abs())
        .fold(0.0f32, f32::max);

    neon_gemv(&a, &x, &mut y_neon, N_EXPERTS, H);
    let max_diff_neon: f32 = y_scalar
        .iter()
        .zip(y_neon.iter())
        .map(|(s, c)| (s - c).abs())
        .fold(0.0f32, f32::max);

    println!(
        "  Max diff vs scalar: cblas={:.2e}, neon={:.2e}",
        max_diff_cblas, max_diff_neon
    );

    y_cblas.fill(0.0);
    y_neon.fill(0.0);

    // ── Benchmark ──────────────────────────────────────────────────
    let results = vec![
        run_benchmark(
            "Pure scalar",
            N_EXPERTS,
            H,
            &a,
            &x,
            &mut y_scalar,
            scalar_gemv,
        ),
        run_benchmark(
            "cblas_sgemv",
            N_EXPERTS,
            H,
            &a,
            &x,
            &mut y_cblas,
            cblas_gemv,
        ),
        run_benchmark(
            "NEON intrinsics",
            N_EXPERTS,
            H,
            &a,
            &x,
            &mut y_neon,
            neon_gemv,
        ),
    ];

    print_results(&results, "ROUTER MATMUL", N_EXPERTS, H);
}
