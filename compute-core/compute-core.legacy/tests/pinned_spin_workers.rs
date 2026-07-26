//! Pinned spin-worker benchmark — compares three CPU-parallelism strategies
//! for the S=512 attention-score matmul (nh×q@K^T).
//!
//! Approaches:
//!   1. Serial — single thread
//!   2. Rayon — parallel chunks via rayon::par_iter
//!   3. Pinned workers — N threads pinned to P-cores via Mach thread_policy_set,
//!      spin-waiting on AtomicU64 work slots
//!
//! Matmul: nh=8 heads, each 128-dim query @ 128×512 K cache = 64K FMAs/head.
//!
//! Run: cargo test --test pinned_spin_workers --features prism-backend -- --nocapture

#![cfg(all(target_os = "macos", feature = "prism-backend"))]

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Instant;

// ── Constants ──────────────────────────────────────────────────────────────

const NH: usize = 8; // Number of attention heads
const HD: usize = 128; // Head dimension
const S: usize = 512; // Sequence length (K cache)
const WARMUP: usize = 100; // Warmup iterations
const ITERS: usize = 1000; // Measured iterations
const NUM_WORKERS: usize = 4; // M1 has 4 P-cores (0-3)
const SHUTDOWN: u64 = u64::MAX;

// ── Mach thread affinity FFI ──────────────────────────────────────────────

/// THREAD_AFFINITY_POLICY tells the scheduler to co-locate threads with the
/// same affinity tag on the same core cluster.  On M1 we assign P-core tags
/// 0-3 to pin workers to individual P-cores.
const THREAD_AFFINITY_POLICY: u32 = 1;

extern "C" {
    fn thread_policy_set(thread: u32, flavor: u32, policy_info: *const u32, count: u32) -> u32;
}

/// Pin the calling thread to a specific P-core via Mach affinity policy.
fn pin_to_p_core(core_id: usize) {
    unsafe {
        let mach_thread = libc::pthread_mach_thread_np(libc::pthread_self());
        let tag = core_id as u32;
        let ret = thread_policy_set(mach_thread, THREAD_AFFINITY_POLICY, &tag as *const u32, 1);
        if ret != 0 {
            // Non-fatal: scheduler may ignore affinity hints on overcommit.
            eprintln!(
                "  [core {}] thread_policy_set returned {} (Mach KERN_*)",
                core_id, ret
            );
        }
    }
}

// ── Work item for pinned workers ──────────────────────────────────────────

/// Describes a slice of attention heads for one worker to process.
///
/// # Safety
/// Pointers must be valid, correctly aligned, and non-overlapping with other
/// workers' output ranges.  Workers only read `queries`/`keys` and write
/// disjoint `scores` slices.
#[repr(C, align(8))]
struct HeadMatmulWork {
    queries: *const f32,
    keys: *const f32,
    scores: *mut f32,
    first_head: usize,
    num_heads: usize,
    seq_len: usize,
}

unsafe impl Send for HeadMatmulWork {}
unsafe impl Sync for HeadMatmulWork {}

/// Execute the assigned heads' score computation: q[h] @ K^T for each head.
///
/// Each head is a dot-product of 128-dim query against 512 key vectors.
unsafe fn execute_head_matmul(work: &HeadMatmulWork) {
    for h in work.first_head..work.first_head + work.num_heads {
        let q_h = std::slice::from_raw_parts(work.queries.add(h * HD), HD);
        let out = std::slice::from_raw_parts_mut(work.scores.add(h * work.seq_len), work.seq_len);
        for s in 0..work.seq_len {
            let k_s = std::slice::from_raw_parts(work.keys.add(s * HD), HD);
            let mut sum = 0.0f32;
            for i in 0..HD {
                sum += q_h[i] * k_s[i];
            }
            out[s] = sum;
        }
    }
}

// ── Pinned worker pool ────────────────────────────────────────────────────

/// A pool of worker threads, each pinned to a specific P-core, that spin-wait
/// on an `AtomicU64` work slot.  The main thread writes a `*const HeadMatmulWork`
/// into the slot to dispatch work; workers signal completion by resetting to 0.
struct PinnedWorkerPool {
    slots: Vec<Arc<AtomicU64>>,
    handles: Vec<Option<JoinHandle<()>>>,
    num_workers: usize,
}

impl PinnedWorkerPool {
    /// Spawn `p_cores.len()` workers, each pinned to the given core id.
    fn new(p_cores: &[usize]) -> Self {
        let num = p_cores.len();
        let mut slots = Vec::with_capacity(num);
        let mut handles = Vec::with_capacity(num);

        for &core in p_cores {
            let slot = Arc::new(AtomicU64::new(0));
            let w = Arc::clone(&slot);
            let h = thread::spawn(move || {
                pin_to_p_core(core);
                loop {
                    let raw = w.load(Ordering::Acquire);
                    if raw == SHUTDOWN {
                        break;
                    }
                    if raw == 0 {
                        std::hint::spin_loop();
                        continue;
                    }
                    // SAFETY: main thread guarantees `raw` points to a valid,
                    // live HeadMatmulWork for the duration of execution.
                    let work = unsafe { &*(raw as *const HeadMatmulWork) };
                    unsafe { execute_head_matmul(work) };
                    w.store(0, Ordering::Release);
                }
            });
            slots.push(slot);
            handles.push(Some(h));
        }

        Self {
            slots,
            handles,
            num_workers: num,
        }
    }

    /// Dispatch work and spin-wait until every worker finishes.
    ///
    /// `work_items` must remain valid (not moved/dropped) until this returns.
    fn run(&self, work_items: &[HeadMatmulWork]) {
        // Release-store: workers must see the pointer before executing.
        for i in 0..self.num_workers {
            let ptr = &work_items[i] as *const HeadMatmulWork as u64;
            self.slots[i].store(ptr, Ordering::Release);
        }
        // Acquire-load: detect when each worker resets its slot to 0.
        for i in 0..self.num_workers {
            while self.slots[i].load(Ordering::Acquire) != 0 {
                std::hint::spin_loop();
            }
        }
    }

    /// Send shutdown signal and join all worker threads.
    fn shutdown(&mut self) {
        for slot in &self.slots {
            slot.store(SHUTDOWN, Ordering::Release);
        }
        for h in self.handles.iter_mut() {
            if let Some(jh) = h.take() {
                let _ = jh.join();
            }
        }
    }
}

impl Drop for PinnedWorkerPool {
    fn drop(&mut self) {
        // Gentle shutdown: signal workers to exit.  If shutdown() was already
        // called the stores are redundant but harmless.
        for slot in &self.slots {
            slot.store(SHUTDOWN, Ordering::Release);
        }
        // Don't join here — the shutdown() method handles that.
    }
}

// ── Matmul implementations ────────────────────────────────────────────────

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

/// Serial: single thread iterates heads sequentially.
fn serial_matmul(query: &[f32], key: &[f32], scores: &mut [f32]) {
    for h in 0..NH {
        let q_h = &query[h * HD..(h + 1) * HD];
        let out = &mut scores[h * S..(h + 1) * S];
        for s in 0..S {
            out[s] = dot(q_h, &key[s * HD..(s + 1) * HD]);
        }
    }
}

/// Rayon: parallel iteration over heads via rayon's work-stealing thread pool.
fn rayon_matmul(query: &[f32], key: &[f32], scores: &mut [f32]) {
    use rayon::prelude::*;
    scores.par_chunks_mut(S).enumerate().for_each(|(h, out)| {
        let q_h = &query[h * HD..(h + 1) * HD];
        for s in 0..S {
            out[s] = dot(q_h, &key[s * HD..(s + 1) * HD]);
        }
    });
}

// ── Benchmark helper ──────────────────────────────────────────────────────

/// Measure median latency of `f` over ITERS iterations (after WARMUP).
/// Returns nanoseconds.
fn bench<F>(mut f: F) -> f64
where
    F: FnMut(),
{
    for _ in 0..WARMUP {
        f();
    }

    let mut samples = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t0 = Instant::now();
        f();
        samples.push(t0.elapsed().as_nanos() as f64);
    }
    samples.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    samples[ITERS / 2]
}

// ── Deterministic data ────────────────────────────────────────────────────

fn generate_data() -> (Vec<f32>, Vec<f32>) {
    use std::hash::{Hash, Hasher};
    let query: Vec<f32> = (0..NH * HD)
        .map(|i| {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            (i as u64).hash(&mut h);
            (h.finish() as f32 % 1000.0 - 500.0) / 500.0
        })
        .collect();
    let key: Vec<f32> = (0..HD * S)
        .map(|i| {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            (i as u64 + 0x1_0000).hash(&mut h);
            (h.finish() as f32 % 1000.0 - 500.0) / 500.0
        })
        .collect();
    (query, key)
}

fn make_data(n: usize, seed: u64) -> Vec<f32> {
    use std::hash::{Hash, Hasher};
    (0..n)
        .map(|i| {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            (i as u64 + seed).hash(&mut h);
            (h.finish() as f32 % 1000.0 - 500.0) / 500.0
        })
        .collect()
}

// ── Test ──────────────────────────────────────────────────────────────────

#[test]
fn test_pinned_spin_workers() {
    println!();
    println!("=== PINNED SPIN WORKERS: CPU PARALLELISM BENCHMARK ===");
    println!("Hardware: Apple M1 (P-cores 0-3, E-cores 4-7)");
    println!(
        "Workload: {} heads \\u00d7 {}d query @ {}d \\u00d7 {} K-cache = {} FMAs",
        NH,
        HD,
        HD,
        S,
        NH * HD * S
    );
    println!(
        "Workers: {} pinned to P-cores 0-{}",
        NUM_WORKERS,
        NUM_WORKERS - 1
    );
    println!("Rayon pool: default (all {} logical cores)", num_cpus());
    println!("Iterations: {} warmup + {} measured", WARMUP, ITERS);
    println!();

    // ── Generate data ────────────────────────────────────────────────
    let (query, key) = generate_data();
    let mut scores_serial = vec![0.0f32; NH * S];
    let mut scores_rayon = vec![0.0f32; NH * S];
    let mut scores_pinned = vec![0.0f32; NH * S];

    // ── Correctness check: all three produce matching results ─────────
    serial_matmul(&query, &key, &mut scores_serial);
    rayon_matmul(&query, &key, &mut scores_rayon);

    let max_err_rayon = scores_serial
        .iter()
        .zip(scores_rayon.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err_rayon < 1e-6,
        "Rayon mismatch vs serial: max_err={}",
        max_err_rayon
    );
    println!(
        "  Correctness: serial vs rayon match (max_err={:.2e})",
        max_err_rayon
    );

    // ── Build pinned worker pool ─────────────────────────────────────
    let p_cores: Vec<usize> = (0..NUM_WORKERS).collect();
    let mut pool = PinnedWorkerPool::new(&p_cores);

    // Distribute heads evenly across workers.
    let heads_per = NH / NUM_WORKERS;
    let extra = NH % NUM_WORKERS;
    let work_items: Vec<HeadMatmulWork> = (0..NUM_WORKERS)
        .map(|i| {
            let start = i * heads_per + extra.min(i);
            let count = heads_per + if i < extra { 1 } else { 0 };
            HeadMatmulWork {
                queries: query.as_ptr(),
                keys: key.as_ptr(),
                scores: scores_pinned.as_mut_ptr(),
                first_head: start,
                num_heads: count,
                seq_len: S,
            }
        })
        .collect();

    // Warmup pinned workers once; verify result.
    pool.run(&work_items);
    let max_err_pinned = scores_serial
        .iter()
        .zip(scores_pinned.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err_pinned < 1e-6,
        "Pinned mismatch vs serial: max_err={}",
        max_err_pinned
    );
    println!(
        "  Correctness: serial vs pinned workers match (max_err={:.2e})",
        max_err_pinned
    );
    println!();

    // ── Benchmark serial ──────────────────────────────────────────────
    let serial_ns = bench(|| {
        serial_matmul(&query, &key, &mut scores_serial);
    });
    println!(
        "  Serial:          {:>9.1} ns  ({:>7.2} us)",
        serial_ns,
        serial_ns / 1000.0
    );

    // ── Benchmark rayon ───────────────────────────────────────────────
    let rayon_ns = bench(|| {
        rayon_matmul(&query, &key, &mut scores_rayon);
    });
    let rayon_speedup = serial_ns / rayon_ns;
    println!(
        "  Rayon:           {:>9.1} ns  ({:>7.2} us)  speedup={:.2}x",
        rayon_ns,
        rayon_ns / 1000.0,
        rayon_speedup
    );

    // ── Benchmark pinned workers ──────────────────────────────────────
    // work_items is already configured — just dispatch each iteration.
    let pinned_ns = bench(|| {
        pool.run(&work_items);
    });
    let pinned_speedup = serial_ns / pinned_ns;
    println!(
        "  Pinned workers:  {:>9.1} ns  ({:>7.2} us)  speedup={:.2}x",
        pinned_ns,
        pinned_ns / 1000.0,
        pinned_speedup
    );

    // ── Shutdown pinned workers ───────────────────────────────────────
    pool.shutdown();

    // ── Summary ───────────────────────────────────────────────────────
    println!();
    println!("=== SUMMARY ===");
    println!("  Serial:          {:>9.1} ns", serial_ns);
    println!(
        "  Rayon:           {:>9.1} ns  speedup={:.2}x vs serial",
        rayon_ns, rayon_speedup
    );
    println!(
        "  Pinned workers:  {:>9.1} ns  speedup={:.2}x vs serial",
        pinned_ns, pinned_speedup
    );

    let ray_vs_pin = if pinned_ns < rayon_ns {
        format!("Pinned workers beat Rayon by {:.2}x", rayon_ns / pinned_ns)
    } else if rayon_ns < pinned_ns {
        format!("Rayon beats Pinned workers by {:.2}x", pinned_ns / rayon_ns)
    } else {
        "Rayon and Pinned workers are tied (within noise)".into()
    };
    println!("  Result: {}", ray_vs_pin);
    println!(
        "  Latency measured as median over {} iterations after {} warmup",
        ITERS, WARMUP
    );
}

/// Number of logical CPUs (for the rayon pool size note).
fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8)
}
/// Sweep over head counts and sequence lengths to measure scaling.
#[test]
fn test_pinned_scaling() {
    println!("\n=== PINNED WORKERS: SCALING SWEEP ===");
    println!("  Workers: {} (M1 P-cores)", NUM_WORKERS);
    println!();
    println!(
        "{:<8} {:>6} {:>6} {:>10} {:>10} {:>10} {:>8} {:>8}",
        "label", "nh", "S", "serial", "rayon", "pinned", "R/S", "P/S"
    );
    println!(
        "{:-<8} {:->6} {:->6} {:->10} {:->10} {:->10} {:->8} {:->8}",
        "", "", "", "", "", "", "", ""
    );

    let sizes: &[(usize, usize, &str)] = &[
        (8, 512, "small"),
        (16, 1024, "med"),
        (32, 2048, "large"),
        (64, 4096, "xl"),
        (128, 8192, "xxl"),
    ];

    for &(nh, seq_len, label) in sizes {
        let query = make_data(nh * HD, 0xCAFE);
        let key = make_data(HD * seq_len, 0xBEEF);
        let mut scores_s = vec![0.0f32; nh * seq_len];
        let mut scores_r = vec![0.0f32; nh * seq_len];
        let mut scores_p = vec![0.0f32; nh * seq_len];

        // Matmul (inline, no global constants)
        let matmul = |q: &[f32], k: &[f32], out: &mut [f32]| {
            for h in 0..nh {
                let qh = &q[h * HD..(h + 1) * HD];
                let o = &mut out[h * seq_len..(h + 1) * seq_len];
                for s in 0..seq_len {
                    let mut sum = 0.0f32;
                    for i in 0..HD {
                        sum += qh[i] * k[s * HD + i];
                    }
                    o[s] = sum;
                }
            }
        };

        let matmul_rayon = |q: &[f32], k: &[f32], out: &mut [f32]| {
            use rayon::prelude::*;
            out.par_chunks_mut(seq_len).enumerate().for_each(|(h, o)| {
                let qh = &q[h * HD..(h + 1) * HD];
                for s in 0..seq_len {
                    let mut sum = 0.0f32;
                    for i in 0..HD {
                        sum += qh[i] * k[s * HD + i];
                    }
                    o[s] = sum;
                }
            });
        };

        // Verify
        matmul(&query, &key, &mut scores_s);
        matmul_rayon(&query, &key, &mut scores_r);
        let max_err = scores_s
            .iter()
            .zip(scores_r.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-6, "mismatch");

        // Pinned workers
        let p_cores: Vec<usize> = (0..NUM_WORKERS.min(nh)).collect();
        let mut pool = PinnedWorkerPool::new(&p_cores);
        let h_per = nh / p_cores.len();
        let extra = nh % p_cores.len();
        let items: Vec<HeadMatmulWork> = (0..p_cores.len())
            .map(|i| {
                let start = i * h_per + extra.min(i);
                let count = h_per + if i < extra { 1 } else { 0 };
                HeadMatmulWork {
                    queries: query.as_ptr(),
                    keys: key.as_ptr(),
                    scores: scores_p.as_mut_ptr(),
                    first_head: start,
                    num_heads: count,
                    seq_len,
                }
            })
            .collect();
        pool.run(&items);
        let max_err_p = scores_s
            .iter()
            .zip(scores_p.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err_p < 1e-6, "pinned mismatch");

        // Warmup
        for _ in 0..10 {
            matmul(&query, &key, &mut scores_s);
        }

        // Benchmark
        let bench = |f: &mut dyn FnMut()| -> f64 {
            let t0 = Instant::now();
            for _ in 0..10 {
                f();
            }
            t0.elapsed().as_nanos() as f64 / 10.0
        };
        let s_ns = bench(&mut || matmul(&query, &key, &mut scores_s));
        let r_ns = bench(&mut || matmul_rayon(&query, &key, &mut scores_r));
        let p_ns = bench(&mut || pool.run(&items));

        pool.shutdown();

        println!(
            "{:<8} {:>6} {:>6} {:>10.0} {:>10.0} {:>10.0} {:>7.2}x {:>7.2}x{}",
            label,
            nh,
            seq_len,
            s_ns,
            r_ns,
            p_ns,
            s_ns / r_ns,
            s_ns / p_ns,
            if p_ns < r_ns { "*" } else { "" }
        );
    }
    println!();
    println!("  *=pinned beats rayon  *=advantage grows with size when true");
}
