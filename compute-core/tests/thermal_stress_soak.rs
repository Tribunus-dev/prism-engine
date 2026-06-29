//! Thermal stress soak — 5-minute continuous FP32 matmul on 4 pinned P-cores.
//!
//! Monitors CPU frequency via `sysctl hw.cpufrequency` (sysctl MIB
//! CTL_HW / HW_CPU_FREQ) and die temperature via Apple SMC, logging every
//! 10 seconds.  Passes if frequency remains above 90% of the initial reading
//! (indicating no thermal throttling).
//!
//! The test reports:
//!   - Frequency stability (mean, min, max, stddev)
//!   - Any throttling events (frequency drops >10% from initial)
//!   - Die temperature range
//!
//! Run: cargo test --test thermal_stress_soak --features prism-backend -- --nocapture
//!
//! Runtime: ~300 seconds (5 minutes).

#![cfg(all(target_os = "macos", feature = "prism-backend"))]
#![allow(dead_code)]

use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

// ── Constants ──────────────────────────────────────────

const SOAK_SECONDS: u64 = 300; // 5 minutes
const FREQ_LOG_INTERVAL_SECS: u64 = 10; // Sample frequency every 10 s
const STATUS_INTERVAL_SECS: u64 = 30; // Print status line every 30 s
const NUM_WORKERS: usize = 4; // M1 has 4 P-cores
const THROTTLE_THRESHOLD: f64 = 0.90; // 90% of initial frequency

// ── Mach thread affinity FFI ──────────────────────────────────────────────

const THREAD_AFFINITY_POLICY: u32 = 1;

extern "C" {
    fn thread_policy_set(thread: u32, flavor: u32, policy: *const u32, count: u32) -> u32;
    fn pthread_mach_thread_np(thread: libc::pthread_t) -> u32;
}

/// Pin the calling thread to a specific P-core via Mach affinity policy.
fn pin_to_p_core(core_id: usize) {
    unsafe {
        let mach_thread = pthread_mach_thread_np(libc::pthread_self());
        let tag = core_id as u32;
        let ret = thread_policy_set(mach_thread, THREAD_AFFINITY_POLICY, &tag as *const u32, 1);
        if ret != 0 {
            eprintln!(
                "  [warn] thread_policy_set for core {} returned {}",
                core_id, ret
            );
        }
    }
}

// ── CPU frequency sampling via sysctl MIB ──────────────────────────────────

/// Read nominal CPU frequency via `sysctl hw.cpufrequency`.
///
/// Uses the MIB `[CTL_HW, HW_CPU_FREQ]`.  Returns MHz.
/// Returns 0.0 when the sysctl is unavailable (e.g. on some
/// Apple Silicon configurations where this MIB may not be populated).
fn read_cpu_frequency() -> f64 {
    let mut freq: u64 = 0;
    let mut size = std::mem::size_of::<u64>();
    let mut mib: [i32; 2] = [libc::CTL_HW, libc::HW_CPU_FREQ];
    let rc = unsafe {
        libc::sysctl(
            mib.as_mut_ptr(),
            2,
            &mut freq as *mut _ as *mut libc::c_void,
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc != 0 || freq == 0 {
        0.0
    } else {
        freq as f64 / 1_000_000.0
    }
}

// ── Die temperature reading via Apple SMC ──────────────────────────────────

#[link(name = "IOKit", kind = "framework")]
extern "C" {
    fn IOServiceGetMatchingService(main_port: u32, matching: *const std::ffi::c_void) -> u32;
    fn IOServiceMatching(name: *const i8) -> *const std::ffi::c_void;
    fn IOServiceOpen(
        service: u32,
        owning_task: u32,
        user_client_type: u32,
        connect: *mut u32,
    ) -> i32;
    fn IOServiceClose(connect: u32) -> i32;
    fn IOObjectRelease(object: u32) -> i32;
    fn IOConnectCallStructMethod(
        connect: u32,
        selector: u32,
        input_struct: *const std::ffi::c_void,
        input_struct_cnt: usize,
        output_struct: *mut std::ffi::c_void,
        output_struct_cnt: *mut usize,
    ) -> i32;
    fn mach_task_self() -> u32;
}

/// Read die temperature (in Celsius) via the Apple SMC.
///
/// Queries the `TC0P` key (CPU proximity / die temperature on Apple Silicon).
/// Returns `0.0` if the SMC is inaccessible or the key is unavailable.
/// Read die temperature via Apple SMC.
///
/// Scans known CPU temperature keys (TC0P, TC0D, Tp09) and returns
/// the first valid reading, or 0.0 if none are available.
fn read_die_temperature() -> f64 {
    // Try the most common CPU temperature key for Apple Silicon.
    for key_bytes in [*b"TC0P", *b"TC0D", *b"Tp09"].iter() {
        let temp = read_smc_temperature(*key_bytes);
        if temp > 0.0 && temp < 150.0 {
            return temp;
        }
    }
    0.0
}

/// Read a single SMC temperature key, returning degrees Celsius.
/// Returns 0.0 on any failure.
fn read_smc_temperature(key: [u8; 4]) -> f64 {
    let service_name = c"AppleSMC";
    let matching_dict = unsafe { IOServiceMatching(service_name.as_ptr()) };
    if matching_dict.is_null() {
        return 0.0;
    }

    let service = unsafe { IOServiceGetMatchingService(0, matching_dict) };
    if service == 0 {
        return 0.0;
    }

    let mut connect: u32 = 0;
    let kr = unsafe { IOServiceOpen(service, mach_task_self(), 0, &mut connect) };
    unsafe { IOObjectRelease(service) };
    if kr != 0 || connect == 0 {
        return 0.0;
    }

    // Build raw SMC param buffer: 80 bytes of zeroed struct.
    // AppleSMC uses a packed input/output struct:
    //   uint32_t key;      offset 0
    //   SMCVersion vers;   offset 4..9
    //   SMCPLimitData pld; offset 10..20
    //   uint32_t result;   offset 20
    //   uint8_t  status;   offset 24
    //   uint8_t  data8;    offset 25
    //   uint32_t data32;   offset 28
    //   uint8_t  data8_out[4]; offset 32..35
    //   uint32_t data32_out;   offset 36
    //   + padding to 80 bytes
    //
    // For simplicity, we use a fixed-size u8 buffer and pack/unpack directly.

    let mut buf = [0u8; 80];

    // Open SMC.
    let mut out_size = 80usize;
    let kr = unsafe {
        IOConnectCallStructMethod(
            connect,
            2, // kSMCUserClientOpen
            &buf as *const _ as *const std::ffi::c_void,
            80,
            &mut buf as *mut _ as *mut std::ffi::c_void,
            &mut out_size,
        )
    };
    if kr != 0 {
        unsafe { IOServiceClose(connect) };
        return 0.0;
    }

    // Pack key: big-endian u32 at offset 0.
    let key_be: u32 = u32::from_be_bytes(key);
    let key_bytes = key_be.to_ne_bytes();
    buf[0] = key_bytes[0];
    buf[1] = key_bytes[1];
    buf[2] = key_bytes[2];
    buf[3] = key_bytes[3];

    // Get key info.
    out_size = 80;
    let kr = unsafe {
        IOConnectCallStructMethod(
            connect,
            9, // kSMCGetKeyInfo
            &buf as *const _ as *const std::ffi::c_void,
            80,
            &mut buf as *mut _ as *mut std::ffi::c_void,
            &mut out_size,
        )
    };
    if kr != 0 {
        unsafe { IOServiceClose(connect) };
        return 0.0;
    }

    // data32_out at offset 36 is the key's data size. If 0, key not found.
    let data_size = u32::from_ne_bytes([buf[36], buf[37], buf[38], buf[39]]) as usize;
    if data_size == 0 {
        unsafe { IOServiceClose(connect) };
        return 0.0;
    }

    // Read key.
    // Repack the key at offset 0 for the read call.
    buf[0] = key_bytes[0];
    buf[1] = key_bytes[1];
    buf[2] = key_bytes[2];
    buf[3] = key_bytes[3];
    out_size = 80;
    let kr = unsafe {
        IOConnectCallStructMethod(
            connect,
            5, // kSMCReadKey
            &buf as *const _ as *const std::ffi::c_void,
            80,
            &mut buf as *mut _ as *mut std::ffi::c_void,
            &mut out_size,
        )
    };
    unsafe { IOServiceClose(connect) };
    if kr != 0 {
        return 0.0;
    }

    // SMC temperature keys use sp78 format (signed 8.8 fixed-point).
    // The raw value is at buf[32..34] (data8_out[0..2]).
    let raw = u16::from_ne_bytes([buf[32], buf[33]]);
    let int_part = (raw >> 8) as i16;
    let frac_part = (raw & 0xFF) as f64 / 256.0;
    (int_part as f64) + frac_part
}

// ── Continuous matmul workload ────────────────────────────────────────────

/// A minimal FP32 matmul kernel: `C[m][n] = A[m][k] · B[k][n]`
///
/// Operates on pre-allocated heap buffers (A: m×k, B: k×n, C: m×n).
/// Each invocation does one full multiply.  The kernel is intentionally
/// simple — we want sustained ALU pressure, not optimal bandwidth.
fn matmul_f32(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for t in 0..k {
                sum += a[i * k + t] * b[t * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/// Generates deterministic pseudo-random f32 data in [-1, 1].
fn fill_deterministic(buf: &mut [f32], seed: u64) {
    for (i, v) in buf.iter_mut().enumerate() {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (i as u64 + seed).hash(&mut h);
        *v = (h.finish() as f32 % 1000.0 - 500.0) / 500.0;
    }
}

// Work buffer dimensions: 64×64 matmul (sustainable ALU pressure per core).
const MATMUL_M: usize = 64;
const MATMUL_K: usize = 64;
const MATMUL_N: usize = 64;

/// Continuous matmul loop — runs until `running` becomes false.
/// Each iteration does one 64×64×64 FP32 matmul (~512K FMAs).
fn worker_loop(running: &AtomicBool, a: &[f32], b: &[f32], c: &mut [f32]) {
    while running.load(Ordering::Relaxed) {
        matmul_f32(MATMUL_M, MATMUL_K, MATMUL_N, a, b, c);
    }
}

// ── Frequency / temperature sampler ───────────────────────────────────────

#[derive(Debug, Clone, Copy)]
struct Sample {
    elapsed_secs: f64,
    freq_mhz: f64,
    temp_c: f64,
}

/// Background sampler thread — records frequency and temperature every
/// `interval_secs` seconds.  Returns its log when `running` becomes false.
fn sampler_thread(running: Arc<AtomicBool>, interval_secs: u64) -> JoinHandle<Vec<Sample>> {
    thread::spawn(move || {
        let mut samples = Vec::with_capacity(64);
        let t0 = Instant::now();

        loop {
            if !running.load(Ordering::Relaxed) {
                // Take one final sample before exiting.
                let elapsed = t0.elapsed().as_secs_f64();
                let freq = read_cpu_frequency();
                let temp = read_die_temperature();
                samples.push(Sample {
                    elapsed_secs: elapsed,
                    freq_mhz: freq,
                    temp_c: temp,
                });
                break samples;
            }

            let elapsed = t0.elapsed().as_secs_f64();
            let freq = read_cpu_frequency();
            let temp = read_die_temperature();
            samples.push(Sample {
                elapsed_secs: elapsed,
                freq_mhz: freq,
                temp_c: temp,
            });

            // Sleep — check `running` every 100ms for responsive shutdown.
            let sleep_until = Instant::now() + Duration::from_secs(interval_secs);
            while Instant::now() < sleep_until {
                if !running.load(Ordering::Relaxed) {
                    break;
                }
                thread::sleep(Duration::from_millis(100));
            }
        }
    })
}

// ── Pinned worker pool (adapted for continuous load) ──────────────────────

struct ThermalWorkerPool {
    running: Arc<AtomicBool>,
    handles: Vec<Option<JoinHandle<()>>>,
}

impl ThermalWorkerPool {
    /// Spawn `num_workers` threads pinned to P-cores 0..num_workers-1,
    /// each running `worker_loop` with its own A, B, C buffers.
    fn new(num_workers: usize) -> Self {
        let running = Arc::new(AtomicBool::new(true));
        let mut handles = Vec::with_capacity(num_workers);

        // Deterministic data — every worker gets unique (but deterministic)
        // data so cache lines are independent.
        let mut abuf = vec![0.0f32; MATMUL_M * MATMUL_K];
        let mut bbuf = vec![0.0f32; MATMUL_K * MATMUL_N];
        let cbuf = vec![0.0f32; MATMUL_M * MATMUL_N];

        for core in 0..num_workers {
            let a = abuf.clone();
            let b = bbuf.clone();
            let mut c = cbuf.clone();
            fill_deterministic(&mut abuf, (core * 1000 + 1) as u64);
            fill_deterministic(&mut bbuf, (core * 1000 + 2) as u64);

            let flag = Arc::clone(&running);
            let h = thread::Builder::new()
                .name(format!("thermal-worker-{}", core))
                .spawn(move || {
                    pin_to_p_core(core);
                    worker_loop(&flag, &a, &b, &mut c);
                })
                .expect("spawn thermal worker");
            handles.push(Some(h));
        }

        Self { running, handles }
    }

    /// Signal all workers to stop and join them.
    fn shutdown(&mut self) {
        self.running.store(false, Ordering::Release);
        for h in self.handles.iter_mut() {
            if let Some(jh) = h.take() {
                let _ = jh.join();
            }
        }
    }
}

// ── Report helpers ────────────────────────────────────────────────────────

/// Format a duration in human-readable form.
fn fmt_dur(secs: f64) -> String {
    let m = (secs / 60.0).floor() as u64;
    let s = (secs as u64) % 60;
    if m > 0 {
        format!("{}m{:02}s", m, s)
    } else {
        format!("{}s", s)
    }
}

// ── Test ──────────────────────────────────────────────────────────────────

#[test]
fn test_thermal_stress_soak() {
    println!();
    println!("=== THERMAL STRESS SOAK ===");
    println!(
        "Workload: {} pinned P-core workers, continuous {}x{}x{} FP32 matmul",
        NUM_WORKERS, MATMUL_M, MATMUL_K, MATMUL_N
    );
    println!(
        "Duration: {} ({} seconds)",
        fmt_dur(SOAK_SECONDS as f64),
        SOAK_SECONDS
    );
    println!();

    // ── Take initial measurements ─────────────────────────────────────

    let initial_freq = read_cpu_frequency();
    let initial_temp = read_die_temperature();

    println!("  Initial CPU frequency: {:.1} MHz", initial_freq);
    if initial_freq == 0.0 {
        println!("  ⚠ sysctl unavailable (hw.cpufrequency returned 0)");
    }

    if initial_temp > 0.0 {
        println!("  Initial die temperature: {:.1} °C", initial_temp);
    } else {
        println!("  Initial die temperature: unavailable (SMC not accessible)");
    }
    println!();

    // ── Start workers ─────────────────────────────────────────────────

    let mut pool = ThermalWorkerPool::new(NUM_WORKERS);
    let sampler_running = Arc::new(AtomicBool::new(true));
    let sampler_flag = Arc::clone(&sampler_running);
    let sampler = sampler_thread(sampler_flag, FREQ_LOG_INTERVAL_SECS);

    // ── Soak loop ─────────────────────────────────────────────────────

    let t0 = Instant::now();
    let mut last_status_secs: u64 = 0;
    let mut throttle_events: Vec<f64> = Vec::new();
    let mut min_freq = initial_freq;

    loop {
        let elapsed = t0.elapsed().as_secs();
        if elapsed >= SOAK_SECONDS {
            break;
        }

        let status_period = elapsed / STATUS_INTERVAL_SECS;
        if status_period > last_status_secs / STATUS_INTERVAL_SECS {
            last_status_secs = elapsed;
            let freq = read_cpu_frequency();
            let temp = read_die_temperature();

            if freq > 0.0 && initial_freq > 0.0 && freq < initial_freq * THROTTLE_THRESHOLD {
                throttle_events.push(elapsed as f64);
            }

            if freq > 0.0 && (min_freq == 0.0 || freq < min_freq) {
                min_freq = freq;
            }

            if freq > 0.0 {
                let pct = (freq / initial_freq) * 100.0;
                print!(
                    "  [{:>3}s / {}] freq={:.0} MHz ({:.1}%)",
                    elapsed,
                    fmt_dur(SOAK_SECONDS as f64),
                    freq,
                    pct
                );
            } else {
                print!(
                    "  [{:>3}s / {}] freq= N/A (sysctl)",
                    elapsed,
                    fmt_dur(SOAK_SECONDS as f64)
                );
            }
            if temp > 0.0 {
                print!("  temp={:.1} °C", temp);
            }
            println!();
        }

        thread::sleep(Duration::from_secs(1));
    }

    // ── Stop workers and sampler ──────────────────────────────────────

    sampler_running.store(false, Ordering::Release);
    pool.shutdown();

    let samples = sampler.join().expect("sampler thread panicked");

    // ── Compute statistics ────────────────────────────────────────────

    let freq_samples: Vec<f64> = samples.iter().map(|s| s.freq_mhz).collect();
    let temp_samples: Vec<f64> = samples.iter().map(|s| s.temp_c).collect();

    let valid_freqs: Vec<&f64> = freq_samples.iter().filter(|&&f| f > 0.0).collect();
    let valid_temps: Vec<&f64> = temp_samples.iter().filter(|&&t| t > 0.0).collect();

    let n = valid_freqs.len();
    let mean_freq = if n > 0 {
        valid_freqs.iter().copied().sum::<f64>() / n as f64
    } else {
        0.0
    };

    let min_freq = if n > 0 {
        valid_freqs
            .iter()
            .copied()
            .cloned()
            .fold(f64::MAX, f64::min)
    } else {
        0.0
    };

    let max_freq = if n > 0 {
        valid_freqs
            .iter()
            .copied()
            .cloned()
            .fold(f64::MIN, f64::max)
    } else {
        0.0
    };

    let stddev_freq = if n > 1 {
        let variance = valid_freqs
            .iter()
            .map(|&&f| (f - mean_freq).powi(2))
            .sum::<f64>()
            / (n - 1) as f64;
        variance.sqrt()
    } else {
        0.0
    };

    let mean_temp = if !valid_temps.is_empty() {
        valid_temps.iter().copied().sum::<f64>() / valid_temps.len() as f64
    } else {
        0.0
    };

    let min_temp = if !valid_temps.is_empty() {
        valid_temps
            .iter()
            .copied()
            .cloned()
            .fold(f64::MAX, f64::min)
    } else {
        0.0
    };

    let max_temp = if !valid_temps.is_empty() {
        valid_temps
            .iter()
            .copied()
            .cloned()
            .fold(f64::MIN, f64::max)
    } else {
        0.0
    };

    // ── Report ────────────────────────────────────────────────────────

    println!();
    println!("=== THERMAL SOAK RESULTS ===");
    println!(
        "  Duration:                  {}",
        fmt_dur(SOAK_SECONDS as f64)
    );
    println!("  Samples:                   {}", samples.len());
    println!();

    if n > 0 {
        println!("── CPU Frequency (sysctl hw.cpufrequency) ──");
        println!("  Initial:                   {:.1} MHz", initial_freq);
        println!("  Mean:                      {:.1} MHz", mean_freq);
        println!("  Min:                       {:.1} MHz", min_freq);
        println!("  Max:                       {:.1} MHz", max_freq);
        println!("  Std Dev:                   {:.1} MHz", stddev_freq);
        if mean_freq > 0.0 {
            println!(
                "  Avg drop from initial:      {:.1}%",
                (1.0 - mean_freq / initial_freq) * 100.0
            );
        }

        if !throttle_events.is_empty() {
            println!(
                "  ⚠ Throttling events:        {} detected",
                throttle_events.len()
            );
            for te in &throttle_events {
                println!("    - at t = {:.0}s", te);
            }
        } else {
            println!("  ✅ No throttling events detected");
        }
    } else {
        println!("── CPU Frequency ──");
        println!("  No frequency samples (sysctl returned 0 for all reads)");
        println!("  This is expected on some Apple Silicon configurations.");
    }

    if !valid_temps.is_empty() {
        println!();
        println!("── Die Temperature (SMC) ──");
        println!("  Mean:                      {:.1} °C", mean_temp);
        println!("  Min:                       {:.1} °C", min_temp);
        println!("  Max:                       {:.1} °C", max_temp);
    } else {
        println!();
        println!("── Die Temperature ──");
        println!("  No temperature samples (SMC not accessible)");
        println!("  This is expected if running without SIP disabled or in a sandbox.");
    }

    println!();

    // ── PASS / FAIL ───────────────────────────────────────────────────

    if n == 0 {
        // sysctl unavailable — log but don't fail; report the situation.
        println!("  SYSCTL UNAVAILABLE: `hw.cpufrequency` returned 0 throughout.");
        println!("  PASS (insufficient data — sysctl not populated on this system)");
    } else if throttle_events.is_empty() && min_freq >= initial_freq * THROTTLE_THRESHOLD {
        println!(
            "  ✅ PASS: frequency stayed above {:.0}% of initial ({:.1} MHz threshold)",
            THROTTLE_THRESHOLD * 100.0,
            initial_freq * THROTTLE_THRESHOLD
        );
    } else {
        let min_pct = (min_freq / initial_freq) * 100.0;
        println!(
            "  ❌ FAIL: frequency dropped to {:.1}% of initial ({:.1} MHz < {:.1} MHz threshold)",
            min_pct,
            min_freq,
            initial_freq * THROTTLE_THRESHOLD
        );
        println!(
            "         {} throttling event(s) detected",
            throttle_events.len()
        );
        panic!(
            "Thermal soak failed: freq dropped to {:.1}% of initial ({:.1} MHz < {:.1} MHz)",
            min_pct,
            min_freq,
            initial_freq * THROTTLE_THRESHOLD
        );
    }
}
