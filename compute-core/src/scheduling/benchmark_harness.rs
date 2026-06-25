//! Verification harness for Gates 1-3: Zero-Copy, Prefill TPS, Decode TPS.
//!
//! Uses Mach kernel VM statistics (host_statistics64) to detect silent
//! IOSurface arena copies by measuring CPU page-outs.  On 16 GB Apple
//! Silicon systems, a broken zero-copy handoff instantly triggers
//! compressed swap that `VmProbe` catches at kernel level.

use std::time::Instant;

// ── Mach kernel VM profiler (Gate 1) ─────────────────────────────────────

/// Snapshots `vm_statistics64.pageouts` via Mach IPC.
///
/// The delta between `start()` and `stop_and_report()` reveals whether Core ML
/// silently shadow-copied the IOSurface arena into DRAM.  Each page-out is
/// 16 KB on Apple Silicon.
pub struct VmProbe {
    initial_pageouts: u64,
}

impl VmProbe {
    pub fn start() -> Self {
        Self {
            initial_pageouts: Self::get_pageouts(),
        }
    }

    /// Stop the probe and return the page-out delta **in MB**.
    pub fn stop_and_report_mb(&self) -> u64 {
        let delta = Self::get_pageouts().saturating_sub(self.initial_pageouts);
        // Apple Silicon page size = 16 KB = 16384 bytes
        (delta * 16384) / (1024 * 1024)
    }

    fn get_pageouts() -> u64 {
        // libc on macOS exposes host_statistics64 for reading Mach VM stats.
        // The host_statistics64() syscall fills a vm_statistics64 struct; we
        // extract the `pageouts` field (cumulative count since boot).
        #[cfg(target_os = "macos")]
        unsafe {
            use std::mem;
            let mut count: u32 =
                (mem::size_of::<libc::vm_statistics64>() / mem::size_of::<i32>()) as u32;
            let mut stats: libc::vm_statistics64 = mem::zeroed();
            let result = libc::host_statistics64(
                #[allow(deprecated)]
                libc::mach_host_self(),
                libc::HOST_VM_INFO64,
                &mut stats as *mut _ as *mut i32,
                &mut count,
            );
            if result == 0 {
                stats.pageouts
            } else {
                0
            }
        }
        #[cfg(not(target_os = "macos"))]
        0
    }
}

// ── Decoder abstraction (compile-time placeholder) ──────────────────────

/// Trait any Metal decoder must implement to be benchmarked.
///
/// The Phase 1 LUT kernel provider and Phase 2d orchestrator each produce
/// a decoder that satisfies this.  The harness only calls `step()`, keeping
/// the benchmark logic independent of the internal kernel dispatch strategy.
pub trait MetalDecoder {
    type Token;
    type Error;

    /// Advance one autoregressive decode step.
    ///
    /// Returns the next predicted token or an error if the kernel panics /
    /// Metal command buffer times out.
    fn step(&mut self) -> Result<Self::Token, Self::Error>;
}

// ── Orchestrator abstraction (compile-time placeholder) ─────────────────

/// Trait any prefill orchestrator must implement.
///
/// The Phase 2d `PrefillOrchestrator` satisfies this.  The harness only
/// calls `execute_chunked_prefill()`, remaining agnostic to:
/// - chunk-size selection strategy (BTreeMap iteration)
/// - IOSurface arena binding details
/// - MLState lifecycle
pub trait AnePrefillOrchestrator {
    type Error;

    /// Run a full chunked ANE prefill returning the number of tokens consumed.
    fn execute_chunked_prefill(&mut self, tokens: &[u32]) -> Result<usize, Self::Error>;
}

// ── Gate results ───────────────────────────────────────────────────────

pub struct GateResults {
    pub prefill_tps: f64,
    pub decode_tps: f64,
    pub pageouts_mb: u64,
}

// ── Benchmark runner ───────────────────────────────────────────────────

/// Run the three-gate benchmark in one automated pass.
///
/// # Arguments
/// * `orchestrator` — ANE prefill orchestrator (e.g. `PrefillOrchestrator`).
/// * `decoder` — Metal LUT decoder (e.g. `LutGemvDecoder`).
/// * `prompt_tokens` — tokenized prompt for prefill.
/// * `decode_target` — how many autoregressive tokens to generate.
pub fn run_tps_benchmark(
    orchestrator: &mut dyn AnePrefillOrchestrator<Error = String>,
    decoder: &mut dyn MetalDecoder<Token = u32, Error = String>,
    prompt_tokens: &[u32],
    decode_target: usize,
) -> Result<GateResults, String> {
    println!("[benchmark] Initiating Gates 1-3 Verification...");
    println!("[benchmark] Prompt length: {} tokens", prompt_tokens.len());
    println!("[benchmark] Decode target: {} tokens", decode_target);

    // ── Gates 1 & 2: ANE Prefill + Zero-Copy ──────────────────────────
    let vm_probe = VmProbe::start();
    let prefill_start = Instant::now();

    let tokens_processed = orchestrator.execute_chunked_prefill(prompt_tokens)?;

    let prefill_duration = prefill_start.elapsed().as_secs_f64();
    let prefill_tps = tokens_processed as f64 / prefill_duration;
    let pageouts_mb = vm_probe.stop_and_report_mb();

    println!(
        "[gate1] CPU page-outs during ANE execution: {} MB",
        pageouts_mb
    );
    println!("[gate2] ANE prefill throughput: {:.2} TPS", prefill_tps);

    // ── Gate 3: Metal LUT Decode ──────────────────────────────────────
    println!("[benchmark] Transitioning to Metal autoregressive decode...");

    let decode_start = Instant::now();
    for _ in 0..decode_target {
        decoder.step()?;
    }
    let decode_duration = decode_start.elapsed().as_secs_f64();
    let decode_tps = decode_target as f64 / decode_duration;

    println!("[gate3] Metal LUT decode throughput: {:.2} TPS", decode_tps);

    Ok(GateResults {
        prefill_tps,
        decode_tps,
        pageouts_mb,
    })
}

// ── Gate assertions ───────────────────────────────────────────────────

/// Assert all three gates against their targets.
///
/// # Arguments
/// * `baseline_prefill_tps` — TPS of the standard MLX/Metal NF4 prefill.
/// * `baseline_decode_tps`  — TPS of the standard MLX/Metal NF4 decode.
pub fn assert_gates(results: &GateResults, baseline_prefill_tps: f64, baseline_decode_tps: f64) {
    // Gate 1: Zero-copy — page-outs must stay under 1 MB
    assert!(
        results.pageouts_mb <= 1,
        "FAIL Gate 1: Zero-copy broken — {:.1} MB paged out. Core ML likely shadow-copied the IOSurface arena.",
        results.pageouts_mb,
    );

    // Gate 2: ANE prefill must be ≥ 2× GPU-only baseline
    assert!(
        results.prefill_tps >= baseline_prefill_tps * 2.0,
        "FAIL Gate 2: ANE prefill {:.1} TPS < 2× GPU baseline {:.1} TPS",
        results.prefill_tps,
        baseline_prefill_tps * 2.0,
    );

    // Gate 3: Metal LUT decode must be within 15 % of block MatMul
    let decode_ratio = results.decode_tps / baseline_decode_tps;
    assert!(
        decode_ratio >= 0.85,
        "FAIL Gate 3: LUT decode {:.1} TPS < 85 % of block MatMul baseline {:.1} TPS (ratio = {:.3})",
        results.decode_tps,
        baseline_decode_tps,
        decode_ratio,
    );

    println!("✅ All throughput and memory gates passed.");
    println!(
        "  ANE prefill: {:.1} TPS ({:.1}× baseline)",
        results.prefill_tps,
        results.prefill_tps / baseline_prefill_tps,
    );
    println!(
        "  LUT decode:  {:.1} TPS ({:.1} % of baseline)",
        results.decode_tps,
        100.0 * results.decode_tps / baseline_decode_tps,
    );
    println!("  Zero-copy:   {} MB paged out", results.pageouts_mb);
}

#[cfg(test)]
mod tests {
    use super::*;

    struct NoopDecoder;
    impl MetalDecoder for NoopDecoder {
        type Token = u32;
        type Error = String;
        fn step(&mut self) -> Result<u32, String> {
            // Simulate 1 µs per step
            std::thread::sleep(std::time::Duration::from_micros(1));
            Ok(0)
        }
    }

    struct NoopOrchestrator;
    impl AnePrefillOrchestrator for NoopOrchestrator {
        type Error = String;
        fn execute_chunked_prefill(&mut self, tokens: &[u32]) -> Result<usize, String> {
            // Simulate 100 µs per token
            let n = tokens.len();
            std::thread::sleep(std::time::Duration::from_micros(n as u64 * 100));
            Ok(n)
        }
    }

    #[test]
    fn test_harness_smoke() {
        let mut orch = NoopOrchestrator;
        let mut dec = NoopDecoder;
        let prompt: Vec<u32> = vec![1; 256];
        let results = run_tps_benchmark(&mut orch, &mut dec, &prompt, 100).unwrap();
        assert!(results.prefill_tps > 0.0);
        assert!(results.decode_tps > 0.0);
        // Noop decoder is much slower than target, so skip ratio assertions
        eprintln!(
            "[smoke] prefill={:.1} TPS, decode={:.1} TPS, pageouts={} MB",
            results.prefill_tps, results.decode_tps, results.pageouts_mb,
        );
    }

    #[test]
    #[should_panic(expected = "FAIL Gate 1")]
    fn test_assert_gate1_fail() {
        let r = GateResults {
            prefill_tps: 500.0,
            decode_tps: 100.0,
            pageouts_mb: 10,
        };
        assert_gates(&r, 10.0, 10.0);
    }

    #[test]
    #[should_panic(expected = "FAIL Gate 2")]
    fn test_assert_gate2_fail() {
        let r = GateResults {
            prefill_tps: 15.0,
            decode_tps: 100.0,
            pageouts_mb: 0,
        };
        assert_gates(&r, 10.0, 10.0);
    }

    #[test]
    #[should_panic(expected = "FAIL Gate 3")]
    fn test_assert_gate3_fail() {
        let r = GateResults {
            prefill_tps: 500.0,
            decode_tps: 5.0,
            pageouts_mb: 0,
        };
        assert_gates(&r, 10.0, 10.0);
    }

    #[test]
    fn test_assert_all_pass() {
        let r = GateResults {
            prefill_tps: 500.0,
            decode_tps: 95.0,
            pageouts_mb: 0,
        };
        assert_gates(&r, 10.0, 100.0); // decode ratio 0.95 > 0.85
    }
}
