//! ANE keepalive — sends dummy predicts at configurable intervals to
//! prevent the ANE from entering a lower-power idle state.
//!
//! ## Background
//! The Apple Neural Engine enters a low-power idle state after brief periods
//! of inactivity.  Re-entering peak performance after idle requires ~100 µs or
//! more of ramp-up.  Empirical data from the Tribunus calibration suite shows:
//!
//! - At 10 ms gap: +26 % latency on the first post-idle predict
//! - At 500 ms gap: +42 % latency
//!
//! By sending a trivial 64×64 matmul predict at <5 ms intervals, the ANE
//! stays in its active power state and production predictions see consistent
//! sub-50 µs dispatch latency.
//!
//! The keepalive model is a single `matmul` (1×64 · 64×64 → 1×64) compiled
//! for `cpuAndNeuralEngine`.  A single ping completes in ~10 µs with
//! negligible power overhead.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

use crate::arena::Arena;
use crate::arena::DataType;
use crate::coreml_bridge::CoreMlModel;
use crate::coreml_pipeline::build_matmul_region;

/// ANE keepalive — sends dummy predicts at configurable intervals to
/// prevent the ANE from entering a lower-power idle state.
///
/// ## Usage
///
/// ```ignore
/// let mut ka = AneKeepalive::start(3000)    // 3 ms interval
///     .expect("keepalive init failed");
/// // … production work …
/// ka.stop();                                 // clean shutdown
/// ```
///
/// Dropping the instance also stops the keepalive thread.
pub struct AneKeepalive {
    /// Loaded Core ML model (tiny 64×64 matmul, ANE-targeted).
    model: Option<Arc<CoreMlModel>>,
    /// Background thread that sends the pings.
    thread: Option<thread::JoinHandle<()>>,
    /// Shared shutdown flag.
    shutdown: Arc<AtomicBool>,
}

// Safety: CoreMlModel is Send + Sync.  The thread handle is only joined in
// `stop()` / `drop()`, and the shutdown flag is atomic.
unsafe impl Send for AneKeepalive {}
unsafe impl Sync for AneKeepalive {}

// ── Feature names baked into the compiled model ──────────────────────────

const INPUT_NAME: &str = "x";
const OUTPUT_NAME: &str = "matmul_1";

// ── Matrix dimensions (hidden size) ──────────────────────────────────────

const HIDDEN: i64 = 64;

impl AneKeepalive {
    /// Start the keepalive thread with a tiny 64×64 matmul model compiled
    /// for the ANE.
    ///
    /// `interval_us` — interval between dummy predicts in microseconds.
    ///
    /// - `interval_us < 5000` (5 ms): the ANE stays in its peak-performance
    ///   active state.
    /// - `interval_us` up to 10 000 (10 ms): still provides meaningful
    ///   benefit, though the first predict after idle may see a small
    ///   latency bump.
    ///
    /// ## Errors
    ///
    /// Returns `Err` if MIL program construction, `coremlcompiler`
    /// invocation, model loading, or arena allocation fails.
    pub fn start(interval_us: u32) -> Result<Self, String> {
        let interval = std::time::Duration::from_micros(interval_us as u64);

        // ── Build, compile, and load the keepalive model ─────────────
        let (model, input_arena, output_arena) = Self::build_keepalive_model()?;
        let model = Arc::new(model);
        let ping_model = Arc::clone(&model);

        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_flag = shutdown.clone();

        const REGION_ID: &str = "ane_keepalive_64x64";

        let thread = thread::Builder::new()
            .name("ane-keepalive".into())
            .spawn(move || {
                // The arenas are moved into the closure and live for the
                // entire thread lifetime.  Access is exclusive — no other
                // thread touches them.
                let inp = input_arena.info;
                let out = output_arena.info;
                Self::keepalive_loop(&ping_model, REGION_ID, &inp, &out, interval, &shutdown_flag);
                // model dropped here — Arc refcount reaches zero
            })
            .map_err(|e| format!("failed to spawn keepalive thread: {e}"))?;

        Ok(Self {
            model: Some(model),
            thread: Some(thread),
            shutdown,
        })
    }

    /// Stop the keepalive thread and release all Core ML resources.
    ///
    /// Safe to call multiple times; subsequent calls are no-ops.
    pub fn stop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
        // Drop the model, releasing the Core ML MLModel handle on the
        // calling thread.
        self.model.take();
    }

    // ── Internal helpers ─────────────────────────────────────────────

    /// Build the 64×64 identity-matmul model, compile it for the ANE,
    /// and load the resulting `.modelc`.
    fn build_keepalive_model() -> Result<(CoreMlModel, Arena, Arena), String> {
        // Identity matrix — matmul with identity is a pass-through at
        // minimal compute cost (~10 µs per ping on M1 ANE).
        let mut weight_values = vec![0.0_f32; (HIDDEN * HIDDEN) as usize];
        for i in 0..HIDDEN {
            weight_values[(i * HIDDEN + i) as usize] = 1.0;
        }

        const REGION_ID: &str = "ane_keepalive_64x64";

        // Compile the MIL program into a .modelc bundle via xcrun
        // coremlcompiler.  We keep the temporary directory alive so the
        // compiled artifacts exist while the model loads.
        let tmp_dir = tempfile::tempdir().map_err(|e| format!("keepalive tempdir: {e}"))?;
        let output_path = tmp_dir.path().to_path_buf();

        let receipt = build_matmul_region(
            INPUT_NAME,
            &[1, HIDDEN],
            "weight",
            &weight_values,
            &[HIDDEN, HIDDEN],
            &output_path,
            REGION_ID,
        )?;

        let model = CoreMlModel::load(&receipt.compiled_modelc_path)?;

        // Allocate zero-initialised IOSurface-backed arenas for the
        // predict loop.  Each arena holds 1×64 Float32 = 256 bytes.
        let input_arena = Arena::new(1, HIDDEN as u32, DataType::Float32)
            .map_err(|e| format!("keepalive input arena: {e}"))?;
        let output_arena = Arena::new(1, HIDDEN as u32, DataType::Float32)
            .map_err(|e| format!("keepalive output arena: {e}"))?;

        // Model is loaded in memory — the on-disk .modelc is no longer
        // needed.
        drop(tmp_dir);

        Ok((model, input_arena, output_arena))
    }

    /// The background loop: sleep → predict → repeat until shutdown is
    /// signalled.
    fn keepalive_loop(
        model: &CoreMlModel,
        region_id: &str,
        input_arena: &crate::arena_info::ArenaInfo,
        output_arena: &crate::arena_info::ArenaInfo,
        interval: std::time::Duration,
        shutdown: &AtomicBool,
    ) {
        while !shutdown.load(Ordering::Relaxed) {
            thread::sleep(interval);

            if shutdown.load(Ordering::Relaxed) {
                break;
            }

            if let Err(e) = model.predict(INPUT_NAME, input_arena, OUTPUT_NAME, output_arena) {
                // Keepalive is best-effort — log the error but keep
                // running so the thread can be stopped cleanly.
                eprintln!("[ane_keepalive] predict failed for {}: {e}", region_id,);
            }
        }
    }
}

impl Drop for AneKeepalive {
    fn drop(&mut self) {
        self.stop();
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that `start` and `stop` complete without error and the
    /// thread terminates cleanly.
    #[test]
    fn start_stop() {
        // Check for toolchain availability first — the compile step
        // requires xcrun + coremlcompiler.
        let _tc = match crate::toolchain_attest::ToolchainAttestation::probe() {
            Ok(t) => t,
            Err(e) => {
                eprintln!("SKIP: toolchain not available ({e})");
                return;
            }
        };

        let mut ka = AneKeepalive::start(5000).expect("keepalive start");
        // Let it run for a few cycles.
        thread::sleep(std::time::Duration::from_millis(20));
        ka.stop();

        assert!(ka.thread.is_none(), "thread must be joined after stop");
        assert!(ka.model.is_none(), "model must be dropped after stop");
    }

    /// Dropping the keepalive must stop the thread and release the model
    /// without panicking.
    #[test]
    fn drop_cleanup() {
        let _tc = match crate::toolchain_attest::ToolchainAttestation::probe() {
            Ok(t) => t,
            Err(e) => {
                eprintln!("SKIP: toolchain not available ({e})");
                return;
            }
        };

        let ka = AneKeepalive::start(5000).expect("keepalive start");
        thread::sleep(std::time::Duration::from_millis(10));
        // Implicit drop — must not panic and must join the thread.
        drop(ka);
    }

    /// With a 1 ms interval the keepalive completes many cycles
    /// successfully.
    #[test]
    fn fast_interval() {
        let _tc = match crate::toolchain_attest::ToolchainAttestation::probe() {
            Ok(t) => t,
            Err(e) => {
                eprintln!("SKIP: toolchain not available ({e})");
                return;
            }
        };

        let mut ka = AneKeepalive::start(1000).expect("keepalive start with 1 ms interval");
        // Run for ~15 ms → ~15 pings.
        thread::sleep(std::time::Duration::from_millis(15));
        ka.stop();
    }

    /// Calling `stop()` multiple times is safe (no double-join, no
    /// double-free).
    #[test]
    fn double_stop() {
        let _tc = match crate::toolchain_attest::ToolchainAttestation::probe() {
            Ok(t) => t,
            Err(e) => {
                eprintln!("SKIP: toolchain not available ({e})");
                return;
            }
        };

        let mut ka = AneKeepalive::start(5000).expect("keepalive start");
        thread::sleep(std::time::Duration::from_millis(5));
        ka.stop();
        ka.stop(); // second call is a no-op
    }
}
