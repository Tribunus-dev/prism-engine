//! Production-grade pinned P-core worker pool.
//!
//! Spawns N threads pinned to P-cores 0..N-1 via Mach thread affinity
//! (`thread_policy_set` with `THREAD_AFFINITY_POLICY`).  Workers spin-wait
//! on `AtomicU64` work pointers and signal completion by resetting to 0.
//!
//! ## Dispatch model
//!
//! `PinnedWorkerPool::run()` partitions the caller's `&[WorkItem]` into N
//! contiguous chunks (one per worker) and writes a tagged pointer to each
//! worker's atomic slot.  Workers cast the pointer back to `WorkerAssignment`,
//! iterate their chunk, and store 0 to signal done.  The calling thread
//! spin-waits on all slots reaching 0.
//!
//! ## Panic isolation
//!
//! Each worker wraps the function dispatch in `catch_unwind`.  If a function
//! panics, the worker records the panic payload in a shared slot, signals
//! completion, and continues spinning.  `run()` checks the shared panic slot
//! after all workers finish and re-panics with the original message if one
/// was recorded.
///
/// ## Safety
///
/// `WorkItem` contains raw function and argument pointers.  Callers must
/// guarantee that:
/// - `fn_ptr` points to a valid `unsafe fn(usize, usize)` (Rust ABI — supports
///   panic unwinding; workers use `catch_unwind` for isolation).
/// - `arg_ptr` is valid, correctly aligned, and lives until `run()` returns.
/// - No two items alias the same mutable state in conflicting ways.
/// - The function does not rely on any particular calling convention.
#[cfg(target_os = "macos")]
use std::any::Any;
use std::hint::spin_loop;
use std::mem;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

// ── Mach thread affinity FFI ──────────────────────────────────────────────

/// `THREAD_AFFINITY_POLICY` tells the scheduler to co-locate threads with
/// the same affinity tag on the same core cluster.  On Apple Silicon, P-cores
/// are available on tags 0..N-1 (N = number of performance cores).
const THREAD_AFFINITY_POLICY: u32 = 1;

extern "C" {
    fn thread_policy_set(thread: u32, flavor: u32, policy_info: *const u32, count: u32) -> u32;
}

/// Pin the calling thread to a specific P-core via Mach affinity policy.
///
/// Uses `libc::pthread_mach_thread_np` to get the Mach thread port, then
/// sets `THREAD_AFFINITY_POLICY` with `core_id` as the affinity tag.
/// Non-fatal on failure — the scheduler may ignore affinity hints during
/// overcommit.
fn pin_to_p_core(core_id: usize) {
    let mach_thread = unsafe { libc::pthread_mach_thread_np(libc::pthread_self()) };
    let tag = core_id as u32;
    let ret =
        unsafe { thread_policy_set(mach_thread, THREAD_AFFINITY_POLICY, &tag as *const u32, 1) };
    if ret != 0 {
        // Non-fatal: the scheduler may still co-locate naturally.
        log_warn!(
            "cpu_worker_pool[core={}] thread_policy_set returned {} (Mach KERN_*)",
            core_id,
            ret,
        );
    }
}

// ── Public types ──────────────────────────────────────────────────────────

/// A generic CPU work item dispatched to a pinned worker.
///
/// # Safety
///
/// See module-level safety notes.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct WorkItem {
    /// Raw function pointer.  The receiving worker transmutes this to
    /// `unsafe extern "C" fn(arg_ptr: usize, arg_len: usize)`.
    pub fn_ptr: usize,
    /// Raw argument pointer (first parameter of the function).
    pub arg_ptr: usize,
    /// Argument length in bytes (second parameter of the function).
    pub arg_len: usize,
}

// ── Internal dispatch types ───────────────────────────────────────────────

/// Per-worker assignment: a slice of work items carved from the caller's
/// `run()` batch.  Written by the dispatching thread, read once by the
/// worker.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct WorkerAssignment {
    items: *const WorkItem,
    count: usize,
}

/// Sentinel value stored in the atomic slot to signal worker shutdown.
const SHUTDOWN: u64 = u64::MAX;

/// Shared panic state accessible from both workers and the dispatching thread.
struct PanicState {
    occurred: AtomicBool,
    payload: Mutex<Option<Box<dyn Any + Send + 'static>>>,
}

impl PanicState {
    fn new() -> Self {
        Self {
            occurred: AtomicBool::new(false),
            payload: Mutex::new(None),
        }
    }

    /// Record a panic.  Only the first panic is preserved.
    fn record(&self, panic: Box<dyn Any + Send + 'static>) {
        if self.occurred.swap(true, Ordering::Release) {
            return; // already recorded
        }
        if let Ok(mut guard) = self.payload.lock() {
            *guard = Some(panic);
        }
    }

    /// Check whether a panic was recorded.  If so, re-panic with the
    /// original message.
    fn check_and_repanic(&self) {
        if !self.occurred.load(Ordering::Acquire) {
            return;
        }
        // Re-panic on the calling thread with the original payload.
        if let Ok(mut guard) = self.payload.lock() {
            if let Some(payload) = guard.take() {
                // Try to extract a meaningful message for the panic hook.
                let msg = if let Some(s) = payload.downcast_ref::<&str>() {
                    s.to_string()
                } else if let Some(s) = payload.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "worker panicked (opaque payload)".to_string()
                };
                panic!("cpu_worker_pool: worker panic propagated: {msg}");
            }
        }
    }
}

// ── PinnedWorkerPool ──────────────────────────────────────────────────────

/// A pool of worker threads pinned to specific P-cores via Mach thread affinity.
///
/// Workers spin-wait on `AtomicU64` work slots.  The calling thread dispatches
/// work by writing a tagged pointer into each slot and spin-waits for all to
/// reset to 0, achieving microsecond-scale dispatch latency.
///
/// ## Panic propagation
///
/// If a work function panics, the panic is caught, recorded, and re-thrown on
/// the calling thread when `run()` returns.  The pool remains usable after a
/// caught panic — subsequent `run()` calls will work, though the original
/// panic payload is consumed once.
///
/// ## Shutdown
///
/// Call `shutdown()` to signal all workers to exit and join their threads.
/// `Drop` also signals shutdown but does **not** join — call `shutdown()`
/// explicitly before dropping to guarantee clean thread cleanup.
pub struct PinnedWorkerPool {
    slots: Vec<Arc<AtomicU64>>,
    handles: Vec<Option<JoinHandle<()>>>,
    panic_state: Arc<PanicState>,
    num_workers: usize,
}

impl PinnedWorkerPool {
    /// Create a pool with `num_workers` threads pinned to P-cores 0..N-1.
    ///
    /// Each worker thread is named `"cpu-wkr-{core}"` for debugging.
    ///
    /// ## Panics
    ///
    /// Panics if `num_workers == 0`.
    pub fn new(num_workers: usize) -> Self {
        assert!(
            num_workers > 0,
            "PinnedWorkerPool requires at least one worker"
        );

        let panic_state = Arc::new(PanicState::new());
        let mut slots = Vec::with_capacity(num_workers);
        let mut handles = Vec::with_capacity(num_workers);

        for core in 0..num_workers {
            let slot = Arc::new(AtomicU64::new(0));
            let worker_slot = Arc::clone(&slot);
            let worker_panic = Arc::clone(&panic_state);

            let builder = thread::Builder::new().name(format!("cpu-wkr-{core}"));

            let handle = builder
                .spawn(move || {
                    pin_to_p_core(core);

                    loop {
                        let raw = worker_slot.load(Ordering::Acquire);
                        if raw == SHUTDOWN {
                            break;
                        }
                        if raw == 0 {
                            spin_loop();
                            continue;
                        }

                        // SAFETY: the dispatching thread guarantees `raw` points
                        // to a valid, live `WorkerAssignment` for the duration
                        // of execution.  No other thread writes to this slot
                        // until the worker stores 0.
                        let assignment = unsafe { &*(raw as *const WorkerAssignment) };

                        // Empty assignment — no work for this worker this round.
                        if assignment.count == 0 {
                            worker_slot.store(0, Ordering::Release);
                            continue;
                        }

                        let items = unsafe {
                            std::slice::from_raw_parts(assignment.items, assignment.count)
                        };

                        // Process each work item, catching panics.
                        for item in items {
                            // SAFETY: caller guarantees fn_ptr is a valid
                            // function pointer — pass-through Rust ABI
                            // (supports catch_unwind / panic isolation).
                            // arg_ptr/arg_len must be valid for that function.
                            let func: unsafe fn(usize, usize) = unsafe {
                                mem::transmute::<usize, unsafe fn(usize, usize)>(item.fn_ptr)
                            };

                            let result =
                                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
                                    func(item.arg_ptr, item.arg_len)
                                }));

                            if let Err(panic_payload) = result {
                                worker_panic.record(panic_payload);
                            }
                        }

                        worker_slot.store(0, Ordering::Release);
                    }
                })
                .expect("failed to spawn pinned worker thread");

            slots.push(slot);
            handles.push(Some(handle));
        }

        Self {
            slots,
            handles,
            panic_state,
            num_workers,
        }
    }

    /// Distribute `items` across workers and block until all complete.
    ///
    /// Items are partitioned into `num_workers` contiguous chunks.  If there
    /// are fewer items than workers, the extra workers receive an empty
    /// assignment and immediately signal completion.
    ///
    /// ## Panics
    ///
    /// Panics if a worker caught a panic during execution; the original panic
    /// is re-thrown on the calling thread.
    pub fn run(&mut self, items: &[WorkItem]) {
        let nw = self.num_workers;
        if nw == 0 || items.is_empty() {
            return;
        }

        // Partition items into nw contiguous chunks.
        let per_worker = items.len() / nw;
        let extra = items.len() % nw;

        // Build assignments on the stack.  Each assignment points into the
        // caller's `items` slice, which must outlive `run()`.
        // Use a Vec to handle dynamic num_workers cleanly.
        let mut assignments: Vec<WorkerAssignment> = Vec::with_capacity(nw);
        let mut cursor = 0usize;
        for i in 0..nw {
            let count = per_worker + if i < extra { 1 } else { 0 };
            assignments.push(WorkerAssignment {
                items: if count > 0 {
                    items[cursor..].as_ptr()
                } else {
                    std::ptr::null()
                },
                count,
            });
            cursor += count;
        }

        // Clear any stale panic flag from a prior run.
        self.panic_state.occurred.store(false, Ordering::Release);
        // Use `drain(..)` so the Vec is consumed and can't be aliased.
        let drained: Vec<WorkerAssignment> = assignments.drain(..).collect();

        // Dispatch: Release-store so workers see the pointer.
        for (i, assignment) in drained.iter().enumerate() {
            let ptr = assignment as *const WorkerAssignment as u64;
            self.slots[i].store(ptr, Ordering::Release);
        }

        // Wait for all workers to signal completion.
        for i in 0..nw {
            while self.slots[i].load(Ordering::Acquire) != 0 {
                spin_loop();
            }
        }

        // Propagate any worker panic to the calling thread.
        self.panic_state.check_and_repanic();
    }

    /// Signal all workers to shut down and join their threads.
    ///
    /// After this call the pool is no longer usable; callers should drop it.
    pub fn shutdown(&mut self) {
        for slot in &self.slots {
            slot.store(SHUTDOWN, Ordering::Release);
        }
        for h in self.handles.iter_mut() {
            if let Some(jh) = h.take() {
                let _ = jh.join();
            }
        }
    }

    /// Returns the number of worker threads in this pool.
    pub fn num_workers(&self) -> usize {
        self.num_workers
    }
}

impl Drop for PinnedWorkerPool {
    fn drop(&mut self) {
        // Signal workers to exit.  If `shutdown()` was already called these
        // stores are redundant but harmless.
        for slot in &self.slots {
            slot.store(SHUTDOWN, Ordering::Release);
        }
        // Do NOT join in Drop — blocking in Drop is surprising.  Callers
        // must call `shutdown()` explicitly for clean teardown.  The workers
        // will eventually exit on their own when the process drops the Arc.
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    /// A simple work function: increment a counter at `arg_ptr`.
    unsafe fn increment_counter(counter_ptr: usize, _len: usize) {
        let counter = &*(counter_ptr as *const AtomicUsize);
        counter.fetch_add(1, Ordering::SeqCst);
    }

    /// A work function that multiplies a value by a scalar.
    unsafe fn scale_value(ptr: usize, len: usize) {
        let slice = std::slice::from_raw_parts_mut(ptr as *mut f32, len / 4);
        for v in slice.iter_mut() {
            *v *= 2.0;
        }
    }

    #[test]
    fn test_basic_dispatch() {
        let mut pool = PinnedWorkerPool::new(4);
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_ptr = Arc::as_ptr(&counter) as usize;

        let items: Vec<WorkItem> = (0..8)
            .map(|_| WorkItem {
                fn_ptr: increment_counter as *const () as usize,
                arg_ptr: counter_ptr,
                arg_len: 0,
            })
            .collect();

        pool.run(&items);
        assert_eq!(counter.load(Ordering::SeqCst), 8);

        pool.shutdown();
    }

    #[test]
    fn test_all_items_executed() {
        let mut pool = PinnedWorkerPool::new(4);
        let total = Arc::new(AtomicUsize::new(0));
        let ptr = Arc::as_ptr(&total) as usize;

        const N: usize = 1000;
        let items: Vec<WorkItem> = (0..N)
            .map(|_| WorkItem {
                fn_ptr: increment_counter as *const () as usize,
                arg_ptr: ptr,
                arg_len: 0,
            })
            .collect();

        pool.run(&items);
        assert_eq!(total.load(Ordering::SeqCst), N);
        pool.shutdown();
    }

    #[test]
    fn test_more_items_than_workers() {
        let mut pool = PinnedWorkerPool::new(2);

        // Each item zeroes 8 f32 values
        let mut data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let items: Vec<WorkItem> = data
            .chunks_mut(8)
            .map(|chunk| WorkItem {
                fn_ptr: scale_value as *const () as usize,
                arg_ptr: chunk.as_mut_ptr() as usize,
                arg_len: chunk.len() * 4,
            })
            .collect();

        pool.run(&items);

        for (i, &v) in data.iter().enumerate() {
            assert_eq!(v, (i as f32) * 2.0, "item {i}");
        }
        pool.shutdown();
    }

    #[test]
    fn test_fewer_items_than_workers() {
        let mut pool = PinnedWorkerPool::new(4);
        let counter = Arc::new(AtomicUsize::new(0));
        let ptr = Arc::as_ptr(&counter) as usize;

        let items: Vec<WorkItem> = (0..2)
            .map(|_| WorkItem {
                fn_ptr: increment_counter as *const () as usize,
                arg_ptr: ptr,
                arg_len: 0,
            })
            .collect();

        pool.run(&items);
        // Only 2 items dispatched, only 2 workers process work (others
        // immediately signal completion with empty assignments).
        assert_eq!(counter.load(Ordering::SeqCst), 2);
        pool.shutdown();
    }

    #[test]
    fn test_multiple_runs() {
        let mut pool = PinnedWorkerPool::new(4);
        let counter = Arc::new(AtomicUsize::new(0));
        let ptr = Arc::as_ptr(&counter) as usize;

        for _ in 0..10 {
            let items: Vec<WorkItem> = (0..4)
                .map(|_| WorkItem {
                    fn_ptr: increment_counter as *const () as usize,
                    arg_ptr: ptr,
                    arg_len: 0,
                })
                .collect();
            pool.run(&items);
        }
        assert_eq!(counter.load(Ordering::SeqCst), 40);
        pool.shutdown();
    }

    #[test]
    fn test_empty_items_no_op() {
        let mut pool = PinnedWorkerPool::new(2);
        // Should not hang or panic.
        pool.run(&[]);
        pool.shutdown();
    }

    #[test]
    fn test_panic_isolation() {
        let mut pool = PinnedWorkerPool::new(2);

        unsafe fn panicking_fn(_ptr: usize, _len: usize) {
            panic!("intentional panic in worker");
        }

        let items: Vec<WorkItem> = (0..4)
            .map(|_| WorkItem {
                fn_ptr: panicking_fn as *const () as usize,
                arg_ptr: 0,
                arg_len: 0,
            })
            .collect();

        // run() should re-panic with the worker's message.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pool.run(&items);
        }));

        assert!(result.is_err(), "expected re-panic from worker");
        if let Err(payload) = result {
            let msg = payload
                .downcast_ref::<String>()
                .map(|s| s.as_str())
                .or_else(|| payload.downcast_ref::<&str>().copied())
                .unwrap_or("");
            assert!(
                msg.contains("intentional panic in worker"),
                "unexpected panic message: {msg:?}"
            );
        }

        // Ensure pool is still alive enough to shut down cleanly.
        pool.shutdown();
    }

    #[test]
    fn test_latency_benchmark() {
        let mut pool = PinnedWorkerPool::new(4);
        let counter = Arc::new(AtomicUsize::new(0));
        let ptr = Arc::as_ptr(&counter) as usize;

        // Benchmark dispatch latency
        let items: Vec<WorkItem> = (0..8)
            .map(|_| WorkItem {
                fn_ptr: increment_counter as *const () as usize,
                arg_ptr: ptr,
                arg_len: 0,
            })
            .collect();

        // Warmup
        for _ in 0..50 {
            pool.run(&items);
        }
        counter.store(0, Ordering::SeqCst);

        let start = Instant::now();
        const ITERS: usize = 500;
        for _ in 0..ITERS {
            pool.run(&items);
        }
        let elapsed = start.elapsed();

        let avg_ns = elapsed.as_nanos() as f64 / (ITERS as f64);
        eprintln!(
            "dispatch+exec overhead (8 items, 4 workers): {avg_ns:.0} ns avg over {ITERS} runs"
        );

        assert_eq!(counter.load(Ordering::SeqCst), 8 * ITERS);
        pool.shutdown();
    }

    /// Thread join eventually succeeds after shutdown.
    #[test]
    fn test_shutdown_joins_workers() {
        let mut pool = PinnedWorkerPool::new(2);
        // Give workers a trivial workload to confirm the loop runs.
        let counter = Arc::new(AtomicUsize::new(0));
        let ptr = Arc::as_ptr(&counter) as usize;
        let items: Vec<WorkItem> = (0..4)
            .map(|_| WorkItem {
                fn_ptr: increment_counter as *const () as usize,
                arg_ptr: ptr,
                arg_len: 0,
            })
            .collect();
        pool.run(&items);
        assert_eq!(counter.load(Ordering::SeqCst), 4);

        // Join should return promptly.
        let deadline = Instant::now() + Duration::from_secs(5);
        pool.shutdown();
        assert!(Instant::now() < deadline, "shutdown took too long");
    }
}
