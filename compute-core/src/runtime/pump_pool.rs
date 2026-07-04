//! Pump pool — a thread pool that decompresses tile640 weights and
//! materialises them in NPU-native format on per-device staging buffers.
//!
//! # Architecture
//!
//! Each NPU device gets one dedicated worker thread, pinned to a P-core.
//! Workers run independently — no barrier, no lock-step.  The coordinator
//! enqueues a layer to all workers via per-worker `mpsc::Sender` channels.
//! Workers decompress the full layer (batch-per-layer, not row-slice),
//! format it for their device, push to the staging buffer, and signal
//! completion via a shared `DoneTracker`.
//!
//! The ECS execution loop (ANE multiplexer or NPU dispatcher) calls
//! `wait_for_layer(layer)` to block until all devices finish.
//!
//! # Thread count
//!
//! One worker per `NpuDevice`.  The ANE gets one worker, two Coral TPUs
//! get two workers, etc.  Workers are pinned to P-cores via QoS policy.
//! The coordinator runs on whatever thread enqueues work (typically the
//! P-core multiplexer or the main inference loop).

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{mpsc, Arc};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use super::npu_pump::NpuWeightPump;
use crate::compute_image::compile::ternary::SegmentKind;

// ── Shared block scales cache ─────────────────────────────────────

/// Pre-parsed FP16 block scales for all layers, stored in the pool so
/// each worker can slice out the range for its layer without re-parsing
/// the cimage segment.
struct BlockScaleCache {
    /// Raw FP16 bytes from the cimage BlockScales segment.
    raw: Vec<u8>,
}

impl BlockScaleCache {
    fn from_mmap(
        mmap: &[u8],
        header: &crate::compute_image::compile::ternary::CimageHeader,
    ) -> Self {
        let entry = header.segment(SegmentKind::BlockScales);
        let raw = match entry {
            Some(seg) if seg.length > 0 => {
                let start = seg.offset as usize;
                let len = seg.length as usize;
                let end = (start + len).min(mmap.len());
                let slice = &mmap[start..end];
                slice.to_vec()
            }
            _ => Vec::new(),
        };
        Self { raw }
    }
}

// ── Completion tracking ───────────────────────────────────────────

/// Signal that a layer's pump work is done for a specific device.
struct DoneTracker {
    /// For each layer: a bitmask of device indices that have completed.
    /// `1 << device_index` when done.
    layers: Vec<AtomicU32>,
    /// Total number of devices.
    n_devices: u32,
}

impl DoneTracker {
    fn new(capacity: usize, n_devices: u32) -> Self {
        Self {
            layers: (0..capacity).map(|_| AtomicU32::new(0)).collect(),
            n_devices,
        }
    }

    fn mark_done(&self, layer: usize, device_index: u32) {
        self.layers[layer].fetch_or(1u32 << device_index, Ordering::Release);
    }

    fn all_done(&self, layer: usize) -> bool {
        let mask = (1u32 << self.n_devices) - 1;
        self.layers[layer].load(Ordering::Acquire) == mask
    }
}

// ── Types ─────────────────────────────────────────────────────────

/// How to submit a staging buffer to an NPU device.
///
/// For the ANE: no-op (the pump writes directly to SLC, which is the
/// device's working memory).  For a Coral TPU: `write()` to `/dev/apex_N`
/// followed by an `ioctl(SUBMIT)`.  For Intel NCE: write to VPU DMA
/// staging and submit a descriptor.
pub enum SubmitMethod {
    /// No explicit submission — staging buffer *is* device memory.
    DirectMapped,
    /// Submit via a callback.
    Submit(Box<dyn Fn(&[u8]) -> Result<(), String> + Send + Sync>),
}

/// One NPU device managed by the pool.
pub struct NpuDevice {
    /// The weight format converter.
    pub pump: Box<dyn NpuWeightPump>,
    /// Staging buffer for this device's native weights.
    /// Pre-allocated to `pump.output_buffer_size(max_rows, max_cols)`.
    pub staging: Vec<u8>,
    /// How to submit the staging buffer to the device.
    pub submit: SubmitMethod,
    /// Human-readable label for diagnostics.
    pub name: String,
}

/// Work item sent to a worker thread.
struct WorkItem {
    /// 0-based layer index.
    layer_index: u32,
    /// Byte range into the cimage mmap for this layer's ternary weights.
    ternary_offset: usize,
    ternary_len: usize,
    /// Byte range into BlockScaleCache.raw for this layer's block scales.
    scale_offset: usize,
    scale_len: usize,
    /// Logical weight matrix dimensions.
    rows: usize,
    cols: usize,
}

// ── Pump pool ────────────────────────────────────────────────────

/// Manages a pool of NPU pump workers — one thread per device.
///
/// Usage:
/// ```
/// let pool = PumpPool::spawn(mmap, devices, &header);
/// pool.enqueue_layer(0, &layer_info);
/// pool.wait_for_layer(0);
/// ```
pub struct PumpPool {
    /// Shared mmap for all workers to read ternary weights.
    _mmap: Arc<memmap2::Mmap>,
    /// Pre-parsed block scales for all layers.
    _scale_cache: BlockScaleCache,
    /// Completion tracker shared with workers.
    done: Arc<DoneTracker>,
    /// Per-worker senders.
    senders: Vec<mpsc::Sender<WorkItem>>,
    /// Worker threads.
    _workers: Vec<JoinHandle<()>>,
    /// Stop signal.
    _stop: Arc<AtomicBool>,
}

impl PumpPool {
    /// Spawn the pump pool with the given devices.
    ///
    /// `mmap` — the cimage mmap that all workers read from.
    /// `header` — the cimage header (for block scales segment lookup).
    /// `devices` — one `NpuDevice` per NPU backend to drive.
    ///
    /// Returns a `PumpPool` with one worker thread per device, ready to
    /// accept work via `enqueue_layer`.
    pub fn spawn(
        mmap: Arc<memmap2::Mmap>,
        header: &crate::compute_image::compile::ternary::CimageHeader,
        devices: Vec<NpuDevice>,
    ) -> Self {
        let n_devices = devices.len() as u32;
        let scale_cache = BlockScaleCache::from_mmap(&mmap, header);
        let done = Arc::new(DoneTracker::new(1024, n_devices)); // 1024-layer capacity
        let stop = Arc::new(AtomicBool::new(false));

        let mut senders = Vec::with_capacity(devices.len());
        let mut workers = Vec::with_capacity(devices.len());

        for (di, device) in devices.into_iter().enumerate() {
            let (tx, rx) = mpsc::channel::<WorkItem>();
            senders.push(tx);

            let done = done.clone();
            let stop = stop.clone();
            let mmap = mmap.clone();
            let scales_raw = scale_cache.raw.clone();

            workers.push(
                thread::Builder::new()
                    .name(format!("pump-{}", device.name))
                    .spawn(move || {
                        // Pin to any available core (the caller handles QoS).
                        // Workers use P-core-style spinning: no yielding on hot path.
                        Self::worker_loop(
                            &mmap,
                            &scales_raw,
                            &rx,
                            &done,
                            &stop,
                            &device,
                            di as u32,
                        );
                    })
                    .expect("pump worker thread spawn failed"),
            );
        }

        Self {
            _mmap: mmap,
            _scale_cache: scale_cache,
            done,
            senders,
            _workers: workers,
            _stop: stop,
        }
    }

    /// Enqueue a layer for all devices to pump.
    ///
    /// Non-blocking: returns immediately after sending to all workers.
    /// The work is picked up by each worker's thread and processed
    /// concurrently.  Call `wait_for_layer` before dispatching execution.
    pub fn enqueue_layer(
        &self,
        layer_index: u32,
        ternary_offset: usize,
        ternary_len: usize,
        rows: usize,
        cols: usize,
    ) {
        // Compute the block scale range for this layer's weight count.
        let n_vals = rows * cols;
        let n_blocks = (n_vals + 255) / 256;
        let scale_len = n_blocks * 2;
        let scale_offset = 0; // the cache is per-layer; the worker slices from the raw vec

        for (_di, sender) in self.senders.iter().enumerate() {
            let item = WorkItem {
                layer_index,
                ternary_offset,
                ternary_len,
                scale_offset,
                scale_len,
                rows,
                cols,
            };
            let _ = sender.send(item); // ignore errors (worker may be shutting down)
        }
    }

    /// Block until all devices have finished pumping `layer_index`.
    ///
    /// Spins with a short yield — the pump workers are on other cores
    /// and the critical path is memory-bound, not CPU-bound.
    pub fn wait_for_layer(&self, layer_index: u32) {
        let idx = layer_index as usize;
        while !self.done.all_done(idx) {
            std::hint::spin_loop();
        }
    }

    // ── Worker implementation ──────────────────────────────────────

    fn worker_loop(
        mmap: &[u8],
        scales_raw: &[u8],
        rx: &mpsc::Receiver<WorkItem>,
        done: &DoneTracker,
        stop: &AtomicBool,
        device: &NpuDevice,
        device_index: u32,
    ) {
        // Pre-compute the max buffer size used for staging.
        // Workers reuse their staging Vec across repack calls.
        let mut staging = device.staging.clone();

        loop {
            if stop.load(Ordering::Relaxed) {
                return;
            }

            // Try to receive a work item.  Block with a short timeout
            // so we can check the stop flag.
            let item = match rx.recv_timeout(Duration::from_millis(100)) {
                Ok(work) => work,
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(mpsc::RecvTimeoutError::Disconnected) => return,
            };

            // Ensure staging buffer is large enough.
            let needed = device.pump.output_buffer_size(item.rows, item.cols);
            if staging.len() < needed {
                staging.resize(needed, 0);
            }

            // Slice ternary weights from the mmap.
            let ternary_end = item.ternary_offset + item.ternary_len;
            let ternary_slice = if ternary_end <= mmap.len() {
                &mmap[item.ternary_offset..ternary_end]
            } else {
                &[]
            };

            // Slice block scales from the raw cache.
            let scale_end = item.scale_offset + item.scale_len;
            let scale_slice = if scale_end <= scales_raw.len() {
                &scales_raw[item.scale_offset..scale_end]
            } else {
                &[]
            };

            // Decompress, format, and write to staging buffer.
            device.pump.repack(
                ternary_slice,
                scale_slice,
                item.rows,
                item.cols,
                &mut staging[..needed],
            );

            // Submit to the device.
            match &device.submit {
                SubmitMethod::DirectMapped => {
                    // No-op: staging buffer *is* device memory (ANE SLC).
                }
                SubmitMethod::Submit(submit) => {
                    if let Err(e) = submit(&staging[..needed]) {
                        eprintln!("[pump] {} submit failed: {}", device.name, e);
                    }
                }
            }

            // Signal completion.
            done.mark_done(item.layer_index as usize, device_index);
        }
    }
}

impl Drop for PumpPool {
    fn drop(&mut self) {
        self._stop.store(true, Ordering::Release);
        // Drop the senders so workers unblock and exit.
        self.senders.clear();
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_image::compile::ternary::{
        CimageHeader, SegmentEntry, CIMAGE_PAGE_SIZE, PRISM_MAGIC,
    };
    use crate::runtime::npu_pump::AneWeightPump;

    /// Build a minimal cimage header + ternary weights in a temp mmap.
    fn make_minimal_cimage(
        rows: usize,
        cols: usize,
    ) -> (tempfile::TempDir, Arc<memmap2::Mmap>, CimageHeader) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.cimage");

        let nt = (cols + 639) / 640;
        let weights_len = rows * nt * 32 * 4;

        // Write a valid cimage with header + weights
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .unwrap();

        let page_aligned =
            ((std::mem::size_of::<CimageHeader>() + weights_len) as u64 + CIMAGE_PAGE_SIZE - 1)
                & !(CIMAGE_PAGE_SIZE - 1);

        file.set_len(page_aligned).unwrap();

        let mut mmap = unsafe { memmap2::MmapMut::map_mut(&file).unwrap() };

        // Build header with one TernaryWeights segment
        let weights_offset = std::mem::size_of::<CimageHeader>() as u64;
        let header = CimageHeader {
            magic: PRISM_MAGIC,
            version: 4,
            segment_count: 2,
            payload_hash: [0u8; 32],
            num_layers: 1,
            num_heads: 0,
            head_dim: 0,
            hidden_dim: cols as u32,
            intermediate_dim: rows as u32,
            vocab_size: 0,
            quantization_schema: 0,
            draft_num_layers: 0,
            segments: [
                SegmentEntry::new(SegmentKind::MetalLib, 0, 0),
                SegmentEntry::new(
                    SegmentKind::TernaryWeights,
                    weights_offset,
                    weights_len as u64,
                ),
                SegmentEntry {
                    kind: 0,
                    offset: 0,
                    length: 0,
                },
                SegmentEntry {
                    kind: 0,
                    offset: 0,
                    length: 0,
                },
                SegmentEntry {
                    kind: 0,
                    offset: 0,
                    length: 0,
                },
                SegmentEntry {
                    kind: 0,
                    offset: 0,
                    length: 0,
                },
                SegmentEntry {
                    kind: 0,
                    offset: 0,
                    length: 0,
                },
                SegmentEntry {
                    kind: 0,
                    offset: 0,
                    length: 0,
                },
                SegmentEntry {
                    kind: 0,
                    offset: 0,
                    length: 0,
                },
            ],
            _pad: [0u8; 8],
        };

        // Write header
        let header_bytes = unsafe {
            std::slice::from_raw_parts(
                &header as *const CimageHeader as *const u8,
                std::mem::size_of::<CimageHeader>(),
            )
        };
        mmap[..header_bytes.len()].copy_from_slice(header_bytes);

        // Write dummy ternary data (all zeros = all -1)
        let data_start = weights_offset as usize;
        let data_end = data_start + weights_len;
        mmap[data_start..data_end].fill(0);

        // Freeze to Mmap
        drop(mmap);
        let mmap = unsafe { memmap2::Mmap::map(&file).unwrap() };

        (dir, Arc::new(mmap), header)
    }

    #[test]
    fn test_pool_spawn_and_dispatch() {
        let rows = 32;
        let cols = 640;
        let (_dir, mmap, header) = make_minimal_cimage(rows, cols);

        // One ANE device.
        let ane_device = NpuDevice {
            pump: Box::new(AneWeightPump),
            staging: vec![0u8; 32 * 640], // minimum
            submit: SubmitMethod::DirectMapped,
            name: "ane-test".into(),
        };

        let pool = PumpPool::spawn(mmap.clone(), &header, vec![ane_device]);

        // Enqueue layer 0
        let weights_entry = header.segment(SegmentKind::TernaryWeights).unwrap();
        pool.enqueue_layer(
            0,
            weights_entry.offset as usize,
            weights_entry.length as usize,
            rows,
            cols,
        );

        // Wait for it.
        pool.wait_for_layer(0);
        // If we get here without hanging, the pool works.
    }

    #[test]
    fn test_two_devices_parallel() {
        let rows = 32;
        let cols = 640;
        let (_dir, mmap, header) = make_minimal_cimage(rows, cols);

        let ane_device = |name: &str| -> NpuDevice {
            NpuDevice {
                pump: Box::new(AneWeightPump),
                staging: vec![0u8; 32 * 640],
                submit: SubmitMethod::DirectMapped,
                name: name.into(),
            }
        };

        let pool = PumpPool::spawn(
            mmap.clone(),
            &header,
            vec![ane_device("dev0"), ane_device("dev1")],
        );

        let weights_entry = header.segment(SegmentKind::TernaryWeights).unwrap();
        pool.enqueue_layer(
            0,
            weights_entry.offset as usize,
            weights_entry.length as usize,
            rows,
            cols,
        );

        pool.wait_for_layer(0);
    }

    #[test]
    fn test_three_layers_sequential() {
        let rows = 32;
        let cols = 640;
        let (_dir, mmap, header) = make_minimal_cimage(rows, cols);

        let device = NpuDevice {
            pump: Box::new(AneWeightPump),
            staging: vec![0u8; 32 * 640],
            submit: SubmitMethod::DirectMapped,
            name: "seq-test".into(),
        };

        let pool = PumpPool::spawn(mmap.clone(), &header, vec![device]);
        let weights_entry = header.segment(SegmentKind::TernaryWeights).unwrap();

        for layer in 0..3 {
            pool.enqueue_layer(
                layer,
                weights_entry.offset as usize,
                weights_entry.length as usize,
                rows,
                cols,
            );
            pool.wait_for_layer(layer);
        }
    }
}
