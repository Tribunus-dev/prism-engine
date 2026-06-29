//! KV Interleave Pipeline — Multi-Threadgroup Prefetch ABI.
//!
//! Shared ABI types for the producer-consumer KV prefetch pipeline between
//! decode and cache-prefetch threadgroups.  Layout must be byte-identical
//! between Rust host code and Metal Shading Language device structs.
//!
//! Queue-counter padding (128 bytes) is a tuning parameter, not a universal
//! cache-line constant.  Benchmark 64/128/256-byte variants per GPU family
//! and seal the chosen stride into the ComputeImage device profile.

#![allow(dead_code)]

use core::sync::atomic::{AtomicU32, Ordering};
use std::collections::VecDeque;

// ── ABI constants ─────────────────────────────────────────────────
pub const CLAIM_UNOWNED: u32 = 0;
pub const CLAIM_HELPER: u32 = 1;
pub const CLAIM_DECODE_FALLBACK: u32 = 2;
pub const CLAIM_DECODE_CONSUMER: u32 = 3;

pub const OUTCOME_NONE: u32 = 0;
pub const OUTCOME_READY_CONSUMABLE: u32 = 1;
pub const OUTCOME_CANCELED: u32 = 2;
pub const OUTCOME_POISONED: u32 = 3;
pub const OUTCOME_BYPASSED: u32 = 4;

pub const FAULT_NONE: u32 = 0;
pub const FAULT_HANDOFF_INTEGRITY: u32 = 1;
pub const FAULT_INVALID_READY_STATE: u32 = 2;
pub const FAULT_GENERATION_MISMATCH: u32 = 3;
pub const FAULT_UNRECOGNIZED_OUTCOME: u32 = 4;

pub const KV_STATE_EMPTY: u32 = 0;
pub const KV_STATE_QUEUED: u32 = 1;
pub const KV_STATE_FILLING: u32 = 2;
pub const KV_STATE_READY: u32 = 3;
pub const KV_STATE_CONSUMING: u32 = 7;
pub const KV_STATE_RECLAIMABLE: u32 = 8;
pub const KV_STATE_POISONED: u32 = 5;
pub const KV_STATE_CANCELED: u32 = 6;

// ── Queue counter stride ──────────────────────────────────────────
pub const QUEUE_COUNTER_STRIDE_U32: usize = 32;

// ── Buffer state machine ──────────────────────────────────────────
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvBufferState {
    Empty = 0,
    Queued = 1,
    Filling = 2,
    Ready = 3,
    Consuming = 7,
    Poisoned = 5,
    Canceled = 6,
    Reclaimable = 8,
}

impl KvBufferState {
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::Empty),
            1 => Some(Self::Queued),
            2 => Some(Self::Filling),
            3 => Some(Self::Ready),
            7 => Some(Self::Consuming),
            5 => Some(Self::Poisoned),
            6 => Some(Self::Canceled),
            8 => Some(Self::Reclaimable),
            _ => None,
        }
    }
}

// ── Scratch buffer metadata (host-visible ABI) ────────────────────
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct KvScratchMetadataAbi {
    pub request_id: u32,
    pub session_id: u32,
    pub sequence_id: u32,
    pub target_layer: u32,
    pub token_epoch: u32,
    pub kv_generation: u32,
    pub page_table_generation: u32,
    pub data_offset: u32,
}

// ── Scratch buffer device control block (opaque atomics) ──────────
//
// The first 4 bytes alias the buffer state AtomicU32 used by
// Metal threadgroups for CAS transitions.  The remaining 28 bytes
// carry device-side cancel, producer, and completion fields.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct KvScratchDeviceControl {
    pub _opaque_atomic_storage: [u8; 32],
}

impl Default for KvScratchDeviceControl {
    fn default() -> Self {
        Self {
            _opaque_atomic_storage: [0u8; 32],
        }
    }
}

impl KvScratchDeviceControl {
    pub const SIZE: usize = 32;
}

// ── Scratch buffer header (64 bytes) ──────────────────────────────
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct KvScratchHeader {
    pub metadata: KvScratchMetadataAbi,
    pub control: KvScratchDeviceControl,
}

impl Default for KvScratchHeader {
    fn default() -> Self {
        Self {
            metadata: KvScratchMetadataAbi::default(),
            control: KvScratchDeviceControl::default(),
        }
    }
}

impl KvScratchHeader {
    /// Return a reference to the buffer-state AtomicU32 alias within the
    /// opaque device-control block (offset 0).
    fn state_atomic(&self) -> &AtomicU32 {
        let ptr = &self.control._opaque_atomic_storage as *const [u8; 32]
            as *const AtomicU32;
        unsafe { &*ptr }
    }

    pub fn new() -> Self {
        let s = Self::default();
        // The zeroed state reads as KV_STATE_EMPTY (0), but write it
        // explicitly so the initialisation is visible even when the
        // opaque block is later changed to non-zero defaults.
        s.state_atomic().store(KvBufferState::Empty as u32, Ordering::Relaxed);
        s
    }

    pub fn load_state(&self) -> KvBufferState {
        let v = self.state_atomic().load(Ordering::Acquire);
        KvBufferState::from_u32(v).unwrap_or(KvBufferState::Poisoned)
    }

    pub fn store_state(&self, s: KvBufferState) {
        self.state_atomic().store(s as u32, Ordering::Release);
    }

    pub fn compare_exchange_state(
        &self,
        expected: KvBufferState,
        desired: KvBufferState,
    ) -> Result<KvBufferState, KvBufferState> {
        let prev = self.state_atomic().compare_exchange(
            expected as u32,
            desired as u32,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
        match prev {
            Ok(v) => Ok(KvBufferState::from_u32(v).unwrap_or(KvBufferState::Poisoned)),
            Err(v) => Err(KvBufferState::from_u32(v).unwrap_or(KvBufferState::Poisoned)),
        }
    }
}

// ── Queue counter slot (padded) ──────────────────────────────────
#[repr(C)]
#[derive(Debug)]
pub struct KvQueueCounterSlot {
    pub value: AtomicU32,
    _pad: [u32; QUEUE_COUNTER_STRIDE_U32 - 1],
}

impl Default for KvQueueCounterSlot {
    fn default() -> Self {
        Self {
            value: AtomicU32::new(0),
            _pad: [0u32; QUEUE_COUNTER_STRIDE_U32 - 1],
        }
    }
}

// ── Prefetch request (16 u32 fields, 64 bytes) ────────────────────
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct KvPrefetchRequest {
    pub request_id: u32,
    pub session_id: u32,
    pub sequence_id: u32,
    pub target_layer: u32,
    pub token_epoch: u32,
    pub kv_generation: u32,
    pub page_table_generation: u32,
    pub scratch_set_index: u32,
    pub source_k_base: u32,
    pub source_v_base: u32,
    pub source_scale_base: u32,
    pub source_page_count: u32,
    pub destination_k_offset: u32,
    pub destination_v_offset: u32,
    pub deadline_ticks: u32,
    pub flags: u32,
}

impl KvPrefetchRequest {
    pub const SIZE: usize = 64;
    pub const fn zeroed() -> Self {
        Self {
            request_id: 0,
            session_id: 0,
            sequence_id: 0,
            target_layer: 0,
            token_epoch: 0,
            kv_generation: 0,
            page_table_generation: 0,
            scratch_set_index: 0,
            source_k_base: 0,
            source_v_base: 0,
            source_scale_base: 0,
            source_page_count: 0,
            destination_k_offset: 0,
            destination_v_offset: 0,
            deadline_ticks: 0,
            flags: 0,
        }
    }
}

// ── Bounded lock-free ring queue ─────────────────────────────────
#[repr(C)]
#[derive(Debug)]
pub struct KvPrefetchQueueAbi {
    pub enqueue_pos: KvQueueCounterSlot,
    pub dequeue_pos: KvQueueCounterSlot,
    pub completed_pos: KvQueueCounterSlot,
    pub dropped_count: KvQueueCounterSlot,
    pub overflow_count: u32,
    pub capacity: u32,
    pub mask: u32,
    pub abi_version: u32,
    pub entries: [KvPrefetchRequest; 16],
}

impl Default for KvPrefetchQueueAbi {
    fn default() -> Self {
        Self {
            enqueue_pos: KvQueueCounterSlot::default(),
            dequeue_pos: KvQueueCounterSlot::default(),
            completed_pos: KvQueueCounterSlot::default(),
            dropped_count: KvQueueCounterSlot::default(),
            overflow_count: 0,
            capacity: 16,
            mask: 15,
            abi_version: 1,
            entries: [KvPrefetchRequest::zeroed(); 16],
        }
    }
}

impl KvPrefetchQueueAbi {
    pub const CAPACITY: usize = 16;
    pub const ABI_VERSION: u32 = 1;

    pub fn new() -> Self {
        Self::default()
    }

    /// Enqueue a prefetch request. Returns slot index on success, None on full.
    pub fn enqueue(&self, req: &KvPrefetchRequest) -> Option<u32> {
        let enq = self.enqueue_pos.value.fetch_add(1, Ordering::AcqRel);
        let deq = self.dequeue_pos.value.load(Ordering::Acquire);
        if enq.wrapping_sub(deq) >= self.capacity {
            // Rollback
            self.enqueue_pos.value.store(enq, Ordering::Release);
            self.dropped_count.value.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        let idx = (enq & self.mask) as usize;
        unsafe {
            let slot = &self.entries[idx] as *const KvPrefetchRequest as *mut KvPrefetchRequest;
            slot.write_volatile(req.clone());
        }
        Some(enq)
    }

    /// Dequeue a prefetch request. Returns (id, request) or None if empty.
    pub fn dequeue(&self) -> Option<(u32, KvPrefetchRequest)> {
        let deq = self.dequeue_pos.value.load(Ordering::Acquire);
        let enq = self.enqueue_pos.value.load(Ordering::Acquire);
        if deq == enq {
            return None;
        }
        let idx = (deq & self.mask) as usize;
        let req = unsafe {
            (self.entries.as_ptr().add(idx) as *const KvPrefetchRequest).read_volatile()
        };
        self.dequeue_pos.value.store(deq.wrapping_add(1), Ordering::Release);
        Some((deq, req))
    }

    pub fn mark_completed(&self) {
        self.completed_pos.value.fetch_add(1, Ordering::Release);
    }

    pub fn depth(&self) -> u32 {
        let enq = self.enqueue_pos.value.load(Ordering::Acquire);
        let deq = self.dequeue_pos.value.load(Ordering::Acquire);
        enq.wrapping_sub(deq)
    }
}

// ── Scratch set — pair of K/V staging buffers ─────────────────────
#[derive(Debug, Clone)]
pub struct KvScratchSet {
    pub header_offset: usize,
    pub k_offset: usize,
    pub v_offset: usize,
    pub k_bytes: usize,
    pub v_bytes: usize,
}

// ── Prefetch arena — double-buffered ──────────────────────────────
#[derive(Debug)]
pub struct KvPrefetchArena {
    pub sets: [KvScratchSet; 2],
    pub queue_offset: usize,
    pub telemetry_offset: usize,
    pub total_bytes: usize,
    pub active_set: usize,
}

impl KvPrefetchArena {
    pub fn bytes_per_set(max_context: u32, num_kv_heads: u32, head_dim: u32) -> (usize, usize) {
        let per_position = (num_kv_heads * head_dim) as usize;
        let k_bytes = (max_context as usize) * per_position * 2;
        let v_bytes = k_bytes;
        (k_bytes, v_bytes)
    }

    fn align_up(x: usize, a: usize) -> usize {
        (x + a - 1) & !(a - 1)
    }

    pub fn total_arena_bytes(
        scratch_set_count: usize,
        k_bytes: usize,
        v_bytes: usize,
    ) -> usize {
        const SIMD_ALIGN: usize = 256;
        const PAGE_ALIGN: usize = 4096;

        let header_size = Self::align_up(size_of::<KvScratchHeader>(), PAGE_ALIGN);
        let k_aligned = Self::align_up(k_bytes, SIMD_ALIGN);
        let v_aligned = Self::align_up(v_bytes, SIMD_ALIGN);
        let set_bytes = scratch_set_count * (header_size + k_aligned + v_aligned);
        let queue_bytes = Self::align_up(size_of::<KvPrefetchQueueAbi>(), PAGE_ALIGN);
        let telemetry_bytes = PAGE_ALIGN;
        set_bytes + queue_bytes + telemetry_bytes
    }

    pub fn layout(
        max_context: u32,
        num_kv_heads: u32,
        head_dim: u32,
    ) -> Self {
        const SIMD_ALIGN: usize = 256;
        const PAGE_ALIGN: usize = 4096;
        let (k_bytes, v_bytes) = Self::bytes_per_set(max_context, num_kv_heads, head_dim);
        let header_size = Self::align_up(size_of::<KvScratchHeader>(), PAGE_ALIGN);
        let k_aligned = Self::align_up(k_bytes, SIMD_ALIGN);
        let v_aligned = Self::align_up(v_bytes, SIMD_ALIGN);
        let set_stride = header_size + k_aligned + v_aligned;

        let set0 = KvScratchSet {
            header_offset: 0,
            k_offset: header_size,
            v_offset: header_size + k_aligned,
            k_bytes,
            v_bytes,
        };
        let set1 = KvScratchSet {
            header_offset: set_stride,
            k_offset: set_stride + header_size,
            v_offset: set_stride + header_size + k_aligned,
            k_bytes,
            v_bytes,
        };
        let queue_offset = Self::align_up(2 * set_stride, PAGE_ALIGN);
        let telemetry_offset = queue_offset + Self::align_up(size_of::<KvPrefetchQueueAbi>(), PAGE_ALIGN);
        let total = telemetry_offset + PAGE_ALIGN;

        Self {
            sets: [set0, set1],
            queue_offset,
            telemetry_offset,
            total_bytes: total,
            active_set: 0,
        }
    }
}

// ── Epoch receipt counters ────────────────────────────────────────
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct KvEpochReceipt {
    // Counters written atomically by GPU during epoch (plain on CPU side)
    pub requests_claimed: u32,
    pub requests_ready_consumable: u32,
    pub requests_canceled: u32,
    pub requests_poisoned: u32,
    pub requests_bypassed: u32,
    pub staging_consumptions: u32,
    pub late_ready_discarded_diagnostic: u32,
    pub duplicate_write_detected: u32,
    pub requests_unresolved: u32,
    pub epoch_fatal_fault: u32,
    pub epoch_fatal_fault_generation: u32,
    pub epoch_fatal_fault_request_id: u32,
    pub epoch_secondary_fault_count: u32,
    pub epoch_fatal_claim: u32,
    // Completion fields (written by GPU at end of epoch)
    pub outcome: u32,
    pub epoch_id: u64,
    pub kv_bytes_prefetched: u64,
    pub kv_bytes_consumed: u64,
    pub decode_duration_us: u64,
    pub prefetch_duration_us: u64,
}

impl Default for KvEpochReceipt {
    fn default() -> Self {
        Self {
            requests_claimed: 0,
            requests_ready_consumable: 0,
            requests_canceled: 0,
            requests_poisoned: 0,
            requests_bypassed: 0,
            staging_consumptions: 0,
            late_ready_discarded_diagnostic: 0,
            duplicate_write_detected: 0,
            requests_unresolved: 0,
            epoch_fatal_fault: 0,
            epoch_fatal_fault_generation: 0,
            epoch_fatal_fault_request_id: 0,
            epoch_secondary_fault_count: 0,
            epoch_fatal_claim: 0,
            outcome: 0,
            epoch_id: 0,
            kv_bytes_prefetched: 0,
            kv_bytes_consumed: 0,
            decode_duration_us: 0,
            prefetch_duration_us: 0,
        }
    }
}

// ── Epoch control registers ──────────────────────────────────────
#[derive(Debug)]
#[repr(C)]
pub struct EpochControl {
    pub epoch_close_requested: AtomicU32,
    pub epoch_enqueue_limit: AtomicU32,
    pub epoch_fatal_claim: AtomicU32,
    pub epoch_fatal_fault: AtomicU32,
    pub epoch_fatal_fault_generation: AtomicU32,
    pub epoch_fatal_fault_request_id: AtomicU32,
}

impl Default for EpochControl {
    fn default() -> Self {
        Self {
            epoch_close_requested: AtomicU32::new(0),
            epoch_enqueue_limit: AtomicU32::new(0),
            epoch_fatal_claim: AtomicU32::new(0),
            epoch_fatal_fault: AtomicU32::new(0),
            epoch_fatal_fault_generation: AtomicU32::new(0),
            epoch_fatal_fault_request_id: AtomicU32::new(0),
        }
    }
}

// ── Token timing metric ──────────────────────────────────────────
#[derive(Debug, Clone, Copy)]
pub struct TokenMetric {
    pub token_index: u64,
    pub request_submission_us: u64,
    pub prefetch_completion_us: u64,
    pub decode_begin_us: u64,
    pub decode_end_us: u64,
    pub was_cache_hit: bool,
    pub hit_layer_count: u32,
    pub fallback_count: u32,
}

// ── Sliding interleave window ────────────────────────────────────
#[derive(Debug, Clone)]
pub struct InterleaveWindow {
    pub metrics: VecDeque<TokenMetric>,
    pub window_size: usize,
}

impl InterleaveWindow {
    pub fn new(window_size: usize) -> Self {
        Self {
            metrics: VecDeque::with_capacity(window_size),
            window_size,
        }
    }

    pub fn push(&mut self, metric: TokenMetric) {
        if self.metrics.len() >= self.window_size {
            self.metrics.pop_front();
        }
        self.metrics.push_back(metric);
    }

    pub fn len(&self) -> usize {
        self.metrics.len()
    }

    pub fn is_empty(&self) -> bool {
        self.metrics.is_empty()
    }
}

// ── Aggregated window statistics ─────────────────────────────────
#[derive(Debug, Clone, Copy, Default)]
pub struct WindowStats {
    pub p50_us: f64,
    pub p95_us: f64,
    pub request_admission_rate: f64,
    pub prefetch_hit_rate: f64,
    pub fallback_rate: f64,
}

// ── Admission state machine ──────────────────────────────────────
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdmissionState {
    Disabled = 0,
    WarmupBaseline = 1,
    QualificationInterleave = 2,
    InterleaveActive = 3,
    InterleaveCoolingDown = 4,
    ForcedBaseline = 5,
}

// ── Pipeline telemetry tracker ───────────────────────────────────
#[derive(Debug, Clone)]
pub struct PipelineTelemetryTracker {
    pub state: AdmissionState,
    pub window: InterleaveWindow,
    pub baseline_stats: WindowStats,
    pub tokens_until_requalification: u32,
    pub epoch_count: u64,
}

impl PipelineTelemetryTracker {
    pub const DEFAULT_WINDOW_SIZE: usize = 64;

    pub fn default() -> Self {
        Self::new(Self::DEFAULT_WINDOW_SIZE)
    }

    pub fn new(window_size: usize) -> Self {
        Self {
            state: AdmissionState::Disabled,
            window: InterleaveWindow::new(window_size),
            baseline_stats: WindowStats::default(),
            tokens_until_requalification: 0,
            epoch_count: 0,
        }
    }

    /// Evaluate whether the pipeline should transition to a different
    /// admission state based on the current window statistics.
    pub fn evaluate_admission(&mut self, current_stats: &WindowStats) -> AdmissionState {
        match self.state {
            AdmissionState::Disabled => {
                // No transition while disabled — external control must
                // set WarmupBaseline or ForcedBaseline explicitly.
                AdmissionState::Disabled
            }
            AdmissionState::WarmupBaseline => {
                // Once the window is populated, move to qualification.
                if self.window.len() >= self.window.window_size {
                    AdmissionState::QualificationInterleave
                } else {
                    AdmissionState::WarmupBaseline
                }
            }
            AdmissionState::QualificationInterleave => {
                // Compare against baseline.  If interleave performs
                // within threshold, activate; otherwise stay in
                // qualification or fall back.
                if current_stats.p95_us <= self.baseline_stats.p95_us * 1.05
                    && current_stats.fallback_rate < 0.05
                {
                    AdmissionState::InterleaveActive
                } else if self.tokens_until_requalification > 0 {
                    self.tokens_until_requalification =
                        self.tokens_until_requalification.saturating_sub(1);
                    AdmissionState::QualificationInterleave
                } else {
                    AdmissionState::ForcedBaseline
                }
            }
            AdmissionState::InterleaveActive => {
                // Monitoring: if p95 or fallback rate degrade
                // significantly, start cooling down.
                if current_stats.p95_us > self.baseline_stats.p95_us * 1.15
                    || current_stats.fallback_rate > 0.10
                {
                    AdmissionState::InterleaveCoolingDown
                } else {
                    AdmissionState::InterleaveActive
                }
            }
            AdmissionState::InterleaveCoolingDown => {
                // Cool-down period then back to baseline.
                if self.tokens_until_requalification > 0 {
                    self.tokens_until_requalification =
                        self.tokens_until_requalification.saturating_sub(1);
                    AdmissionState::InterleaveCoolingDown
                } else {
                    AdmissionState::ForcedBaseline
                }
            }
            AdmissionState::ForcedBaseline => {
                // Remain in baseline until external reset.
                AdmissionState::ForcedBaseline
            }
        }
    }
}

// ── Device capability key (probe result cache) ───────────────────
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceCapabilityKey {
    pub gpu_family: u32,
    pub gpu_variant: u32,
    pub driver_version_major: u32,
    pub driver_version_minor: u32,
}

impl DeviceCapabilityKey {
    pub fn new(gpu_family: u32, gpu_variant: u32) -> Self {
        Self {
            gpu_family,
            gpu_variant,
            driver_version_major: 0,
            driver_version_minor: 0,
        }
    }

    pub fn with_driver(mut self, major: u32, minor: u32) -> Self {
        self.driver_version_major = major;
        self.driver_version_minor = minor;
        self
    }
}

/// Probe whether the device supports 64-bit atomic operations by
/// compiling and dispatching a minimal test kernel.
///
/// Returns `true` when the device responds with a correct result,
/// `false` on any failure or unsupported feature.
pub fn probe_64bit_atomic_execution(gpu_family: u32, gpu_variant: u32) -> bool {
    // A real implementation would:
    //  1. Compile a minimal Metal shader that reads/writes a 64-bit
    //     atomic counter.
    //  2. Dispatch a single threadgroup with one thread.
    //  3. Check the result equals the expected value.
    //
    // For now, return known OS support based on GPU family.
    // Apple Silicon (family >= 7) supports 64-bit buffer atomics;
    // older families are likely unsupported.
    gpu_family >= 7 && gpu_variant != 0
}

// ── Epoch conservation validation ────────────────────────────────
//
/// Validate that the number of claimed requests is conserved against
/// recorded outcomes within an epoch.  Returns `Ok(())` when the
/// epoch exhibits no lost claims or phantom completions.
pub fn validate_epoch_conservation(receipt: &KvEpochReceipt) -> Result<(), u32> {
    let claimed = receipt.requests_claimed;
    let consumable = receipt.requests_ready_consumable;
    let canceled = receipt.requests_canceled;
    let poisoned = receipt.requests_poisoned;
    let bypassed = receipt.requests_bypassed;

    let accounted = consumable
        .wrapping_add(canceled)
        .wrapping_add(poisoned)
        .wrapping_add(bypassed);

    if claimed == accounted {
        Ok(())
    } else {
        Err(claimed.wrapping_sub(accounted))
    }
}

// ── Concurrency qualification ────────────────────────────────────
//
/// Evaluate whether the pipeline should proceed with interleave
/// concurrency at the requested level based on device capability
/// and epoch health.
pub fn evaluate_concurrency_qualification(
    device_key: &DeviceCapabilityKey,
    _requested_decode_workers: u32,
    requested_cache_workers: u32,
    epoch_health: &KvEpochReceipt,
) -> bool {
    // Step 1: Gate on 64-bit atomic support on the target GPU.
    if !probe_64bit_atomic_execution(device_key.gpu_family, device_key.gpu_variant) {
        return false;
    }

    // Step 2: Reject concurrency > 2 cache workers unless the
    // epoch has zero fatal faults and at most one secondary fault.
    if requested_cache_workers > 2 {
        let fatal_faults = epoch_health.epoch_fatal_fault;
        let secondary_faults = epoch_health.epoch_secondary_fault_count;
        if fatal_faults > 0 || secondary_faults > 1 {
            return false;
        }
    }

    // Step 3: A high unresolved-request count (>25%) is a sign of
    // pipeline back-pressure — do not increase concurrency.
    let claimed = epoch_health.requests_claimed;
    if claimed > 0 {
        let unresolved = epoch_health.requests_unresolved;
        if unresolved > 0 && unresolved * 4 > claimed {
            return false;
        }
    }

    true
}

// ── Pipeline mode enum ───────────────────────────────────────────
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KvPipelineMode {
    #[default]
    Disabled = 0,
    FullLayerDoubleBuffer = 1,
    PageGroupStreaming = 2,
    Adaptive = 3,
    QualificationOnly = 4,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WorkerTopology {
    #[default]
    SingleDecode = 0,
    TwoDecodeOneCache = 1,
    OneDecodeTwoCache = 2,
    TwoDecodeTwoCache = 3,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PipelineCorrectnessStatus {
    #[default]
    Unknown = 0,
    Verified = 1,
    Degraded = 2,
    FallbackActive = 3,
}

// ── Pipeline receipt ─────────────────────────────────────────────
#[derive(Debug, Clone)]
pub struct KvPipelineReceipt {
    pub session_id: u64,
    pub sequence_id: u64,
    pub token_epoch: u64,
    pub model_layer_count: u32,
    pub prefetch_requests: u32,
    pub prefetch_hits: u32,
    pub prefetch_misses: u32,
    pub fallback_decompressions: u32,
    pub canceled_prefetches: u32,
    pub stale_prefetches: u32,
    pub historical_kv_bytes_prefetched: u64,
    pub historical_kv_bytes_consumed: u64,
    pub decode_lane_us: u64,
    pub prefetch_lane_us: u64,
    pub overlap_window_us: u64,
    pub stall_wait_us: u64,
    pub fallback_us: u64,
    pub estimated_bandwidth_contention_penalty_us: u64,
    pub pipeline_mode: KvPipelineMode,
    pub selected_worker_topology: WorkerTopology,
    pub correctness_status: PipelineCorrectnessStatus,
}

impl Default for KvPipelineReceipt {
    fn default() -> Self {
        Self {
            session_id: 0,
            sequence_id: 0,
            token_epoch: 0,
            model_layer_count: 0,
            prefetch_requests: 0,
            prefetch_hits: 0,
            prefetch_misses: 0,
            fallback_decompressions: 0,
            canceled_prefetches: 0,
            stale_prefetches: 0,
            historical_kv_bytes_prefetched: 0,
            historical_kv_bytes_consumed: 0,
            decode_lane_us: 0,
            prefetch_lane_us: 0,
            overlap_window_us: 0,
            stall_wait_us: 0,
            fallback_us: 0,
            estimated_bandwidth_contention_penalty_us: 0,
            pipeline_mode: KvPipelineMode::Disabled,
            selected_worker_topology: WorkerTopology::SingleDecode,
            correctness_status: PipelineCorrectnessStatus::Unknown,
        }
    }
}

// ── Tier1 telemetry (always available) ───────────────────────────
#[derive(Debug, Clone)]
pub struct Tier1Telemetry {
    pub token_latency_ms: f32,
    pub p95_duration_ms: f32,
    pub prefetch_hit_rate: f32,
    pub fallback_rate: f32,
    pub queue_residency_ticks: u32,
    pub scratch_buffer_stalls: u32,
}

impl Default for Tier1Telemetry {
    fn default() -> Self {
        Self {
            token_latency_ms: 0.0,
            p95_duration_ms: 0.0,
            prefetch_hit_rate: 0.0,
            fallback_rate: 0.0,
            queue_residency_ticks: 0,
            scratch_buffer_stalls: 0,
        }
    }
}

// ── Tier2 telemetry (counter-dependent) ──────────────────────────
#[derive(Debug, Clone)]
pub struct Tier2Telemetry {
    pub is_available: bool,
    pub memory_stall_ratio: f32,
}

impl Default for Tier2Telemetry {
    fn default() -> Self {
        Self { is_available: false, memory_stall_ratio: 0.0 }
    }
}

// ── Interleave plan ──────────────────────────────────────────────
pub struct KvInterleavePlan {
    pub enabled: bool,
    pub mode: KvPipelineMode,
    pub scratch_set_count: u8,
    pub prefetch_queue_capacity: u16,
    pub cache_worker_count: u8,
    pub decode_worker_count: u8,
    pub elastic_worker_count: u8,
    pub page_group_strategy: KvPageGroupStrategy,
    pub fallback_policy: KvFallbackPolicy,
    pub readiness_spin_budget: u32,
    pub minimum_overlap_gain_percent: f32,
    pub max_p95_regression_percent: f32,
    pub required_memory_bytes: u64,
    pub required_atomic_features: AtomicFeatureSet,
}

impl KvInterleavePlan {
    /// Build a baseline interleave plan from device capabilities.
    #[cfg(feature = "metal-dispatch")]
    pub fn build(_device: &metal::Device) -> Self {
        // Probe 64-bit atomics; in production read the family from the device.
        let has_atomics = probe_64bit_atomic_execution(0, 1);

        Self {
            enabled: has_atomics,
            mode: if has_atomics { KvPipelineMode::Adaptive } else { KvPipelineMode::Disabled },
            scratch_set_count: 2,
            prefetch_queue_capacity: 16,
            cache_worker_count: 1,
            decode_worker_count: 1,
            elastic_worker_count: 0,
            page_group_strategy: KvPageGroupStrategy::SingleBlock,
            fallback_policy: KvFallbackPolicy::SynchronousSelfHelp,
            readiness_spin_budget: 1000,
            minimum_overlap_gain_percent: 10.0,
            max_p95_regression_percent: 5.0,
            required_memory_bytes: 0,
            required_atomic_features: AtomicFeatureSet::FullDeviceScope,
        }
    }
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvPageGroupStrategy {
    SingleBlock = 0,
    PerHeadGroup = 1,
    PerContiguousSpan = 2,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvFallbackPolicy {
    SynchronousSelfHelp = 0,
    StealCacheWorker = 1,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomicFeatureSet {
    BasicDeviceScope = 0,
    FullDeviceScope = 1,
}

// ── Layout tests ─────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::size_of;

    // ── Size assertions ──────────────────────────────────────────

    #[test]
    fn kv_scratch_metadata_abi_size() {
        assert_eq!(size_of::<KvScratchMetadataAbi>(), 32);
    }

    #[test]
    fn kv_scratch_device_control_size() {
        assert_eq!(size_of::<KvScratchDeviceControl>(), 32);
    }

    #[test]
    fn kv_scratch_header_size() {
        assert_eq!(size_of::<KvScratchHeader>(), 64);
    }

    #[test]
    fn queue_counter_slot_size() {
        assert_eq!(size_of::<KvQueueCounterSlot>(), 128);
    }

    #[test]
    fn prefetch_request_size() {
        assert_eq!(size_of::<KvPrefetchRequest>(), 64);
    }

    // ── Enum / constant values ───────────────────────────────────

    #[test]
    fn buffer_state_enum_values() {
        assert_eq!(KvBufferState::Empty as u32, 0);
        assert_eq!(KvBufferState::Queued as u32, 1);
        assert_eq!(KvBufferState::Filling as u32, 2);
        assert_eq!(KvBufferState::Ready as u32, 3);
        assert_eq!(KvBufferState::Consuming as u32, 7);
        assert_eq!(KvBufferState::Poisoned as u32, 5);
        assert_eq!(KvBufferState::Canceled as u32, 6);
        assert_eq!(KvBufferState::Reclaimable as u32, 8);
    }

    #[test]
    fn claim_constants() {
        assert_eq!(CLAIM_UNOWNED, 0);
        assert_eq!(CLAIM_HELPER, 1);
        assert_eq!(CLAIM_DECODE_FALLBACK, 2);
        assert_eq!(CLAIM_DECODE_CONSUMER, 3);
    }

    #[test]
    fn outcome_constants() {
        assert_eq!(OUTCOME_NONE, 0);
        assert_eq!(OUTCOME_READY_CONSUMABLE, 1);
        assert_eq!(OUTCOME_CANCELED, 2);
        assert_eq!(OUTCOME_POISONED, 3);
        assert_eq!(OUTCOME_BYPASSED, 4);
    }

    #[test]
    fn fault_constants() {
        assert_eq!(FAULT_NONE, 0);
        assert_eq!(FAULT_HANDOFF_INTEGRITY, 1);
        assert_eq!(FAULT_INVALID_READY_STATE, 2);
        assert_eq!(FAULT_GENERATION_MISMATCH, 3);
        assert_eq!(FAULT_UNRECOGNIZED_OUTCOME, 4);
    }

    #[test]
    fn kv_state_constants() {
        assert_eq!(KV_STATE_EMPTY, 0);
        assert_eq!(KV_STATE_QUEUED, 1);
        assert_eq!(KV_STATE_FILLING, 2);
        assert_eq!(KV_STATE_READY, 3);
        assert_eq!(KV_STATE_CONSUMING, 7);
        assert_eq!(KV_STATE_RECLAIMABLE, 8);
        assert_eq!(KV_STATE_POISONED, 5);
        assert_eq!(KV_STATE_CANCELED, 6);
    }

    // ── State transitions ────────────────────────────────────────

    #[test]
    fn scratch_header_state_transitions() {
        let h = KvScratchHeader::new();
        assert_eq!(h.load_state(), KvBufferState::Empty);
        assert!(h.compare_exchange_state(KvBufferState::Empty, KvBufferState::Queued).is_ok());
        assert_eq!(h.load_state(), KvBufferState::Queued);
        assert!(h.compare_exchange_state(KvBufferState::Queued, KvBufferState::Filling).is_ok());
        assert_eq!(h.load_state(), KvBufferState::Filling);
        assert!(h.compare_exchange_state(KvBufferState::Filling, KvBufferState::Ready).is_ok());
        assert_eq!(h.load_state(), KvBufferState::Ready);
        assert!(h.compare_exchange_state(KvBufferState::Ready, KvBufferState::Consuming).is_ok());
        assert_eq!(h.load_state(), KvBufferState::Consuming);
        assert!(h.compare_exchange_state(KvBufferState::Consuming, KvBufferState::Reclaimable).is_ok());
        assert_eq!(h.load_state(), KvBufferState::Reclaimable);
        assert!(h.compare_exchange_state(KvBufferState::Reclaimable, KvBufferState::Empty).is_ok());
        assert_eq!(h.load_state(), KvBufferState::Empty);
    }

    #[test]
    fn generation_tagged_reclaim() {
        // Simulate a generation check during reclaim: the header
        // transitions through Ready -> Consuming -> Reclaimable,
        // and the kv_generation in metadata is preserved.
        let h = KvScratchHeader::new();
        assert!(h.compare_exchange_state(KvBufferState::Empty, KvBufferState::Queued).is_ok());
        assert!(h.compare_exchange_state(KvBufferState::Queued, KvBufferState::Filling).is_ok());
        assert!(h.compare_exchange_state(KvBufferState::Filling, KvBufferState::Ready).is_ok());

        // Set the generation tag in metadata.
        unsafe {
            let ptr = &h.metadata as *const KvScratchMetadataAbi as *mut KvScratchMetadataAbi;
            (*ptr).kv_generation = 7;
        }
        assert_eq!(h.metadata.kv_generation, 7);

        assert!(h.compare_exchange_state(KvBufferState::Ready, KvBufferState::Consuming).is_ok());
        assert!(h.compare_exchange_state(KvBufferState::Consuming, KvBufferState::Reclaimable).is_ok());
        assert_eq!(h.load_state(), KvBufferState::Reclaimable);

        // Verify generation is preserved through the reclaim cycle.
        assert_eq!(h.metadata.kv_generation, 7);
    }

    #[test]
    fn outcome_cas_protocol() {
        // Simulate the outcome CAS protocol with ABI constants.
        let outcome = AtomicU32::new(OUTCOME_NONE);

        // Decode consumer claims and marks consumable.
        assert!(outcome.compare_exchange(
            OUTCOME_NONE,
            OUTCOME_READY_CONSUMABLE,
            Ordering::AcqRel,
            Ordering::Acquire,
        ).is_ok());
        assert_eq!(outcome.load(Ordering::Acquire), OUTCOME_READY_CONSUMABLE);

        // CAS from consumable -> none must fail (already consumed).
        assert!(outcome.compare_exchange(
            OUTCOME_NONE,
            OUTCOME_BYPASSED,
            Ordering::AcqRel,
            Ordering::Acquire,
        ).is_err());

        // Helper could still cancel from consumable.
        assert!(outcome.compare_exchange(
            OUTCOME_READY_CONSUMABLE,
            OUTCOME_CANCELED,
            Ordering::AcqRel,
            Ordering::Acquire,
        ).is_ok());
        assert_eq!(outcome.load(Ordering::Acquire), OUTCOME_CANCELED);
    }

    #[test]
    fn from_u32_all_variants() {
        assert_eq!(KvBufferState::from_u32(0), Some(KvBufferState::Empty));
        assert_eq!(KvBufferState::from_u32(1), Some(KvBufferState::Queued));
        assert_eq!(KvBufferState::from_u32(2), Some(KvBufferState::Filling));
        assert_eq!(KvBufferState::from_u32(3), Some(KvBufferState::Ready));
        assert_eq!(KvBufferState::from_u32(7), Some(KvBufferState::Consuming));
        assert_eq!(KvBufferState::from_u32(5), Some(KvBufferState::Poisoned));
        assert_eq!(KvBufferState::from_u32(6), Some(KvBufferState::Canceled));
        assert_eq!(KvBufferState::from_u32(8), Some(KvBufferState::Reclaimable));
        assert_eq!(KvBufferState::from_u32(4), None);
        assert_eq!(KvBufferState::from_u32(9), None);
    }

    // ── Queue operations ─────────────────────────────────────────

    #[test]
    fn prefetch_queue_enqueue_dequeue() {
        let q = KvPrefetchQueueAbi::new();
        assert_eq!(q.depth(), 0);

        let req = KvPrefetchRequest {
            request_id: 42,
            session_id: 1,
            sequence_id: 2,
            target_layer: 3,
            token_epoch: 0,
            kv_generation: 1,
            page_table_generation: 0,
            scratch_set_index: 1,
            source_k_base: 0x1000,
            source_v_base: 0x2000,
            source_scale_base: 0x3000,
            source_page_count: 8,
            destination_k_offset: 0,
            destination_v_offset: 0x8000,
            deadline_ticks: 1000,
            flags: 0,
        };

        let slot = q.enqueue(&req);
        assert!(slot.is_some());
        assert_eq!(q.depth(), 1);

        let (id, dequeued) = q.dequeue().unwrap();
        assert_eq!(id, 0);
        assert_eq!(dequeued.request_id, 42);
        assert_eq!(dequeued.target_layer, 3);
        assert_eq!(q.depth(), 0);
    }

    #[test]
    fn prefetch_queue_overflow() {
        let q = KvPrefetchQueueAbi::new();
        for i in 0..KvPrefetchQueueAbi::CAPACITY {
            let req = KvPrefetchRequest {
                request_id: i as u32,
                ..KvPrefetchRequest::zeroed()
            };
            assert!(q.enqueue(&req).is_some(), "enqueue {i}");
        }
        let req = KvPrefetchRequest {
            request_id: 999,
            ..KvPrefetchRequest::zeroed()
        };
        assert!(q.enqueue(&req).is_none(), "overflow should return None");
    }

    // ── Arena layout ─────────────────────────────────────────────

    #[test]
    fn kv_prefetch_arena_layout() {
        let ctx = 2048;
        let heads = 8;
        let hdim = 512;
        let (k_bytes, v_bytes) = KvPrefetchArena::bytes_per_set(ctx, heads, hdim);
        assert_eq!(k_bytes, 2048 * 8 * 512 * 2);
        assert_eq!(v_bytes, k_bytes);

        let total = KvPrefetchArena::total_arena_bytes(2, k_bytes, v_bytes);
        let a = KvPrefetchArena::layout(ctx, heads, hdim);
        assert_eq!(a.total_bytes, total);
        assert_eq!(a.sets.len(), 2);
        assert_eq!(a.active_set, 0);
    }

    // ── Epoch functions ──────────────────────────────────────────

    #[test]
    fn epoch_conservation_balanced() {
        let mut receipt = KvEpochReceipt::default();
        receipt.requests_claimed = 10;
        receipt.requests_ready_consumable = 7;
        receipt.requests_canceled = 2;
        receipt.requests_bypassed = 1;
        assert!(validate_epoch_conservation(&receipt).is_ok());
    }

    #[test]
    fn epoch_conservation_imbalanced() {
        let mut receipt = KvEpochReceipt::default();
        receipt.requests_claimed = 10;
        receipt.requests_ready_consumable = 5;
        // Only 5 accounted for instead of 10.
        let err = validate_epoch_conservation(&receipt);
        assert!(err.is_err());
    }

    #[test]
    fn concurrency_qualification_gates_on_atomics() {
        let key = DeviceCapabilityKey::new(0, 0); // old GPU, fails probe
        let health = KvEpochReceipt::default();
        assert!(!evaluate_concurrency_qualification(&key, 1, 1, &health));
    }

    #[test]
    fn concurrency_qualification_passes_apple_silicon() {
        let key = DeviceCapabilityKey::new(7, 1); // Apple silicon
        let mut health = KvEpochReceipt::default();
        health.requests_claimed = 10;
        health.requests_unresolved = 1;
        assert!(evaluate_concurrency_qualification(&key, 1, 1, &health));
    }

    #[test]
    fn concurrency_qualification_rejects_high_unresolved() {
        let key = DeviceCapabilityKey::new(7, 1);
        let mut health = KvEpochReceipt::default();
        health.requests_claimed = 10;
        health.requests_unresolved = 5; // 50% -> exceeds 25%
        assert!(!evaluate_concurrency_qualification(&key, 1, 2, &health));
    }

    // ── Telemetry tracker ────────────────────────────────────────

    #[test]
    fn telemetry_tracker_warmup_transition() {
        let mut tracker = PipelineTelemetryTracker::new(16);
        assert_eq!(tracker.state, AdmissionState::Disabled);
        tracker.state = AdmissionState::WarmupBaseline;

        // Fill the window.
        for i in 0..16 {
            tracker.window.push(TokenMetric {
                token_index: i,
                request_submission_us: 0,
                prefetch_completion_us: 100,
                decode_begin_us: 100,
                decode_end_us: 200,
                was_cache_hit: true,
                hit_layer_count: 1,
                fallback_count: 0,
            });
        }

        let stats = WindowStats {
            p50_us: 150.0,
            p95_us: 200.0,
            request_admission_rate: 1.0,
            prefetch_hit_rate: 0.9,
            fallback_rate: 0.01,
        };

        let next = tracker.evaluate_admission(&stats);
        assert_eq!(next, AdmissionState::QualificationInterleave);
    }

    #[test]
    fn telemetry_tracker_interleave_active_maintains() {
        let mut tracker = PipelineTelemetryTracker::new(16);
        tracker.state = AdmissionState::InterleaveActive;
        tracker.baseline_stats.p95_us = 200.0;

        let good_stats = WindowStats {
            p95_us: 210.0,
            fallback_rate: 0.02,
            ..WindowStats::default()
        };

        let next = tracker.evaluate_admission(&good_stats);
        assert_eq!(next, AdmissionState::InterleaveActive);
    }

    #[test]
    fn telemetry_tracker_interleave_cooling_down() {
        let mut tracker = PipelineTelemetryTracker::new(16);
        tracker.state = AdmissionState::InterleaveActive;
        tracker.baseline_stats.p95_us = 200.0;

        let bad_stats = WindowStats {
            p95_us: 280.0, // > 1.15 * 200 = 230
            fallback_rate: 0.02,
            ..WindowStats::default()
        };

        let next = tracker.evaluate_admission(&bad_stats);
        assert_eq!(next, AdmissionState::InterleaveCoolingDown);
    }

    // ── Probe function ───────────────────────────────────────────

    #[test]
    fn probe_64bit_rejects_old_gpu() {
        assert!(!probe_64bit_atomic_execution(0, 0));
        assert!(!probe_64bit_atomic_execution(6, 0));
    }

    #[test]
    fn probe_64bit_accepts_apple_silicon() {
        assert!(probe_64bit_atomic_execution(7, 1));
        assert!(probe_64bit_atomic_execution(8, 1));
    }

    // ── Interleave window ────────────────────────────────────────

    #[test]
    fn interleave_window_bounded_capacity() {
        let mut w = InterleaveWindow::new(4);
        for i in 0..10u64 {
            w.push(TokenMetric {
                token_index: i,
                request_submission_us: 0,
                prefetch_completion_us: 100,
                decode_begin_us: 100,
                decode_end_us: 200,
                was_cache_hit: true,
                hit_layer_count: 1,
                fallback_count: 0,
            });
        }
        assert_eq!(w.len(), 4);
        // The oldest entry should have token_index 6 (indices 6,7,8,9 remain).
        assert_eq!(w.metrics.front().unwrap().token_index, 6);
        assert_eq!(w.metrics.back().unwrap().token_index, 9);
    }
}
