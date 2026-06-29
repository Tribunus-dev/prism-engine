//! Dispatch orchestration for the persistent GPU megakernel.
//!
//! The [`Megakernel`] struct owns the compiled Metal compute pipeline state
//! and provides methods to allocate KV cache buffers, submit decode work
//! via an atomic ring buffer, poll for completion, and read back
//! logits and entropy data.

use super::*;
use crate::compute_image::cimage_loader::CimageDeployment;
use metal::*;
/// Logits per slot: 1 main head + N MTP heads, each VOCAB_SIZE half values.
pub const LOGITS_PER_SLOT: u64 = (1 + NUM_MTP_HEADS as u64) * VOCAB_SIZE as u64 * 2;
use std::sync::atomic::{AtomicU32, Ordering};


/// Size of the submission ring buffer (must match shader constant).
pub const RING_SIZE: usize = 512;


/// Compute the maximum number of decode slots the device can support
/// based on available memory and per-slot KV cache cost.
fn compute_num_slots(device: &Device) -> u64 {
    let working_set = device.recommended_max_working_set_size();

    // Per-slot KV cache cost in bytes (ternary K+V + FP16 scratch + scales + logits)
    let blocks_per_slot = (LAYERS * MAX_CONTEXT * NUM_KV_HEADS * (GLOBAL_HEAD_DIM + 255) / 256) as u64;
    let ternary_kv_per_slot = blocks_per_slot * KV_BLOCK_BYTES * 2; // K+V nibbles
    let scales_per_slot   = blocks_per_slot * 2 * 2;                // K+V half scales
    let scratch_per_slot  = (MAX_CONTEXT * NUM_KV_HEADS * GLOBAL_HEAD_DIM) as u64 * 2 * 2; // K+V FP16
    let logits_per_slot   = LOGITS_PER_SLOT;
    let per_slot_total    = ternary_kv_per_slot + scales_per_slot + scratch_per_slot + logits_per_slot;

    // Reserve ~1.5 GB for model weights, scales, embed table, centroids, norms
    let kv_budget = working_set.saturating_sub(1_500_000_000);
    let max_slots = if per_slot_total > 0 { kv_budget / per_slot_total } else { 0 };

    // Cap at compile-time ceiling
    (max_slots as u64).clamp(1, NUM_SLOTS as u64)
}

pub struct Megakernel {
    pso: ComputePipelineState,
    queue: CommandQueue,
    device: Device,
    pub num_slots: u64,
    ring_head: AtomicU32,
    last_completed: AtomicU32,
}

impl Megakernel {
    pub fn new(
        device: &Device,
        queue: &CommandQueue,
        deployment: &CimageDeployment,
    ) -> Result<Self, String> {
        let num_slots = compute_num_slots(device);
        let pso = if let Some(metallib_buf) = &deployment.metallib_buffer {
            let ptr = metallib_buf.contents() as *const u8;
            let len = metallib_buf.length() as usize;
            let data = unsafe { std::slice::from_raw_parts(ptr, len) };
            compile_kernel_from_metallib(device, data)?
        } else {
            super::kernels::compile_kernel(device)?
        };
        Ok(Self {
            pso,
            queue: queue.clone(),
            device: device.clone(),
            num_slots,
            ring_head: AtomicU32::new(0),
            last_completed: AtomicU32::new(0),
        })
    }

    /// Launch with KV cache allocation.  Returns buffers needed by the
    /// orchestrator for decode steps.
    pub fn launch(
        &self,
        deployment: &CimageDeployment,
        _batch_size: u32,
    ) -> Result<KernelBuffers, String> {
        let num_slots = compute_num_slots(&self.device);

        // ── Ternary KV cache buffers (per slot) ──────────────────────
        // Ternary packed: 256 values → 13 u32 nibbles + 1 half scale = 54 bytes/block
        // h_dim = 256 → 1 block, h_dim = 512 → 2 blocks
        // Total blocks per slot = LAYERS × MAX_CONTEXT × NUM_KV_HEADS × ceil(GLOBAL_HEAD_DIM / 256)
        let total_blocks_per_slot =
            (LAYERS * MAX_CONTEXT * NUM_KV_HEADS * (GLOBAL_HEAD_DIM + 255) / 256) as u64;
        let ternary_kv_bytes_per_slot = total_blocks_per_slot * KV_BLOCK_BYTES;
        let ternary_kv_total = ternary_kv_bytes_per_slot * num_slots;

        let kv_k_nibbles = self
            .device
            .new_buffer(ternary_kv_total, MTLResourceOptions::StorageModeShared);
        let kv_v_nibbles = self
            .device
            .new_buffer(ternary_kv_total, MTLResourceOptions::StorageModeShared);

        // Block scales: one half per block
        let total_blocks = total_blocks_per_slot * num_slots;
        let scales_bytes = total_blocks * 2; // one half (2 bytes) per block
        let kv_k_scales = self
            .device
            .new_buffer(scales_bytes, MTLResourceOptions::StorageModeShared);
        let kv_v_scales = self
            .device
            .new_buffer(scales_bytes, MTLResourceOptions::StorageModeShared);

        // ── FP16 scratch buffers (1 layer per slot) ──────────────────
        // Scratch holds decompressed FP16 K/V for one layer, used during decode.
        let scratch_elems_per_slot = (MAX_CONTEXT * NUM_KV_HEADS * GLOBAL_HEAD_DIM) as u64;
        let scratch_bytes_per_slot = scratch_elems_per_slot * 2; // half = 2 bytes
        let scratch_total = scratch_bytes_per_slot * num_slots;

        let kv_scratch_k = self
            .device
            .new_buffer(scratch_total, MTLResourceOptions::StorageModeShared);
        let kv_scratch_v = self
            .device
            .new_buffer(scratch_total, MTLResourceOptions::StorageModeShared);

        // Zero-initialize all buffers
        unsafe {
            std::ptr::write_bytes(kv_k_nibbles.contents(), 0, ternary_kv_total as usize);
            std::ptr::write_bytes(kv_v_nibbles.contents(), 0, ternary_kv_total as usize);
            std::ptr::write_bytes(kv_k_scales.contents(), 0, scales_bytes as usize);
            std::ptr::write_bytes(kv_v_scales.contents(), 0, scales_bytes as usize);
            std::ptr::write_bytes(kv_scratch_k.contents(), 0, scratch_total as usize);
            std::ptr::write_bytes(kv_scratch_v.contents(), 0, scratch_total as usize);
        }

        // ── Atomic ring buffer for work submission ───────────────────
        // ring_entries: RING_SIZE entries × 4 u32s each (state, token_id, seq_pos, kv_slot_id)
        // CPUCacheModeWriteCombined: CPU writes bypass SLC entirely, go directly to DRAM.
        // This prevents evicting ANE's hot weights from the 8 MB SLC.
        let ring_entries = self
            .device
            .new_buffer(
                RING_SIZE as u64 * 4 * 4,
                MTLResourceOptions::CPUCacheModeWriteCombined | MTLResourceOptions::StorageModeShared,
            );
        unsafe {
            std::ptr::write_bytes(ring_entries.contents(), 0, RING_SIZE * 4 * 4);
        }

        // ring_tail: atomic u32 (GPU produces)
        let ring_tail = self
            .device
            .new_buffer(4, MTLResourceOptions::StorageModeShared);
        unsafe {
            *(ring_tail.contents() as *mut u32) = 0;
        }

        let logits_total = num_slots * LOGITS_PER_SLOT;
        let slot_logits = self
            .device
            .new_buffer(logits_total, MTLResourceOptions::StorageModeShared);

        // completion_counter: atomic u32 (GPU increments after each work item)
        // CPUCacheModeWriteCombined — CPU only reads, but ensures no SLC pollution.
        let completion_counter = self
            .device
            .new_buffer(
                4,
                MTLResourceOptions::CPUCacheModeWriteCombined | MTLResourceOptions::StorageModeShared,
            );
        unsafe {
            *(completion_counter.contents() as *mut u32) = 0;
        }

        // ── Ternary-centroid decompress scratch ────────────────────
        let cent_tiles = ((HIDDEN_DIM + 255) / 256) as u64;
        let centroid_scales = deployment
            .centroid_scales_buffer
            .as_ref()
            .map(|b| b.clone())
            .unwrap_or_else(|| {
                self.device.new_buffer(
                    NUM_CENTROIDS as u64 * cent_tiles * 2,
                    MTLResourceOptions::StorageModeShared,
                )
            });

        let centroid_scratch = self.device.new_buffer(
            NUM_CENTROIDS as u64 * HIDDEN_DIM as u64 * 2,
            MTLResourceOptions::StorageModeShared,
        );

        let decompress_progress = self
            .device
            .new_buffer(4, MTLResourceOptions::StorageModeShared);
        unsafe {
            *(decompress_progress.contents() as *mut u32) = 0;
        }

        // ── Entropy map buffer (half per position per slot) ──────────
        let entropy_size = MAX_CONTEXT as u64 * 2 * num_slots;
        let entropy_map = self
            .device
            .new_buffer(entropy_size, MTLResourceOptions::StorageModeShared);

        // One-shot dispatch of persistent kernel (runs forever)
        let cmd_buf = self.queue.new_command_buffer();
        let enc = cmd_buf.new_compute_command_encoder();

        enc.set_compute_pipeline_state(&self.pso);
        enc.set_buffer(0, Some(&deployment.weights_buffer), 0);
        enc.set_buffer(1, Some(&deployment.scales_buffer), 0);
        if let Some(b) = &deployment.norms_buffer {
            enc.set_buffer(2, Some(b), 0);
        }
        if let Some(b) = &deployment.embed_buffer {
            enc.set_buffer(3, Some(b), 0);
        }
        if let Some(b) = &deployment.centroid_buffer {
            enc.set_buffer(4, Some(b), 0);
        }
        if let Some(b) = &deployment.cluster_map_buffer {
            enc.set_buffer(5, Some(b), 0);
        }
        enc.set_buffer(6, Some(&*kv_k_nibbles), 0);
        enc.set_buffer(7, Some(&*kv_v_nibbles), 0);
        enc.set_buffer(8, Some(&*kv_k_scales), 0);
        enc.set_buffer(9, Some(&*kv_v_scales), 0);
        enc.set_buffer(14, deployment.embed_scales_buffer.as_ref().map(|b| &**b), 0);
        enc.set_buffer(15, Some(&*centroid_scales), 0);
        enc.set_buffer(16, Some(&*centroid_scratch), 0);
        enc.set_buffer(17, Some(&*decompress_progress), 0);
        // slot 18: removed (old work_queue)
        enc.set_buffer(19, Some(&*kv_scratch_k), 0);
        enc.set_buffer(20, Some(&*kv_scratch_v), 0);
        enc.set_buffer(21, Some(&*entropy_map), 0);
        enc.set_buffer(22, Some(&*ring_entries), 0);
        enc.set_buffer(23, Some(&*ring_tail), 0);
        enc.set_buffer(24, Some(&*slot_logits), 0);
        enc.set_buffer(25, Some(&*completion_counter), 0);

        enc.dispatch_thread_groups(
            MTLSize {
                width: num_slots,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );

        enc.end_encoding();
        cmd_buf.commit();
        // Do NOT wait — the persistent kernel runs forever

        Ok(KernelBuffers {
            kv_k_nibbles,
            kv_v_nibbles,
            kv_k_scales,
            kv_v_scales,
            kv_scratch_k,
            kv_scratch_v,
            ring_entries,
            ring_tail,
            slot_logits,
            completion_counter,
            centroid_scratch,
            centroid_scales,
            decompress_progress,
            entropy_map,
        })
    }

    /// Prefill a slot with a batch of tokens using the GPU work queue.
    ///
    /// # Design decision: host-side sequential submission
    ///
    /// The persistent GPU kernel processes one token per work queue submission.
    /// Each submission runs embedding lookup, 48 transformer layers, writes K/V
    /// to the ternary-packed KV cache at `seq_pos`, then signals completion.
    ///
    /// For batched prefill, we simply loop on the host: submit token N, spin-wait
    /// for GPU completion (which populates KV cache at position start_pos+N),
    /// then submit token N+1.  This is correct because position `p+1` attends to
    /// positions 0..p (causal attention), so the KV entry for position `p` must
    /// exist before we compute position `p+1`.
    ///
    /// ## Why not modify the kernel?
    ///
    /// - **True batched GEMV** (sharing weights across N tokens) would require
    ///   N × HIDDEN_DIM × 2 bytes of SRAM — 1.9 MB for N=256, HIDDEN_DIM=3840.
    ///   The threadgroup SRAM limit is 32 KB.  Micro-batching fits only 1–2
    ///   additional hidden states with the current ~19 KB SRAM budget.
    /// - **Sequential per-token loop inside the kernel** would avoid the host
    ///   round-trip but would re-read all weights from DRAM for every token.
    ///   The per-token throughput would match single-token decode (~33 t/s),
    ///   which is memory-bandwidth bound, not compute bound.
    ///
    /// ## Prefill throughput
    ///
    /// With 32 slots processing prompts concurrently, each at ~33 t/s:
    ///   32 × 33 = ~1,056 t/s aggregate prefill throughput.
    /// A 1,000-token prompt takes ~30 s per slot.
    ///
    /// ## Cross-architecture note
    ///
    /// The ANE path (via [`Orchestrator::prefill_slot`]) provides faster prefill
    /// by processing all tokens in one MIL program invocation.  This GPU-based
    /// prefill is intended as a fallback or for systems without ANE support.
    pub fn prefill_slot_batched(
        &self,
        buffers: &KernelBuffers,
        slot_id: u32,
        tokens: &[u32],
        start_pos: u32,
        kv_slot_id: u32,
    ) {
        for (i, &token) in tokens.iter().enumerate() {
            let pos = start_pos + i as u32;

            self.submit_work(buffers, slot_id, token, pos, kv_slot_id);

            // Spin-wait for GPU completion
            while !self.poll_work(buffers, slot_id) {
                std::hint::spin_loop();
            }

            // Reset slot for the next token in the batch.
            // We skip reading logits — the KV cache is the only output we need.
            self.reset_work_slot(buffers, slot_id);
        }
    }

    /// One-shot decode: dispatch the full transformer kernel for one token.
    /// Submit a decode request to the ring buffer.
    /// Returns immediately after writing the ring entry.
    /// The GPU will dequeue it asynchronously.
    pub fn submit_work(
        &self,
        buffers: &KernelBuffers,
        _slot_id: u32,
        token_id: u32,
        seq_pos: u32,
        kv_slot_id: u32,
    ) {
        unsafe {
            let head = self.ring_head.fetch_add(1, Ordering::Release);
            let idx = head as usize % RING_SIZE;
            let entries = buffers.ring_entries.contents() as *mut u32;
            let entry = entries.add(idx * 4);
            *entry.add(0) = 1; // SUBMITTED
            entry.add(1).write(token_id);
            entry.add(2).write(seq_pos);
            entry.add(3).write(kv_slot_id);
            std::sync::atomic::fence(Ordering::SeqCst); // ensure store visibility
        }
    }

    /// Poll whether the latest submitted work has completed.
    /// Returns true if the GPU has incremented completion_counter
    /// past the last_known value.
    pub fn poll_work(&self, buffers: &KernelBuffers, _slot_id: u32) -> bool {
        let completed = unsafe { *(buffers.completion_counter.contents() as *const u32) };
        let known = self.last_completed.load(Ordering::Acquire);
        if completed > known {
            self.last_completed.store(completed, Ordering::Release);
            true
        } else {
            false
        }
    }

    /// Read logits from a completed slot.
    pub fn read_slot_logits(&self, buffers: &KernelBuffers, slot_id: u32, head: u32) -> Vec<u16> {
        let offset = (slot_id as u64) * LOGITS_PER_SLOT + (head as u64) * (VOCAB_SIZE as u64) * 2;
        let ptr = unsafe { buffers.slot_logits.contents().add(offset as usize) as *const u16 };
        let n = VOCAB_SIZE as usize;
        unsafe { std::slice::from_raw_parts(ptr, n).to_vec() }
    }

    /// Read entropy map for a completed slot.
    /// Returns per-position entropy values (half-float) as u16 slice.
    /// Only the first `num_cached` positions contain valid data.
    pub fn read_entropy_map(&self, buffers: &KernelBuffers, slot_id: u32) -> Vec<u16> {
        let slot_offset = (slot_id as u64) * (MAX_CONTEXT as u64) * 2;
        let ptr = unsafe { buffers.entropy_map.contents().add(slot_offset as usize) as *const u16 };
        let n = MAX_CONTEXT as usize;
        unsafe { std::slice::from_raw_parts(ptr, n).to_vec() }
    }

    /// Reset slot state after reading results.
    /// With the ring-buffer design each work item is a unique ring entry,
    /// so there is no per-slot state to clear.  Kept for API compatibility.
    pub fn reset_work_slot(&self, _buffers: &KernelBuffers, _slot_id: u32) {
        // no-op: ring entries are naturally consumed by the GPU
    }
}

/// Per-decode buffers returned by [`Megakernel::launch`].
pub struct KernelBuffers {
    pub kv_k_nibbles: metal::Buffer,
    pub kv_v_nibbles: metal::Buffer,
    pub kv_k_scales: metal::Buffer,
    pub kv_v_scales: metal::Buffer,
    pub kv_scratch_k: metal::Buffer,
    pub kv_scratch_v: metal::Buffer,
    pub ring_entries: metal::Buffer,        // RING_SIZE * 4 * 4 bytes (WorkEntry[512])
    pub ring_tail: metal::Buffer,           // 4 bytes (atomic u32, GPU-produced)
    pub slot_logits: metal::Buffer,         // NUM_SLOTS * VOCAB_SIZE * 2 bytes (half-float)
    pub completion_counter: metal::Buffer,  // 4 bytes (atomic u32, GPU-incremented)
    pub centroid_scratch: metal::Buffer,
    pub centroid_scales: metal::Buffer,
    pub decompress_progress: metal::Buffer,
    pub entropy_map: metal::Buffer,
}
