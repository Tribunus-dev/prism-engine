
//! Dispatch orchestration for the persistent GPU megakernel (Metal 4 + MPP TensorOps).
//!
//! The [`Megakernel`] struct owns the compiled Metal compute pipeline state
//! and provides methods to allocate KV cache buffers, submit decode work
//! via an atomic ring buffer, poll for completion, and read back
//! logits and entropy data.
//!
//! Compiled with `-std=metal4`.  The GEMV tile decompression uses threadgroup
//! scratch memory at index 0 (1280 bytes = 640 halves) to decompress ternary
//! weights to FP16 before issuing `mpp::tensor_ops` operations.  Backward
//! compatible with M1–M4 (TensorOps auto-fall back to ALU on pre-M5 hardware).
//! All non-GEMV shader paths (RMSNorm, RoPE, SwiGLU, GQA attention, centroid
//! scout, MTP) are unchanged from the M3 baseline.

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

    // Per-slot KV cache cost in bytes (ternary32 K+V + outlier bypass + FP16 scratch + logits)
    let blocks_per_head   = ((GLOBAL_HEAD_DIM + 31) / 32) as u64;  // 16 for 512-dim
    let blocks_per_slot   = (LAYERS as u64) * (MAX_CONTEXT as u64) * (NUM_KV_HEADS as u64) * blocks_per_head;
    let ternary_kv_per_slot = blocks_per_slot * 9 * 2;             // K+V TernaryBlock32: 7 packed + 2 scale = 9
    let outlier_per_slot    = blocks_per_slot * 2 * 2;             // K+V FP16 outlier bypass: 2 bytes/block
    let scratch_per_slot    = (MAX_CONTEXT as u64) * (NUM_KV_HEADS as u64) * (GLOBAL_HEAD_DIM as u64) * 2 * 2; // K+V FP16
    let logits_per_slot     = LOGITS_PER_SLOT;
    let per_slot_total      = ternary_kv_per_slot + outlier_per_slot + scratch_per_slot + logits_per_slot;


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
    pub int4_mode: bool,
    pub num_slots: u64,
    ring_head: AtomicU32,
    last_completed: AtomicU32,
}

impl Megakernel {
    pub fn new(
        device: &Device,
        queue: &CommandQueue,
        deployment: &CimageDeployment,
        int4_mode: bool,
    ) -> Result<Self, String> {
        let num_slots = compute_num_slots(device);
        let pso = if let Some(metallib_buf) = &deployment.metallib_buffer {
            let ptr = metallib_buf.contents() as *const u8;
            let len = metallib_buf.length() as usize;
            let data = unsafe { std::slice::from_raw_parts(ptr, len) };
            if int4_mode { compile_kernel_from_metallib_int4(device, data)? }
            else { compile_kernel_from_metallib(device, data)? }
        } else {
            super::kernels::compile_kernel(device, int4_mode)?
        };
        Ok(Self {
            pso,
            queue: queue.clone(),
            device: device.clone(),
            int4_mode,
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


        // ── KV cache buffers (per slot) ─────────────────────────────
        // Ternary: TernaryBlock32 = 7 packed trits + 2 FP16 scale = 9 bytes/block
        // + outlier bypass: 1 FP16 per block = 2 bytes/block
        // Non-INT4 path: 256-elem blocks = 54 bytes/block + separate FP16 scales
        let (kv_k_nibbles, kv_v_nibbles, kv_k_scales, kv_v_scales, kv_k_outliers, kv_v_outliers) = if self.int4_mode {
            // Ternary 5-per-byte blocks with outlier isolation (TernaryBlock32)
            let int4_blocks_per_head = (GLOBAL_HEAD_DIM + 31) / 32;  // 16 for 512-dim
            let total_blocks_per_slot =
                (LAYERS * MAX_CONTEXT * NUM_KV_HEADS * int4_blocks_per_head) as u64;
            // TernaryBlock32: 7 bytes packed trits + 2 bytes FP16 scale = 9 bytes/block
            let ternary_bytes = total_blocks_per_slot * 9;
            let ternary_total = ternary_bytes * num_slots;
            // Outlier bypass: 1 FP16 value per block worst case = 2 bytes/block
            let outlier_bytes = total_blocks_per_slot * 2;
            let outlier_total = outlier_bytes * num_slots;

            let k_ternary = self
                .device
                .new_buffer(ternary_total, MTLResourceOptions::StorageModeShared);
            let v_ternary = self
                .device
                .new_buffer(ternary_total, MTLResourceOptions::StorageModeShared);
            let k_outliers = self
                .device
                .new_buffer(outlier_total, MTLResourceOptions::StorageModeShared);
            let v_outliers = self
                .device
                .new_buffer(outlier_total, MTLResourceOptions::StorageModeShared);
            unsafe {
                std::ptr::write_bytes(k_ternary.contents(), 0, ternary_total as usize);
                std::ptr::write_bytes(v_ternary.contents(), 0, ternary_total as usize);
                std::ptr::write_bytes(k_outliers.contents(), 0, outlier_total as usize);
                std::ptr::write_bytes(v_outliers.contents(), 0, outlier_total as usize);
            }
            (k_ternary, v_ternary, None, None, Some(k_outliers), Some(v_outliers))
        } else {
            // Ternary: 256 values per block, 54 bytes/block + separate FP16 scales
            let total_blocks_per_slot =
                (LAYERS * MAX_CONTEXT * NUM_KV_HEADS * (GLOBAL_HEAD_DIM + 255) / 256) as u64;
            let ternary_kv_bytes_per_slot = total_blocks_per_slot * KV_BLOCK_BYTES;
            let ternary_kv_total = ternary_kv_bytes_per_slot * num_slots;

            let k_buf = self
                .device
                .new_buffer(ternary_kv_total, MTLResourceOptions::StorageModeShared);
            let v_buf = self
                .device
                .new_buffer(ternary_kv_total, MTLResourceOptions::StorageModeShared);

            let total_blocks = total_blocks_per_slot * num_slots;
            let scales_bytes = total_blocks * 2;
            let k_scales = self
                .device
                .new_buffer(scales_bytes, MTLResourceOptions::StorageModeShared);
            let v_scales = self
                .device
                .new_buffer(scales_bytes, MTLResourceOptions::StorageModeShared);

            unsafe {
                std::ptr::write_bytes(k_buf.contents(), 0, ternary_kv_total as usize);
                std::ptr::write_bytes(v_buf.contents(), 0, ternary_kv_total as usize);
                std::ptr::write_bytes(k_scales.contents(), 0, scales_bytes as usize);
                std::ptr::write_bytes(v_scales.contents(), 0, scales_bytes as usize);
            }
            (k_buf, v_buf, Some(k_scales), Some(v_scales), None, None)
        };

        // ── Outlier bitmask LUT (static, shared across slots) ────────
        // One u32 per 32-element block, zero-initialized (all inlier by default).
        let blocks_per_head = (GLOBAL_HEAD_DIM + 31) / 32;  // 16 for 512-dim
        let total_masks = (LAYERS * NUM_KV_HEADS * blocks_per_head) as u64;
        let mask_bytes = total_masks * 4;
        let outlier_masks = self
            .device
            .new_buffer(mask_bytes, MTLResourceOptions::StorageModeShared);
        unsafe {
            std::ptr::write_bytes(outlier_masks.contents(), 0, mask_bytes as usize);
        }

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
            std::ptr::write_bytes(kv_scratch_k.contents(), 0, scratch_total as usize);
            std::ptr::write_bytes(kv_scratch_v.contents(), 0, scratch_total as usize);
        }

        // ── Atomic ring buffer for work submission ───────────────────
        // ring_entries: RING_SIZE entries × 5 u32s each (state|kind, token_id/chunk_pos, seq_pos/num_prior, kv_slot_id, reserved)
        // CPUCacheModeWriteCombined: CPU writes bypass SLC entirely, go directly to DRAM.
        // This prevents evicting ANE's hot weights from the 8 MB SLC.
        let ring_entries = self
            .device
            .new_buffer(
                RING_SIZE as u64 * 5 * 4,
                MTLResourceOptions::CPUCacheModeWriteCombined | MTLResourceOptions::StorageModeShared,
            );
        unsafe {
            std::ptr::write_bytes(ring_entries.contents(), 0, RING_SIZE * 5 * 4);
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
        // CPUCacheModeWriteCombined CPU only reads, but ensures no SLC pollution.
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

        // ── Active token mask (continuous compaction) ──
        // One u32 per MAX_CONTEXT position per slot. 1 = active, 0 = evicted.
        // CPU updates after each decode step based on running entropy scores.
        let active_mask_bytes = (MAX_CONTEXT as u64) * 4 * num_slots;
        let active_mask = self
            .device
            .new_buffer(active_mask_bytes, MTLResourceOptions::StorageModeShared);
        // Initialize all to 1 (all positions active by default)
        unsafe {
            let ptr = active_mask.contents() as *mut u32;
            for i in 0..(MAX_CONTEXT as usize * num_slots as usize) {
                *ptr.add(i) = 1;
            }
        }

        // ── Draft model output buffer ──────────────────────────────
        // Per slot: [u32 count] + [MAX_DRAFT_CANDIDATES × u32 token_ids] +
        // [MAX_DRAFT_CANDIDATES × f32 logprobs]
        let draft_output_bytes = (num_slots as u64) * (1 + MAX_DRAFT_CANDIDATES as u64 * 2) * 4;
        let draft_output = self
            .device
            .new_buffer(draft_output_bytes, MTLResourceOptions::StorageModeShared);
        unsafe {
            std::ptr::write_bytes(draft_output.contents(), 0, draft_output_bytes as usize);
        }

        // One-shot dispatch of persistent kernel (runs forever)
        let cmd_buf = self.queue.new_command_buffer();
        let enc = cmd_buf.new_compute_command_encoder();

        enc.set_compute_pipeline_state(&self.pso);
        if self.int4_mode {
            if let Some(fused) = &deployment.fused_int4_buffer {
                enc.set_buffer(0, Some(&**fused), 0);
            } else {
                enc.set_buffer(0, deployment.weights_int4_buffer.as_ref().map(|b| &**b), 0);
            }
        } else {
            enc.set_buffer(0, Some(&deployment.weights_buffer), 0);
        }
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
        if self.int4_mode {
            // Ternary+outlier: buffer(8)=kv_k_outliers, buffer(9)=kv_v_outliers, buffer(10)=outlier_masks
            enc.set_buffer(8, kv_k_outliers.as_deref(), 0);
            enc.set_buffer(9, kv_v_outliers.as_deref(), 0);
            enc.set_buffer(10, Some(&*outlier_masks), 0);
        } else {
            enc.set_buffer(8, kv_k_scales.as_deref(), 0);
            enc.set_buffer(9, kv_v_scales.as_deref(), 0);
        }
        enc.set_buffer(14, deployment.embed_scales_buffer.as_ref().map(|b| &**b), 0);
        enc.set_buffer(15, Some(&*centroid_scales), 0);
        enc.set_buffer(16, Some(&*centroid_scratch), 0);
        enc.set_buffer(17, Some(&*decompress_progress), 0);
        // slot 18: removed (old work_queue)
        enc.set_buffer(19, Some(&*kv_scratch_k), 0);
        enc.set_buffer(20, Some(&*kv_scratch_v), 0);
        enc.set_buffer(21, Some(&*entropy_map), 0);
        enc.set_buffer(11, Some(&*active_mask), 0);
        enc.set_buffer(22, Some(&*ring_entries), 0);
        enc.set_buffer(23, Some(&*ring_tail), 0);
        enc.set_buffer(24, Some(&*slot_logits), 0);
        enc.set_buffer(25, Some(&*completion_counter), 0);
        enc.set_buffer(28, Some(&*draft_output), 0);

        // Threadgroup scratch for ternary->FP16 decompress in tile-GEMV:
        // 640 halves = 1280 bytes at index 0 (consumed by the `tile_scratch`
        // threadgroup buffer in the Metal 4 shader).
        enc.set_threadgroup_memory_length(0, (TILE as u64) * 2);

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
        // Do NOT wait -- the persistent kernel runs forever

        Ok(KernelBuffers {
            kv_k_nibbles,
            kv_v_nibbles,
            kv_k_scales,
            kv_v_scales,
            kv_k_outliers,
            kv_v_outliers,
            outlier_masks,
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
            active_mask,
            draft_output,
        })
    }

    /// Prefill a slot with a batch of tokens using the GPU work queue.
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
            // We skip reading logits the KV cache is the only output we need.
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
            let entry = entries.add(idx * 5);
            *entry.add(0) = 1; // SUBMITTED | (kind << 2) — kind=0 for decode
            entry.add(1).write(token_id);
            entry.add(2).write(seq_pos);
            entry.add(3).write(kv_slot_id);
            entry.add(4).write(0); // reserved
            std::sync::atomic::fence(Ordering::SeqCst); // ensure store visibility
        }
    }

    pub fn submit_prefill_work(&self, buffers: &KernelBuffers, slot_id: u32, chunk_pos: u32, num_prior: u32) {
        unsafe {
        let head = self.ring_head.fetch_add(1, Ordering::Release);
        let idx = head as usize % RING_SIZE;
        let entries = buffers.ring_entries.contents() as *mut u32;
        let entry = entries.add(idx * 5);
        *entry.add(0) = 1 | (1 << 2); // SUBMITTED | (PREFILL << 2)
        entry.add(1).write(chunk_pos);
        entry.add(2).write(num_prior);
        entry.add(3).write(slot_id);
        entry.add(4).write(0);
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
    pub fn reset_work_slot(&self, _buffers: &KernelBuffers, _slot_id: u32) {
        // no-op: ring entries are naturally consumed by the GPU
    }

    /// Submit a draft model decode request (kind=3).
    ///
    /// The GPU runs the fast draft model forward pass at `seq_pos` for slot 0,
    /// writing up to `num_candidates` candidate token IDs + log-probs into
    /// the `draft_output` buffer.
    pub fn submit_draft(
        &self,
        buffers: &KernelBuffers,
        _token_id: u32,
        seq_pos: u32,
        num_candidates: u32,
    ) {
        unsafe {
            let head = self.ring_head.fetch_add(1, Ordering::Release);
            let idx = head as usize % RING_SIZE;
            let entries = buffers.ring_entries.contents() as *mut u32;
            let entry = entries.add(idx * 5);
            *entry.add(0) = 1 | (3 << 2);  // SUBMITTED | (DRAFT << 2)
            entry.add(1).write(num_candidates);  // number of tokens to draft
            entry.add(2).write(seq_pos);
            entry.add(3).write(0);  // slot 0
            entry.add(4).write(0);
            std::sync::atomic::fence(Ordering::SeqCst);
        }
    }

    /// Read draft model output for slot 0.
    ///
    /// Returns a vector of `(token_id, logprob)` pairs for each candidate
    /// token the draft model produced, in order.  The logprob is the draft
    /// model's log-probability for that token (already converted to f32).
    pub fn read_draft_output(&self, buffers: &KernelBuffers) -> Vec<(u32, f32)> {
        let slot = 0usize;
        let slot_offset = slot * (1 + MAX_DRAFT_CANDIDATES as usize * 2);
        let ptr = buffers.draft_output.contents() as *const u32;
        let count = unsafe { *ptr.add(slot_offset) }.min(MAX_DRAFT_CANDIDATES);
        unsafe {
            let token_ptr = ptr.add(slot_offset + 1) as *const u32;
            let prob_ptr = ptr.add(slot_offset + 1 + MAX_DRAFT_CANDIDATES as usize) as *const f32;
            (0..count as usize)
                .map(|i| (*token_ptr.add(i), *prob_ptr.add(i)))
                .collect()
        }
    }
}

/// Per-decode buffers returned by [`Megakernel::launch`].
pub struct KernelBuffers {
    pub kv_k_nibbles: metal::Buffer,
    pub kv_v_nibbles: metal::Buffer,
    pub kv_k_scales: Option<metal::Buffer>,
    pub kv_v_scales: Option<metal::Buffer>,
    pub kv_k_outliers: Option<metal::Buffer>,  // FP16 outlier bypass (INT4 path)
    pub kv_v_outliers: Option<metal::Buffer>,
    pub outlier_masks: metal::Buffer,          // static u32 bitmask LUT per block
    pub kv_scratch_k: metal::Buffer,
    pub kv_scratch_v: metal::Buffer,
    pub ring_entries: metal::Buffer,        // RING_SIZE * 5 * 4 bytes (WorkEntry[512])
    pub ring_tail: metal::Buffer,           // 4 bytes (atomic u32, GPU-produced)
    pub slot_logits: metal::Buffer,         // NUM_SLOTS * VOCAB_SIZE * 2 bytes (half-float)
    pub completion_counter: metal::Buffer,  // 4 bytes (atomic u32, GPU-incremented)
    pub centroid_scratch: metal::Buffer,
    pub centroid_scales: metal::Buffer,
    pub decompress_progress: metal::Buffer,
    pub entropy_map: metal::Buffer,
    pub active_mask: metal::Buffer,
    /// Draft model output buffer: per-slot, first u32 = candidate count, then
    /// MAX_DRAFT_CANDIDATES × u32 token IDs, then MAX_DRAFT_CANDIDATES × f32 log-probs.
    pub draft_output: metal::Buffer,
}
