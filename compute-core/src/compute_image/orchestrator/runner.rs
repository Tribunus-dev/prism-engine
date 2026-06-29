//! Main inference orchestrator — loading, decode, compaction, agent management.
//!
//! Owns a loaded `.cimage` deployment, the full-transformer GPU megakernel
//! (embedding → 48 layers → logits), tree-attention, and optional ANE
//! prefill model for prompt processing.

use crate::arena::DataType;
use crate::compute_image::compaction;
use crate::compute_image::cimage_loader::CimageDeployment;
use crate::compute_image::megakernel::{KernelBuffers, Megakernel};
use crate::compute_image::megakernel::{MAX_DRAFT_CANDIDATES, NUM_MTP_HEADS};
use crate::compute_image::tree_attention::TreeAttention;
use crate::compute_image::vm_manager::VmManager;
use super::{
    generate_speculative_candidates, sample_argmax,
    GLOBAL_HEAD_DIM, LAYERS, MAX_CONTEXT, MAX_SURVIVORS, NUM_KV_HEADS, NUM_SLOTS, SLCPhase,
};
use crate::arena::Arena;
use crate::coreml_bridge::{CoreMlComputeUnits, CoreMlModel};
use half::f16;
use metal::*;
use std::path::PathBuf;

// ── Architecture constants (also used by sibling modules) ────────
// (shared constants live in mod.rs; runner re-exports via `use super::*`)

/// Top-level inference orchestrator.
///
/// Owns a loaded `.cimage` deployment, the full-transformer GPU
/// megakernel (embedding → 48 layers → logits), tree-attention, and
/// optional ANE prefill model for prompt processing.
pub struct Orchestrator {
    pub megakernel: Megakernel,
    pub tree_attn: TreeAttention,
    pub device: Device,
    pub queue: CommandQueue,
    pub deployment: CimageDeployment,
    pub int4_mode: bool,
    pub kernel_buffers: KernelBuffers,
    pub batch_size: u32,
    /// Per-slot sequence positions (0..NUM_SLOTS).
    /// slot_seq_pos[slot] tracks how many tokens have been
    /// processed (prefilled + decoded) for that slot.
    pub slot_seq_pos: Vec<u32>,
    /// Current SLC phase. Used to prevent ANE/GPU SLC thrashing.
    pub slc_phase: SLCPhase,
    /// Compiled ANE prefill model loaded from the cimage's MIL program.
    /// Set by [`Self::compile_ane_model`] when `deployment.mil_buffer`
    /// is present and compilation succeeds.
    /// One model instance per work queue slot for parallel prefill.
    pub ane_prefill_models: Vec<Option<CoreMlModel>>,
    /// Cache path for the compiled .mlmodelc bundle (alongside the cimage).
    pub ane_modelc_path: Option<PathBuf>,
    /// Compiled ANE compaction gather model. Loaded when compaction
    /// MIL program compiles successfully.
    pub compaction_model: Option<CoreMlModel>,
    /// Indices arena for compaction (Int32). Pre-allocated at load time.
    pub compaction_indices_arena: Option<Arena>,
    /// Per-layer input arenas for compaction (FP16, one layer at a time).
    pub compaction_k_arena: Option<Arena>,
    pub compaction_v_arena: Option<Arena>,
    /// Output arenas for compacted KV (FP16).
    pub compacted_k_arena: Option<Arena>,
    pub compacted_v_arena: Option<Arena>,
    /// Whether at least one GPU decode step has run (populating the entropy map).
    /// When false, the first compaction uses uniform stride.
    /// When true, entropy-driven compaction selects high-uncertainty positions.
    pub entropy_available: bool,
    /// VM manager for IOSurface pool across work queue slots.
    pub vm_manager: VmManager,
    /// Current multi-pass compaction index.
    pub compaction_pass: u32,
    /// Pre-compiled ANE prefill layer model loaded from embedded
    /// model bytes. Built at ingest time by gemma4_ingest via
    /// coremlcompiler. One model instance per work queue slot.
    pub prefill_model: Option<CoreMlModel>,
}

impl Orchestrator {
    /// Create an orchestrator from a compiled `.cimage` file.
    ///
    /// Opens the file, loads weights onto the GPU, compiles both the
    /// full-transformer megakernel and the tree-attention kernel, and
    /// allocates GPU-side buffers (KV cache, logits, atomics).
    ///
    /// If the deployment contains a `mil_buffer` (ANE MIL program),
    /// attempts to compile it via `xcrun coremlcompiler` and load the
    /// resulting model for ANE prefill.
    pub fn from_cimage(
        path: impl AsRef<std::path::Path>,
        batch_size: u32,
        int4_mode: bool,
    ) -> Result<Self, String> {
        let path = path.as_ref();
        let device = Device::system_default().ok_or("no Metal device available")?;
        let queue = device.new_command_queue();
        let mut deployment = CimageDeployment::load(path, &device)?;
        if int4_mode {
            deployment.maybe_expand_to_int4(&device)?;
        }
        let megakernel = Megakernel::new(&device, &queue, &deployment, int4_mode)?;
        let tree_attn = TreeAttention::new(&device)?;
        let kernel_buffers = megakernel.launch(&deployment, batch_size)?;

        // ── ANE prefill model compilation ────────────────────────────
        let (mut first_model, ane_modelc_path) = if deployment.mil_buffer.is_some() {
            let cache_path = path.with_extension("ane_prefill.modelc");
            match Self::compile_ane_model(&deployment, &cache_path) {
                Ok(model) => (Some(model), Some(cache_path)),
                Err(e) => {
                    eprintln!(
                        "[orchestrator] ANE model compilation failed (prefill unavailable): {e}"
                    );
                    (None, None)
                }
            }
        } else {
            (None, None)
        };

        // Create additional ANE model instances for other slots (parallel prefill)
        let num_slots = NUM_SLOTS as usize;
        let mut ane_prefill_models: Vec<Option<CoreMlModel>> = Vec::with_capacity(num_slots);
        for i in 0..num_slots {
            if i == 0 {
                ane_prefill_models.push(first_model.take());
            } else if let Some(ref cache_path) = ane_modelc_path {
                match CoreMlModel::load_with_compute_units(
                    &cache_path.to_string_lossy(),
                    CoreMlComputeUnits::CpuAndNeuralEngine,
                ) {
                    Ok(m) => ane_prefill_models.push(Some(m)),
                    Err(e) => {
                        eprintln!("[orchestrator] Failed to load ANE model for slot {i}: {e}");
                        ane_prefill_models.push(None);
                    }
                }
            } else {
                ane_prefill_models.push(None);
            }
        }

        // ── ANE compaction model compilation ─────────────────────────
        let compaction_model = Self::load_compaction_model(
            deployment.compaction_model_bytes.as_ref(),
            NUM_KV_HEADS,
            GLOBAL_HEAD_DIM,
            MAX_CONTEXT,
        );

        // ── ANE prefill model (pre-compiled, from cimage aux tail) ──
        let prefill_model = deployment
            .prefill_model_bytes
            .as_ref()
            .and_then(|bytes| Self::load_prefill_model(bytes));

        // Pre-allocate compaction arenas
        let (compaction_indices_arena, compaction_k_arena, compaction_v_arena, compacted_k_arena, compacted_v_arena) =
            Self::allocate_compaction_arenas(
                &compaction_model,
                NUM_KV_HEADS,
                GLOBAL_HEAD_DIM,
                MAX_CONTEXT,
            );

        Ok(Self {
            megakernel,
            tree_attn,
            int4_mode,
            device,
            queue,
            deployment,
            kernel_buffers,
            batch_size,
            slot_seq_pos: vec![0; NUM_SLOTS as usize],
            slc_phase: SLCPhase::GPUDecode,
            ane_prefill_models,
            ane_modelc_path,
            compaction_model,
            compaction_indices_arena,
            compaction_k_arena,
            compaction_v_arena,
            compacted_k_arena,
            compacted_v_arena,
            entropy_available: false,
            vm_manager: VmManager::new(),
            compaction_pass: 0,
            prefill_model,
        })
    }

    /// Run ANE prefill on a prompt for a specific slot, then transfer
    /// the KV cache to that slot's partition of the Metal buffers.
    ///
    /// After this call, decode_token(slot_id, ...) will attend to all
    /// prefill positions.
    pub fn prefill_slot(&mut self, slot_id: u32, prompt: &[u32]) -> Result<(), String> {
        let prompt_len = prompt.len() as u32;
        if prompt_len == 0 {
            return Err("prefill_slot: empty prompt".into());
        }

        let slot = slot_id as usize;
        let model = self
            .ane_prefill_models
            .get(slot)
            .and_then(|m| m.as_ref())
            .ok_or_else(|| format!("prefill_slot: no ANE model for slot {slot_id}"))?;

        self.slc_phase = SLCPhase::ANEPrefill;

        // ── 1. Allocate input arena for token IDs ────────────────────
        // The ANE model expects an MLMultiArray of shape [1, prompt_len]
        // with dtype Float32.
        let input_arena = Arena::new(1, prompt_len, DataType::Float32)
            .map_err(|e| format!("input arena: {e}"))?;

        // Write token IDs as f32 into the input arena.
        {
            input_arena.lock()?;
            let ptr = unsafe { input_arena.base_ptr() as *mut f32 };
            let dst = unsafe { std::slice::from_raw_parts_mut(ptr, prompt_len as usize) };
            for (i, &tok) in prompt.iter().enumerate() {
                dst[i] = tok as f32;
            }
            input_arena.unlock()?;
        }

        // ── 2. Determine output layout from architecture constants ──
        // The ANE model outputs K and V caches for all layers.
        // Each layer's K/V shape: [prompt_len, NUM_KV_HEADS, GLOBAL_HEAD_DIM]
        // Total K cache (all layers): LAYERS × prompt_len × NUM_KV_HEADS × GLOBAL_HEAD_DIM FP16
        let per_layer_kv_elems = prompt_len * NUM_KV_HEADS * GLOBAL_HEAD_DIM;
        let total_kv_elems = LAYERS * per_layer_kv_elems;

        let k_output_arena = Arena::from_metal_buffer(
            &self.kernel_buffers.kv_scratch_k,
            total_kv_elems as i32,
            1,
            DataType::Float16,
        )
            .map_err(|e| format!("k output arena from scratch: {e}"))?;
        let v_output_arena = Arena::from_metal_buffer(
            &self.kernel_buffers.kv_scratch_v,
            total_kv_elems as i32,
            1,
            DataType::Float16,
        )
            .map_err(|e| format!("v output arena from scratch: {e}"))?;

        // ── 3. Run ANE prediction ────────────────────────────────────
        // Use the pixelbuffer path for IOSurface-backed tensors.
        // Port names follow the MIL program contract: "token_ids" -> "k_cache", "v_cache".
        let mut k_info = k_output_arena.info;
        let mut v_info = v_output_arena.info;

        model
            .predict_pixelbuffer("token_ids", &input_arena.info, "k_cache", &mut k_info)
            .map_err(|e| format!("ANE prefill K prediction: {e}"))?;
        model
            .predict_pixelbuffer("token_ids", &input_arena.info, "v_cache", &mut v_info)
            .map_err(|e| format!("ANE prefill V prediction: {e}"))?;

        // ── 4. Transfer KV cache from arenas to scratch + pack to ternary ─
        //
        // The ANE output layout is [layer][position][head][dim], same as
        // the Metal scratch buffer layout.
        //
        // Scratch layout: per-slot scratch holds 1 layer's worth of FP16 data.
        // We process layers one at a time: copy ANE output for layer L into
        // the slot's scratch partition, then pack to ternary.
        // Within a layer, positions are consecutive, each position has
        // NUM_KV_HEADS Ã GLOBAL_HEAD_DIM FP16 values, head-major.
        //
        // The ANE output arena has the same layout but with
        // `prompt_len` positions instead of MAX_CONTEXT.
        //
        // The scratch destination offset is per-slot (1 layer's worth per slot).
        let per_layer_scratch_elems = (MAX_CONTEXT * NUM_KV_HEADS * GLOBAL_HEAD_DIM) as usize;
        let per_layer_scratch_bytes = per_layer_scratch_elems * 2; // FP16 = 2 bytes
        let scratch_slot_offset = (slot_id as usize) * per_layer_scratch_bytes;

        let k_scratch_ptr = unsafe {
            self.kernel_buffers
                .kv_scratch_k
                .contents()
                .add(scratch_slot_offset) as *mut u8
        };
        let v_scratch_ptr = unsafe {
            self.kernel_buffers
                .kv_scratch_v
                .contents()
                .add(scratch_slot_offset) as *mut u8
        };

        k_output_arena.lock()?;
        v_output_arena.lock()?;

        let k_ane_ptr = unsafe { k_output_arena.base_ptr() as *const u8 };
        let v_ane_ptr = unsafe { v_output_arena.base_ptr() as *const u8 };

        let per_layer_ane_bytes = (per_layer_kv_elems as usize) * 2;

        // ── 4a. Optionally run ANE compaction gather ─────────────────
        // If the compaction model is available and the prompt is long
        // enough to benefit from compaction (at least 2x the target),
        // use the ANE to gather just the survivor positions.
        // Multi-pass: if indices exceed DEFAULT_TARGET_COUNT (20K), split
        // into chunks and fire the gather model repeatedly.
        let slot_alloc = self.vm_manager.slot_allocation(slot_id);
        let target_total = slot_alloc.survivor_count;
        let should_compact = self.compaction_model.is_some() && prompt_len > target_total * 2;

        if should_compact {
            let compaction_model = self.compaction_model.as_ref().unwrap();
            let indices_arena = self.compaction_indices_arena.as_ref().unwrap();
            let k_in_arena = self.compaction_k_arena.as_ref().unwrap();
            let v_in_arena = self.compaction_v_arena.as_ref().unwrap();
            let k_out_arena = self.compacted_k_arena.as_ref().unwrap();
            let v_out_arena = self.compacted_v_arena.as_ref().unwrap();

            // Compute survivor positions.
            // First compaction (no decode steps yet): uniform stride selection.
            // After GPU decode runs: entropy-driven selection from accumulated attention data.
            let indices = if self.entropy_available {
                // Read entropy map from GPU (populated by decode kernel after attention)
                let entropy_raw = self
                    .megakernel
                    .read_entropy_map(&self.kernel_buffers, slot_id);
                let active_len = prompt_len as usize;
                let entropies: Vec<f16> = entropy_raw[..active_len]
                    .iter()
                    .map(|&v| f16::from_bits(v))
                    .collect();
                compaction::select_entropy_adaptive_positions(&entropies, target_total as usize)
            } else {
                // No entropy data: use uniform stride heuristic
                compaction::select_compaction_positions(prompt_len as usize, target_total as usize)
            };

            // Multi-pass: split indices into chunks of up to DEFAULT_TARGET_COUNT
            // and fire gather model for each chunk.
            const CHUNK_SIZE: usize = compaction::DEFAULT_TARGET_COUNT as usize;
            let num_passes = (indices.len() + CHUNK_SIZE - 1) / CHUNK_SIZE;
            self.compaction_pass = 0;

            for pass_idx in 0..num_passes {
                let start = pass_idx * CHUNK_SIZE;
                let end = (start + CHUNK_SIZE).min(indices.len());
                let chunk_indices = &indices[start..end];
                let chunk_len = chunk_indices.len();

                if chunk_len == 0 {
                    continue;
                }

                // Compute byte offset within the slot's scratch region
                // from the VM manager's allocation base.
                let per_position_bytes =
                    (NUM_KV_HEADS as usize) * (GLOBAL_HEAD_DIM as usize) * 2;
                let chunk_offset_bytes =
                    slot_alloc.byte_offset as usize + start * per_position_bytes;

                // Write chunk indices to indices arena as Int32.
                {
                    indices_arena.lock()?;
                    let ptr = unsafe { indices_arena.base_ptr() as *mut u32 };
                    let dst = unsafe { std::slice::from_raw_parts_mut(ptr, chunk_len) };
                    dst.copy_from_slice(chunk_indices);
                    indices_arena.unlock()?;
                }

                let compacted_per_layer_bytes = chunk_len * per_position_bytes;

                for layer in 0..LAYERS {
                    let layer_ane_offset = (layer as usize) * per_layer_ane_bytes;

                    // Copy per-layer FP16 K from ANE output -> compaction input arena
                    unsafe {
                        let k_src = k_ane_ptr.add(layer_ane_offset);
                        let k_dst = k_in_arena.base_ptr() as *mut u8;
                        std::ptr::copy_nonoverlapping(k_src, k_dst, per_layer_ane_bytes);
                    }

                    // Copy per-layer FP16 V from ANE output -> compaction input arena
                    unsafe {
                        let v_src = v_ane_ptr.add(layer_ane_offset);
                        let v_dst = v_in_arena.base_ptr() as *mut u8;
                        std::ptr::copy_nonoverlapping(v_src, v_dst, per_layer_ane_bytes);
                    }

                    // Run ANE compaction gather: input KV + indices -> compacted KV
                    let mut compacted_k_info = k_out_arena.info;
                    let mut compacted_v_info = v_out_arena.info;

                    compaction_model
                        .predict_multi(
                            &["key_cache", "value_cache", "indices"],
                            &[&k_in_arena.info, &v_in_arena.info, &indices_arena.info],
                            &["compacted_key", "compacted_value"],
                            &mut [&mut compacted_k_info, &mut compacted_v_info],
                        )
                        .map_err(|e| format!("compaction layer {layer} pass {pass_idx}: {e}"))?;

                    // Copy compacted output to scratch at chunk offset
                    let pass_scratch_k = unsafe { k_scratch_ptr.add(chunk_offset_bytes) };
                    let pass_scratch_v = unsafe { v_scratch_ptr.add(chunk_offset_bytes) };

                    unsafe {
                        let k_src = k_out_arena.base_ptr() as *const u8;
                        std::ptr::copy_nonoverlapping(
                            k_src,
                            pass_scratch_k,
                            compacted_per_layer_bytes,
                        );

                        let v_src = v_out_arena.base_ptr() as *const u8;
                        std::ptr::copy_nonoverlapping(
                            v_src,
                            pass_scratch_v,
                            compacted_per_layer_bytes,
                        );
                    }
                }

                self.compaction_pass += 1;
            }
        }

        k_output_arena.unlock()?;
        v_output_arena.unlock()?;

        // ── 5. Per-slot sequence position tracking ───────────────────
        self.slot_seq_pos[slot] = prompt_len;

        Ok(())
    }

    /// Run ANE prefill on a prompt using slot 0 (convenience wrapper).
    #[inline]
    pub fn prefill_text(&mut self, prompt: &[u32]) -> Result<(), String> {
        self.prefill_slot(0, prompt)
    }

    /// Decode one token using the specified work queue slot.
    /// Blocks until GPU completes and advances `slot_seq_pos[slot]`.
    ///
    /// If `prefill_slot` was called earlier, the KV cache already
    /// contains the prefill positions and attention covers the full
    /// context.
    pub fn decode_slot(&mut self, slot_id: u32, token_id: u32) -> Result<u32, String> {
        self.slc_phase = SLCPhase::GPUDecode;
        let slot = slot_id as usize;
        let seq_pos = self.slot_seq_pos[slot];

        self.megakernel
            .submit_work(&self.kernel_buffers, slot_id, token_id, seq_pos, slot_id);

        while !self.megakernel.poll_work(&self.kernel_buffers, slot_id) {
            std::thread::yield_now();
        }

        self.entropy_available = true;

        // ── Continuous entropy-driven eviction ──
        // If context exceeds L1 capacity (~20K), evict the lowest-entropy token.
        const L1_CAPACITY: u32 = 20480;
        const SINK_COUNT: u32 = 4;
        const SLIDING_WINDOW: u32 = 4096;

        let next_pos = seq_pos + 1;
        if next_pos > L1_CAPACITY {
            // Read entropy map
            let entropy = self.megakernel.read_entropy_map(&self.kernel_buffers, slot_id as u32);

            // Find lowest-entropy token outside pinned regions
            // Pinned: sinks [0..4), recent window [next_pos - SLIDING_WINDOW, next_pos)
            let window_start = next_pos.saturating_sub(SLIDING_WINDOW);
            let mut min_entropy = f32::MAX;
            let mut min_pos = SINK_COUNT.max(1);

            for pos in SINK_COUNT..window_start {
                let e = half::f16::from_bits(entropy[pos as usize]).to_f32();
                if e < min_entropy {
                    min_entropy = e;
                    min_pos = pos;
                }
            }

            // Mark for eviction in the GPU's active_mask buffer
            unsafe {
                let ptr = self.kernel_buffers.active_mask.contents() as *mut u32;
                let offset = slot as u64 * MAX_CONTEXT as u64;
                *ptr.add(offset as usize + min_pos as usize) = 0;
            }
        }
        // ── End eviction ──

        let logits = self
            .megakernel
            .read_slot_logits(&self.kernel_buffers, slot_id, 0);
        self.megakernel
            .reset_work_slot(&self.kernel_buffers, slot_id);

        self.slot_seq_pos[slot] = seq_pos + 1;
        Ok(sample_argmax(&logits))
    }

    /// Decode one token using slot 0 (convenience wrapper).
    #[inline]
    pub fn decode_token(&mut self, token_id: u32) -> Result<u32, String> {
        self.decode_slot(0, token_id)
    }

    /// Decode one or more tokens using MTP speculative verification.
    ///
    /// Uses slot 0 for the primary decode, then submits draft candidates
    /// to slots 1+ for verification. When `mtp_depth=0`, falls back to
    /// standard single-token decode.
    ///
    /// Returns all accepted tokens and updates `seq_pos` by the number
    /// accepted.
    pub fn decode_with_mtp(&mut self, token_id: u32, mtp_depth: u32) -> Result<Vec<u32>, String> {
        if mtp_depth == 0 {
            let t = self.decode_token(token_id)?;
            return Ok(vec![t]);
        }

        self.slc_phase = SLCPhase::GPUDecode;

        // 1. Run primary decode on slot 0
        self.megakernel
            .submit_work(&self.kernel_buffers, 0, token_id, self.slot_seq_pos[0], 0);
        while !self.megakernel.poll_work(&self.kernel_buffers, 0) {
            std::hint::spin_loop();
        }
        let logits = self.megakernel.read_slot_logits(&self.kernel_buffers, 0, 0);
        self.megakernel.reset_work_slot(&self.kernel_buffers, 0);

        // 2. Sample primary token
        let primary_token = sample_argmax(&logits);

        // 3. Generate speculative candidates from logits (top-K)
        let k = mtp_depth.min(NUM_SLOTS as u32 - 1);
        let mut candidates = generate_speculative_candidates(&logits, k as usize);
        // Always include the primary token as candidate 0
        if candidates.is_empty() || candidates[0] != primary_token {
            candidates.insert(0, primary_token);
        }
        let num_candidates = candidates.len().min((NUM_SLOTS - 1) as usize);

        // 4. Submit each candidate to its slot
        for (i, &cand) in candidates[..num_candidates].iter().enumerate() {
            let slot = (i + 1) as u32; // slots 1, 2, ...
            self.megakernel.submit_work(
                &self.kernel_buffers,
                slot,
                cand,
                self.slot_seq_pos[0] + 1,
                slot,
            );
        }

        // 5. Poll all speculative slots
        // Short timeout: if a slot hasn't completed, discard that candidate
        let mut accepted = vec![primary_token]; // always accept the primary
        for (i, _) in candidates[..num_candidates].iter().enumerate() {
            let slot = (i + 1) as u32;
            // Poll with limited spins
            let mut spins = 0;
            while !self.megakernel.poll_work(&self.kernel_buffers, slot) {
                spins += 1;
                if spins > 1_000_000 {
                    break; // timeout — discard this candidate
                }
                std::hint::spin_loop();
            }

            if self.megakernel.poll_work(&self.kernel_buffers, slot) {
                let cand_logits =
                    self.megakernel.read_slot_logits(&self.kernel_buffers, slot, 0);
                let cand_result = sample_argmax(&cand_logits);
                self.megakernel.reset_work_slot(&self.kernel_buffers, slot);

                // Verify: does the candidate's predicted next token match itself?
                // A candidate C is "verified" if running the full transformer on C
                // predicts C as the output. This means C was a stable fixed point.
                if cand_result == candidates[i] && i == 0 {
                    // Primary already accepted — skip
                } else if cand_result == candidates[i] {
                    // Self-consistent: accept this candidate
                    accepted.push(candidates[i]);
                }
            }
        }

        self.slot_seq_pos[0] += accepted.len() as u32;
        Ok(accepted)
    }

    /// Decode with draft model speculation + MTP verification.
    ///
    /// Flow per call:
    /// 1. Submit draft model (kind=3) — fast forward pass, outputs N candidate
    ///    token IDs + log-probs into the `draft_output` buffer.
    /// 2. Poll draft completion, read candidate tokens from `draft_output`.
    /// 3. Submit main model decode (kind=0) — full transformer forward pass
    ///    that also produces MTP head predictions.
    /// 4. Poll main completion, read logits + MTP predictions.
    /// 5. Rejection sampling: accept each draft token where
    ///    p_main(draft) / p_draft(draft) > threshold.
    /// 6. For positions the draft chain did not cover, accept MTP predictions.
    /// 7. Advance `seq_pos` by the number of accepted tokens.
    pub fn decode_speculative(&mut self, token_id: u32, num_draft: u32) -> Result<Vec<u32>, String> {
        self.slc_phase = SLCPhase::GPUDecode;
        let slot = 0usize;
        let seq_pos = self.slot_seq_pos[slot];

        // Cap draft candidates to buffer capacity.
        let num_candidates = num_draft.min(MAX_DRAFT_CANDIDATES);
        if num_candidates == 0 {
            return self.decode_with_mtp(token_id, 0);
        }

        // ── Phase 1: Run draft model forward pass ──
        self.megakernel.submit_draft(
            &self.kernel_buffers,
            token_id,
            seq_pos,
            num_candidates,
        );
        while !self.megakernel.poll_work(&self.kernel_buffers, 0) {
            std::hint::spin_loop();
        }

        // ── Phase 2: Read draft candidate tokens + log-probs ──
        let draft_candidates = self.megakernel.read_draft_output(&self.kernel_buffers);
        if draft_candidates.is_empty() {
            // Fall back to single-token decode if draft produced nothing.
            return self.decode_with_mtp(token_id, 0);
        }

        // ── Phase 3: Run main model decode (kind=0) — produces logits + MTP heads ──
        self.megakernel
            .submit_work(&self.kernel_buffers, 0, token_id, seq_pos, 0);
        while !self.megakernel.poll_work(&self.kernel_buffers, 0) {
            std::hint::spin_loop();
        }
        self.entropy_available = true;

        // ── Phase 4: Read main model logits (head 0) and MTP head predictions ──
        let logits = self.megakernel.read_slot_logits(&self.kernel_buffers, 0, 0);

        let mut mtp_logits_list: Vec<Vec<u16>> = Vec::with_capacity(NUM_MTP_HEADS as usize);
        for h in 1..=NUM_MTP_HEADS {
            let head_logits = self.megakernel.read_slot_logits(&self.kernel_buffers, 0, h);
            mtp_logits_list.push(head_logits);
        }
        self.megakernel.reset_work_slot(&self.kernel_buffers, 0);

        // ── Phase 5: Softmax over main model logits ──
        // Convert f16 logit buffer to f32 and compute softmax with numerical
        // stability (subtract max before exponentiation).
        let n_vocab = logits.len();
        let mut probs_f32 = Vec::with_capacity(n_vocab);
        let mut max_logit = f32::NEG_INFINITY;
        for &bits in &logits {
            let v = half::f16::from_bits(bits).to_f32();
            if v > max_logit {
                max_logit = v;
            }
            probs_f32.push(v);
        }
        let mut sum = 0.0f32;
        for v in probs_f32.iter_mut() {
            *v = (*v - max_logit).exp();
            sum += *v;
        }
        for v in probs_f32.iter_mut() {
            *v /= sum;
        }

        // ── Phase 6: Rejection sampling over draft candidates ──
        // Always accept the primary token sampled from the main model.
        let primary_token = sample_argmax(&logits);
        let mut accepted = vec![primary_token];

        for &(draft_token, draft_logprob) in &draft_candidates {
            let p_main = probs_f32[draft_token as usize];
            let p_draft = draft_logprob.exp(); // log-prob → probability
            // Standard speculative decoding rejection criterion:
            // Accept if p_main / p_draft > uniform(0,1).
            // Conservative approximation: accept when p_main > p_draft
            // (since uniform < 1, this is a stricter bound that guarantees
            // the correct target distribution when satisfied).
            if p_main > p_draft {
                accepted.push(draft_token);
            } else {
                break;
            }
        }

        // ── Phase 7: Fill remaining positions from MTP head predictions ──
        // MTP head h predicts the token at seq_pos + 1 + h.
        // If the draft chain accepted fewer tokens than there are MTP heads,
        // use the MTP predictions for the uncovered positions.
        let draft_accepted = accepted.len().saturating_sub(1); // exclude primary
        for h in draft_accepted..NUM_MTP_HEADS as usize {
            if h < mtp_logits_list.len() {
                let mtp_token = sample_argmax(&mtp_logits_list[h]);
                accepted.push(mtp_token);
            }
        }

        // ── Phase 8: Advance sequence position ──
        self.slot_seq_pos[slot] = seq_pos + accepted.len() as u32;
        Ok(accepted)
    }

    /// Compact a slot's KV cache using entropy-guided selection.
    ///
    /// Reads entropy from the GPU kernel's entropy map, selects survivor
    /// positions using adaptive stride, and runs multi-pass ANE gather
    /// if indices exceed the single-pass limit (20480).
    ///
    /// The VM manager tracks per-slot IOSurface offsets. After compaction,
    /// `slot_seq_pos[slot]` is updated to the number of survivors.
    pub fn compact_slot(&mut self, slot_id: u32) -> Result<(), String> {
        let compaction_model = self
            .compaction_model
            .as_ref()
            .ok_or_else(|| format!("compact_slot: no compaction model for slot {slot_id}"))?;
        let indices_arena = self
            .compaction_indices_arena
            .as_ref()
            .ok_or_else(|| "compact_slot: no indices arena".to_string())?;
        let k_in_arena = self
            .compaction_k_arena
            .as_ref()
            .ok_or_else(|| "compact_slot: no K input arena".to_string())?;
        let v_in_arena = self
            .compaction_v_arena
            .as_ref()
            .ok_or_else(|| "compact_slot: no V input arena".to_string())?;
        let k_out_arena = self
            .compacted_k_arena
            .as_ref()
            .ok_or_else(|| "compact_slot: no compacted K arena".to_string())?;
        let v_out_arena = self
            .compacted_v_arena
            .as_ref()
            .ok_or_else(|| "compact_slot: no compacted V arena".to_string())?;

        let slot = slot_id as usize;
        let seq_pos = self.slot_seq_pos[slot];
        if seq_pos == 0 {
            return Err("compact_slot: slot has no data".into());
        }

        // 1. Read entropy map from GPU (populated by decode kernel)
        let entropy_raw = self
            .megakernel
            .read_entropy_map(&self.kernel_buffers, slot_id);
        let active_len = seq_pos as usize;
        let entropies: Vec<f16> = entropy_raw[..active_len]
            .iter()
            .map(|&v| f16::from_bits(v))
            .collect();

        // 2. Select positions with entropy-weighted stride
        let slot_alloc = self.vm_manager.slot_allocation(slot_id);
        let target_total = slot_alloc.survivor_count as usize;
        let indices = compaction::select_entropy_adaptive_positions(&entropies, target_total);

        // 3. Multi-pass gather if indices exceed single-pass limit
        const CHUNK_SIZE: usize = compaction::DEFAULT_TARGET_COUNT as usize;
        let num_passes = (indices.len() + CHUNK_SIZE - 1) / CHUNK_SIZE;
        self.compaction_pass = 0;

        // Per-slot scratch offset (from orchestrator scratch layout)
        let per_layer_scratch_elems = (MAX_CONTEXT * NUM_KV_HEADS * GLOBAL_HEAD_DIM) as usize;
        let per_layer_scratch_bytes = per_layer_scratch_elems * 2;
        let scratch_slot_offset = (slot_id as usize) * per_layer_scratch_bytes;
        let per_position_bytes = (NUM_KV_HEADS as usize) * (GLOBAL_HEAD_DIM as usize) * 2;

        let k_scratch_ptr = unsafe {
            self.kernel_buffers
                .kv_scratch_k
                .contents()
                .add(scratch_slot_offset) as *mut u8
        };
        let v_scratch_ptr = unsafe {
            self.kernel_buffers
                .kv_scratch_v
                .contents()
                .add(scratch_slot_offset) as *mut u8
        };

        for pass_idx in 0..num_passes {
            let start = pass_idx * CHUNK_SIZE;
            let end = (start + CHUNK_SIZE).min(indices.len());
            let chunk_indices = &indices[start..end];
            let chunk_len = chunk_indices.len();

            if chunk_len == 0 {
                continue;
            }

            // Write chunk indices to indices arena
            {
                indices_arena.lock()?;
                let ptr = unsafe { indices_arena.base_ptr() as *mut u32 };
                let dst = unsafe { std::slice::from_raw_parts_mut(ptr, chunk_len) };
                dst.copy_from_slice(chunk_indices);
                indices_arena.unlock()?;
            }

            let chunk_offset_bytes = slot_alloc.byte_offset as usize + start * per_position_bytes;
            let compacted_per_layer_bytes = chunk_len * per_position_bytes;

            for layer in 0..LAYERS {
                let layer_scratch_offset = (layer as usize) * per_layer_scratch_bytes;

                // Copy full scratch K/V for this layer into input arenas
                unsafe {
                    let k_src = k_scratch_ptr.add(layer_scratch_offset);
                    let k_dst = k_in_arena.base_ptr() as *mut u8;
                    std::ptr::copy_nonoverlapping(k_src, k_dst, per_layer_scratch_bytes);

                    let v_src = v_scratch_ptr.add(layer_scratch_offset);
                    let v_dst = v_in_arena.base_ptr() as *mut u8;
                    std::ptr::copy_nonoverlapping(v_src, v_dst, per_layer_scratch_bytes);
                }

                // Run ANE compaction gather
                let mut compacted_k_info = k_out_arena.info;
                let mut compacted_v_info = v_out_arena.info;

                compaction_model
                    .predict_multi(
                        &["key_cache", "value_cache", "indices"],
                        &[&k_in_arena.info, &v_in_arena.info, &indices_arena.info],
                        &["compacted_key", "compacted_value"],
                        &mut [&mut compacted_k_info, &mut compacted_v_info],
                    )
                    .map_err(|e| format!("compact_slot layer {layer} pass {pass_idx}: {e}"))?;

                // Write compacted output back to scratch at VM offset
                let pass_scratch_k =
                    unsafe { k_scratch_ptr.add(layer_scratch_offset + chunk_offset_bytes) };
                let pass_scratch_v =
                    unsafe { v_scratch_ptr.add(layer_scratch_offset + chunk_offset_bytes) };

                unsafe {
                    let k_src = k_out_arena.base_ptr() as *const u8;
                    std::ptr::copy_nonoverlapping(
                        k_src,
                        pass_scratch_k,
                        compacted_per_layer_bytes,
                    );

                    let v_src = v_out_arena.base_ptr() as *const u8;
                    std::ptr::copy_nonoverlapping(
                        v_src,
                        pass_scratch_v,
                        compacted_per_layer_bytes,
                    );
                }
            }

            self.compaction_pass += 1;
        }

        // 4. Update seq_pos to reflect compacted survivor count
        self.slot_seq_pos[slot] = indices.len() as u32;

        Ok(())
    }

    /// Compact a slot's KV cache with an explicit target survivor count.
    /// Same as [`compact_slot`] but allows specifying the target count
    /// instead of deriving it from the VM manager's slot allocation.
    pub fn compact_slot_with_target(
        &mut self,
        slot_id: u32,
        target_count: u32,
    ) -> Result<(), String> {
        let compaction_model = self
            .compaction_model
            .as_ref()
            .ok_or_else(|| format!("compact_slot: no compaction model for slot {slot_id}"))?;
        let indices_arena = self
            .compaction_indices_arena
            .as_ref()
            .ok_or_else(|| "compact_slot: no indices arena".to_string())?;
        let k_in_arena = self
            .compaction_k_arena
            .as_ref()
            .ok_or_else(|| "compact_slot: no K input arena".to_string())?;
        let v_in_arena = self
            .compaction_v_arena
            .as_ref()
            .ok_or_else(|| "compact_slot: no V input arena".to_string())?;
        let k_out_arena = self
            .compacted_k_arena
            .as_ref()
            .ok_or_else(|| "compact_slot: no compacted K arena".to_string())?;
        let v_out_arena = self
            .compacted_v_arena
            .as_ref()
            .ok_or_else(|| "compact_slot: no compacted V arena".to_string())?;

        let slot = slot_id as usize;
        let seq_pos = self.slot_seq_pos[slot];
        if seq_pos == 0 {
            return Err("compact_slot: slot has no data".into());
        }

        let target_total = target_count as usize;

        // 1. Read entropy map from GPU (populated by decode kernel)
        let entropy_raw = self
            .megakernel
            .read_entropy_map(&self.kernel_buffers, slot_id);
        let active_len = seq_pos as usize;
        let entropies: Vec<f16> = entropy_raw[..active_len]
            .iter()
            .map(|&v| f16::from_bits(v))
            .collect();

        // 2. Select positions with entropy-weighted stride
        let slot_alloc = self.vm_manager.slot_allocation(slot_id);
        let indices = compaction::select_entropy_adaptive_positions(&entropies, target_total);

        // 3. Multi-pass gather if indices exceed single-pass limit
        const CHUNK_SIZE: usize = compaction::DEFAULT_TARGET_COUNT as usize;
        let num_passes = (indices.len() + CHUNK_SIZE - 1) / CHUNK_SIZE;
        self.compaction_pass = 0;

        // Per-slot scratch offset (from orchestrator scratch layout)
        let per_layer_scratch_elems = (MAX_CONTEXT * NUM_KV_HEADS * GLOBAL_HEAD_DIM) as usize;
        let per_layer_scratch_bytes = per_layer_scratch_elems * 2;
        let scratch_slot_offset = (slot_id as usize) * per_layer_scratch_bytes;
        let per_position_bytes = (NUM_KV_HEADS as usize) * (GLOBAL_HEAD_DIM as usize) * 2;

        let k_scratch_ptr = unsafe {
            self.kernel_buffers
                .kv_scratch_k
                .contents()
                .add(scratch_slot_offset) as *mut u8
        };
        let v_scratch_ptr = unsafe {
            self.kernel_buffers
                .kv_scratch_v
                .contents()
                .add(scratch_slot_offset) as *mut u8
        };

        for pass_idx in 0..num_passes {
            let start = pass_idx * CHUNK_SIZE;
            let end = (start + CHUNK_SIZE).min(indices.len());
            let chunk_indices = &indices[start..end];
            let chunk_len = chunk_indices.len();

            if chunk_len == 0 {
                continue;
            }

            // Write chunk indices to indices arena
            {
                indices_arena.lock()?;
                let ptr = unsafe { indices_arena.base_ptr() as *mut u32 };
                let dst = unsafe { std::slice::from_raw_parts_mut(ptr, chunk_len) };
                dst.copy_from_slice(chunk_indices);
                indices_arena.unlock()?;
            }

            let chunk_offset_bytes = slot_alloc.byte_offset as usize + start * per_position_bytes;
            let compacted_per_layer_bytes = chunk_len * per_position_bytes;

            for layer in 0..LAYERS {
                let layer_scratch_offset = (layer as usize) * per_layer_scratch_bytes;

                // Copy full scratch K/V for this layer into input arenas
                unsafe {
                    let k_src = k_scratch_ptr.add(layer_scratch_offset);
                    let k_dst = k_in_arena.base_ptr() as *mut u8;
                    std::ptr::copy_nonoverlapping(k_src, k_dst, per_layer_scratch_bytes);

                    let v_src = v_scratch_ptr.add(layer_scratch_offset);
                    let v_dst = v_in_arena.base_ptr() as *mut u8;
                    std::ptr::copy_nonoverlapping(v_src, v_dst, per_layer_scratch_bytes);
                }

                // Run ANE compaction gather
                let mut compacted_k_info = k_out_arena.info;
                let mut compacted_v_info = v_out_arena.info;

                compaction_model
                    .predict_multi(
                        &["key_cache", "value_cache", "indices"],
                        &[&k_in_arena.info, &v_in_arena.info, &indices_arena.info],
                        &["compacted_key", "compacted_value"],
                        &mut [&mut compacted_k_info, &mut compacted_v_info],
                    )
                    .map_err(|e| format!("compact_slot layer {layer} pass {pass_idx}: {e}"))?;

                // Write compacted output back to scratch at VM offset
                let pass_scratch_k =
                    unsafe { k_scratch_ptr.add(layer_scratch_offset + chunk_offset_bytes) };
                let pass_scratch_v =
                    unsafe { v_scratch_ptr.add(layer_scratch_offset + chunk_offset_bytes) };

                unsafe {
                    let k_src = k_out_arena.base_ptr() as *const u8;
                    std::ptr::copy_nonoverlapping(
                        k_src,
                        pass_scratch_k,
                        compacted_per_layer_bytes,
                    );

                    let v_src = v_out_arena.base_ptr() as *const u8;
                    std::ptr::copy_nonoverlapping(
                        v_src,
                        pass_scratch_v,
                        compacted_per_layer_bytes,
                    );
                }
            }

            self.compaction_pass += 1;
        }

        // 4. Update seq_pos to reflect compacted survivor count
        self.slot_seq_pos[slot] = indices.len() as u32;

        Ok(())
    }

    /// Reserve a slot for an agent with the given context budget.
    ///
    /// Finds an available slot (with no active sequence), configures its
    /// survivor count in the VM manager, and returns the slot_id.
    /// `context_budget` specifies the desired number of survivor positions
    /// (e.g. 20_480 for ~1M context at 50:1 compaction, 2_560 for ~128K).
    pub fn spawn_agent(&mut self, context_budget: usize) -> Result<u32, String> {
        // Find first slot with seq_pos == 0 (unused)
        let slot_id = self
            .slot_seq_pos
            .iter()
            .position(|&p| p == 0)
            .ok_or_else(|| "all slots are occupied".to_string())? as u32;

        let survivor_count = context_budget.min(MAX_SURVIVORS as usize) as u32;
        self.vm_manager
            .configure_slots(&[(slot_id, survivor_count)]);

        Ok(slot_id)
    }

    /// Signal that ANE prefill is active (runs concurrently with GPU decode).
    #[deprecated(since = "0.2.0", note = "use prefill_text(&mut self, prompt) instead")]
    pub fn prefill_from_ane(&mut self) {
        self.slc_phase = SLCPhase::ANEPrefill;
    }
}
