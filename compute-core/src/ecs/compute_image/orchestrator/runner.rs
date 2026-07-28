#![cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
//! Main inference orchestrator — loading, decode, compaction, agent management.
//!
//! Owns a loaded `.cimage` deployment, the full-transformer GPU megakernel
//! (embedding → 48 layers → logits), tree-attention, and optional ANE
//! prefill model for prompt processing.

use super::kernel_fusion;
use super::{
    generate_speculative_candidates, sample_argmax, sample_argmax_f32, GLOBAL_HEAD_DIM, LAYERS,
    MAX_CONTEXT, MAX_SURVIVORS, NUM_KV_HEADS, NUM_SLOTS,
};
use crate::arena::Arena;
use crate::arena::DataType;
use crate::coreai_bridge::{CoreAiComputeUnits, CoreAiModel};
use crate::ecs::compute_image::cimage_loader::CimageDeployment;
use crate::ecs::compute_image::compaction;
use crate::ecs::compute_image::compile::execution_graph::{ExecutionGraphDescriptor, NodeKind};
use crate::ecs::compute_image::compile::kernel_dispatch::{
    Nf4Tile640Offsets, Nf4Tile640ProjectionDispatcher,
};
use crate::ecs::compute_image::compile::kernel_registry::KernelRegistry;
use crate::ecs::compute_image::compile::kernel_types::{KernelReceipt, ProjectionParams};
pub use crate::ecs::compute_image::legacy_compute_image_runtime::megakernel::kernels::TapMode;
use crate::ecs::compute_image::legacy_compute_image_runtime::megakernel::{KernelBuffers, Megakernel};
use crate::ecs::compute_image::legacy_compute_image_runtime::megakernel::{MAX_DRAFT_CANDIDATES, NUM_MTP_HEADS};
use crate::ecs::compute_image::legacy_compute_image_runtime::multimodal::binding::SealedMultimodalBindings;
use crate::ecs::compute_image::legacy_compute_image_runtime::multimodal::descriptor::ProjectionTensorRecord;
use crate::ecs::compute_image::tree_attention::TreeAttention;
use crate::ecs::compute_image::vm_manager::VmManager;
use half::f16;
use metal::*;
use parking_lot::Mutex;
use std::path::PathBuf;
use std::sync::Arc;

pub struct Nf4GraphPrefixInputs<'a> {
    pub vision_patch: Option<&'a [f32]>,
    pub vision_projection: Option<&'a [f32]>,
    pub audio_frame: Option<&'a [f32]>,
    pub audio_projection: Option<&'a [f32]>,
}

pub struct Nf4GraphPrefixOutputs {
    pub vision_patch: Option<Vec<f32>>,
    pub vision_projection: Option<Vec<f32>>,
    pub audio_frame: Option<Vec<f32>>,
    pub audio_projection: Option<Vec<f32>>,
    pub next_node_index: usize,
}

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
    /// Compiled ANE prefill model loaded from the cimage's MIL program.
    /// Set by [`Self::compile_ane_model`] when `deployment.mil_buffer`
    /// is present and compilation succeeds.
    /// One model instance per work queue slot for parallel prefill.
    pub ane_prefill_models: Vec<Option<CoreAiModel>>,
    /// Cache path for the compiled .mlmodelc bundle (alongside the cimage).
    pub ane_modelc_path: Option<PathBuf>,
    /// Compiled ANE compaction gather model. Loaded when compaction
    /// MIL program compiles successfully.
    pub compaction_model: Option<CoreAiModel>,
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
    /// model bytes. Built at compile time by the ECS packer path via
    /// coremlcompiler. One model instance per work queue slot.
    pub prefill_model: Option<CoreAiModel>,
    /// How this orchestrator was built with respect to Stage 0 activation
    /// taps — decided at construction, recorded in operational receipts. The
    /// parity audit refuses to run against an `Untapped` teacher before any
    /// decoding begins ([`Self::decode_token_logits_with_taps`]).
    pub tap_mode: TapMode,
}

impl Orchestrator {
    pub fn validate_nf4_execution_graph(
        &self,
        graph: &ExecutionGraphDescriptor,
    ) -> Result<usize, String> {
        if !self.deployment.is_nf4_tile640() {
            return Err("validate_nf4_execution_graph requires an NF4Tile640 deployment".into());
        }
        let _ = self.deployment.require_nf4_biases()?;
        let bindings = SealedMultimodalBindings::from_deployment(&self.deployment)?;
        if bindings.projection_precision
            != crate::ecs::compute_image::legacy_compute_image_runtime::multimodal::ProjectionPrecision::Nf4Tile640
        {
            return Err(format!(
                "sealed multimodal projection precision is not NF4Tile640: {:?}",
                bindings.projection_precision
            ));
        }
        bindings.validate_graph_multimodal_prefix(graph)
    }

    /// Execute one multimodal NF4Tile640 projection record directly against the
    /// sealed shared-weight arena.
    pub fn run_nf4_multimodal_projection(
        &self,
        record: &ProjectionTensorRecord,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let _layout = record.validate_nf4_tile640()?;
        let weights = self
            .deployment
            .multimodal_projection_weights_buffer
            .as_ref()
            .ok_or_else(|| "multimodal NF4 projection weights unavailable".to_string())?;
        let scales = self
            .deployment
            .multimodal_projection_scales_buffer
            .as_ref()
            .ok_or_else(|| "multimodal NF4 projection scales unavailable".to_string())?;

        let expected_in = record.input_width as usize;
        let expected_out = record.output_width as usize;
        if expected_in == 0 || expected_out == 0 {
            return Err("projection record has zero input/output width".into());
        }
        if input.len() != expected_in {
            return Err(format!(
                "projection input width mismatch: got {}, expected {}",
                input.len(),
                expected_in
            ));
        }

        let input_buf = self.device.new_buffer_with_data(
            input.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(input) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let output_buf = self.device.new_buffer(
            (expected_out * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Bias residency (kernels/MULTIMODAL_NF4_BIAS_ABI.md, implemented):
        // when the record carries FLAG_HAS_BIAS and the artifact seals a bias
        // segment, bind the REAL resident biases — addressed by the record's
        // scale geometry per the parallel-layout contract. Otherwise fall
        // back to a zero buffer, which is numerically exact for every
        // artifact the symmetric NF4 quantizer produces (bias ≡ 0 by
        // construction). The taken path is logged per projection so there is
        // never ambiguity about whether the shared bias arena was used
        // (PRODUCTION_CONTRACT.md).
        let resident_biases = if record.has_bias() {
            match self.deployment.multimodal_projection_biases_buffer.as_ref() {
                Some(buf) => Some(buf),
                None => {
                    return Err(format!(
                        "projection record {:#x} sets FLAG_HAS_BIAS but the loaded \
                         artifact has no MultimodalProjectionBiases segment — refusing \
                         the silent zero fallback for a record that declares residency",
                        record.logical_name_hash
                    ));
                }
            }
        } else {
            None
        };
        let zero_biases;
        let (biases_buf, biases_offset): (&metal::Buffer, u64) = match resident_biases {
            Some(buf) => {
                eprintln!(
                    "[multimodal-nf4] projection {:#x}: bias residency = RESIDENT \
                     (segment-backed, offset {}, {} bytes)",
                    record.logical_name_hash, record.scale_offset, record.scale_length
                );
                (buf, record.scale_offset)
            }
            None => {
                eprintln!(
                    "[multimodal-nf4] projection {:#x}: bias residency = ZERO-FALLBACK \
                     (v1-compat: record has no FLAG_HAS_BIAS)",
                    record.logical_name_hash
                );
                zero_biases = self.device.new_buffer(
                    record.scale_length.max(4),
                    MTLResourceOptions::StorageModeShared,
                );
                // Metal shared buffers are zero-initialized only on some
                // paths; make the fallback contract explicit.
                unsafe {
                    std::ptr::write_bytes(
                        zero_biases.contents() as *mut u8,
                        0,
                        record.scale_length.max(4) as usize,
                    );
                }
                (&zero_biases, 0)
            }
        };

        let registry = Arc::new(Mutex::new(KernelRegistry::new(&self.device)));
        let dispatcher = Nf4Tile640ProjectionDispatcher::new(registry);
        let params = ProjectionParams {
            in_dim: record.input_width,
            out_dim: record.output_width,
            page_count: record.input_width.div_ceil(640),
            page_width: 640,
            mode_flags: 0,
            probe_seed: 0,
            reserved: [0; 5],
        };
        let mut receipt = KernelReceipt {
            kernel_id: 0,
            phase_id: 0,
            page_count: 0,
            sidecar_hits: 0,
            sidecar_entries_read: 0,
            threadgroups: 0,
            threads_per_threadgroup: 0,
            output_elements: 0,
            flags: 0,
            logical_weight_bytes: 0,
            logical_sidecar_bytes: 0,
            logical_activation_bytes: 0,
        };

        let cb = self.queue.new_command_buffer();
        dispatcher.dispatch_with_offsets(
            &cb,
            weights,
            scales,
            biases_buf,
            &input_buf,
            &output_buf,
            &params,
            Nf4Tile640Offsets {
                weights_offset: record.weight_offset,
                scales_offset: record.scale_offset,
                biases_offset,
            },
            &mut receipt,
        );
        cb.commit();
        cb.wait_until_completed();

        let out = unsafe {
            std::slice::from_raw_parts(output_buf.contents() as *const f32, expected_out)
        };
        Ok(out.to_vec())
    }

    /// Resolve one multimodal execution-graph node against the sealed binding
    /// table and execute it through the explicit NF4Tile640 Metal projection
    /// path.
    pub fn run_nf4_multimodal_node(
        &self,
        node: &crate::ecs::compute_image::compile::execution_graph::LayerExecutionNode,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let bindings = SealedMultimodalBindings::from_deployment(&self.deployment)?;
        if bindings.projection_precision
            != crate::ecs::compute_image::legacy_compute_image_runtime::multimodal::ProjectionPrecision::Nf4Tile640
        {
            return Err(format!(
                "multimodal projection graph is not sealed as NF4Tile640: {:?}",
                bindings.projection_precision
            ));
        }
        let binding = bindings.validate_node_binding(node)?;
        self.run_nf4_multimodal_projection(&binding.record, input)
    }

    pub fn run_nf4_graph_multimodal_prefix(
        &self,
        graph: &ExecutionGraphDescriptor,
        inputs: Nf4GraphPrefixInputs<'_>,
    ) -> Result<Nf4GraphPrefixOutputs, String> {
        let next_node_index = self.validate_nf4_execution_graph(graph)?;
        let mut outputs = Nf4GraphPrefixOutputs {
            vision_patch: None,
            vision_projection: None,
            audio_frame: None,
            audio_projection: None,
            next_node_index,
        };

        for node in graph.layers.iter().take(next_node_index) {
            match node.node_kind {
                x if x == NodeKind::VisionPatchEmbed as u8 => {
                    let input = inputs.vision_patch.ok_or_else(|| {
                        "NF4 graph requires vision patch input for VisionPatchEmbed".to_string()
                    })?;
                    outputs.vision_patch = Some(self.run_nf4_multimodal_node(node, input)?);
                }
                x if x == NodeKind::VisionFinalProjection as u8 => {
                    let input = inputs.vision_projection.ok_or_else(|| {
                        "NF4 graph requires vision projection input for VisionFinalProjection"
                            .to_string()
                    })?;
                    outputs.vision_projection = Some(self.run_nf4_multimodal_node(node, input)?);
                }
                x if x == NodeKind::AudioFrameEmbed as u8 => {
                    let input = inputs.audio_frame.ok_or_else(|| {
                        "NF4 graph requires audio frame input for AudioFrameEmbed".to_string()
                    })?;
                    outputs.audio_frame = Some(self.run_nf4_multimodal_node(node, input)?);
                }
                x if x == NodeKind::AudioProjection as u8 => {
                    let input = inputs.audio_projection.ok_or_else(|| {
                        "NF4 graph requires audio projection input for AudioProjection".to_string()
                    })?;
                    outputs.audio_projection = Some(self.run_nf4_multimodal_node(node, input)?);
                }
                x if x == NodeKind::EmbeddingAssembly as u8 => {}
                other => {
                    return Err(format!(
                        "run_nf4_graph_multimodal_prefix encountered non-prefix node kind {}",
                        other
                    ));
                }
            }
        }

        Ok(outputs)
    }

    /// Look up or create a compute pipeline state for a kernel function name.
    /// Loads the metallib from the deployment on first call; Metal caches PSOs internally.
    fn get_pso(&self, kernel_name: &str) -> Result<metal::ComputePipelineState, String> {
        let metallib_buf = self
            .deployment
            .metallib_buffer
            .as_ref()
            .ok_or_else(|| "get_pso: no metallib buffer in deployment".to_string())?;
        let data = unsafe {
            std::slice::from_raw_parts(
                metallib_buf.contents() as *const u8,
                metallib_buf.length() as usize,
            )
        };
        let library = self
            .device
            .new_library_with_data(data)
            .map_err(|e| format!("get_pso: failed to load metallib: {e}"))?;
        let function = library
            .get_function(kernel_name, None)
            .map_err(|e| format!("get_pso: kernel '{kernel_name}' not found: {e}"))?;
        self.device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| format!("get_pso: failed to create PSO for '{kernel_name}': {e}"))
    }

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
        // Back-compat: the tap mode defaults from the environment
        // (`TRIBUNUS_TAPS=1` ⇒ tapped-audit). New call sites — the parity
        // stage in particular — pass the mode explicitly.
        Self::from_cimage_with_mode(path, batch_size, int4_mode, TapMode::from_env())
    }

    /// [`Self::from_cimage`] with the tap mode stated explicitly — the
    /// declared-mode construction path (PRODUCTION_CONTRACT.md): audit
    /// tooling passes [`TapMode::TappedAudit`] and never depends on ambient
    /// environment variables; production passes [`TapMode::Untapped`].
    pub fn from_cimage_with_mode(
        path: impl AsRef<std::path::Path>,
        batch_size: u32,
        int4_mode: bool,
        tap_mode: TapMode,
    ) -> Result<Self, String> {
        let path = path.as_ref();
        let device = Device::system_default().ok_or("no Metal device available")?;
        let queue = device.new_command_queue();
        let mut deployment = CimageDeployment::load(path, &device)?;
        if int4_mode {
            deployment.maybe_expand_to_int4(&device)?;
        }
        let megakernel = Megakernel::new(&device, &queue, &deployment, int4_mode, tap_mode)?;
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
        let mut ane_prefill_models: Vec<Option<CoreAiModel>> = Vec::with_capacity(num_slots);
        for i in 0..num_slots {
            if i == 0 {
                ane_prefill_models.push(first_model.take());
            } else if let Some(ref cache_path) = ane_modelc_path {
                match CoreAiModel::load_with_compute_units(
                    &cache_path.to_string_lossy(),
                    CoreAiComputeUnits::CpuAndNeuralEngine,
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
        let (
            compaction_indices_arena,
            compaction_k_arena,
            compaction_v_arena,
            compacted_k_arena,
            compacted_v_arena,
        ) = Self::allocate_compaction_arenas(
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
            tap_mode,
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
                let per_position_bytes = (NUM_KV_HEADS as usize) * (GLOBAL_HEAD_DIM as usize) * 2;
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
        Ok(sample_argmax_f32(
            &self.decode_slot_logits(slot_id, token_id)?,
        ))
    }

    /// Like [`decode_slot`] but returns the full output logit vector instead of
    /// just the argmax token. Used by the benchmark harness to score perplexity,
    /// KL divergence, and top-1 agreement. Behaviour (KV advance, eviction) is
    /// identical — `decode_slot` is a thin argmax wrapper over this.
    pub fn decode_slot_logits(&mut self, slot_id: u32, token_id: u32) -> Result<Vec<f32>, String> {
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
            let entropy = self
                .megakernel
                .read_entropy_map(&self.kernel_buffers, slot_id as u32);

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

        // read_slot_logits returns the megakernel's raw FP16 logits as u16
        // half-bits; the scoring API contract here is f32 (the bench harness,
        // KD gate, and Gemma4Teacher::teacher_forced all consume f32).
        let raw = self
            .megakernel
            .read_slot_logits(&self.kernel_buffers, slot_id, 0);
        self.megakernel
            .reset_work_slot(&self.kernel_buffers, slot_id);

        self.slot_seq_pos[slot] = seq_pos + 1;
        Ok(raw.iter().map(|&b| f16::from_bits(b).to_f32()).collect())
    }

    /// Decode one token using slot 0 (convenience wrapper).
    #[inline]
    pub fn decode_token(&mut self, token_id: u32) -> Result<u32, String> {
        self.decode_slot(0, token_id)
    }

    /// Decode on slot 0 returning `(argmax_token, full_logits)` — the scoring
    /// hook for the benchmark harness (perplexity / KL / top-1 agreement).
    #[inline]
    pub fn decode_token_logits(&mut self, token_id: u32) -> Result<(u32, Vec<f32>), String> {
        let logits = self.decode_slot_logits(0, token_id)?;
        Ok((sample_argmax_f32(&logits), logits))
    }

    /// Decode one token on slot 0 and return `(argmax, logits, taps)` — the
    /// Stage 0 audit hook (kernels/STAGE0_TAPS_SPEC.md, Transport A).
    ///
    /// Requires the megakernel to have been compiled with taps — set
    /// `TRIBUNUS_TAPS=1` BEFORE constructing this Orchestrator (the persistent
    /// kernel compiles taps in at construction time). Errors rather than
    /// returning stale/zeroed taps otherwise, and verifies the in-kernel
    /// progress counter reached the final slot before trusting the buffer.
    pub fn decode_token_logits_with_taps(
        &mut self,
        token_id: u32,
    ) -> Result<(u32, Vec<f32>, LayerTaps), String> {
        if !self.tap_mode.is_tapped() {
            return Err(
                "taps not enabled: this Orchestrator was constructed Untapped — build it \
                 with from_cimage_with_mode(.., TapMode::TappedAudit) (or TRIBUNUS_TAPS=1 \
                 for the env-default constructor) before requesting taps"
                    .into(),
            );
        }
        let logits = self.decode_slot_logits(0, token_id)?;
        let progress = self.megakernel.read_tap_progress(&self.kernel_buffers);
        let expected = 2 * LAYERS + 1;
        if progress != expected {
            return Err(format!(
                "tap progress {progress} != expected final slot {expected} — \
                 kernel not compiled with -DPRISM_TAPS?"
            ));
        }
        let raw = self.megakernel.read_layer_taps(&self.kernel_buffers);
        let taps = LayerTaps::from_raw(raw)?;
        Ok((sample_argmax_f32(&logits), logits, taps))
    }

    /// Transport B, fusion group size 1 (the audit-pass configuration from
    /// STAGE0_TAPS_SPEC.md): dispatch `decode_layer_full_real` once per layer
    /// in a single command buffer, CHAINING per-layer device buffers — layer
    /// k's output buffer is layer k+1's input, and those 48 resident buffers
    /// ARE the blit-free boundary taps (read back after completion, no
    /// copies, no function constants, zero shader deltas).
    ///
    /// KV is the fused path's fp16 clean mode with per-layer caches sized for
    /// short audit windows. At `seq_position == 0` each layer is numerically
    /// identical to the megakernel's math; across the 48-layer chain the
    /// output drifts from the megakernel taps only by the megakernel's own
    /// ternary-KV noise (zero at pos 0) plus fp16 accumulation-order deltas —
    /// the parity test emits the per-layer drift curve.
    ///
    /// Production fusion (2–4 layers/dispatch) migrates separately once the
    /// pair/triple/quad bodies are real; this is the tap-bearing audit lane.
    pub fn decode_audit_group1(
        &self,
        device: &metal::Device,
        hidden_in: &[f32],
        seq_position: u32,
        audit_max_seq: u32,
    ) -> Result<Vec<Vec<f32>>, String> {
        use crate::ecs::compute_image::legacy_compute_image_runtime::megakernel::kernels::compile_layer_library;

        const HIDDEN: usize = 3840;
        if hidden_in.len() != HIDDEN {
            return Err(format!("hidden_in len {} != {HIDDEN}", hidden_in.len()));
        }
        let norms = self
            .deployment
            .norms_buffer
            .as_ref()
            .ok_or("deployment missing norms buffer")?;

        let lib = compile_layer_library(device)?;
        let f = lib
            .get_function("decode_layer_full_real", None)
            .map_err(|e| format!("decode_layer_full_real: {e}"))?;
        let pso = device
            .new_compute_pipeline_state_with_function(&f)
            .map_err(|e| format!("layer PSO: {e}"))?;

        let opts = metal::MTLResourceOptions::StorageModeShared;
        let mk_hidden = || device.new_buffer((HIDDEN * 2) as u64, opts);
        // boundary[0] = input; boundary[k+1] = layer k's output (the tap).
        let mut boundaries: Vec<metal::Buffer> = Vec::with_capacity(LAYERS as usize + 1);
        let in_buf = mk_hidden();
        unsafe {
            let dst = in_buf.contents() as *mut u16;
            for (i, &v) in hidden_in.iter().enumerate() {
                *dst.add(i) = f16::from_f32(v).to_bits();
            }
        }
        boundaries.push(in_buf);
        for _ in 0..LAYERS {
            boundaries.push(mk_hidden());
        }

        let stride = (NUM_KV_HEADS * GLOBAL_HEAD_DIM) as u64;
        let kv_bytes = audit_max_seq.max(1) as u64 * stride * 2;
        let kv: Vec<(metal::Buffer, metal::Buffer)> = (0..LAYERS)
            .map(|_| {
                (
                    device.new_buffer(kv_bytes, opts),
                    device.new_buffer(kv_bytes, opts),
                )
            })
            .collect();
        let ffn_scratch = device.new_buffer(2 * 15360 * 2, opts);

        let queue = device.new_command_queue();
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pso);
        for layer in 0..LAYERS as usize {
            enc.set_buffer(0, Some(&boundaries[layer]), 0);
            enc.set_buffer(1, Some(&boundaries[layer + 1]), 0);
            enc.set_buffer(2, Some(&kv[layer].0), 0);
            enc.set_buffer(3, Some(&kv[layer].1), 0);
            enc.set_buffer(4, Some(&self.deployment.weights_buffer), 0);
            enc.set_buffer(5, Some(norms), 0);
            enc.set_buffer(6, Some(&self.kernel_buffers.head_gates), 0);
            enc.set_buffer(7, Some(&ffn_scratch), 0);
            let li = layer as u32;
            enc.set_bytes(8, 4, &li as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(9, 4, &seq_position as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                metal::MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                metal::MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
        }
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        // The chained buffers are the taps — read every layer boundary back.
        Ok(boundaries[1..]
            .iter()
            .map(|b| {
                unsafe { std::slice::from_raw_parts(b.contents() as *const u16, HIDDEN) }
                    .iter()
                    .map(|&bits| f16::from_bits(bits).to_f32())
                    .collect()
            })
            .collect())
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
                let cand_logits = self
                    .megakernel
                    .read_slot_logits(&self.kernel_buffers, slot, 0);
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
    /// Transport B fused decode over the REAL kernels for ALL group sizes.
    /// Chains ceil(48 / group_size) dispatches — one of
    /// decode_layer_full_real / fused_full_{pair,triple,quad}_real per group —
    /// with per-layer fp16 KV sliced (by offset) out of one arena, and the
    /// group-boundary buffers chained (the blit-free taps at group
    /// granularity, exactly the STAGE0_TAPS_SPEC Transport B policy: audit
    /// passes use group_size 1, production fuses 2–4 and taps only group
    /// boundaries). Returns the group-end boundary states in order.
    ///
    /// This is the working replacement for the legacy `decode_fused` binding
    /// (which still targets the identity stubs' old ABI); the fusion-analyzer
    /// wiring migrates here once the graph descriptor carries real layers.
    pub fn decode_fused_real(
        &self,
        device: &metal::Device,
        hidden_in: &[f32],
        seq_position: u32,
        audit_max_seq: u32,
        group_size: u32,
    ) -> Result<Vec<Vec<f32>>, String> {
        if !(1..=4).contains(&group_size) {
            return Err(format!("group_size {group_size} not in 1..=4"));
        }
        // Uniform tiling of all 48 layers (remainder gets a smaller group).
        let mut groups = Vec::new();
        let mut layer = 0u32;
        while layer < LAYERS {
            let n = group_size.min(LAYERS - layer);
            groups.push((layer, n));
            layer += n;
        }
        self.fused_chain(device, hidden_in, seq_position, audit_max_seq, &groups)
    }

    /// Analyzer-driven fused decode: `kernel_fusion::analyze_graph` on the
    /// execution-graph descriptor decides the group sizes (1–4, same-kind
    /// consecutive decoder layers), and each group dispatches its ladder
    /// kernel via [`Self::fused_chain`]. This is the graph-descriptor path
    /// the legacy `decode_fused` stub binding is deprecated in favor of.
    pub fn decode_fused_graph(
        &self,
        device: &metal::Device,
        graph: &ExecutionGraphDescriptor,
        hidden_in: &[f32],
        seq_position: u32,
        audit_max_seq: u32,
    ) -> Result<Vec<Vec<f32>>, String> {
        let analysis = kernel_fusion::analyze_graph(graph);
        let mut groups: Vec<(u32, u32)> = Vec::new();
        let mut expected_next: Option<u32> = None;
        for g in &analysis {
            let node = graph
                .layers
                .get(g.start_layer)
                .ok_or_else(|| format!("fusion group start {} out of range", g.start_layer))?;
            if node.node_kind != NodeKind::DecoderLayer as u8 {
                continue; // non-decoder nodes (multimodal prefix etc.) are not layer groups
            }
            if !(1..=4).contains(&g.count) {
                return Err(format!(
                    "analyzer produced group of {} layers at {} — ladder kernels cover 1..=4",
                    g.count, g.start_layer
                ));
            }
            let layer_index = node.layer_index;
            if let Some(exp) = expected_next {
                if layer_index != exp {
                    return Err(format!(
                        "non-contiguous decoder coverage: expected layer {exp}, group starts at {layer_index}"
                    ));
                }
            }
            expected_next = Some(layer_index + g.count);
            groups.push((layer_index, g.count));
        }
        match expected_next {
            Some(n) if n == LAYERS => {}
            other => {
                return Err(format!(
                    "graph covers decoder layers up to {other:?}, expected exactly {LAYERS}"
                ))
            }
        }
        self.fused_chain(device, hidden_in, seq_position, audit_max_seq, &groups)
    }

    /// Shared fused-ladder chain: one dispatch per `(start_layer, count)`
    /// group, boundary buffers chained (the Transport B group-boundary taps),
    /// per-layer fp16 KV sliced by offset from one arena.
    fn fused_chain(
        &self,
        device: &metal::Device,
        hidden_in: &[f32],
        seq_position: u32,
        audit_max_seq: u32,
        groups: &[(u32, u32)],
    ) -> Result<Vec<Vec<f32>>, String> {
        use crate::ecs::compute_image::legacy_compute_image_runtime::megakernel::kernels::compile_layer_library;

        const HIDDEN: usize = 3840;
        if hidden_in.len() != HIDDEN {
            return Err(format!("hidden_in len {} != {HIDDEN}", hidden_in.len()));
        }
        if groups.is_empty() {
            return Err("empty fusion group list".into());
        }
        let norms = self
            .deployment
            .norms_buffer
            .as_ref()
            .ok_or("deployment missing norms buffer")?;
        let lib = compile_layer_library(device)?;
        let entry = |n: u32| match n {
            1 => "decode_layer_full_real",
            2 => "fused_full_pair_real",
            3 => "fused_full_triple_real",
            _ => "fused_full_quad_real",
        };
        // PSOs for every distinct group size in the plan.
        let mut psos: std::collections::HashMap<u32, metal::ComputePipelineState> =
            std::collections::HashMap::new();
        let mut sizes: Vec<u32> = groups.iter().map(|&(_, n)| n).collect();
        sizes.sort_unstable();
        sizes.dedup();
        for n in sizes {
            let f = lib
                .get_function(entry(n), None)
                .map_err(|e| format!("{}: {e}", entry(n)))?;
            psos.insert(
                n,
                device
                    .new_compute_pipeline_state_with_function(&f)
                    .map_err(|e| format!("PSO {}: {e}", entry(n)))?,
            );
        }

        let opts = metal::MTLResourceOptions::StorageModeShared;
        let in_buf = device.new_buffer((HIDDEN * 2) as u64, opts);
        unsafe {
            let dst = in_buf.contents() as *mut u16;
            for (i, &v) in hidden_in.iter().enumerate() {
                *dst.add(i) = f16::from_f32(v).to_bits();
            }
        }
        // Per-layer KV: one arena, sliced by offset per layer.
        let stride = (NUM_KV_HEADS * GLOBAL_HEAD_DIM) as u64;
        let per_layer_kv = audit_max_seq.max(1) as u64 * stride * 2;
        let kv_k = device.new_buffer(per_layer_kv * LAYERS as u64, opts);
        let kv_v = device.new_buffer(per_layer_kv * LAYERS as u64, opts);
        let ffn_scratch = device.new_buffer(2 * 15360 * 2, opts);

        let queue = device.new_command_queue();
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        let mut boundaries: Vec<metal::Buffer> = Vec::new();
        let mut current = in_buf;
        for &(layer, n) in groups {
            let pso = &psos[&n];
            let out = device.new_buffer((HIDDEN * 2) as u64, opts);
            enc.set_compute_pipeline_state(pso);
            enc.set_buffer(0, Some(&current), 0);
            enc.set_buffer(1, Some(&out), 0);
            enc.set_buffer(2, Some(&kv_k), (layer as u64) * per_layer_kv);
            enc.set_buffer(3, Some(&kv_v), (layer as u64) * per_layer_kv);
            enc.set_buffer(4, Some(&self.deployment.weights_buffer), 0);
            enc.set_buffer(5, Some(norms), 0);
            enc.set_buffer(6, Some(&self.kernel_buffers.head_gates), 0);
            enc.set_buffer(7, Some(&ffn_scratch), 0);
            enc.set_bytes(8, 4, &layer as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(9, 4, &seq_position as *const u32 as *const std::ffi::c_void);
            // Layers b/c/d: KV slices + indices at 10.. in steps of 3.
            for j in 1..n {
                let l = layer + j;
                let base = 10 + (j - 1) * 3;
                enc.set_buffer(base as u64, Some(&kv_k), (l as u64) * per_layer_kv);
                enc.set_buffer((base + 1) as u64, Some(&kv_v), (l as u64) * per_layer_kv);
                enc.set_bytes(
                    (base + 2) as u64,
                    4,
                    &l as *const u32 as *const std::ffi::c_void,
                );
            }
            enc.dispatch_thread_groups(
                metal::MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                metal::MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            boundaries.push(out.clone());
            current = out;
        }
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        Ok(boundaries
            .iter()
            .map(|b| {
                unsafe { std::slice::from_raw_parts(b.contents() as *const u16, HIDDEN) }
                    .iter()
                    .map(|&bits| f16::from_bits(bits).to_f32())
                    .collect()
            })
            .collect())
    }

    /// Run fused per-layer Metal decode driven by graph fusion analysis.
    ///
    /// NOTE: the group-size-1 AUDIT configuration now lives in
    /// [`Self::decode_audit_group1`], dispatching the REAL
    /// `decode_layer_full_real` body with chained boundary-tap buffers; the
    /// pair/triple/quad kernels this path dispatches are still identity
    /// stubs, so this function remains non-functional for real decode until
    /// those bodies are authored and this binding migrates to the new ABI.
    /// Groups of up to 4 consecutive same-kind decoder layers are dispatched
    /// as a single fused kernel, reducing command-buffer overhead and
    /// eliminating intermediate global buffer writes.
    #[deprecated(
        note = "binds the identity-stub kernels via the old ABI and computes nothing; \
                use decode_fused_graph (analyzer-driven) or decode_fused_real (fixed \
                group size) — both dispatch the real fused ladder"
    )]
    pub fn decode_fused(
        &self,
        device: &metal::Device,
        queue: &metal::CommandQueue,
        graph: &ExecutionGraphDescriptor,
        input_hidden: &metal::Buffer,
        kv_cache: &metal::Buffer,
        seq_position: u32,
    ) -> Result<metal::Buffer, String> {
        if self.deployment.is_nf4_tile640() {
            let next_node_index = self.validate_nf4_execution_graph(graph)?;
            let first_unsupported = graph.layers.get(next_node_index);
            return Err(
                match first_unsupported {
                    Some(node) if node.node_kind == NodeKind::DecoderLayer as u8 => format!(
                        "decode_fused: NF4Tile640 graph validated through multimodal prefix, but decoder layer {} still requires a dedicated NF4 decode kernel",
                        node.layer_index
                    ),
                    Some(node) if node.node_kind == NodeKind::DraftLayer as u8 => format!(
                        "decode_fused: NF4Tile640 graph validated through multimodal prefix, but draft decoder layer {} still requires a dedicated NF4 decode kernel",
                        node.layer_index
                    ),
                    Some(node) => format!(
                        "decode_fused: NF4Tile640 graph hit unsupported node kind {} at index {}",
                        node.node_kind, next_node_index
                    ),
                    None => "decode_fused: NF4Tile640 graph contains no decoder nodes; use the explicit graph prefix runner for multimodal execution".into(),
                }
            );
        }

        let fusion_groups = kernel_fusion::analyze_graph(graph);
        let hidden_dim = graph.layers[0].hidden_dim;
        let mut current = input_hidden.clone();

        let weights_buf = &self.deployment.weights_buffer;
        let scales_buf = &self.deployment.scales_buffer;

        for group in &fusion_groups {
            if group.count == 1 {
                // Single layer: use per-layer kernel
                let layer = &graph.layers[group.start_layer];
                let kernel_fn = if layer.attention_kind == 0 {
                    "decode_layer_swa"
                } else if layer.attention_kind == 1 {
                    "decode_layer_full"
                } else {
                    continue; // Skip non-attention nodes (multimodal ops handled separately)
                };

                let next = device.new_buffer(
                    (hidden_dim as u64) * std::mem::size_of::<half::f16>() as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );

                let cb = queue.new_command_buffer();
                let encoder = cb.new_compute_command_encoder();
                let pipeline = self.get_pso(kernel_fn)?;
                encoder.set_compute_pipeline_state(&pipeline);
                encoder.set_buffer(0, Some(&current), 0);
                encoder.set_buffer(1, Some(&next), 0);
                encoder.set_buffer(2, Some(kv_cache), 0);
                encoder.set_buffer(4, Some(weights_buf), 0);
                encoder.set_buffer(5, Some(scales_buf), 0);
                encoder.dispatch_thread_groups(
                    metal::MTLSize::new((hidden_dim as u64 + 255) / 256, 1, 1),
                    metal::MTLSize::new(256, 1, 1),
                );
                encoder.end_encoding();
                cb.commit();
                current = next;
            } else if group.count == 2 && group.attention_kind <= 1 {
                // Fused pair: use fused_swa_pair or fused_full_pair
                let layer_a = &graph.layers[group.start_layer];
                let layer_b = &graph.layers[group.start_layer + 1];
                let kernel_fn = if group.attention_kind == 0 {
                    "fused_swa_pair"
                } else {
                    "fused_full_pair"
                };

                let next = device.new_buffer(
                    (hidden_dim as u64) * std::mem::size_of::<half::f16>() as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );

                let cb = queue.new_command_buffer();
                let encoder = cb.new_compute_command_encoder();
                let pipeline = self.get_pso(kernel_fn)?;
                encoder.set_compute_pipeline_state(&pipeline);
                encoder.set_buffer(0, Some(&current), 0);
                encoder.set_buffer(1, Some(&next), 0);
                encoder.set_buffer(2, Some(kv_cache), 0);
                encoder.set_buffer(3, Some(kv_cache), 0);
                encoder.set_buffer(4, Some(weights_buf), 0);
                encoder.set_buffer(5, Some(scales_buf), 0);
                // layer A weight offset (for intra-kernel layer A weight base)
                encoder.set_bytes(
                    6,
                    4,
                    &(layer_a.weight_offset as u32) as *const u32 as *const _,
                );
                // layer B weight offset
                encoder.set_bytes(
                    7,
                    4,
                    &(layer_b.weight_offset as u32) as *const u32 as *const _,
                );
                // layer A scale offset
                encoder.set_bytes(
                    8,
                    4,
                    &(layer_a.scale_offset as u32) as *const u32 as *const _,
                );
                // layer B scale offset
                encoder.set_bytes(
                    9,
                    4,
                    &(layer_b.scale_offset as u32) as *const u32 as *const _,
                );
                // layer indices
                encoder.set_bytes(10, 4, &layer_a.layer_index as *const u32 as *const _);
                encoder.set_bytes(11, 4, &layer_b.layer_index as *const u32 as *const _);
                // current sequence position (for KV cache slot addressing)
                encoder.set_bytes(12, 4, &seq_position as *const u32 as *const _);
                // hidden dimension
                encoder.set_bytes(13, 4, &hidden_dim as *const u32 as *const _);
                encoder.dispatch_thread_groups(
                    metal::MTLSize::new((hidden_dim as u64 + 255) / 256, 1, 1),
                    metal::MTLSize::new(256, 1, 1),
                );
                encoder.end_encoding();
                cb.commit();
                current = next;
            }
            // Groups of 3-4: recurse with pair + single dispatch
        }

        Ok(current)
    }

    pub fn decode_speculative(
        &mut self,
        token_id: u32,
        num_draft: u32,
    ) -> Result<Vec<u32>, String> {
        let slot = 0usize;
        let seq_pos = self.slot_seq_pos[slot];

        // Cap draft candidates to buffer capacity.
        let num_candidates = num_draft.min(MAX_DRAFT_CANDIDATES);
        if num_candidates == 0 {
            return self.decode_with_mtp(token_id, 0);
        }

        // ── Phase 1: Run draft model forward pass ──
        self.megakernel
            .submit_draft(&self.kernel_buffers, token_id, seq_pos, num_candidates);
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
                    std::ptr::copy_nonoverlapping(k_src, pass_scratch_k, compacted_per_layer_bytes);

                    let v_src = v_out_arena.base_ptr() as *const u8;
                    std::ptr::copy_nonoverlapping(v_src, pass_scratch_v, compacted_per_layer_bytes);
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
                    std::ptr::copy_nonoverlapping(k_src, pass_scratch_k, compacted_per_layer_bytes);

                    let v_src = v_out_arena.base_ptr() as *const u8;
                    std::ptr::copy_nonoverlapping(v_src, pass_scratch_v, compacted_per_layer_bytes);
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
    pub fn prefill_from_ane(&mut self) {}
}

/// Stage 0 activation taps for one decoded token (STAGE0_TAPS_SPEC.md slot
/// map). Holds the raw f16 bits; accessors convert to f32. The hidden width
/// is derived from the buffer length, so the view cannot desync from the
/// kernel's HIDDEN_DIM.
pub struct LayerTaps {
    raw: Vec<u16>,
    hidden: usize,
}

impl LayerTaps {
    fn from_raw(raw: Vec<u16>) -> Result<Self, String> {
        let slots = 2 * LAYERS as usize + 2;
        if raw.is_empty() || raw.len() % slots != 0 {
            return Err(format!(
                "tap buffer len {} is not a multiple of {slots} slots",
                raw.len()
            ));
        }
        let hidden = raw.len() / slots;
        Ok(LayerTaps { raw, hidden })
    }

    fn slot(&self, s: usize) -> Vec<f32> {
        self.raw[s * self.hidden..(s + 1) * self.hidden]
            .iter()
            .map(|&b| f16::from_bits(b).to_f32())
            .collect()
    }

    pub fn hidden_dim(&self) -> usize {
        self.hidden
    }
    /// Slot 0: embedding output, before layer 0.
    pub fn post_embed(&self) -> Vec<f32> {
        self.slot(0)
    }
    /// Slot 2k+1: layer `k` after the attention residual.
    pub fn post_attention(&self, layer: usize) -> Vec<f32> {
        self.slot(2 * layer + 1)
    }
    /// Slot 2k+2: layer `k` after the FFN residual — the layer boundary.
    pub fn post_layer(&self, layer: usize) -> Vec<f32> {
        self.slot(2 * layer + 2)
    }
    /// Final pre-logits hidden (post final norm).
    pub fn final_hidden(&self) -> Vec<f32> {
        self.slot(2 * LAYERS as usize + 1)
    }
}

// ── Stage 0 tap tests (Mac; real cimage; env-gated) ─────────────────────────
// Taps are a process-env, kernel-compile-time property, so these tests mutate
// TRIBUNUS_TAPS around Orchestrator construction and MUST run serially:
//   TRIBUNUS_TEST_CIMAGE=/path/to/model.cimage \
//   cargo test --features prism-backend stage0_taps -- --test-threads=1
#[cfg(all(test, feature = "prism-backend"))]
mod stage0_tap_tests {
    use super::*;

    fn cimage() -> Option<String> {
        std::env::var("TRIBUNUS_TEST_CIMAGE").ok()
    }

    fn with_taps_env<T>(on: bool, f: impl FnOnce() -> T) -> T {
        if on {
            std::env::set_var("TRIBUNUS_TAPS", "1");
        } else {
            std::env::remove_var("TRIBUNUS_TAPS");
        }
        let out = f();
        std::env::remove_var("TRIBUNUS_TAPS");
        out
    }

    fn finite_nonzero(v: &[f32]) -> bool {
        v.iter().all(|x| x.is_finite()) && v.iter().any(|x| *x != 0.0)
    }

    /// Tap mode is an explicit construction parameter, not an ambient env
    /// convention: with TRIBUNUS_TAPS **unset**, `from_cimage_with_mode(..,
    /// TappedAudit)` must produce a fully tapped orchestrator, and an
    /// explicitly `Untapped` one must refuse the taps API even if the env
    /// var IS set (the mode wins over the environment).
    #[test]
    fn tap_mode_explicit_construction_beats_env() {
        let Some(path) = cimage() else { return };
        // Explicit TappedAudit, env unset → taps work.
        let (argmax_a, taps_ok) = with_taps_env(false, || {
            let mut orch =
                Orchestrator::from_cimage_with_mode(&path, 1, false, TapMode::TappedAudit)
                    .expect("tapped orchestrator");
            assert_eq!(orch.tap_mode, TapMode::TappedAudit);
            let (next, _logits, taps) = orch
                .decode_token_logits_with_taps(7)
                .expect("explicit mode must not need TRIBUNUS_TAPS");
            (next, finite_nonzero(&taps.post_embed()))
        });
        assert!(taps_ok, "post-embed tap must be populated");
        // Explicit Untapped, env SET → the taps API refuses (mode wins).
        let argmax_b = with_taps_env(true, || {
            let mut orch = Orchestrator::from_cimage_with_mode(&path, 1, false, TapMode::Untapped)
                .expect("untapped orchestrator");
            assert_eq!(orch.tap_mode, TapMode::Untapped);
            assert!(
                orch.decode_token_logits_with_taps(7).is_err(),
                "Untapped orchestrator must refuse the taps API even with TRIBUNUS_TAPS=1"
            );
            let (next, _logits) = orch.decode_token_logits(7).expect("plain decode");
            next
        });
        // Same token, same artifact → same greedy pick on both modes.
        assert_eq!(argmax_a, argmax_b, "tap mode must not change decode output");
    }

    /// (a) Self-consistency: every tap slot is populated and finite; the
    /// residual stream evolves across layers; the final norm visibly
    /// transforms the last layer boundary.
    #[test]
    fn stage0_taps_self_consistency() {
        let Some(path) = cimage() else {
            eprintln!("skipping: set TRIBUNUS_TEST_CIMAGE (and run --test-threads=1)");
            return;
        };
        with_taps_env(true, || {
            let mut orch = Orchestrator::from_cimage(&path, 1, false).expect("load");
            let (_tok, logits, taps) = orch
                .decode_token_logits_with_taps(7)
                .expect("tapped decode");
            assert!(!logits.is_empty());
            assert!(finite_nonzero(&taps.post_embed()), "embed tap empty");
            for k in [0usize, 5, 24, LAYERS as usize - 1] {
                assert!(finite_nonzero(&taps.post_attention(k)), "attn tap {k}");
                assert!(finite_nonzero(&taps.post_layer(k)), "layer tap {k}");
            }
            assert!(finite_nonzero(&taps.final_hidden()));
            // The stream must actually evolve (identity layers would be a bug).
            assert_ne!(taps.post_embed(), taps.post_layer(0));
            // The final norm transforms the last boundary state.
            assert_ne!(taps.post_layer(LAYERS as usize - 1), taps.final_hidden());
        });
    }

    /// (b) Determinism: two fresh tapped orchestrators produce bitwise-equal
    /// tap buffers for the same token stream.
    #[test]
    fn stage0_taps_deterministic() {
        let Some(path) = cimage() else {
            eprintln!("skipping: set TRIBUNUS_TEST_CIMAGE (and run --test-threads=1)");
            return;
        };
        with_taps_env(true, || {
            let run = || {
                let mut orch = Orchestrator::from_cimage(&path, 1, false).expect("load");
                let mut all = Vec::new();
                for t in [7u32, 42, 99] {
                    let (_a, _l, taps) = orch.decode_token_logits_with_taps(t).expect("decode");
                    all.push(taps.raw.clone());
                }
                all
            };
            assert_eq!(run(), run(), "taps not bitwise deterministic");
        });
    }

    /// Transport B gate: the REAL decode_layer_full_real body must reproduce
    /// the megakernel's layer-0 boundary (Transport A taps) at position 0 —
    /// where the fp16-KV fused kernel and the megakernel are numerically
    /// identical by construction (the megakernel also attends over fresh fp16
    /// scratch before ternary-packing).
    #[test]
    fn transport_b_layer0_matches_taps() {
        let Some(path) = cimage() else {
            eprintln!("skipping: set TRIBUNUS_TEST_CIMAGE (and run --test-threads=1)");
            return;
        };
        with_taps_env(true, || {
            use crate::ecs::compute_image::legacy_compute_image_runtime::megakernel::kernels::compile_layer_library;
            use metal::MTLResourceOptions;

            let mut orch = Orchestrator::from_cimage(&path, 1, false).expect("load");
            // Position 0: first decode on a fresh orchestrator.
            let (_t, _l, taps) = orch
                .decode_token_logits_with_taps(7)
                .expect("tapped decode");
            let input = taps.post_embed();
            let expect = taps.post_layer(0);
            let hidden = taps.hidden_dim();

            let device = metal::Device::system_default().expect("metal device");
            let lib = compile_layer_library(&device).expect("compile decode_per_layer");
            let f = lib
                .get_function("decode_layer_full_real", None)
                .expect("entry point");
            let pso = device
                .new_compute_pipeline_state_with_function(&f)
                .expect("pso");

            let half_bits = |v: &[f32]| -> Vec<u8> {
                v.iter()
                    .flat_map(|&x| half::f16::from_f32(x).to_bits().to_le_bytes())
                    .collect()
            };
            let mk = |bytes: &[u8]| {
                device.new_buffer_with_data(
                    bytes.as_ptr() as *const std::ffi::c_void,
                    bytes.len() as u64,
                    MTLResourceOptions::StorageModeShared,
                )
            };
            let in_buf = mk(&half_bits(&input));
            let out_buf =
                device.new_buffer((hidden * 2) as u64, MTLResourceOptions::StorageModeShared);
            let stride = (NUM_KV_HEADS * GLOBAL_HEAD_DIM) as u64;
            let kv_len = 4 * stride * 2; // tiny max_seq for pos 0
            let kv_k = device.new_buffer(kv_len, MTLResourceOptions::StorageModeShared);
            let kv_v = device.new_buffer(kv_len, MTLResourceOptions::StorageModeShared);
            let ffn = device.new_buffer(2 * 15360 * 2, MTLResourceOptions::StorageModeShared);
            let norms = self_norms(&orch);
            let queue = device.new_command_queue();
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pso);
            enc.set_buffer(0, Some(&in_buf), 0);
            enc.set_buffer(1, Some(&out_buf), 0);
            enc.set_buffer(2, Some(&kv_k), 0);
            enc.set_buffer(3, Some(&kv_v), 0);
            enc.set_buffer(4, Some(&orch.deployment.weights_buffer), 0);
            enc.set_buffer(5, Some(norms), 0);
            enc.set_buffer(6, Some(&orch.kernel_buffers.head_gates), 0);
            enc.set_buffer(7, Some(&ffn), 0);
            let layer0 = 0u32;
            let pos0 = 0u32;
            enc.set_bytes(8, 4, &layer0 as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(9, 4, &pos0 as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                metal::MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                metal::MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();

            let got: Vec<f32> =
                unsafe { std::slice::from_raw_parts(out_buf.contents() as *const u16, hidden) }
                    .iter()
                    .map(|&b| f16::from_bits(b).to_f32())
                    .collect();
            let (mut se, mut den) = (0.0f64, 0.0f64);
            for (a, g) in got.iter().zip(&expect) {
                se += (*a as f64 - *g as f64).powi(2);
                den += (*g as f64).powi(2);
            }
            let rel = (se / den.max(1e-30)).sqrt();
            assert!(
                rel < 1e-3,
                "Transport B layer 0 vs Transport A tap: rel-L2 {rel:.3e} (expected fp16-tight at pos 0)"
            );
            eprintln!("[transport-b] layer0 parity rel-L2 = {rel:.3e}  PASS");
        });
    }

    /// Transport B end-to-end (all 48 layers, group size 1): chain the real
    /// fused layer body from the post-embed tap and compare EVERY boundary
    /// buffer (the blit-free taps) against Transport A's post_layer taps.
    /// Emits the per-layer drift curve; gates on the final boundary.
    /// Full logits parity additionally needs the embed/final-norm/logits
    /// stages migrated — the final boundary here is the last fused-path
    /// state before those megakernel-only stages.
    #[test]
    fn transport_b_full_depth_matches_taps() {
        let Some(path) = cimage() else {
            eprintln!("skipping: set TRIBUNUS_TEST_CIMAGE (and run --test-threads=1)");
            return;
        };
        with_taps_env(true, || {
            let mut orch = Orchestrator::from_cimage(&path, 1, false).expect("load");
            let (_t, _l, taps) = orch
                .decode_token_logits_with_taps(7)
                .expect("tapped decode");
            let device = metal::Device::system_default().expect("metal device");
            let boundaries = orch
                .decode_audit_group1(&device, &taps.post_embed(), 0, 4)
                .expect("group-1 audit chain");
            assert_eq!(boundaries.len(), LAYERS as usize);
            let rel = |a: &[f32], g: &[f32]| -> f64 {
                let (mut se, mut den) = (0.0f64, 0.0f64);
                for (x, y) in a.iter().zip(g) {
                    se += (*x as f64 - *y as f64).powi(2);
                    den += (*y as f64).powi(2);
                }
                (se / den.max(1e-30)).sqrt()
            };
            let mut worst = (0usize, 0.0f64);
            for k in 0..LAYERS as usize {
                let d = rel(&boundaries[k], &taps.post_layer(k));
                if d > worst.1 {
                    worst = (k, d);
                }
                if k % 8 == 0 || k == LAYERS as usize - 1 {
                    eprintln!("[transport-b] layer {k:2} drift rel-L2 = {d:.3e}");
                }
            }
            let final_drift = rel(
                &boundaries[LAYERS as usize - 1],
                &taps.post_layer(LAYERS as usize - 1),
            );
            eprintln!(
                "[transport-b] worst layer {} ({:.3e}); final boundary {:.3e}",
                worst.0, worst.1, final_drift
            );
            assert!(
                final_drift < 5e-3,
                "48-layer chained drift {final_drift:.3e} exceeds the 5e-3 gate (worst at layer {})",
                worst.0
            );
        });
    }

    /// Fused-pair gate: fused_full_pair_real(k, k+1) must equal two chained
    /// decode_layer_full_real dispatches — bitwise-expected, because BOTH
    /// paths hold the intermediate boundary in half precision (threadgroup
    /// h_buf vs device round-trip both quantize to f16).
    #[test]
    fn transport_b_pair_matches_two_singles() {
        let Some(path) = cimage() else {
            eprintln!("skipping: set TRIBUNUS_TEST_CIMAGE (and run --test-threads=1)");
            return;
        };
        with_taps_env(true, || {
            use crate::ecs::compute_image::legacy_compute_image_runtime::megakernel::kernels::compile_layer_library;
            use metal::MTLResourceOptions;

            let mut orch = Orchestrator::from_cimage(&path, 1, false).expect("load");
            let (_t, _l, taps) = orch
                .decode_token_logits_with_taps(7)
                .expect("tapped decode");
            let input = taps.post_embed();
            let hidden = taps.hidden_dim();

            // Reference: layers 0 then 1 via the group-1 audit chain.
            let device = metal::Device::system_default().expect("metal device");
            let singles = orch
                .decode_audit_group1(&device, &input, 0, 4)
                .expect("group-1 chain");
            let expect = &singles[1]; // boundary after layer 1

            // Fused pair (0, 1) in one dispatch.
            let lib = compile_layer_library(&device).expect("compile");
            let f = lib
                .get_function("fused_full_pair_real", None)
                .expect("entry");
            let pso = device
                .new_compute_pipeline_state_with_function(&f)
                .expect("pso");
            let opts = MTLResourceOptions::StorageModeShared;
            let in_buf = device.new_buffer((hidden * 2) as u64, opts);
            unsafe {
                let dst = in_buf.contents() as *mut u16;
                for (i, &v) in input.iter().enumerate() {
                    *dst.add(i) = f16::from_f32(v).to_bits();
                }
            }
            let out_buf = device.new_buffer((hidden * 2) as u64, opts);
            let stride = (NUM_KV_HEADS * GLOBAL_HEAD_DIM) as u64;
            let kv = |_| device.new_buffer(4 * stride * 2, opts);
            let (ka, va, kb, vb) = (kv(0), kv(1), kv(2), kv(3));
            let ffn = device.new_buffer(2 * 15360 * 2, opts);
            let norms = self_norms(&orch);
            let queue = device.new_command_queue();
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pso);
            enc.set_buffer(0, Some(&in_buf), 0);
            enc.set_buffer(1, Some(&out_buf), 0);
            enc.set_buffer(2, Some(&ka), 0);
            enc.set_buffer(3, Some(&va), 0);
            enc.set_buffer(4, Some(&orch.deployment.weights_buffer), 0);
            enc.set_buffer(5, Some(norms), 0);
            enc.set_buffer(6, Some(&orch.kernel_buffers.head_gates), 0);
            enc.set_buffer(7, Some(&ffn), 0);
            let (la, pos, lb) = (0u32, 0u32, 1u32);
            enc.set_bytes(8, 4, &la as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(9, 4, &pos as *const u32 as *const std::ffi::c_void);
            enc.set_buffer(10, Some(&kb), 0);
            enc.set_buffer(11, Some(&vb), 0);
            enc.set_bytes(12, 4, &lb as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                metal::MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                metal::MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();

            let got: Vec<f32> =
                unsafe { std::slice::from_raw_parts(out_buf.contents() as *const u16, hidden) }
                    .iter()
                    .map(|&b| f16::from_bits(b).to_f32())
                    .collect();
            let max_abs = got
                .iter()
                .zip(expect)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_abs == 0.0,
                "fused pair vs two singles: expected bitwise-identical (both quantize the \
                 intermediate to f16), max |Δ| = {max_abs}"
            );
            eprintln!("[transport-b] pair == two singles (bitwise)  PASS");
        });
    }

    /// Fusion-ladder gate: for every production group size, the fused chain's
    /// group-end boundaries must be BITWISE-equal to the group-1 chain's
    /// boundaries at the same layers (both paths quantize every compared
    /// boundary to f16; the fused path's INTERNAL boundaries stay in
    /// threadgroup memory and are intentionally not compared — that is what
    /// fusion elides).
    #[test]
    fn transport_b_groups_match_singles() {
        let Some(path) = cimage() else {
            eprintln!("skipping: set TRIBUNUS_TEST_CIMAGE (and run --test-threads=1)");
            return;
        };
        with_taps_env(true, || {
            let mut orch = Orchestrator::from_cimage(&path, 1, false).expect("load");
            let (_t, _l, taps) = orch
                .decode_token_logits_with_taps(7)
                .expect("tapped decode");
            let input = taps.post_embed();
            let device = metal::Device::system_default().expect("metal device");
            let singles = orch
                .decode_audit_group1(&device, &input, 0, 4)
                .expect("group-1 chain");
            for gs in [2u32, 3, 4] {
                let groups = orch
                    .decode_fused_real(&device, &input, 0, 4, gs)
                    .expect("fused chain");
                let mut layer = 0u32;
                for (gi, got) in groups.iter().enumerate() {
                    let n = gs.min(LAYERS - layer);
                    let end_layer = (layer + n - 1) as usize;
                    let expect = &singles[end_layer];
                    let max_abs = got
                        .iter()
                        .zip(expect)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max);
                    assert!(
                        max_abs == 0.0,
                        "group_size {gs}, group {gi} (end layer {end_layer}): expected \
                         bitwise-identical, max |Δ| = {max_abs}"
                    );
                    layer += n;
                }
                eprintln!("[transport-b] group_size {gs}: all group boundaries bitwise  PASS");
            }
        });
    }

    fn self_norms(orch: &Orchestrator) -> &metal::Buffer {
        orch.deployment
            .norms_buffer
            .as_ref()
            .expect("deployment carries a norms buffer")
    }

    /// (c) Taps-off identity: without the define the kernel is compiled from
    /// byte-identical preprocessed source — logits must match the tapped
    /// build (bitwise expected; asserted via argmax + tight max-abs so a
    /// compiler-scheduling delta is visible but diagnosable), and the taps
    /// API must refuse rather than serve a stale buffer.
    #[test]
    fn stage0_taps_off_identity() {
        let Some(path) = cimage() else {
            eprintln!("skipping: set TRIBUNUS_TEST_CIMAGE (and run --test-threads=1)");
            return;
        };
        let tapped = with_taps_env(true, || {
            let mut orch = Orchestrator::from_cimage(&path, 1, false).expect("load");
            orch.decode_token_logits(7).expect("tapped decode").1
        });
        let plain = with_taps_env(false, || {
            let mut orch = Orchestrator::from_cimage(&path, 1, false).expect("load");
            let out = orch.decode_token_logits(7).expect("plain decode").1;
            assert!(
                orch.decode_token_logits_with_taps(7).is_err(),
                "taps API must refuse when the kernel was compiled without taps"
            );
            out
        });
        assert_eq!(
            sample_argmax_f32(&tapped),
            sample_argmax_f32(&plain),
            "argmax must be identical between tapped and untapped builds"
        );
        let max_abs = tapped
            .iter()
            .zip(&plain)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs == 0.0,
            "expected bitwise-identical logits, max |Δ| = {max_abs}"
        );
    }
}
