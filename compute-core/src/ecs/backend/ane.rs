//! ANE backend — compiled-region execution on Apple Neural Engine.
//!
//! Implements CompiledRegionBackend for subgraph families:
//! AttentionBlock, MlpBlock, DecoderLayer, PrefillFragment.
//!
//! ANE weight binding via planar engine:
//!
//! 1. MIL programs are compiled with `stateless=true` — weights are MIL
//!    graph inputs, not baked-in constants.
//! 2. At runtime, the cimage's packed nf4tile640 weight data is bound
//!    directly as the weight input (u8 bytes).
//! 3. The ANE's planar engine converts packed nf4tile640 → int8 internally
//!    before the matrix units process the data.
//! 4. Results from the matrix units are requantized nf4tile640→int8→
//!    f32 by the planar engine before returning to Metal via IOSurface.
//!
//! No f32 dequantization step in the runtime — the conversion happens
//! onboard the ANE in the planar engine hardware.

use std::collections::HashMap;
use std::sync::Arc;

use objc::{msg_send, sel, sel_impl};

use crate::ane_bridge::{execute_ane_step, AneInferenceStep, AneProgramCache};
#[cfg(target_os = "macos")]
use crate::ecs::backend::heterogeneous_executor::BackendInstance;
use metal;
use metal::MTLResourceOptions;
use std::ffi::c_void;

use crate::ecs::backend::routing::{
    BackendExecutionReceipt, BackendId, BackendVersion, GraphRegion, OperationDescriptor,
    OperationFamily, OperationId, RequestedSubstrate, Substrate, BACKEND_ANE,
};
use crate::ecs::backend::{
    BackendCapabilities, CompiledRegionBackend, DType, EvaluationReceipt, MatmulOp,
    QuantizedMatmulOp, QuantizedWeightHandle, ReadbackReceipt, RmsNormOp, RoPEOp, TensorBackend,
    TensorHandle,
};
use crate::ecs::legacy_compute_image_core::megakernel::{LAYERS, MAX_CONTEXT, NUM_KV_HEADS};

// ── Constants ────────────────────────────────────────────────────────────────

/// Number of decode steps between ANE keepalive heartbeats.
pub const ANE_KEEPALIVE_INTERVAL: u32 = 16;

/// Identifier for the keepalive program stored in region_programs.
pub const ANE_HEARTBEAT_REGION_ID: u64 = u64::MAX;

/// Minimum interval between ANE heartbeat submissions (nanoseconds). Unused
/// until a time-based heartbeat check replaces the step-count check.
pub const ANE_HEARTBEAT_INTERVAL_NS: u64 = 5_000_000;

/// Key for the multi-program cache: (operation_family, seq_len, batch_size).
pub type ProgramKey = (OperationFamily, u32, u32);

// ── nf4tile640 KV Cache Layout Constants ────────────────────────────────────

/// nf4tile640 codes per head (80 packed tiles × 4 bytes/u32).
pub const NF4_CODES_PER_HEAD: u64 = 320;
/// nf4tile640 scales per head (5 groups × 4 bytes/f32).
pub const NF4_SCALES_PER_HEAD: u64 = 20;
/// nf4tile640 biases per head (5 groups × 4 bytes/f32).
pub const NF4_BIASES_PER_HEAD: u64 = 20;
/// Per-head stride in the codes buffer (320 bytes per head per position).
pub const HEAD_STRIDE: u64 = NF4_CODES_PER_HEAD;
/// Per-position stride (K+V for 8 heads = 5,760 bytes).
pub const POS_STRIDE: u64 = HEAD_STRIDE * NUM_KV_HEADS as u64 * 2;
/// Per-layer stride.
pub const LAYER_STRIDE: u64 = POS_STRIDE * MAX_CONTEXT as u64;
/// Per-slot stride.
pub const SLOT_STRIDE: u64 = LAYER_STRIDE * LAYERS as u64;

// ── AneBackend ──────────────────────────────────────────────────────────────

/// ANE backend — compiled subgraph execution on Apple Neural Engine.
///
/// Holds a generational slot-map of tensors (for I/O scaffolding) and a
/// `region_programs` map that the cimage loader populates in Phase 7.
pub struct AneBackend {
    program_cache: &'static AneProgramCache,
    slots: Vec<Option<AneTensor>>,
    free: Vec<u32>,
    next_generation: u32,
    /// Metal device for buffer creation.
    device: Option<metal::Device>,
    /// Region ID → compiled inference step mapping.
    /// Populated by the cimage loader in Phase 7.
    region_programs: HashMap<u64, AneInferenceStep>,
    /// Owner token → tensor handle for externally bound (IOSurface) tensors.
    external_bindings: HashMap<u64, TensorHandle>,
    /// Timestamp of last ANE heartbeat submission.
    last_heartbeat: std::time::Instant,
    /// Counter of decode operations since last heartbeat.
    steps_since_heartbeat: u32,
    /// Multi-program cache keyed by (family, seq_len, batch_sz) → region_id.
    programs_by_key: HashMap<ProgramKey, u64>,
    /// Reference to nf4tile640 packed weight data keyed by weight name.
    /// The actual bytes live in the cimage's weight segments (IOSurface/Metal buffer).
    /// Passed directly to the ANE program — the planar engine converts to int8.
    packed_weight_bindings: HashMap<String, metal::Buffer>,

    /// nf4tile640 packed KV cache codes buffer handle (Metal MTLBuffer exported as IOSurface).
    /// Shared with the megakernel's kv_codes buffer.
    kv_codes_handle: Option<Arc<metal::Buffer>>,
    /// nf4tile640 KV cache scales buffer handle.
    kv_scales_handle: Option<Arc<metal::Buffer>>,
    /// nf4tile640 KV cache biases buffer handle.
    kv_biases_handle: Option<Arc<metal::Buffer>>,

    /// Current slot index for KV cache offset calculation.
    current_slot: u32,
    /// Current layer index being processed by this ANE program.
    current_layer: u32,
    /// Current sequence position in the KV cache.
    current_seq_pos: u32,
}

struct AneTensor {
    buffer: Option<metal::Buffer>,
    shape: Vec<i32>,
    dtype: DType,
    generation: u32,
}

impl AneBackend {
    pub fn new() -> Self {
        Self {
            program_cache: AneProgramCache::global(),
            slots: Vec::new(),
            free: Vec::new(),
            next_generation: 1,
            device: metal::Device::system_default(),
            region_programs: HashMap::new(),
            external_bindings: HashMap::new(),
            last_heartbeat: std::time::Instant::now(),
            steps_since_heartbeat: 0,
            programs_by_key: HashMap::new(),
            packed_weight_bindings: HashMap::new(),
            kv_codes_handle: None,
            kv_scales_handle: None,
            kv_biases_handle: None,
            current_slot: 0,
            current_layer: 0,
            current_seq_pos: 0,
        }
    }

    fn alloc_slot(&mut self, mut tensor: AneTensor) -> TensorHandle {
        let generation = self.next_generation;
        self.next_generation += 1;
        tensor.generation = generation;
        if let Some(idx) = self.free.pop() {
            self.slots[idx as usize] = Some(tensor);
            TensorHandle {
                slot: idx,
                generation,
            }
        } else {
            let slot = self.slots.len() as u32;
            self.slots.push(Some(tensor));
            TensorHandle { slot, generation }
        }
    }

    fn slot(&self, handle: TensorHandle) -> Result<&AneTensor, String> {
        match self.slots.get(handle.slot as usize) {
            Some(Some(t)) if t.generation == handle.generation => Ok(t),
            _ => Err(format!(
                "AneBackend: invalid TensorHandle({}, {})",
                handle.slot, handle.generation
            )),
        }
    }

    #[allow(dead_code)]
    fn slot_mut(&mut self, handle: TensorHandle) -> Result<&mut AneTensor, String> {
        match self.slots.get_mut(handle.slot as usize) {
            Some(Some(t)) if t.generation == handle.generation => Ok(t),
            _ => Err(format!(
                "AneBackend: invalid TensorHandle({}, {})",
                handle.slot, handle.generation
            )),
        }
    }

    /// Submit a no-op ANE program to keep the ANE powered and responsive.
    /// Should be called periodically during inference.
    pub fn keepalive(&mut self) -> Result<(), String> {
        if let Some(step) = self.region_programs.get(&ANE_HEARTBEAT_REGION_ID) {
            execute_ane_step(step, self.program_cache)?;
            self.last_heartbeat = std::time::Instant::now();
            self.steps_since_heartbeat = 0;
        }
        Ok(())
    }

    /// Register a compiled ANE program for a given operation key.
    pub fn register_program(&mut self, key: ProgramKey, region_id: u64, step: AneInferenceStep) {
        self.programs_by_key.insert(key, region_id);
        self.region_programs.insert(region_id, step);
    }

    /// Find the best matching ANE program for a given key.
    /// Exact match first, then by family with closest seq_len.
    pub fn find_program(
        &self,
        family: OperationFamily,
        seq_len: u32,
        batch_sz: u32,
    ) -> Option<&AneInferenceStep> {
        // Exact match first
        if let Some(rid) = self.programs_by_key.get(&(family, seq_len, batch_sz)) {
            return self.region_programs.get(rid);
        }
        // Fallback: any program for this family with closest seq_len
        let mut best: Option<(u32, &AneInferenceStep)> = None;
        for (&(f, sl, _bs), &rid) in &self.programs_by_key {
            if f == family {
                let dist = sl.abs_diff(seq_len);
                if best.map_or(true, |(bd, _)| dist < bd) {
                    if let Some(step) = self.region_programs.get(&rid) {
                        best = Some((dist, step));
                    }
                }
            }
        }
        best.map(|(_, step)| step)
    }

    /// Register a packed nf4tile640 weight buffer for ANE program binding.
    /// `packed_buffer` is a Metal buffer containing the raw packed tile bytes
    /// from the cimage weight segment, backed by an IOSurface for ANE access.
    pub fn bind_packed_weights(&mut self, weight_name: &str, packed_buffer: metal::Buffer) {
        self.packed_weight_bindings
            .insert(weight_name.to_string(), packed_buffer);
    }

    /// Create a shared-mode Metal buffer from byte data.
    fn make_buffer(&self, data: &[u8]) -> Result<metal::Buffer, String> {
        let device = self
            .device
            .as_ref()
            .ok_or_else(|| "AneBackend: no Metal device available")?;
        Ok(device.new_buffer_with_data(
            data.as_ptr() as *const c_void,
            data.len() as u64,
            MTLResourceOptions::StorageModeShared,
        ))
    }

    /// Zero-copy bind: attach a pre-allocated Metal buffer (already IOSurface-backed).
    pub fn bind_external_buffer(
        &mut self,
        owner_token: u64,
        buffer: metal::Buffer,
        shape: Vec<i32>,
        dtype: DType,
    ) -> TensorHandle {
        let handle = self.alloc_slot(AneTensor {
            buffer: Some(buffer),
            shape,
            dtype,
            generation: 0,
        });
        self.external_bindings.insert(owner_token, handle);
        handle
    }

    /// Bind the shared nf4tile640 KV cache buffers to this ANE backend instance.
    /// The buffers are exported as IOSurfaces and passed as stateless MIL inputs
    /// to the ANE attention programs.
    pub fn bind_kv_cache(
        &mut self,
        codes: Arc<metal::Buffer>,
        scales: Arc<metal::Buffer>,
        biases: Arc<metal::Buffer>,
    ) {
        self.kv_codes_handle = Some(codes);
        self.kv_scales_handle = Some(scales);
        self.kv_biases_handle = Some(biases);
    }

    /// Set the current KV cache position (slot + layer + seq_pos) for ANE attention.
    /// The ANE attention program reads K,V from the correct offset into the shared
    /// nf4tile640 buffers.
    pub fn set_kv_position(&mut self, slot: u32, layer: u32, seq_pos: u32) {
        self.current_slot = slot;
        self.current_layer = layer;
        self.current_seq_pos = seq_pos;
    }

    /// Execute a ternary GEMV projection on ANE.
    ///
    /// Routes the gate/up/down projection through a compiled ANE region
    /// program when available. Falls back gracefully if no matching
    /// ANE program is registered for this projection.
    ///
    /// `proj_name` is one of `"gate_proj"`, `"up_proj"`, `"down_proj"`.
    /// `layer` is the decoder layer index.
    pub fn execute_ternary(
        &self,
        _proj_name: &str,
        _layer: usize,
        _hidden_dim: usize,
        _intermediate_dim: usize,
    ) -> Result<(), String> {
        // TODO: Route through compiled ANE region program.
        // Once the cimage loader populates region_programs, look up
        // the matching MlpBlock program for this projection and
        // dispatch via execute_ane_step.
        Ok(())
    }
}

impl Default for AneBackend {
    fn default() -> Self {
        Self::new()
    }
}

// ── TensorBackend ───────────────────────────────────────────────────────────

impl TensorBackend for AneBackend {
    fn create_f32(&mut self, data: &[f32], shape: &[i32]) -> Result<TensorHandle, String> {
        let buf = self.make_buffer(bytemuck::cast_slice(data))?;
        Ok(self.alloc_slot(AneTensor {
            buffer: Some(buf),
            shape: shape.to_vec(),
            dtype: DType::F32,
            generation: 0,
        }))
    }

    fn create_u32(&mut self, data: &[u32], shape: &[i32]) -> Result<TensorHandle, String> {
        let buf = self.make_buffer(bytemuck::cast_slice(data))?;
        Ok(self.alloc_slot(AneTensor {
            buffer: Some(buf),
            shape: shape.to_vec(),
            dtype: DType::U32,
            generation: 0,
        }))
    }

    fn create_f32_from_bf16_bits(
        &mut self,
        _data: &[u16],
        _shape: &[i32],
    ) -> Result<TensorHandle, String> {
        Err("AneBackend: create_f32_from_bf16_bits not implemented".into())
    }

    fn create_owned_from_bytes(
        &mut self,
        data: &[u8],
        shape: &[i32],
        dtype: DType,
    ) -> Result<TensorHandle, String> {
        let buf = self.make_buffer(data)?;
        Ok(self.alloc_slot(AneTensor {
            buffer: Some(buf),
            shape: shape.to_vec(),
            dtype,
            generation: 0,
        }))
    }

    fn bind_external(
        &mut self,
        owner_token: u64,
        data: &[u8],
        shape: &[i32],
        dtype: DType,
    ) -> Result<TensorHandle, String> {
        let buf = self.make_buffer(data)?;
        let handle = self.alloc_slot(AneTensor {
            buffer: Some(buf),
            shape: shape.to_vec(),
            dtype,
            generation: 0,
        });
        self.external_bindings.insert(owner_token, handle);
        Ok(handle)
    }

    fn quantized_matmul(
        &mut self,
        _op: &QuantizedMatmulOp,
        _x: TensorHandle,
        _w: QuantizedWeightHandle,
        _scales: TensorHandle,
        _biases: TensorHandle,
    ) -> Result<TensorHandle, String> {
        Err("AneBackend: ANE does not execute primitive ops".into())
    }

    fn matmul(
        &mut self,
        _op: &MatmulOp,
        _a: TensorHandle,
        _b: TensorHandle,
    ) -> Result<TensorHandle, String> {
        Err("AneBackend: ANE does not execute primitive ops".into())
    }

    fn rms_norm(
        &mut self,
        _op: &RmsNormOp,
        _x: TensorHandle,
        _weight: TensorHandle,
    ) -> Result<TensorHandle, String> {
        Err("AneBackend: ANE does not execute primitive ops".into())
    }

    fn rope(&mut self, _op: &RoPEOp, _x: TensorHandle) -> Result<TensorHandle, String> {
        Err("AneBackend: ANE does not execute primitive ops".into())
    }

    fn add(&mut self, _a: TensorHandle, _b: TensorHandle) -> Result<TensorHandle, String> {
        Err("AneBackend: ANE does not execute primitive ops".into())
    }

    fn multiply(&mut self, _a: TensorHandle, _b: TensorHandle) -> Result<TensorHandle, String> {
        Err("AneBackend: ANE does not execute primitive ops".into())
    }

    fn silu(&mut self, _x: TensorHandle) -> Result<TensorHandle, String> {
        Err("AneBackend: ANE does not execute primitive ops".into())
    }

    fn transpose(&mut self, _x: TensorHandle, _dims: &[i32]) -> Result<TensorHandle, String> {
        Err("AneBackend: ANE does not execute primitive ops".into())
    }

    fn reshape(&mut self, x: TensorHandle, shape: &[i32]) -> Result<TensorHandle, String> {
        let t = self.slot(x)?;
        Ok(self.alloc_slot(AneTensor {
            buffer: t.buffer.clone(),
            shape: shape.to_vec(),
            dtype: t.dtype,
            generation: 0,
        }))
    }

    fn softmax(&mut self, _x: TensorHandle, _axis: i32) -> Result<TensorHandle, String> {
        Err("AneBackend: ANE does not execute primitive ops".into())
    }

    fn index_select(
        &mut self,
        _x: TensorHandle,
        _indices: &[u32],
        _axis: i32,
    ) -> Result<TensorHandle, String> {
        Err("AneBackend: ANE does not execute primitive ops".into())
    }

    fn evaluate(
        &mut self,
        group_id: u64,
        _outputs: &[TensorHandle],
    ) -> Result<EvaluationReceipt, String> {
        Ok(EvaluationReceipt {
            group_id,
            graph_build_ns: 0,
            submit_ns: 0,
            sync_ns: 0,
            output_count: 0,
            active_memory_after: 0,
            cache_memory_after: 0,
            observed_substrate: None,
            eval_calls: 1,
        })
    }

    fn read_f32(&mut self, handle: TensorHandle) -> Result<ReadbackReceipt, String> {
        let tensor = self.slot(handle)?;
        let buf = tensor
            .buffer
            .as_ref()
            .ok_or_else(|| "AneBackend: tensor has no buffer".to_string())?;
        let len = buf.length() as usize;
        let ptr = buf.contents() as *const f32;
        let data: Vec<f32> = if len >= 4 {
            unsafe { std::slice::from_raw_parts(ptr, len / 4) }.to_vec()
        } else {
            Vec::new()
        };
        Ok(ReadbackReceipt {
            data,
            forced_eval: false,
            sync_ns: 0,
            observed_substrate: None,
        })
    }

    fn shape(&self, handle: TensorHandle) -> Result<Vec<i32>, String> {
        Ok(self.slot(handle)?.shape.clone())
    }

    fn release(&mut self, handle: TensorHandle) -> Result<(), String> {
        let idx = handle.slot as usize;
        if self.slots.get(idx).and_then(|s| s.as_ref()).is_some() {
            self.slots[idx] = None;
            self.free.push(handle.slot);
            Ok(())
        } else {
            Err(format!(
                "AneBackend: invalid TensorHandle({}, {})",
                handle.slot, handle.generation
            ))
        }
    }

    fn active_memory(&self) -> (u64, u64) {
        (0, 0)
    }

    fn backend_capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            can_gpu: false,
            can_cpu: false,
            supports_quantized: false,
            supports_bf16_native: false,
            backend_name: "ane".into(),
        }
    }
}

// ── BackendInstance ─────────────────────────────────────────────────────────

#[cfg(target_os = "macos")]
impl BackendInstance for AneBackend {
    fn backend_kind(&self) -> BackendId {
        BACKEND_ANE
    }

    fn supports(&self, family: OperationFamily) -> bool {
        matches!(
            family,
            OperationFamily::AttentionBlock
                | OperationFamily::MlpBlock
                | OperationFamily::DecoderLayer
                | OperationFamily::PrefillFragment
        )
    }

    fn execute(
        &mut self,
        _op: &OperationDescriptor,
        _inputs: &[TensorHandle],
    ) -> Result<BackendExecutionReceipt, String> {
        // ANE keepalive heartbeat even on unsupported primitive ops
        self.steps_since_heartbeat += 1;
        if self.steps_since_heartbeat >= ANE_KEEPALIVE_INTERVAL {
            let _ = self.keepalive();
        }
        Err("AneBackend: ANE does not execute primitive ops".into())
    }

    fn as_compiled_region_backend(&mut self) -> Option<&mut dyn CompiledRegionBackend> {
        Some(self)
    }
}

// ── CompiledRegionBackend ───────────────────────────────────────────────────

impl CompiledRegionBackend for AneBackend {
    fn supports_region(&self, family: OperationFamily) -> bool {
        matches!(
            family,
            OperationFamily::AttentionBlock
                | OperationFamily::MlpBlock
                | OperationFamily::DecoderLayer
                | OperationFamily::PrefillFragment
        )
    }

    fn execute_compiled_region(
        &mut self,
        region: &GraphRegion,
        _inputs: &[TensorHandle],
        _outputs: &[TensorHandle],
    ) -> Result<BackendExecutionReceipt, String> {
        // ANE keepalive heartbeat — run before program lookup to avoid
        // borrowing conflicts and to warm the ANE before the real work.
        self.steps_since_heartbeat += 1;
        if self.steps_since_heartbeat >= ANE_KEEPALIVE_INTERVAL {
            let _ = self.keepalive();
        }

        // Extract seq_len and batch_sz from the first shape constraint.
        let seq_len = region
            .shape_constraints
            .first()
            .and_then(|ts| ts.dims.first().copied())
            .unwrap_or(0);
        let batch_sz = region
            .shape_constraints
            .first()
            .and_then(|ts| ts.dims.get(1).copied())
            .unwrap_or(0);

        // Try multi-key cache first, then fall back to region_id.
        let step = self
            .find_program(region.family, seq_len, batch_sz)
            .or_else(|| self.region_programs.get(&region.region_id))
            .ok_or_else(|| {
                format!(
                    "AneBackend: no compiled program for region {}",
                    region.region_id
                )
            })?;

        // ── Build IOSurface input pointers ──
        // Ordered by step.inputs to match the MIL program's declared input order.
        let weight_name = format!("region_{}_weights", region.region_id);
        let mut input_ptrs: Vec<*mut c_void> = Vec::with_capacity(step.inputs.len());
        for input_name in &step.inputs {
            // Resolve input by priority: activation tensors → weight bindings → KV cache
            let iosurface = if let Some(handle) = region.inputs.get(input_name) {
                // 1. Region activation tensors (named by the graph compiler)
                match self.slot(*handle) {
                    Ok(t) => t.buffer.as_ref().and_then(|b| buffer_to_ane_input(b).ok()),
                    Err(_) => None,
                }
            } else if *input_name == weight_name {
                // 2. Packed nf4tile640 weight buffer
                self.packed_weight_bindings
                    .get(&weight_name)
                    .and_then(|b| buffer_to_ane_input(b).ok())
            } else if matches!(region.family, OperationFamily::AttentionBlock) {
                // 3. KV cache buffers for attention programs
                if input_name == "kv_codes" {
                    self.kv_codes_handle
                        .as_ref()
                        .and_then(|b| buffer_to_ane_input(b.as_ref()).ok())
                } else if input_name == "kv_scales" {
                    self.kv_scales_handle
                        .as_ref()
                        .and_then(|b| buffer_to_ane_input(b.as_ref()).ok())
                } else if input_name == "kv_biases" {
                    self.kv_biases_handle
                        .as_ref()
                        .and_then(|b| buffer_to_ane_input(b.as_ref()).ok())
                } else {
                    None
                }
            } else {
                None
            };
            input_ptrs.push(iosurface.unwrap_or(std::ptr::null_mut()));
        }

        // ── Build IOSurface output pointers ──
        // Ordered by step.outputs to match the MIL program's output declaration.
        let mut output_ptrs: Vec<*mut c_void> = Vec::with_capacity(step.outputs.len());
        for output_name in &step.outputs {
            let iosurface = region.outputs.get(output_name).and_then(|handle| {
                self.slot(*handle)
                    .ok()
                    .and_then(|t| t.buffer.as_ref())
                    .and_then(|b| buffer_to_ane_input(b).ok())
            });
            output_ptrs.push(iosurface.unwrap_or(std::ptr::null_mut()));
        }

        // ── Execute the ANE program with IOSurface bindings ──
        let program = self
            .program_cache
            .get_or_compile(&step.mil_text, &step.tag)?;
        let start = std::time::Instant::now();
        program.evaluate(&input_ptrs, &output_ptrs)?;
        let elapsed = start.elapsed().as_nanos() as u64;

        Ok(BackendExecutionReceipt {
            operation_id: OperationId(region.region_id),
            backend_id: BACKEND_ANE,
            backend_version: BackendVersion {
                backend_name: "ane".into(),
                version: "0.1".into(),
                git_commit: None,
            },
            requested_substrate: Some(RequestedSubstrate::NeuralEngine),
            observed_substrate: Some(Substrate::NeuralEngine),
            graph_build_ns: Some(0),
            compile_ns: None,
            queue_wait_ns: None,
            submit_ns: Some(0),
            execution_ns: Some(elapsed),
            synchronization_ns: None,
            total_wall_ns: elapsed,
            bytes_read: None,
            bytes_written: None,
            temporary_bytes: Some(0),
            active_memory_before: None,
            active_memory_after: None,
            cache_memory_before: None,
            cache_memory_after: None,
            transfer_in_ns: None,
            transfer_out_ns: None,
            fallback_occurred: false,
        })
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Register an AneInferenceStep for a given region ID.
///
/// Called by the cimage loader in Phase 7 to populate the region→program
/// mapping before compiled-region execution begins.
pub fn register_ane_region(backend: &mut AneBackend, region_id: u64, step: AneInferenceStep) {
    backend.region_programs.insert(region_id, step);
}

/// Export a Metal buffer's IOSurface for ANE program input binding.
///
/// The ANE C API (`tribunus_ane_eval`) accepts IOSurface pointers as the
/// `*mut c_void` entries in the input and output pointer arrays.
/// Only buffers created with IOSurface support (StorageModeShared or
/// Private with .allowIOSurface) return a valid IOSurface.
fn buffer_to_ane_input(buffer: &metal::Buffer) -> Result<*mut c_void, String> {
    unsafe {
        let iosurface: *mut c_void = msg_send![buffer.as_ref(), iosurface];
        if iosurface.is_null() {
            Err("Buffer has no IOSurface backing".into())
        } else {
            Ok(iosurface)
        }
    }
}
