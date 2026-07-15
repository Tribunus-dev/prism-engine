//! Metal backend — production [`TensorBackend`] + [`BackendInstance`]
//! using `mpsgraph` (Apple Metal Performance Shaders Graph).
//!
//! Gated behind `target_os = "macos"` — Linux builds are unaffected.

#![cfg(target_os = "macos")]

use std::collections::HashMap;
use std::sync::Arc;

use crate::completion::{ComellationToken, ComputationToken};
use mpsgraph::{DataType, Device, Graph, ShapedType, TensorData};

use crate::heterogeneous_executor::BackendInstance;
use crate::routing::{
    BackendExecutionReceipt, BackendId, BackendVersion, OperationDescriptor, OperationFamily,
    BACKEND_METAL,
};
use crate::{
    BackendCapabilities, DType, EvaluationReceipt, MatmulOp, QuantizedMatmulOp,
    QuantizedWeightHandle, ReadbackReceipt, RmsNormOp, RoPEOp, TensorBackend, TensorHandle,
};
use prism_ecs_core::canonical::kernel_abi::KernelSemanticId;
use prism_ecs_core::metal_backend::{catalogue_source_for, MetalImplementationCatalogue};
use prism_ecs_core::mlir::runtime_lowering_artifact;

/// One live tensor stored in the Metal backend.
/// Host-side representation of `Nf4Tile640DispatchParams` in `shaders/nf4tile640.metal`.
#[repr(C, align(16))]
struct Nf4Tile640DispatchParams {
    abi_version: u32,
    m: u32,
    k: u32,
    n: u32,
    group_size: u32,
    reserved: [u32; 3],
}

const _: () = assert!(std::mem::size_of::<Nf4Tile640DispatchParams>() == 32);
const _: () = assert!(std::mem::align_of::<Nf4Tile640DispatchParams>() == 16);

#[repr(C, align(16))]
struct MlpConstants {
    hidden_dim: u32,
    intermediate_dim: u32,
    group_size: u32,
    codec_id: u32,
    epsilon: f32,
    pad: [u32; 3],
}

#[repr(C, align(16))]
struct TernaryGemvConstants {
    rows: u32,
    cols: u32,
    group_size: u32,
    groups_per_row: u32,
    bytes_per_group: u32,
    output_dtype: u32,
    padding: [u32; 3],
}

struct MetalTensor {
    buffer: Option<metal::Buffer>,
    shape: Vec<i32>,
    dtype: DType,
    generation: u32,
}

pub struct MetalBackend {
    mtl_device: metal::Device,
    slots: Vec<Option<MetalTensor>>,
    free: Vec<u32>,
    next_generation: u32,
    /// Slot map for quantized weight bytes (packed nf4tile640 format).
    weight_slots: Vec<Option<Vec<u8>>>,
    weight_generations: Vec<u32>,
    #[allow(dead_code)]
    weight_free: Vec<u32>,
    /// Cached precision pipelines keyed by the canonical semantic kernel ID.
    precision_pipelines: HashMap<String, metal::ComputePipelineState>,
    /// Libraries must outlive their pipeline states.
    precision_libraries: HashMap<String, metal::Library>,
    /// Owner token → tensor handle for externally bound (IOSurface) tensors.
    external_bindings: HashMap<u64, TensorHandle>,
}

impl MetalBackend {
    pub fn new() -> Result<Self, String> {
        let mtl_device =
            metal::Device::system_default().ok_or_else(|| "no Metal device found".to_string())?;
        Ok(Self {
            mtl_device,
            slots: Vec::new(),
            free: Vec::new(),
            next_generation: 1,
            weight_slots: Vec::new(),
            weight_generations: Vec::new(),
            weight_free: Vec::new(),
            precision_pipelines: HashMap::new(),
            precision_libraries: HashMap::new(),
            external_bindings: HashMap::new(),
        })
    }

    fn alloc_slot(&mut self, mut tensor: MetalTensor) -> TensorHandle {
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

    fn slot(&self, handle: TensorHandle) -> Result<&MetalTensor, String> {
        match self.slots.get(handle.slot as usize) {
            Some(Some(t)) if t.generation == handle.generation => Ok(t),
            _ => Err(format!(
                "invalid TensorHandle({}, {})",
                handle.slot, handle.generation
            )),
        }
    }

    /// Register a quantized weight blob (packed nf4tile640 bytes).
    pub fn alloc_weight(&mut self, data: Vec<u8>) -> QuantizedWeightHandle {
        if let Some(idx) = self.weight_free.pop() {
            let i = idx as usize;
            self.weight_generations[i] = self.weight_generations[i].checked_add(1).unwrap_or(1);
            self.weight_slots[i] = Some(data);
            QuantizedWeightHandle {
                slot: idx,
                generation: self.weight_generations[i],
            }
        } else {
            let slot = self.weight_slots.len() as u32;
            self.weight_slots.push(Some(data));
            self.weight_generations.push(1);
            QuantizedWeightHandle {
                slot,
                generation: 1,
            }
        }
    }

    fn get_weight(&self, handle: QuantizedWeightHandle) -> Result<&[u8], String> {
        let i = handle.slot as usize;
        match self.weight_slots.get(i) {
            Some(Some(data)) if handle.generation == self.weight_generations[i] => Ok(data),
            _ => Err(format!(
                "invalid QuantizedWeightHandle({}, {})",
                handle.slot, handle.generation
            )),
        }
    }

    /// Lazily compile any catalogue-registered precision kernel and cache its
    /// pipeline by semantic identity.
    pub fn ensure_precision_kernel(
        &mut self,
        semantic_id: &KernelSemanticId,
    ) -> Result<(), String> {
        if self.precision_pipelines.contains_key(&semantic_id.0) {
            return Ok(());
        }
        let source = catalogue_source_for(semantic_id)
            .ok_or_else(|| format!("Metal catalogue has no source for {}", semantic_id.0))?;
        let catalogue = MetalImplementationCatalogue::default();
        let registration = catalogue
            .for_semantic(semantic_id)
            .into_iter()
            .find(|registration| registration.source_entry_point.is_some())
            .ok_or_else(|| {
                format!(
                    "Metal catalogue has no executable entry point for {}",
                    semantic_id.0
                )
            })?;
        let entry_point = registration.source_entry_point.clone().unwrap();
        self.compile_precision_pipeline(semantic_id, &source, &entry_point)
    }

    fn compile_precision_pipeline(
        &mut self,
        semantic_id: &KernelSemanticId,
        source: &str,
        entry_point: &str,
    ) -> Result<(), String> {
        if self.precision_pipelines.contains_key(&semantic_id.0) {
            return Ok(());
        }
        let lib = self
            .mtl_device
            .new_library_with_source(source, &metal::CompileOptions::new())
            .map_err(|e| format!("Metal library compile failed: {e}"))?;
        let kernel = lib
            .get_function(entry_point, None::<metal::FunctionConstantValues>)
            .map_err(|e| format!("Metal kernel not found: {e}"))?;
        let pipeline = self
            .mtl_device
            .new_compute_pipeline_state_with_function(&kernel)
            .map_err(|e| format!("Metal pipeline state creation failed: {e}"))?;
        self.precision_libraries.insert(semantic_id.0.clone(), lib);
        self.precision_pipelines
            .insert(semantic_id.0.clone(), pipeline);
        Ok(())
    }

    /// Dispatch a catalogue-registered precision kernel using its declared
    /// buffer slots. The caller owns ABI-specific buffer construction; this
    /// boundary owns semantic selection, pipeline caching, slot binding, and
    /// synchronization.
    pub fn dispatch_precision_kernel(
        &mut self,
        semantic_id: &KernelSemanticId,
        buffers: &[Option<&metal::Buffer>],
        grid: metal::MTLSize,
        threadgroup: metal::MTLSize,
    ) -> Result<(), String> {
        let translated = runtime_lowering_artifact(
            semantic_id,
            [grid.width as u32, grid.height as u32, grid.depth as u32],
            [
                threadgroup.width as u32,
                threadgroup.height as u32,
                threadgroup.depth as u32,
            ],
        )?;
        self.compile_precision_pipeline(semantic_id, &translated.source, &translated.entry_point)?;
        let pipeline = self
            .precision_pipelines
            .get(&semantic_id.0)
            .ok_or_else(|| format!("pipeline missing after compilation for {}", semantic_id.0))?;
        let queue = self.mtl_device.new_command_queue();
        let command = queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        for (slot, buffer) in buffers.iter().enumerate() {
            if let Some(buffer) = buffer {
                encoder.set_buffer(slot as u64, Some(buffer), 0);
            }
        }
        encoder.dispatch_threads(
            metal::MTLSize::new(
                translated.dispatch.grid[0] as u64,
                translated.dispatch.grid[1] as u64,
                translated.dispatch.grid[2] as u64,
            ),
            metal::MTLSize::new(
                translated.dispatch.threadgroup[0] as u64,
                translated.dispatch.threadgroup[1] as u64,
                translated.dispatch.threadgroup[2] as u64,
            ),
        );
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        Ok(())
    }

    fn quantized_matmul_int8(
        &mut self,
        op: &QuantizedMatmulOp,
        x: TensorHandle,
        w: QuantizedWeightHandle,
        scales: TensorHandle,
        biases: TensorHandle,
    ) -> Result<TensorHandle, String> {
        let x_buf = self
            .slot(x)?
            .buffer
            .as_ref()
            .ok_or("INT8 input tensor has no buffer")?
            .clone();
        let weights = self.get_weight(w)?.to_vec();
        let scale_buf = self
            .slot(scales)?
            .buffer
            .as_ref()
            .ok_or("INT8 scales tensor has no buffer")?
            .clone();
        let bias_buf = self
            .slot(biases)?
            .buffer
            .as_ref()
            .ok_or("INT8 biases tensor has no buffer")?
            .clone();
        let expected = (op.k as usize).saturating_mul(op.n as usize);
        if weights.len() != expected {
            return Err(format!(
                "INT8 weights size {} != expected {}",
                weights.len(),
                expected
            ));
        }
        let weight_buf = self.mtl_device.new_buffer_with_data(
            weights.as_ptr() as *const std::ffi::c_void,
            weights.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output_buf = self.mtl_device.new_buffer(
            (op.m as usize * op.n as usize * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let constants = MlpConstants {
            hidden_dim: op.k,
            intermediate_dim: op.n,
            group_size: op.group_size,
            codec_id: 1,
            epsilon: 0.0,
            pad: [0; 3],
        };
        let constants_buf = self.mtl_device.new_buffer_with_data(
            &constants as *const MlpConstants as *const std::ffi::c_void,
            std::mem::size_of::<MlpConstants>() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        self.dispatch_precision_kernel(
            &KernelSemanticId("prism.linear.int8.v1".into()),
            &[
                Some(&x_buf),
                Some(&weight_buf),
                Some(&scale_buf),
                Some(&bias_buf),
                Some(&output_buf),
                Some(&constants_buf),
            ],
            metal::MTLSize::new(op.n as u64, op.m as u64, 1),
            metal::MTLSize::new(64, 1, 1),
        )?;
        Ok(self.alloc_slot(MetalTensor {
            buffer: Some(output_buf),
            shape: vec![op.m as i32, op.n as i32],
            dtype: DType::F32,
            generation: 0,
        }))
    }

    fn quantized_matmul_ternary_tile640(
        &mut self,
        op: &QuantizedMatmulOp,
        x: TensorHandle,
        w: QuantizedWeightHandle,
        page_scales: TensorHandle,
        lane_scales: TensorHandle,
    ) -> Result<TensorHandle, String> {
        if op.m != 1 || op.input_dtype != DType::F16 || op.output_dtype != DType::F16 {
            return Err("tile640 ternary dispatch requires m=1 and F16 input/output".into());
        }
        let input = self
            .slot(x)?
            .buffer
            .as_ref()
            .ok_or("ternary input has no buffer")?
            .clone();
        let packed = self.get_weight(w)?.to_vec();
        let page = self
            .slot(page_scales)?
            .buffer
            .as_ref()
            .ok_or("ternary page scales have no buffer")?
            .clone();
        let lane = self
            .slot(lane_scales)?
            .buffer
            .as_ref()
            .ok_or("ternary lane scales have no buffer")?
            .clone();
        let output = self.mtl_device.new_buffer(
            op.n as u64 * std::mem::size_of::<u16>() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let packed_buf = self.mtl_device.new_buffer_with_data(
            packed.as_ptr() as *const std::ffi::c_void,
            packed.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let in_dim = self.mtl_device.new_buffer_with_data(
            &op.k as *const u32 as *const std::ffi::c_void,
            4,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let out_dim = self.mtl_device.new_buffer_with_data(
            &op.n as *const u32 as *const std::ffi::c_void,
            4,
            metal::MTLResourceOptions::StorageModeShared,
        );
        self.dispatch_precision_kernel(
            &KernelSemanticId("prism.ternary.gemv.v1".into()),
            &[
                Some(&packed_buf),
                Some(&input),
                Some(&page),
                Some(&lane),
                Some(&output),
                Some(&in_dim),
                Some(&out_dim),
            ],
            metal::MTLSize::new(op.n as u64, 1, 1),
            metal::MTLSize::new(64, 1, 1),
        )?;
        Ok(self.alloc_slot(MetalTensor {
            buffer: Some(output),
            shape: vec![1, op.n as i32],
            dtype: DType::F16,
            generation: 0,
        }))
    }

    fn quantized_matmul_q4(
        &mut self,
        op: &QuantizedMatmulOp,
        x: TensorHandle,
        w: QuantizedWeightHandle,
        scales: TensorHandle,
    ) -> Result<TensorHandle, String> {
        if op.m != 1 || op.input_dtype != DType::F16 || op.output_dtype != DType::F16 {
            return Err("Q4 block-symmetric dispatch requires m=1 and F16 input/output".into());
        }
        if op.k % 8 != 0 || op.k % op.group_size != 0 {
            return Err("Q4 dimensions must be divisible by 8 and group_size".into());
        }
        let input = self
            .slot(x)?
            .buffer
            .as_ref()
            .ok_or("Q4 input has no buffer")?
            .clone();
        let packed = self.get_weight(w)?.to_vec();
        let scale = self
            .slot(scales)?
            .buffer
            .as_ref()
            .ok_or("Q4 scales have no buffer")?
            .clone();
        let expected = (op.n as usize) * (op.k as usize) / 2;
        if packed.len() != expected {
            return Err(format!(
                "Q4 packed weights size {} != expected {}",
                packed.len(),
                expected
            ));
        }
        let weights = self.mtl_device.new_buffer_with_data(
            packed.as_ptr() as *const std::ffi::c_void,
            packed.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output = self.mtl_device.new_buffer(
            op.n as u64 * 2,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let k_buf = self.mtl_device.new_buffer_with_data(
            &op.k as *const u32 as *const std::ffi::c_void,
            4,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let n_buf = self.mtl_device.new_buffer_with_data(
            &op.n as *const u32 as *const std::ffi::c_void,
            4,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let group_buf = self.mtl_device.new_buffer_with_data(
            &op.group_size as *const u32 as *const std::ffi::c_void,
            4,
            metal::MTLResourceOptions::StorageModeShared,
        );
        self.dispatch_precision_kernel(
            &KernelSemanticId("prism.q4.block_sym.gemv.v1".into()),
            &[
                Some(&input),
                Some(&weights),
                Some(&scale),
                Some(&output),
                Some(&k_buf),
                Some(&n_buf),
                Some(&group_buf),
            ],
            metal::MTLSize::new(op.n as u64, 1, 1),
            metal::MTLSize::new(64, 1, 1),
        )?;
        Ok(self.alloc_slot(MetalTensor {
            buffer: Some(output),
            shape: vec![1, op.n as i32],
            dtype: DType::F16,
            generation: 0,
        }))
    }

    fn quantized_matmul_palettized(
        &mut self,
        op: &QuantizedMatmulOp,
        x: TensorHandle,
        w: QuantizedWeightHandle,
        codebook: TensorHandle,
    ) -> Result<TensorHandle, String> {
        if op.m != 1 || op.input_dtype != DType::F16 || op.output_dtype != DType::F16 {
            return Err("palettized GEMV requires m=1 and F16 input/output".into());
        }
        if op.k % 8 != 0 {
            return Err("palettized GEMV input dimension must be divisible by 8".into());
        }
        let input = self
            .slot(x)?
            .buffer
            .as_ref()
            .ok_or("palette input has no buffer")?
            .clone();
        let indices = self.get_weight(w)?.to_vec();
        let codebook = self
            .slot(codebook)?
            .buffer
            .as_ref()
            .ok_or("palette codebook has no buffer")?
            .clone();
        let expected = (op.n as usize) * (op.k as usize) / 2;
        if indices.len() != expected {
            return Err(format!(
                "palette indices size {} != expected {}",
                indices.len(),
                expected
            ));
        }
        let index_buf = self.mtl_device.new_buffer_with_data(
            indices.as_ptr() as *const std::ffi::c_void,
            indices.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output = self.mtl_device.new_buffer(
            op.n as u64 * 2,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let in_dim = self.mtl_device.new_buffer_with_data(
            &op.k as *const u32 as *const std::ffi::c_void,
            4,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let out_dim = self.mtl_device.new_buffer_with_data(
            &op.n as *const u32 as *const std::ffi::c_void,
            4,
            metal::MTLResourceOptions::StorageModeShared,
        );
        self.dispatch_precision_kernel(
            &KernelSemanticId("prism.palettized.gemv.v1".into()),
            &[
                Some(&input),
                Some(&codebook),
                Some(&index_buf),
                Some(&output),
                Some(&in_dim),
                Some(&out_dim),
            ],
            metal::MTLSize::new(op.n as u64, 1, 1),
            metal::MTLSize::new(64, 1, 1),
        )?;
        Ok(self.alloc_slot(MetalTensor {
            buffer: Some(output),
            shape: vec![1, op.n as i32],
            dtype: DType::F16,
            generation: 0,
        }))
    }

    fn quantized_matmul_palettized_gemm(
        &mut self,
        op: &QuantizedMatmulOp,
        x: TensorHandle,
        w: QuantizedWeightHandle,
        codebook: TensorHandle,
    ) -> Result<TensorHandle, String> {
        if op.input_dtype != DType::F16 || op.output_dtype != DType::F16 || op.k % 2 != 0 {
            return Err("palettized GEMM requires F16 tensors and even input dimension".into());
        }
        let input = self
            .slot(x)?
            .buffer
            .as_ref()
            .ok_or("palette input has no buffer")?
            .clone();
        let indices = self.get_weight(w)?.to_vec();
        let cb_buf = self
            .slot(codebook)?
            .buffer
            .as_ref()
            .ok_or("palette codebook has no buffer")?
            .clone();
        let cb_bytes = op.n as usize * 16 * 2;
        let cb_ptr = cb_buf.contents() as *const u8;
        let codebook_bytes = unsafe { std::slice::from_raw_parts(cb_ptr, cb_bytes) };
        let row_stride = 32 + op.k as usize / 2;
        let expected_indices = op.n as usize * op.k as usize / 2;
        if indices.len() != expected_indices {
            return Err(format!(
                "palette GEMM indices size {} != expected {}",
                indices.len(),
                expected_indices
            ));
        }
        let mut arena = vec![0u8; op.n as usize * row_stride];
        for row in 0..op.n as usize {
            arena[row * row_stride..row * row_stride + 32]
                .copy_from_slice(&codebook_bytes[row * 32..row * 32 + 32]);
            let src = row * (op.k as usize / 2);
            arena[row * row_stride + 32..(row + 1) * row_stride]
                .copy_from_slice(&indices[src..src + op.k as usize / 2]);
        }
        let arena_buf = self.mtl_device.new_buffer_with_data(
            arena.as_ptr() as *const std::ffi::c_void,
            arena.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output = self.mtl_device.new_buffer(
            op.m as u64 * op.n as u64 * 2,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let m_buf = self.mtl_device.new_buffer_with_data(
            &op.m as *const u32 as *const std::ffi::c_void,
            4,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let k_buf = self.mtl_device.new_buffer_with_data(
            &op.k as *const u32 as *const std::ffi::c_void,
            4,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let n_buf = self.mtl_device.new_buffer_with_data(
            &op.n as *const u32 as *const std::ffi::c_void,
            4,
            metal::MTLResourceOptions::StorageModeShared,
        );
        self.dispatch_precision_kernel(
            &KernelSemanticId("prism.palettized.gemm.v1".into()),
            &[
                Some(&arena_buf),
                Some(&input),
                Some(&output),
                Some(&m_buf),
                Some(&k_buf),
                Some(&n_buf),
            ],
            metal::MTLSize::new(op.n.div_ceil(16) as u64, op.m.div_ceil(16) as u64, 1),
            metal::MTLSize::new(16, 16, 1),
        )?;
        Ok(self.alloc_slot(MetalTensor {
            buffer: Some(output),
            shape: vec![op.m as i32, op.n as i32],
            dtype: DType::F16,
            generation: 0,
        }))
    }

    pub fn dispatch_palettized_swiglu(
        &mut self,
        input: TensorHandle,
        gate_weights: QuantizedWeightHandle,
        gate_codebook: TensorHandle,
        up_weights: QuantizedWeightHandle,
        up_codebook: TensorHandle,
        in_dim: u32,
        out_dim: u32,
    ) -> Result<TensorHandle, String> {
        let input_buf = self
            .slot(input)?
            .buffer
            .as_ref()
            .ok_or("SwiGLU input has no buffer")?
            .clone();
        let make_arena = |backend: &MetalBackend,
                          weights: QuantizedWeightHandle,
                          cb: TensorHandle|
         -> Result<Vec<u8>, String> {
            let indices = backend.get_weight(weights)?.to_vec();
            let cb_buf = backend
                .slot(cb)?
                .buffer
                .as_ref()
                .ok_or("SwiGLU codebook has no buffer")?
                .clone();
            let cb_slice = unsafe {
                std::slice::from_raw_parts(cb_buf.contents() as *const u8, out_dim as usize * 32)
            };
            let row_stride = 32 + in_dim as usize / 2;
            let mut arena = vec![0u8; out_dim as usize * row_stride];
            for row in 0..out_dim as usize {
                arena[row * row_stride..row * row_stride + 32]
                    .copy_from_slice(&cb_slice[row * 32..row * 32 + 32]);
                let start = row * in_dim as usize / 2;
                arena[row * row_stride + 32..(row + 1) * row_stride]
                    .copy_from_slice(&indices[start..start + in_dim as usize / 2]);
            }
            Ok(arena)
        };
        let gate = make_arena(self, gate_weights, gate_codebook)?;
        let up = make_arena(self, up_weights, up_codebook)?;
        let gate_buf = self.mtl_device.new_buffer_with_data(
            gate.as_ptr() as *const std::ffi::c_void,
            gate.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let up_buf = self.mtl_device.new_buffer_with_data(
            up.as_ptr() as *const std::ffi::c_void,
            up.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output = self.mtl_device.new_buffer(
            out_dim as u64 * 2,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let in_buf = self.mtl_device.new_buffer_with_data(
            &in_dim as *const u32 as *const std::ffi::c_void,
            4,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let out_buf = self.mtl_device.new_buffer_with_data(
            &out_dim as *const u32 as *const std::ffi::c_void,
            4,
            metal::MTLResourceOptions::StorageModeShared,
        );
        self.dispatch_precision_kernel(
            &KernelSemanticId("prism.palettized.swiglu.v1".into()),
            &[
                Some(&gate_buf),
                Some(&up_buf),
                Some(&input_buf),
                Some(&output),
                Some(&in_buf),
                Some(&out_buf),
            ],
            metal::MTLSize::new(out_dim as u64, 1, 1),
            metal::MTLSize::new(64, 1, 1),
        )?;
        Ok(self.alloc_slot(MetalTensor {
            buffer: Some(output),
            shape: vec![1, out_dim as i32],
            dtype: DType::F16,
            generation: 0,
        }))
    }

    pub fn dispatch_linear_nf4_cimage(
        &mut self,
        input: TensorHandle,
        weights: QuantizedWeightHandle,
        scales: TensorHandle,
        biases: TensorHandle,
        in_dim: u32,
        out_dim: u32,
        group_size: u32,
    ) -> Result<TensorHandle, String> {
        if group_size != 128 || out_dim > 640 {
            return Err("CImage NF4 requires group_size=128 and out_dim<=640".into());
        }
        let input_buf = self
            .slot(input)?
            .buffer
            .as_ref()
            .ok_or("CImage NF4 input has no buffer")?
            .clone();
        let scale_buf = self
            .slot(scales)?
            .buffer
            .as_ref()
            .ok_or("CImage NF4 scales have no buffer")?
            .clone();
        let bias_buf = self
            .slot(biases)?
            .buffer
            .as_ref()
            .ok_or("CImage NF4 biases have no buffer")?
            .clone();
        let codes = self.get_weight(weights)?.to_vec();
        let expected = in_dim as usize * 320;
        if codes.len() != expected {
            return Err(format!(
                "CImage NF4 codes size {} != expected {}",
                codes.len(),
                expected
            ));
        }
        let codes_buf = self.mtl_device.new_buffer_with_data(
            codes.as_ptr() as *const std::ffi::c_void,
            codes.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output = self.mtl_device.new_buffer(
            out_dim as u64 * 4,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let constants = MlpConstants {
            hidden_dim: in_dim,
            intermediate_dim: out_dim,
            group_size,
            codec_id: 0,
            epsilon: 0.0,
            pad: [0; 3],
        };
        let constants_buf = self.mtl_device.new_buffer_with_data(
            &constants as *const MlpConstants as *const std::ffi::c_void,
            std::mem::size_of::<MlpConstants>() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        self.dispatch_precision_kernel(
            &KernelSemanticId("prism.linear.nf4.v1".into()),
            &[
                Some(&input_buf),
                Some(&codes_buf),
                Some(&scale_buf),
                Some(&bias_buf),
                Some(&output),
                Some(&constants_buf),
            ],
            metal::MTLSize::new(out_dim as u64, 1, 1),
            metal::MTLSize::new(64, 1, 1),
        )?;
        Ok(self.alloc_slot(MetalTensor {
            buffer: Some(output),
            shape: vec![1, out_dim as i32],
            dtype: DType::F32,
            generation: 0,
        }))
    }

    fn quantized_matmul_ternary_cimage(
        &mut self,
        op: &QuantizedMatmulOp,
        x: TensorHandle,
        w: QuantizedWeightHandle,
        scales: TensorHandle,
    ) -> Result<TensorHandle, String> {
        if op.m != 1 || op.input_dtype != DType::F16 || op.output_dtype != DType::F16 {
            return Err("CImage ternary GEMV requires m=1 and F16 input/output".into());
        }
        let groups = op.k.div_ceil(op.group_size);
        let bytes_per_group = op.group_size.div_ceil(4);
        let codes = self.get_weight(w)?.to_vec();
        let expected = op.n as usize * groups as usize * bytes_per_group as usize;
        if codes.len() != expected {
            return Err(format!(
                "CImage ternary codes size {} != expected {}",
                codes.len(),
                expected
            ));
        }
        let input = self
            .slot(x)?
            .buffer
            .as_ref()
            .ok_or("CImage ternary input has no buffer")?
            .clone();
        let scale = self
            .slot(scales)?
            .buffer
            .as_ref()
            .ok_or("CImage ternary scales have no buffer")?
            .clone();
        let codes_buf = self.mtl_device.new_buffer_with_data(
            codes.as_ptr() as *const std::ffi::c_void,
            codes.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output = self.mtl_device.new_buffer(
            op.n as u64 * 2,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let constants = TernaryGemvConstants {
            rows: op.n,
            cols: op.k,
            group_size: op.group_size,
            groups_per_row: groups,
            bytes_per_group,
            output_dtype: 0,
            padding: [0; 3],
        };
        let constants_buf = self.mtl_device.new_buffer_with_data(
            &constants as *const TernaryGemvConstants as *const std::ffi::c_void,
            std::mem::size_of::<TernaryGemvConstants>() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        self.dispatch_precision_kernel(
            &KernelSemanticId("prism.ternary.cimage.gemv.v1".into()),
            &[
                Some(&input),
                Some(&codes_buf),
                Some(&scale),
                Some(&output),
                Some(&constants_buf),
            ],
            metal::MTLSize::new(op.n as u64, 1, 1),
            metal::MTLSize::new(64, 1, 1),
        )?;
        Ok(self.alloc_slot(MetalTensor {
            buffer: Some(output),
            shape: vec![1, op.n as i32],
            dtype: DType::F16,
            generation: 0,
        }))
    }

    fn quantized_matmul_ternary_legacy(
        &mut self,
        op: &QuantizedMatmulOp,
        x: TensorHandle,
        w: QuantizedWeightHandle,
    ) -> Result<TensorHandle, String> {
        if op.m != 1
            || op.input_dtype != DType::F16
            || op.output_dtype != DType::F16
            || op.k % 4 != 0
        {
            return Err(
                "legacy ternary GEMV requires m=1, F16 tensors, and K divisible by 4".into(),
            );
        }
        let input = self
            .slot(x)?
            .buffer
            .as_ref()
            .ok_or("ternary input has no buffer")?
            .clone();
        let codes = self.get_weight(w)?.to_vec();
        let expected = op.n as usize * op.k as usize / 4;
        if codes.len() != expected {
            return Err(format!(
                "ternary packed weights size {} != expected {}",
                codes.len(),
                expected
            ));
        }
        let codes_buf = self.mtl_device.new_buffer_with_data(
            codes.as_ptr() as *const std::ffi::c_void,
            codes.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output = self.mtl_device.new_buffer(
            op.n as u64 * 2,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let k_buf = self.mtl_device.new_buffer_with_data(
            &op.k as *const u32 as *const std::ffi::c_void,
            4,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let n_buf = self.mtl_device.new_buffer_with_data(
            &op.n as *const u32 as *const std::ffi::c_void,
            4,
            metal::MTLResourceOptions::StorageModeShared,
        );
        self.dispatch_precision_kernel(
            &KernelSemanticId("prism.ternary.gemv.v2".into()),
            &[
                Some(&codes_buf),
                Some(&input),
                Some(&output),
                Some(&k_buf),
                Some(&n_buf),
            ],
            metal::MTLSize::new(op.n as u64, 1, 1),
            metal::MTLSize::new(64, 1, 1),
        )?;
        Ok(self.alloc_slot(MetalTensor {
            buffer: Some(output),
            shape: vec![1, op.n as i32],
            dtype: DType::F16,
            generation: 0,
        }))
    }

    fn quantized_matmul_ternary_gemm(
        &mut self,
        op: &QuantizedMatmulOp,
        x: TensorHandle,
        w: QuantizedWeightHandle,
        scales: TensorHandle,
    ) -> Result<TensorHandle, String> {
        if op.input_dtype != DType::F16 || op.output_dtype != DType::F16 || op.k % 16 != 0 {
            return Err("ternary GEMM requires F16 tensors and K divisible by 16".into());
        }
        let input = self
            .slot(x)?
            .buffer
            .as_ref()
            .ok_or("ternary GEMM input has no buffer")?
            .clone();
        let weights = self.get_weight(w)?.to_vec();
        let scale = self
            .slot(scales)?
            .buffer
            .as_ref()
            .ok_or("ternary GEMM scales have no buffer")?
            .clone();
        let packed_per_row = op.k.div_ceil(16) as usize * 4;
        let expected_weights = op.n as usize * packed_per_row;
        if weights.len() != expected_weights {
            return Err(format!(
                "ternary GEMM weights size {} != expected {}",
                weights.len(),
                expected_weights
            ));
        }
        let weight_buf = self.mtl_device.new_buffer_with_data(
            weights.as_ptr() as *const std::ffi::c_void,
            weights.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output = self.mtl_device.new_buffer(
            op.m as u64 * op.n as u64 * 2,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let m_buf = self.mtl_device.new_buffer_with_data(
            &op.m as *const u32 as *const std::ffi::c_void,
            4,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let k_buf = self.mtl_device.new_buffer_with_data(
            &op.k as *const u32 as *const std::ffi::c_void,
            4,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let n_buf = self.mtl_device.new_buffer_with_data(
            &op.n as *const u32 as *const std::ffi::c_void,
            4,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let group_buf = self.mtl_device.new_buffer_with_data(
            &op.group_size as *const u32 as *const std::ffi::c_void,
            4,
            metal::MTLResourceOptions::StorageModeShared,
        );
        self.dispatch_precision_kernel(
            &KernelSemanticId("prism.ternary.gemm.v1".into()),
            &[
                Some(&input),
                Some(&weight_buf),
                Some(&scale),
                Some(&output),
                Some(&m_buf),
                Some(&k_buf),
                Some(&n_buf),
                Some(&group_buf),
            ],
            metal::MTLSize::new(op.m.div_ceil(16) as u64, op.n.div_ceil(16) as u64, 1),
            metal::MTLSize::new(16, 16, 1),
        )?;
        Ok(self.alloc_slot(MetalTensor {
            buffer: Some(output),
            shape: vec![op.m as i32, op.n as i32],
            dtype: DType::F16,
            generation: 0,
        }))
    }

    fn matmul_rawf32_gemv(
        &mut self,
        op: &MatmulOp,
        a: TensorHandle,
        b: TensorHandle,
    ) -> Result<TensorHandle, String> {
        if op.m != 1 {
            return Err("raw F32 Metal GEMV requires m=1".into());
        }
        let input = self
            .slot(a)?
            .buffer
            .as_ref()
            .ok_or("F32 input has no buffer")?
            .clone();
        let weights = self
            .slot(b)?
            .buffer
            .as_ref()
            .ok_or("F32 weights have no buffer")?
            .clone();
        let output = self.mtl_device.new_buffer(
            op.n as u64 * 4,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let unused = self
            .mtl_device
            .new_buffer(4, metal::MTLResourceOptions::StorageModeShared);
        let constants = MlpConstants {
            hidden_dim: op.k,
            intermediate_dim: op.n,
            group_size: 1,
            codec_id: 0,
            epsilon: 0.0,
            pad: [0; 3],
        };
        let constants_buf = self.mtl_device.new_buffer_with_data(
            &constants as *const MlpConstants as *const std::ffi::c_void,
            std::mem::size_of::<MlpConstants>() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        self.dispatch_precision_kernel(
            &KernelSemanticId("prism.linear.rawf32.v1".into()),
            &[
                Some(&input),
                Some(&weights),
                Some(&unused),
                Some(&unused),
                Some(&output),
                Some(&constants_buf),
            ],
            metal::MTLSize::new(op.n as u64, 1, 1),
            metal::MTLSize::new(64, 1, 1),
        )?;
        Ok(self.alloc_slot(MetalTensor {
            buffer: Some(output),
            shape: vec![1, op.n as i32],
            dtype: DType::F32,
            generation: 0,
        }))
    }

    /// Bind an externally-owned Metal buffer as a tensor.
    /// This is the zero-copy path — no bytes enter CPU memory.
    pub fn bind_external_buffer(
        &mut self,
        owner_token: u64,
        buffer: Arc<metal::Buffer>,
        shape: Vec<i32>,
        dtype: DType,
    ) -> TensorHandle {
        let handle = self.alloc_slot(MetalTensor {
            buffer: Some((*buffer).clone()),
            shape,
            dtype,
            generation: 0,
        });
        self.external_bindings.insert(owner_token, handle);
        handle
    }
}

impl TensorBackend for MetalBackend {
    fn create_f32(&mut self, data: &[f32], shape: &[i32]) -> Result<TensorHandle, String> {
        let buf = self.mtl_device.new_buffer_with_data(
            data.as_ptr() as *const std::ffi::c_void,
            (data.len() * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        Ok(self.alloc_slot(MetalTensor {
            buffer: Some(buf),
            shape: shape.to_vec(),
            dtype: DType::F32,
            generation: 0,
        }))
    }

    fn create_u32(&mut self, data: &[u32], shape: &[i32]) -> Result<TensorHandle, String> {
        let buf = self.mtl_device.new_buffer_with_data(
            data.as_ptr() as *const std::ffi::c_void,
            (data.len() * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        Ok(self.alloc_slot(MetalTensor {
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
        Err("MetalBackend: create_f32_from_bf16_bits not implemented".into())
    }

    fn create_owned_from_bytes(
        &mut self,
        data: &[u8],
        shape: &[i32],
        dtype: DType,
    ) -> Result<TensorHandle, String> {
        let buf = self.mtl_device.new_buffer_with_data(
            data.as_ptr() as *const std::ffi::c_void,
            data.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        Ok(self.alloc_slot(MetalTensor {
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
        let handle = self.create_owned_from_bytes(data, shape, dtype)?;
        self.external_bindings.insert(owner_token, handle);
        Ok(handle)
    }

    fn matmul(
        &mut self,
        op: &MatmulOp,
        a: TensorHandle,
        b: TensorHandle,
    ) -> Result<TensorHandle, String> {
        if op.m == 1 {
            return self.matmul_rawf32_gemv(op, a, b);
        }
        let ta = self.slot(a)?;
        let tb = self.slot(b)?;
        let a_buf = ta.buffer.as_ref().ok_or("Tensor A has no buffer")?.clone();
        let b_buf = tb.buffer.as_ref().ok_or("Tensor B has no buffer")?.clone();
        let m = op.m as usize;
        let k = op.k as usize;
        let n = op.n as usize;

        // Build MPSGraph.
        let graph = Graph::new();
        let a_ph = graph.placeholder(Some(&[m as isize, k as isize]), DataType::Float32, None);
        let b_ph = graph.placeholder(Some(&[k as isize, n as isize]), DataType::Float32, None);
        let c = graph.matrix_multiplication(&a_ph, &b_ph, None);

        // Create shared-memory output buffer.
        let out_buf = self.mtl_device.new_buffer(
            (m * n * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Shaped types for compilation.
        let a_st = ShapedType::new_with_shape_data_type(
            Some(&[m as isize, k as isize]),
            DataType::Float32,
        );
        let b_st = ShapedType::new_with_shape_data_type(
            Some(&[k as isize, n as isize]),
            DataType::Float32,
        );

        let mut compile_feeds = HashMap::new();
        compile_feeds.insert(&*a_ph, &*a_st);
        compile_feeds.insert(&*b_ph, &*b_st);

        let executable = graph.compile(
            &*Device::with_device(&self.mtl_device),
            &compile_feeds,
            &[&*c],
            None,
            None,
        );

        // Wrap MTLBuffers in TensorData.
        let a_td = TensorData::new_with_mtl_buffer(&a_buf, &[m, k], DataType::Float32, None);
        let b_td = TensorData::new_with_mtl_buffer(&b_buf, &[k, n], DataType::Float32, None);
        let out_td = TensorData::new_with_mtl_buffer(&out_buf, &[m, n], DataType::Float32, None);

        let queue = self.mtl_device.new_command_queue();

        // Synchronous execution — blocks until GPU finishes.
        let _results =
            executable.run_with_command_queue(&queue, &[&*a_td, &*b_td], Some(&[&*out_td]), None);

        Ok(self.alloc_slot(MetalTensor {
            buffer: Some(out_buf),
            shape: vec![op.m as i32, op.n as i32],
            dtype: DType::F32,
            generation: 0,
        }))
    }

    fn quantized_matmul(
        &mut self,
        op: &QuantizedMatmulOp,
        x: TensorHandle,
        w: QuantizedWeightHandle,
        scales: TensorHandle,
        biases: TensorHandle,
    ) -> Result<TensorHandle, String> {
        let semantic_id = op.kernel_semantic_id()?;
        let nf4_semantic = KernelSemanticId("prism.nf4tile640.dequant_mul.v1".into());
        if semantic_id != nf4_semantic {
            if semantic_id.0 == "prism.linear.int8.v1" {
                return self.quantized_matmul_int8(op, x, w, scales, biases);
            }
            if semantic_id.0 == "prism.ternary.gemv.v1" {
                return self.quantized_matmul_ternary_tile640(op, x, w, scales, biases);
            }
            if semantic_id.0 == "prism.ternary.gemm.v1" {
                return self.quantized_matmul_ternary_gemm(op, x, w, scales);
            }
            if semantic_id.0 == "prism.ternary.gemv.v2" {
                return self.quantized_matmul_ternary_legacy(op, x, w);
            }
            if semantic_id.0 == "prism.ternary.cimage.gemv.v1" {
                return self.quantized_matmul_ternary_cimage(op, x, w, scales);
            }
            if semantic_id.0 == "prism.q4.block_sym.gemv.v1" {
                return self.quantized_matmul_q4(op, x, w, scales);
            }
            if semantic_id.0 == "prism.palettized.gemv.v1" {
                return self.quantized_matmul_palettized(op, x, w, scales);
            }
            if semantic_id.0 == "prism.palettized.gemm.v1" {
                return self.quantized_matmul_palettized_gemm(op, x, w, scales);
            }
            return Err(format!("Metal precision kernel {} compiled, but its ABI binder is not yet wired for quantized_matmul", semantic_id.0));
        }
        let tx = self.slot(x)?;
        let w_data = self.get_weight(w)?;
        let ts = self.slot(scales)?;
        let tb = self.slot(biases)?;

        let m = op.m as usize;
        let k = op.k as usize;
        let n = op.n as usize;

        // Validate input dimensions.
        if tx.shape.len() != 2 {
            return Err("quantized_matmul: input tensor must be 2D".into());
        }
        if tx.shape[0] as u32 != op.m || tx.shape[1] as u32 != op.k {
            return Err(format!(
                "quantized_matmul: input shape ({:?}) does not match op m={} k={}",
                tx.shape, op.m, op.k
            ));
        }

        // Validate sizes for the compiler's nf4tile640 format.
        use prism_ecs_core::nf4tile640;
        let tiles_per_col = n.div_ceil(nf4tile640::TILE_ELEMENTS);
        let total_tiles = k * tiles_per_col;
        let expected_codes_size = total_tiles * nf4tile640::PACKED_BYTES_PER_TILE; // 320
        let expected_meta_count = total_tiles * nf4tile640::GROUPS_PER_TILE; // 5 per tile
        if w_data.len() != expected_codes_size {
            return Err(format!(
                "quantized_matmul: packed codes size {} != expected {} (k={} n={} tiles_per_col={})",
                w_data.len(), expected_codes_size, k, n, tiles_per_col,
            ));
        }

        // Read scale and bias data from their Metal buffers.
        let s_buf = ts.buffer.as_ref().ok_or("scales tensor has no buffer")?;
        let b_buf = tb.buffer.as_ref().ok_or("biases tensor has no buffer")?;
        let s_ptr = s_buf.contents() as *const u8;
        let b_ptr = b_buf.contents() as *const u8;
        let s_data: &[f32] =
            unsafe { std::slice::from_raw_parts(s_ptr as *const f32, expected_meta_count) };
        let b_data: &[f32] =
            unsafe { std::slice::from_raw_parts(b_ptr as *const f32, expected_meta_count) };

        if s_data.len() != expected_meta_count || b_data.len() != expected_meta_count {
            return Err(format!(
                "quantized_matmul: scale/bias count {} != expected {} (k={} n={})",
                s_data.len(),
                expected_meta_count,
                k,
                n,
            ));
        }

        // Clone data to release the immutable borrow on self before ensure_nf4_kernel borrows mutably.
        let x_buf = tx
            .buffer
            .as_ref()
            .ok_or("input tensor has no buffer")?
            .clone();
        let w_bytes = w_data.to_vec();
        let s_bytes = s_data.to_vec();
        let b_bytes = b_data.to_vec();
        // s_data and b_data borrows end above; drop slot references.
        let _ = tx;
        let _ = s_buf;
        let _ = b_buf;
        let _ = w_data;
        let _ = ts;
        let _ = tb;

        // Create MTLBuffers.
        let codes_buf = self.mtl_device.new_buffer_with_data(
            w_bytes.as_ptr() as *const std::ffi::c_void,
            w_bytes.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let scale_buf = self.mtl_device.new_buffer_with_data(
            s_bytes.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(s_bytes.as_slice()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let bias_buf = self.mtl_device.new_buffer_with_data(
            b_bytes.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(b_bytes.as_slice()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let out_buf = self.mtl_device.new_buffer(
            (m * n * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Versioned Nf4Tile640DispatchParams at buffer[5]
        let params = Nf4Tile640DispatchParams {
            abi_version: 1,
            m: m as u32,
            k: k as u32,
            n: n as u32,
            group_size: op.group_size,
            reserved: [0; 3],
        };
        let params_buf = self.mtl_device.new_buffer_with_data(
            &params as *const Nf4Tile640DispatchParams as *const std::ffi::c_void,
            std::mem::size_of::<Nf4Tile640DispatchParams>() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        self.dispatch_precision_kernel(
            &nf4_semantic,
            &[
                Some(&codes_buf),
                Some(&scale_buf),
                Some(&bias_buf),
                Some(&x_buf),
                Some(&out_buf),
                Some(&params_buf),
            ],
            metal::MTLSize::new(n as u64, m as u64, 1),
            metal::MTLSize::new(16, 1, 1),
        )?;

        Ok(self.alloc_slot(MetalTensor {
            buffer: Some(out_buf),
            shape: vec![op.m as i32, op.n as i32],
            dtype: DType::F32,
            generation: 0,
        }))
    }
    fn rms_norm(
        &mut self,
        op: &RmsNormOp,
        x: TensorHandle,
        weight: TensorHandle,
    ) -> Result<TensorHandle, String> {
        let tx = self.slot(x)?;
        let tw = self.slot(weight)?;
        let x_buf = tx
            .buffer
            .as_ref()
            .ok_or("Input tensor has no buffer")?
            .clone();
        let w_buf = tw
            .buffer
            .as_ref()
            .ok_or("Weight tensor has no buffer")?
            .clone();
        let dim = op.dim as usize;
        let total: usize = tx.shape.iter().map(|&d| d as usize).product();
        let batch = total / dim;
        let out_shape = tx.shape.clone();

        // Build MPSGraph.
        let graph = Graph::new();
        let x_ph = graph.placeholder(
            Some(&[batch as isize, dim as isize]),
            DataType::Float32,
            None,
        );
        let w_ph = graph.placeholder(Some(&[dim as isize]), DataType::Float32, None);

        // Compute mean(x^2) via matmul with ones: [batch,dim] @ [dim,1] = [batch,1].
        // We use matmul because mpsgraph v0.2 does not wrap reductionSumWithTensor.
        let x_sq = graph.square(&x_ph, None);
        let ones = graph.constant_with_scalar(1.0_f64, Some(&[dim, 1usize]), DataType::Float32);
        let sum_sq = graph.matrix_multiplication(&x_sq, &ones, None);
        let dim_c =
            graph.constant_with_scalar(dim as f64, Some(&[1usize, 1usize]), DataType::Float32);
        let mean_sq = graph.division(&sum_sq, &dim_c, None);

        // Add epsilon.
        let eps_c =
            graph.constant_with_scalar(op.eps as f64, Some(&[1usize, 1usize]), DataType::Float32);
        let mean_eps = graph.addition(&mean_sq, &eps_c, None);

        // sqrt -> divide -> multiply by weight.
        let rms = graph.square_root(&mean_eps, None);
        let normalized = graph.division(&x_ph, &rms, None);
        let result = graph.multiplication(&normalized, &w_ph, None);

        // Create shared-memory output buffer.
        let out_buf = self.mtl_device.new_buffer(
            (total * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Shaped types.
        let x_st = ShapedType::new_with_shape_data_type(
            Some(&[batch as isize, dim as isize]),
            DataType::Float32,
        );
        let w_st = ShapedType::new_with_shape_data_type(Some(&[dim as isize]), DataType::Float32);

        let mut compile_feeds = HashMap::new();
        compile_feeds.insert(&*x_ph, &*x_st);
        compile_feeds.insert(&*w_ph, &*w_st);

        let executable = graph.compile(
            &*Device::with_device(&self.mtl_device),
            &compile_feeds,
            &[&*result],
            None,
            None,
        );

        // MTLBuffer wrapper TensorData.
        let x_td = TensorData::new_with_mtl_buffer(&x_buf, &[batch, dim], DataType::Float32, None);
        let w_td = TensorData::new_with_mtl_buffer(&w_buf, &[dim], DataType::Float32, None);
        let out_td =
            TensorData::new_with_mtl_buffer(&out_buf, &[batch, dim], DataType::Float32, None);

        let queue = self.mtl_device.new_command_queue();
        let _results =
            executable.run_with_command_queue(&queue, &[&*x_td, &*w_td], Some(&[&*out_td]), None);

        Ok(self.alloc_slot(MetalTensor {
            buffer: Some(out_buf),
            shape: out_shape,
            dtype: DType::F32,
            generation: 0,
        }))
    }

    fn rope(&mut self, op: &RoPEOp, x: TensorHandle) -> Result<TensorHandle, String> {
        let tx = self.slot(x)?;
        let shape = tx.shape.clone();
        let head_dim = op.head_dim as usize;
        let total: usize = shape.iter().map(|&d| d as usize).product();
        let x_buf = tx.buffer.as_ref().ok_or("Input tensor has no buffer")?;
        let x_ptr = x_buf.contents() as *const u8;
        let x_f32: &[f32] = unsafe { std::slice::from_raw_parts(x_ptr as *const f32, total) };

        if head_dim == 0 || total % head_dim != 0 {
            return Err(format!(
                "rope: head_dim {head_dim} does not divide total elements {total}"
            ));
        }

        // CPU-fallback RoPE: for each leading dimension and each pair (x_{2i}, x_{2i+1})
        // within head_dim, apply the rotary rotation.
        let leading = total / head_dim;
        let offset = op.positions.first().copied().unwrap_or(0) as f32;
        let base: f32 = 10000.0;
        let half = head_dim / 2;

        let mut out = x_f32.to_vec();
        for l in 0..leading {
            let p = offset + l as f32;
            for i in 0..half {
                let theta = p / base.powf(2.0 * i as f32 / head_dim as f32);
                let cos = theta.cos();
                let sin = theta.sin();
                let idx = l * head_dim;
                let x1 = out[idx + 2 * i];
                let x2 = out[idx + 2 * i + 1];
                out[idx + 2 * i] = x1 * cos - x2 * sin;
                out[idx + 2 * i + 1] = x1 * sin + x2 * cos;
            }
        }

        let out_buf = self.mtl_device.new_buffer_with_data(
            out.as_ptr() as *const std::ffi::c_void,
            (out.len() * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        Ok(self.alloc_slot(MetalTensor {
            buffer: Some(out_buf),
            shape,
            dtype: DType::F32,
            generation: 0,
        }))
    }

    fn add(&mut self, a: TensorHandle, b: TensorHandle) -> Result<TensorHandle, String> {
        let ta = self.slot(a)?;
        let tb = self.slot(b)?;
        let a_buf = ta.buffer.as_ref().ok_or("Tensor A has no buffer")?;
        let b_buf = tb.buffer.as_ref().ok_or("Tensor B has no buffer")?;
        let a_ptr = a_buf.contents() as *const u8;
        let b_ptr = b_buf.contents() as *const u8;
        let a_len = a_buf.length() as usize;
        let b_len = b_buf.length() as usize;
        let a_f32: &[f32] = unsafe { std::slice::from_raw_parts(a_ptr as *const f32, a_len / 4) };
        let b_f32: &[f32] = unsafe { std::slice::from_raw_parts(b_ptr as *const f32, b_len / 4) };
        let sum: Vec<f32> = a_f32.iter().zip(b_f32.iter()).map(|(x, y)| x + y).collect();
        let out_buf = self.mtl_device.new_buffer_with_data(
            sum.as_ptr() as *const std::ffi::c_void,
            (sum.len() * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        Ok(self.alloc_slot(MetalTensor {
            buffer: Some(out_buf),
            shape: ta.shape.clone(),
            dtype: DType::F32,
            generation: 0,
        }))
    }

    fn multiply(&mut self, a: TensorHandle, b: TensorHandle) -> Result<TensorHandle, String> {
        let ta = self.slot(a)?;
        let tb = self.slot(b)?;
        let a_buf = ta.buffer.as_ref().ok_or("Tensor A has no buffer")?;
        let b_buf = tb.buffer.as_ref().ok_or("Tensor B has no buffer")?;
        let a_ptr = a_buf.contents() as *const u8;
        let b_ptr = b_buf.contents() as *const u8;
        let a_len = a_buf.length() as usize;
        let b_len = b_buf.length() as usize;
        let a_f32: &[f32] = unsafe { std::slice::from_raw_parts(a_ptr as *const f32, a_len / 4) };
        let b_f32: &[f32] = unsafe { std::slice::from_raw_parts(b_ptr as *const f32, b_len / 4) };
        let prod: Vec<f32> = a_f32.iter().zip(b_f32.iter()).map(|(x, y)| x * y).collect();
        let out_buf = self.mtl_device.new_buffer_with_data(
            prod.as_ptr() as *const std::ffi::c_void,
            (prod.len() * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        Ok(self.alloc_slot(MetalTensor {
            buffer: Some(out_buf),
            shape: ta.shape.clone(),
            dtype: DType::F32,
            generation: 0,
        }))
    }

    fn silu(&mut self, x: TensorHandle) -> Result<TensorHandle, String> {
        let tx = self.slot(x)?;
        let x_buf = tx
            .buffer
            .as_ref()
            .ok_or("Input tensor has no buffer")?
            .clone();
        let shape: Vec<isize> = tx.shape.iter().map(|&d| d as isize).collect();
        let shape_usize: Vec<usize> = tx.shape.iter().map(|&d| d as usize).collect();
        let out_shape = tx.shape.clone();
        let count: usize = shape_usize.iter().product();
        // Build MPSGraph: SiLU(x) = x * sigmoid(x)
        let graph = Graph::new();
        let x_ph = graph.placeholder(Some(&shape), DataType::Float32, None);
        let sig = graph.sigmoid(&x_ph, None);
        let result = graph.multiplication(&x_ph, &sig, None);

        let out_buf = self.mtl_device.new_buffer(
            (count * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let x_st = ShapedType::new_with_shape_data_type(Some(&shape), DataType::Float32);
        let mut compile_feeds = HashMap::new();
        compile_feeds.insert(&*x_ph, &*x_st);

        let executable = graph.compile(
            &*Device::with_device(&self.mtl_device),
            &compile_feeds,
            &[&*result],
            None,
            None,
        );

        let x_td = TensorData::new_with_mtl_buffer(&x_buf, &shape_usize, DataType::Float32, None);
        let out_td =
            TensorData::new_with_mtl_buffer(&out_buf, &shape_usize, DataType::Float32, None);

        let queue = self.mtl_device.new_command_queue();
        let _results =
            executable.run_with_command_queue(&queue, &[&*x_td], Some(&[&*out_td]), None);

        Ok(self.alloc_slot(MetalTensor {
            buffer: Some(out_buf),
            shape: out_shape,
            dtype: DType::F32,
            generation: 0,
        }))
    }

    fn transpose(&mut self, _x: TensorHandle, _dims: &[i32]) -> Result<TensorHandle, String> {
        Err("MetalBackend: transpose not implemented".into())
    }

    fn reshape(&mut self, x: TensorHandle, shape: &[i32]) -> Result<TensorHandle, String> {
        let tensor = self.slot(x)?;
        let buf = tensor
            .buffer
            .as_ref()
            .ok_or("Tensor has no buffer")?
            .clone();
        let dtype = tensor.dtype;
        Ok(self.alloc_slot(MetalTensor {
            buffer: Some(buf),
            shape: shape.to_vec(),
            dtype,
            generation: 0,
        }))
    }

    fn softmax(&mut self, x: TensorHandle, axis: i32) -> Result<TensorHandle, String> {
        let tx = self.slot(x)?;
        let x_buf = tx
            .buffer
            .as_ref()
            .ok_or("Input tensor has no buffer")?
            .clone();
        let shape: Vec<isize> = tx.shape.iter().map(|&d| d as isize).collect();
        let shape_usize: Vec<usize> = tx.shape.iter().map(|&d| d as usize).collect();
        let out_shape = tx.shape.clone();
        let count: usize = shape_usize.iter().product();
        // Build MPSGraph with native soft_max.
        let graph = Graph::new();
        let x_ph = graph.placeholder(Some(&shape), DataType::Float32, None);
        let result = graph.soft_max(&x_ph, axis as i64, None);

        let out_buf = self.mtl_device.new_buffer(
            (count * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let x_st = ShapedType::new_with_shape_data_type(Some(&shape), DataType::Float32);
        let mut compile_feeds = HashMap::new();
        compile_feeds.insert(&*x_ph, &*x_st);

        let executable = graph.compile(
            &*Device::with_device(&self.mtl_device),
            &compile_feeds,
            &[&*result],
            None,
            None,
        );

        let x_td = TensorData::new_with_mtl_buffer(&x_buf, &shape_usize, DataType::Float32, None);
        let out_td =
            TensorData::new_with_mtl_buffer(&out_buf, &shape_usize, DataType::Float32, None);

        let queue = self.mtl_device.new_command_queue();
        let _results =
            executable.run_with_command_queue(&queue, &[&*x_td], Some(&[&*out_td]), None);

        Ok(self.alloc_slot(MetalTensor {
            buffer: Some(out_buf),
            shape: out_shape,
            dtype: DType::F32,
            generation: 0,
        }))
    }

    fn index_select(
        &mut self,
        _x: TensorHandle,
        _indices: &[u32],
        _axis: i32,
    ) -> Result<TensorHandle, String> {
        Err("MetalBackend: index_select not implemented".into())
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

    fn submit_compute(
        &mut self,
        group_id: u64,
        outputs: &[TensorHandle],
    ) -> Result<ComputationToken, String> {
        self.evaluate(group_id, outputs)?;

        // Create a Metal command buffer with an async completion handler.
        let queue = self.mtl_device.new_command_queue();
        let cmd_buf = queue.new_command_buffer();

        // Create a completion token backed by the MTLCommandBuffer.
        // The completion handler will fire when the GPU finishes.
        let token = ComellationToken::from_command_buffer(cmd_buf);
        cmd_buf.commit();

        Ok(ComputationToken::Metal(token))
    }

    /// Evaluate outputs into a pre-allocated arena (IOSurface-backed).
    /// Creates a Metal command buffer with a blit encoder for memory
    /// synchronization, commits, and waits for completion.
    #[cfg(any(feature = "mlx-backend", feature = "prism-backend"))]
    fn evaluate_into(
        &mut self,
        group_id: u64,
        outputs: &[TensorHandle],
        _arena: &tribunus_compute_core::arena::Arena,
    ) -> Result<EvaluationReceipt, String> {
        let receipt = self.evaluate(group_id, outputs)?;

        // Create a command buffer with a blit encoder to synchronise memory.
        let queue = self.mtl_device.new_command_queue();
        let cmd_buf = queue.new_command_buffer();
        let blit_enc = cmd_buf.new_blit_command_encoder();
        blit_enc.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        Ok(receipt)
    }

    fn read_f32(&mut self, handle: TensorHandle) -> Result<ReadbackReceipt, String> {
        let tensor = self.slot(handle)?;
        let buf = tensor
            .buffer
            .as_ref()
            .ok_or("read_f32: tensor has no buffer")?;
        let count: usize = tensor.shape.iter().map(|&d| d as usize).product();
        let ptr = buf.contents() as *const f32;
        let data: Vec<f32> = unsafe { std::slice::from_raw_parts(ptr, count) }.to_vec();
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
                "invalid TensorHandle({}, {})",
                handle.slot, handle.generation
            ))
        }
    }

    fn active_memory(&self) -> (u64, u64) {
        (0, 0)
    }

    fn backend_capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            can_gpu: true,
            can_cpu: false,
            supports_quantized: true,
            supports_bf16_native: false,
            backend_name: "metal".into(),
        }
    }
}

impl BackendInstance for MetalBackend {
    fn backend_kind(&self) -> BackendId {
        BACKEND_METAL
    }
    fn supports(&self, family: OperationFamily) -> bool {
        matches!(
            family,
            OperationFamily::Matmul
                | OperationFamily::QuantizedMatmul
                | OperationFamily::RmsNorm
                | OperationFamily::Softmax
                | OperationFamily::Silu
                | OperationFamily::RoPE
        )
    }

    fn execute(
        &mut self,
        op: &OperationDescriptor,
        inputs: &[TensorHandle],
    ) -> Result<BackendExecutionReceipt, String> {
        let start = std::time::Instant::now();
        match op.family {
            OperationFamily::Matmul => {
                let m = op.logical_shape.dims.first().copied().unwrap_or(1);
                let n = op.logical_shape.dims.get(1).copied().unwrap_or(1);
                let k = op.logical_shape.dims.get(2).copied().unwrap_or(1);
                let mo = MatmulOp { m, n, k };
                let a = *inputs.first().ok_or_else(|| "need 2 inputs")?;
                let b = *inputs.get(1).ok_or_else(|| "need 2 inputs")?;
                self.matmul(&mo, a, b)?;
            }
            OperationFamily::RmsNorm => {
                let x = *inputs.first().ok_or_else(|| "need 2 inputs")?;
                let w = *inputs.get(1).ok_or_else(|| "need 2 inputs")?;
                let tx = self.slot(x)?;
                let dim = tx.shape.last().copied().unwrap_or(1) as u32;
                let rms_op = RmsNormOp { dim, eps: 1e-5 };
                self.rms_norm(&rms_op, x, w)?;
            }
            OperationFamily::Softmax => {
                let x = *inputs.first().ok_or_else(|| "need 1 input")?;
                let tx = self.slot(x)?;
                let axis = (tx.shape.len() as i32 - 1).max(0);
                self.softmax(x, axis)?;
            }
            OperationFamily::Silu => {
                let x = *inputs.first().ok_or_else(|| "need 1 input")?;
                self.silu(x)?;
            }
            OperationFamily::RoPE => {
                let x = *inputs.first().ok_or_else(|| "need 1 input")?;
                let tx = self.slot(x)?;
                let head_dim = tx.shape.last().copied().unwrap_or(64) as u32;
                let rope_op = RoPEOp {
                    head_dim,
                    positions: vec![0],
                };
                self.rope(&rope_op, x)?;
            }
            _ => {
                return Err(format!("MetalBackend: unsupported {:?}", op.family));
            }
        }
        let elapsed = start.elapsed().as_nanos() as u64;
        Ok(BackendExecutionReceipt {
            operation_id: op.operation_id,
            backend_id: BACKEND_METAL,
            backend_version: BackendVersion {
                backend_name: "metal".into(),
                version: "0.1".into(),
                git_commit: None,
            },
            requested_substrate: None,
            observed_substrate: None,
            graph_build_ns: None,
            compile_ns: None,
            queue_wait_ns: None,
            submit_ns: Some(elapsed),
            execution_ns: Some(elapsed),
            synchronization_ns: None,
            total_wall_ns: elapsed,
            bytes_read: None,
            bytes_written: None,
            temporary_bytes: None,
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

#[cfg(test)]
mod shadow_test {
    use super::*;
    use crate::routing::{
        CorrectnessCheckpointPolicy, LogicalShape, OperationId, Phase, PhysicalLayout, TensorShape,
    };

    #[test]
    fn generated_nf4_runtime_matches_reference() {
        let Ok(mut backend) = MetalBackend::new() else {
            return;
        };
        let input = backend.create_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
        let weights = backend.alloc_weight(vec![0x77; 4 * 320]);
        let scales = backend.create_f32(&[0.0; 20], &[4, 5]).unwrap();
        let biases = backend.create_f32(&[1.0; 20], &[4, 5]).unwrap();

        let output = backend
            .dispatch_linear_nf4_cimage(input, weights, scales, biases, 4, 4, 128)
            .unwrap();
        let result = backend.read_f32(output).unwrap().data;
        assert_eq!(result.len(), 4);
        for (index, value) in result.iter().enumerate() {
            assert!((value - 10.0).abs() < 1e-4, "output[{index}] = {value}");
        }
    }

    #[test]
    fn generated_int8_runtime_matches_reference() {
        let Ok(mut backend) = MetalBackend::new() else {
            return;
        };
        let input = backend.create_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
        let weights = backend.alloc_weight(vec![1i8 as u8; 4 * 4]);
        let scales = backend.create_f32(&[2.0; 4], &[4]).unwrap();
        let biases = backend.create_f32(&[0.0; 4], &[4]).unwrap();
        let op = QuantizedMatmulOp {
            m: 1,
            n: 4,
            k: 4,
            input_dtype: DType::F32,
            weight_dtype: DType::I8,
            scale_dtype: DType::F32,
            bias_dtype: DType::F32,
            output_dtype: DType::F32,
            group_size: 1,
            bits: 8,
            transpose: false,
        };

        let output = backend
            .quantized_matmul_int8(&op, input, weights, scales, biases)
            .unwrap();
        let result = backend.read_f32(output).unwrap().data;
        assert_eq!(result, vec![20.0; 4]);
    }

    #[test]
    fn generated_q4_runtime_matches_reference() {
        let Ok(mut backend) = MetalBackend::new() else {
            return;
        };
        let half_tensor = |backend: &mut MetalBackend, bits: &[u16], shape: &[i32]| {
            let buffer = backend.mtl_device.new_buffer_with_data(
                bits.as_ptr() as *const std::ffi::c_void,
                (bits.len() * 2) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            backend.alloc_slot(MetalTensor {
                buffer: Some(buffer),
                shape: shape.to_vec(),
                dtype: DType::F16,
                generation: 0,
            })
        };
        let input = half_tensor(&mut backend, &[0x3c00; 8], &[1, 8]);
        let weights = backend.alloc_weight(vec![0x99; 4]);
        let scales = half_tensor(&mut backend, &[0x4000], &[1]);
        let op = QuantizedMatmulOp {
            m: 1,
            n: 1,
            k: 8,
            input_dtype: DType::F16,
            weight_dtype: DType::U8,
            scale_dtype: DType::F16,
            bias_dtype: DType::F16,
            output_dtype: DType::F16,
            group_size: 8,
            bits: 4,
            transpose: false,
        };

        let output = backend
            .quantized_matmul_q4(&op, input, weights, scales)
            .unwrap();
        let tensor = backend.slot(output).unwrap();
        let bits = unsafe { *(tensor.buffer.as_ref().unwrap().contents() as *const u16) };
        assert_eq!(
            bits, 0x4c00,
            "expected half-precision 16.0, got {bits:#06x}"
        );
    }

    #[test]
    fn generated_ternary_runtime_matches_reference() {
        let Ok(mut backend) = MetalBackend::new() else {
            return;
        };
        let half_tensor = |backend: &mut MetalBackend, bits: &[u16], shape: &[i32]| {
            let buffer = backend.mtl_device.new_buffer_with_data(
                bits.as_ptr() as *const std::ffi::c_void,
                (bits.len() * 2) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            backend.alloc_slot(MetalTensor {
                buffer: Some(buffer),
                shape: shape.to_vec(),
                dtype: DType::F16,
                generation: 0,
            })
        };
        let input = half_tensor(&mut backend, &[0x3c00; 4], &[1, 4]);
        let op = QuantizedMatmulOp {
            m: 1,
            n: 1,
            k: 4,
            input_dtype: DType::F16,
            weight_dtype: DType::U8,
            scale_dtype: DType::F16,
            bias_dtype: DType::F16,
            output_dtype: DType::F16,
            group_size: 4,
            bits: 2,
            transpose: false,
        };
        let legacy_weights = backend.alloc_weight(vec![0x55]);
        let legacy = backend
            .quantized_matmul_ternary_legacy(&op, input, legacy_weights)
            .unwrap();
        let legacy_bits = unsafe {
            *(backend
                .slot(legacy)
                .unwrap()
                .buffer
                .as_ref()
                .unwrap()
                .contents() as *const u16)
        };
        assert_eq!(legacy_bits, 0x4400);

        let input = half_tensor(&mut backend, &[0x3c00; 4], &[1, 4]);
        let scales = half_tensor(&mut backend, &[0x3c00], &[1]);
        let cimage_weights = backend.alloc_weight(vec![0xaa]);
        let cimage = backend
            .quantized_matmul_ternary_cimage(
                &QuantizedMatmulOp {
                    group_size: 4,
                    ..op
                },
                input,
                cimage_weights,
                scales,
            )
            .unwrap();
        let cimage_bits = unsafe {
            *(backend
                .slot(cimage)
                .unwrap()
                .buffer
                .as_ref()
                .unwrap()
                .contents() as *const u16)
        };
        assert_eq!(cimage_bits, 0x4400);
    }

    #[test]
    fn generated_palettized_runtime_matches_reference() {
        let Ok(mut backend) = MetalBackend::new() else {
            return;
        };
        let half_tensor = |backend: &mut MetalBackend, bits: &[u16], shape: &[i32]| {
            let buffer = backend.mtl_device.new_buffer_with_data(
                bits.as_ptr() as *const std::ffi::c_void,
                (bits.len() * 2) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            backend.alloc_slot(MetalTensor {
                buffer: Some(buffer),
                shape: shape.to_vec(),
                dtype: DType::F16,
                generation: 0,
            })
        };
        let input = half_tensor(&mut backend, &[0x3c00; 8], &[1, 8]);
        let codebook = half_tensor(&mut backend, &[0x4000; 16], &[1, 16]);
        let indices = backend.alloc_weight(vec![0x00; 4]);
        let op = QuantizedMatmulOp {
            m: 1,
            n: 1,
            k: 8,
            input_dtype: DType::F16,
            weight_dtype: DType::U8,
            scale_dtype: DType::F16,
            bias_dtype: DType::F16,
            output_dtype: DType::F16,
            group_size: 16,
            bits: 4,
            transpose: false,
        };
        let output = backend
            .quantized_matmul_palettized(&op, input, indices, codebook)
            .unwrap();
        let bits = unsafe {
            *(backend
                .slot(output)
                .unwrap()
                .buffer
                .as_ref()
                .unwrap()
                .contents() as *const u16)
        };
        assert_eq!(bits, 0x4c00);
    }

    #[test]
    fn generated_nf4_tile640_runtime_matches_reference() {
        let Ok(mut backend) = MetalBackend::new() else {
            return;
        };
        let input = backend.create_f32(&[2.0], &[1, 1]).unwrap();
        let weights = backend.alloc_weight(vec![0x77; 320]);
        let scales = backend.create_f32(&[0.0; 5], &[1, 5]).unwrap();
        let biases = backend.create_f32(&[1.0; 5], &[1, 5]).unwrap();
        let op = QuantizedMatmulOp {
            m: 1,
            n: 640,
            k: 1,
            input_dtype: DType::F32,
            weight_dtype: DType::U8,
            scale_dtype: DType::F32,
            bias_dtype: DType::F32,
            output_dtype: DType::F32,
            group_size: 128,
            bits: 4,
            transpose: false,
        };
        let output = backend
            .quantized_matmul(&op, input, weights, scales, biases)
            .unwrap();
        let result = backend.read_f32(output).unwrap().data;
        assert_eq!(result.len(), 640);
        for (index, value) in result.iter().enumerate() {
            assert!((*value - 2.0).abs() < 1e-4, "output[{index}] = {value}");
        }
    }

    #[test]
    fn generated_palettized_gemm_and_swiglu_match_reference() {
        let Ok(mut backend) = MetalBackend::new() else {
            return;
        };
        let half_tensor = |backend: &mut MetalBackend, bits: &[u16], shape: &[i32]| {
            let buffer = backend.mtl_device.new_buffer_with_data(
                bits.as_ptr() as *const std::ffi::c_void,
                (bits.len() * 2) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            backend.alloc_slot(MetalTensor {
                buffer: Some(buffer),
                shape: shape.to_vec(),
                dtype: DType::F16,
                generation: 0,
            })
        };
        let half_to_f32 = |bits: u16| {
            let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
            let exponent = ((bits >> 10) & 0x1f) as i32;
            let mantissa = (bits & 0x03ff) as u32;
            if exponent == 0 {
                sign * (mantissa as f32 / 1024.0) * 2f32.powi(-14)
            } else {
                sign * (1.0 + mantissa as f32 / 1024.0) * 2f32.powi(exponent - 15)
            }
        };

        let input = half_tensor(&mut backend, &[0x3c00; 2], &[1, 2]);
        let codebook = half_tensor(&mut backend, &[0x4000; 16], &[1, 16]);
        let indices = backend.alloc_weight(vec![0]);
        let gemm_op = QuantizedMatmulOp {
            m: 1,
            n: 1,
            k: 2,
            input_dtype: DType::F16,
            weight_dtype: DType::U8,
            scale_dtype: DType::F16,
            bias_dtype: DType::F16,
            output_dtype: DType::F16,
            group_size: 16,
            bits: 4,
            transpose: false,
        };
        let gemm = backend
            .quantized_matmul_palettized_gemm(&gemm_op, input, indices, codebook)
            .unwrap();
        let gemm_bits = unsafe {
            *(backend
                .slot(gemm)
                .unwrap()
                .buffer
                .as_ref()
                .unwrap()
                .contents() as *const u16)
        };
        assert!((half_to_f32(gemm_bits) - 4.0).abs() < 0.01);

        let input = half_tensor(&mut backend, &[0x3c00; 2], &[1, 2]);
        let gate_codebook = half_tensor(&mut backend, &[0x3c00; 16], &[1, 16]);
        let up_codebook = half_tensor(&mut backend, &[0x3c00; 16], &[1, 16]);
        let gate_weights = backend.alloc_weight(vec![0]);
        let up_weights = backend.alloc_weight(vec![0]);
        let swiglu = backend
            .dispatch_palettized_swiglu(
                input,
                gate_weights,
                gate_codebook,
                up_weights,
                up_codebook,
                2,
                1,
            )
            .unwrap();
        let swiglu_bits = unsafe {
            *(backend
                .slot(swiglu)
                .unwrap()
                .buffer
                .as_ref()
                .unwrap()
                .contents() as *const u16)
        };
        let swiglu_value = half_to_f32(swiglu_bits);
        let expected_swiglu = 2.0 * (1.0 / (1.0 + (-2.0f32).exp())) * 2.0;
        assert!(
            (swiglu_value - expected_swiglu).abs() < 0.01,
            "SwiGLU output = {swiglu_value}, expected {expected_swiglu}"
        );
    }

    #[test]
    fn every_generated_precision_target_dispatches_on_metal() {
        let Ok(mut backend) = MetalBackend::new() else {
            return;
        };
        let targets = [
            "prism.linear.rawf32.v1",
            "prism.linear.nf4.v1",
            "prism.linear.int8.v1",
            "prism.ternary.gemv.v1",
            "prism.ternary.gemv.v2",
            "prism.ternary.gemm.v1",
            "prism.nf4.tile640.gemv.v1",
            "prism.nf4tile640.dequant_mul.v1",
            "prism.q4.block_sym.gemv.v1",
            "prism.palettized.gemv.v1",
            "prism.palettized.swiglu.v1",
            "prism.palettized.gemm.v1",
            "prism.ternary.cimage.gemv.v1",
        ];
        for semantic_id in targets {
            let buffers: Vec<metal::Buffer> = (0..10)
                .map(|_| {
                    backend
                        .mtl_device
                        .new_buffer(1024 * 1024, metal::MTLResourceOptions::StorageModeShared)
                })
                .collect();
            let bindings: Vec<Option<&metal::Buffer>> = buffers.iter().map(Some).collect();
            backend
                .dispatch_precision_kernel(
                    &KernelSemanticId(semantic_id.into()),
                    &bindings,
                    metal::MTLSize::new(1, 1, 1),
                    metal::MTLSize::new(64, 1, 1),
                )
                .unwrap_or_else(|error| panic!("{semantic_id} failed Metal dispatch: {error}"));
        }
    }

    /// NB: These tests are ignored because `mpsgraph` v0.2.0 passes
    /// `MTLCommandQueue*` as a raw pointer (`^v`) where the ObjC runtime
    /// on this macOS version expects an ObjC object reference (`@`).
    /// The crate's upstream repo (mirai-audio/mpsgraph-rs) is 404 — no fix
    /// is available. Re-enable when mpsgraph is upgraded to a version that
    /// uses `objc2`-compatible type encoding.
    #[ignore = "mpsgraph 0.2.0 FFI: MTLCommandQueue* encod as ^v, runtime expects @"]
    #[test]
    fn shadow_matmul_matches_cpu() {
        let mut backend = MetalBackend::new().expect("Metal device should be available");
        let m = 4u32;
        let k = 4u32;
        let n = 4u32;
        let a_data: Vec<f32> = (0..(m * k)).map(|i| i as f32).collect();
        let b_data: Vec<f32> = (0..(k * n)).map(|i| (i as f32) * 2.0).collect();
        let a = backend.create_f32(&a_data, &[m as i32, k as i32]).unwrap();
        let b = backend.create_f32(&b_data, &[k as i32, n as i32]).unwrap();
        let out = backend.matmul(&MatmulOp { m, n, k }, a, b).unwrap();
        let metal_result = &backend.read_f32(out).unwrap().data;

        let mut cpu_ref = vec![0.0f32; (m * n) as usize];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for kk in 0..k {
                    s += a_data[(i * k + kk) as usize] * b_data[(kk * n + j) as usize];
                }
                cpu_ref[(i * n + j) as usize] = s;
            }
        }
        assert_eq!(metal_result.len(), cpu_ref.len());
        for (i, (&m, &c)) in metal_result.iter().zip(cpu_ref.iter()).enumerate() {
            assert!((m - c).abs() < 1e-4, "Mismatch at {i}: Metal={m} CPU={c}");
        }
    }

    #[test]
    #[ignore = "mpsgraph 0.2.0 FFI: MTLCommandQueue* encod as ^v, runtime expects @"]
    fn shadow_execute_through_backend_instance() {
        let m = 4u32;
        let mut backend = MetalBackend::new().expect("Metal device should be available");
        let k = 4u32;
        let n = 4u32;
        let a_data: Vec<f32> = (0..(m * k)).map(|i| i as f32).collect();
        let b_data: Vec<f32> = (0..(k * n)).map(|i| (i as f32) * 2.0).collect();
        let a = backend.create_f32(&a_data, &[m as i32, k as i32]).unwrap();
        let b = backend.create_f32(&b_data, &[k as i32, n as i32]).unwrap();
        let op = OperationDescriptor {
            operation_id: OperationId(1),
            family: OperationFamily::Matmul,
            layer_index: None,
            phase: Phase::Decode,
            logical_shape: LogicalShape {
                dims: vec![m, n, k],
            },
            physical_layout: PhysicalLayout::RowMajor,
            input_dtypes: vec![DType::F32, DType::F32],
            output_dtype: DType::F32,
            quantization: None,
            expected_output_shape: TensorShape { dims: vec![m, n] },
            correctness_checkpoint: CorrectnessCheckpointPolicy::None,
        };
        let receipt = backend
            .execute(&op, &[a, b])
            .expect("execute should succeed");
        assert!(
            receipt.total_wall_ns > 0,
            "execution should take measurable time"
        );
    }
}
