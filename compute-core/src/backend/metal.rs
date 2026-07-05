//! Metal backend — production [`TensorBackend`] + [`BackendInstance`]
//! using `mpsgraph` (Apple Metal Performance Shaders Graph).
//!
//! Gated behind `target_os = "macos"` — Linux builds are unaffected.

#![cfg(target_os = "macos")]

use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::Mutex;

use mpsgraph::{DataType, Device, Graph, ShapedType, TensorData};
use crate::backend::completion::ComellationToken;

use crate::backend::heterogeneous_executor::BackendInstance;
use crate::backend::routing::{
    BackendExecutionReceipt, BackendId, BackendVersion, OperationDescriptor, OperationFamily,
    BACKEND_METAL,
};
use crate::backend::{
    BackendCapabilities, DType, EvaluationReceipt, MatmulOp, QuantizedMatmulOp,
    QuantizedWeightHandle, ReadbackReceipt, RmsNormOp, RoPEOp, TensorBackend, TensorHandle,
};

/// One live tensor stored in the Metal backend.
struct MetalTensor {
    data: Vec<u8>,
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
    /// Cached compute pipeline state for the dequant_mul_nf4tile640 kernel.
    nf4_pipeline: Option<metal::ComputePipelineState>,
    /// The Metal library must outlive the pipeline state.
    nf4_library: Option<metal::Library>,
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
            nf4_pipeline: None,
            nf4_library: None,
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

    #[allow(dead_code)]
    /// Register a quantized weight blob (packed nf4tile640 bytes).
    fn alloc_weight(&mut self, data: Vec<u8>) -> QuantizedWeightHandle {
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

    /// Lazily compile the NF4 dequantize+matmul kernel and cache the pipeline.
    fn ensure_nf4_kernel(&mut self) -> Result<(), String> {
        if self.nf4_pipeline.is_some() {
            return Ok(());
        }
        let src = include_str!("../../shaders/nf4tile640.metal");
        let lib = self
            .mtl_device
            .new_library_with_source(src, &metal::CompileOptions::new())
            .map_err(|e| format!("Metal library compile failed: {e}"))?;
        let kernel = lib
            .get_function("dequant_mul_nf4tile640", None::<metal::FunctionConstantValues>)
            .map_err(|e| format!("Metal kernel not found: {e}"))?;
        let pipeline = self
            .mtl_device
            .new_compute_pipeline_state_with_function(&kernel)
            .map_err(|e| format!("Metal pipeline state creation failed: {e}"))?;
        self.nf4_library = Some(lib);
        self.nf4_pipeline = Some(pipeline);
        Ok(())
    }
}

impl TensorBackend for MetalBackend {
    fn create_f32(&mut self, data: &[f32], shape: &[i32]) -> Result<TensorHandle, String> {
        Ok(self.alloc_slot(MetalTensor {
            data: bytemuck::cast_slice(data).to_vec(),
            shape: shape.to_vec(),
            dtype: DType::F32,
            generation: 0,
        }))
    }

    fn create_u32(&mut self, data: &[u32], shape: &[i32]) -> Result<TensorHandle, String> {
        Ok(self.alloc_slot(MetalTensor {
            data: bytemuck::cast_slice(data).to_vec(),
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
        Ok(self.alloc_slot(MetalTensor {
            data: data.to_vec(),
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
        let handle = self.alloc_slot(MetalTensor {
            data: data.to_vec(),
            shape: shape.to_vec(),
            dtype,
            generation: 0,
        });
        self.external_bindings.insert(owner_token, handle);
        Ok(handle)
    }

    fn matmul(
        &mut self,
        op: &MatmulOp,
        a: TensorHandle,
        b: TensorHandle,
    ) -> Result<TensorHandle, String> {
        let ta = self.slot(a)?;
        let tb = self.slot(b)?;
        let a_f32: &[f32] = bytemuck::cast_slice(&ta.data);
        let b_f32: &[f32] = bytemuck::cast_slice(&tb.data);
        let m = op.m as usize;
        let k = op.k as usize;
        let n = op.n as usize;

        // Build MPSGraph.
        let graph = Graph::new();
        let a_ph = graph.placeholder(Some(&[m as isize, k as isize]), DataType::Float32, None);
        let b_ph = graph.placeholder(Some(&[k as isize, n as isize]), DataType::Float32, None);
        let c = graph.matrix_multiplication(&a_ph, &b_ph, None);

        // Create shared-memory MTLBuffers for inputs and output.
        let a_buf = self.mtl_device.new_buffer_with_data(
            a_f32.as_ptr() as *const std::ffi::c_void,
            (m * k * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let b_buf = self.mtl_device.new_buffer_with_data(
            b_f32.as_ptr() as *const std::ffi::c_void,
            (k * n * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
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

        // Read back from the shared-memory output buffer.
        let ptr = out_buf.contents() as *const u8;
        let out_bytes = unsafe { std::slice::from_raw_parts(ptr, (m * n * 4) as usize) };

        Ok(self.alloc_slot(MetalTensor {
            data: out_bytes.to_vec(),
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
        _scales: TensorHandle,
        _biases: TensorHandle,
    ) -> Result<TensorHandle, String> {
        let tx = self.slot(x)?;
        let w_data = self.get_weight(w)?;

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

        // Validate that weight data is the expected size.
        let expected_w_size = crate::nf4tile640::packed_size(k, n);
        if w_data.len() != expected_w_size {
            return Err(format!(
                "quantized_matmul: packed weight size {} != expected {} (k={} n={})",
                w_data.len(),
                expected_w_size,
                k,
                n,
            ));
        }

        // Clone data to release the immutable borrow on self before ensure_nf4_kernel borrows mutably.
        let x_bytes = tx.data.clone();
        let w_bytes = w_data.to_vec();
        let _ = tx;
        let _ = w_data;

        self.ensure_nf4_kernel()?;
        let pipeline = self.nf4_pipeline.as_ref().unwrap();

        // Create MTLBuffers.
        let w_buf = self.mtl_device.new_buffer_with_data(
            w_bytes.as_ptr() as *const std::ffi::c_void,
            w_bytes.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let in_buf = self.mtl_device.new_buffer_with_data(
            x_bytes.as_ptr() as *const std::ffi::c_void,
            (m * k * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let out_buf = self.mtl_device.new_buffer(
            (m * n * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Dispatch the compute kernel.
        let queue = self.mtl_device.new_command_queue();
        let cmd_buf = queue.new_command_buffer();
        let enc = cmd_buf.new_compute_command_encoder();

        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(&w_buf), 0);
        enc.set_buffer(1, Some(&in_buf), 0);
        enc.set_buffer(2, Some(&out_buf), 0);

        let m_val: u32 = m as u32;
        let k_val: u32 = k as u32;
        let n_val: u32 = n as u32;
        enc.set_bytes(3, 4, &m_val as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &k_val as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(5, 4, &n_val as *const u32 as *const std::ffi::c_void);

        let grid = metal::MTLSize { width: n as u64, height: m as u64, depth: 1 };
        let group = metal::MTLSize { width: 16, height: 16, depth: 1 };
        enc.dispatch_threads(grid, group);
        enc.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        // Read back.
        let ptr = out_buf.contents() as *const u8;
        let out_bytes = unsafe { std::slice::from_raw_parts(ptr, (m * n * 4) as usize) };

        Ok(self.alloc_slot(MetalTensor {
            data: out_bytes.to_vec(),
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
        let x_f32: &[f32] = bytemuck::cast_slice(&tx.data);
        let w_f32: &[f32] = bytemuck::cast_slice(&tw.data);
        let dim = op.dim as usize;
        let total: usize = tx.shape.iter().map(|&d| d as usize).product();
        let batch = total / dim;

        // Build MPSGraph.
        let graph = Graph::new();
        let x_ph = graph.placeholder(Some(&[batch as isize, dim as isize]), DataType::Float32, None);
        let w_ph = graph.placeholder(Some(&[dim as isize]), DataType::Float32, None);

        // Compute mean(x^2) via matmul with ones: [batch,dim] @ [dim,1] = [batch,1].
        // We use matmul because mpsgraph v0.2 does not wrap reductionSumWithTensor.
        let x_sq = graph.square(&x_ph, None);
        let ones = graph.constant_with_scalar(1.0_f64, Some(&[dim, 1usize]), DataType::Float32);
        let sum_sq = graph.matrix_multiplication(&x_sq, &ones, None);
        let dim_c = graph.constant_with_scalar(dim as f64, Some(&[1usize, 1usize]), DataType::Float32);
        let mean_sq = graph.division(&sum_sq, &dim_c, None);

        // Add epsilon.
        let eps_c = graph.constant_with_scalar(op.eps as f64, Some(&[1usize, 1usize]), DataType::Float32);
        let mean_eps = graph.addition(&mean_sq, &eps_c, None);

        // sqrt -> divide -> multiply by weight.
        let rms = graph.square_root(&mean_eps, None);
        let normalized = graph.division(&x_ph, &rms, None);
        let result = graph.multiplication(&normalized, &w_ph, None);

        // Create buffers.
        let x_buf = self.mtl_device.new_buffer_with_data(
            x_f32.as_ptr() as *const std::ffi::c_void,
            (total * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let w_buf = self.mtl_device.new_buffer_with_data(
            w_f32.as_ptr() as *const std::ffi::c_void,
            (dim * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let out_buf = self.mtl_device.new_buffer(
            (total * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Shaped types.
        let x_st = ShapedType::new_with_shape_data_type(
            Some(&[batch as isize, dim as isize]),
            DataType::Float32,
        );
        let w_st = ShapedType::new_with_shape_data_type(
            Some(&[dim as isize]),
            DataType::Float32,
        );

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
        let out_td = TensorData::new_with_mtl_buffer(&out_buf, &[batch, dim], DataType::Float32, None);

        let queue = self.mtl_device.new_command_queue();
        let _results = executable.run_with_command_queue(
            &queue,
            &[&*x_td, &*w_td],
            Some(&[&*out_td]),
            None,
        );

        // Read back.
        let ptr = out_buf.contents() as *const u8;
        let out_bytes = unsafe { std::slice::from_raw_parts(ptr, (total * 4) as usize) };

        Ok(self.alloc_slot(MetalTensor {
            data: out_bytes.to_vec(),
            shape: tx.shape.clone(),
            dtype: DType::F32,
            generation: 0,
        }))
    }

    fn rope(&mut self, op: &RoPEOp, x: TensorHandle) -> Result<TensorHandle, String> {
        let tx = self.slot(x)?;
        let x_f32: &[f32] = bytemuck::cast_slice(&tx.data);
        let shape = tx.shape.clone();
        let head_dim = op.head_dim as usize;
        let total = x_f32.len();

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

        Ok(self.alloc_slot(MetalTensor {
            data: bytemuck::cast_slice(&out).to_vec(),
            shape,
            dtype: DType::F32,
            generation: 0,
        }))
    }

    fn add(&mut self, a: TensorHandle, b: TensorHandle) -> Result<TensorHandle, String> {
        let ta = self.slot(a)?;
        let tb = self.slot(b)?;
        let a_f32: &[f32] = bytemuck::cast_slice(&ta.data);
        let b_f32: &[f32] = bytemuck::cast_slice(&tb.data);
        let sum: Vec<f32> = a_f32.iter().zip(b_f32.iter()).map(|(x, y)| x + y).collect();
        Ok(self.alloc_slot(MetalTensor {
            data: bytemuck::cast_slice(&sum).to_vec(),
            shape: ta.shape.clone(),
            dtype: DType::F32,
            generation: 0,
        }))
    }

    fn multiply(&mut self, a: TensorHandle, b: TensorHandle) -> Result<TensorHandle, String> {
        let ta = self.slot(a)?;
        let tb = self.slot(b)?;
        let a_f32: &[f32] = bytemuck::cast_slice(&ta.data);
        let b_f32: &[f32] = bytemuck::cast_slice(&tb.data);
        let prod: Vec<f32> = a_f32.iter().zip(b_f32.iter()).map(|(x, y)| x * y).collect();
        Ok(self.alloc_slot(MetalTensor {
            data: bytemuck::cast_slice(&prod).to_vec(),
            shape: ta.shape.clone(),
            dtype: DType::F32,
            generation: 0,
        }))
    }

    fn silu(&mut self, x: TensorHandle) -> Result<TensorHandle, String> {
        let tx = self.slot(x)?;
        let x_f32: &[f32] = bytemuck::cast_slice(&tx.data);
        let shape: Vec<isize> = tx.shape.iter().map(|&d| d as isize).collect();
        let shape_usize: Vec<usize> = tx.shape.iter().map(|&d| d as usize).collect();
        let count: usize = shape_usize.iter().product();

        // Build MPSGraph: SiLU(x) = x * sigmoid(x)
        let graph = Graph::new();
        let x_ph = graph.placeholder(Some(&shape), DataType::Float32, None);
        let sig = graph.sigmoid(&x_ph, None);
        let result = graph.multiplication(&x_ph, &sig, None);

        let x_buf = self.mtl_device.new_buffer_with_data(
            x_f32.as_ptr() as *const std::ffi::c_void,
            (count * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
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

        let ptr = out_buf.contents() as *const u8;
        let out_bytes = unsafe { std::slice::from_raw_parts(ptr, (count * 4) as usize) };

        Ok(self.alloc_slot(MetalTensor {
            data: out_bytes.to_vec(),
            shape: tx.shape.clone(),
            dtype: DType::F32,
            generation: 0,
        }))
    }

    fn transpose(&mut self, _x: TensorHandle, _dims: &[i32]) -> Result<TensorHandle, String> {
        Err("MetalBackend: transpose not implemented".into())
    }

    fn reshape(&mut self, x: TensorHandle, shape: &[i32]) -> Result<TensorHandle, String> {
        let tensor = self.slot(x)?;
        Ok(self.alloc_slot(MetalTensor {
            data: tensor.data.clone(),
            shape: shape.to_vec(),
            dtype: tensor.dtype,
            generation: 0,
        }))
    }

    fn softmax(&mut self, x: TensorHandle, axis: i32) -> Result<TensorHandle, String> {
        let tx = self.slot(x)?;
        let x_f32: &[f32] = bytemuck::cast_slice(&tx.data);
        let shape: Vec<isize> = tx.shape.iter().map(|&d| d as isize).collect();
        let shape_usize: Vec<usize> = tx.shape.iter().map(|&d| d as usize).collect();
        let count: usize = shape_usize.iter().product();

        // Build MPSGraph with native soft_max.
        let graph = Graph::new();
        let x_ph = graph.placeholder(Some(&shape), DataType::Float32, None);
        let result = graph.soft_max(&x_ph, axis as i64, None);

        let x_buf = self.mtl_device.new_buffer_with_data(
            x_f32.as_ptr() as *const std::ffi::c_void,
            (count * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
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

        let ptr = out_buf.contents() as *const u8;
        let out_bytes = unsafe { std::slice::from_raw_parts(ptr, (count * 4) as usize) };

        Ok(self.alloc_slot(MetalTensor {
            data: out_bytes.to_vec(),
            shape: tx.shape.clone(),
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
    ) -> Result<(ComellationToken, EvaluationReceipt), String>
    {
        let (token, completer) = ComellationToken::new();
        let mut receipt = self.evaluate(group_id, outputs)?;

        // Create a Metal command buffer and set a completion handler.
        let queue = self.mtl_device.new_command_queue();
        let cmd_buf = queue.new_command_buffer();

        // First implementation: synchronous execution, signal token immediately.
        // The Arc<Mutex<Option<...>>> pattern sets up the structure for a future
        // async path via MTLCommandBuffer addCompletedHandler.
        let completer = Arc::new(Mutex::new(Some(completer)));

        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        if let Some(completer) = Arc::clone(&completer).lock().take() {
            completer.complete();
        }

        receipt.submit_ns = 0;
        Ok((token, receipt))
    }

    /// Evaluate outputs into a pre-allocated arena (IOSurface-backed).
    /// Creates a Metal command buffer with a blit encoder for memory
    /// synchronization, commits, and waits for completion.
    #[cfg(any(feature = "mlx-backend", feature = "prism-backend"))]
    fn evaluate_into(
        &mut self,
        group_id: u64,
        outputs: &[TensorHandle],
        _arena: &crate::arena::Arena,
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
        let data: Vec<f32> = bytemuck::cast_slice(&tensor.data).to_vec();
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
    use crate::backend::routing::{
        CorrectnessCheckpointPolicy, LogicalShape, OperationId, Phase, PhysicalLayout, TensorShape,
    };

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
    fn shadow_execute_through_backend_instance() {
        let mut backend = MetalBackend::new().expect("Metal device should be available");
        let m = 4u32;
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
