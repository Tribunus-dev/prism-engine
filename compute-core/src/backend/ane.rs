//! ANE backend — compiled-region execution on Apple Neural Engine.
//!
//! Implements CompiledRegionBackend for subgraph families:
//! AttentionBlock, MlpBlock, DecoderLayer, PrefillFragment.

#![cfg(target_os = "macos")]

use std::collections::HashMap;

use crate::ane_bridge::{execute_ane_step, AneInferenceStep, AneProgramCache};
use crate::backend::heterogeneous_executor::BackendInstance;
use crate::backend::routing::{
    BackendExecutionReceipt, BackendVersion, GraphRegion, OperationDescriptor,
    BackendId, OperationFamily, OperationId, RequestedSubstrate, Substrate, BACKEND_ANE,
};
use crate::backend::{
    BackendCapabilities, CompiledRegionBackend, DType, EvaluationReceipt, MatmulOp,
    QuantizedMatmulOp, QuantizedWeightHandle, ReadbackReceipt, RmsNormOp, RoPEOp,
    TensorBackend, TensorHandle,
};

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
    /// Region ID → compiled inference step mapping.
    /// Populated by the cimage loader in Phase 7.
    region_programs: HashMap<u64, AneInferenceStep>,
    /// Owner token → tensor handle for externally bound (IOSurface) tensors.
    external_bindings: HashMap<u64, TensorHandle>,
}

struct AneTensor {
    data: Vec<u8>,
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
            region_programs: HashMap::new(),
            external_bindings: HashMap::new(),
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
}

impl Default for AneBackend {
    fn default() -> Self {
        Self::new()
    }
}

// ── TensorBackend ───────────────────────────────────────────────────────────

impl TensorBackend for AneBackend {
    fn create_f32(&mut self, data: &[f32], shape: &[i32]) -> Result<TensorHandle, String> {
        Ok(self.alloc_slot(AneTensor {
            data: bytemuck::cast_slice(data).to_vec(),
            shape: shape.to_vec(),
            dtype: DType::F32,
            generation: 0,
        }))
    }

    fn create_u32(&mut self, data: &[u32], shape: &[i32]) -> Result<TensorHandle, String> {
        Ok(self.alloc_slot(AneTensor {
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
        Err("AneBackend: create_f32_from_bf16_bits not implemented".into())
    }

    fn create_owned_from_bytes(
        &mut self,
        data: &[u8],
        shape: &[i32],
        dtype: DType,
    ) -> Result<TensorHandle, String> {
        Ok(self.alloc_slot(AneTensor {
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
        let handle = self.alloc_slot(AneTensor {
            data: data.to_vec(),
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

    fn rope(
        &mut self,
        _op: &RoPEOp,
        _x: TensorHandle,
    ) -> Result<TensorHandle, String> {
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
            data: t.data.clone(),
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
        let step = self.region_programs.get(&region.region_id).ok_or_else(|| {
            format!(
                "AneBackend: no compiled program for region {}",
                region.region_id
            )
        })?;

        let start = std::time::Instant::now();
        let boundary = execute_ane_step(step, self.program_cache)?;
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
            graph_build_ns: Some(boundary.graph_build_ns),
            compile_ns: None,
            queue_wait_ns: None,
            submit_ns: Some(boundary.submit_ns),
            execution_ns: Some(boundary.execution_ns),
            synchronization_ns: None,
            total_wall_ns: elapsed,
            bytes_read: None,
            bytes_written: None,
            temporary_bytes: Some(boundary.temporary_bytes),
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
pub fn register_ane_region(
    backend: &mut AneBackend,
    region_id: u64,
    step: AneInferenceStep,
) {
    backend.region_programs.insert(region_id, step);
}
