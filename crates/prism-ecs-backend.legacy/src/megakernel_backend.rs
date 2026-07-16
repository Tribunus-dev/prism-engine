//! Megakernel backend — fused Metal GPU decode via the existing Megakernel.
//!
//! Wraps the Orchestrator's megakernel decode path as a BackendInstance
//! so FlexDispatch can route DecoderLayer operations between this,
//! MetalBackend (per-op MPSGraph), and AneBackend.
//!
//! This is the production decode path — NOT a shadow implementation.

#![cfg(target_os = "macos")]

use crate::heterogeneous_executor::BackendInstance;
use crate::routing::{
    BackendExecutionReceipt, BackendId, BackendVersion, GraphRegion, OperationDescriptor,
    OperationFamily, BACKEND_MEGAKERNEL,
};
use crate::{
    BackendCapabilities, CompiledRegionBackend, DType, EvaluationReceipt, MatmulOp,
    QuantizedMatmulOp, QuantizedWeightHandle, ReadbackReceipt, RmsNormOp, RoPEOp, TensorBackend,
    TensorHandle,
};
use prism_ecs_core::compute_image::orchestrator::Orchestrator;

/// Megakernel backend — owns the Orchestrator and its Metal resources.
pub struct MegakernelBackend {
    orchestrator: Orchestrator,
    #[allow(dead_code)]
    batch_size: u32,
    #[allow(dead_code)]
    int4_mode: bool,
    decode_count: u64,
    /// Most recently decoded token ID.
    pub last_decoded_token: u64,
}

impl MegakernelBackend {
    /// Load a cimage and build the megakernel.
    pub fn from_cimage(
        path: impl AsRef<std::path::Path>,
        batch_size: u32,
        int4_mode: bool,
    ) -> Result<Self, String> {
        let orchestrator = Orchestrator::from_cimage(path, batch_size, int4_mode)?;
        Ok(Self {
            orchestrator,
            batch_size,
            int4_mode,
            decode_count: 0,
            last_decoded_token: 0,
        })
    }
}

impl TensorBackend for MegakernelBackend {
    // Stub — the megakernel manages its own buffers internally.
    // Returns Err for any tensor operations since the megakernel
    // allocates and manages its own Metal buffers via launch().
    fn create_f32(&mut self, _data: &[f32], _shape: &[i32]) -> Result<TensorHandle, String> {
        Err("MegakernelBackend: use Orchestrator's internal buffers".into())
    }
    fn create_u32(&mut self, _data: &[u32], _shape: &[i32]) -> Result<TensorHandle, String> {
        Err("MegakernelBackend: not supported".into())
    }
    fn create_f32_from_bf16_bits(
        &mut self,
        _data: &[u16],
        _shape: &[i32],
    ) -> Result<TensorHandle, String> {
        Err("MegakernelBackend: not supported".into())
    }
    fn create_owned_from_bytes(
        &mut self,
        _data: &[u8],
        _shape: &[i32],
        _dtype: DType,
    ) -> Result<TensorHandle, String> {
        Err("MegakernelBackend: not supported".into())
    }
    fn bind_external(
        &mut self,
        _owner_token: u64,
        _data: &[u8],
        _shape: &[i32],
        _dtype: DType,
    ) -> Result<TensorHandle, String> {
        Err("MegakernelBackend: not supported".into())
    }
    fn matmul(
        &mut self,
        _op: &MatmulOp,
        _a: TensorHandle,
        _b: TensorHandle,
    ) -> Result<TensorHandle, String> {
        Err("MegakernelBackend: use Orchestrator".into())
    }
    fn quantized_matmul(
        &mut self,
        _op: &QuantizedMatmulOp,
        _x: TensorHandle,
        _w: QuantizedWeightHandle,
        _scales: TensorHandle,
        _biases: TensorHandle,
    ) -> Result<TensorHandle, String> {
        Err("MegakernelBackend: not supported".into())
    }
    fn rms_norm(
        &mut self,
        _op: &RmsNormOp,
        _x: TensorHandle,
        _weight: TensorHandle,
    ) -> Result<TensorHandle, String> {
        Err("MegakernelBackend: not supported".into())
    }
    fn rope(&mut self, _op: &RoPEOp, _x: TensorHandle) -> Result<TensorHandle, String> {
        Err("MegakernelBackend: not supported".into())
    }
    fn add(&mut self, _a: TensorHandle, _b: TensorHandle) -> Result<TensorHandle, String> {
        Err("MegakernelBackend: not supported".into())
    }
    fn multiply(&mut self, _a: TensorHandle, _b: TensorHandle) -> Result<TensorHandle, String> {
        Err("MegakernelBackend: not supported".into())
    }
    fn silu(&mut self, _x: TensorHandle) -> Result<TensorHandle, String> {
        Err("MegakernelBackend: not supported".into())
    }
    fn transpose(&mut self, _x: TensorHandle, _dims: &[i32]) -> Result<TensorHandle, String> {
        Err("MegakernelBackend: not supported".into())
    }
    fn reshape(&mut self, _x: TensorHandle, _shape: &[i32]) -> Result<TensorHandle, String> {
        Err("MegakernelBackend: not supported".into())
    }
    fn softmax(&mut self, _x: TensorHandle, _axis: i32) -> Result<TensorHandle, String> {
        Err("MegakernelBackend: not supported".into())
    }
    fn index_select(
        &mut self,
        _x: TensorHandle,
        _indices: &[u32],
        _axis: i32,
    ) -> Result<TensorHandle, String> {
        Err("MegakernelBackend: not supported".into())
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
    fn read_f32(&mut self, _handle: TensorHandle) -> Result<ReadbackReceipt, String> {
        Err("MegakernelBackend: readback not supported directly".into())
    }
    fn shape(&self, _handle: TensorHandle) -> Result<Vec<i32>, String> {
        Err("MegakernelBackend: not supported".into())
    }
    fn release(&mut self, _handle: TensorHandle) -> Result<(), String> {
        Err("MegakernelBackend: not supported".into())
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
            backend_name: "megakernel".into(),
        }
    }
}

impl BackendInstance for MegakernelBackend {
    fn backend_kind(&self) -> BackendId {
        BACKEND_MEGAKERNEL
    }

    fn last_decoded_token(&self) -> Option<u64> {
        Some(self.last_decoded_token)
    }

    fn supports(&self, family: OperationFamily) -> bool {
        matches!(
            family,
            OperationFamily::DecoderLayer
                | OperationFamily::PrefillFragment
                | OperationFamily::VisionEncode
                | OperationFamily::AudioEncode
                | OperationFamily::MultimodalProject
        )
    }

    fn execute(
        &mut self,
        op: &OperationDescriptor,
        _inputs: &[TensorHandle],
    ) -> Result<BackendExecutionReceipt, String> {
        match op.family {
            OperationFamily::DecoderLayer => {
                let start = std::time::Instant::now();
                // Decode one token on slot 0.
                let token_id = op.operation_id.0 as u32;
                let (next_token, _logits) = self.orchestrator.decode_token_logits(token_id)?;
                self.last_decoded_token = next_token as u64;
                let elapsed = start.elapsed().as_nanos() as u64;
                self.decode_count += 1;
                Ok(BackendExecutionReceipt {
                    operation_id: op.operation_id,
                    backend_id: BACKEND_MEGAKERNEL,
                    backend_version: BackendVersion {
                        backend_name: "megakernel".into(),
                        version: "0.1".into(),
                        git_commit: None,
                    },
                    requested_substrate: None,
                    observed_substrate: None,
                    graph_build_ns: None,
                    compile_ns: None,
                    queue_wait_ns: None,
                    submit_ns: None,
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
            OperationFamily::PrefillFragment => {
                Err("MegakernelBackend: prefill not implemented on megakernel path".into())
            }
            OperationFamily::VisionEncode => {
                Err("VisionEncode: use the server's multimodal path".into())
            }
            OperationFamily::AudioEncode => {
                // The actual nf4tile640 projection dispatch is done by the
                // orchestrator once weights are loaded. For now, users must
                // go through the server's multimodal encode path which
                // handles WAV decode, mel extraction, and AudioEncode
                // operation descriptor creation.
                Err("AudioEncode: dispatch via orchestrator nf4 projection".into())
            }
            OperationFamily::MultimodalProject => {
                Err("MultimodalProject: use the server's multimodal path".into())
            }
            _ => Err(format!("MegakernelBackend: unsupported {:?}", op.family)),
        }
    }

    fn as_compiled_region_backend(&mut self) -> Option<&mut dyn CompiledRegionBackend> {
        Some(self)
    }
}

impl CompiledRegionBackend for MegakernelBackend {
    fn execute_compiled_region(
        &mut self,
        region: &GraphRegion,
        _inputs: &[TensorHandle],
        _outputs: &[TensorHandle],
    ) -> Result<BackendExecutionReceipt, String> {
        match region.family {
            OperationFamily::DecoderLayer => {
                let start = std::time::Instant::now();
                // Decode via slot 0. The token id is not carried in GraphRegion
                // directly — this is a stub that decodes the first available.
                // FlexDispatch will wire the token id through a side channel
                // before calling into the compiled region path.
                let (next_token, _logits) = self.orchestrator.decode_token_logits(0)?;
                self.last_decoded_token = next_token as u64;
                let elapsed = start.elapsed().as_nanos() as u64;
                self.decode_count += 1;
                Ok(BackendExecutionReceipt {
                    operation_id: region
                        .operations
                        .first()
                        .copied()
                        .unwrap_or(crate::routing::OperationId(0)),
                    backend_id: BACKEND_MEGAKERNEL,
                    backend_version: BackendVersion {
                        backend_name: "megakernel".into(),
                        version: "0.1".into(),
                        git_commit: None,
                    },
                    requested_substrate: None,
                    observed_substrate: None,
                    graph_build_ns: None,
                    compile_ns: None,
                    queue_wait_ns: None,
                    submit_ns: None,
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
            _ => Err(format!(
                "MegakernelBackend: unsupported compiled region {:?}",
                region.family
            )),
        }
    }

    fn supports_region(&self, family: OperationFamily) -> bool {
        matches!(family, OperationFamily::DecoderLayer)
    }
}
