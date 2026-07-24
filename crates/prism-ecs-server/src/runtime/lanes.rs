// ── Prism LLM — Lane Router ────────────────────────────────────────────
//
// Routes dispatches to execution lanes and returns typed execution
// receipts.  All dispatch methods in this stub "succeed" immediately
// with fake timing data.

use std::collections::HashMap;
use std::sync::Mutex;
use crate::runtime::manifest::{ExecutionLane, QualificationStatus};
use crate::runtime::server_types::{
    AccelerateExecutionReceipt, ArtifactDigest, CoreMlAuxiliaryReceipt, DispatchId, LaneDispatch,
    LaneExecutionReceipt, MetalExecutionReceipt,
};
use prism_ecs_runtime::{BackendExecutionRegistry, KernelDispatchSpec};

// ── Helpers ──────────────────────────────────────────────────────────

// ── LaneRouter ───────────────────────────────────────────────────────

/// Routes inference dispatches to the appropriate execution lane and
/// produces typed execution receipts.
///
/// Each `dispatch_*` method "executes" the lane immediately in this
/// stub, returning a receipt populated with the dispatch's identifiers
/// and fake timing data.  The receipt is cached internally so that
/// [`await_completion`](Self::await_completion) can return a
/// [`LaneExecutionReceipt`] wrapping the lane-specific receipt.
pub struct LaneRouter {
    /// Cached lane execution receipts, keyed by dispatch id.
    receipts: Mutex<HashMap<DispatchId, LaneExecutionReceipt>>,
    /// The sole backend resource owner for kernel-backed lane execution.
    /// Legacy receipt methods remain available for compatibility, while new
    /// callers must use `dispatch_metal_kernel` to obtain an authoritative
    /// result from the persistent ECS registry.
    kernel_registry: BackendExecutionRegistry,
}

pub struct LaneCapabilities { pub coreml_ane: bool, pub metal: bool, pub accelerate: bool }
impl LaneCapabilities { pub fn host() -> Self { Self { coreml_ane: false, metal: false, accelerate: true } } }

impl LaneRouter {
    /// Create a new, empty lane router with no cached receipts.
    pub fn new() -> Self {
        Self {
            receipts: Mutex::new(HashMap::new()),
            kernel_registry: BackendExecutionRegistry::new(),
        }
    }

    /// Dispatch a registered Metal kernel through the ECS-owned backend
    /// registry. This is the execution boundary used by modality and model
    /// work items; it cannot claim completion until the backend returns.
    pub fn dispatch_metal_kernel(
        &self,
        spec: &KernelDispatchSpec,
    ) -> Result<prism_ecs_kernel::KernelOutput, String> {
        self.kernel_registry
            .dispatch("metal", spec)
            .map_err(|error| format!("Metal ECS dispatch failed: {error}"))
    }

    pub fn kernel_registry(&self) -> BackendExecutionRegistry {
        self.kernel_registry.clone()
    }

    /// Dispatch a Metal prefill (prompt evaluation) operation.
    pub fn dispatch_metal_prefill(
        &self,
        _dispatch: &LaneDispatch,
    ) -> Result<MetalExecutionReceipt, String> {
        Err("legacy Metal prefill router is non-authoritative; dispatch through ECS registry".into())
    }

    /// Dispatch a Metal decode (token generation) operation.
    pub fn dispatch_metal_decode(
        &self,
        _dispatch: &LaneDispatch,
    ) -> Result<MetalExecutionReceipt, String> {
        Err("legacy Metal decode router is non-authoritative; dispatch through ECS registry".into())
    }

    /// Dispatch an Accelerate framework operation.
    pub fn dispatch_accelerate(
        &self,
        dispatch: &LaneDispatch,
        operations: Vec<String>,
    ) -> Result<AccelerateExecutionReceipt, String> {
        let receipt = AccelerateExecutionReceipt {
            dispatch_id: dispatch.dispatch_id,
            operations,
            shared_memory_mapped: true,
            cpu_readback: false,
            fallback_used: false,
        };

        self.cache_lane_receipt(
            dispatch.dispatch_id,
            ExecutionLane::Accelerate,
            None,
            Some(receipt.clone()),
            None,
        )?;

        Ok(receipt)
    }

    /// Dispatch a Core ML auxiliary island execution.
    pub fn dispatch_coreml_auxiliary(
        &self,
        dispatch: &LaneDispatch,
        island_id: &str,
    ) -> Result<CoreMlAuxiliaryReceipt, String> {
        let receipt = CoreMlAuxiliaryReceipt {
            auxiliary_island_id: island_id.to_string(),
            artifact_digest: ArtifactDigest("coreml-auxiliary-stub".to_string()),
            source_epoch: dispatch.required_epoch,
            qualification_status: QualificationStatus::Accepted,
            input_contract_verified: true,
            output_contract_verified: true,
            provider_opaque_materialization: false,
        };

        self.cache_lane_receipt(
            dispatch.dispatch_id,
            ExecutionLane::CoreMlAne,
            None,
            None,
            Some(receipt.clone()),
        )?;

        Ok(receipt)
    }

    /// Await completion of a previously dispatched lane and return the
    /// full lane execution receipt.
    ///
    /// In this stub every dispatch completes synchronously, so this
    /// method simply returns the cached receipt.
    pub fn await_completion(
        &self,
        dispatch_id: &DispatchId,
    ) -> Result<LaneExecutionReceipt, String> {
        let map = self.receipts.lock().map_err(|e| e.to_string())?;
        map.get(dispatch_id)
            .cloned()
            .ok_or_else(|| format!("no cached receipt for dispatch {:?}", dispatch_id))
    }

    // ── Internals ──────────────────────────────────────────────────

    /// Store a lane execution receipt for later retrieval by
    /// [`await_completion`](Self::await_completion).
    fn cache_lane_receipt(
        &self,
        dispatch_id: DispatchId,
        lane: ExecutionLane,
        metal: Option<MetalExecutionReceipt>,
        accelerate: Option<AccelerateExecutionReceipt>,
        coreml: Option<CoreMlAuxiliaryReceipt>,
    ) -> Result<(), String> {
        let receipt = LaneExecutionReceipt {
            lane,
            metal,
            accelerate,
            coreml,
        };
        let mut map = self.receipts.lock().map_err(|e| e.to_string())?;
        map.insert(dispatch_id, receipt);
        Ok(())
    }
}

impl Default for LaneRouter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod ecs_metal_tests {
    use super::*;
    use prism_ecs_kernel::{
        BackendKind, DispatchGeometry, KernelBackend, KernelCompileRequest, KernelDescriptor,
        KernelVariant, MetalBackend,
    };

    #[test]
    fn metal_lane_dispatches_registered_artifact_through_ecs_registry() {
        let router = LaneRouter::new();
        let artifact = MetalBackend::new()
            .compile(&KernelCompileRequest {
                source: b"kernel void fp16_gemv() {}".to_vec(),
                descriptor: KernelDescriptor {
                    name: "fp16_gemv".into(),
                    variant: KernelVariant::FP16GEMV,
                    backend: BackendKind::Metal,
                    source_digest: String::new(),
                    binary_digest: String::new(),
                    binding_signature: Vec::new(),
                    dispatch_geometry: DispatchGeometry {
                        threads_per_threadgroup: [1, 1, 1],
                        threadgroups_per_grid: [1, 1, 1],
                        threads_per_grid: [1, 1, 1],
                    },
                },
                source_path: None,
            })
            .expect("compile Metal artifact");
        let binding = router
            .kernel_registry()
            .register_artifact(artifact)
            .expect("register Metal artifact");
        let result = router.dispatch_metal_kernel(&binding.dispatch_spec());
        #[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
        assert!(result.is_ok(), "native Metal dispatch failed: {result:?}");
        #[cfg(not(all(target_os = "macos", feature = "metal-dispatch")))]
        assert!(result.unwrap_err().contains("Metal dispatch requires"));
    }
}
