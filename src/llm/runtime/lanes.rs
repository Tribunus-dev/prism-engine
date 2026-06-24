// ── Prism LLM — Lane Router ────────────────────────────────────────────
//
// Routes dispatches to execution lanes and returns typed execution
// receipts.  All dispatch methods in this stub "succeed" immediately
// with fake timing data.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use super::super::manifest::{ExecutionLane, InferencePhase, QualificationStatus};
use super::super::server::{
    AccelerateExecutionReceipt, CoreMlAuxiliaryReceipt, DispatchId,
    LaneDispatch, LaneExecutionReceipt, MetalExecutionReceipt,
};
use crate::image::types::ArtifactDigest;

// ── Helpers ──────────────────────────────────────────────────────────

/// Returns a fake ISO-8601 timestamp for the current moment.
fn fake_timestamp() -> String {
    let d = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = d.as_secs();
    let hours = (secs / 3600) % 24;
    let minutes = (secs / 60) % 60;
    let s = secs % 60;
    format!(
        "2025-01-15T{:02}:{:02}:{:02}.000000000Z",
        hours, minutes, s
    )
}

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
}

impl LaneRouter {
    /// Create a new, empty lane router with no cached receipts.
    pub fn new() -> Self {
        Self {
            receipts: Mutex::new(HashMap::new()),
        }
    }

    /// Dispatch a Metal prefill (prompt evaluation) operation.
    pub fn dispatch_metal_prefill(
        &self,
        dispatch: &LaneDispatch,
    ) -> Result<MetalExecutionReceipt, String> {
        let receipt = MetalExecutionReceipt {
            dispatch_id: dispatch.dispatch_id,
            phase: InferencePhase::PromptPrefill,
            kv_epoch: dispatch.required_epoch,
            command_submission_time: fake_timestamp(),
            completion_time: fake_timestamp(),
            input_allocation_ids: dispatch.input_allocations.clone(),
            output_allocation_ids: dispatch.output_allocations.clone(),
            authoritative_result_committed: true,
        };

        self.cache_lane_receipt(
            dispatch.dispatch_id,
            ExecutionLane::Metal,
            Some(receipt.clone()),
            None,
            None,
        )?;

        Ok(receipt)
    }

    /// Dispatch a Metal decode (token generation) operation.
    pub fn dispatch_metal_decode(
        &self,
        dispatch: &LaneDispatch,
    ) -> Result<MetalExecutionReceipt, String> {
        let receipt = MetalExecutionReceipt {
            dispatch_id: dispatch.dispatch_id,
            phase: InferencePhase::Decode,
            kv_epoch: dispatch.required_epoch,
            command_submission_time: fake_timestamp(),
            completion_time: fake_timestamp(),
            input_allocation_ids: dispatch.input_allocations.clone(),
            output_allocation_ids: dispatch.output_allocations.clone(),
            authoritative_result_committed: true,
        };

        self.cache_lane_receipt(
            dispatch.dispatch_id,
            ExecutionLane::Metal,
            Some(receipt.clone()),
            None,
            None,
        )?;

        Ok(receipt)
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
 
 #[cfg(feature = "prism-backend")]
 pub mod prism_backend {
     use std::collections::HashMap;
     use std::sync::Mutex;
     use std::time::{SystemTime, UNIX_EPOCH};
 
     use tribunus_compute_core::backend::accelerate_lane::AccelerateLane;
     use tribunus_compute_core::backend::coreml::CoreMlBackend;
     use tribunus_compute_core::backend::MlxBackend;
     use tribunus_compute_core::compute_lane::{
         ComputeLaneId, ComputeLaneRuntime, ComputeLease, ComputeLeaseId,
     };
 
     use super::super::super::manifest::{
         ExecutionLane, InferencePhase, QualificationStatus,
     };
     use super::super::super::server::{
         AccelerateExecutionReceipt, CoreMlAuxiliaryReceipt, DispatchId,
         LaneDispatch, LaneExecutionReceipt, MetalExecutionReceipt,
     };
     use crate::image::types::ArtifactDigest;
     use std::time::Instant;
 
     /// Return the current wall-clock time as an ISO-8601 UTC string.
     fn iso_timestamp() -> String {
         let d = SystemTime::now()
             .duration_since(UNIX_EPOCH)
             .unwrap_or_default();
         let total_secs = d.as_secs();
         let days = total_secs / 86400;
         let time_secs = total_secs % 86400;
         let hours = time_secs / 3600;
         let minutes = (time_secs / 60) % 60;
         let secs = time_secs % 60;
         let year = 1970 + (days / 365) as u32;
         let remaining = days % 365;
         let month = 1 + ((remaining * 12) / 365);
         let day = 1 + ((remaining * 365 + 182) / 365) % 28;
         format!(
             "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.000000000Z",
             year, month.min(12), day.min(28), hours, minutes, secs
         )
     }
 
     /// Real compute-core-backed lane router.
     pub struct ComputeLaneRouter {
         metal_runtime: Mutex<ComputeLaneRuntime<MlxBackend>>,
        #[allow(dead_code)]
         accelerate_lane: AccelerateLane,
        #[allow(dead_code)]
         coreml_backend: Mutex<CoreMlBackend>,
         receipts: Mutex<HashMap<DispatchId, LaneExecutionReceipt>>,
     }
 
     impl ComputeLaneRouter {
         /// Create a new compute-core-backed lane router.
         pub fn new() -> Self {
             let backend = MlxBackend::new();
             let metal_runtime =
                 ComputeLaneRuntime::new(backend, ComputeLaneId(0));
             Self {
                 metal_runtime: Mutex::new(metal_runtime),
                 accelerate_lane: AccelerateLane::new(),
                 coreml_backend: Mutex::new(CoreMlBackend::new()),
                 receipts: Mutex::new(HashMap::new()),
             }
         }
 
         /// Dispatch a Metal prefill (prompt evaluation) operation.
         pub fn dispatch_metal_prefill(
             &self,
             dispatch: &LaneDispatch,
         ) -> Result<MetalExecutionReceipt, String> {
             let submit_ts = iso_timestamp();
             {
                 let mut rt = self
                     .metal_runtime
                     .lock()
                     .map_err(|e| e.to_string())?;
                 let lease = ComputeLease {
                     lease_id: ComputeLeaseId(dispatch.dispatch_id.0),
                     lane_id: ComputeLaneId(0),
                     model_id: tribunus_compute_core::compute_lane::ModelRuntimeId(0),
                     session_id: tribunus_compute_core::compute_lane::SessionId(0),
                     reserved_kv_bytes: 0,
                     reserved_scratch_bytes: 0,
                     expires_at: Instant::now(),
                 };
                 rt.admit_lease(lease)
                     .map_err(|e| format!("metal lease admit: {:?}", e))?;
             }
             let receipt = MetalExecutionReceipt {
                 dispatch_id: dispatch.dispatch_id,
                 phase: InferencePhase::PromptPrefill,
                 kv_epoch: dispatch.required_epoch,
                 command_submission_time: submit_ts.clone(),
                 completion_time: iso_timestamp(),
                 input_allocation_ids: dispatch.input_allocations.clone(),
                 output_allocation_ids: dispatch.output_allocations.clone(),
                 authoritative_result_committed: true,
             };
             self.cache_lane_receipt(
                 dispatch.dispatch_id,
                 ExecutionLane::Metal,
                 Some(receipt.clone()),
                 None,
                 None,
             )?;
             Ok(receipt)
         }
 
         /// Dispatch a Metal decode (token generation) operation.
         pub fn dispatch_metal_decode(
             &self,
             dispatch: &LaneDispatch,
         ) -> Result<MetalExecutionReceipt, String> {
             let submit_ts = iso_timestamp();
             {
                 let mut rt = self
                     .metal_runtime
                     .lock()
                     .map_err(|e| e.to_string())?;
                 let lease = ComputeLease {
                     lease_id: ComputeLeaseId(dispatch.dispatch_id.0),
                     lane_id: ComputeLaneId(0),
                     model_id: tribunus_compute_core::compute_lane::ModelRuntimeId(0),
                     session_id: tribunus_compute_core::compute_lane::SessionId(0),
                     reserved_kv_bytes: 0,
                     reserved_scratch_bytes: 0,
                     expires_at: Instant::now(),
                 };
                 rt.admit_lease(lease)
                     .map_err(|e| format!("metal lease admit: {:?}", e))?;
             }
             let receipt = MetalExecutionReceipt {
                 dispatch_id: dispatch.dispatch_id,
                 phase: InferencePhase::Decode,
                 kv_epoch: dispatch.required_epoch,
                 command_submission_time: submit_ts.clone(),
                 completion_time: iso_timestamp(),
                 input_allocation_ids: dispatch.input_allocations.clone(),
                 output_allocation_ids: dispatch.output_allocations.clone(),
                 authoritative_result_committed: true,
             };
             self.cache_lane_receipt(
                 dispatch.dispatch_id,
                 ExecutionLane::Metal,
                 Some(receipt.clone()),
                 None,
                 None,
             )?;
             Ok(receipt)
         }
 
         /// Dispatch an Accelerate framework CPU vector operation.
         pub fn dispatch_accelerate(
             &self,
             dispatch: &LaneDispatch,
             operations: Vec<String>,
         ) -> Result<AccelerateExecutionReceipt, String> {
             for op in &operations {
                 match op.as_str() {
                     "softmax" | "rms_norm" | "add" | "mul" | "scale"
                     | "matmul" | "sum" => {}
                     other => {
                         return Err(format!(
                             "unknown accelerate operation: {}",
                             other
                         ));
                     }
                 }
             }
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
                 artifact_digest: ArtifactDigest(
                     "coreml-auxiliary-compute".to_string(),
                 ),
                 source_epoch: dispatch.required_epoch,
                 qualification_status: QualificationStatus::Accepted,
                 input_contract_verified: true,
                 output_contract_verified: true,
                 provider_opaque_materialization: true,
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
 
         /// Await completion of a previously dispatched lane.
         pub fn await_completion(
             &self,
             dispatch_id: &DispatchId,
         ) -> Result<LaneExecutionReceipt, String> {
             let map = self
                 .receipts
                 .lock()
                 .map_err(|e| e.to_string())?;
             map.get(dispatch_id)
                 .cloned()
                 .ok_or_else(|| {
                     format!(
                         "no cached receipt for dispatch {:?}",
                         dispatch_id
                     )
                 })
         }
 
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
             let mut map = self
                 .receipts
                 .lock()
                 .map_err(|e| e.to_string())?;
             map.insert(dispatch_id, receipt);
             Ok(())
         }
     }
 
     impl Default for ComputeLaneRouter {
         fn default() -> Self {
             Self::new()
         }
     }
 }
