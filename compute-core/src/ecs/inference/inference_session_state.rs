#![cfg(feature = "mlx-backend")]

use crate::ecs::compute_image::kv_plan::{KvCachePlan, KvCodec};
use crate::ecs::inference::inference_step_state::StepReceiptLedger;
use crate::ecs::kv_cache::KvCache;
use crate::ecs::kv_cache::LiveKvCache;
use crate::ecs::runtime::resources::kv_cache_coordinator::CompressedKvCache;
use crate::ecs::scheduling::receipts::PhaseReceipt;
use crate::executor::SinkState;
use crate::profiled_executor::WorkingSetManager;
use crate::quantization::turboquant_kv::KvQuantMode as TqKvQuantMode;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

/// Unique identifier for an inference session.
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct InferenceSessionId(pub String);

/// Mutable per-session state owned by the PhaseEngine.
///
/// Contains KV caches, sink states, the active working set for weight
/// streaming, lane registries, and cancellation sources.
pub struct InferenceSessionState {
    pub session_id: InferenceSessionId,
    pub kv_caches: Vec<LiveKvCache>,
    pub sink_states: Vec<SinkState>,
    pub working_set: Option<WorkingSetManager>,
    pub coreai_models: CoreAiModelRegistryStub,
    pub lane_registry: LaneRegistryStub,
    pub cancellation: Arc<AtomicBool>,
    pub session_epoch: AtomicU64,
    pub receipt_ledger: StepReceiptLedger,
}

/// Stub for the Core ML model registry.
/// In a full implementation this loads artifacts once at session creation time.
pub struct CoreAiModelRegistryStub;

/// Stub for the lane registry.
pub struct LaneRegistryStub;

impl InferenceSessionState {
    pub fn new(
        session_id: String,
        plan: &KvCachePlan,
        kv_caches: Vec<KvCache>,
        sink_states: Vec<SinkState>,
    ) -> Self {
        let live_caches = kv_caches
            .into_iter()
            .map(|c| match &plan.codec {
                KvCodec::Fp16 => LiveKvCache::Fp16(c),
                KvCodec::TurboQuant {
                    key_mode,
                    value_mode: _,
                    group_size,
                } => {
                    let tq_mode = kv_plan_mode_to_tq(key_mode);
                    LiveKvCache::Compressed(CompressedKvCache::new(
                        tq_mode,
                        *group_size as usize,
                        plan.max_blocks as usize,
                    ))
                }
                KvCodec::Fp32 => LiveKvCache::Fp16(c),
            })
            .collect();
        Self {
            session_id: InferenceSessionId(session_id),
            kv_caches: live_caches,
            sink_states,
            working_set: None,
            coreai_models: CoreAiModelRegistryStub,
            lane_registry: LaneRegistryStub,
            cancellation: Arc::new(AtomicBool::new(false)),
            session_epoch: AtomicU64::new(0),
            receipt_ledger: StepReceiptLedger::new(),
        }
    }

    /// Check whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.cancellation.load(Ordering::Relaxed)
    }

    /// Request cancellation.
    pub fn cancel(&self) {
        self.cancellation.store(true, Ordering::Relaxed);
    }

    /// Increment and return the session epoch.
    pub fn next_epoch(&self) -> u64 {
        self.session_epoch.fetch_add(1, Ordering::Relaxed)
    }

    /// Record a phase receipt in the session receipt ledger.
    pub fn push_receipt(&mut self, receipt: PhaseReceipt) {
        self.receipt_ledger.push(receipt);
    }
}
/// Convert a kv_plan KvQuantMode (serialization-friendly) to a
/// turboquant_kv KvQuantMode (implementation type) with conservative
/// defaults for fields absent in the plan representation.
fn kv_plan_mode_to_tq(mode: &crate::ecs::compute_image::kv_plan::KvQuantMode) -> TqKvQuantMode {
    use crate::ecs::compute_image::kv_plan::KvQuantMode as PlanMode;
    match mode {
        PlanMode::Polar(b) => TqKvQuantMode::Polar(*b),
        PlanMode::Prod(b) => TqKvQuantMode::Prod(*b),
        PlanMode::Split(b) => TqKvQuantMode::Split(*b),
        PlanMode::Mse(b) => TqKvQuantMode::Mse {
            bits: *b,
            state_bits: 4,
        },
        PlanMode::PolarProd(b) => TqKvQuantMode::PolarProd(*b),
        PlanMode::PolarHadamard(b) => TqKvQuantMode::PolarHadamard(*b),
        PlanMode::TurboQuant3 => TqKvQuantMode::TurboQuant3 {
            bits: 3,
            qjl_bits: 2,
        },
    }
}
