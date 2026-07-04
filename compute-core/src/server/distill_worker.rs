//! Background distillation worker — orchestrates the Level 1/2/3 compiler
//! pipeline as an async service within prism-server.
//!
//! # Pipeline per block
//!   1. Level1Scheduler (always): Metal teacher + ternary student + Accelerate reducer
//!   2. Level2Scheduler (macOS only): Core ML teacher + Metal student + Accelerate reducer
//!   3. Level 1 numerical gate on reducer metrics
//!   4. Level 2 gates (semantic equivalence, failure containment, joint acceptance)
//!   5. BlockReceipt with metrics + execution provenance
//!   6. Repeat for each block

use crate::compilation::level1::gates::check_numerical;
use crate::compilation::level1::scheduler::{Level1Config, Level1Scheduler};
use crate::compilation::level2::bridge::CoreMLTeacher;
use crate::compilation::level2::gates::{
    check_joint_acceptance_rate, AcceptanceThresholds, JointAcceptanceResult,
};
use crate::compilation::level2::scheduler::Level2Scheduler;
use crate::compilation::memory_budget::MemoryBudget;
use crate::compilation::receipt::{BlockReceipt, EngineExecutionLog};
use crate::server::state::{MemoryAllocationBroker, ServerOperationalMode};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Request payload for `/v1/distill`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DistillationRequest {
    pub job_id: String,
    pub teacher_checkpoint: String,
    pub assistant_checkpoint: Option<String>,
    #[serde(default = "default_representation")]
    pub target_representation: String,
    #[serde(default = "default_memory_ceiling")]
    pub memory_ceiling_gb: f64,
    /// Path to directory containing .mlmodelc bundles (Level 2 path).
    #[serde(default)]
    pub model_dir: Option<String>,
    #[serde(default)]
    pub modality_profiles: HashMap<String, HashMap<String, f64>>,
    #[serde(default)]
    pub gates: HashMap<String, serde_json::Value>,
}

fn default_representation() -> String {
    "ternarypage640".into()
}

fn default_memory_ceiling() -> f64 {
    10.5
}

/// Status of a running or completed distillation job.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DistillationJobStatus {
    pub job_id: String,
    pub state: DistillationState,
    pub teacher_checkpoint: String,
    pub target_representation: String,
    pub memory_ceiling_gb: f64,
    pub current_block: usize,
    pub total_blocks: usize,
    pub blocks_completed: usize,
    pub receipt_count: usize,
    pub level2_used: bool,
    pub level2_fallback_verified: bool,
    pub joint_acceptance_rate: Option<f64>,
    pub joint_acceptance_passed: bool,
    pub error: Option<String>,
}

/// Phase of a distillation job.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum DistillationState {
    Queued,
    Ingesting,
    Compiling,
    Verifying,
    Completed,
    Failed,
}

/// Internal state for a single distillation job.
struct DistillationJob {
    request: DistillationRequest,
    state: DistillationState,
    current_block: usize,
    blocks_completed: usize,
    block_receipts: Vec<BlockReceipt>,
    level2_used: bool,
    level2_fallback_verified: bool,
    joint_acceptance_result: Option<JointAcceptanceResult>,
    error: Option<String>,
}

/// Shared engine that manages all active distillation jobs.
pub struct DistillationEngine {
    jobs: Arc<Mutex<HashMap<String, DistillationJob>>>,
    memory_broker: Arc<MemoryAllocationBroker>,
}

impl DistillationEngine {
    pub fn new(memory_broker: Arc<MemoryAllocationBroker>) -> Self {
        DistillationEngine {
            jobs: Arc::new(Mutex::new(HashMap::new())),
            memory_broker,
        }
    }

    pub async fn submit(
        &self,
        request: DistillationRequest,
    ) -> Result<String, String> {
        let mut jobs = self.jobs.lock().await;
        let job_id = request.job_id.clone();
        if jobs.contains_key(&job_id) {
            return Err(format!("job {} already exists", job_id));
        }
        let total_blocks = 48;
        let has_model_dir = request.model_dir.is_some();
        let job = DistillationJob {
            request: request.clone(),
            state: DistillationState::Queued,
            current_block: 0,
            blocks_completed: 0,
            block_receipts: Vec::with_capacity(total_blocks),
            level2_used: false,
            level2_fallback_verified: false,
            joint_acceptance_result: None,
            error: None,
        };
        jobs.insert(job_id.clone(), job);
        let j = self.jobs.clone();
        let b = self.memory_broker.clone();
        let jid = job_id.clone();
        let md = request.model_dir.clone();
        let enabled = has_model_dir;
        tokio::spawn(async move {
            run_distillation_loop(j, b, jid, total_blocks, md, enabled).await;
        });
        Ok(job_id)
    }

    pub async fn status(&self, job_id: &str) -> Option<DistillationJobStatus> {
        let jobs = self.jobs.lock().await;
        jobs.get(job_id).map(|j| DistillationJobStatus {
            job_id: job_id.to_string(),
            state: j.state.clone(),
            teacher_checkpoint: j.request.teacher_checkpoint.clone(),
            target_representation: j.request.target_representation.clone(),
            memory_ceiling_gb: j.request.memory_ceiling_gb,
            current_block: j.current_block,
            total_blocks: 48,
            blocks_completed: j.blocks_completed,
            receipt_count: j.block_receipts.len(),
            level2_used: j.level2_used,
            level2_fallback_verified: j.level2_fallback_verified,
            joint_acceptance_rate: j.joint_acceptance_result
                .as_ref()
                .map(|r| r.acceptance_rate),
            joint_acceptance_passed: j.joint_acceptance_result
                .as_ref()
                .map(|r| r.passed)
                .unwrap_or(false),
            error: j.error.clone(),
        })
    }

    pub fn memory_broker(&self) -> &Arc<MemoryAllocationBroker> {
        &self.memory_broker
    }
}

// ── Level 2 helper (gated on macOS + prism-backend) ──────────────────────

/// Run the Level 2 Core ML teacher pipeline for one block.
/// Returns (peak_bytes, fallback_verified) or None if not available.
#[cfg(all(target_os = "macos", feature = "prism-backend"))]
fn run_level2_block(
    config: &Level1Config,
    model_dir: &str,
    block_idx: usize,
) -> Option<(u64, bool)> {
    let teacher = CoreMLTeacher::new(model_dir);
    let mut sched = Level2Scheduler::new(
        config.clone(),
        8,
        teacher,
        true, // coreai_available
    );
    sched.initialize();
    while sched.step() {}

    let peak = sched.peak_memory();
    let bridge_receipts = sched.bridge_receipts();
    let fallback_occurred = bridge_receipts
        .iter()
        .any(|r| r.actual_route.contains("Metal-fallback"));

    // Drop scheduler → all Core ML + Metal resources released.
    drop(sched);

    Some((peak, !fallback_occurred))
}

#[cfg(not(all(target_os = "macos", feature = "prism-backend")))]
fn run_level2_block(
    _config: &Level1Config,
    _model_dir: &str,
    _block_idx: usize,
) -> Option<(u64, bool)> {
    None
}

// ── Main distillation loop ────────────────────────────────────────────────

/// Background loop driving a single distillation job block-by-block.
///
/// Per block:
///   a. Level1Scheduler: Metal teacher + ternary student + Accelerate reducer
///   b. Level2Scheduler (when model_dir is Some + macOS): Core ML teacher
///   c. check_numerical() — Level 1 gate
///   d. BlockReceipt with metrics + execution provenance
///   e. Release block resources
///
/// After all blocks:
///   f. check_joint_acceptance_rate() — Level 2 gate
///   g. Store final receipts + gate results
async fn run_distillation_loop(
    jobs: Arc<Mutex<HashMap<String, DistillationJob>>>,
    broker: Arc<MemoryAllocationBroker>,
    job_id: String,
    total_blocks: usize,
    model_dir: Option<String>,
    level2_requested: bool,
) {
    broker.set_mode(ServerOperationalMode::Distilling);
    let ceiling = MemoryAllocationBroker::DISTILL_SUB_CEILING_BYTES;

    // ── Ingesting ────────────────────────────────────────────────────────
    {
        let mut j = jobs.lock().await;
        if let Some(job) = j.get_mut(&job_id) {
            job.state = DistillationState::Ingesting;
            job.level2_used = level2_requested;
        }
    }

    // ── Compiling (block-by-block) ───────────────────────────────────────
    for block_idx in 0..total_blocks {
        if broker.available() < 100_000_000 {
            let mut j = jobs.lock().await;
            if let Some(job) = j.get_mut(&job_id) {
                job.state = DistillationState::Failed;
                job.error = Some(format!("out of memory at block {}", block_idx));
            }
            broker.set_mode(ServerOperationalMode::Idle);
            return;
        }

        broker.declare(ceiling);

        // ── Level 1: Metal teacher + Ternary student + Accelerate reducer ──
        let config = Level1Config {
            microbatch: 4096,
            hidden_dim: 3840,
            pages_per_row: 2,
            budget: MemoryBudget::m1_16gb_default(),
            objective_weights: None,
        };

        let mut l1 = Level1Scheduler::new(config.clone(), 8);
        l1.initialize();
        while l1.step() {}

        let mse = l1.reducer().output_mse.unwrap_or(f64::INFINITY);
        let cosine = l1.reducer().cosine_similarity.unwrap_or(0.0);
        let residual = l1.reducer().residual_relative_error.unwrap_or(f64::INFINITY);
        let l1_peak = l1.peak_memory();

        // ── Level 2: Core ML teacher (when available) ──────────────────────
        let mut l2_peak = 0u64;
        let mut l2_fallback_verified = false;

        if let Some(ref md) = model_dir {
            if let Some((pk, fv)) = run_level2_block(&config, md, block_idx) {
                l2_peak = pk;
                l2_fallback_verified = fv;
            }
        }

        // ── Level 1 gates ─────────────────────────────────────────────
        let num_result = check_numerical();

        // Build block receipt.
        let mut numerical_drift = HashMap::new();
        numerical_drift.insert("output_mse".into(), mse as f32);
        numerical_drift.insert("cosine_similarity".into(), cosine as f32);
        numerical_drift.insert("residual_error".into(), residual as f32);
        numerical_drift.insert("gate_passed".into(), if num_result.passed { 1.0 } else { 0.0 });

        let receipt = BlockReceipt {
            block_index: block_idx,
            modality_tag: "text".into(),
            input_frontier_hash: String::new(),
            output_frontier_hash: String::new(),
            sidecar_fraction: 0.0,
            optimal_scale_dtype: "two_level_int8".into(),
            numerical_drift,
            execution_provenance: EngineExecutionLog {
                backend_requested: if level2_requested { "Level2-CoreML".into() } else { "Level1-Metal".into() },
                backend_observed: if l2_fallback_verified { "Level2-CoreML".into() } else { "Level1-Metal".into() },
                zero_copy_verified: false,
                wall_time_ms: 0.0,
                peak_arena_bytes: l2_peak.max(l1_peak),
            },
        };

        broker.release(ceiling);

        // Store.
        {
            let mut j = jobs.lock().await;
            if let Some(job) = j.get_mut(&job_id) {
                job.current_block = block_idx + 1;
                job.blocks_completed = block_idx + 1;
                job.block_receipts.push(receipt);
                job.level2_fallback_verified = job.level2_fallback_verified || l2_fallback_verified;
            }
        }

        // Brief yield to allow other Tokio tasks to run.
        tokio::task::yield_now().await;
    }

    // ── Verifying (Level 2 gates) ────────────────────────────────────────
    {
        let mut j = jobs.lock().await;
        if let Some(job) = j.get_mut(&job_id) {
            job.state = DistillationState::Verifying;
        }
    }

    // TODO(#distill): Wire real speculative decoding acceptance measurement
    // once the ternary MTP drafter is loaded. The gate currently FAILS by
    // default to prevent silent pass-through without measurement.
    let ja = check_joint_acceptance_rate(
        &AcceptanceThresholds::default(),
        None,
    );

    {
        let mut j = jobs.lock().await;
        if let Some(job) = j.get_mut(&job_id) {
            job.joint_acceptance_result = Some(ja);
            if job.joint_acceptance_result.as_ref().map(|r| r.passed).unwrap_or(false) {
                job.state = DistillationState::Completed;
            } else {
                // Gate failed — still mark Verifying so the user can inspect receipts.
                // The joint acceptance gate requires MTP drafter to pass.
            }
        }
    }

    broker.set_mode(ServerOperationalMode::Idle);
}
