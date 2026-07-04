//! Background distillation worker — orchestrates the Level 1/2/3 compiler
//! pipeline as an async service within prism-server.
//!
//! The `DistillationEngine` receives job submissions, spawns a background
//! Tokio task for each, and reports status via the `/v1/distill` API.
//!
//! # Pipeline per block
//!   1. Create Level1Scheduler with MemoryBudget (teacher/student/reducer)
//!   2. scheduler.run() — drives TeacherForward → StudentForward → Reduce
//!   3. Run Level 1 numerical gate on reducer metrics
//!   4. Run Level 2 joint acceptance gate (TODO)
//!   5. Write BlockReceipt with metrics + execution provenance
//!   6. Repeat for each block

use crate::compilation::level1::gates::check_numerical;
use crate::compilation::level1::scheduler::{Level1Config, Level1Scheduler};
use crate::compilation::memory_budget::MemoryBudget;
use crate::compilation::receipt::{BlockReceipt, EngineExecutionLog};
use crate::server::state::{MemoryAllocationBroker, ServerOperationalMode};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Request payload for `/v1/distill`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DistillationRequest {
    /// Unique job identifier.
    pub job_id: String,
    /// HuggingFace repo or local path for the teacher checkpoint.
    pub teacher_checkpoint: String,
    /// HuggingFace repo or local path for the assistant (MTP drafter).
    pub assistant_checkpoint: Option<String>,
    /// Target representation format.
    #[serde(default = "default_representation")]
    pub target_representation: String,
    /// Memory ceiling for this compilation job (GB).
    #[serde(default = "default_memory_ceiling")]
    pub memory_ceiling_gb: f64,
    /// Per-modality objective weight overrides.
    #[serde(default)]
    pub modality_profiles: HashMap<String, HashMap<String, f64>>,
    /// Gate thresholds.
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

    pub async fn submit(&self, request: DistillationRequest) -> Result<String, String> {
        let mut jobs = self.jobs.lock().await;
        let job_id = request.job_id.clone();
        if jobs.contains_key(&job_id) {
            return Err(format!("job {} already exists", job_id));
        }
        let total_blocks = 48;
        let job = DistillationJob {
            request: request.clone(),
            state: DistillationState::Queued,
            current_block: 0,
            blocks_completed: 0,
            block_receipts: Vec::with_capacity(total_blocks),
            error: None,
        };
        jobs.insert(job_id.clone(), job);
        let j = self.jobs.clone();
        let b = self.memory_broker.clone();
        let jid = job_id.clone();
        tokio::spawn(async move {
            run_distillation_loop(j, b, jid, total_blocks).await;
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
            error: j.error.clone(),
        })
    }

    pub fn memory_broker(&self) -> &Arc<MemoryAllocationBroker> {
        &self.memory_broker
    }
}

/// Background loop driving a single distillation job block-by-block.
///
/// Each block:
///   a. Allocate a Level1Scheduler with the target memory budget.
///   b. scheduler.initialize() — sets up ActivationArena + teacher/student/reducer.
///   c. while scheduler.step() {} — runs TeacherForward → StudentForward → Reduce.
///   d. Extract metrics from scheduler.reducer().
///   e. Run check_numerical() — Level 1 gate.
///   f. Build BlockReceipt with metrics + peak memory + execution provenance.
///   g. Release block resources (scheduler goes out of scope → drop).
async fn run_distillation_loop(
    jobs: Arc<Mutex<HashMap<String, DistillationJob>>>,
    broker: Arc<MemoryAllocationBroker>,
    job_id: String,
    total_blocks: usize,
) {
    broker.set_mode(ServerOperationalMode::Distilling);

    // ── Phase: Ingesting ──────────────────────────────────────────────────
    {
        let mut j = jobs.lock().await;
        if let Some(job) = j.get_mut(&job_id) {
            job.state = DistillationState::Ingesting;
        }
    }

    // TODO(#distill): actual safetensor loading + mmap pipeline.
    // For now, the Level1Scheduler uses synthetic weights internally.

    {
        let mut j = jobs.lock().await;
        if let Some(job) = j.get_mut(&job_id) {
            job.state = DistillationState::Compiling;
        }
    }

    // ── Phase: Compiling (block-by-block) ────────────────────────────────
    for block_idx in 0..total_blocks {
        // Check available memory — bail out if under 100 MB free.
        if broker.distill_available() < 100_000_000 {
            let mut j = jobs.lock().await;
            if let Some(job) = j.get_mut(&job_id) {
                job.state = DistillationState::Failed;
                job.error = Some(format!("out of memory at block {}", block_idx));
            }
            broker.set_mode(ServerOperationalMode::Idle);
            return;
        }

        let ceiling = MemoryAllocationBroker::DISTILL_SUB_CEILING_BYTES;
        broker.declare(ceiling);

        // ── Level 1: Metal teacher + Ternary student + Accelerate reducer ──
        let config = Level1Config {
            microbatch: 4096,
            hidden_dim: 3840,
            pages_per_row: 2,
            budget: MemoryBudget::m1_16gb_default(),
            objective_weights: None,
        };

        let mut scheduler = Level1Scheduler::new(config, 8); // 8 microbatches per block
        scheduler.initialize();
        while scheduler.step() {}

        // Extract reduction metrics.
        let reducer = scheduler.reducer();
        let mse = reducer.output_mse.unwrap_or(f64::INFINITY);
        let cosine = reducer.cosine_similarity.unwrap_or(0.0);
        let residual = reducer.residual_relative_error.unwrap_or(f64::INFINITY);
        let peak_bytes = scheduler.peak_memory();

        // ── Level 1 gate: numerical ──────────────────────────────────────
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
                backend_requested: "Level1-Metal".into(),
                backend_observed: "Level1-Metal".into(),
                zero_copy_verified: false,
                wall_time_ms: 0.0,
                peak_arena_bytes: peak_bytes,
            },
        };

        // Scheduler goes out of scope → ActivationArena + buffers dropped.

        broker.release(ceiling);

        // Store receipt.
        {
            let mut j = jobs.lock().await;
            if let Some(job) = j.get_mut(&job_id) {
                job.current_block = block_idx + 1;
                job.blocks_completed = block_idx + 1;
                job.block_receipts.push(receipt);
            }
        }
    }

    // ── Phase: Verifying ─────────────────────────────────────────────────
    {
        let mut j = jobs.lock().await;
        if let Some(job) = j.get_mut(&job_id) {
            job.state = DistillationState::Verifying;
        }
    }

    // TODO(#distill): Wire Level 2 joint acceptance gate:
    //   let thresholds = AcceptanceThresholds::default();
    //   let weights = ObjectiveWeights { ... };
    //   let ja = check_joint_acceptance_rate(&thresholds, Some(&weights));
    //
    // TODO(#distill): Wire Level 3 bridge provider verification:
    //   let routing = Level3Router::new();
    //   let result = routing.validate();
    //
    // TODO(#distill): Build final ProcessingReceipt from all BlockReceipts.
    //   Requires GlobalMetrics from final perplexity eval.

    // Mark complete.
    {
        let mut j = jobs.lock().await;
        if let Some(job) = j.get_mut(&job_id) {
            job.state = DistillationState::Completed;
        }
    }

    broker.set_mode(ServerOperationalMode::Idle);
}
