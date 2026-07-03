//! Background distillation worker — orchestrates the multi-resolution compiler
//! pipeline as an async service within prism-server.
//!
//! The `DistillationEngine` receives job submissions, spawns a background
//! Tokio task for each, and reports status + receipts via the `/v1/distill`
//! API endpoint.
//!
//! # Memory model
//! Tensors are streamed from disk via `memmap2`, processed block-by-block
//! within the `ActivationArena` out-of-core frontier, and dropped after
//! each block — the full teacher or student model is never resident.

use crate::compilation::receipt::ProcessingReceipt;
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
    pub current_block: usize,
    pub total_blocks: usize,
    pub blocks_completed: usize,
    pub receipts: Vec<ProcessingReceipt>,
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
    receipts: Vec<ProcessingReceipt>,
    error: Option<String>,
}

/// Shared engine that manages all active distillation jobs.
pub struct DistillationEngine {
    jobs: Arc<Mutex<HashMap<String, DistillationJob>>>,
    memory_broker: Arc<MemoryAllocationBroker>,
}

impl DistillationEngine {
    /// Create a new engine sharing the server's memory broker.
    pub fn new(memory_broker: Arc<MemoryAllocationBroker>) -> Self {
        DistillationEngine {
            jobs: Arc::new(Mutex::new(HashMap::new())),
            memory_broker,
        }
    }

    /// Submit a new distillation job. Returns the job_id on success.
    ///
    /// The job is queued immediately and a background task is spawned.
    pub async fn submit(
        &self,
        request: DistillationRequest,
    ) -> Result<String, String> {
        let mut jobs = self.jobs.lock().await;
        let job_id = request.job_id.clone();

        if jobs.contains_key(&job_id) {
            return Err(format!("job {} already exists", job_id));
        }

        let total_blocks = 48; // TODO(#distill): resolve from model config
        let job = DistillationJob {
            request: request.clone(),
            state: DistillationState::Queued,
            current_block: 0,
            blocks_completed: 0,
            receipts: Vec::with_capacity(total_blocks),
            error: None,
        };
        jobs.insert(job_id.clone(), job);

        // Spawn the background worker.
        let jobs = self.jobs.clone();
        let broker = self.memory_broker.clone();
        let job_id_clone = job_id.clone();
        tokio::spawn(async move {
            run_distillation_loop(jobs, broker, job_id_clone, total_blocks).await;
        });

        Ok(job_id)
    }

    /// Get the status of a job by its ID.
    pub async fn status(&self, job_id: &str) -> Option<DistillationJobStatus> {
        let jobs = self.jobs.lock().await;
        jobs.get(job_id).map(|job| DistillationJobStatus {
            job_id: job_id.to_string(),
            state: job.state.clone(),
            current_block: job.current_block,
            total_blocks: 48, // TODO(#distill): resolve from model config
            blocks_completed: job.blocks_completed,
            receipts: job.receipts.clone(),
            error: job.error.clone(),
        })
    }

    /// Get a shared reference to the memory broker.
    pub fn memory_broker(&self) -> &Arc<MemoryAllocationBroker> {
        &self.memory_broker
    }
}

/// Background loop that drives a single distillation job.
///
/// # Pipeline
/// 1. Transition to `Distilling` mode on the memory broker.
/// 2. For each block (0..total_blocks):
///    a. Map safetensors via `memmap2` into the activation arena.
///    b. Run TeacherForward (Core ML or Accelerate teacher).
///    c. Run StudentCandidateForward (Metal ternary student).
///    d. Run ActivationCompare (Accelerate reducer).
///    e. Run TritCommit (ternary page commit).
///    f. Run ReceiptSeal (produce processing receipt).
///    g. Drop all temporary tensors.
/// 3. Run global verification gates.
/// 4. Transition back to `Idle`.
async fn run_distillation_loop(
    jobs: Arc<Mutex<HashMap<String, DistillationJob>>>,
    broker: Arc<MemoryAllocationBroker>,
    job_id: String,
    total_blocks: usize,
) {
    // Transition to distilling mode.
    broker.set_mode(ServerOperationalMode::Distilling);

    // Update state to Ingesting.
    {
        let mut jobs = jobs.lock().await;
        if let Some(job) = jobs.get_mut(&job_id) {
            job.state = DistillationState::Compiling;
        }
    }

    for block_idx in 0..total_blocks {
        // Check available memory — bail if we're over budget.
        if broker.distill_available() < 100_000_000 {
            // Less than 100 MB left — abort.
            let mut jobs = jobs.lock().await;
            if let Some(job) = jobs.get_mut(&job_id) {
                job.state = DistillationState::Failed;
                job.error = Some(format!(
                    "out of memory at block {}",
                    block_idx
                ));
            }
            broker.set_mode(ServerOperationalMode::Idle);
            return;
        }

        // TODO(#distill): Wire real Level 1/2/3 pipeline here.
        //   1. Map safetensors -> ActivationArena
        //   2. CoreAiTeacher::predict() for teacher output
        //   3. Metal student forward (ternary tile640)
        //   4. AccelerateReducer::reduce() for metrics
        //   5. ObjectiveWeights::resolve(modality) -> weighted loss
        //   6. Write receipt
        broker.declare(256_000_000); // ~256 MB per block

        // Simulated block work — replace with real pipeline calls.
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        broker.release(256_000_000);

        // Update progress.
        let mut jobs = jobs.lock().await;
        if let Some(job) = jobs.get_mut(&job_id) {
            job.current_block = block_idx + 1;
            job.blocks_completed = block_idx + 1;
            if block_idx == total_blocks - 1 {
                job.state = DistillationState::Completed;
            }
        }
    }

    // TODO(#distill): Run verification gates (check_numerical, modality distributions, joint acceptance).

    broker.set_mode(ServerOperationalMode::Idle);
}
