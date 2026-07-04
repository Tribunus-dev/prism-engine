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

use crate::compilation::level1::checkpoint::validate_teacher_checkpoint_against_ternary;
use crate::compilation::level1::gates::check_numerical;
use crate::compilation::level1::scheduler::{Level1Config, Level1Scheduler};
use crate::compilation::level2::bridge::CoreMLTeacher;
use crate::compilation::level2::compiler::ensure_teacher_bundles;
use crate::compilation::level2::gates::{
    check_joint_acceptance_rate, AcceptanceThresholds, JointAcceptanceResult,
};
use crate::compilation::level2::scheduler::Level2Scheduler;
use crate::compilation::level3::gates::run_all_gates as run_level3_gates;
use crate::compilation::level3::routing::Level3Router;
use crate::compilation::memory_budget::MemoryBudget;
use crate::compilation::phase_types::{
    ElementType, PhysicalLayout, ProviderKind, ResidencyClass, TensorDescriptor,
};
use crate::compilation::receipt::{BlockReceipt, EngineExecutionLog};
use crate::server::state::{MemoryAllocationBroker, ServerOperationalMode};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
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
    pub checkpoint_validation_passed: bool,
    pub level2_used: bool,
    pub level2_fallback_verified: bool,
    pub joint_acceptance_rate: Option<f64>,
    pub joint_acceptance_passed: bool,
    pub level3_pass: bool,
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
    checkpoint_validation_passed: bool,
    level2_used: bool,
    level2_fallback_verified: bool,
    joint_acceptance_result: Option<JointAcceptanceResult>,
    level3_pass: bool,
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
        let mut request = request;
        let mut jobs = self.jobs.lock().await;
        let job_id = request.job_id.clone();
        if jobs.contains_key(&job_id) {
            return Err(format!("job {} already exists", job_id));
        }
        if request.model_dir.is_none() {
            if let Some(path) = default_level2_model_dir(&job_id) {
                request.model_dir = Some(path);
            }
        }
        let total_blocks = 48;
        let has_model_dir = request.model_dir.is_some();
        let job = DistillationJob {
            request: request.clone(),
            state: DistillationState::Queued,
            current_block: 0,
            blocks_completed: 0,
            block_receipts: Vec::with_capacity(total_blocks),
            checkpoint_validation_passed: false,
            level2_used: false,
            level2_fallback_verified: false,
            joint_acceptance_result: None,
            level3_pass: false,
            error: None,
        };
        jobs.insert(job_id.clone(), job);
        let j = self.jobs.clone();
        let b = self.memory_broker.clone();
        let jid = job_id.clone();
        let teacher_checkpoint = request.teacher_checkpoint.clone();
        let md = request.model_dir.clone();
        let enabled = has_model_dir;
        tokio::spawn(async move {
            run_distillation_loop(j, b, jid, total_blocks, teacher_checkpoint, md, enabled).await;
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
            checkpoint_validation_passed: j.checkpoint_validation_passed,
            level2_used: j.level2_used,
            level2_fallback_verified: j.level2_fallback_verified,
            joint_acceptance_rate: j
                .joint_acceptance_result
                .as_ref()
                .map(|r| r.acceptance_rate),
            joint_acceptance_passed: j
                .joint_acceptance_result
                .as_ref()
                .map(|r| r.passed)
                .unwrap_or(false),
            level3_pass: j.level3_pass,
            error: j.error.clone(),
        })
    }

    pub fn memory_broker(&self) -> &Arc<MemoryAllocationBroker> {
        &self.memory_broker
    }
}

#[cfg(all(target_os = "macos", feature = "prism-backend"))]
fn default_level2_model_dir(job_id: &str) -> Option<String> {
    Some(
        std::env::temp_dir()
            .join("prism-distill-coreml")
            .join(job_id)
            .join("teacher-models")
            .to_string_lossy()
            .to_string(),
    )
}

#[cfg(not(all(target_os = "macos", feature = "prism-backend")))]
fn default_level2_model_dir(_job_id: &str) -> Option<String> {
    None
}

#[cfg(all(target_os = "macos", feature = "prism-backend"))]
fn prepare_level2_model_dir(
    job_id: &str,
    model_dir: &str,
    config: &Level1Config,
    total_microbatches: usize,
) -> Result<PathBuf, String> {
    let path = PathBuf::from(model_dir);
    ensure_teacher_bundles(&path, config.hidden_dim, total_microbatches)
        .map_err(|e| format!("job {job_id}: prepare Core ML teacher bundles: {e}"))?;
    Ok(path)
}

#[cfg(not(all(target_os = "macos", feature = "prism-backend")))]
fn prepare_level2_model_dir(
    _job_id: &str,
    model_dir: &str,
    _config: &Level1Config,
    _total_microbatches: usize,
) -> Result<PathBuf, String> {
    Ok(PathBuf::from(model_dir))
}

// ── Level 2 helper (gated on macOS + prism-backend) ──────────────────────

/// Run the Level 2 Core ML teacher pipeline for one block.
/// Returns (peak_bytes, fallback_verified) or None if not available.
#[cfg(all(target_os = "macos", feature = "prism-backend"))]
fn run_level2_block(
    config: &Level1Config,
    model_dir: &str,
    _block_idx: usize,
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
    teacher_checkpoint: String,
    model_dir: Option<String>,
    level2_requested: bool,
) {
    broker.set_mode(ServerOperationalMode::Distilling);
    let ceiling = MemoryAllocationBroker::DISTILL_SUB_CEILING_BYTES;
    const LEVEL2_PIPELINE_MICROBATCHES: usize = 8;
    let config = Level1Config {
        microbatch: 4096,
        hidden_dim: 3840,
        pages_per_row: 2,
        budget: MemoryBudget::m1_16gb_default(),
        objective_weights: None,
    };

    // ── Ingesting ────────────────────────────────────────────────────────
    {
        let mut j = jobs.lock().await;
        if let Some(job) = j.get_mut(&job_id) {
            job.state = DistillationState::Ingesting;
            job.level2_used = level2_requested;
        }
    }

    let prepared_model_dir = if let Some(ref model_dir) = model_dir {
        match prepare_level2_model_dir(&job_id, model_dir, &config, LEVEL2_PIPELINE_MICROBATCHES) {
            Ok(path) => Some(path),
            Err(error) => {
                let mut j = jobs.lock().await;
                if let Some(job) = j.get_mut(&job_id) {
                    job.state = DistillationState::Failed;
                    job.error = Some(error);
                }
                broker.set_mode(ServerOperationalMode::Idle);
                return;
            }
        }
    } else {
        None
    };

    let checkpoint_validation = tokio::task::spawn_blocking({
        let teacher_checkpoint = teacher_checkpoint.clone();
        move || {
            validate_teacher_checkpoint_against_ternary(std::path::Path::new(&teacher_checkpoint))
        }
    })
    .await
    .map_err(|error| format!("checkpoint validation task join error: {}", error));

    match checkpoint_validation {
        Err(error) => {
            let mut j = jobs.lock().await;
            if let Some(job) = j.get_mut(&job_id) {
                job.state = DistillationState::Failed;
                job.error = Some(error);
            }
            broker.set_mode(ServerOperationalMode::Idle);
            return;
        }
        Ok(result) => match result {
            Ok(result) if result.passed => {
                let mut j = jobs.lock().await;
                if let Some(job) = j.get_mut(&job_id) {
                    job.checkpoint_validation_passed = true;
                }
            }
            Ok(result) => {
                let mut j = jobs.lock().await;
                if let Some(job) = j.get_mut(&job_id) {
                    job.state = DistillationState::Failed;
                    job.error = Some(result.failure_reason.unwrap_or_else(|| {
                    format!(
                        "checkpoint validation failed after {} sampled layers and {} sampled projections",
                        result.validated_layers, result.validated_projections
                    )
                }));
                }
                broker.set_mode(ServerOperationalMode::Idle);
                return;
            }
            Err(error) => {
                let mut j = jobs.lock().await;
                if let Some(job) = j.get_mut(&job_id) {
                    job.state = DistillationState::Failed;
                    job.error = Some(format!(
                        "checkpoint-backed teacher validation failed for {}: {}",
                        teacher_checkpoint, error
                    ));
                }
                broker.set_mode(ServerOperationalMode::Idle);
                return;
            }
        },
    }

    // ── Compiling (block-by-block) ───────────────────────────────────────
    for block_idx in 0..total_blocks {
        let block_started_at = Instant::now();
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
        let mut l1 = Level1Scheduler::new(config.clone(), LEVEL2_PIPELINE_MICROBATCHES);
        l1.initialize();
        while l1.step() {}

        let mse = l1.reducer().output_mse.unwrap_or(f64::INFINITY);
        let cosine = l1.reducer().cosine_similarity.unwrap_or(0.0);
        let residual = l1
            .reducer()
            .residual_relative_error
            .unwrap_or(f64::INFINITY);
        let l1_peak = l1.peak_memory();

        // ── Level 2: Core ML teacher (when available) ──────────────────────
        let mut l2_peak = 0u64;
        let mut l2_fallback_verified = false;

        if let Some(ref md) = prepared_model_dir {
            if let Some((pk, fv)) = run_level2_block(&config, &md.to_string_lossy(), block_idx) {
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
        numerical_drift.insert(
            "gate_passed".into(),
            if num_result.passed { 1.0 } else { 0.0 },
        );

        let receipt = BlockReceipt {
            block_index: block_idx,
            modality_tag: "text".into(),
            input_frontier_hash: String::new(),
            output_frontier_hash: String::new(),
            sidecar_fraction: 0.0,
            optimal_scale_dtype: "two_level_int8".into(),
            numerical_drift,
            execution_provenance: EngineExecutionLog {
                backend_requested: if level2_requested {
                    "Level2-CoreML".into()
                } else {
                    "Level1-Metal".into()
                },
                backend_observed: if l2_fallback_verified {
                    "Level2-CoreML".into()
                } else {
                    "Level1-Metal".into()
                },
                zero_copy_verified: false,
                wall_time_ms: block_started_at.elapsed().as_secs_f64() * 1000.0,
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

    let ja = check_joint_acceptance_rate(&AcceptanceThresholds::default(), None);

    {
        let mut j = jobs.lock().await;
        if let Some(job) = j.get_mut(&job_id) {
            job.joint_acceptance_result = Some(ja);
            if job
                .joint_acceptance_result
                .as_ref()
                .map(|r| r.passed)
                .unwrap_or(false)
            {
                job.state = DistillationState::Completed;
            } else {
                // Gate failed — still mark Verifying so the user can inspect receipts.
                // The joint acceptance gate requires MTP drafter to pass.
            }
        }
    }

    // ── Level 3: Bridge provider validation ─────────────────────────────
    let l3_cert = {
        let router = Level3Router::new();
        let source = TensorDescriptor {
            logical_shape: vec![1, 1, 1],
            element_type: ElementType::F32,
            physical_layout: PhysicalLayout::DenseRowMajor,
            alignment: 64,
            producer_phase: None,
            consumer_phases: vec![],
            permitted_providers: vec![ProviderKind::CoreML, ProviderKind::Metal],
            residency_class: ResidencyClass::Unified,
            max_bytes: 0,
            mutable: false,
            content_digest: None,
        };
        run_level3_gates(&router, &source)
    };
    {
        let mut j = jobs.lock().await;
        if let Some(job) = j.get_mut(&job_id) {
            job.level3_pass = l3_cert.level3_pass;
        }
    }

    broker.set_mode(ServerOperationalMode::Idle);
}
