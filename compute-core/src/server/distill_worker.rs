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
use crate::compilation::level1::kd_gate::{
    compute_calibration_logits, kd_available, kd_gate, load_calibration_tokens,
    score_student_logits, KdGateConfig, KdGateResult, KdReport, ParityRun, ParityThresholds,
};
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

    // ── Real-teacher KD scoring (Gemma4Teacher → distill_core) ──────────
    /// Compiled ternary student `.cimage` to score against the teacher. When
    /// set (and this build can run Metal), the loop runs teacher-forced passes
    /// over the calibration tokens on BOTH cimages and gates on KD divergence.
    #[serde(default)]
    pub student_checkpoint: Option<String>,
    /// File of comma/whitespace-separated u32 token IDs. Omitted → the
    /// deterministic built-in stream (same LCG as prism-bench-ab).
    #[serde(default)]
    pub calibration_tokens_path: Option<String>,
    /// Built-in stream length (default 128 → ~134 MB of held logits per model
    /// at vocab 262144).
    #[serde(default)]
    pub calibration_len: Option<usize>,
    /// Built-in stream token-ID cap (keep below every model's vocab).
    #[serde(default)]
    pub calibration_vocab_cap: Option<u32>,
    /// KD softmax temperature (default 2.0).
    #[serde(default)]
    pub kd_temperature: Option<f32>,
    /// Gate: fail if mean KD exceeds this (default 0.75).
    #[serde(default)]
    pub kd_max: Option<f32>,
    /// Gate: fail if top-1 agreement falls below this (default 0.55).
    #[serde(default)]
    pub kd_min_top1: Option<f32>,

    // ── Pipelined parity validator (STAGE0_TAPS_SPEC.md) ────────────────
    /// Directory of golden tap dumps: `token_{i:05}.bin`, each the
    /// concatenated f32-LE tap slots (2·layers+2 slots × hidden) for
    /// calibration token i, produced by the bf16 anchor oracle. When set
    /// (and this build can run Metal with TRIBUNUS_TAPS=1), the loop runs
    /// the pipelined parity audit over the calibration stream.
    #[serde(default)]
    pub parity_golden_dir: Option<String>,
    /// Hard breach threshold (rel-L2; default 0.35). Breach ⇒ between-token
    /// early exit + taint dump + job failure.
    #[serde(default)]
    pub parity_hard: Option<f64>,
    /// Warn threshold (rel-L2; default 0.10) — telemetry only.
    #[serde(default)]
    pub parity_warn: Option<f64>,
    /// `.parity` sidecar path (default `<teacher_checkpoint>.parity.json`).
    #[serde(default)]
    pub parity_output: Option<String>,
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
    /// Model-level KD (T²·KL) of the ternary student vs the NF4 teacher over
    /// the calibration stream. `None` until the KD stage has run (or when no
    /// student_checkpoint was provided / this build cannot run Metal).
    pub kd_divergence: Option<f32>,
    pub kd_top1_agreement: Option<f32>,
    pub kd_worst_window: Option<f32>,
    pub kd_gate_passed: Option<bool>,
    /// Why KD scoring was skipped, when it was (e.g. non-Metal build).
    pub kd_skipped_reason: Option<String>,
    /// SHA-256 of the `.parity` sidecar (also stamped into each BlockReceipt).
    pub parity_digest: Option<String>,
    pub parity_passed: Option<bool>,
    pub parity_tokens_validated: Option<usize>,
    pub parity_skipped_reason: Option<String>,
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
    kd_report: Option<KdReport>,
    kd_gate_result: Option<KdGateResult>,
    kd_skipped_reason: Option<String>,
    parity_run: Option<ParityRun>,
    parity_digest: Option<String>,
    parity_skipped_reason: Option<String>,
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
            kd_report: None,
            kd_gate_result: None,
            kd_skipped_reason: None,
            parity_run: None,
            parity_digest: None,
            parity_skipped_reason: None,
            error: None,
        };
        jobs.insert(job_id.clone(), job);
        let j = self.jobs.clone();
        let b = self.memory_broker.clone();
        let jid = job_id.clone();
        let req = request.clone();
        let enabled = has_model_dir;
        tokio::spawn(async move {
            run_distillation_loop(j, b, jid, total_blocks, req, enabled).await;
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
            kd_divergence: j.kd_report.as_ref().map(|r| r.kd),
            kd_top1_agreement: j.kd_report.as_ref().map(|r| r.top1),
            kd_worst_window: j.kd_report.as_ref().map(|r| r.worst_window_kd),
            kd_gate_passed: j.kd_gate_result.as_ref().map(|g| g.passed),
            kd_skipped_reason: j.kd_skipped_reason.clone(),
            parity_digest: j.parity_digest.clone(),
            parity_passed: j.parity_run.as_ref().map(|r| r.all_passed()),
            parity_tokens_validated: j.parity_run.as_ref().map(|r| r.tokens_validated()),
            parity_skipped_reason: j.parity_skipped_reason.clone(),
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

/// Run the real-teacher KD stage: load the NF4 teacher `.cimage` via
/// [`Gemma4Teacher`], run `teacher_forced` over the calibration tokens, do the
/// same for the ternary student `.cimage`, and score with
/// `distill_core::{kd_divergence, top1_agreement}`.
///
/// Blocking (drives the Metal megakernel) — call via `spawn_blocking`. The two
/// orchestrators are loaded sequentially (teacher drops before student loads).
/// Returns `Ok(None)` when no student checkpoint was requested.
fn run_kd_stage(
    request: &DistillationRequest,
) -> Result<Option<(KdReport, KdGateResult)>, String> {
    let Some(student_ckpt) = request.student_checkpoint.as_ref() else {
        return Ok(None);
    };
    let cfg = KdGateConfig {
        temperature: request.kd_temperature.unwrap_or(2.0),
        max_kd: request.kd_max.unwrap_or(0.75),
        min_top1: request.kd_min_top1.unwrap_or(0.55),
        ..KdGateConfig::default()
    };
    let tokens = load_calibration_tokens(
        request
            .calibration_tokens_path
            .as_deref()
            .map(std::path::Path::new),
        request.calibration_len.unwrap_or(128),
        request.calibration_vocab_cap.unwrap_or(1000),
    )?;

    let teacher = compute_calibration_logits(
        std::path::Path::new(&request.teacher_checkpoint),
        &tokens,
    )
    .map_err(|e| format!("teacher logits: {e}"))?;
    let student = compute_calibration_logits(std::path::Path::new(student_ckpt), &tokens)
        .map_err(|e| format!("student logits: {e}"))?;

    let report = score_student_logits(&teacher, &student, &cfg)?;
    let verdict = kd_gate(&report, &cfg);
    Ok(Some((report, verdict)))
}

/// Whether the pipelined parity audit can run in this build (needs the Metal
/// megakernel with Stage 0 taps).
const fn parity_available() -> bool {
    cfg!(all(target_os = "macos", feature = "prism-backend"))
}

/// Load one token's golden tap slots: `<dir>/token_{i:05}.bin`, the
/// concatenated f32-LE slots produced by the bf16 anchor oracle.
#[cfg(all(target_os = "macos", feature = "prism-backend"))]
fn load_golden_slots(
    dir: &std::path::Path,
    token_index: u64,
    slots: usize,
    hidden: usize,
) -> Result<Vec<Vec<f32>>, String> {
    let path = dir.join(format!("token_{token_index:05}.bin"));
    let bytes = std::fs::read(&path).map_err(|e| format!("read golden {}: {e}", path.display()))?;
    let expect = slots * hidden * 4;
    if bytes.len() != expect {
        return Err(format!(
            "golden {} is {} bytes, expected {expect} ({slots} slots × {hidden} × f32) — \
             wrong model geometry or truncated dump",
            path.display(),
            bytes.len()
        ));
    }
    Ok((0..slots)
        .map(|si| {
            bytes[si * hidden * 4..(si + 1) * hidden * 4]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        })
        .collect())
}

/// Run the pipelined parity audit: decode the calibration stream through the
/// taps-enabled teacher megakernel while a validator thread scores each
/// token's taps against the bf16 anchor goldens (STAGE0_TAPS_SPEC.md).
///
/// Pipelining: the validator consumes token t−1 while the GPU decodes token
/// t; a hard breach flips `stop` and the main loop exits BETWEEN tokens
/// (no new submissions), preserving the taint. On breach the failing token's
/// raw tap slots are dumped beside the sidecar.
///
/// Returns `(run, digest, sidecar_path)`; `Ok(None)` when not requested.
#[cfg(all(target_os = "macos", feature = "prism-backend"))]
fn run_parity_stage(
    request: &DistillationRequest,
    total_blocks: usize,
) -> Result<Option<(ParityRun, String, PathBuf)>, String> {
    use crate::compilation::level1::kd_gate::validate_token_taps;
    use crate::compute_image::orchestrator::Orchestrator;
    use std::sync::atomic::{AtomicBool, Ordering};

    let Some(golden_dir) = request.parity_golden_dir.as_ref() else {
        return Ok(None);
    };
    if std::env::var("TRIBUNUS_TAPS").as_deref() != Ok("1") {
        return Err(
            "parity audit requires TRIBUNUS_TAPS=1 in the environment BEFORE the worker \
             constructs the teacher (taps compile into the persistent kernel at construction)"
                .into(),
        );
    }
    let layers = total_blocks as u32;
    let slots = 2 * total_blocks + 2;
    let thresholds = ParityThresholds {
        hard: request.parity_hard.unwrap_or(0.35),
        warn: request.parity_warn.unwrap_or(0.10),
    };
    let tokens = load_calibration_tokens(
        request
            .calibration_tokens_path
            .as_deref()
            .map(std::path::Path::new),
        request.calibration_len.unwrap_or(128),
        request.calibration_vocab_cap.unwrap_or(1000),
    )?;
    let golden_dir = PathBuf::from(golden_dir);
    let sidecar = request
        .parity_output
        .clone()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(format!("{}.parity.json", request.teacher_checkpoint)));

    let mut orch = Orchestrator::from_cimage(
        std::path::Path::new(&request.teacher_checkpoint),
        1,
        false,
    )
    .map_err(|e| format!("load teacher for parity audit: {e}"))?;

    let stop = std::sync::Arc::new(AtomicBool::new(false));
    let (tx, rx) = std::sync::mpsc::channel::<(u64, Vec<Vec<f32>>)>();
    let v_stop = std::sync::Arc::clone(&stop);
    let v_dir = golden_dir.clone();
    let validator = std::thread::spawn(
        move || -> Result<(ParityRun, Option<(u64, Vec<Vec<f32>>)>), String> {
            let mut run = ParityRun::new(thresholds);
            let mut taint = None;
            for (idx, actual) in rx {
                let hidden = actual.first().map(Vec::len).unwrap_or(0);
                let golden = load_golden_slots(&v_dir, idx, actual.len(), hidden)?;
                let manifest = validate_token_taps(idx, layers, &actual, &golden, thresholds)?;
                if !run.push(manifest) {
                    taint = Some((idx, actual));
                    v_stop.store(true, Ordering::Release);
                    break;
                }
            }
            Ok((run, taint))
        },
    );

    for (i, &tok) in tokens.iter().enumerate() {
        // Between-token early exit: a hard breach on token t−1 stops token
        // t+1 from ever being submitted; token t (in flight) completes.
        if stop.load(Ordering::Acquire) {
            break;
        }
        let (_next, _logits, taps) = orch
            .decode_token_logits_with_taps(tok)
            .map_err(|e| format!("tapped decode @ token {i}: {e}"))?;
        let mut slot_vec = Vec::with_capacity(slots);
        slot_vec.push(taps.post_embed());
        for k in 0..total_blocks {
            slot_vec.push(taps.post_attention(k));
            slot_vec.push(taps.post_layer(k));
        }
        slot_vec.push(taps.final_hidden());
        if tx.send((i as u64, slot_vec)).is_err() {
            break; // validator ended (error path) — join below surfaces it
        }
    }
    drop(tx);
    let (run, taint) = validator
        .join()
        .map_err(|_| "parity validator thread panicked".to_string())??;

    let json = run.to_parity_json()?;
    std::fs::write(&sidecar, &json)
        .map_err(|e| format!("write {}: {e}", sidecar.display()))?;
    let digest = run.parity_digest()?;
    if let Some((idx, slots_data)) = taint {
        // Taint dump: the failing token's raw tap slots, full fidelity.
        let taint_path = sidecar.with_extension("taint.bin");
        let mut bytes = Vec::with_capacity(slots_data.iter().map(|v| v.len() * 4).sum());
        for sv in &slots_data {
            for v in sv {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        std::fs::write(&taint_path, &bytes)
            .map_err(|e| format!("write taint {}: {e}", taint_path.display()))?;
        eprintln!(
            "[parity] HARD BREACH at token {idx}: taint → {}, manifests → {}",
            taint_path.display(),
            sidecar.display()
        );
    }
    Ok(Some((run, digest, sidecar)))
}

#[cfg(not(all(target_os = "macos", feature = "prism-backend")))]
fn run_parity_stage(
    _request: &DistillationRequest,
    _total_blocks: usize,
) -> Result<Option<(ParityRun, String, PathBuf)>, String> {
    Err("parity audit requires macOS + the prism-backend feature (Metal megakernel taps)".into())
}

/// Background loop driving a single distillation job block-by-block.
///
/// Job-level, before blocks:
///   0. KD stage (when `student_checkpoint` is set + Metal available):
///      NF4 teacher + ternary student teacher-forced over calibration tokens →
///      distill_core KD divergence / top-1 agreement → KD gate verdict
///
/// Per block:
///   a. Level1Scheduler: Metal teacher + ternary student + Accelerate reducer
///   b. Level2Scheduler (when model_dir is Some + macOS): Core ML teacher
///   c. check_numerical() — Level 1 gate
///   d. BlockReceipt with metrics (incl. model-level KD) + execution provenance
///   e. Release block resources
///
/// After all blocks:
///   f. check_joint_acceptance_rate() — Level 2 gate
///   g. KD gate enforcement — a failing KD gate fails the job (receipts remain)
///   h. Store final receipts + gate results
async fn run_distillation_loop(
    jobs: Arc<Mutex<HashMap<String, DistillationJob>>>,
    broker: Arc<MemoryAllocationBroker>,
    job_id: String,
    total_blocks: usize,
    request: DistillationRequest,
    level2_requested: bool,
) {
    let teacher_checkpoint = request.teacher_checkpoint.clone();
    let model_dir = request.model_dir.clone();
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

    // ── KD stage: real NF4 teacher vs ternary student (Gemma4Teacher) ────
    // Model-level end-to-end KD. Per-block *isolation* (block-swap KD) needs
    // the per-op forward (kernels/PER_OP_FORWARD_PLAN.md Stage 7); until then
    // every block receipt carries these model-level numbers.
    let kd_metrics: Option<(f32, f32, f32, bool)> = if request.student_checkpoint.is_none() {
        None
    } else if !kd_available() {
        let reason =
            "KD scoring skipped: requires macOS + prism-backend (Metal megakernel)".to_string();
        let mut j = jobs.lock().await;
        if let Some(job) = j.get_mut(&job_id) {
            job.kd_skipped_reason = Some(reason);
        }
        None
    } else {
        broker.declare(ceiling);
        let staged = tokio::task::spawn_blocking({
            let request = request.clone();
            move || run_kd_stage(&request)
        })
        .await;
        broker.release(ceiling);
        match staged {
            Err(join_err) => {
                let mut j = jobs.lock().await;
                if let Some(job) = j.get_mut(&job_id) {
                    job.state = DistillationState::Failed;
                    job.error = Some(format!("KD stage task join error: {join_err}"));
                }
                broker.set_mode(ServerOperationalMode::Idle);
                return;
            }
            Ok(Err(error)) => {
                let mut j = jobs.lock().await;
                if let Some(job) = j.get_mut(&job_id) {
                    job.state = DistillationState::Failed;
                    job.error = Some(format!("KD stage failed: {error}"));
                }
                broker.set_mode(ServerOperationalMode::Idle);
                return;
            }
            Ok(Ok(None)) => None,
            Ok(Ok(Some((report, verdict)))) => {
                let summary = (
                    report.kd,
                    report.top1,
                    report.worst_window_kd,
                    verdict.passed,
                );
                let mut j = jobs.lock().await;
                if let Some(job) = j.get_mut(&job_id) {
                    job.kd_report = Some(report);
                    job.kd_gate_result = Some(verdict);
                }
                Some(summary)
            }
        }
    };

    // ── Pipelined parity audit (Stage 0 taps vs bf16 anchor goldens) ─────
    let parity_metrics: Option<(f64, usize, bool)> = if request.parity_golden_dir.is_none() {
        None
    } else if !parity_available() {
        let reason =
            "parity audit skipped: requires macOS + prism-backend (Metal megakernel taps)"
                .to_string();
        let mut j = jobs.lock().await;
        if let Some(job) = j.get_mut(&job_id) {
            job.parity_skipped_reason = Some(reason);
        }
        None
    } else {
        broker.declare(ceiling);
        let staged = tokio::task::spawn_blocking({
            let request = request.clone();
            move || run_parity_stage(&request, total_blocks)
        })
        .await;
        broker.release(ceiling);
        match staged {
            Err(join_err) => {
                let mut j = jobs.lock().await;
                if let Some(job) = j.get_mut(&job_id) {
                    job.state = DistillationState::Failed;
                    job.error = Some(format!("parity stage task join error: {join_err}"));
                }
                broker.set_mode(ServerOperationalMode::Idle);
                return;
            }
            Ok(Err(error)) => {
                let mut j = jobs.lock().await;
                if let Some(job) = j.get_mut(&job_id) {
                    job.state = DistillationState::Failed;
                    job.error = Some(format!("parity stage failed: {error}"));
                }
                broker.set_mode(ServerOperationalMode::Idle);
                return;
            }
            Ok(Ok(None)) => None,
            Ok(Ok(Some((run, digest, sidecar)))) => {
                let summary = (run.worst_rel_l2, run.tokens_validated(), run.all_passed());
                eprintln!(
                    "[parity] {} tokens validated, worst rel-L2 {:.3e}, sidecar {}",
                    summary.1,
                    summary.0,
                    sidecar.display()
                );
                let mut j = jobs.lock().await;
                if let Some(job) = j.get_mut(&job_id) {
                    job.parity_run = Some(run);
                    job.parity_digest = Some(digest);
                }
                Some(summary)
            }
        }
    };
    let parity_digest_for_receipts: Option<String> = {
        let j = jobs.lock().await;
        j.get(&job_id).and_then(|job| job.parity_digest.clone())
    };

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
        // Model-level KD vs the real NF4 teacher (same value on every block
        // until block-swap KD lands — see the KD stage comment above).
        if let Some((kd, top1, worst, kd_passed)) = kd_metrics {
            numerical_drift.insert("kd_divergence".into(), kd);
            numerical_drift.insert("kd_top1_agreement".into(), top1);
            numerical_drift.insert("kd_worst_window".into(), worst);
            numerical_drift.insert(
                "kd_gate_passed".into(),
                if kd_passed { 1.0 } else { 0.0 },
            );
        }
        // Parity audit summary (same values on every block — the audit is
        // model-level per token; per-block isolation arrives with block-swap).
        if let Some((worst_rel_l2, tokens, passed)) = parity_metrics {
            numerical_drift.insert("parity_worst_rel_l2".into(), worst_rel_l2 as f32);
            numerical_drift.insert("parity_tokens_validated".into(), tokens as f32);
            numerical_drift.insert("parity_passed".into(), if passed { 1.0 } else { 0.0 });
        }

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
            parity_digest: parity_digest_for_receipts.clone(),
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
            let kd_ok = job
                .kd_gate_result
                .as_ref()
                .map(|g| g.passed)
                .unwrap_or(true); // no KD stage requested → gate vacuously passes
            let parity_ok = job
                .parity_run
                .as_ref()
                .map(|r| r.all_passed())
                .unwrap_or(true); // no parity stage requested → vacuously ok
            if !parity_ok {
                job.state = DistillationState::Failed;
                let stopped = job
                    .parity_run
                    .as_ref()
                    .and_then(|r| r.stopped_at_token)
                    .map(|t| format!(" (hard breach at token {t}; taint dumped)"))
                    .unwrap_or_default();
                job.error = Some(format!(
                    "parity audit failed: teacher taps diverged from the bf16 anchor \
                     beyond the hard threshold{stopped}"
                ));
            } else if !kd_ok {
                // KD gate failed — the ternary student diverges from the real
                // NF4 teacher beyond thresholds. Fail the job (all block
                // receipts remain inspectable, each carries the KD numbers).
                job.state = DistillationState::Failed;
                job.error = Some(format!(
                    "KD gate failed: {}",
                    job.kd_gate_result
                        .as_ref()
                        .map(|g| g.reasons.join("; "))
                        .unwrap_or_default()
                ));
            } else if job
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
