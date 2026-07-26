//! Engine execution receipts — structured records of the engine lifecycle.
//!
//! This file owns the canonical authority for engine execution telemetry in
//! the runtime: model load metrics, request admission decisions, per-phase
//! and per-step timing, terminal request outcomes, and worker exit summaries.
//!
//! Receipts are plain `serde` records, durably storable through the runtime's
//! [`crate::ports::EvidenceSink`]. They participate in the canonical change
//! flow as evidence; they are not constitutional commands and they do not
//! mutate world state. The execution telemetry authority belongs to the
//! runtime, not to the engine, so the engine can be retired as a parallel
//! authority after the remaining engine surfaces are absorbed.
//!
//! # Receipt identity
//!
//! Every receipt carries a [`ReceiptId`] drawn from
//! [`prism_ecs_constitutional::ReceiptId`] — a typed newtype, not a raw
//! string. Receipts use a builder (`new()` + `with_*()` + `build() -> Self`)
//! pattern so partial construction is explicit and validated at the boundary.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};

pub use prism_ecs_constitutional::ReceiptId;

// ── 1. ModelLoadReceipt ─────────────────────────────────────────────────────

/// Captures the full cost of loading a compute image into a worker.
///
/// A `ModelLoadReceipt` is produced once per model load and durably recorded.
/// It carries the image identity (hash + ABI pair), the worker process identity,
/// the residency cost (mapped / persistent / materialized / copied bytes), and
/// the admission estimate that justified the load.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelLoadReceipt {
    /// Stable identity of the loaded image.
    pub receipt_id: ReceiptId,
    /// Image hash (e.g. BLAKE3 digest of the cimage).
    pub image_hash: String,
    /// Storage ABI version the image was loaded under.
    pub storage_abi: String,
    /// Runtime ABI version the worker speaks.
    pub runtime_abi: String,
    /// Worker process id (host identity, not authority-bearing).
    pub worker_pid: u32,
    /// Wall-clock time to open the model from storage, in milliseconds.
    pub model_open_ms: u64,
    /// Total bytes mapped into the worker's virtual address space.
    pub mapped_virtual_bytes: u64,
    /// Bytes pinned in physical memory after admission.
    pub persistent_resident_bytes: u64,
    /// Bytes materialized (decoded/decompressed) on demand.
    pub materialized_bytes: u64,
    /// Bytes copied between tiers (storage ↔ accelerator).
    pub copied_bytes: u64,
    /// Number of tensor bindings resolved during load.
    pub tensor_binding_count: u32,
    /// Number of distinct segments in the image.
    pub segment_count: u32,
    /// MLX active memory limit at load time, in bytes (0 if not MLX).
    pub mlx_active_limit_bytes: u64,
    /// MLX cache limit at load time, in bytes (0 if not MLX).
    pub mlx_cache_limit_bytes: u64,
    /// Worker RSS before the load, in bytes.
    pub rss_before_bytes: u64,
    /// Worker RSS after the load, in bytes.
    pub rss_after_bytes: u64,
    /// Serialized admission estimate for replay auditing.
    pub admission_estimate_json: String,
}

/// Builder for [`ModelLoadReceipt`]. Fields default to zero / empty so a
/// partial builder is still constructible; the caller is expected to set the
/// fields that apply to the load path actually used.
#[derive(Debug, Clone, Default)]
pub struct ModelLoadReceiptBuilder {
    receipt_id: Option<ReceiptId>,
    image_hash: String,
    storage_abi: String,
    runtime_abi: String,
    worker_pid: u32,
    model_open_ms: u64,
    mapped_virtual_bytes: u64,
    persistent_resident_bytes: u64,
    materialized_bytes: u64,
    copied_bytes: u64,
    tensor_binding_count: u32,
    segment_count: u32,
    mlx_active_limit_bytes: u64,
    mlx_cache_limit_bytes: u64,
    rss_before_bytes: u64,
    rss_after_bytes: u64,
    admission_estimate_json: String,
}

impl ModelLoadReceiptBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the receipt id explicitly (e.g. when the id is provided by the
    /// event store). If unset, [`build`](Self::build) generates one.
    pub fn with_receipt_id(mut self, id: ReceiptId) -> Self {
        self.receipt_id = Some(id);
        self
    }

    pub fn with_image_hash(mut self, v: impl Into<String>) -> Self {
        self.image_hash = v.into();
        self
    }

    pub fn with_storage_abi(mut self, v: impl Into<String>) -> Self {
        self.storage_abi = v.into();
        self
    }

    pub fn with_runtime_abi(mut self, v: impl Into<String>) -> Self {
        self.runtime_abi = v.into();
        self
    }

    pub fn with_worker_pid(mut self, v: u32) -> Self {
        self.worker_pid = v;
        self
    }

    pub fn with_model_open_ms(mut self, v: u64) -> Self {
        self.model_open_ms = v;
        self
    }

    pub fn with_mapped_virtual_bytes(mut self, v: u64) -> Self {
        self.mapped_virtual_bytes = v;
        self
    }

    pub fn with_persistent_resident_bytes(mut self, v: u64) -> Self {
        self.persistent_resident_bytes = v;
        self
    }

    pub fn with_materialized_bytes(mut self, v: u64) -> Self {
        self.materialized_bytes = v;
        self
    }

    pub fn with_copied_bytes(mut self, v: u64) -> Self {
        self.copied_bytes = v;
        self
    }

    pub fn with_tensor_binding_count(mut self, v: u32) -> Self {
        self.tensor_binding_count = v;
        self
    }

    pub fn with_segment_count(mut self, v: u32) -> Self {
        self.segment_count = v;
        self
    }

    pub fn with_mlx_active_limit_bytes(mut self, v: u64) -> Self {
        self.mlx_active_limit_bytes = v;
        self
    }

    pub fn with_mlx_cache_limit_bytes(mut self, v: u64) -> Self {
        self.mlx_cache_limit_bytes = v;
        self
    }

    pub fn with_rss_before_bytes(mut self, v: u64) -> Self {
        self.rss_before_bytes = v;
        self
    }

    pub fn with_rss_after_bytes(mut self, v: u64) -> Self {
        self.rss_after_bytes = v;
        self
    }

    pub fn with_admission_estimate_json(mut self, v: impl Into<String>) -> Self {
        self.admission_estimate_json = v.into();
        self
    }

    /// Finalize the builder. Generates a fresh [`ReceiptId`] if the caller
    /// did not supply one. Returns an error only if the builder is in an
    /// internally inconsistent state (currently unreachable — the builder
    /// always produces a valid record).
    pub fn build(self) -> ModelLoadReceipt {
        let receipt_id = self
            .receipt_id
            .unwrap_or_else(|| ReceiptId(uuid_v4_string()));
        ModelLoadReceipt {
            receipt_id,
            image_hash: self.image_hash,
            storage_abi: self.storage_abi,
            runtime_abi: self.runtime_abi,
            worker_pid: self.worker_pid,
            model_open_ms: self.model_open_ms,
            mapped_virtual_bytes: self.mapped_virtual_bytes,
            persistent_resident_bytes: self.persistent_resident_bytes,
            materialized_bytes: self.materialized_bytes,
            copied_bytes: self.copied_bytes,
            tensor_binding_count: self.tensor_binding_count,
            segment_count: self.segment_count,
            mlx_active_limit_bytes: self.mlx_active_limit_bytes,
            mlx_cache_limit_bytes: self.mlx_cache_limit_bytes,
            rss_before_bytes: self.rss_before_bytes,
            rss_after_bytes: self.rss_after_bytes,
            admission_estimate_json: self.admission_estimate_json,
        }
    }
}

// ── 2. RequestAdmissionReceipt ──────────────────────────────────────────────

/// Admission decision for an incoming inference request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestAdmissionReceipt {
    pub receipt_id: ReceiptId,
    /// Identifier of the policy that produced the decision.
    pub policy_id: String,
    /// Whether the request passed the policy qualification gate.
    pub qualification: bool,
    /// Prompt token count.
    pub prompt_tokens: u32,
    /// Output token budget granted.
    pub output_token_budget: u32,
    /// Context (KV) budget granted.
    pub context_budget: u32,
    /// Estimated KV state size, in bytes.
    pub estimated_kv_bytes: u64,
    /// Estimated attention workspace size, in bytes.
    pub estimated_attention_workspace_bytes: u64,
    /// Absolute deadline for the request, in ms since epoch.
    pub deadline_ms: u64,
    /// Worker RSS soft ceiling at admission time, in bytes.
    pub worker_rss_soft_ceiling_bytes: u64,
    /// Worker RSS hard ceiling at admission time, in bytes.
    pub worker_rss_hard_ceiling_bytes: u64,
    /// The decision string: `"admitted"` or `"rejected"`.
    pub decision: AdmissionDecision,
    /// Reason for rejection (only set when `decision` is
    /// [`AdmissionDecision::Rejected`]).
    pub reject_reason: Option<String>,
}

/// Admission decision as a typed enum — the engine previously used a free-form
/// `String`, which made the admit/reject branching an opaque contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdmissionDecision {
    Admitted,
    Rejected,
}

impl AdmissionDecision {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Admitted => "admitted",
            Self::Rejected => "rejected",
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct RequestAdmissionReceiptBuilder {
    receipt_id: Option<ReceiptId>,
    policy_id: String,
    qualification: bool,
    prompt_tokens: u32,
    output_token_budget: u32,
    context_budget: u32,
    estimated_kv_bytes: u64,
    estimated_attention_workspace_bytes: u64,
    deadline_ms: u64,
    worker_rss_soft_ceiling_bytes: u64,
    worker_rss_hard_ceiling_bytes: u64,
    decision: Option<AdmissionDecision>,
    reject_reason: Option<String>,
}

impl RequestAdmissionReceiptBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_receipt_id(mut self, id: ReceiptId) -> Self {
        self.receipt_id = Some(id);
        self
    }

    pub fn with_policy_id(mut self, v: impl Into<String>) -> Self {
        self.policy_id = v.into();
        self
    }

    pub fn with_qualification(mut self, v: bool) -> Self {
        self.qualification = v;
        self
    }

    pub fn with_prompt_tokens(mut self, v: u32) -> Self {
        self.prompt_tokens = v;
        self
    }

    pub fn with_output_token_budget(mut self, v: u32) -> Self {
        self.output_token_budget = v;
        self
    }

    pub fn with_context_budget(mut self, v: u32) -> Self {
        self.context_budget = v;
        self
    }

    pub fn with_estimated_kv_bytes(mut self, v: u64) -> Self {
        self.estimated_kv_bytes = v;
        self
    }

    pub fn with_estimated_attention_workspace_bytes(mut self, v: u64) -> Self {
        self.estimated_attention_workspace_bytes = v;
        self
    }

    pub fn with_deadline_ms(mut self, v: u64) -> Self {
        self.deadline_ms = v;
        self
    }

    pub fn with_worker_rss_soft_ceiling_bytes(mut self, v: u64) -> Self {
        self.worker_rss_soft_ceiling_bytes = v;
        self
    }

    pub fn with_worker_rss_hard_ceiling_bytes(mut self, v: u64) -> Self {
        self.worker_rss_hard_ceiling_bytes = v;
        self
    }

    pub fn with_decision(mut self, v: AdmissionDecision) -> Self {
        self.decision = Some(v);
        self
    }

    /// Set the rejection reason. This is a thin wrapper that defaults the
    /// decision to [`AdmissionDecision::Rejected`] if the caller has not
    /// already set one. The reason is ignored when the decision is
    /// [`AdmissionDecision::Admitted`].
    pub fn with_reject_reason(mut self, v: Option<String>) -> Self {
        let has_reason = v.is_some();
        self.reject_reason = v;
        if has_reason && self.decision.is_none() {
            self.decision = Some(AdmissionDecision::Rejected);
        }
        self
    }

    /// Finalize the builder. Defaults the decision to
    /// [`AdmissionDecision::Admitted`] when no decision and no reject
    /// reason was supplied.
    pub fn build(self) -> RequestAdmissionReceipt {
        let receipt_id = self
            .receipt_id
            .unwrap_or_else(|| ReceiptId(uuid_v4_string()));
        let decision = self
            .decision
            .unwrap_or(if self.reject_reason.is_some() {
                AdmissionDecision::Rejected
            } else {
                AdmissionDecision::Admitted
            });
        RequestAdmissionReceipt {
            receipt_id,
            policy_id: self.policy_id,
            qualification: self.qualification,
            prompt_tokens: self.prompt_tokens,
            output_token_budget: self.output_token_budget,
            context_budget: self.context_budget,
            estimated_kv_bytes: self.estimated_kv_bytes,
            estimated_attention_workspace_bytes: self.estimated_attention_workspace_bytes,
            deadline_ms: self.deadline_ms,
            worker_rss_soft_ceiling_bytes: self.worker_rss_soft_ceiling_bytes,
            worker_rss_hard_ceiling_bytes: self.worker_rss_hard_ceiling_bytes,
            decision,
            reject_reason: self.reject_reason,
        }
    }
}

// ── 3. PhaseReceipt ─────────────────────────────────────────────────────────

/// The execution phase that produced this telemetry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionPhase {
    Prefill,
    Decode,
}

impl ExecutionPhase {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Prefill => "prefill",
            Self::Decode => "decode",
        }
    }
}

/// Per-phase telemetry: timing, KV state, memory, and copy classification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseReceipt {
    pub receipt_id: ReceiptId,
    /// Phase that produced this telemetry.
    pub phase: ExecutionPhase,
    /// Wall-clock time for the phase, in milliseconds.
    pub wall_time_ms: u64,
    /// Time spent building the compute graph, in milliseconds.
    pub graph_build_ms: u64,
    /// Time spent evaluating the graph, in milliseconds.
    pub eval_ms: u64,
    /// Time spent queued waiting for the executor, in milliseconds.
    pub queue_ms: u64,
    /// File bytes read during the phase.
    pub file_bytes_read: u64,
    /// Tensor views created during the phase.
    pub tensor_view_creations: u32,
    /// Tensor views reused from cache.
    pub tensor_view_reuses: u32,
    /// MLX active bytes during the phase (0 if not MLX).
    pub mlx_active_bytes: u64,
    /// MLX cache bytes during the phase.
    pub mlx_cache_bytes: u64,
    /// MLX peak bytes during the phase.
    pub mlx_peak_bytes: u64,
    /// Worker RSS during the phase, in bytes.
    pub worker_rss_bytes: u64,
    /// Logical position in the KV cache at the end of the phase.
    pub kv_logical_position: u32,
    /// KV cache allocated bytes at the end of the phase.
    pub kv_allocated_bytes: u64,
    /// Labels for each copy event during the phase (e.g.
    /// `"application_copy_free"`, `"zero_copy"`, `"eager_materialize"`).
    pub copy_classifications: Vec<String>,
    /// Bytes of attention mask material.
    pub mask_bytes: u64,
}

#[derive(Debug, Clone, Default)]
pub struct PhaseReceiptBuilder {
    receipt_id: Option<ReceiptId>,
    phase: Option<ExecutionPhase>,
    wall_time_ms: u64,
    graph_build_ms: u64,
    eval_ms: u64,
    queue_ms: u64,
    file_bytes_read: u64,
    tensor_view_creations: u32,
    tensor_view_reuses: u32,
    mlx_active_bytes: u64,
    mlx_cache_bytes: u64,
    mlx_peak_bytes: u64,
    worker_rss_bytes: u64,
    kv_logical_position: u32,
    kv_allocated_bytes: u64,
    copy_classifications: Vec<String>,
    mask_bytes: u64,
}

impl PhaseReceiptBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_receipt_id(mut self, id: ReceiptId) -> Self {
        self.receipt_id = Some(id);
        self
    }

    pub fn with_phase(mut self, v: ExecutionPhase) -> Self {
        self.phase = Some(v);
        self
    }

    pub fn with_wall_time_ms(mut self, v: u64) -> Self {
        self.wall_time_ms = v;
        self
    }

    pub fn with_graph_build_ms(mut self, v: u64) -> Self {
        self.graph_build_ms = v;
        self
    }

    pub fn with_eval_ms(mut self, v: u64) -> Self {
        self.eval_ms = v;
        self
    }

    pub fn with_queue_ms(mut self, v: u64) -> Self {
        self.queue_ms = v;
        self
    }

    pub fn with_file_bytes_read(mut self, v: u64) -> Self {
        self.file_bytes_read = v;
        self
    }

    pub fn with_tensor_view_creations(mut self, v: u32) -> Self {
        self.tensor_view_creations = v;
        self
    }

    pub fn with_tensor_view_reuses(mut self, v: u32) -> Self {
        self.tensor_view_reuses = v;
        self
    }

    pub fn with_mlx_active_bytes(mut self, v: u64) -> Self {
        self.mlx_active_bytes = v;
        self
    }

    pub fn with_mlx_cache_bytes(mut self, v: u64) -> Self {
        self.mlx_cache_bytes = v;
        self
    }

    pub fn with_mlx_peak_bytes(mut self, v: u64) -> Self {
        self.mlx_peak_bytes = v;
        self
    }

    pub fn with_worker_rss_bytes(mut self, v: u64) -> Self {
        self.worker_rss_bytes = v;
        self
    }

    pub fn with_kv_logical_position(mut self, v: u32) -> Self {
        self.kv_logical_position = v;
        self
    }

    pub fn with_kv_allocated_bytes(mut self, v: u64) -> Self {
        self.kv_allocated_bytes = v;
        self
    }

    pub fn with_copy_classifications(mut self, v: Vec<String>) -> Self {
        self.copy_classifications = v;
        self
    }

    pub fn with_mask_bytes(mut self, v: u64) -> Self {
        self.mask_bytes = v;
        self
    }

    /// Finalize the builder. Defaults `phase` to
    /// [`ExecutionPhase::Decode`] if unset.
    pub fn build(self) -> PhaseReceipt {
        let receipt_id = self
            .receipt_id
            .unwrap_or_else(|| ReceiptId(uuid_v4_string()));
        let phase = self.phase.unwrap_or(ExecutionPhase::Decode);
        PhaseReceipt {
            receipt_id,
            phase,
            wall_time_ms: self.wall_time_ms,
            graph_build_ms: self.graph_build_ms,
            eval_ms: self.eval_ms,
            queue_ms: self.queue_ms,
            file_bytes_read: self.file_bytes_read,
            tensor_view_creations: self.tensor_view_creations,
            tensor_view_reuses: self.tensor_view_reuses,
            mlx_active_bytes: self.mlx_active_bytes,
            mlx_cache_bytes: self.mlx_cache_bytes,
            mlx_peak_bytes: self.mlx_peak_bytes,
            worker_rss_bytes: self.worker_rss_bytes,
            kv_logical_position: self.kv_logical_position,
            kv_allocated_bytes: self.kv_allocated_bytes,
            copy_classifications: self.copy_classifications,
            mask_bytes: self.mask_bytes,
        }
    }
}

// ── 4. StepReceipt ──────────────────────────────────────────────────────────

/// Per-decode-step telemetry. Wraps a [`PhaseReceipt`] with decode-step
/// identifying fields (step index, sampled token, KV position, wall time in
/// microseconds).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepReceipt {
    pub receipt_id: ReceiptId,
    /// Decode step index (0 = first generated token after prefill).
    pub step_index: u32,
    /// Sampled token id for this step.
    pub token_id: u32,
    /// Sequence position of this step in the KV cache.
    pub position: u32,
    /// Wall-clock time for this step, in microseconds.
    pub wall_time_us: u64,
    /// Per-phase telemetry for the step.
    pub phase_receipt: PhaseReceipt,
}

#[derive(Debug, Clone, Default)]
pub struct StepReceiptBuilder {
    receipt_id: Option<ReceiptId>,
    step_index: u32,
    token_id: u32,
    position: u32,
    wall_time_us: u64,
    phase_receipt: Option<PhaseReceipt>,
}

impl StepReceiptBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_receipt_id(mut self, id: ReceiptId) -> Self {
        self.receipt_id = Some(id);
        self
    }

    pub fn with_step_index(mut self, v: u32) -> Self {
        self.step_index = v;
        self
    }

    pub fn with_token_id(mut self, v: u32) -> Self {
        self.token_id = v;
        self
    }

    pub fn with_position(mut self, v: u32) -> Self {
        self.position = v;
        self
    }

    pub fn with_wall_time_us(mut self, v: u64) -> Self {
        self.wall_time_us = v;
        self
    }

    pub fn with_phase_receipt(mut self, v: PhaseReceipt) -> Self {
        self.phase_receipt = Some(v);
        self
    }

    pub fn build(self) -> StepReceipt {
        let receipt_id = self
            .receipt_id
            .unwrap_or_else(|| ReceiptId(uuid_v4_string()));
        StepReceipt {
            receipt_id,
            step_index: self.step_index,
            token_id: self.token_id,
            position: self.position,
            wall_time_us: self.wall_time_us,
            phase_receipt: self.phase_receipt.unwrap_or_else(|| {
                PhaseReceiptBuilder::new()
                    .with_phase(ExecutionPhase::Decode)
                    .build()
            }),
        }
    }
}

// ── 5. TerminalRequestReceipt ──────────────────────────────────────────────

/// Terminal outcome class for an inference request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestOutcome {
    Completed,
    Cancelled,
    Failed,
    TimedOut,
}

impl RequestOutcome {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Cancelled => "cancelled",
            Self::Failed => "failed",
            Self::TimedOut => "timed_out",
        }
    }
}

/// Optional reason a request was cancelled. Only set when `outcome` is
/// [`RequestOutcome::Cancelled`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CancellationMode {
    ClientDisconnect,
    ServerShutdown,
    DeadlineExceeded,
    Preempted,
    Other,
}

/// Terminal summary for a single inference request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerminalRequestReceipt {
    pub receipt_id: ReceiptId,
    /// Stable id of the request (e.g. HTTP request id).
    pub request_id: String,
    /// Terminal outcome.
    pub outcome: RequestOutcome,
    /// Number of tokens generated before termination.
    pub generated_token_count: u32,
    /// Time to first token, in milliseconds.
    pub ttft_ms: u64,
    /// Per-token latency in milliseconds, in step order.
    pub per_token_latency_ms: Vec<u64>,
    /// Peak RSS during the request, in bytes.
    pub peak_rss_bytes: u64,
    /// Peak MLX active bytes during the request.
    pub peak_mlx_active_bytes: u64,
    /// Peak MLX cache bytes during the request.
    pub peak_mlx_cache_bytes: u64,
    /// Whether the request had to be forcibly terminated.
    pub forced_termination: bool,
    /// Why the request was cancelled (only when `outcome` is
    /// [`RequestOutcome::Cancelled`]).
    pub cancellation_mode: Option<CancellationMode>,
    /// Whether the worker must restart before serving the next request.
    pub worker_restart_required: bool,
    /// The last phase that completed before termination.
    pub last_completed_phase: ExecutionPhase,
    /// Stable error code, set when `outcome` is [`RequestOutcome::Failed`]
    /// or [`RequestOutcome::TimedOut`].
    pub error_code: Option<String>,
    /// Human-readable error message.
    pub error_message: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct TerminalRequestReceiptBuilder {
    receipt_id: Option<ReceiptId>,
    request_id: String,
    outcome: Option<RequestOutcome>,
    generated_token_count: u32,
    ttft_ms: u64,
    per_token_latency_ms: Vec<u64>,
    peak_rss_bytes: u64,
    peak_mlx_active_bytes: u64,
    peak_mlx_cache_bytes: u64,
    forced_termination: bool,
    cancellation_mode: Option<CancellationMode>,
    worker_restart_required: bool,
    last_completed_phase: Option<ExecutionPhase>,
    error_code: Option<String>,
    error_message: Option<String>,
}

impl TerminalRequestReceiptBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_receipt_id(mut self, id: ReceiptId) -> Self {
        self.receipt_id = Some(id);
        self
    }

    pub fn with_request_id(mut self, v: impl Into<String>) -> Self {
        self.request_id = v.into();
        self
    }

    pub fn with_outcome(mut self, v: RequestOutcome) -> Self {
        self.outcome = Some(v);
        self
    }

    pub fn with_generated_token_count(mut self, v: u32) -> Self {
        self.generated_token_count = v;
        self
    }

    pub fn with_ttft_ms(mut self, v: u64) -> Self {
        self.ttft_ms = v;
        self
    }

    pub fn with_per_token_latency_ms(mut self, v: Vec<u64>) -> Self {
        self.per_token_latency_ms = v;
        self
    }

    pub fn with_peak_rss_bytes(mut self, v: u64) -> Self {
        self.peak_rss_bytes = v;
        self
    }

    pub fn with_peak_mlx_active_bytes(mut self, v: u64) -> Self {
        self.peak_mlx_active_bytes = v;
        self
    }

    pub fn with_peak_mlx_cache_bytes(mut self, v: u64) -> Self {
        self.peak_mlx_cache_bytes = v;
        self
    }

    pub fn with_forced_termination(mut self, v: bool) -> Self {
        self.forced_termination = v;
        self
    }

    pub fn with_cancellation_mode(mut self, v: Option<CancellationMode>) -> Self {
        self.cancellation_mode = v;
        if v.is_some() && self.outcome.is_none() {
            self.outcome = Some(RequestOutcome::Cancelled);
        }
        self
    }

    pub fn with_worker_restart_required(mut self, v: bool) -> Self {
        self.worker_restart_required = v;
        self
    }

    pub fn with_last_completed_phase(mut self, v: ExecutionPhase) -> Self {
        self.last_completed_phase = Some(v);
        self
    }

    pub fn with_error_code(mut self, v: Option<String>) -> Self {
        self.error_code = v;
        self
    }

    pub fn with_error_message(mut self, v: Option<String>) -> Self {
        self.error_message = v;
        self
    }

    pub fn build(self) -> TerminalRequestReceipt {
        let receipt_id = self
            .receipt_id
            .unwrap_or_else(|| ReceiptId(uuid_v4_string()));
        let outcome = self.outcome.unwrap_or(RequestOutcome::Completed);
        TerminalRequestReceipt {
            receipt_id,
            request_id: self.request_id,
            outcome,
            generated_token_count: self.generated_token_count,
            ttft_ms: self.ttft_ms,
            per_token_latency_ms: self.per_token_latency_ms,
            peak_rss_bytes: self.peak_rss_bytes,
            peak_mlx_active_bytes: self.peak_mlx_active_bytes,
            peak_mlx_cache_bytes: self.peak_mlx_cache_bytes,
            forced_termination: self.forced_termination,
            cancellation_mode: self.cancellation_mode,
            worker_restart_required: self.worker_restart_required,
            last_completed_phase: self
                .last_completed_phase
                .unwrap_or(ExecutionPhase::Decode),
            error_code: self.error_code,
            error_message: self.error_message,
        }
    }
}

// ── 6. WorkerExitReceipt ────────────────────────────────────────────────────

/// Worker-level exit summary. Recorded when a worker process terminates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerExitReceipt {
    pub receipt_id: ReceiptId,
    /// Worker process id (host identity, not authority-bearing).
    pub worker_pid: u32,
    /// POSIX-style exit code, when the worker exited normally.
    pub exit_code: Option<i32>,
    /// Signal number, when the worker was killed.
    pub signal: Option<i32>,
    /// Wall-clock uptime before exit, in milliseconds.
    pub uptime_ms: u64,
    /// Number of requests completed before exit.
    pub requests_completed: u32,
    /// Number of requests that failed before exit.
    pub requests_failed: u32,
    /// Peak RSS observed during the worker's lifetime, in bytes.
    pub peak_rss_bytes: u64,
    /// Whether the worker exited due to a fault (assertion / segfault / panic).
    pub faulted: bool,
    /// Timestamp of the last heartbeat before exit, in ms since epoch.
    pub last_heartbeat_ms: u64,
}

#[derive(Debug, Clone, Default)]
pub struct WorkerExitReceiptBuilder {
    receipt_id: Option<ReceiptId>,
    worker_pid: u32,
    exit_code: Option<i32>,
    signal: Option<i32>,
    uptime_ms: u64,
    requests_completed: u32,
    requests_failed: u32,
    peak_rss_bytes: u64,
    faulted: bool,
    last_heartbeat_ms: u64,
}

impl WorkerExitReceiptBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_receipt_id(mut self, id: ReceiptId) -> Self {
        self.receipt_id = Some(id);
        self
    }

    pub fn with_worker_pid(mut self, v: u32) -> Self {
        self.worker_pid = v;
        self
    }

    pub fn with_exit_code(mut self, v: Option<i32>) -> Self {
        self.exit_code = v;
        self
    }

    pub fn with_signal(mut self, v: Option<i32>) -> Self {
        self.signal = v;
        self
    }

    pub fn with_uptime_ms(mut self, v: u64) -> Self {
        self.uptime_ms = v;
        self
    }

    pub fn with_requests_completed(mut self, v: u32) -> Self {
        self.requests_completed = v;
        self
    }

    pub fn with_requests_failed(mut self, v: u32) -> Self {
        self.requests_failed = v;
        self
    }

    pub fn with_peak_rss_bytes(mut self, v: u64) -> Self {
        self.peak_rss_bytes = v;
        self
    }

    pub fn with_faulted(mut self, v: bool) -> Self {
        self.faulted = v;
        self
    }

    pub fn with_last_heartbeat_ms(mut self, v: u64) -> Self {
        self.last_heartbeat_ms = v;
        self
    }

    pub fn build(self) -> WorkerExitReceipt {
        let receipt_id = self
            .receipt_id
            .unwrap_or_else(|| ReceiptId(uuid_v4_string()));
        WorkerExitReceipt {
            receipt_id,
            worker_pid: self.worker_pid,
            exit_code: self.exit_code,
            signal: self.signal,
            uptime_ms: self.uptime_ms,
            requests_completed: self.requests_completed,
            requests_failed: self.requests_failed,
            peak_rss_bytes: self.peak_rss_bytes,
            faulted: self.faulted,
            last_heartbeat_ms: self.last_heartbeat_ms,
        }
    }
}

// ── 7. ReceiptBuilder — unified entry point ─────────────────────────────────

/// Unified entry point for constructing engine receipts.
///
/// Each method returns a typed builder. The receipt identity is allocated
/// lazily by [`build`](ModelLoadReceiptBuilder::build) unless the caller
/// supplies one explicitly. Receipts are storable through the runtime's
/// evidence sink; this type is the only public surface for receipt
/// construction in the runtime crate.
#[derive(Debug, Clone, Copy, Default)]
pub struct ReceiptBuilder;

impl ReceiptBuilder {
    pub fn new() -> Self {
        Self
    }

    pub fn model_load() -> ModelLoadReceiptBuilder {
        ModelLoadReceiptBuilder::new()
    }

    pub fn request_admission() -> RequestAdmissionReceiptBuilder {
        RequestAdmissionReceiptBuilder::new()
    }

    pub fn phase() -> PhaseReceiptBuilder {
        PhaseReceiptBuilder::new()
    }

    pub fn step() -> StepReceiptBuilder {
        StepReceiptBuilder::new()
    }

    pub fn terminal_request() -> TerminalRequestReceiptBuilder {
        TerminalRequestReceiptBuilder::new()
    }

    pub fn worker_exit() -> WorkerExitReceiptBuilder {
        WorkerExitReceiptBuilder::new()
    }
}

// ── 8. DiffusionStepReceipt ─────────────────────────────────────────────────

/// Captures the result of a single diffusion denoising step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffusionStepReceipt {
    pub receipt_id: ReceiptId,
    pub step: u32,
    pub timestep: u32,
    pub sampler_temperature: f32,
    pub commit_mask_hash: String,
    pub convergence_result: Option<String>,
}

// ── 9. Timeline — bounded event buffer ──────────────────────────────────────

/// A single timestamped event in the engine timeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineEvent {
    pub timestamp: String,
    pub event_type: String,
    pub data: serde_json::Value,
}

/// Bounded event buffer. The buffer drops the oldest entry once `max_events`
/// is exceeded. Ordering is insertion order; the timeline is a ring buffer
/// of the most recent events, not a complete history. For durable history,
/// record events through the evidence sink.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Timeline {
    pub events: Vec<TimelineEvent>,
    pub max_events: usize,
}

impl Timeline {
    /// Create a new timeline with the given capacity. The internal buffer is
    /// pre-allocated to `min(max_events, 16)` to avoid thrashing for
    /// short-lived timelines.
    pub fn new(max_events: usize) -> Self {
        Self {
            events: Vec::with_capacity(max_events.min(16)),
            max_events,
        }
    }

    /// Append an event. If `events.len()` already equals `max_events`, the
    /// oldest event is removed to make room.
    pub fn append(
        &mut self,
        timestamp: impl Into<String>,
        event_type: impl Into<String>,
        data: serde_json::Value,
    ) {
        if self.events.len() >= self.max_events {
            self.events.remove(0);
        }
        self.events.push(TimelineEvent {
            timestamp: timestamp.into(),
            event_type: event_type.into(),
            data,
        });
    }

    /// Serialize this timeline as a JSON value. Returns `Value::Null` on
    /// serialization failure; the failure is not propagated because the
    /// timeline is itself a debug surface, not an authority-bearing record.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    pub fn clear(&mut self) {
        self.events.clear();
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────

/// Generate a UUID v4 string. We use the workspace `uuid` crate through
/// the runtime kernel — the receipt id is opaque to the engine.
///
/// The implementation is here (rather than re-exported from the runtime
/// kernel) to keep this module self-contained and to avoid adding a
/// dependency on a particular `uuid` feature set.
fn uuid_v4_string() -> String {
    uuid::Uuid::new_v4().to_string()
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ModelLoadReceipt ────────────────────────────────────────────────

    #[test]
    fn model_load_receipt_builder_round_trip() {
        let r = ReceiptBuilder::model_load()
            .with_image_hash("abc123")
            .with_storage_abi("v1")
            .with_runtime_abi("v2")
            .with_worker_pid(42)
            .with_model_open_ms(1500)
            .with_mapped_virtual_bytes(6_000_000_000)
            .with_persistent_resident_bytes(4_000_000_000)
            .with_materialized_bytes(2_000_000_000)
            .with_copied_bytes(500_000_000)
            .with_tensor_binding_count(128)
            .with_segment_count(16)
            .with_mlx_active_limit_bytes(8_000_000_000)
            .with_mlx_cache_limit_bytes(2_000_000_000)
            .with_rss_before_bytes(1_000_000_000)
            .with_rss_after_bytes(3_000_000_000)
            .with_admission_estimate_json("{}")
            .build();

        assert_eq!(r.image_hash, "abc123");
        assert_eq!(r.worker_pid, 42);
        assert_eq!(r.model_open_ms, 1500);
        assert_eq!(r.tensor_binding_count, 128);

        // Receipt id is non-empty and a valid uuid v4.
        assert!(!r.receipt_id.0.is_empty());

        // Round-trip through serde_json.
        let json = serde_json::to_value(&r).expect("serialize");
        let de: ModelLoadReceipt = serde_json::from_value(json).expect("deserialize");
        assert_eq!(de.image_hash, "abc123");
        assert_eq!(de.worker_pid, 42);
        assert_eq!(de.receipt_id.0, r.receipt_id.0);
    }

    // ── RequestAdmissionReceipt ─────────────────────────────────────────

    #[test]
    fn request_admission_receipt_admitted() {
        let r = ReceiptBuilder::request_admission()
            .with_policy_id("policy-01")
            .with_qualification(true)
            .with_prompt_tokens(512)
            .with_output_token_budget(128)
            .with_context_budget(1024)
            .with_estimated_kv_bytes(300_000_000)
            .with_estimated_attention_workspace_bytes(100_000_000)
            .with_deadline_ms(5000)
            .with_worker_rss_soft_ceiling_bytes(8_000_000_000)
            .with_worker_rss_hard_ceiling_bytes(10_000_000_000)
            .with_decision(AdmissionDecision::Admitted)
            .build();

        assert_eq!(r.policy_id, "policy-01");
        assert!(r.qualification);
        assert_eq!(r.decision, AdmissionDecision::Admitted);
        assert!(r.reject_reason.is_none());
    }

    #[test]
    fn request_admission_receipt_rejected() {
        let r = ReceiptBuilder::request_admission()
            .with_decision(AdmissionDecision::Rejected)
            .with_reject_reason(Some("OOM: estimated kv exceeds ceiling".to_string()))
            .build();

        assert_eq!(r.decision, AdmissionDecision::Rejected);
        assert_eq!(
            r.reject_reason.as_deref(),
            Some("OOM: estimated kv exceeds ceiling")
        );
    }

    #[test]
    fn request_admission_reject_reason_implies_rejected_decision() {
        // When only a reject reason is set, the decision defaults to Rejected.
        let r = ReceiptBuilder::request_admission()
            .with_reject_reason(Some("too many tokens".to_string()))
            .build();
        assert_eq!(r.decision, AdmissionDecision::Rejected);
        assert_eq!(r.reject_reason.as_deref(), Some("too many tokens"));
    }

    #[test]
    fn admission_decision_serde_uses_snake_case() {
        let json = serde_json::to_string(&AdmissionDecision::Admitted).unwrap();
        assert_eq!(json, "\"admitted\"");
        let de: AdmissionDecision = serde_json::from_str("\"rejected\"").unwrap();
        assert_eq!(de, AdmissionDecision::Rejected);
    }

    // ── PhaseReceipt ────────────────────────────────────────────────────

    #[test]
    fn phase_receipt_builder_prefill() {
        let r = ReceiptBuilder::phase()
            .with_phase(ExecutionPhase::Prefill)
            .with_wall_time_ms(1200)
            .with_graph_build_ms(300)
            .with_eval_ms(800)
            .with_queue_ms(100)
            .with_file_bytes_read(400_000)
            .with_tensor_view_creations(64)
            .with_tensor_view_reuses(32)
            .with_mlx_active_bytes(4_000_000_000)
            .with_mlx_cache_bytes(1_000_000_000)
            .with_mlx_peak_bytes(4_500_000_000)
            .with_worker_rss_bytes(5_000_000_000)
            .with_kv_logical_position(512)
            .with_kv_allocated_bytes(200_000_000)
            .with_copy_classifications(vec!["application_copy_free".to_string()])
            .with_mask_bytes(16_384)
            .build();

        assert_eq!(r.phase, ExecutionPhase::Prefill);
        assert_eq!(r.wall_time_ms, 1200);
        assert_eq!(r.copy_classifications.len(), 1);
    }

    #[test]
    fn phase_receipt_default_phase_is_decode() {
        let r = ReceiptBuilder::phase().with_wall_time_ms(50).build();
        assert_eq!(r.phase, ExecutionPhase::Decode);
    }

    // ── StepReceipt ─────────────────────────────────────────────────────

    #[test]
    fn step_receipt_builder_wraps_phase_receipt() {
        let phase = ReceiptBuilder::phase()
            .with_phase(ExecutionPhase::Decode)
            .with_wall_time_ms(50)
            .with_eval_ms(40)
            .build();

        let r = ReceiptBuilder::step()
            .with_step_index(5)
            .with_token_id(1234)
            .with_position(5)
            .with_wall_time_us(50_000)
            .with_phase_receipt(phase)
            .build();

        assert_eq!(r.step_index, 5);
        assert_eq!(r.token_id, 1234);
        assert_eq!(r.phase_receipt.phase, ExecutionPhase::Decode);
    }

    // ── TerminalRequestReceipt ──────────────────────────────────────────

    #[test]
    fn terminal_request_receipt_completed() {
        let r = ReceiptBuilder::terminal_request()
            .with_request_id("req-001")
            .with_outcome(RequestOutcome::Completed)
            .with_generated_token_count(256)
            .with_ttft_ms(1500)
            .with_per_token_latency_ms(vec![45, 42, 48, 44])
            .with_peak_rss_bytes(6_000_000_000)
            .with_peak_mlx_active_bytes(5_000_000_000)
            .with_peak_mlx_cache_bytes(1_500_000_000)
            .with_forced_termination(false)
            .with_worker_restart_required(false)
            .with_last_completed_phase(ExecutionPhase::Decode)
            .build();

        assert_eq!(r.request_id, "req-001");
        assert_eq!(r.outcome, RequestOutcome::Completed);
        assert_eq!(r.generated_token_count, 256);
        assert!(r.cancellation_mode.is_none());
    }

    #[test]
    fn terminal_request_receipt_cancelled() {
        let r = ReceiptBuilder::terminal_request()
            .with_outcome(RequestOutcome::Cancelled)
            .with_forced_termination(true)
            .with_cancellation_mode(Some(CancellationMode::ClientDisconnect))
            .with_last_completed_phase(ExecutionPhase::Prefill)
            .build();

        assert_eq!(r.outcome, RequestOutcome::Cancelled);
        assert!(r.forced_termination);
        assert_eq!(r.cancellation_mode, Some(CancellationMode::ClientDisconnect));
    }

    #[test]
    fn terminal_request_receipt_failed() {
        let r = ReceiptBuilder::terminal_request()
            .with_outcome(RequestOutcome::Failed)
            .with_worker_restart_required(true)
            .with_error_code(Some("E_ENGINE_CRASH".to_string()))
            .with_error_message(Some("Worker segfaulted during prefill".to_string()))
            .build();

        assert_eq!(r.outcome, RequestOutcome::Failed);
        assert!(r.worker_restart_required);
        assert_eq!(r.error_code.as_deref(), Some("E_ENGINE_CRASH"));
    }

    #[test]
    fn cancellation_mode_implies_cancelled_outcome() {
        let r = ReceiptBuilder::terminal_request()
            .with_cancellation_mode(Some(CancellationMode::ServerShutdown))
            .build();
        assert_eq!(r.outcome, RequestOutcome::Cancelled);
        assert_eq!(r.cancellation_mode, Some(CancellationMode::ServerShutdown));
    }

    // ── WorkerExitReceipt ───────────────────────────────────────────────

    #[test]
    fn worker_exit_receipt_normal_exit() {
        let r = ReceiptBuilder::worker_exit()
            .with_worker_pid(42)
            .with_exit_code(Some(0))
            .with_uptime_ms(3_600_000)
            .with_requests_completed(150)
            .with_requests_failed(2)
            .with_peak_rss_bytes(8_000_000_000)
            .with_faulted(false)
            .with_last_heartbeat_ms(3_599_900)
            .build();

        assert_eq!(r.worker_pid, 42);
        assert_eq!(r.exit_code, Some(0));
        assert!(!r.faulted);
    }

    #[test]
    fn worker_exit_receipt_signaled() {
        let r = ReceiptBuilder::worker_exit()
            .with_worker_pid(99)
            .with_exit_code(None)
            .with_signal(Some(11))
            .with_uptime_ms(120_000)
            .with_requests_completed(10)
            .with_requests_failed(1)
            .with_peak_rss_bytes(5_000_000_000)
            .with_faulted(true)
            .with_last_heartbeat_ms(119_900)
            .build();

        assert_eq!(r.signal, Some(11));
        assert!(r.faulted);
        assert_eq!(r.exit_code, None);
    }

    #[test]
    fn worker_exit_receipt_round_trip() {
        let r = ReceiptBuilder::worker_exit()
            .with_worker_pid(7)
            .with_signal(Some(9))
            .with_faulted(true)
            .build();

        let json = serde_json::to_value(&r).expect("serialize");
        let de: WorkerExitReceipt = serde_json::from_value(json).expect("deserialize");
        assert_eq!(de.worker_pid, 7);
        assert_eq!(de.signal, Some(9));
        assert!(de.faulted);
    }

    // ── Timeline ────────────────────────────────────────────────────────

    #[test]
    fn timeline_drops_oldest_when_full() {
        let mut tl = Timeline::new(3);
        assert!(tl.is_empty());

        tl.append("t1", "load", serde_json::Value::Null);
        tl.append("t2", "admit", serde_json::Value::Null);
        tl.append("t3", "prefill", serde_json::Value::Null);
        assert_eq!(tl.len(), 3);

        // Fourth push drops the oldest.
        tl.append("t4", "decode", serde_json::Value::Null);
        assert_eq!(tl.len(), 3);
        assert_eq!(tl.events[0].timestamp, "t2");
        assert_eq!(tl.events[0].event_type, "admit");

        tl.clear();
        assert!(tl.is_empty());
    }

    #[test]
    fn timeline_to_json() {
        let mut tl = Timeline::new(10);
        tl.append(
            "ts1",
            "load",
            serde_json::json!({"hash": "abc"}),
        );

        let json = tl.to_json();
        assert!(json.is_object());
        let events = json["events"].as_array().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0]["event_type"], "load");
        assert_eq!(events[0]["data"]["hash"], "abc");
    }

    #[test]
    fn timeline_to_json_empty() {
        let tl = Timeline::new(5);
        let json = tl.to_json();
        let events = json["events"].as_array().unwrap();
        assert!(events.is_empty());
    }

    // ── Receipt identity ────────────────────────────────────────────────

    #[test]
    fn explicit_receipt_id_is_honored() {
        let id = ReceiptId("explicit-id-001".to_string());
        let r = ReceiptBuilder::worker_exit()
            .with_receipt_id(id.clone())
            .with_worker_pid(1)
            .build();
        assert_eq!(r.receipt_id, id);
    }

    #[test]
    fn auto_assigned_receipt_id_is_unique() {
        let r1 = ReceiptBuilder::worker_exit().build();
        let r2 = ReceiptBuilder::worker_exit().build();
        assert_ne!(r1.receipt_id, r2.receipt_id);
    }
}
