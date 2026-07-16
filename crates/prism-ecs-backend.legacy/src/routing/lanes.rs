//! Lane routing logic — how the router selects a backend and records
//! the decision, along with execution and transfer receipts.

use super::*;

// ── Routing ────────────────────────────────────────────────────────────────

/// How the router selects a backend.
#[derive(Debug, Clone)]
pub enum RoutingMode {
    /// Execute only on the specified backend; fail if unavailable.
    Forced(BackendId),
    /// Execute on the control/authority backend.
    Baseline,
    /// Authority produces result; candidate executes for measurement only.
    ShadowCompare {
        authority: BackendId,
        candidate: BackendId,
    },
    /// Use a measured, evidence-authorized selection.
    MeasuredSelection,
    /// Production policy (sealed profile).
    ProductionPolicy,
}

/// A routing request for one operation.
#[derive(Debug, Clone)]
pub struct RouteRequest {
    pub operation: OperationDescriptor,
    pub candidate_backends: Vec<BackendId>,
    pub routing_mode: RoutingMode,
    pub session_id: u64, // SessionId from compute_lane
    pub evaluation_group_id: EvaluationGroupId,
}

/// Reason the router selected a specific backend.
#[derive(Debug, Clone)]
pub enum RouteSelectionReason {
    Forced,
    BaselineAuthority,
    OnlyCandidate,
    PolicyMatch { evidence: Option<EvidenceDigest> },
    Fallback { original_error: String },
}

/// Information about a candidate backend considered during routing.
#[derive(Debug, Clone)]
pub struct BackendCandidate {
    pub backend_id: BackendId,
    pub eligible: bool,
    pub reason: String,
}

/// Receipt emitted before execution — the router's decision.
#[derive(Debug, Clone)]
pub struct RouteDecisionReceipt {
    pub operation_id: OperationId,
    pub requested_backend: BackendId,
    pub selected_backend: BackendId,
    pub selection_reason: RouteSelectionReason,
    pub candidate_backends: Vec<BackendCandidate>,
    pub forced: bool,
    pub fallback_allowed: bool,
    pub decision_duration_ns: u64,
}

// ── Execution receipts ─────────────────────────────────────────────────────

/// Backend version identity.
#[derive(Debug, Clone)]
pub struct BackendVersion {
    pub backend_name: String,
    pub version: String,
    pub git_commit: Option<String>,
}

/// Physical execution receipt — what the backend actually did.
#[derive(Debug, Clone)]
pub struct BackendExecutionReceipt {
    pub operation_id: OperationId,
    pub backend_id: BackendId,
    pub backend_version: BackendVersion,
    pub requested_substrate: Option<RequestedSubstrate>,
    pub observed_substrate: Option<Substrate>,
    pub graph_build_ns: Option<u64>,
    pub compile_ns: Option<u64>,
    pub queue_wait_ns: Option<u64>,
    pub submit_ns: Option<u64>,
    pub execution_ns: Option<u64>,
    pub synchronization_ns: Option<u64>,
    pub total_wall_ns: u64,
    pub bytes_read: Option<u64>,
    pub bytes_written: Option<u64>,
    pub temporary_bytes: Option<u64>,
    pub active_memory_before: Option<u64>,
    pub active_memory_after: Option<u64>,
    pub cache_memory_before: Option<u64>,
    pub cache_memory_after: Option<u64>,
    pub transfer_in_ns: Option<u64>,
    pub transfer_out_ns: Option<u64>,
    pub fallback_occurred: bool,
}

// ── Tensor transfer ────────────────────────────────────────────────────────

/// Conversion required when moving a tensor between backends.
#[derive(Debug, Clone)]
pub enum LayoutConversion {
    None,
    Transpose,
    Cast { from: DType, to: DType },
    Pack { group_size: u32, bits: u8 },
    Unpack { group_size: u32, bits: u8 },
    Contiguous,
}

/// Receipt for explicit cross-backend tensor movement.
///
/// Preserves exact source and destination layouts, dtypes, and timings.
/// No invented detail — every field is observed or absent.
#[derive(Debug, Clone)]
pub struct TensorTransferReceipt {
    pub tensor_id: TensorId,
    pub tensor_version: TensorVersion,
    pub source_materialization: TensorMaterializationId,
    pub destination_materialization: TensorMaterializationId,
    pub source_backend: BackendId,
    pub destination_backend: BackendId,
    pub source_layout: PhysicalLayout,
    pub destination_layout: PhysicalLayout,
    pub source_dtype: DType,
    pub destination_dtype: DType,
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub transfer_ns: u64,
    pub conversion_ns: u64,
    pub zero_copy: bool,
}

// ── Transfer plan ──────────────────────────────────────────────────────────

/// Kind of conversion in a compile-time transfer plan.
#[derive(Debug, Clone)]
pub enum ConversionKind {
    None,
    LayoutConversion,
    DtypeCast,
    OwnedCopy,
    SharedReference,
}

/// Compile-time plan for moving a tensor between backends.
#[derive(Debug, Clone)]
pub struct TensorTransferPlan {
    pub tensor_id: TensorId,
    pub source_backend: BackendId,
    pub destination_backend: BackendId,
    pub source_layout: PhysicalLayout,
    pub destination_layout: PhysicalLayout,
    pub conversion: ConversionKind,
    pub expected_bytes: u64,
    pub synchronization_before: bool,
    pub synchronization_after: bool,
}
